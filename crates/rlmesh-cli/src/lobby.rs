//! The home daemon's lobby and runs (scene design §3, §4): rlmeshd hosts
//! the session `_lobby`, where robots wait and clients run scenes, and
//! hosts each run's session in-process, or has a satellite host it
//! (§4.4: a paired machine's rlmeshd waiting here as `rlmeshd.<device>`,
//! [`crate::satellite`]).
//!
//! A run goes through its stages (§4.1), each failing with its code:
//! 1. **check** the scene file against `devices.toml` and its pins
//!    ([`crate::scene::check`]): `INVALID_REQUEST`, `NOT_AUTHORIZED`;
//! 2. **claim** its robots, all or none: `ROBOT_BUSY`, naming the run;
//! 3. **host** its session, with the scene's places and member table: here,
//!    or on the satellite `host =` names (by default the machine of the
//!    first controller that runs rlmeshd, §4.1) through a `HostOrder`
//!    answered by `HostStarted`;
//! 4. **assign**: a `Join` to each robot once it waits in the lobby,
//!    `NOT_PRESENT` at `wait`;
//! 5. **declare**: each robot joins and declares, `NOT_PRESENT` (or the
//!    refusal its next wait reports) at `wait`;
//! 6. **bind**: each controller member binds, `NOT_READY` at `wait`;
//! 7. **arm**: by the daemon (`arm = "auto"`) once all of that is done, or on
//!    an `ArmScene` (`manual`, `--arm manual`);
//! 8. **run** until stopped, or a member is lost: the run then waits in
//!    `REJOINING` for it (a robot gets its join again in the lobby and must
//!    declare the same map) until the session's rejoin window closes, which
//!    fails it, and once everyone is back it waits in `ARMING` for an
//!    explicit arm, whatever the scene's `arm`;
//! 9. **tear down**: the scene closes (`SCENE_ENDED`, or `SCENE_FAILED` with
//!    the stage's error; a `StopOrder` on a satellite), robots detach and
//!    come back to the lobby, and the host stops once they left (or after
//!    2 s); the run ends once its robots are back (or 2 s later).
//!
//! The first two answer `RunScene` itself (a refused scene has no run);
//! the rest show in the run's state. A robot's claim is released when it is
//! back in the lobby, or `wait` after the run ended if it never comes back.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::net::ToSocketAddrs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::time::Duration;

use rlmesh_session::address::{SessionAddress, TENANT, session_id};
use rlmesh_session::auth::address::unbase32;
use rlmesh_session::auth::{HostIdentity, LOCAL_DEVICE, Pin};
use rlmesh_session::authority::{Member as HostMember, Verb};
use rlmesh_session::lobby::{
    LOBBY_SESSION, LobbyAnswer, LobbyReply, LobbyResult, LobbyService, SATELLITE_PREFIX,
};
use rlmesh_session::participant::Participant;
use rlmesh_session::scene::{Arm, Role as SceneRole, Scene, Source};
use rlmesh_session::server::SessionServer;
use rlmesh_session::wire::{
    Assignment, AuthoritySnapshot, Command, CtlError, CtlStatus, DeviceRecord, HostOrder,
    HostStarted, Join, LobbyWait, MemberStage, MemberState, Mode, ParticipantId, PriorityClass,
    Reason, ReasonCode, RobotPin, Role, RunStage, RunState, SessionId, StopOrder,
};
use rlmesh_session::{HostConfig, SessionError};
use tokio::sync::{mpsc, oneshot, watch};

use crate::devices::{DEVICES, Table};

/// The id the home daemon watches and arms its runs as.
pub const HOME: &str = "rlmeshd";
/// How long teardown waits for members to detach.
const STRAGGLERS: Duration = Duration::from_secs(2);

/// What a robot waiting in the lobby is answered with, once a run assigns
/// it.
struct Waiter {
    answer: oneshot::Sender<LobbyResult>,
}

/// A run, as the lobby sees it.
struct Run {
    /// Its scene's name.
    scene: String,
    state: watch::Sender<RunState>,
    commands: mpsc::UnboundedSender<RunCommand>,
    /// The join sent to each robot, re-sent while the run waits for it.
    joins: Mutex<HashMap<ParticipantId, Join>>,
    /// Set when a client stops the run, before its scene closes: from then
    /// on a robot back in the lobby is returned, never sent its join again.
    stopping: AtomicBool,
    /// Hosted on a satellite: the scene and the order it was hosted with,
    /// whose device records and pins the satellite holds.
    away: OnceLock<(Scene, HostOrder)>,
}

/// What a client asks of a running run.
enum RunCommand {
    /// Stop it (a client), or fail it (the device table changed under a
    /// satellite's run).
    End(Outcome),
    Arm {
        answer: oneshot::Sender<LobbyResult>,
    },
}

/// What a satellite's host order is answered with: the run host's key and
/// addresses.
type Started = Result<([u8; 32], Vec<String>), CtlError>;

/// A satellite daemon in the lobby: its pending wait, the orders it hasn't
/// taken yet, and the host orders waiting for its `HostStarted`, by run.
#[derive(Default)]
struct Satellite {
    waiter: Option<Waiter>,
    orders: VecDeque<Assignment>,
    started: HashMap<String, oneshot::Sender<Started>>,
}

#[derive(Default)]
struct Rooms {
    waiting: HashMap<ParticipantId, Waiter>,
    /// By device.
    satellites: HashMap<String, Satellite>,
    claims: HashMap<ParticipantId, String>,
    // TODO: every run since the daemon started is kept for `SceneStatus`;
    // drop ended ones after a while if a long-lived daemon runs thousands.
    runs: HashMap<String, Arc<Run>>,
}

/// The home daemon's lobby: its waiting robots, claims and runs.
pub struct Home {
    dir: PathBuf,
    template: HostConfig,
    addresses: Vec<String>,
    server: Weak<SessionServer>,
    /// The lobby's own authority state: who is attached.
    lobby: OnceLock<watch::Receiver<AuthoritySnapshot>>,
    rooms: Mutex<Rooms>,
    /// Bumped whenever a robot starts waiting.
    arrivals: watch::Sender<u64>,
    me: Weak<Home>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn refused(status: CtlStatus, message: impl std::fmt::Display) -> CtlError {
    CtlError::new(status, message)
}

fn over(stage: RunStage) -> bool {
    matches!(stage, RunStage::Ending | RunStage::Ended | RunStage::Failed)
}

/// Run `run`'s host order for a satellite (§4.4): the scene file, its
/// members' device records from `table` (the home machine's `local` key
/// among them, so `safety` stays with the robots' own machine), and its
/// robots' pins.
fn host_order(
    run: &str,
    session: SessionId,
    scene_toml: &str,
    scene: &Scene,
    (table, pins): (&Table, &BTreeMap<ParticipantId, Pin>),
    template: &HostConfig,
) -> HostOrder {
    let mut devices = vec![DeviceRecord {
        id: LOCAL_DEVICE.to_string(),
        public_key: template.identity.local_public(),
        max_priority: Some(PriorityClass::Safety),
        parts: vec!["**".into()],
        verbs: vec!["arm".into(), "disarm".into()],
    }];
    let mut robots = Vec::new();
    for member in &scene.members {
        let pin = pins.get(&member.name);
        if let Some(pin) = pin {
            robots.push(RobotPin {
                robot: member.name.clone(),
                device: pin.device.clone(),
                board_id: pin.board_id,
            });
        }
        let device = pin.map(|p| p.device.as_str()).or(member.device());
        let Some((id, entry)) = device.and_then(|d| table.devices.get_key_value(d)) else {
            continue;
        };
        if devices.iter().any(|d| &d.id == id) {
            continue;
        }
        let (Some(public_key), Ok(access)) = (unbase32(&entry.public_key), entry.access()) else {
            continue;
        };
        devices.push(DeviceRecord {
            id: id.clone(),
            public_key,
            max_priority: access.max_priority,
            parts: access.parts,
            verbs: access
                .verbs
                .iter()
                .map(|v| match v {
                    Verb::Arm => "arm".to_string(),
                    Verb::Disarm => "disarm".to_string(),
                })
                .collect(),
        });
    }
    HostOrder {
        run: run.to_string(),
        session,
        scene_toml: scene_toml.to_string(),
        devices,
        pins: robots,
    }
}

/// A fresh run id: `r-` and four letters or digits.
fn run_id(taken: &HashMap<String, Arc<Run>>) -> String {
    const ALPHABET: &[u8] = b"abcdefghijkmnpqrstuvwxyz23456789";
    loop {
        let bits = SessionId::new().as_uuid().as_u128() as u32;
        let id: String = (0..4)
            .map(|i| char::from(ALPHABET[((bits >> (5 * i)) & 31) as usize]))
            .collect();
        let id = format!("r-{id}");
        if !taken.contains_key(&id) {
            return id;
        }
    }
}

impl Home {
    /// The lobby of the daemon whose state is in `dir`: its runs are hosted
    /// by `server` with `template`'s identity, clock and lease grace, and
    /// joined at `addresses`.
    fn new(
        dir: PathBuf,
        template: HostConfig,
        addresses: Vec<String>,
        server: Weak<SessionServer>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            dir,
            template,
            addresses,
            server,
            lobby: OnceLock::new(),
            rooms: Mutex::default(),
            arrivals: watch::channel(0).0,
            me: me.clone(),
        })
    }

    /// Hosts the lobby `_lobby` on `server` (the daemon whose state is in
    /// `dir`, with `config`'s identity, clock and lease grace), its runs
    /// joined at `addresses`.
    ///
    /// # Errors
    /// What starting its session failed with.
    pub async fn start(
        server: &Arc<SessionServer>,
        dir: PathBuf,
        config: HostConfig,
        addresses: Vec<String>,
    ) -> Result<Arc<Self>, SessionError> {
        let home = Self::new(dir, config.clone(), addresses, Arc::downgrade(server));
        let lobby = server
            .start_session(HostConfig {
                session: session_id(LOBBY_SESSION),
                lobby: Some(home.clone()),
                ..config
            })
            .await?;
        let _ = home.lobby.set(lobby);
        Ok(home)
    }

    /// `devices.toml` is now `table`: a run on a satellite whose members'
    /// device records or pins it changed fails (scene design §4.4), since
    /// the satellite holds the ones its order carried. Runs hosted here
    /// take the table live.
    pub fn devices_changed(&self, table: &Table) {
        let Ok((_, pins)) = table.host() else {
            return;
        };
        let runs: Vec<Arc<Run>> = lock(&self.rooms).runs.values().cloned().collect();
        for run in runs {
            let Some((scene, order)) = run.away.get() else {
                continue;
            };
            if over(run.state.borrow().stage) {
                continue;
            }
            let now = host_order(
                &order.run,
                order.session,
                &order.scene_toml,
                scene,
                (table, &pins),
                &self.template,
            );
            if now.devices != order.devices || now.pins != order.pins {
                run.stopping.store(true, Ordering::SeqCst);
                let _ = run.commands.send(RunCommand::End(Outcome::Failed(refused(
                    CtlStatus::NotAuthorized,
                    format!(
                        "{DEVICES} changed a member's device or pin, and run {} on a satellite \
                         holds the old ones: run it again",
                        order.run
                    ),
                ))));
            }
        }
    }

    fn identity(&self) -> &HostIdentity {
        &self.template.identity
    }

    fn run(&self, id: &str) -> Result<Arc<Run>, CtlError> {
        lock(&self.rooms).runs.get(id).cloned().ok_or_else(|| {
            refused(
                CtlStatus::InvalidRequest,
                format!("no run {id} on this daemon"),
            )
        })
    }

    /// A robot's wait: its last failure goes on its run's member, a robot
    /// back from a finished run frees its claim, one its run still waits
    /// for gets its join again, and anyone else waits for an assignment.
    fn wait(&self, from: &ParticipantId, device: &str, wait: LobbyWait) -> LobbyAnswer {
        // Only a device that may embody the name waits under it (§3.2): its
        // pinned device, or one that may pin it.
        let may = Table::read(&self.dir).is_ok_and(|table| match table.robots.get(from.as_str()) {
            Some(pin) => pin.device == device,
            None => table.may_pin(from.as_str(), device),
        });
        if !may {
            return LobbyAnswer::Now(Err(refused(
                CtlStatus::NotAuthorized,
                format!(
                    "{device} may not wait as robot {from}: it is pinned elsewhere, or {device} \
                     may not pin it (provision the board as {from}, or rlmesh devices allow \
                     {device} --verbs arm)"
                ),
            )));
        }
        let mut rooms = lock(&self.rooms);
        if let (Some(last), Some(run)) = (wait.last, rooms.runs.get(&wait.run)) {
            member(&run.state, from, |m| m.error = Some(last));
        }
        if let Some(id) = rooms.claims.get(from).cloned() {
            match rooms.runs.get(&id).cloned() {
                Some(run)
                    if !over(run.state.borrow().stage) && !run.stopping.load(Ordering::SeqCst) =>
                {
                    if let Some(join) = lock(&run.joins).get(from).cloned() {
                        return LobbyAnswer::Now(Ok(LobbyReply::Assignment(Assignment::Join(
                            join,
                        ))));
                    }
                }
                run => {
                    rooms.claims.remove(from);
                    if let Some(run) = run {
                        member(&run.state, from, |m| m.stage = MemberStage::Returned);
                    }
                }
            }
        }
        let (answer, waited) = oneshot::channel();
        rooms.waiting.insert(from.clone(), Waiter { answer });
        drop(rooms);
        self.arrivals.send_modify(|n| *n += 1);
        LobbyAnswer::Later(waited)
    }

    /// Stages 1 and 2: the scene checked and its robots claimed, all or
    /// none; then the rest of the run starts.
    fn run_scene(&self, scene_toml: &str, manual_arm: bool) -> LobbyResult {
        let scene = Scene::parse(scene_toml)
            .map_err(|refusal| refused(CtlStatus::InvalidRequest, refusal))?;
        let table = Table::read(&self.dir)
            .map_err(|err| refused(CtlStatus::InvalidRequest, format!("{err:#}")))?;
        let problems = crate::scene::check(&scene, &table);
        if let Some(first) = problems.first() {
            let status = first.status;
            return Err(refused(
                status,
                rlmesh_session::scene::Refusal {
                    scene: scene.name.clone(),
                    problems,
                },
            ));
        }
        // TODO: a served member needs a catalog model that binds as a scene
        // controller; send its ServeOrder (scene design §4.1 stage 4) once
        // `rlmesh serve` can run one.
        if let Some(m) = scene.members.iter().find(|m| {
            matches!(
                m.role,
                SceneRole::Controller {
                    source: Source::Serve { .. },
                    ..
                }
            )
        }) {
            return Err(refused(
                CtlStatus::Unsupported,
                format!(
                    "{} is served, and no daemon serves a catalog model as a scene controller \
                     yet: run the model yourself as a device member",
                    m.name
                ),
            ));
        }
        let (_, pins) = table
            .host()
            .map_err(|err| refused(CtlStatus::InvalidRequest, format!("{err:#}")))?;
        let satellite = self.placement(&scene, &table)?;
        let robots: Vec<ParticipantId> = scene
            .members
            .iter()
            .filter(|m| matches!(m.role, SceneRole::Robot { .. }))
            .map(|m| m.name.clone())
            .collect();
        let mut rooms = lock(&self.rooms);
        if let Some((robot, other)) = robots
            .iter()
            .find_map(|r| rooms.claims.get(r).map(|run| (r, run.clone())))
        {
            let scene = rooms
                .runs
                .get(&other)
                .map(|r| r.scene.clone())
                .unwrap_or_default();
            return Err(refused(
                CtlStatus::RobotBusy,
                format!("{robot} is in run {other} ({scene}); rlmesh stop {other}"),
            ));
        }
        let id = run_id(&rooms.runs);
        for robot in &robots {
            rooms.claims.insert(robot.clone(), id.clone());
        }
        let members = scene
            .members
            .iter()
            .map(|m| MemberState {
                name: m.name.clone(),
                stage: MemberStage::Waiting,
                error: None,
            })
            .collect();
        let state = RunState {
            run: id.clone(),
            version: 1,
            stage: RunStage::Hosting,
            members,
            error: None,
            session: None,
            addresses: Vec::new(),
            host_key: None,
            scene_fingerprint: Some(scene.fingerprint()),
        };
        let (commands, inbox) = mpsc::unbounded_channel();
        let run = Arc::new(Run {
            scene: scene.name.clone(),
            state: watch::channel(state.clone()).0,
            commands,
            joins: Mutex::default(),
            stopping: AtomicBool::new(false),
            away: OnceLock::new(),
        });
        rooms.runs.insert(id.clone(), run.clone());
        drop(rooms);
        // TODO: `record` is not acted on: rlmeshd records no session yet;
        // start the run's recorder once the daemon takes a recording directory.
        let manual = manual_arm || scene.arm == Arm::Manual;
        let mut host = scene.host(&id, &pins);
        for (name, member) in &mut host.members {
            if let HostMember::Robot { may_pin, .. } = member {
                may_pin.extend(
                    table
                        .devices
                        .keys()
                        .filter(|device| table.may_pin(name.as_str(), device))
                        .cloned(),
                );
            }
        }
        let config = HostConfig {
            session: SessionId::new(),
            scene: Some(host),
            seal: scene.seal(),
            lobby: None,
            ..self.template.clone()
        };
        let satellite = satellite.map(|device| {
            let order = host_order(
                &id,
                config.session,
                scene_toml,
                &scene,
                (&table, &pins),
                &self.template,
            );
            let _ = run.away.set((scene.clone(), order.clone()));
            (device, order)
        });
        tokio::spawn(
            Driver {
                home: self
                    .me
                    .upgrade()
                    .ok_or_else(|| refused(CtlStatus::NotReady, "the daemon is stopping"))?,
                run,
                id,
                scene,
                manual,
                armed: 0,
                inbox,
                daemon: None,
                satellite,
            }
            .drive(config),
        );
        Ok(LobbyReply::Run(state))
    }

    /// Where run `scene` is hosted (§4.1): `None` here, or the satellite
    /// `host =` names, else the machine of its first controller that runs
    /// rlmeshd, when every robot is pinned (only the home daemon pins at
    /// first embodiment).
    ///
    /// # Errors
    /// `NOT_PRESENT` when `host =` names a satellite not in the lobby.
    fn placement(&self, scene: &Scene, table: &Table) -> Result<Option<String>, CtlError> {
        match scene.host.as_deref() {
            Some("home") => return Ok(None),
            Some(host) if self.attached(host) => return Ok(Some(host.to_string())),
            Some(host) => {
                return Err(refused(
                    CtlStatus::NotPresent,
                    format!(
                        "{host} is not attached to this lobby as a satellite: run rlmeshd there, \
                         or host = \"home\""
                    ),
                ));
            }
            None => {}
        }
        let pinned = scene.members.iter().all(|m| {
            !matches!(m.role, SceneRole::Robot { .. }) || table.robots.contains_key(m.name.as_str())
        });
        let first = scene.members.iter().find_map(|m| match m.device() {
            Some(LOCAL_DEVICE) if matches!(m.role, SceneRole::Controller { .. }) => Some(None),
            Some(device)
                if matches!(m.role, SceneRole::Controller { .. })
                    && table.may_host(device)
                    && self.attached(device) =>
            {
                Some(Some(device.to_string()))
            }
            _ => None,
        });
        Ok(first.flatten().filter(|_| pinned))
    }

    /// Whether device `device`'s satellite waits in the lobby.
    fn attached(&self, device: &str) -> bool {
        let Ok(id) = ParticipantId::new(format!("{SATELLITE_PREFIX}{device}")) else {
            return false;
        };
        self.lobby.get().is_some_and(|lobby| {
            lobby
                .borrow()
                .participant(&id)
                .is_some_and(|p| p.present && p.device == device)
        })
    }

    /// A satellite's wait: its next order, now or once there is one. Only
    /// a paired machine with the `arm` verb waits as a satellite, under its
    /// own device's name.
    fn satellite_wait(&self, from: &ParticipantId, device: &str) -> LobbyAnswer {
        let may = Table::read(&self.dir).is_ok_and(|table| table.may_host(device));
        if !may || from.as_str() != format!("{SATELLITE_PREFIX}{device}") {
            return LobbyAnswer::Now(Err(refused(
                CtlStatus::NotAuthorized,
                format!(
                    "{from} from {device}: a satellite is a paired machine with the arm verb, \
                     waiting as {SATELLITE_PREFIX}{device}"
                ),
            )));
        }
        let mut rooms = lock(&self.rooms);
        let satellite = rooms.satellites.entry(device.to_string()).or_default();
        if let Some(order) = satellite.orders.pop_front() {
            return LobbyAnswer::Now(Ok(LobbyReply::Assignment(order)));
        }
        let (answer, waited) = oneshot::channel();
        satellite.waiter = Some(Waiter { answer });
        LobbyAnswer::Later(waited)
    }

    /// Sends satellite `device` `order`: to its wait, else when it next
    /// waits.
    fn order(&self, device: &str, order: Assignment) {
        let mut rooms = lock(&self.rooms);
        let satellite = rooms.satellites.entry(device.to_string()).or_default();
        let order = match satellite.waiter.take() {
            Some(waiter) => match waiter.answer.send(Ok(LobbyReply::Assignment(order))) {
                Ok(()) => return,
                Err(Ok(LobbyReply::Assignment(order))) => order,
                Err(_) => return,
            },
            None => order,
        };
        satellite.orders.push_back(order);
    }

    /// Satellite `device`'s answer to the host order of `started.run`.
    fn host_started(&self, device: &str, started: HostStarted) -> LobbyResult {
        let waiting = lock(&self.rooms)
            .satellites
            .get_mut(device)
            .and_then(|s| s.started.remove(&started.run));
        let Some(waiting) = waiting else {
            return Err(refused(
                CtlStatus::InvalidRequest,
                format!("{device} has no host order for run {}", started.run),
            ));
        };
        let run = self.run(&started.run)?;
        let _ = waiting.send(started.result);
        Ok(LobbyReply::Run(run.state.borrow().clone()))
    }

    /// The robot `robot`'s wait, if it is still waiting.
    fn take_waiter(&self, robot: &ParticipantId) -> Option<Waiter> {
        let mut rooms = lock(&self.rooms);
        let waiter = rooms.waiting.remove(robot)?;
        (!waiter.answer.is_closed()).then_some(waiter)
    }

    /// Releases run `id`'s claims on `robots`.
    fn release(&self, id: &str, robots: &[ParticipantId]) {
        let mut rooms = lock(&self.rooms);
        for robot in robots {
            if rooms.claims.get(robot).is_some_and(|run| run == id) {
                rooms.claims.remove(robot);
            }
        }
    }
}

impl LobbyService for Home {
    fn handle(&self, from: &ParticipantId, device: &str, command: Command) -> LobbyAnswer {
        let now = |result| LobbyAnswer::Now(result);
        match command {
            Command::LobbyWait(_) if from.as_str().starts_with(SATELLITE_PREFIX) => {
                self.satellite_wait(from, device)
            }
            Command::LobbyWait(wait) => self.wait(from, device, wait),
            Command::RunScene {
                scene_toml,
                manual_arm,
            } => now(self.run_scene(&scene_toml, manual_arm)),
            Command::SceneStatus { run, after } => {
                let run = match self.run(&run) {
                    Ok(run) => run,
                    Err(err) => return now(Err(err)),
                };
                let mut states = run.state.subscribe();
                let current = states.borrow_and_update().clone();
                if current.version > after {
                    return now(Ok(LobbyReply::Run(current)));
                }
                let (mut answer, waited) = oneshot::channel();
                tokio::spawn(async move {
                    tokio::select! {
                        () = answer.closed() => {}
                        changed = states.wait_for(|s| s.version > after) => {
                            let result = changed
                                .map(|s| LobbyReply::Run(s.clone()))
                                .map_err(|_| refused(CtlStatus::Expired, "the run is gone"));
                            let _ = answer.send(result);
                        }
                    }
                });
                LobbyAnswer::Later(waited)
            }
            Command::StopScene { run } => now(self.run(&run).map(|run| {
                run.stopping.store(true, Ordering::SeqCst);
                let _ = run
                    .commands
                    .send(RunCommand::End(Outcome::Stopped(from.clone())));
                LobbyReply::Run(run.state.borrow().clone())
            })),
            Command::ArmScene { run } => match self.run(&run) {
                Ok(run) => {
                    let (answer, waited) = oneshot::channel();
                    if let Err(mpsc::error::SendError(RunCommand::Arm { answer })) =
                        run.commands.send(RunCommand::Arm { answer })
                    {
                        let _ = answer.send(Err(refused(
                            CtlStatus::InvalidTransition,
                            format!("run {} is over", run.state.borrow().run),
                        )));
                    }
                    LobbyAnswer::Later(waited)
                }
                Err(err) => now(Err(err)),
            },
            Command::HostStarted(started) => now(self.host_started(device, started)),
            _ => now(Err(refused(CtlStatus::InvalidRequest, "not a lobby verb"))),
        }
    }
}

/// Changes member `name` of a run's state (the version rises if it
/// changed).
fn member(state: &watch::Sender<RunState>, name: &ParticipantId, f: impl FnOnce(&mut MemberState)) {
    update(state, |s| {
        if let Some(m) = s.members.iter_mut().find(|m| &m.name == name) {
            f(m);
        }
    });
}

fn update(state: &watch::Sender<RunState>, f: impl FnOnce(&mut RunState)) {
    state.send_if_modified(|s| {
        let before = s.clone();
        f(s);
        let changed = *s != before;
        if changed {
            s.version += 1;
        }
        changed
    });
}

/// How a run left its stages.
enum Outcome {
    /// Stopped by a client.
    Stopped(ParticipantId),
    /// Failed: why.
    Failed(CtlError),
}

/// One run's stages 3 to 9.
struct Driver {
    home: Arc<Home>,
    run: Arc<Run>,
    id: String,
    scene: Scene,
    manual: bool,
    /// The authority revision the run was last armed at: an older snapshot
    /// is from before that arm.
    armed: u64,
    inbox: mpsc::UnboundedReceiver<RunCommand>,
    daemon: Option<Participant>,
    /// The satellite hosting the run, by device, and its host order.
    satellite: Option<(String, HostOrder)>,
}

/// Each member's stage in `snapshot`, from what it had: present, declared,
/// bound; a member present before and absent now is lost. A robot back in
/// the lobby stays returned: a snapshot from before it left can still arrive.
fn stage_in(snapshot: &AuthoritySnapshot, scene: &Scene, m: &MemberState) -> MemberStage {
    if m.stage == MemberStage::Returned {
        return MemberStage::Returned;
    }
    let info = snapshot.participant(&m.name);
    let present = info.is_some_and(|p| p.present);
    let robot = matches!(
        scene.member(m.name.as_str()).map(|m| &m.role),
        Some(SceneRole::Robot { .. })
    );
    let joined = matches!(
        m.stage,
        MemberStage::Joined | MemberStage::Declared | MemberStage::Bound | MemberStage::Lost
    );
    if !present {
        return if joined { MemberStage::Lost } else { m.stage };
    }
    // A robot back from a loss stays `left` until it declared again.
    let declared = info.is_some_and(|p| p.left.is_none());
    if robot && declared && snapshot.parts.iter().any(|p| p.owner == m.name) {
        MemberStage::Declared
    } else if !robot && snapshot.binding(&m.name).is_some() {
        MemberStage::Bound
    } else {
        MemberStage::Joined
    }
}

impl Driver {
    fn server(&self) -> Result<Arc<SessionServer>, CtlError> {
        self.home
            .server
            .upgrade()
            .ok_or_else(|| refused(CtlStatus::NotReady, "the daemon is stopping"))
    }

    fn state(&self) -> RunState {
        self.run.state.borrow().clone()
    }

    fn update(&self, f: impl FnOnce(&mut RunState)) {
        update(&self.run.state, f);
    }

    fn robots(&self) -> Vec<ParticipantId> {
        self.scene
            .members
            .iter()
            .filter(|m| matches!(m.role, SceneRole::Robot { .. }))
            .map(|m| m.name.clone())
            .collect()
    }

    async fn drive(mut self, config: HostConfig) {
        let session = config.session;
        let outcome = self.stages(config).await;
        self.tear_down(session, outcome).await;
    }

    /// Stages 3 to 8, until the run stops or fails.
    async fn stages(&mut self, config: HostConfig) -> Outcome {
        let snapshots = match self.host(config).await {
            Ok(snapshots) => snapshots,
            Err(err) => return Outcome::Failed(err),
        };
        match self.gather(snapshots.clone()).await {
            Ok(()) => {}
            Err(outcome) => return outcome,
        }
        let mut manual = self.manual;
        loop {
            if let Err(outcome) = self.arm_stage(snapshots.clone(), manual).await {
                return outcome;
            }
            if let Some(outcome) = self.running(snapshots.clone()).await {
                return outcome;
            }
            if let Err(outcome) = self.rejoin(snapshots.clone()).await {
                return outcome;
            }
            // Back from a loss, the run arms only when asked (§4.3).
            manual = true;
        }
    }

    /// Stage 3: the run's session, hosted here or on its satellite, and the
    /// daemon in it.
    async fn host(
        &mut self,
        config: HostConfig,
    ) -> Result<watch::Receiver<AuthoritySnapshot>, CtlError> {
        if let Some((device, order)) = self.satellite.clone() {
            return self.host_on(&device, order).await;
        }
        let session = config.session;
        let server = self.server()?;
        let started = async {
            let snapshots = server.start_session(config).await?;
            let daemon = server
                .attach(
                    session,
                    ParticipantId::new(HOME)?,
                    vec![Role::Observer],
                    PriorityClass::Scripted,
                )
                .await?;
            Ok::<_, SessionError>((snapshots, daemon))
        };
        let (snapshots, daemon) = started
            .await
            .map_err(|err| refused(CtlStatus::NotReady, format!("the run's host: {err}")))?;
        self.daemon = Some(daemon);
        let host_key = *self.home.identity().static_public();
        let addresses = self.home.addresses.clone();
        self.update(|s| {
            s.session = Some(session);
            s.addresses = addresses;
            s.host_key = Some(host_key);
            s.stage = RunStage::Assigning;
        });
        Ok(snapshots)
    }

    /// Stage 3 on satellite `device`: its host order, its answer, and the
    /// daemon attached there as `local`, the run's state followed through
    /// it until the satellite's host is gone.
    async fn host_on(
        &mut self,
        device: &str,
        order: HostOrder,
    ) -> Result<watch::Receiver<AuthoritySnapshot>, CtlError> {
        let session = order.session;
        let (answer, started) = oneshot::channel();
        lock(&self.home.rooms)
            .satellites
            .entry(device.to_string())
            .or_default()
            .started
            .insert(self.id.clone(), answer);
        self.home.order(device, Assignment::Host(order));
        let wait = self.scene.wait;
        let not_started = || {
            refused(
                CtlStatus::NotPresent,
                format!(
                    "{device} did not start the run's host (waited {} s)",
                    wait.as_secs_f64()
                ),
            )
        };
        let (host_key, addresses) = tokio::time::timeout(wait, started)
            .await
            .map_err(|_| not_started())?
            .map_err(|_| not_started())??;
        let base = addresses
            .iter()
            .find_map(|a| a.to_socket_addrs().ok()?.next())
            .ok_or_else(|| {
                refused(
                    CtlStatus::NotReady,
                    format!("{device} hosts the run at none of {addresses:?}"),
                )
            })?;
        let address = SessionAddress {
            credential: Some(self.home.identity().local_credential().with_host(host_key)),
            ..SessionAddress::new(base, &session.to_string())
                .map_err(|err| refused(CtlStatus::NotReady, err))?
        };
        let daemon = address
            .join(HOME, vec![Role::Observer], PriorityClass::Scripted, wait)
            .await
            .map_err(|err| {
                refused(
                    CtlStatus::NotReady,
                    format!("the run's host on {device}: {err}"),
                )
            })?;
        let snapshots = follow(&daemon).await.map_err(|err| {
            refused(
                CtlStatus::NotReady,
                format!("the run's host on {device}: {err}"),
            )
        })?;
        self.daemon = Some(daemon);
        self.update(|s| {
            s.session = Some(session);
            s.addresses = addresses;
            s.host_key = Some(host_key);
            s.stage = RunStage::Assigning;
        });
        Ok(snapshots)
    }

    /// Stages 4 to 6: robots assigned as they wait, declared, and every
    /// controller bound, all by `wait`.
    async fn gather(
        &mut self,
        mut snapshots: watch::Receiver<AuthoritySnapshot>,
    ) -> Result<(), Outcome> {
        let deadline = tokio::time::Instant::now() + self.scene.wait;
        let mut arrivals = self.home.arrivals.subscribe();
        loop {
            self.assign();
            let snapshot = snapshots.borrow_and_update().clone();
            self.refresh(&snapshot);
            let state = self.state();
            let stage = gathering(&state, &self.robots());
            if stage == RunStage::Arming {
                return Ok(());
            }
            self.update(|s| s.stage = stage);
            tokio::select! {
                () = tokio::time::sleep_until(deadline) => {
                    return Err(Outcome::Failed(self.late(&self.state())));
                }
                changed = snapshots.changed() => if changed.is_err() {
                    return Err(Outcome::Failed(refused(CtlStatus::NotReady, "the run's host stopped")));
                },
                _ = arrivals.changed() => {}
                Some(command) = self.inbox.recv() => match command {
                    RunCommand::End(outcome) => return Err(outcome),
                    RunCommand::Arm { answer } => {
                        let _ = answer.send(Err(refused(
                            CtlStatus::InvalidTransition,
                            format!("run {} is not ready to arm: {:?}", self.id, self.state().stage),
                        )));
                    }
                },
            }
        }
    }

    /// Sends a join to every robot member waiting in the lobby that has
    /// none yet.
    fn assign(&self) {
        let state = self.state();
        let Some(session) = state.session else {
            return;
        };
        for robot in self.robots() {
            if lock(&self.run.joins).contains_key(&robot) {
                continue;
            }
            let Some(waiter) = self.home.take_waiter(&robot) else {
                continue;
            };
            let Some(host_key) = state.host_key else {
                return;
            };
            // A satellite's run host has keys of its own and no host id:
            // a robot finds it at the join's addresses, or through this
            // lobby again.
            let host_id = match self.satellite {
                Some(_) => String::new(),
                None => self.home.identity().host_id(),
            };
            let join = Join {
                run: self.id.clone(),
                session,
                robot: robot.clone(),
                addresses: state.addresses.clone(),
                host_key,
                host_id,
                tenant: TENANT.to_string(),
            };
            let sent = waiter
                .answer
                .send(Ok(LobbyReply::Assignment(Assignment::Join(join.clone()))));
            if sent.is_ok() {
                lock(&self.run.joins).insert(robot.clone(), join);
                member(&self.run.state, &robot, |m| m.stage = MemberStage::Assigned);
            }
        }
    }

    /// Every member's stage from `snapshot`.
    fn refresh(&self, snapshot: &AuthoritySnapshot) {
        let scene = &self.scene;
        self.update(|s| {
            for m in &mut s.members {
                m.stage = stage_in(snapshot, scene, m);
            }
        });
    }

    /// Why the run failed at `wait`: its first member not done.
    fn late(&self, state: &RunState) -> CtlError {
        let wait = self.scene.wait.as_secs_f64();
        for m in &state.members {
            let robot = self.robots().contains(&m.name);
            let error = match (robot, m.stage) {
                (true, MemberStage::Waiting) => refused(
                    CtlStatus::NotPresent,
                    format!("{} is not in the lobby (waited {wait} s)", m.name),
                ),
                (true, MemberStage::Assigned | MemberStage::Joined) => {
                    m.error.clone().unwrap_or_else(|| {
                        refused(
                            CtlStatus::NotPresent,
                            format!("{} has not declared (waited {wait} s)", m.name),
                        )
                    })
                }
                (false, MemberStage::Waiting | MemberStage::Joined) => refused(
                    CtlStatus::NotReady,
                    format!("{} has not bound (waited {wait} s)", m.name),
                ),
                _ => continue,
            };
            let failed = error.clone();
            member(&self.run.state, &m.name, |m| m.error = Some(failed));
            return error;
        }
        refused(CtlStatus::NotReady, "the run did not get ready in time")
    }

    /// Stage 7: armed by the daemon, or on an `ArmScene` (or straight at
    /// the host by an operator).
    async fn arm_stage(
        &mut self,
        mut snapshots: watch::Receiver<AuthoritySnapshot>,
        manual: bool,
    ) -> Result<(), Outcome> {
        self.update(|s| s.stage = RunStage::Arming);
        if !manual {
            return self.arm().await.map_err(Outcome::Failed);
        }
        loop {
            let snapshot = snapshots.borrow_and_update().clone();
            if matches!(snapshot.mode, Mode::Armed | Mode::Active) {
                self.armed = snapshot.revision;
                self.update(|s| s.stage = RunStage::Running);
                return Ok(());
            }
            tokio::select! {
                changed = snapshots.changed() => if changed.is_err() {
                    return Err(Outcome::Failed(refused(CtlStatus::NotReady, "the run's host stopped")));
                },
                Some(command) = self.inbox.recv() => match command {
                    RunCommand::End(outcome) => return Err(outcome),
                    RunCommand::Arm { answer } => {
                        let armed = self.arm().await;
                        let _ = answer.send(armed.clone().map(|()| LobbyReply::Run(self.state())));
                        if armed.is_ok() {
                            return Ok(());
                        }
                    }
                },
            }
        }
    }

    /// Arms the run's session as the daemon; `RUNNING` once armed.
    async fn arm(&mut self) -> Result<(), CtlError> {
        let daemon = self
            .daemon
            .as_ref()
            .ok_or_else(|| refused(CtlStatus::NotReady, "the daemon is not in the run"))?;
        match daemon.arm().await {
            Ok(armed) => {
                self.armed = armed.revision;
                self.update(|s| s.stage = RunStage::Running);
                Ok(())
            }
            Err(SessionError::Rejected(err)) => Err(err),
            Err(err) => Err(refused(CtlStatus::NotReady, format!("arm: {err}"))),
        }
    }

    /// Stage 8: until stopped (`Some`), or a member is lost or the session
    /// is no longer armed (`None`). A loss and its rejoin may come in one
    /// snapshot (a watch keeps only the latest), so a session that left
    /// ARMED and ACTIVE after the arm needs an arm again whether or not the
    /// rejoin window was seen.
    async fn running(
        &mut self,
        mut snapshots: watch::Receiver<AuthoritySnapshot>,
    ) -> Option<Outcome> {
        loop {
            let snapshot = snapshots.borrow_and_update().clone();
            self.refresh(&snapshot);
            let rejoining = snapshot
                .scene
                .as_ref()
                .is_some_and(|s| s.rejoin_until_ns.is_some());
            let disarmed = snapshot.revision >= self.armed
                && !matches!(snapshot.mode, Mode::Armed | Mode::Active);
            if rejoining || disarmed {
                return None;
            }
            tokio::select! {
                changed = snapshots.changed() => if changed.is_err() {
                    return Some(Outcome::Failed(refused(CtlStatus::NotReady, "the run's host stopped")));
                },
                Some(command) = self.inbox.recv() => match command {
                    RunCommand::End(outcome) => return Some(outcome),
                    RunCommand::Arm { answer } => {
                        let _ = answer.send(Err(refused(
                            CtlStatus::InvalidTransition,
                            format!("run {} is already running", self.id),
                        )));
                    }
                },
            }
        }
    }

    /// Stage 8a (§4.3): waits for every lost member to come back (a robot
    /// re-sent its join through the lobby, declaring its map again) until
    /// the session's rejoin window closes, which fails the run. Back means
    /// the host closed the window and every member is where it was: robots
    /// declared, controllers bound.
    async fn rejoin(
        &mut self,
        mut snapshots: watch::Receiver<AuthoritySnapshot>,
    ) -> Result<(), Outcome> {
        let mut deadline = None;
        loop {
            let snapshot = snapshots.borrow_and_update().clone();
            self.refresh(&snapshot);
            let window = snapshot.scene.as_ref().and_then(|s| s.rejoin_until_ns);
            if window.is_none() && gathering(&self.state(), &self.robots()) == RunStage::Arming {
                return Ok(());
            }
            self.update(|s| s.stage = RunStage::Rejoining);
            // On the host's clock, as of this snapshot.
            let left = window.map_or(self.scene.rejoin, |until| {
                Duration::from_nanos(until.saturating_sub(snapshot.host_ns).max(0).unsigned_abs())
            });
            let deadline = *deadline.get_or_insert(tokio::time::Instant::now() + left);
            tokio::select! {
                () = tokio::time::sleep_until(deadline) => {
                    let lost: Vec<String> = self
                        .state()
                        .members
                        .iter()
                        .filter(|m| matches!(m.stage, MemberStage::Lost | MemberStage::Joined))
                        .map(|m| m.name.to_string())
                        .collect();
                    return Err(Outcome::Failed(refused(
                        CtlStatus::NotPresent,
                        format!(
                            "{} did not rejoin within {} s",
                            lost.join(", "),
                            self.scene.rejoin.as_secs_f64()
                        ),
                    )));
                }
                changed = snapshots.changed() => if changed.is_err() {
                    return Err(Outcome::Failed(refused(CtlStatus::NotReady, "the run's host stopped")));
                },
                Some(command) = self.inbox.recv() => match command {
                    RunCommand::End(outcome) => return Err(outcome),
                    RunCommand::Arm { answer } => {
                        let _ = answer.send(Err(refused(
                            CtlStatus::NotReady,
                            format!("run {} waits for its lost members to rejoin", self.id),
                        )));
                    }
                },
            }
        }
    }

    /// Stage 9: closes the scene, waits up to 2 s for its members to
    /// detach, stops its host, and frees the claims of robots that never
    /// left the lobby (the others' on their return, or after `wait`).
    async fn tear_down(mut self, session: SessionId, outcome: Outcome) {
        let (code, by, detail, error) = match &outcome {
            Outcome::Stopped(by) => (ReasonCode::SceneEnded, by.to_string(), String::new(), None),
            Outcome::Failed(err) => (
                ReasonCode::SceneFailed,
                HOME.to_string(),
                err.message.clone(),
                Some(err.clone()),
            ),
        };
        self.update(|s| {
            s.stage = RunStage::Ending;
            s.error.clone_from(&error);
        });
        if let Some((device, _)) = self.satellite.clone() {
            self.home.order(
                &device,
                Assignment::Stop(StopOrder {
                    run: self.id.clone(),
                    reason: Reason {
                        by,
                        detail,
                        ..Reason::new(code)
                    },
                }),
            );
            if let Some(daemon) = self.daemon.take() {
                // The satellite gives the members 2 s, then stops its host.
                let mut present = daemon.host_present();
                let _ = tokio::time::timeout(STRAGGLERS * 2, present.wait_for(|p| !p)).await;
                let _ = daemon.detach().await;
            }
        } else if let Ok(server) = self.server() {
            if server
                .close_scene(session, code, &by, &detail)
                .await
                .is_ok()
                && let Some(daemon) = &self.daemon
            {
                let members: Vec<ParticipantId> = self
                    .state()
                    .members
                    .iter()
                    .map(|m| m.name.clone())
                    .collect();
                let _ = daemon
                    .wait_for("members to detach", STRAGGLERS, |s| {
                        !members
                            .iter()
                            .any(|m| s.participant(m).is_some_and(|p| p.present))
                    })
                    .await;
            }
            if let Some(daemon) = self.daemon.take() {
                let _ = daemon.detach().await;
            }
            server.stop(session).await;
        }
        let never_left: Vec<ParticipantId> = self
            .robots()
            .into_iter()
            .filter(|r| !lock(&self.run.joins).contains_key(r))
            .collect();
        self.home.release(&self.id, &never_left);
        // A robot back in the lobby within milliseconds may come after the
        // host stopped: the run ends once its robots are back (their claims
        // released, `RETURNED`), or 2 s on.
        let joined: Vec<ParticipantId> = lock(&self.run.joins).keys().cloned().collect();
        let mut states = self.run.state.subscribe();
        let _ = tokio::time::timeout(
            STRAGGLERS,
            states.wait_for(|s| {
                s.members
                    .iter()
                    .filter(|m| joined.contains(&m.name))
                    .all(|m| m.stage == MemberStage::Returned)
            }),
        )
        .await;
        self.update(|s| {
            s.stage = if error.is_some() {
                RunStage::Failed
            } else {
                RunStage::Ended
            };
        });
        let (home, id, robots, wait) = (
            self.home.clone(),
            self.id.clone(),
            self.robots(),
            self.scene.wait,
        );
        tokio::spawn(async move {
            tokio::time::sleep(wait).await;
            home.release(&id, &robots);
        });
    }
}

/// `daemon`'s authority state as it changes, until its host is gone.
async fn follow(daemon: &Participant) -> Result<watch::Receiver<AuthoritySnapshot>, SessionError> {
    let mut states = daemon.watch();
    let mut present = daemon.host_present();
    let held = states.borrow_and_update().clone();
    let first = match held {
        Some(state) => state,
        None => daemon.fetch_authority().await?,
    };
    let (send, snapshots) = watch::channel(first);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                changed = states.changed() => {
                    if changed.is_err() {
                        return;
                    }
                    if let Some(state) = states.borrow_and_update().clone() {
                        send.send_replace(state);
                    }
                }
                _ = present.wait_for(|present| !present) => return,
                () = send.closed() => return,
            }
        }
    });
    Ok(snapshots)
}

/// The stage a run gathering its members is at: `ASSIGNING` while a robot
/// has no join, `DECLARING` while one hasn't declared, `BINDING` while a
/// controller hasn't bound, else ready to arm.
fn gathering(state: &RunState, robots: &[ParticipantId]) -> RunStage {
    let robot = |m: &&MemberState| robots.contains(&m.name);
    let members = || state.members.iter();
    if members()
        .filter(robot)
        .any(|m| m.stage == MemberStage::Waiting)
    {
        RunStage::Assigning
    } else if members()
        .filter(robot)
        .any(|m| m.stage != MemberStage::Declared)
    {
        RunStage::Declaring
    } else if members()
        .filter(|m| !robot(m))
        .any(|m| m.stage != MemberStage::Bound)
    {
        RunStage::Binding
    } else {
        RunStage::Arming
    }
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, SocketAddr};

    use rlmesh_session::SystemMonotonicClock;
    use rlmesh_session::address::{self, DEFAULT_SESSION, SessionAddress};
    use rlmesh_session::auth::Credential;
    use rlmesh_session::auth::address::{AddressKey, base32};
    use rlmesh_session::crypto::StaticKeypair;
    use rlmesh_session::edge::{EdgeError, Robot};
    use rlmesh_session::lobby::{
        Lobby, LobbyClient, SatelliteClient, home_lobby, run_address, wait_for_join,
    };
    use rlmesh_session::pairing::{HostFile, Paired};
    use rlmesh_session::sdk;
    use rlmesh_session::spec::RunOptions;
    use rlmesh_session::spec::apply::{Tensor, Value};
    use rlmesh_session::wire::{DType, PairInfo, TenantId};
    use rlmesh_session::{Controller, Observation, Predictor, RunError};

    use super::*;

    const WAIT: Duration = Duration::from_secs(10);
    const ARM: &str = r#"{"arm": {"kind": "joints", "count": 2, "fields": ["pos"], "hz": 100,
        "command": "position", "hold_max_ms": 100}}"#;
    const PILOT: &str = r#"{"hz": 20, "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]}},
        "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position", "hz": 20, "horizon": 2}}}"#;

    fn scene(wait: &str) -> String {
        format!(
            "[scene]\nformat = 1\nname = \"bench\"\nwait = \"{wait}\"\n\n\
             [members.arm]\nrole = \"robot\"\n\n\
             [members.pilot]\nrole = \"controller\"\ndevice = \"local\"\n"
        )
    }

    /// A home daemon in `dir` on a loopback port: its lobby's address.
    async fn home(dir: &std::path::Path) -> (Arc<SessionServer>, SessionAddress) {
        let (server, lobby, _) = home_gated(dir, true).await;
        (server, lobby)
    }

    /// [`home`], its links gated by its key's password or (`gated` false)
    /// by none, as rlmeshd's: paired machines link to it too. Its devices
    /// are `devices.toml`'s.
    async fn home_gated(
        dir: &std::path::Path,
        gated: bool,
    ) -> (Arc<SessionServer>, SessionAddress, Arc<Home>) {
        let identity = Arc::new(crate::daemon::identity(dir).expect("identity"));
        let key = AddressKey::of(&identity);
        let password = gated.then(|| key.link_password());
        let (transport, base) = address::open_listeners(
            SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
            password.as_ref(),
        )
        .await
        .expect("listeners");
        let mut config = HostConfig::new(
            TenantId::new(TENANT).expect("tenant"),
            session_id(DEFAULT_SESSION),
        );
        config.identity = identity;
        config.clock = SystemMonotonicClock.shared();
        config.health_log = None;
        (config.devices, config.pins) = Table::read(dir)
            .and_then(|table| table.host())
            .expect("devices.toml");
        let server = Arc::new(
            SessionServer::start(transport, config.clone())
                .await
                .expect("server"),
        );
        let home = Home::start(&server, dir.into(), config, vec![base.to_string()])
            .await
            .expect("lobby");
        let lobby = SessionAddress {
            key: Some(key),
            ..home_lobby(Some(&base.to_string())).expect("lobby address")
        };
        (server, lobby, home)
    }

    /// A robot process: waits in the lobby, joins its run, stays until the
    /// run closes, then comes back to the lobby.
    async fn robot(lobby: SessionAddress) {
        let mut me = Lobby::new(ParticipantId::new("arm").expect("id"));
        let (_, address) = wait_for_join(&lobby, &mut me, "arm", Some(WAIT * 3))
            .await
            .expect("a join");
        let robot = sdk::connect_robot(&address, "arm", ARM, WAIT)
            .await
            .expect("joined");
        robot
            .participant()
            .wait_for("closing", WAIT * 3, |s| {
                s.scene.as_ref().is_some_and(|s| s.closing.is_some())
            })
            .await
            .expect("closing");
        robot.detach().await.expect("detach");
        let back = wait_for_join(&lobby, &mut me, "arm", Some(Duration::from_secs(1))).await;
        assert!(back.is_err(), "no second run assigns it");
    }

    async fn until(
        client: &LobbyClient,
        mut state: RunState,
        done: impl Fn(&RunState) -> bool,
    ) -> RunState {
        while !done(&state) {
            assert!(
                !matches!(state.stage, RunStage::Ended | RunStage::Failed) || done(&state),
                "run ended early: {state:?}"
            );
            state = client
                .status(&state.run, state.version, Some(WAIT * 2))
                .await
                .expect("status");
        }
        state
    }

    fn stage(state: &RunState, name: &str) -> MemberStage {
        state
            .members
            .iter()
            .find(|m| m.name.as_str() == name)
            .map(|m| m.stage)
            .expect("member")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_lobby_robot_is_assigned_armed_once_its_policy_bound_claimed_and_returned() {
        let dir = tempfile::tempdir().expect("dir");
        let (server, lobby) = home(dir.path()).await;
        let robot = tokio::spawn(robot(lobby.clone()));
        let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
        let state = client.run_scene(&scene("20s"), false).await.expect("run");
        assert_eq!(state.stage, RunStage::Hosting);

        // Declared, but its controller hasn't bound: no arm yet.
        let state = until(&client, state, |s| s.stage == RunStage::Binding).await;
        assert_eq!(stage(&state, "arm"), MemberStage::Declared);
        let address = run_address(&lobby, &state)
            .expect("credential")
            .expect("hosted");
        let busy = client
            .run_scene(&scene("20s"), false)
            .await
            .expect_err("arm is claimed");
        match busy {
            SessionError::Rejected(err) => {
                assert_eq!(err.status, CtlStatus::RobotBusy);
                assert!(err.message.contains(&state.run), "{}", err.message);
            }
            other => panic!("{other}"),
        }

        let pilot = sdk::bind_policy(
            &address,
            "pilot",
            PILOT,
            &RunOptions::named("pilot"),
            PriorityClass::Model,
            WAIT,
        )
        .await
        .expect("bound");
        let state = until(&client, state, |s| s.stage == RunStage::Running).await;
        assert_eq!(stage(&state, "pilot"), MemberStage::Bound);
        pilot
            .participant()
            .wait_for("armed", WAIT, |s| {
                matches!(s.mode, Mode::Armed | Mode::Active)
            })
            .await
            .expect("armed");

        client.stop(&state.run).await.expect("stop");
        let state = until(&client, state, |s| {
            s.stage == RunStage::Ended && stage(s, "arm") == MemberStage::Returned
        })
        .await;
        assert_eq!(state.error, None);
        drop(pilot);
        robot.await.expect("robot");
        // Its claim went with its return: the robot may run again.
        let again = client.run_scene(&scene("1s"), false).await.expect("free");
        assert_eq!(again.stage, RunStage::Hosting);
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_robot_back_in_the_lobby_as_its_run_ends_is_returned_and_free_for_the_next() {
        const RUNS: usize = 8;
        let dir = tempfile::tempdir().expect("dir");
        let (server, lobby, service) = home_gated(dir.path(), true).await;
        // A robot that is back in the lobby within milliseconds of seeing
        // its run close, as a board is.
        let robot = tokio::spawn({
            let lobby = lobby.clone();
            async move {
                let mut me = Lobby::new(ParticipantId::new("arm").expect("id"));
                for _ in 0..RUNS {
                    let (_, address) = wait_for_join(&lobby, &mut me, "arm", Some(WAIT * 3))
                        .await
                        .expect("a join");
                    let robot = sdk::connect_robot(&address, "arm", ARM, WAIT)
                        .await
                        .expect("joined");
                    robot
                        .participant()
                        .wait_for("closing", WAIT * 3, |s| {
                            s.scene.as_ref().is_some_and(|s| s.closing.is_some())
                        })
                        .await
                        .expect("closing");
                    robot.detach().await.expect("detach");
                }
                let back =
                    wait_for_join(&lobby, &mut me, "arm", Some(Duration::from_secs(1))).await;
                assert!(back.is_err(), "no other run assigns it");
            }
        });
        let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
        // The robot alone: its run stops as soon as it left.
        let alone = scene("20s").replace(
            "\n[members.pilot]\nrole = \"controller\"\ndevice = \"local\"\n",
            "",
        );
        for _ in 0..RUNS {
            let state = client.run_scene(&alone, false).await.expect("free");
            let state = until(&client, state, |s| s.stage == RunStage::Running).await;
            client.stop(&state.run).await.expect("stop");
            let state = until(&client, state, |s| s.stage == RunStage::Ended).await;
            assert_eq!(stage(&state, "arm"), MemberStage::Returned, "{state:?}");
        }
        robot.await.expect("robot");

        // Back in the lobby once its run is stopping, before it is ending:
        // returned at once, not sent into the closing run.
        let arm = ParticipantId::new("arm").expect("id");
        let wait = || match service.handle(&arm, "local", Command::LobbyWait(LobbyWait::default()))
        {
            LobbyAnswer::Later(waited) => waited,
            LobbyAnswer::Now(answer) => panic!("not a wait: {answer:?}"),
        };
        let waited = wait();
        let state = client.run_scene(&alone, false).await.expect("free");
        let assigned = waited.await.expect("answered");
        assert!(
            matches!(assigned, Ok(LobbyReply::Assignment(Assignment::Join(_)))),
            "{assigned:?}"
        );
        let stopped = std::time::Instant::now();
        // As a stop does before the run hears of it.
        let run = service.run(&state.run).expect("run");
        run.stopping.store(true, Ordering::SeqCst);
        let back = wait();
        let stage_then = run.state.borrow().stage;
        assert!(!over(stage_then), "{stage_then:?}");
        client.stop(&state.run).await.expect("stop");
        let state = until(&client, state, |s| s.stage == RunStage::Ended).await;
        assert_eq!(stage(&state, "arm"), MemberStage::Returned, "{state:?}");
        assert!(
            stopped.elapsed() < STRAGGLERS / 2,
            "{:?}",
            stopped.elapsed()
        );
        client.run_scene(&alone, false).await.expect("free at once");
        let next = back.await.expect("answered");
        assert!(
            matches!(next, Ok(LobbyReply::Assignment(Assignment::Join(_)))),
            "{next:?}"
        );
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_robot_process_carries_its_uncleared_estop_into_its_next_run() {
        let dir = tempfile::tempdir().expect("dir");
        let (server, lobby) = home(dir.path()).await;
        let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
        let alone = scene("20s").replace(
            "\n[members.pilot]\nrole = \"controller\"\ndevice = \"local\"\n",
            "",
        );
        let from_lobby = || sdk::robot_from_lobby_at(&lobby, "arm", ARM, Some(WAIT * 3));
        let state = client.run_scene(&alone, false).await.expect("run");
        let (robot, _) = from_lobby().await.expect("joined");
        let state = until(&client, state, |s| s.stage == RunStage::Running).await;
        // Its button, then the run ends: the documented loop drops the
        // robot and waits again, never clearing.
        robot.estop("button");
        client.stop(&state.run).await.expect("stop");
        robot
            .participant()
            .wait_for("closing", WAIT, |s| {
                s.scene.as_ref().is_some_and(|s| s.closing.is_some())
            })
            .await
            .expect("closing");
        robot.detach().await.expect("detach");
        let back = tokio::spawn({
            let lobby = lobby.clone();
            async move { sdk::robot_from_lobby_at(&lobby, "arm", ARM, Some(WAIT * 3)).await }
        });
        until(&client, state, |s| s.stage == RunStage::Ended).await;

        let state = client.run_scene(&alone, false).await.expect("free");
        let (robot, _) = back.await.expect("task").expect("joined again");
        assert!(robot.participant().estop_latched(), "the same latch");
        let state = until(&client, state, |s| s.stage == RunStage::Failed).await;
        let error = state.error.expect("why");
        assert_eq!(error.status, CtlStatus::EstopLatched, "{}", error.message);
        let mode = robot.participant().fetch_authority().await.map(|s| s.mode);
        assert!(matches!(mode, Ok(Mode::Estop) | Err(_)), "{mode:?}");
        drop(robot);
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_run_fails_at_its_stage_with_its_code() {
        let dir = tempfile::tempdir().expect("dir");
        let (server, lobby) = home(dir.path()).await;
        let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
        let refused = client
            .run_scene("[scene]\nformat = 2\n", false)
            .await
            .expect_err("a bad file");
        assert!(
            matches!(&refused, SessionError::Rejected(e) if e.status == CtlStatus::InvalidRequest),
            "{refused}"
        );
        // No robot waits: it fails at assignment once `wait` passes.
        let state = client.run_scene(&scene("1s"), false).await.expect("run");
        let state = until(&client, state, |s| s.stage == RunStage::Failed).await;
        let error = state.error.expect("why");
        assert_eq!(error.status, CtlStatus::NotPresent);
        assert!(
            error.message.contains("arm is not in the lobby"),
            "{}",
            error.message
        );
        // A robot that declared, and a controller that never bound.
        let robot = tokio::spawn(robot(lobby.clone()));
        let state = client.run_scene(&scene("3s"), false).await.expect("run");
        let state = until(&client, state, |s| s.stage == RunStage::Failed).await;
        let error = state.error.expect("why");
        assert_eq!(error.status, CtlStatus::NotReady);
        assert!(
            error.message.contains("pilot has not bound"),
            "{}",
            error.message
        );
        robot.await.expect("robot");
        server.shutdown().await;
    }

    const ARM_CHANGED: &str = r#"{"arm": {"kind": "joints", "count": 3, "fields": ["pos"], "hz": 100,
        "command": "position", "hold_max_ms": 100}}"#;

    struct Hold;

    impl Predictor for Hold {
        type Error = String;

        fn predict(&mut self, _: Observation, _: i64) -> Result<Vec<(String, Value)>, String> {
            let hold = Tensor::from_f64(DType::F32, vec![2, 2], &[0.0; 4]);
            Ok(vec![("arm".into(), Value::Tensor(hold))])
        }
    }

    /// Robot `arm` from the lobby, declaring `map` and publishing its
    /// joints; a refused join is reported in its next wait.
    async fn joined(lobby: &SessionAddress, me: &mut Lobby, map: &str) -> Option<Arc<Robot>> {
        let (join, address) = wait_for_join(lobby, me, "arm", Some(WAIT * 3))
            .await
            .expect("a join");
        let robot = match sdk::connect_robot(&address, "arm", map, WAIT).await {
            Ok(robot) => Arc::new(robot),
            Err(EdgeError::Session(SessionError::Rejected(err))) => {
                me.failed(&join.run, err);
                return None;
            }
            Err(err) => panic!("{err}"),
        };
        let weak = Arc::downgrade(&robot);
        tokio::spawn(async move {
            while let Some(robot) = weak.upgrade() {
                let now = robot.participant().clock().now_ns();
                let _ = robot.publish("arm", now, &[&[0_u8; 8]]);
                drop(robot);
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
        Some(robot)
    }

    /// Stops `robot`'s heartbeats and drops it: the host loses it.
    fn lose(robot: Arc<Robot>) {
        robot.participant().set_heartbeats_enabled(false);
        drop(robot);
    }

    async fn holding(pilot: &Controller) {
        let deadline = tokio::time::Instant::now() + WAIT;
        while !pilot.holds() {
            assert!(tokio::time::Instant::now() < deadline, "no lease");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    fn rejoin_scene(rejoin: &str) -> String {
        scene("20s").replace(
            "wait = \"20s\"\n",
            &format!("wait = \"20s\"\nrejoin = \"{rejoin}\"\n"),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_lost_robot_rejoins_with_its_map_and_the_run_rearms_only_when_asked() {
        let dir = tempfile::tempdir().expect("dir");
        let (server, lobby) = home(dir.path()).await;
        let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
        let mut me = Lobby::new(ParticipantId::new("arm").expect("id"));
        let state = client
            .run_scene(&rejoin_scene("20s"), false)
            .await
            .expect("run");
        let robot = joined(&lobby, &mut me, ARM).await.expect("declared");
        let state = until(&client, state, |s| s.stage == RunStage::Binding).await;
        let address = run_address(&lobby, &state)
            .expect("credential")
            .expect("hosted");
        let pilot = Arc::new(
            sdk::bind_policy(
                &address,
                "pilot",
                PILOT,
                &RunOptions::named("pilot"),
                PriorityClass::Model,
                WAIT,
            )
            .await
            .expect("bound"),
        );
        // Lost while ARMED, before its policy leased: the session goes SAFE
        // all the same, and is never ACTIVE again until the explicit arm.
        let state = until(&client, state, |s| s.stage == RunStage::Running).await;
        lose(robot);
        let state = client
            .wait_until(&state.run, RunStage::Rejoining, WAIT)
            .await
            .expect("rejoining");
        assert_eq!(stage(&state, "arm"), MemberStage::Lost);
        let active = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut modes = pilot.participant().watch();
        let watching = tokio::spawn({
            let active = active.clone();
            async move {
                while modes.changed().await.is_ok() {
                    if modes
                        .borrow()
                        .as_ref()
                        .is_some_and(|s| s.mode == Mode::Active)
                    {
                        active.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            }
        });
        let running = std::thread::spawn({
            let pilot = pilot.clone();
            move || pilot.run(&mut Hold, None)
        });
        // Attached under its name but not declared: still rejoining.
        let (_, back) = wait_for_join(&lobby, &mut me, "arm", Some(WAIT * 3))
            .await
            .expect("its join again");
        let back = back
            .join("arm", vec![Role::Observer], PriorityClass::Scripted, WAIT)
            .await
            .expect("attached");
        tokio::time::sleep(Duration::from_millis(300)).await;
        let status = client.status(&state.run, 0, Some(WAIT)).await;
        assert_eq!(status.expect("status").stage, RunStage::Rejoining);
        back.set_heartbeats_enabled(false);
        drop(back);
        assert!(
            !running.is_finished(),
            "ROBOT_LOST in a scene is not the end"
        );
        // Re-flashed with another map: refused, and the run waits on.
        assert!(joined(&lobby, &mut me, ARM_CHANGED).await.is_none());
        let robot = joined(&lobby, &mut me, ARM).await.expect("the same map");
        let state = until(&client, state, |s| s.stage == RunStage::Arming).await;
        let refusal = state
            .members
            .iter()
            .find(|m| m.name.as_str() == "arm")
            .and_then(|m| m.error.clone());
        assert_eq!(refusal.map(|e| e.status), Some(CtlStatus::Incompatible));
        // `arm = "auto"`, and still it waits for an explicit arm.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let state = client
            .status(&state.run, 0, Some(WAIT))
            .await
            .expect("status");
        assert_eq!(state.stage, RunStage::Arming);
        assert!(
            !active.load(std::sync::atomic::Ordering::SeqCst),
            "ACTIVE before the explicit arm"
        );
        watching.abort();
        let state = client.arm(&state.run).await.expect("arm");
        assert_eq!(state.stage, RunStage::Running);
        holding(&pilot).await;
        // Reset and back at once, likely before its loss was seen: a new
        // generation whose rejoin window the run may never see. SAFE all the
        // same, and armed again on request.
        lose(robot);
        let restarted = joined(&lobby, &mut me, ARM).await.expect("restarted");
        let state = until(&client, state, |s| s.stage == RunStage::Arming).await;
        let state = client
            .arm(&state.run)
            .await
            .expect("armed after the restart");
        assert_eq!(state.stage, RunStage::Running);
        holding(&pilot).await;
        let robot = restarted;

        client.stop(&state.run).await.expect("stop");
        // This process stopped the run: the policy's run returns.
        running.join().expect("thread").expect("returned");
        let state = until(&client, state, |s| s.stage == RunStage::Ended).await;
        assert_eq!(state.error, None);
        drop(robot);
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_lost_robot_that_misses_the_rejoin_window_fails_the_run() {
        let dir = tempfile::tempdir().expect("dir");
        let (server, lobby) = home(dir.path()).await;
        let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
        let mut me = Lobby::new(ParticipantId::new("arm").expect("id"));
        let state = client
            .run_scene(&rejoin_scene("1s"), false)
            .await
            .expect("run");
        let robot = joined(&lobby, &mut me, ARM).await.expect("declared");
        let state = until(&client, state, |s| s.stage == RunStage::Binding).await;
        let address = run_address(&lobby, &state)
            .expect("credential")
            .expect("hosted");
        let pilot = Arc::new(
            sdk::bind_policy(
                &address,
                "pilot",
                PILOT,
                &RunOptions::named("pilot"),
                PriorityClass::Model,
                WAIT,
            )
            .await
            .expect("bound"),
        );
        let running = std::thread::spawn({
            let pilot = pilot.clone();
            move || pilot.run(&mut Hold, None)
        });
        let state = until(&client, state, |s| s.stage == RunStage::Running).await;
        lose(robot);
        let state = until(&client, state, |s| s.stage == RunStage::Failed).await;
        let error = state.error.expect("why");
        assert!(
            error.message.contains("arm did not rejoin within 1 s"),
            "{}",
            error.message
        );
        match running.join().expect("thread") {
            Err(RunError::Lost(reason)) => assert_eq!(reason.code, ReasonCode::SceneFailed),
            other => panic!("{other:?}"),
        }
        server.shutdown().await;
    }

    /// A `devices.toml` entry for device `id` with `key`.
    fn device(id: &str, key: &StaticKeypair, kind: &str, max: &str) -> String {
        format!(
            "[devices.{id}]\npublic_key = \"{}\"\nkind = \"{kind}\"\nname = \"{id}\"\n\
             max_priority = \"{max}\"\n\n",
            base32(key.public())
        )
    }

    /// `scene("20s")` hosted on `host`.
    fn hosted_on(host: &str) -> String {
        scene("20s").replace("wait = ", &format!("host = \"{host}\"\nwait = "))
    }

    /// rlmeshd on another machine paired with the home daemon in `home` as
    /// device `device` (`key`): its server on its own loopback port, its
    /// host file in `hosts`, and its satellite's tasks.
    async fn satellite(
        home: &std::path::Path,
        lobby: &SessionAddress,
        device: &str,
        key: &StaticKeypair,
        hosts: &std::path::Path,
    ) -> (
        Arc<SessionServer>,
        SocketAddr,
        Vec<tokio::task::JoinHandle<()>>,
    ) {
        let identity = crate::daemon::identity(home).expect("identity");
        let paired = Paired {
            info: PairInfo {
                device: device.into(),
                host_key: *identity.static_public(),
                host_id: identity.host_id(),
                addresses: Vec::new(),
            },
            device: key.clone(),
        };
        HostFile::new(&paired, &lobby.base.to_string())
            .write(hosts)
            .expect("host file");
        let (transport, base) =
            address::open_listeners(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)), None)
                .await
                .expect("listeners");
        let mut config = HostConfig::new(
            TenantId::new(TENANT).expect("tenant"),
            session_id(DEFAULT_SESSION),
        );
        config.health_log = None;
        let own = config.identity.host_id();
        let server = Arc::new(
            SessionServer::start(transport, config)
                .await
                .expect("server"),
        );
        let tasks = crate::satellite::Satellite::start(
            server.clone(),
            Some(hosts.into()),
            &own,
            vec![base.to_string()],
        );
        (server, base, tasks)
    }

    /// `client` runs `text`, retried while its satellite isn't in the lobby
    /// yet.
    async fn run_when_attached(client: &LobbyClient, text: &str) -> RunState {
        let deadline = tokio::time::Instant::now() + WAIT;
        loop {
            match client.run_scene(text, false).await {
                Ok(state) => return state,
                Err(SessionError::Rejected(err))
                    if err.status == CtlStatus::NotPresent
                        && tokio::time::Instant::now() < deadline =>
                {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(err) => panic!("{err}"),
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_scene_hosted_on_a_satellite_runs_there_until_a_members_device_changes_at_home() {
        let dir = tempfile::tempdir().expect("dir");
        let hosts = tempfile::tempdir().expect("hosts");
        let gpu = StaticKeypair::from_private(&mut [11; 32]);
        let arm_pc = StaticKeypair::from_private(&mut [12; 32]);
        let arm_pc_key = arm_pc.clone();
        std::fs::write(
            dir.path().join(crate::devices::DEVICES),
            device("gpu-desktop", &gpu, "machine", "operator")
                + &device("arm-pc", &arm_pc, "machine", "model")
                + "[robots.arm]\ndevice = \"arm-pc\"\n",
        )
        .expect("devices.toml");
        let (server, lobby, home) = home_gated(dir.path(), false).await;
        let (remote, remote_base, tasks) =
            satellite(dir.path(), &lobby, "gpu-desktop", &gpu, hosts.path()).await;
        let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");

        // The arm, a robot process on arm-pc: its join takes it to the
        // satellite, where it attaches with its own device key.
        let home_key = *lobby.credential().expect("credential").host_static();
        let robot_lobby = SessionAddress {
            key: None,
            credential: Some(Credential::new(arm_pc, home_key)),
            ..lobby.clone()
        };
        let robot = tokio::spawn(async move {
            let mut me = Lobby::new(ParticipantId::new("arm").expect("id"));
            let (join, address) = wait_for_join(&robot_lobby, &mut me, "arm", Some(WAIT * 3))
                .await
                .expect("a join");
            assert_eq!(
                (address.base, join.addresses),
                (remote_base, vec![remote_base.to_string()])
            );
            let robot = sdk::connect_robot(&address, "arm", ARM, WAIT)
                .await
                .expect("joined on the satellite");
            let arm = ParticipantId::new("arm").expect("id");
            let attached = robot
                .participant()
                .fetch_authority()
                .await
                .expect("state")
                .participant(&arm)
                .map(|p| p.device.clone());
            assert_eq!(attached.as_deref(), Some("arm-pc"));
            robot
                .participant()
                .wait_for("closing", WAIT * 3, |s| {
                    s.scene.as_ref().is_some_and(|s| s.closing.is_some())
                })
                .await
                .expect("closing");
            robot.detach().await.expect("detach");
        });

        let state = run_when_attached(&client, &hosted_on("gpu-desktop")).await;
        let state = until(&client, state, |s| s.stage == RunStage::Binding).await;
        let session = state.session.expect("hosted");
        assert_ne!(state.host_key, Some(home_key), "the satellite's own keys");
        assert!(remote.sessions().await.contains(&session));
        assert!(!server.sessions().await.contains(&session));
        let address = run_address(&lobby, &state)
            .expect("credential")
            .expect("hosted");
        let pilot = sdk::bind_policy(
            &address,
            "pilot",
            PILOT,
            &RunOptions::named("pilot"),
            PriorityClass::Model,
            WAIT,
        )
        .await
        .expect("bound on the satellite");
        let state = until(&client, state, |s| s.stage == RunStage::Running).await;

        // arm-pc narrowed at home: the satellite holds its old record, so
        // the run fails, and its StopOrder stops the satellite's host.
        std::fs::write(
            dir.path().join(crate::devices::DEVICES),
            device("gpu-desktop", &gpu, "machine", "operator")
                + &device("arm-pc", &arm_pc_key, "machine", "scripted")
                + "[robots.arm]\ndevice = \"arm-pc\"\n",
        )
        .expect("devices.toml");
        home.devices_changed(&Table::read(dir.path()).expect("table"));
        let state = until(&client, state, |s| s.stage == RunStage::Failed).await;
        let error = state.error.expect("why");
        assert_eq!(error.status, CtlStatus::NotAuthorized, "{}", error.message);
        drop(pilot);
        robot.await.expect("robot");
        let deadline = tokio::time::Instant::now() + WAIT;
        while remote.sessions().await.contains(&session) {
            assert!(tokio::time::Instant::now() < deadline, "still hosted");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        for task in tasks {
            task.abort();
        }
        remote.shutdown().await;
        server.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn only_a_paired_machine_waits_as_a_satellite_and_hosts_a_run() {
        let dir = tempfile::tempdir().expect("dir");
        let gpu = StaticKeypair::from_private(&mut [11; 32]);
        let eye = StaticKeypair::from_private(&mut [13; 32]);
        let guest = StaticKeypair::from_private(&mut [14; 32]);
        std::fs::write(
            dir.path().join(crate::devices::DEVICES),
            device("gpu-desktop", &gpu, "machine", "operator")
                + &device("eye-l", &eye, "board", "operator")
                + &device("guest", &guest, "machine", "model")
                + "[robots.arm]\ndevice = \"local\"\n",
        )
        .expect("devices.toml");
        let (server, lobby, _) = home_gated(dir.path(), false).await;
        let home_key = *lobby.credential().expect("credential").host_static();
        let as_device = |key: &StaticKeypair| SessionAddress {
            key: None,
            credential: Some(Credential::new(key.clone(), home_key)),
            ..lobby.clone()
        };
        // A board, and a machine under another's name: refused at the wait.
        for (key, name) in [(&eye, "eye-l"), (&gpu, "laptop-josh")] {
            let refused = SatelliteClient::connect(&as_device(key), name, WAIT)
                .await
                .expect("attached as an observer")
                .next_order()
                .await
                .expect_err("not a satellite");
            assert!(
                matches!(&refused, SessionError::Rejected(e) if e.status == CtlStatus::NotAuthorized),
                "{name}: {refused}"
            );
        }
        // A robot waits only from a device that may embody it: a pinned
        // name from its pin, an unpinned one from a device that may pin it
        // (not a guest without `arm`, nor a board under another name).
        for (key, robot) in [(&eye, "arm"), (&guest, "skull"), (&eye, "skull")] {
            let mut me = Lobby::new(ParticipantId::new(robot).expect("id"));
            let refused = wait_for_join(&as_device(key), &mut me, robot, Some(WAIT))
                .await
                .expect_err("may not wait as it");
            assert!(
                matches!(&refused, SessionError::Rejected(e) if e.status == CtlStatus::NotAuthorized),
                "{robot}: {refused}"
            );
        }
        // A satellite whose home's authenticated echoes stop, its liveliness
        // still up (an unsigned token anyone on the link can keep), counts
        // the home gone.
        let silenced = SatelliteClient::connect(&as_device(&gpu), "gpu-desktop", WAIT)
            .await
            .expect("a satellite");
        silenced.participant().set_heartbeats_enabled(false);
        let gone = tokio::time::timeout(
            rlmesh_session::lobby::HOME_SILENT + Duration::from_secs(2),
            silenced.next_order(),
        )
        .await
        .expect("bounded");
        assert!(matches!(gone, Err(SessionError::Closed)), "{gone:?}");
        assert!(*silenced.participant().host_present().borrow());
        // Hosted on the board: refused at check; on a machine not in the
        // lobby: not present.
        let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
        for (host, status) in [
            ("eye-l", CtlStatus::NotAuthorized),
            ("gpu-desktop", CtlStatus::NotPresent),
        ] {
            let refused = client
                .run_scene(&hosted_on(host), false)
                .await
                .expect_err("refused");
            assert!(
                matches!(&refused, SessionError::Rejected(e) if e.status == status),
                "{host}: {refused}"
            );
        }
        server.shutdown().await;
    }
}
