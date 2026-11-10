//! The home daemon's lobby and runs: rlmeshd hosts
//! the session `_lobby`, where robots wait and clients run scenes, and
//! hosts each run's session in-process, or has a satellite host it
//! (a paired machine's rlmeshd waiting here as `rlmeshd.<device>`,
//! [`crate::satellite`]).
//!
//! A run goes through its stages, each failing with its code:
//! 1. **check** the scene file against `devices.toml` and its pins
//!    ([`crate::scene::check`]): `INVALID_REQUEST`, `NOT_AUTHORIZED`;
//! 2. **claim** its robots, all or none: `ROBOT_BUSY`, naming the run;
//! 3. **host** its session, with the scene's places and member table: here,
//!    or on the satellite `host =` names (by default the machine of the
//!    first controller that runs rlmeshd) through a `HostOrder`
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

mod driver;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::time::Duration;

use rlmesh_session::address::session_id;
use rlmesh_session::auth::address::unbase32;
use rlmesh_session::auth::{HostIdentity, LOCAL_DEVICE, Pin};
use rlmesh_session::authority::{Member as HostMember, Verb};
use rlmesh_session::lobby::{
    LOBBY_SESSION, LobbyAnswer, LobbyReply, LobbyResult, LobbyService, SATELLITE_PREFIX,
};
use rlmesh_session::participant::Participant;
use rlmesh_session::scene::{Arm, Role as SceneRole, Scene};
use rlmesh_session::server::SessionServer;
use rlmesh_session::wire::{
    Assignment, AuthoritySnapshot, Command, CtlError, CtlStatus, DeviceRecord, HostOrder,
    HostStarted, Join, LobbyWait, MemberStage, MemberState, ParticipantId, PriorityClass, RobotPin,
    RunStage, RunState, SessionId,
};
use rlmesh_session::{HostConfig, SessionError};
use tokio::sync::{mpsc, oneshot, watch};

use crate::devices::{DEVICES, Table};
use crate::lobby::driver::Outcome;

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

/// Run `run`'s host order for a satellite: the scene file, its
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
    /// device records or pins it changed fails, since
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
        // Only a device that may embody the name waits under it: its
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
                format!("{robot} is in run {other} ({scene}); rlmesh scene stop {other}"),
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
            places: scene.places(),
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

    /// Where run `scene` is hosted: `None` here, or the satellite
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

#[cfg(test)]
mod tests;
