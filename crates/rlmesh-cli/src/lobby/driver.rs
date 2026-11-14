//! A run's driver: it takes a claimed run through its stages after the claim,
//! following the run's session, and tears it down.

use std::net::ToSocketAddrs;
use std::sync::Arc;
use std::time::Duration;

use rlmesh_session::address::{SessionAddress, TENANT};
use rlmesh_session::lobby::LobbyReply;
use rlmesh_session::participant::Participant;
use rlmesh_session::scene::{Role as SceneRole, Scene};
use rlmesh_session::server::SessionServer;
use rlmesh_session::wire::{
    Assignment, AuthoritySnapshot, CtlError, CtlStatus, HostOrder, Join, MemberStage, MemberState,
    Mode, ParticipantId, PriorityClass, Reason, ReasonCode, Role, RunStage, RunState, SessionId,
    StopOrder,
};
use rlmesh_session::{HostConfig, SessionError};
use tokio::sync::{oneshot, watch};

use crate::lobby::{Driver, HOME, RunCommand, STRAGGLERS, lock, member, refused, update};

/// How a run left its stages.
pub(super) enum Outcome {
    /// Stopped by a client.
    Stopped(ParticipantId),
    /// Failed: why.
    Failed(CtlError),
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

    pub(super) async fn drive(mut self, config: HostConfig) {
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
            // Back from a loss, the run arms only when asked.
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
                // The robot may already be back in the lobby, returned,
                // between the answer and this line.
                member(&self.run.state, &robot, |m| {
                    if m.stage != MemberStage::Returned {
                        m.stage = MemberStage::Assigned;
                    }
                });
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
                        let _ = answer.send(Ok(LobbyReply::Run(self.state())));
                    }
                },
            }
        }
    }

    /// Stage 8a: waits for every lost member to come back (a robot
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
