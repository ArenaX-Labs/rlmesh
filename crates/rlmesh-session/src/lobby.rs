//! The lobby (scene design §3, §4): the session `_lobby` on a robot's home
//! daemon, where idle robots wait for a run and clients run scenes.
//!
//! - [`LobbyService`]: what the `_lobby` host hands the lobby verbs to
//!   (rlmeshd's runs). A verb it can't answer at once (a robot's wait, a
//!   status poll) stays pending in the host, under its request id, and never
//!   blocks the sender's other verbs.
//! - [`LobbyClient`]: a client in the lobby (`cli.<8 hex>`, an observer):
//!   run, follow, stop and arm scenes, and list the robots waiting.
//! - [`wait_for_join`]: a robot process in the lobby until a run assigns it.
//! - [`SatelliteClient`]: a satellite daemon taking its home's orders.
//!
//! The robot's own rules (what it reports, which order it takes, when it
//! leaves a run) are sans-IO in [`rlmesh_session_core::lobby`], shared
//! with boards.

use std::net::ToSocketAddrs;
use std::time::Duration;

use tokio::sync::oneshot;

use crate::address::{Address, SessionAddress, remember_credential, session_id};
use crate::discovery;
use crate::error::SessionError;
use crate::participant::Participant;
use crate::wire::{
    Assignment, Command, CtlError, CtlReply, CtlStatus, HostStarted, Join, ParticipantId,
    ParticipantInfo, PriorityClass, Role, RunStage, RunState, SessionId,
};
pub use rlmesh_session_core::lobby::{LOBBY_SESSION, Leave, Lobby, WAIT_RETRY, leave};

/// What a lobby verb is answered with.
#[derive(Clone, Debug, PartialEq)]
pub enum LobbyReply {
    /// A wait's order.
    Assignment(Assignment),
    /// A run's state.
    Run(RunState),
}

/// A lobby verb's outcome.
pub type LobbyResult = Result<LobbyReply, CtlError>;

/// When a lobby verb is answered.
#[derive(Debug)]
pub enum LobbyAnswer {
    /// Now.
    Now(LobbyResult),
    /// Once the receiver resolves (a long poll). The host drops it when the
    /// sender's generation ends, which closes the sending half.
    Later(oneshot::Receiver<LobbyResult>),
}

/// The home daemon's lobby, behind its `_lobby` session host
/// ([`crate::HostConfig::lobby`]).
pub trait LobbyService: Send + Sync + 'static {
    /// Answers lobby verb `command` from `from` on `device`. The host has
    /// checked that the sender is attached, and that `RunScene`,
    /// `StopScene` and `ArmScene` come from a device with the `arm` verb.
    fn handle(&self, from: &ParticipantId, device: &str, command: Command) -> LobbyAnswer;
}

/// Whether `command` is a lobby verb.
pub fn is_lobby_verb(command: &Command) -> bool {
    matches!(
        command,
        Command::LobbyWait(_)
            | Command::RunScene { .. }
            | Command::SceneStatus { .. }
            | Command::StopScene { .. }
            | Command::HostStarted(_)
            | Command::ArmScene { .. }
    )
}

/// The lobby of the home daemon `home` names (`rig-pi.local`,
/// `10.0.0.5:7447`; `None`: the one on this machine's well-known port).
/// [`find_home`] also takes a host id.
///
/// # Errors
/// [`SessionError::Config`] for an address that doesn't parse.
pub fn home_lobby(home: Option<&str>) -> Result<SessionAddress, SessionError> {
    let base = Address::session(home.unwrap_or(""))?.joining()?;
    Ok(SessionAddress {
        name: LOBBY_SESSION.to_string(),
        session: session_id(LOBBY_SESSION),
        ..base
    })
}

/// The lobby of the home daemon `home` names: a host id (`rlm-3f9a2c`)
/// through this machine's host file for it (its stored addresses, then
/// mDNS, [`discovery::locate`]), anything else as [`home_lobby`] reads it.
///
/// # Errors
/// A host id this machine isn't paired with, what finding it failed with,
/// or an address that doesn't parse.
pub async fn find_home(home: Option<&str>) -> Result<SessionAddress, SessionError> {
    let Some(id) = home.filter(|h| discovery::is_host_id(h)) else {
        return home_lobby(home);
    };
    let dir = crate::pairing::hosts_dir()
        .ok_or_else(|| SessionError::Config("no home directory for ~/.config/rlmesh".into()))?;
    let path = discovery::host_file(&dir, id);
    if !path.exists() {
        return Err(SessionError::Config(format!(
            "this machine is not paired with {id} (no {}): rlmesh pair HOST",
            path.display()
        )));
    }
    let credential = crate::pairing::HostFile::read(&path)?.credential();
    // The host proves its key at an address before it is used or kept.
    let (_, lobby) = discovery::locate(&path, |base| {
        let credential = credential.clone();
        async move {
            let lobby = SessionAddress {
                credential: Some(credential),
                ..SessionAddress::new(base, LOBBY_SESSION)?
            };
            LobbyClient::connect(&lobby, discovery::ATTACH)
                .await?
                .close()
                .await?;
            Ok(lobby)
        }
    })
    .await?;
    Ok(lobby)
}

/// Where a run's session is, for a member of the home daemon whose lobby
/// is `lobby` (scene design §3.3, §4.4): at the lobby's own host when its
/// static key is `host_key` (the home daemon hosts the run), else at the
/// first of `addresses` that resolves (a satellite's), attached with the
/// same device key against `host_key`, which this process keeps for the
/// session ([`remember_credential`]).
///
/// # Errors
/// No credential for the lobby, or no address of a satellite's run that
/// resolves.
pub fn hosted_at(
    lobby: &SessionAddress,
    session: SessionId,
    addresses: &[String],
    host_key: [u8; 32],
) -> Result<SessionAddress, SessionError> {
    let credential = lobby.credential()?;
    if *credential.host_static() == host_key {
        return Ok(SessionAddress {
            name: session.to_string(),
            session,
            ..lobby.clone()
        });
    }
    let base = addresses
        .iter()
        .find_map(|a| a.to_socket_addrs().ok()?.next())
        .ok_or_else(|| {
            SessionError::Config(format!("the run's host is at none of {addresses:?}"))
        })?;
    let address = SessionAddress {
        tenant: lobby.tenant.clone(),
        credential: Some(credential.with_host(host_key)),
        ..SessionAddress::new(base, &session.to_string())?
    };
    remember_credential(&address, credential.with_host(host_key));
    Ok(address)
}

/// Run `run`'s session, for a member of the home daemon whose lobby is
/// `lobby`: [`hosted_at`] once its host is up (`None` before).
///
/// # Errors
/// What [`hosted_at`] fails with.
pub fn run_address(
    lobby: &SessionAddress,
    run: &RunState,
) -> Result<Option<SessionAddress>, SessionError> {
    match (run.session, run.host_key) {
        (Some(session), Some(host_key)) => {
            hosted_at(lobby, session, &run.addresses, host_key).map(Some)
        }
        _ => Ok(None),
    }
}

/// `command`'s reply, however long the host keeps it pending: past the
/// participant's ctl timeout the request goes on being re-sent under its id
/// (§8.6), and the retry joins the pending verb. A refusal is an error.
async fn long_poll(
    participant: &Participant,
    command: Command,
    timeout: Option<Duration>,
) -> Result<CtlReply, SessionError> {
    let poll = async {
        match participant.command(command).await {
            Err(SessionError::Pending { pending, .. }) => pending.reply().await,
            other => other,
        }
    };
    let reply = match timeout {
        Some(timeout) => tokio::time::timeout(timeout, poll)
            .await
            .map_err(|_| SessionError::Timeout(timeout, "the lobby".into()))??,
        None => poll.await?,
    };
    reply.result.clone()?;
    Ok(reply)
}

fn run_of(reply: CtlReply) -> Result<RunState, SessionError> {
    reply.run.ok_or_else(|| {
        SessionError::Rejected(CtlError::new(
            CtlStatus::InvalidRequest,
            "the lobby answered without the run's state",
        ))
    })
}

/// The runs this process stopped: their policies' runs return rather than
/// raise when the run closes (scene design §4.2).
fn stopped() -> &'static std::sync::Mutex<std::collections::HashSet<String>> {
    static STOPPED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    STOPPED.get_or_init(Default::default)
}

/// Whether this process stopped run `run`.
pub fn stopped_here(run: &str) -> bool {
    stopped()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .contains(run)
}

/// A client in a home daemon's lobby.
pub struct LobbyClient {
    participant: Participant,
    lobby: SessionAddress,
}

impl LobbyClient {
    /// Attaches to `lobby` as `cli.<8 random hex>`, an observer, waiting up
    /// to `timeout` for the daemon.
    ///
    /// # Errors
    /// What attaching fails with.
    pub async fn connect(lobby: &SessionAddress, timeout: Duration) -> Result<Self, SessionError> {
        let id = format!("cli.{:08x}", rand::random::<u32>());
        let participant = lobby
            .join(&id, vec![Role::Observer], PriorityClass::Scripted, timeout)
            .await?;
        Ok(Self {
            participant,
            lobby: lobby.clone(),
        })
    }

    /// The lobby it is in.
    pub fn lobby(&self) -> &SessionAddress {
        &self.lobby
    }

    /// Runs the scene `scene_toml` (`manual_arm`: wait for an explicit arm
    /// even with `arm = "auto"`). A scene refused at its check or claim is
    /// [`SessionError::Rejected`] with the stage's code.
    ///
    /// # Errors
    /// The refusal, or what the call failed with.
    pub async fn run_scene(
        &self,
        scene_toml: &str,
        manual_arm: bool,
    ) -> Result<RunState, SessionError> {
        let command = Command::RunScene {
            scene_toml: scene_toml.to_string(),
            manual_arm,
        };
        run_of(long_poll(&self.participant, command, None).await?)
    }

    /// Run `run`'s state once its version passes `after` (`0`: now), or
    /// [`SessionError::Timeout`] after `timeout`.
    ///
    /// # Errors
    /// An unknown run, or what the call failed with.
    pub async fn status(
        &self,
        run: &str,
        after: u64,
        timeout: Option<Duration>,
    ) -> Result<RunState, SessionError> {
        let command = Command::SceneStatus {
            run: run.to_string(),
            after,
        };
        run_of(long_poll(&self.participant, command, timeout).await?)
    }

    /// Follows run `run` until it is at `stage` (scene design §4.3:
    /// `RUNNING` again after a rejoin), or [`SessionError::Timeout`] after
    /// `timeout`. A run that ended or failed first is
    /// [`SessionError::Rejected`] with its error.
    ///
    /// # Errors
    /// The run ending first, the timeout, or what a call failed with.
    pub async fn wait_until(
        &self,
        run: &str,
        stage: RunStage,
        timeout: Duration,
    ) -> Result<RunState, SessionError> {
        let deadline = tokio::time::Instant::now() + timeout;
        let mut state = self.status(run, 0, Some(timeout)).await?;
        loop {
            if state.stage == stage {
                return Ok(state);
            }
            if matches!(state.stage, RunStage::Ended | RunStage::Failed) {
                return Err(SessionError::Rejected(state.error.unwrap_or_else(|| {
                    CtlError::new(CtlStatus::InvalidTransition, format!("run {run} ended"))
                })));
            }
            let left = deadline.saturating_duration_since(tokio::time::Instant::now());
            state = self.status(run, state.version, Some(left)).await?;
        }
    }

    /// Stops run `run`; its state as the stop started.
    ///
    /// # Errors
    /// An unknown run, or what the call failed with.
    pub async fn stop(&self, run: &str) -> Result<RunState, SessionError> {
        stopped()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(run.to_string());
        let command = Command::StopScene {
            run: run.to_string(),
        };
        run_of(long_poll(&self.participant, command, None).await?)
    }

    /// Arms run `run` (`arm = "manual"`, or after a rejoin); its state once
    /// armed.
    ///
    /// # Errors
    /// A run with nothing to arm, the arm's own refusal, or what the call
    /// failed with.
    pub async fn arm(&self, run: &str) -> Result<RunState, SessionError> {
        let command = Command::ArmScene {
            run: run.to_string(),
        };
        run_of(long_poll(&self.participant, command, None).await?)
    }

    /// The robots waiting in the lobby, by its latest state.
    pub fn waiting(&self) -> Vec<ParticipantInfo> {
        self.participant
            .authority()
            .map(|state| {
                state
                    .roster
                    .into_iter()
                    .filter(|p| p.present && p.has_role(Role::Embodiment))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Re-reads the lobby's state, then [`Self::waiting`].
    ///
    /// # Errors
    /// What the query failed with.
    pub async fn fetch_waiting(&self) -> Result<Vec<ParticipantInfo>, SessionError> {
        self.participant.fetch_authority().await?;
        Ok(self.waiting())
    }

    /// Leaves the lobby.
    ///
    /// # Errors
    /// What the detach failed with.
    pub async fn close(self) -> Result<(), SessionError> {
        self.participant.detach().await
    }
}

/// A satellite daemon's id in its home's lobby: `rlmeshd.` and its device
/// (scene design §3.2).
pub const SATELLITE_PREFIX: &str = "rlmeshd.";

/// How long a satellite goes without its home authenticating itself (a
/// heartbeat echo) before it takes the home as lost, whatever the home's
/// unsigned liveliness token says (scene design §4.4).
pub const HOME_SILENT: Duration = Duration::from_secs(3);

/// Resolves once `participant`'s host has gone [`HOME_SILENT`] without an
/// authenticated echo, or its liveliness went away.
async fn home_gone(participant: &Participant) {
    let mut present = participant.host_present();
    let silent = async {
        loop {
            let unheard = participant.host_unheard();
            if unheard >= HOME_SILENT {
                return;
            }
            tokio::time::sleep((HOME_SILENT - unheard).min(Duration::from_millis(250))).await;
        }
    };
    tokio::select! {
        _ = present.wait_for(|present| !present) => {}
        () = silent => {}
    }
}

/// A satellite daemon in its home daemon's lobby (scene design §4.4):
/// `rlmeshd.<device>`, an observer that takes its home's orders one at a
/// time and answers a host order.
pub struct SatelliteClient {
    participant: Participant,
}

impl SatelliteClient {
    /// Attaches to `lobby` as the satellite of device `device`, waiting up
    /// to `timeout` for the daemon.
    ///
    /// # Errors
    /// What attaching fails with.
    pub async fn connect(
        lobby: &SessionAddress,
        device: &str,
        timeout: Duration,
    ) -> Result<Self, SessionError> {
        let participant = lobby
            .join(
                &format!("{SATELLITE_PREFIX}{device}"),
                vec![Role::Observer],
                PriorityClass::Scripted,
                timeout,
            )
            .await?;
        Ok(Self { participant })
    }

    /// The home daemon's next order, however long it takes;
    /// [`SessionError::Closed`] once the home daemon is gone: its
    /// liveliness away, or no authenticated echo for [`HOME_SILENT`].
    ///
    /// # Errors
    /// A refusal (not a satellite of this home), the home lost, or what
    /// the call failed with.
    pub async fn next_order(&self) -> Result<Assignment, SessionError> {
        let wait = long_poll(
            &self.participant,
            Command::LobbyWait(crate::wire::LobbyWait::default()),
            None,
        );
        let reply = tokio::select! {
            reply = wait => reply?,
            () = home_gone(&self.participant) => return Err(SessionError::Closed),
        };
        reply.assignment.ok_or_else(|| {
            SessionError::Rejected(CtlError::new(
                CtlStatus::InvalidRequest,
                "the lobby answered a wait without an order",
            ))
        })
    }

    /// Tells the home daemon how a host order went.
    ///
    /// # Errors
    /// What the call failed with.
    pub async fn host_started(&self, started: HostStarted) -> Result<(), SessionError> {
        tokio::select! {
            reply = long_poll(&self.participant, Command::HostStarted(started), None) => {
                reply.map(drop)
            }
            () = home_gone(&self.participant) => Err(SessionError::Closed),
        }
    }

    /// The home lobby's instance, as its handshake authenticated it: a
    /// restarted home is another.
    pub fn home_instance(&self) -> u64 {
        self.participant.host_instance().unwrap_or_default()
    }

    /// Its lobby participant (tests stop its heartbeats, so the home's
    /// echoes stop with its liveliness still up).
    #[doc(hidden)]
    pub fn participant(&self) -> &Participant {
        &self.participant
    }

    /// Leaves the lobby.
    ///
    /// # Errors
    /// What the detach failed with.
    pub async fn close(self) -> Result<(), SessionError> {
        self.participant.detach().await
    }
}

/// Waits in `lobby` as robot `lobby`'s robot until a run assigns it (or
/// `timeout`), then leaves the lobby: the [`Join`] and the run's session.
/// The wait reports the robot's last failed join ([`Lobby::failed`]).
///
/// # Errors
/// What attaching or waiting failed with; an order that isn't this robot's
/// join as [`SessionError::Rejected`].
pub async fn wait_for_join(
    lobby_address: &SessionAddress,
    lobby: &mut Lobby,
    robot: &str,
    timeout: Option<Duration>,
) -> Result<(Join, SessionAddress), SessionError> {
    let attach = timeout.unwrap_or(Duration::from_secs(10));
    let participant = lobby_address
        .join(
            robot,
            vec![Role::Embodiment],
            PriorityClass::Scripted,
            attach,
        )
        .await?;
    let waited = long_poll(&participant, Command::LobbyWait(lobby.wait()), timeout).await;
    let _ = participant.detach().await;
    let assignment = waited?.assignment.ok_or_else(|| {
        SessionError::Rejected(CtlError::new(
            CtlStatus::InvalidRequest,
            "the lobby answered a wait without an order",
        ))
    })?;
    let join = lobby.take(assignment)?;
    let address = hosted_at(lobby_address, join.session, &join.addresses, join.host_key)?;
    Ok((join, address))
}
