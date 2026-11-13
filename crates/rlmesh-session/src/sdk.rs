//! What every language binding shares (Python's extension, the Node addon):
//! the embedded host, joining as a robot (directly, or from the home
//! daemon's lobby), a policy or a recorder, closing them, running scenes,
//! publishing a robot's fields, a due command's fields, the run-time
//! options, the loop timer and the names errors and counters carry. A
//! binding only converts values to and from its language.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;

use crate::address::{Address, HostingOptions, Placed, SessionAddress, place_with};
use crate::controller::{Controller, ControllerConfig, ControllerError};
use crate::edge::{Due, EdgeConfig, EdgeError, EdgeMode, Published, Rejection, Robot};
use crate::error::SessionError;
#[cfg(feature = "fleet")]
use crate::lobby::{Leave, Lobby, LobbyClient, find_home, leave, run_address, wait_for_join};
#[cfg(feature = "fleet")]
use crate::participant::Participant;
use crate::spec::apply::Tensor;
use crate::spec::bind::Dir;
use crate::spec::policy::Allow;
use crate::spec::{BindError, Policy, RunOptions};
#[cfg(feature = "fleet")]
use crate::wire::{AuthorityView, MemberStage, ParticipantId, RunStage, RunState};
use crate::wire::{CtlStatus, DType, PriorityClass, ReasonCode, Role};

/// How a binding reports a [`SessionError`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// The host refused a verb: its status (`LEASE_BUSY`).
    Refused(String),
    /// Nothing answered in time.
    Timeout,
    /// The value or the part named is wrong.
    Value,
    /// Anything else.
    Session,
}

impl ErrorKind {
    /// The kind of `err`.
    pub fn of(err: &SessionError) -> Self {
        match err {
            SessionError::Rejected(refusal) | SessionError::Unverified(refusal) => {
                Self::Refused(status_name(refusal.status))
            }
            SessionError::NoReply(_) | SessionError::Pending { .. } | SessionError::Timeout(..) => {
                Self::Timeout
            }
            SessionError::InvalidValue { .. }
            | SessionError::UnknownPart(_)
            | SessionError::WrongChannel { .. } => Self::Value,
            _ => Self::Session,
        }
    }
}

/// `CamelCase` as `snake_case` (or `SCREAMING_CASE`).
fn snake(camel: &str, upper: bool) -> String {
    let mut out = String::with_capacity(camel.len() + 4);
    for (i, c) in camel.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(if upper {
            c.to_ascii_uppercase()
        } else {
            c.to_ascii_lowercase()
        });
    }
    out
}

/// A refusal's status as bindings name it (`LEASE_BUSY`).
pub fn status_name(status: CtlStatus) -> String {
    snake(&format!("{status:?}"), true)
}

#[cfg(feature = "fleet")]
/// A run's stage as bindings name it (`rejoining`).
pub fn run_stage_name(stage: RunStage) -> String {
    snake(&format!("{stage:?}"), false)
}

#[cfg(feature = "fleet")]
/// A run member's stage as bindings name it (`declared`).
pub fn member_stage_name(stage: MemberStage) -> String {
    snake(&format!("{stage:?}"), false)
}

/// An edge rejection's reason as bindings name it (`too_old`).
pub fn rejection_name(reason: Rejection) -> String {
    snake(&format!("{reason:?}"), false)
}

/// An edge mode's name (`driving`).
pub fn mode_name(mode: EdgeMode) -> &'static str {
    match mode {
        EdgeMode::Idle => "idle",
        EdgeMode::Driving => "driving",
        EdgeMode::Hold => "hold",
        EdgeMode::Safe => "safe",
        EdgeMode::Estop => "estop",
    }
}

/// A reason code as bindings name it (`robot_lost`).
pub fn reason_name(code: ReasonCode) -> String {
    code.name().to_ascii_lowercase()
}

/// Every edge mode, in ladder order.
pub const EDGE_MODES: [EdgeMode; 5] = [
    EdgeMode::Idle,
    EdgeMode::Driving,
    EdgeMode::Hold,
    EdgeMode::Safe,
    EdgeMode::Estop,
];

/// A priority class by name.
pub fn priority(name: &str) -> Result<PriorityClass, String> {
    match name {
        "safety" => Ok(PriorityClass::Safety),
        "operator" => Ok(PriorityClass::Operator),
        "model" => Ok(PriorityClass::Model),
        "scripted" => Ok(PriorityClass::Scripted),
        other => Err(format!(
            "priority {other:?}: safety, operator, model or scripted"
        )),
    }
}

/// A refusal's problems as `(code, "input" | "action", entry)`.
pub fn bind_problems(err: &BindError) -> Vec<(&'static str, &'static str, String)> {
    match err {
        BindError::Refused(refusal) => refusal
            .problems
            .iter()
            .map(|p| {
                let dir = match p.dir {
                    Dir::In => "input",
                    Dir::Out => "action",
                };
                (p.code.name(), dir, p.entry.clone())
            })
            .collect(),
        BindError::Options(_) => Vec::new(),
    }
}

/// One array from a binding: a numeric dtype by name, a shape and packed
/// native-endian bytes that fill it.
pub fn tensor(entry: &str, dtype: &str, shape: Vec<u32>, data: Vec<u8>) -> Result<Tensor, String> {
    let dtype = DType::from_name(dtype)
        .filter(|d| *d != DType::Utf8)
        .ok_or_else(|| format!("{entry}: dtype {dtype} is not numeric"))?;
    let elements: u64 = shape.iter().map(|d| u64::from(*d)).product();
    if elements.checked_mul(dtype.size()) != u64::try_from(data.len()).ok() {
        return Err(format!(
            "{entry}: {} bytes for {} {shape:?}",
            data.len(),
            dtype.name()
        ));
    }
    Ok(Tensor { dtype, shape, data })
}

/// A policy's run-time options. Maps are applied sorted, so errors
/// come in a stable order.
pub struct BindOptions<'a> {
    /// The policy's participant id.
    pub name: &'a str,
    /// The robot to bind to; `None`: the session's view of every robot
    /// (each under its name; actions name theirs, `eye-1:gaze`).
    pub robot: Option<&'a str>,
    /// A subtree every `at` is under.
    pub under: Option<&'a str>,
    /// Explicit paths, by entry.
    pub at: Vec<(String, String)>,
    /// Entries matched by position.
    pub positional: &'a [String],
    /// Allowed conversions, by name.
    pub allow: &'a [String],
    /// Selector captures.
    pub captures: Vec<(String, String)>,
}

impl BindOptions<'_> {
    /// The options as the binder takes them.
    pub fn run_options(mut self) -> Result<RunOptions, String> {
        let mut options = RunOptions::named(self.name);
        if let Some(robot) = self.robot {
            options = options.robot(robot)?;
        }
        if let Some(under) = self.under {
            options = options.under(under)?;
        }
        self.at.sort();
        for (entry, path) in &self.at {
            options = options.at(entry, path)?;
        }
        for entry in self.positional {
            options = options.positional(entry);
        }
        for name in self.allow {
            let conversion = Allow::from_name(name)
                .ok_or_else(|| format!("allow: unknown conversion {name:?}"))?;
            options = options.allow(conversion);
        }
        self.captures.sort();
        for (capture, value) in &self.captures {
            options = options.capture(capture, value)?;
        }
        Ok(options)
    }
}

/// A session placed by the hosting rule (`address::place`): hosted in
/// this process, on a local TCP port and this process's
/// `CLOCK_MONOTONIC`, or joined when another process holds it.
pub struct EmbeddedHost {
    address: SessionAddress,
    placed: tokio::sync::Mutex<Option<Placed>>,
}

impl EmbeddedHost {
    /// Places the session `address` names (`":0"`: hosted here on an
    /// OS-assigned port); leases in a session hosted here lapse
    /// `lease_grace` after their holder's last heartbeat.
    pub async fn start(address: &Address, lease_grace: Duration) -> Result<Self, SessionError> {
        Self::start_with(address, lease_grace, &HostingOptions::default()).await
    }

    /// [`Self::start`], a session hosted here run with `options` (its board
    /// data path).
    pub async fn start_with(
        address: &Address,
        lease_grace: Duration,
        options: &HostingOptions,
    ) -> Result<Self, SessionError> {
        let placed = place_with(address, lease_grace, options).await?;
        Ok(Self {
            address: placed.address.clone(),
            placed: tokio::sync::Mutex::new(Some(placed)),
        })
    }

    /// A session another process hosts, joined at `address` (a run's,
    /// from the lobby): closing it stops nothing.
    #[cfg(feature = "fleet")]
    pub fn joined(address: SessionAddress) -> Self {
        Self {
            address,
            placed: tokio::sync::Mutex::new(None),
        }
    }

    /// What participants join.
    pub fn address(&self) -> &SessionAddress {
        &self.address
    }

    /// Its participants' link health, when this process hosts the session
    /// (empty otherwise): as of the host's last health interval.
    pub async fn health(&self) -> Vec<crate::ParticipantHealth> {
        match self.placed.lock().await.as_ref() {
            Some(placed) => placed.health().await,
            None => Vec::new(),
        }
    }

    /// Stops the host, if this process runs it (participants see it leave).
    /// Idempotent.
    pub async fn close(&self) -> Result<(), SessionError> {
        if let Some(mut placed) = self.placed.lock().await.take() {
            placed.close().await?;
        }
        Ok(())
    }
}

/// Joins as robot `name` and declares `robot_json` (the authored map).
/// Every robot `name` this process joins to one host shares its e-stop
/// latch, so a stop outlives a rejoin until it is cleared locally.
pub async fn connect_robot(
    address: &SessionAddress,
    name: &str,
    robot_json: &str,
    timeout: Duration,
) -> Result<Robot, EdgeError> {
    let latch = address
        .credential()
        .ok()
        .map(|c| process_latch(*c.host_static(), name));
    let participant = address
        .join_with(
            name,
            vec![Role::Embodiment],
            PriorityClass::Scripted,
            timeout,
            |config| config.latch = latch,
        )
        .await?;
    Robot::declare(participant, robot_json, EdgeConfig::default()).await
}

/// The robot a binding drives: read on every call, swapped by
/// [`keep_joined`] when it rejoins, emptied when it closes.
pub type RobotSlot = Arc<std::sync::Mutex<Option<Arc<Robot>>>>;

/// How long each rejoin waits for the host.
const REJOIN_ATTACH: Duration = Duration::from_secs(5);
/// How often the robot's presence is checked.
const PRESENCE_CHECK: Duration = Duration::from_millis(250);

/// Whether the host has dropped `robot` (its roster entry gone or absent,
/// or the host itself gone); `None` when another process took its name
/// over, which it never takes back.
fn dropped(robot: &Robot) -> Option<bool> {
    let participant = robot.participant();
    if participant.host_lost().is_some() {
        return Some(true);
    }
    let mine = participant.generation();
    participant.authority_view().with(|view| {
        let Some(view) = view else {
            return Some(false);
        };
        match view.participant(participant.id()) {
            None => Some(true),
            Some(me) if !me.present => Some(true),
            Some(me) if mine.is_some_and(|g| g != me.generation) => None,
            Some(_) => Some(false),
        }
    })
}

/// Keeps the robot in `slot` joined to the session at `address`: when the
/// host drops it (a Wi-Fi gap longer than its presence grace) or goes away
/// and comes back, it joins again as `name`, declares `robot_json` and is
/// swapped into `slot`, so the binding's next call uses it. Runs until the
/// slot empties (the robot closed) or another process takes the name over.
pub fn keep_joined(
    slot: RobotSlot,
    address: SessionAddress,
    name: String,
    robot_json: String,
) -> tokio::task::JoinHandle<()> {
    let current = move |slot: &RobotSlot| {
        slot.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    };
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(PRESENCE_CHECK);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            let Some(robot) = current(&slot) else {
                return;
            };
            match dropped(&robot) {
                Some(false) => continue,
                None => {
                    tracing::warn!(robot = %name, "another process took this robot's name over");
                    return;
                }
                Some(true) => {}
            }
            tracing::warn!(robot = %name, "the host dropped this robot: joining again");
            // Its old presence would refuse the new one; the binding keeps
            // the old robot (its edge holds, then goes safe) until then.
            robot.participant().withdraw();
            drop(robot);
            loop {
                if current(&slot).is_none() {
                    return;
                }
                match connect_robot(&address, &name, &robot_json, REJOIN_ATTACH).await {
                    Ok(robot) => {
                        let mut held = slot
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if held.is_none() {
                            return;
                        }
                        *held = Some(Arc::new(robot));
                        tracing::info!(robot = %name, "joined again");
                        break;
                    }
                    Err(err) => {
                        tracing::debug!(robot = %name, %err, "rejoin failed; trying again");
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        }
    })
}

#[cfg(feature = "fleet")]
/// How long a robot from the lobby waits for its run's host.
const RUN_ATTACH: Duration = Duration::from_secs(10);

#[cfg(feature = "fleet")]
/// Waits in the lobby of the home daemon `home` names (`None`: this
/// machine's) as robot `name` until a run assigns it, then joins the run and
/// declares `robot_json`: the robot, and its run's session. A refused join
/// (its pin, another map) is reported to the daemon by the next wait, which
/// goes on waiting; `timeout` (`None`: none) bounds it all.
///
/// # Errors
/// What waiting, attaching or declaring failed with, a refusal aside.
pub async fn robot_from_lobby(
    home: Option<&str>,
    name: &str,
    robot_json: &str,
    timeout: Option<Duration>,
) -> Result<(Robot, SessionAddress), EdgeError> {
    robot_from_lobby_at(&find_home(home).await?, name, robot_json, timeout).await
}

/// The e-stop latch of robot `name` in this process, for the host whose
/// key is `home`: one per robot process, so it outlives each
/// participant (a rejoin, a lobby's run) until cleared locally.
fn process_latch(home: [u8; 32], name: &str) -> Arc<crate::participant::LocalLatch> {
    type Latches =
        std::collections::HashMap<([u8; 32], String), Arc<crate::participant::LocalLatch>>;
    static LATCHES: std::sync::OnceLock<std::sync::Mutex<Latches>> = std::sync::OnceLock::new();
    LATCHES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entry((home, name.to_string()))
        .or_default()
        .clone()
}

#[cfg(feature = "fleet")]
/// [`robot_from_lobby`] at the lobby `lobby_address`. Every run's robot
/// shares this process's latch for the name: a stop not cleared locally
/// starts the next run in ESTOP.
///
/// # Errors
/// As [`robot_from_lobby`].
pub async fn robot_from_lobby_at(
    lobby_address: &SessionAddress,
    name: &str,
    robot_json: &str,
    timeout: Option<Duration>,
) -> Result<(Robot, SessionAddress), EdgeError> {
    let home = *lobby_address.credential()?.host_static();
    let latch = process_latch(home, name);
    let mut lobby = Lobby::new(ParticipantId::new(name).map_err(SessionError::from)?);
    let deadline = timeout.map(|t| tokio::time::Instant::now() + t);
    loop {
        let left = deadline.map(|d| d.saturating_duration_since(tokio::time::Instant::now()));
        let (join, address) = wait_for_join(lobby_address, &mut lobby, name, left).await?;
        let joined = address
            .join_with(
                name,
                vec![Role::Embodiment],
                PriorityClass::Scripted,
                RUN_ATTACH,
                |config| config.latch = Some(latch.clone()),
            )
            .await;
        let robot = match joined {
            Ok(participant) => Robot::declare(participant, robot_json, EdgeConfig::default()).await,
            Err(err) => Err(err.into()),
        };
        match robot {
            Ok(robot) => return Ok((robot, address)),
            Err(EdgeError::Session(SessionError::Rejected(refusal))) => {
                lobby.failed(&join.run, refusal);
            }
            Err(err) => return Err(err),
        }
    }
}

#[cfg(feature = "fleet")]
/// Why `participant`, a robot in a scene's run, should leave it: the run is closing,
/// the host lost it (it waits in the lobby
/// again, where its run re-sends its join), or another process took its
/// name over (it stops). `None` while it is in the run, or outside a scene.
pub fn leaving(participant: &Participant) -> Option<Leave> {
    let mine = participant.generation()?;
    if participant.host_lost().is_some() {
        return Some(Leave::Lost);
    }
    participant.authority_view().with(|view| {
        let view = view?;
        view.scene.as_ref()?;
        let current = view
            .participant(participant.id())
            .filter(|p| p.present)
            .map(|p| p.generation);
        leave(AuthorityView::closing(view), current, mine)
    })
}

#[cfg(feature = "fleet")]
/// A [`Leave`] as bindings name it: the closing reason (`scene_ended`),
/// `robot_lost` or `superseded`.
pub fn leave_name(leave: &Leave) -> String {
    match leave {
        Leave::Closing(reason) => reason_name(reason.code),
        Leave::Lost => reason_name(ReasonCode::RobotLost),
        Leave::Superseded => "superseded".into(),
    }
}

#[cfg(feature = "fleet")]
/// A run's stages, as [`run_stage_name`] names them.
const RUN_STAGES: [RunStage; 11] = [
    RunStage::Checking,
    RunStage::Hosting,
    RunStage::Assigning,
    RunStage::Declaring,
    RunStage::Binding,
    RunStage::Arming,
    RunStage::Running,
    RunStage::Ending,
    RunStage::Ended,
    RunStage::Failed,
    RunStage::Rejoining,
];

#[cfg(feature = "fleet")]
/// The run stage `name` names (`running`).
///
/// # Errors
/// An unknown name, listing the known ones.
pub fn run_stage(name: &str) -> Result<RunStage, String> {
    RUN_STAGES
        .into_iter()
        .find(|s| run_stage_name(*s) == name)
        .ok_or_else(|| {
            let names: Vec<String> = RUN_STAGES.into_iter().map(run_stage_name).collect();
            format!("run stage {name:?}: one of {}", names.join(", "))
        })
}

#[cfg(feature = "fleet")]
/// A scene run this process started from the home daemon's lobby
/// (`run_scene`, `runScene`).
pub struct SceneRun {
    client: LobbyClient,
    run: String,
}

#[cfg(feature = "fleet")]
impl SceneRun {
    /// Runs `scene_toml` at the home daemon `home` names (`None`: this
    /// machine's), waiting up to `timeout` for it (`manual_arm`: wait for
    /// [`Self::arm`]). A refused scene is [`SessionError::Rejected`].
    ///
    /// # Errors
    /// The refusal, or what reaching the daemon failed with.
    pub async fn start(
        home: Option<&str>,
        scene_toml: &str,
        manual_arm: bool,
        timeout: Duration,
    ) -> Result<Self, SessionError> {
        let client = LobbyClient::connect(&find_home(home).await?, timeout).await?;
        match client.run_scene(scene_toml, manual_arm).await {
            Ok(state) => Ok(Self {
                client,
                run: state.run,
            }),
            Err(err) => {
                let _ = client.close().await;
                Err(err)
            }
        }
    }

    /// The run's id.
    pub fn run(&self) -> &str {
        &self.run
    }

    /// The run's state now.
    ///
    /// # Errors
    /// What the call failed with.
    pub async fn status(&self) -> Result<RunState, SessionError> {
        self.client.status(&self.run, 0, None).await
    }

    /// Where the run's members join (its own controllers among them), once
    /// its host is up. On a satellite the text carries no device key: this
    /// process joins with the one it keeps for the run.
    ///
    /// # Errors
    /// What the call failed with.
    pub async fn address(&self) -> Result<Option<String>, SessionError> {
        let state = self.status().await?;
        Ok(run_address(self.client.lobby(), &state)?.map(|a| a.invite()))
    }

    /// Follows the run until it is at `stage` (see
    /// [`LobbyClient::wait_until`]).
    ///
    /// # Errors
    /// The run ending first, the timeout, or what a call failed with.
    pub async fn wait_until(
        &self,
        stage: RunStage,
        timeout: Duration,
    ) -> Result<RunState, SessionError> {
        self.client.wait_until(&self.run, stage, timeout).await
    }

    /// Arms the run (`manual`, or after a rejoin).
    ///
    /// # Errors
    /// The arm's refusal, or what the call failed with.
    pub async fn arm(&self) -> Result<RunState, SessionError> {
        self.client.arm(&self.run).await
    }

    /// Re-arms `robot` in the run: see [`LobbyClient::rearm`].
    ///
    /// # Errors
    /// The re-arm's refusal, or what the call failed with.
    pub async fn rearm(&self, robot: &ParticipantId) -> Result<RunState, SessionError> {
        self.client.rearm(&self.run, robot).await
    }

    /// Stops the run and waits up to `timeout` for it to end.
    ///
    /// # Errors
    /// What the calls failed with.
    pub async fn stop(&self, timeout: Duration) -> Result<RunState, SessionError> {
        let state = self.client.stop(&self.run).await?;
        if matches!(state.stage, RunStage::Ended | RunStage::Failed) {
            return Ok(state);
        }
        match self
            .client
            .wait_until(&self.run, RunStage::Ended, timeout)
            .await
        {
            Err(SessionError::Rejected(_)) => self.status().await,
            other => other,
        }
    }

    /// Stops the run unless it ended, and leaves the lobby.
    ///
    /// # Errors
    /// What leaving failed with.
    pub async fn close(self, timeout: Duration) -> Result<(), SessionError> {
        let over = self
            .status()
            .await
            .is_ok_and(|s| matches!(s.stage, RunStage::Ended | RunStage::Failed));
        if !over {
            let _ = self.stop(timeout).await;
        }
        self.client.close().await
    }
}

/// Publishes one sample from its fields' bytes, in declaration order: a
/// lone field already packed (a multiple of 8 bytes) is sent without a
/// further copy.
pub fn publish(
    robot: &Published,
    path: &str,
    capture_ns: i64,
    mut fields: Vec<Vec<u8>>,
) -> Result<Option<u64>, SessionError> {
    let packed = robot.part(path).is_some_and(|part| {
        matches!(fields.as_slice(), [only] if part.fields.len() == 1 && only.len().is_multiple_of(8))
    });
    if packed {
        let value = fields.pop().map(Bytes::from).unwrap_or_default();
        robot.publish_packed(path, capture_ns, value)
    } else {
        let refs: Vec<&[u8]> = fields.iter().map(Vec::as_slice).collect();
        robot.publish(path, capture_ns, &refs)
    }
}

/// A due command's fields by name, or `None` without a command.
pub fn due_fields(robot: &Robot, due: &Due) -> Result<Option<Vec<(String, Tensor)>>, String> {
    let (Some(action), Some(command)) = (
        &due.action,
        robot.part(&due.path).and_then(|p| p.command.as_ref()),
    ) else {
        return Ok(None);
    };
    let layout = command.layout()?;
    let mut fields = Vec::with_capacity(command.fields.len());
    for (i, field) in command.fields.iter().enumerate() {
        let range = layout.field(i).ok_or("a text command")?;
        let bytes = action
            .value
            .get(range)
            .ok_or("an action shorter than its stride")?;
        fields.push((
            field.name.clone(),
            Tensor {
                dtype: field.dtype,
                shape: field.shape.clone(),
                data: bytes.to_vec(),
            },
        ));
    }
    Ok(Some(fields))
}

/// The inputs a due command's chunk was computed from: `(path, seq_first,
/// seq_last)`, with the robot's paths.
pub fn due_based_on(robot: &Robot, due: &Due) -> Vec<(String, u64, u64)> {
    due.action
        .iter()
        .flat_map(|a| &a.chunk.based_on)
        .map(|r| {
            let path = robot
                .path_of(r.channel)
                .map_or_else(|| r.channel.to_string(), str::to_string);
            (path, r.seq_first, r.seq_last)
        })
        .collect()
}

/// Detaches a robot no other call holds; one still shared is dropped (its
/// presence withdrawn) when the last call returns.
pub async fn close_robot(robot: Arc<Robot>) -> Result<(), SessionError> {
    match Arc::try_unwrap(robot) {
        Ok(robot) => robot.detach().await,
        Err(_) => Ok(()),
    }
}

/// Why [`bind_policy`] failed.
#[derive(Debug, thiserror::Error)]
pub enum BindPolicyError {
    /// The policy map does not parse.
    #[error("binding refused: {0}")]
    Policy(String),
    /// Joining or binding failed.
    #[error(transparent)]
    Controller(#[from] ControllerError),
}

/// Joins as policy `name` at `priority`, waits for the robot's parts, binds
/// `policy_json` with `options` and declares the binding. A refused
/// policy leaves the session.
pub async fn bind_policy(
    address: &SessionAddress,
    name: &str,
    policy_json: &str,
    options: &RunOptions,
    priority: PriorityClass,
    timeout: Duration,
) -> Result<Controller, BindPolicyError> {
    bind_policy_with(address, name, policy_json, None, options, priority, timeout)
        .await
        .map(|(controller, _)| controller)
}

/// [`bind_policy`] for a policy that also publishes annotations: it joins
/// as a controller and an embodiment, declares `publishes_json` (an
/// authored map of parts, none commandable) under its own name, and
/// returns their publishers. Its own parts are never bound to it; other
/// policies may read them.
pub async fn bind_policy_publishing(
    address: &SessionAddress,
    name: &str,
    policy_json: &str,
    publishes_json: &str,
    options: &RunOptions,
    priority: PriorityClass,
    timeout: Duration,
) -> Result<(Controller, Published), BindPolicyError> {
    let (controller, published) = bind_policy_with(
        address,
        name,
        policy_json,
        Some(publishes_json),
        options,
        priority,
        timeout,
    )
    .await?;
    let published =
        published.ok_or_else(|| BindPolicyError::Policy(format!("{name}: no parts")))?;
    Ok((controller, published))
}

async fn bind_policy_with(
    address: &SessionAddress,
    name: &str,
    policy_json: &str,
    publishes_json: Option<&str>,
    options: &RunOptions,
    priority: PriorityClass,
    timeout: Duration,
) -> Result<(Controller, Option<Published>), BindPolicyError> {
    let policy = Policy::parse(policy_json)
        .map_err(|why| BindPolicyError::Policy(format!("{name}: {why}")))?;
    if let Some(json) = publishes_json {
        let commandable = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(json)
            .map_err(|why| BindPolicyError::Policy(format!("{name}: publishes: {why}")))?
            .into_iter()
            .filter(|(_, part)| part.get("command").is_some())
            .map(|(path, _)| path)
            .collect::<Vec<_>>();
        if !commandable.is_empty() {
            return Err(BindPolicyError::Policy(format!(
                "{name}: a policy publishes, never takes commands: {} declare `command`",
                commandable.join(", ")
            )));
        }
    }
    let roles = match publishes_json {
        Some(_) => vec![Role::Controller, Role::Embodiment],
        None => vec![Role::Controller],
    };
    let participant = address
        .join(name, roles, priority, timeout)
        .await
        .map_err(ControllerError::from)?;
    let published = match publishes_json {
        Some(json) => match Published::declare(&participant, json).await {
            Ok(published) => Some(published),
            Err(err) => {
                let _ = participant.detach().await;
                return Err(ControllerError::from(err).into());
            }
        },
        None => None,
    };
    if let Some(robot) = lockstep_robot_named(address, &participant).await {
        drop(published);
        let _ = participant.detach().await;
        return Err(ControllerError::from(BindError::Options(robot)).into());
    }
    let config = ControllerConfig {
        bind_timeout: timeout,
        ..ControllerConfig::default()
    };
    let participant = Arc::new(participant);
    match Controller::bind(participant.clone(), &policy, options, config).await {
        Ok(controller) => Ok((controller, published)),
        Err(err) => {
            drop(published);
            if let Ok(participant) = Arc::try_unwrap(participant) {
                let _ = participant.detach().await;
            }
            Err(err.into())
        }
    }
}

/// The refusal for a session address that is a lockstep robot's
/// participant address: `host:port/cartpole` names session `cartpole`, which
/// holds nothing, when it meant robot `cartpole` of the default session. Only
/// asked of a session with no robot yet, so a policy waiting for its robot
/// pays nothing more.
async fn lockstep_robot_named(
    address: &SessionAddress,
    participant: &crate::participant::Participant,
) -> Option<String> {
    if address.name == crate::address::DEFAULT_SESSION {
        return None;
    }
    let me = participant.id();
    let robots = participant
        .authority()
        .is_some_and(|view| view.parts.iter().any(|p| &p.owner != me));
    if robots {
        return None;
    }
    let robot = crate::wire::ParticipantId::new(&address.name).ok()?;
    let keys = crate::wire::SessionKeys::new(
        &address.tenant,
        crate::address::session_id(crate::address::DEFAULT_SESSION),
    );
    if !crate::controller::serves_lockstep(participant, &keys.lockstep(&robot)).await {
        return None;
    }
    Some(format!(
        "{address} is robot {robot}'s participant address, not a session: {}",
        crate::controller::lockstep_refusal(&robot, Some(&address.to_string())),
    ))
}

/// Stops a policy's run, gives its leases back and detaches, once no other
/// call holds it (a `run` returns within one idle slice; after half a
/// second it is left to drop).
pub async fn close_policy(controller: Arc<Controller>) -> Result<(), SessionError> {
    controller.stop();
    let mut controller = controller;
    for _ in 0..50 {
        match Arc::try_unwrap(controller) {
            Ok(owned) => {
                let participant = owned.close().await;
                return match Arc::try_unwrap(participant) {
                    Ok(participant) => participant.detach().await,
                    Err(_) => Ok(()),
                };
            }
            Err(shared) => {
                controller = shared;
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }
    Ok(())
}

/// A fixed-rate loop timer on `CLOCK_MONOTONIC`: [`Self::next`] says until
/// when to sleep. A tick more than a period late skips the missed ones
/// rather than bursting.
pub struct Rate {
    period_ns: i64,
    next_ns: AtomicU64,
    missed: AtomicU64,
}

impl Rate {
    /// A timer at `hz`.
    pub fn new(hz: f64) -> Result<Self, String> {
        if !(hz.is_finite() && hz > 0.0) {
            return Err(format!("rate {hz} Hz is not positive"));
        }
        Ok(Self {
            period_ns: ((1e9 / hz).round() as i64).max(1),
            next_ns: AtomicU64::new(0),
            missed: AtomicU64::new(0),
        })
    }

    /// The start of the next period, at `now_ns`, and how late this call is
    /// for it.
    pub fn next(&self, now_ns: i64) -> (i64, i64) {
        let mut next = i64::try_from(self.next_ns.load(Ordering::Relaxed)).unwrap_or(0);
        if next == 0 {
            next = now_ns;
        }
        if now_ns > next + self.period_ns {
            let behind = (now_ns - next) / self.period_ns;
            self.missed
                .fetch_add(behind.unsigned_abs(), Ordering::Relaxed);
            next += behind * self.period_ns;
        }
        self.next_ns.store(
            u64::try_from(next + self.period_ns).unwrap_or(0),
            Ordering::Relaxed,
        );
        (next, (now_ns - next).max(0))
    }

    /// Periods skipped because a call came more than a period late.
    pub fn missed(&self) -> u64 {
        self.missed.load(Ordering::Relaxed)
    }
}

/// Joins as observer `name` and records the session into a new MCAP file
/// at `path`.
#[cfg(feature = "record")]
pub async fn start_recorder(
    address: &SessionAddress,
    path: &str,
    name: &str,
    timeout: Duration,
) -> Result<crate::record::Recorder, crate::record::RecordError> {
    let participant = address
        .join(name, vec![Role::Observer], PriorityClass::Scripted, timeout)
        .await?;
    crate::record::Recorder::start(Arc::new(participant), path).await
}

/// Finishes a recording and detaches its observer.
#[cfg(feature = "record")]
pub async fn finish_recorder(
    recorder: crate::record::Recorder,
) -> Result<crate::record::RecordStats, crate::record::RecordError> {
    let participant = recorder.participant().clone();
    let stats = recorder.finish().await?;
    if let Ok(participant) = Arc::try_unwrap(participant) {
        participant.detach().await?;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_the_bindings_spelling() {
        assert_eq!(status_name(CtlStatus::LeaseBusy), "LEASE_BUSY");
        assert_eq!(rejection_name(Rejection::TooOld), "too_old");
        assert_eq!(mode_name(EdgeMode::Driving), "driving");
        assert!(priority("pilot").is_err());
    }

    #[test]
    fn a_late_rate_tick_skips_the_missed_periods() {
        let rate = Rate::new(100.0).expect("rate");
        assert_eq!(rate.next(0), (0, 0));
        assert_eq!(rate.next(5_000_000), (10_000_000, 0));
        // 35 ms later than the next slot: three periods skipped.
        let (next, late) = rate.next(55_000_000);
        assert_eq!((next, late, rate.missed()), (50_000_000, 5_000_000, 3));
    }
}

/// One participant's link in the host's health
/// ([`Participant::health`](crate::participant::Participant::health)),
/// as bindings show it: durations in ms, parts by name (`robot:path`).
#[derive(Clone, Debug, PartialEq)]
pub struct HealthLink {
    /// Whose link.
    pub participant: String,
    /// Heartbeats the host received from it.
    pub heartbeats: u64,
    /// The latest heartbeat round trip (`None`: none measured yet).
    pub rtt_ms: Option<f64>,
    /// Smoothed round-trip jitter.
    pub jitter_ms: f64,
    /// The largest gap between its heartbeats in the host's window.
    pub heartbeat_gap_max_ms: f64,
    /// Since its last heartbeat, when the host published.
    pub last_heartbeat_age_ms: f64,
    /// Its clock mapping, host minus participant, and the mapping's error
    /// bound (`None`: not mapped yet).
    pub offset_ms: Option<(f64, f64)>,
    /// Its self-reported health: `ok`, `degraded`, `unhealthy` or `unknown`.
    pub health: &'static str,
    /// Edges: the newest chunk seq accepted (0: none).
    pub last_chunk_accepted: u64,
    /// Edges: the newest chunk seq applied (0: none).
    pub last_chunk_applied: u64,
    /// Edges: the actions queued.
    pub queue_depth: u32,
    /// Edges: chunks refused since it attached, by reason (nonzero only).
    pub rejected_by: Vec<(String, u64)>,
    /// Edges: each part whose edge refuses chunks for a clock error.
    pub clock_refused: Vec<HealthClockRefusal>,
    /// Each stream it publishes: the part, samples published and samples
    /// its publication plan coalesced.
    pub published: Vec<(String, u64, u64)>,
    /// A bound controller's run counters, from its latest report.
    pub run: Option<HealthRun>,
    /// The newest seq it received per part it subscribes to.
    pub last_seq_seen: Vec<(String, u64)>,
}

/// A part whose edge refuses chunks for a clock error.
#[derive(Clone, Debug, PartialEq)]
pub struct HealthClockRefusal {
    /// The part, `robot:path`.
    pub part: String,
    /// Whose clock mapping is over the bound.
    pub participant: String,
    /// The mapping's error.
    pub error_ms: f64,
    /// The bound the edge holds it to.
    pub bound_ms: f64,
}

/// A controller's run counters as its latest report gave them.
#[derive(Clone, Debug, PartialEq)]
pub struct HealthRun {
    /// `predict` calls started.
    pub starts: u64,
    /// `predict` calls completed.
    pub completions: u64,
    /// Skipped starts by reason (`busy_skip`, `waiting_for`, `stale_input`,
    /// `window_gap`, `decode`, `no_lease`).
    pub skipped: Vec<(&'static str, u64)>,
    /// Chunks sent.
    pub chunks_sent: u64,
    /// The inference in flight's age (`None`: none).
    pub pending_ms: Option<f64>,
    /// Completion latency, median (`None`: none yet).
    pub latency_p50_ms: Option<f64>,
    /// Completion latency, 99th percentile (`None`: none yet).
    pub latency_p99_ms: Option<f64>,
    /// The completion rate is drifting (`rate_drift`).
    pub drifting: bool,
}

fn ms(ns: u64) -> f64 {
    ns as f64 * 1e-6
}

/// The links of `health`, by participant, with the parts `participant`'s
/// authority view names.
pub fn health_links(
    participant: &crate::participant::Participant,
    health: &crate::wire::HostHealth,
) -> Vec<HealthLink> {
    use crate::wire::Health;
    let name = |channel: crate::wire::ChannelId| {
        participant
            .authority_view()
            .with(|view| view.map_or_else(|| channel.to_string(), |v| v.part_name(channel)))
    };
    let nonzero = |ns: u64| (ns > 0).then(|| ms(ns));
    health
        .links
        .iter()
        .filter_map(|link| {
            let id = link.participant.as_ref()?;
            Some(HealthLink {
                participant: id.to_string(),
                heartbeats: link.heartbeats,
                rtt_ms: nonzero(link.rtt_ns),
                jitter_ms: ms(link.jitter_ns),
                heartbeat_gap_max_ms: ms(link.heartbeat_gap_max_ns),
                last_heartbeat_age_ms: ms(link.last_heartbeat_age_ns),
                offset_ms: link
                    .offset
                    .map(|(offset, error)| (offset as f64 * 1e-6, ms(error))),
                health: match link.health {
                    Health::Unknown => "unknown",
                    Health::Ok => "ok",
                    Health::Degraded => "degraded",
                    Health::Unhealthy => "unhealthy",
                },
                last_chunk_accepted: link.last_chunk_accepted,
                last_chunk_applied: link.last_chunk_applied,
                queue_depth: link.queue_depth,
                rejected_by: link
                    .rejected
                    .iter()
                    .filter(|(_, n)| *n > 0)
                    .map(|(reason, n)| (rejection_name(*reason), *n))
                    .collect(),
                clock_refused: link
                    .clock_refused
                    .iter()
                    .map(|r| HealthClockRefusal {
                        part: name(r.channel),
                        participant: r.participant.to_string(),
                        error_ms: ms(r.error_ns),
                        bound_ms: ms(r.bound_ns),
                    })
                    .collect(),
                published: link
                    .report
                    .iter()
                    .flat_map(|r| &r.streams)
                    .map(|s| (name(s.channel), s.published, s.coalesced))
                    .collect(),
                run: link
                    .report
                    .as_ref()
                    .and_then(|r| r.run)
                    .map(|run| HealthRun {
                        starts: run.starts,
                        completions: run.completions,
                        skipped: vec![
                            ("busy_skip", run.busy_skips),
                            ("waiting_for", run.waiting_for),
                            ("stale_input", run.stale_input),
                            ("window_gap", run.window_gap),
                            ("decode", run.decode),
                            ("no_lease", run.no_lease),
                        ],
                        chunks_sent: run.chunks_sent,
                        pending_ms: nonzero(run.pending_ns),
                        latency_p50_ms: nonzero(run.latency_p50_ns),
                        latency_p99_ms: nonzero(run.latency_p99_ns),
                        drifting: run.drifting,
                    }),
                last_seq_seen: link
                    .last_seq_seen
                    .iter()
                    .map(|(channel, seq)| (name(*channel), *seq))
                    .collect(),
            })
        })
        .collect()
}
