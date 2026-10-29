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

use crate::address::{Address, Placed, SessionAddress, place};
use crate::controller::{Controller, ControllerConfig, ControllerError};
use crate::edge::{Due, EdgeConfig, EdgeError, EdgeMode, Rejection, Robot};
use crate::error::SessionError;
use crate::lobby::{Leave, Lobby, LobbyClient, find_home, leave, run_address, wait_for_join};
use crate::participant::Participant;
use crate::spec::apply::Tensor;
use crate::spec::bind::Dir;
use crate::spec::policy::Allow;
use crate::spec::{BindError, Policy, RunOptions};
use crate::wire::{
    AuthorityView, CtlStatus, DType, MemberStage, ParticipantId, PriorityClass, ReasonCode, Role,
    RunStage, RunState,
};

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

/// A run's stage as bindings name it (`rejoining`).
pub fn run_stage_name(stage: RunStage) -> String {
    snake(&format!("{stage:?}"), false)
}

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

/// A policy's run-time options (§6.4). Maps are applied sorted, so errors
/// come in a stable order.
pub struct BindOptions<'a> {
    /// The policy's participant id.
    pub name: &'a str,
    /// The robot to bind to.
    pub robot: &'a str,
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
        let mut options = RunOptions::named(self.name).robot(self.robot)?;
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
/// this process, on local per-class TCP ports and this process's
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
        let placed = place(address, lease_grace).await?;
        Ok(Self {
            address: placed.address.clone(),
            placed: tokio::sync::Mutex::new(Some(placed)),
        })
    }

    /// A session another process hosts, joined at `address` (a run's,
    /// from the lobby): closing it stops nothing.
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

    /// Stops the host, if this process runs it (participants see it leave).
    /// Idempotent.
    pub async fn close(&self) -> Result<(), SessionError> {
        if let Some(mut placed) = self.placed.lock().await.take() {
            placed.close().await?;
        }
        Ok(())
    }
}

/// Joins as robot `name` and declares `robot_json` (the authored map, §3).
pub async fn connect_robot(
    address: &SessionAddress,
    name: &str,
    robot_json: &str,
    timeout: Duration,
) -> Result<Robot, EdgeError> {
    let participant = address
        .join(
            name,
            vec![Role::Embodiment],
            PriorityClass::Scripted,
            timeout,
        )
        .await?;
    Robot::declare(participant, robot_json, EdgeConfig::default()).await
}

/// How long a robot from the lobby waits for its run's host.
const RUN_ATTACH: Duration = Duration::from_secs(10);

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

/// The e-stop latch of robot `name` in this process, for the lobby whose
/// host key is `home`: one per robot process (design §6), so it outlives
/// each run's participant until cleared locally.
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

/// Why `participant`, a robot in a scene's run, should leave it (scene
/// design §3.5): the run is closing, the host lost it (it waits in the lobby
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

/// A [`Leave`] as bindings name it: the closing reason (`scene_ended`),
/// `robot_lost` or `superseded`.
pub fn leave_name(leave: &Leave) -> String {
    match leave {
        Leave::Closing(reason) => reason_name(reason.code),
        Leave::Lost => reason_name(ReasonCode::RobotLost),
        Leave::Superseded => "superseded".into(),
    }
}

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

/// A scene run this process started from the home daemon's lobby
/// (`run_scene`, `runScene`).
pub struct SceneRun {
    client: LobbyClient,
    run: String,
}

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
    robot: &Robot,
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
/// `policy_json` (§6) with `options` and declares the binding. A refused
/// policy leaves the session.
pub async fn bind_policy(
    address: &SessionAddress,
    name: &str,
    policy_json: &str,
    options: &RunOptions,
    priority: PriorityClass,
    timeout: Duration,
) -> Result<Controller, BindPolicyError> {
    let policy = Policy::parse(policy_json)
        .map_err(|why| BindPolicyError::Policy(format!("{name}: {why}")))?;
    let participant = address
        .join(name, vec![Role::Controller], priority, timeout)
        .await
        .map_err(ControllerError::from)?;
    let config = ControllerConfig {
        bind_timeout: timeout,
        ..ControllerConfig::default()
    };
    let participant = Arc::new(participant);
    match Controller::bind(participant.clone(), &policy, options, config).await {
        Ok(controller) => Ok(controller),
        Err(err) => {
            if let Ok(participant) = Arc::try_unwrap(participant) {
                let _ = participant.detach().await;
            }
            Err(err.into())
        }
    }
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
