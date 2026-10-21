//! Node-API bindings for real-time sessions (protocol v2, the v0.2 spec): an
//! embedded session host, a robot ([`rlmesh_edge::Robot`]), a bound policy
//! ([`rlmesh_session::Controller`]) and an MCAP recorder
//! ([`rlmesh_record::Recorder`]), the Node and Electron twin of the PyO3
//! binding. The `rlmesh` npm package wraps them (`src/native.ts`);
//! every rule (packing, binding, the edge, leases, the run loop) stays in
//! the Rust crates.
//!
//! Nothing here blocks the JS thread on the network: verbs that wait are
//! `async` and run on the session runtime; `publish`, `due`, `observe` and
//! `send` never wait. The run loop has a blocking thread of its own and
//! hands each start to JS through a threadsafe function; JS answers with
//! [`NativePolicy::reply`].
//!
//! An error's message starts with a line naming its kind as JSON
//! (`rlmesh:{"kind":"SessionRefused","status":"LEASE_BUSY"}`); the wrapper
//! turns it into one of its error classes.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use napi::bindgen_prelude::{BigInt, Either, Uint8Array, create_custom_tokio_runtime};
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{Error, Result, Status, tokio};
use napi_derive::napi;
use rlmesh_edge::{Due, EdgeConfig, EdgeMode, Rejection, Robot};
use rlmesh_record::{RecordError, Recorder};
use rlmesh_session::controller::sleep_until;
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::spec::policy::Allow;
use rlmesh_session::spec::{BindError, Policy, RunOptions};
use rlmesh_session::wire::{
    CtlStatus, DType, Encoding, PartSpec, PriorityClass, Role, SessionId, TenantId,
};
use rlmesh_session::{
    Clock, Controller, ControllerConfig, ControllerError, HostConfig, Observation, ObserveError,
    Predictor, RunError, RunEvent, SendError, Sent, SessionAddress, SessionError, SessionHost,
    SkipReason, Snapshot, SystemMonotonicClock, Transport,
};

/// The first line of an error's message, naming its kind for the wrapper.
const KIND_PREFIX: &str = "rlmesh:";

/// `kind` plus any fields the wrapper copies onto its error, as the
/// message's first line.
fn error(kind: &str, extra: &[(&str, String)], message: impl Into<String>) -> Error {
    let mut text = format!("{KIND_PREFIX}{{\"kind\":{kind:?}");
    for (name, value) in extra {
        text.push_str(&format!(",{name:?}:{value}"));
    }
    text.push_str("}\n");
    text.push_str(&message.into());
    Error::new(Status::GenericFailure, text)
}

/// An error's message without its kind line.
fn message_of(err: &Error) -> String {
    match err.reason.split_once('\n') {
        Some((kind, message)) if kind.starts_with(KIND_PREFIX) => message.to_string(),
        _ => err.reason.clone(),
    }
}

fn json_list(items: &[String]) -> String {
    let quoted: Vec<String> = items.iter().map(|s| format!("{s:?}")).collect();
    format!("[{}]", quoted.join(","))
}

fn exception(message: impl Into<String>) -> Error {
    error("SessionException", &[], message)
}

fn invalid(message: impl Into<String>) -> Error {
    error("ValueError", &[], message)
}

#[napi_derive::module_init]
fn init() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .thread_name("rlmesh-session")
        .enable_all()
        .build()
        .expect("failed to build the rlmesh session runtime");
    create_custom_tokio_runtime(runtime);
}

fn screaming(status: CtlStatus) -> String {
    let camel = format!("{status:?}");
    let mut out = String::with_capacity(camel.len() + 4);
    for (i, c) in camel.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_ascii_uppercase());
    }
    out
}

fn session_err(err: SessionError) -> Error {
    match err {
        SessionError::Rejected(refusal) => error(
            "SessionRefused",
            &[("status", format!("{:?}", screaming(refusal.status)))],
            refusal.message,
        ),
        SessionError::NoReply(_) | SessionError::Timeout(..) => {
            error("TimeoutError", &[], err.to_string())
        }
        SessionError::InvalidValue { .. }
        | SessionError::UnknownPart(_)
        | SessionError::WrongChannel { .. } => invalid(err.to_string()),
        other => exception(other.to_string()),
    }
}

fn bind_err(err: BindError) -> Error {
    let problems: Vec<String> = match &err {
        BindError::Refused(refusal) => refusal
            .problems
            .iter()
            .map(|p| {
                let dir = match p.dir {
                    rlmesh_session::spec::bind::Dir::In => "input",
                    rlmesh_session::spec::bind::Dir::Out => "action",
                };
                format!("[{:?},{dir:?},{:?}]", p.code.name(), p.entry)
            })
            .collect(),
        BindError::Options(_) => Vec::new(),
    };
    error(
        "BindingRefused",
        &[("problems", format!("[{}]", problems.join(",")))],
        err.to_string(),
    )
}

fn binding_refused(message: String) -> Error {
    error("BindingRefused", &[("problems", "[]".into())], message)
}

fn controller_err(err: ControllerError) -> Error {
    match err {
        ControllerError::Bind(err) => bind_err(err),
        ControllerError::Session(err) => session_err(err),
    }
}

fn send_err(err: SendError) -> Error {
    match err {
        SendError::Value(message) => invalid(message),
        SendError::Session(err) => session_err(err),
    }
}

fn observe_err(err: ObserveError) -> Error {
    match err {
        ObserveError::Skipped(skip) => {
            let reason = if !skip.gaps.is_empty() {
                SkipReason::WindowGap
            } else if !skip.stale.is_empty() {
                SkipReason::StaleInput
            } else {
                SkipReason::Waiting
            };
            error(
                "StartSkipped",
                &[
                    ("reason", format!("{:?}", reason.name())),
                    ("waiting", json_list(&skip.waiting)),
                    ("stale", json_list(&skip.stale)),
                    ("gaps", json_list(&skip.gaps)),
                ],
                format!(
                    "start skipped ({reason}): waiting for {:?}, stale {:?}, window gaps {:?}",
                    skip.waiting, skip.stale, skip.gaps
                ),
            )
        }
        other => exception(other.to_string()),
    }
}

/// A session address (`{ip}:{port}/{tenant}/{session}`); joining it puts
/// capture times on `CLOCK_MONOTONIC`, `process.hrtime.bigint()`'s clock.
fn address(text: &str) -> Result<SessionAddress> {
    text.parse()
        .map_err(|e: SessionError| invalid(e.to_string()))
}

fn timeout(seconds: f64) -> Result<Duration> {
    Duration::try_from_secs_f64(seconds)
        .map_err(|_| invalid(format!("timeout {seconds} is not a duration")))
}

fn priority(name: &str) -> Result<PriorityClass> {
    match name {
        "safety" => Ok(PriorityClass::Safety),
        "operator" => Ok(PriorityClass::Operator),
        "model" => Ok(PriorityClass::Model),
        "scripted" => Ok(PriorityClass::Scripted),
        other => Err(invalid(format!(
            "priority {other:?}: safety, operator, model or scripted"
        ))),
    }
}

/// A `bigint` from JS as nanoseconds.
fn ns(value: &BigInt, what: &str) -> Result<i64> {
    match value.get_i64() {
        (ns, true) => Ok(ns),
        _ => Err(invalid(format!("{what}: out of range for nanoseconds"))),
    }
}

/// A session host in this process, on local per-class TCP ports.
#[napi]
pub struct NativeHost {
    address: SessionAddress,
    host: Mutex<Option<(SessionHost, Transport)>>,
}

/// Starts a session host listening on `ip`.
#[napi]
pub async fn open_host(ip: String, tenant: String, lease_grace_ms: u32) -> Result<NativeHost> {
    let ip: IpAddr = ip.parse().map_err(|e| invalid(format!("ip {ip:?}: {e}")))?;
    let tenant = TenantId::new(&tenant).map_err(|e| invalid(e.to_string()))?;
    let session = SessionId::new();
    let (transport, base) = rlmesh_session::transport::open_local_listeners(ip)
        .await
        .map_err(session_err)?;
    let mut config = HostConfig::new(tenant.clone(), session);
    config.clock = SystemMonotonicClock.shared();
    config.lease_grace = Duration::from_millis(u64::from(lease_grace_ms.max(1)));
    config.health_log = None;
    let host = SessionHost::start(transport.clone(), config)
        .await
        .map_err(session_err)?;
    Ok(NativeHost {
        address: SessionAddress {
            base,
            tenant,
            session,
        },
        host: Mutex::new(Some((host, transport))),
    })
}

#[napi]
impl NativeHost {
    /// `{ip}:{base_port}/{tenant}/{session}`: what participants connect to.
    #[napi(getter)]
    pub fn address(&self) -> String {
        self.address.to_string()
    }

    /// Stops the host (participants see it leave).
    #[napi]
    pub async fn close(&self) -> Result<()> {
        let taken = self.host.lock().map(|mut h| h.take()).unwrap_or(None);
        if let Some((host, transport)) = taken {
            host.shutdown().await;
            transport.close().await.map_err(session_err)?;
        }
        Ok(())
    }
}

/// One field of a part: its name, dtype and shape.
#[napi(object)]
pub struct FieldInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u32>,
}

fn fields_info(fields: &[rlmesh_session::wire::FieldSpec]) -> Vec<FieldInfo> {
    fields
        .iter()
        .map(|f| FieldInfo {
            name: f.name.clone(),
            dtype: f.dtype.name().to_string(),
            shape: f.shape.clone(),
        })
        .collect()
}

/// A commandable part's command.
#[napi(object)]
pub struct CommandInfo {
    pub mode: String,
    pub fields: Vec<FieldInfo>,
    pub group: Option<String>,
    pub safe: String,
    pub control_hz: f64,
}

/// A part as the host normalised it.
#[napi(object)]
pub struct PartInfo {
    pub path: String,
    pub kind: String,
    pub hz: f64,
    pub event: bool,
    pub encoding: String,
    pub fields: Vec<FieldInfo>,
    pub command: Option<CommandInfo>,
}

fn part_info(part: &PartSpec) -> PartInfo {
    PartInfo {
        path: part.path.to_string(),
        kind: part.kind.clone(),
        hz: part.hz,
        event: part.event,
        encoding: part.encoding.wire_name().to_string(),
        fields: fields_info(&part.fields),
        command: part.command.as_ref().map(|command| CommandInfo {
            mode: command.mode.clone(),
            fields: fields_info(&command.fields),
            group: command.group.clone(),
            safe: command.safe_behavior.clone(),
            control_hz: part.control_hz,
        }),
    }
}

/// An array: its dtype and shape, and its bytes (native-endian, packed).
#[napi(object)]
pub struct TensorData {
    /// The field's name, inside a dict value or a command.
    pub name: Option<String>,
    pub dtype: String,
    pub shape: Vec<u32>,
    pub data: Uint8Array,
}

fn tensor_out(name: Option<String>, tensor: Tensor) -> TensorData {
    TensorData {
        name,
        dtype: tensor.dtype.name().to_string(),
        shape: tensor.shape,
        data: Uint8Array::new(tensor.data),
    }
}

fn tensor_in(entry: &str, tensor: &TensorData) -> Result<Tensor> {
    let dtype = DType::from_name(&tensor.dtype)
        .filter(|d| *d != DType::Utf8)
        .ok_or_else(|| invalid(format!("{entry}: dtype {} is not numeric", tensor.dtype)))?;
    let data = tensor.data.to_vec();
    let elements: u64 = tensor.shape.iter().map(|d| u64::from(*d)).product();
    if elements.checked_mul(dtype.size()) != u64::try_from(data.len()).ok() {
        return Err(invalid(format!(
            "{entry}: {} bytes for {} {:?}",
            data.len(),
            dtype.name(),
            tensor.shape
        )));
    }
    Ok(Tensor {
        dtype,
        shape: tensor.shape.clone(),
        data,
    })
}

/// One input or action entry's value: an array, several named arrays, or
/// text.
#[napi(object)]
pub struct EntryValue {
    pub entry: String,
    pub tensor: Option<TensorData>,
    pub fields: Option<Vec<TensorData>>,
    pub text: Option<String>,
}

fn value_out(entry: String, value: Value) -> EntryValue {
    let mut out = EntryValue {
        entry,
        tensor: None,
        fields: None,
        text: None,
    };
    match value {
        Value::Tensor(tensor) => out.tensor = Some(tensor_out(None, tensor)),
        Value::Dict(fields) => {
            out.fields = Some(
                fields
                    .into_iter()
                    .map(|(name, tensor)| tensor_out(Some(name), tensor))
                    .collect(),
            );
        }
        Value::Text(text) => out.text = Some(text),
    }
    out
}

fn values_in(values: &[EntryValue]) -> Result<Vec<(String, Value)>> {
    values
        .iter()
        .map(|v| {
            let value = match (&v.tensor, &v.fields) {
                (Some(tensor), None) => Value::Tensor(tensor_in(&v.entry, tensor)?),
                (None, Some(fields)) => Value::Dict(
                    fields
                        .iter()
                        .map(|t| {
                            let name = t.name.clone().unwrap_or_default();
                            let tensor = tensor_in(&format!("{}.{name}", v.entry), t)?;
                            Ok((name, tensor))
                        })
                        .collect::<Result<_>>()?,
                ),
                _ => {
                    return Err(invalid(format!(
                        "{}: an action is an array or a dict of arrays",
                        v.entry
                    )));
                }
            };
            Ok((v.entry.clone(), value))
        })
        .collect()
}

fn mode_name(mode: EdgeMode) -> &'static str {
    match mode {
        EdgeMode::Idle => "idle",
        EdgeMode::Driving => "driving",
        EdgeMode::Hold => "hold",
        EdgeMode::Safe => "safe",
        EdgeMode::Estop => "estop",
    }
}

/// The inputs a chunk was computed from, by the robot's path.
#[napi(object)]
pub struct InputSeqs {
    pub path: String,
    pub seq_first: i64,
    pub seq_last: i64,
}

/// A command-group handover starting this tick.
#[napi(object)]
pub struct Handover {
    pub group: String,
    pub from: String,
    pub to: String,
}

/// What the robot does this control tick.
#[napi(object)]
pub struct DueTick {
    pub path: String,
    /// `driving`, `hold`, `safe`, `estop` or `idle`.
    pub mode: String,
    /// The command fields (validated and clamped), or none.
    pub fields: Option<Vec<TensorData>>,
    pub safe: Option<String>,
    pub handover: Option<Handover>,
    pub lease_token: Option<BigInt>,
    pub chunk_seq: Option<i64>,
    pub index: Option<u32>,
    pub received_ns: Option<BigInt>,
    pub slot_ns: Option<BigInt>,
    pub based_on: Vec<InputSeqs>,
}

fn seq(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn due_tick(robot: &Robot, due: Due) -> Result<DueTick> {
    let Due {
        path,
        mode,
        action,
        safe,
        handover,
    } = due;
    let fields = match (&action, robot.part(&path).and_then(|p| p.command.as_ref())) {
        (Some(action), Some(command)) => {
            let layout = command.layout().map_err(exception)?;
            let mut out = Vec::with_capacity(command.fields.len());
            for (i, field) in command.fields.iter().enumerate() {
                let range = layout.field(i).ok_or_else(|| exception("a text command"))?;
                let bytes = action
                    .value
                    .get(range)
                    .ok_or_else(|| exception("an action shorter than its stride"))?;
                out.push(tensor_out(
                    Some(field.name.clone()),
                    Tensor {
                        dtype: field.dtype,
                        shape: field.shape.clone(),
                        data: bytes.to_vec(),
                    },
                ));
            }
            Some(out)
        }
        _ => None,
    };
    let chunk = action.as_ref().map(|a| &a.chunk);
    Ok(DueTick {
        mode: mode_name(mode).to_string(),
        fields,
        safe: safe.map(|s| s.to_string()),
        handover: handover.map(|h| Handover {
            group: h.group,
            from: h.from,
            to: h.to,
        }),
        lease_token: chunk.map(|c| BigInt::from(c.lease_token.0)),
        chunk_seq: chunk.map(|c| seq(c.seq)),
        index: action.as_ref().map(|a| a.index),
        received_ns: chunk.map(|c| BigInt::from(c.receive_ns)),
        slot_ns: action.as_ref().map(|a| BigInt::from(a.slot_ns)),
        based_on: chunk
            .map(|c| {
                c.based_on
                    .iter()
                    .map(|r| InputSeqs {
                        path: robot
                            .path_of(r.channel)
                            .map_or_else(|| r.channel.to_string(), str::to_string),
                        seq_first: seq(r.seq_first),
                        seq_last: seq(r.seq_last),
                    })
                    .collect()
            })
            .unwrap_or_default(),
        path,
    })
}

/// Ticks spent in an edge mode, and how often it was entered.
#[napi(object)]
pub struct ModeCount {
    pub mode: String,
    pub ticks: i64,
    pub entries: i64,
}

/// Chunks an edge rejected for one reason (`too_old`, `no_clock_mapping`).
#[napi(object)]
pub struct RejectionCount {
    pub reason: String,
    pub count: i64,
}

/// A rejection reason as a snake_case name.
fn rejection_name(reason: Rejection) -> String {
    let camel = format!("{reason:?}");
    let mut out = String::with_capacity(camel.len() + 4);
    for (i, c) in camel.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

/// The counters of one edge.
#[napi(object)]
pub struct EdgeStats {
    pub accepted: i64,
    pub rejected: i64,
    /// The rejections by reason, those that occurred.
    pub rejected_by: Vec<RejectionCount>,
    pub clamped: i64,
    pub handovers: i64,
    pub safe_reports: i64,
    pub modes: Vec<ModeCount>,
}

/// A robot: its map declared, a publisher per part and an edge per command
/// group.
#[napi]
pub struct NativeRobot {
    name: String,
    robot: Mutex<Option<Arc<Robot>>>,
    encoder: Mutex<Option<Encoder>>,
}

/// A frame for the encoder thread: the part, its capture time, its pixels.
type Frame = (String, i64, Vec<u8>);

/// The robot's JPEG encoder thread. A JPEG part's `offer` encodes on the
/// caller's thread (2 ms for 640x480, 6 ms for 1280x720), which in Node is
/// the JS thread every loop shares; Python's camera thread pays it with the
/// GIL released. Here frames queue to one thread instead, in order.
struct Encoder {
    frames: SyncSender<Frame>,
    thread: std::thread::JoinHandle<()>,
}

impl Encoder {
    /// Frames waiting behind the one being encoded; one more is dropped, as
    /// the image publisher drops a frame that arrives before its slot is
    /// sent.
    const QUEUE: usize = 2;

    fn start(robot: Arc<Robot>) -> std::io::Result<Self> {
        let (frames, queued) = mpsc::sync_channel::<Frame>(Self::QUEUE);
        let thread = std::thread::Builder::new()
            .name("rlmesh-encoder".into())
            .spawn(move || {
                for (path, capture, pixels) in queued {
                    // Its outcome is in the publisher's own counters.
                    let _ = robot.publish(&path, capture, &[pixels.as_slice()]);
                }
            })?;
        Ok(Self { frames, thread })
    }
}

/// Attaches as `name` (the robot's participant id), declares `robot_json`
/// (the authored map, §3) and starts its publishers and edges.
#[napi]
pub async fn connect_robot(
    address: String,
    name: String,
    robot_json: String,
    timeout_s: f64,
) -> Result<NativeRobot> {
    let address = self::address(&address)?;
    let wait = timeout(timeout_s)?;
    let participant = address
        .join(&name, vec![Role::Embodiment], PriorityClass::Scripted, wait)
        .await
        .map_err(session_err)?;
    let robot = Robot::declare(participant, &robot_json, EdgeConfig::default())
        .await
        .map_err(|err| match err {
            rlmesh_edge::EdgeError::Session(err) => session_err(err),
            other => exception(other.to_string()),
        })?;
    Ok(NativeRobot {
        name,
        robot: Mutex::new(Some(Arc::new(robot))),
        encoder: Mutex::new(None),
    })
}

impl NativeRobot {
    fn robot(&self) -> Result<Arc<Robot>> {
        self.robot
            .lock()
            .ok()
            .and_then(|r| r.clone())
            .ok_or_else(|| exception(format!("robot {} is closed", self.name)))
    }
}

#[napi]
impl NativeRobot {
    /// The robot's participant id.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// The parts as the host normalised them, in authored order.
    #[napi]
    pub fn parts(&self) -> Result<Vec<PartInfo>> {
        Ok(self.robot()?.parts().iter().map(part_info).collect())
    }

    /// Publishes one sample of `path` from its fields in declaration order
    /// (packed bytes; a string for text), captured at `capture_ns` (now by
    /// default). The values are copied before this returns. Returns the
    /// sample's seq, or `null` when coalesced or handed to the image
    /// encoder (a JPEG part's frame is encoded off the JS thread).
    #[napi]
    pub fn publish(
        &self,
        path: String,
        fields: Vec<Either<String, Uint8Array>>,
        capture_ns: Option<BigInt>,
    ) -> Result<Option<i64>> {
        let robot = self.robot()?;
        let mut owned: Vec<Vec<u8>> = fields
            .into_iter()
            .map(|field| match field {
                Either::A(text) => text.into_bytes(),
                Either::B(bytes) => bytes.to_vec(),
            })
            .collect();
        let capture = match capture_ns {
            Some(at) => ns(&at, "captureNs")?,
            None => robot.participant().clock().now_ns(),
        };
        if let Some(part) = robot.part(&path)
            && matches!(part.encoding, Encoding::Jpeg { .. })
            && let [pixels] = owned.as_mut_slice()
        {
            let expected: u64 = part
                .fields
                .iter()
                .flat_map(|f| &f.shape)
                .map(|d| u64::from(*d))
                .product();
            if u64::try_from(pixels.len()).ok() != Some(expected) {
                return Err(invalid(format!(
                    "{path}: frame is {} bytes, not {expected}",
                    pixels.len()
                )));
            }
            let frame = (path, capture, std::mem::take(pixels));
            let mut encoder = self.encoder.lock().map_err(|_| exception("poisoned"))?;
            if encoder.is_none() {
                *encoder = Some(Encoder::start(robot).map_err(|e| exception(e.to_string()))?);
            }
            if let Some(encoder) = encoder.as_ref() {
                // A full queue drops this frame: the encoder is behind.
                let _ = encoder.frames.try_send(frame);
            }
            return Ok(None);
        }
        let packed = robot.part(&path).and_then(|part| match owned.as_slice() {
            [only] if part.fields.len() == 1 && only.len().is_multiple_of(8) => Some(()),
            _ => None,
        });
        let sent = match packed {
            Some(()) => {
                let value = owned.pop().map(Bytes::from).unwrap_or_default();
                robot.publish_packed(&path, capture, value)
            }
            None => {
                let refs: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
                robot.publish(&path, capture, &refs)
            }
        };
        Ok(sent.map_err(session_err)?.map(seq))
    }

    /// One control tick of the edge serving `path` and its command group:
    /// what to dispatch now, or `null`. Call it once per tick per group.
    #[napi]
    pub fn due(&self, path: String) -> Result<Option<DueTick>> {
        let robot = self.robot()?;
        let due = robot.due(&path).map_err(session_err)?;
        due.map(|due| due_tick(&robot, due)).transpose()
    }

    /// The robot switched control modes after a handover of `path`'s group:
    /// the group may dispatch again.
    #[napi]
    pub fn handover_done(&self, path: String) -> Result<()> {
        self.robot()?.handover_done(&path).map_err(session_err)
    }

    /// DISARMED or SAFE to ARMED, in a fresh epoch.
    #[napi]
    pub async fn arm(&self) -> Result<()> {
        let robot = self.robot()?;
        robot.participant().arm().await.map_err(session_err)?;
        Ok(())
    }

    /// Back to DISARMED.
    #[napi]
    pub async fn disarm(&self) -> Result<()> {
        let robot = self.robot()?;
        robot.participant().disarm().await.map_err(session_err)?;
        Ok(())
    }

    /// Emergency stop: latched until a local `clear`, then disarm and arm.
    #[napi]
    pub async fn estop(&self, reason: String) -> Result<()> {
        let robot = self.robot()?;
        robot
            .participant()
            .estop(reason)
            .await
            .map_err(session_err)?;
        Ok(())
    }

    /// Releases a latched e-stop (the embodiment's local clear).
    #[napi]
    pub async fn clear(&self) -> Result<()> {
        let robot = self.robot()?;
        robot.participant().clear().await.map_err(session_err)?;
        Ok(())
    }

    /// Tells the host the robot fell back on its own account (a device
    /// watchdog fired): the session goes SAFE. Never blocks.
    #[napi]
    pub fn report_safe(&self, reason: String) -> Result<()> {
        let robot = self.robot()?;
        let participant = robot.participant();
        participant.report_safe(participant.authority_view().epoch(), reason);
        Ok(())
    }

    /// The session mode this robot sees (`disarmed`, `armed`, `active`,
    /// `safe`, `estop`).
    #[napi(getter)]
    pub fn mode(&self) -> Result<String> {
        Ok(self
            .robot()?
            .participant()
            .authority()
            .map_or_else(|| "disarmed".into(), |v| v.mode.to_string().to_lowercase()))
    }

    /// Now on this robot's clock (`process.hrtime.bigint()`).
    #[napi]
    pub fn now_ns(&self) -> Result<BigInt> {
        Ok(BigInt::from(self.robot()?.participant().clock().now_ns()))
    }

    /// The counters of the edge serving `path`.
    #[napi]
    pub fn edge_stats(&self, path: String) -> Result<EdgeStats> {
        let stats = self.robot()?.edge_stats(&path).map_err(session_err)?;
        let count = |n: u64| seq(n);
        Ok(EdgeStats {
            accepted: count(stats.accepted),
            rejected: count(stats.rejected_total()),
            rejected_by: Rejection::ALL
                .into_iter()
                .filter(|r| stats.rejected(*r) > 0)
                .map(|r| RejectionCount {
                    reason: rejection_name(r),
                    count: count(stats.rejected(r)),
                })
                .collect(),
            clamped: count(stats.clamped),
            handovers: count(stats.handovers),
            safe_reports: count(stats.safe_reports),
            modes: [
                EdgeMode::Idle,
                EdgeMode::Driving,
                EdgeMode::Hold,
                EdgeMode::Safe,
                EdgeMode::Estop,
            ]
            .into_iter()
            .map(|mode| {
                let m = stats.mode(mode);
                ModeCount {
                    mode: mode_name(mode).to_string(),
                    ticks: count(m.ticks),
                    entries: count(m.entries),
                }
            })
            .collect(),
        })
    }

    /// Detaches: the edges and publishers go, the host drops the parts.
    #[napi]
    pub async fn close(&self) -> Result<()> {
        // The encoder finishes its queued frames and lets go of the robot.
        let encoder = self.encoder.lock().map(|mut e| e.take()).unwrap_or(None);
        if let Some(Encoder { frames, thread }) = encoder {
            drop(frames);
            let _ = tokio::task::spawn_blocking(move || thread.join()).await;
        }
        let taken = self.robot.lock().map(|mut r| r.take()).unwrap_or(None);
        let Some(robot) = taken else {
            return Ok(());
        };
        match Arc::try_unwrap(robot) {
            Ok(robot) => robot.detach().await.map_err(session_err),
            // A call is in flight: dropping it withdraws presence.
            Err(_) => Ok(()),
        }
    }
}

/// An input's age at the snapshot, in ns (`null`: blank, or no clock
/// mapping).
#[napi(object)]
pub struct InputAge {
    pub entry: String,
    pub age_ns: Option<BigInt>,
}

/// An input's window slots padded by `allow: warmup`.
#[napi(object)]
pub struct InputPadding {
    pub entry: String,
    pub slots: u32,
}

/// Per stream, the seq range a start used.
#[napi(object)]
pub struct StreamSeqs {
    pub stream: u32,
    pub seq_first: i64,
    pub seq_last: i64,
}

/// A policy start: each input as `predict` receives it, and its snapshot.
#[napi]
pub struct NativeObservation {
    values: Mutex<Vec<EntryValue>>,
    meta: Snapshot,
    time_ns: i64,
}

impl NativeObservation {
    fn new(observation: Observation, time_ns: i64) -> Self {
        Self {
            values: Mutex::new(
                observation
                    .values
                    .into_iter()
                    .map(|(entry, value)| value_out(entry, value))
                    .collect(),
            ),
            meta: observation.meta,
            time_ns,
        }
    }
}

#[napi]
impl NativeObservation {
    /// Takes the inputs by entry (the arrays move to JS: a second call
    /// returns none).
    #[napi]
    pub fn take_values(&self) -> Vec<EntryValue> {
        self.values
            .lock()
            .map(|mut v| std::mem::take(&mut *v))
            .unwrap_or_default()
    }

    /// The session time the snapshot was taken at.
    #[napi(getter)]
    pub fn time_ns(&self) -> BigInt {
        BigInt::from(self.time_ns)
    }

    /// Each input's conservative age.
    #[napi(getter)]
    pub fn ages(&self) -> Vec<InputAge> {
        self.meta
            .inputs
            .iter()
            .map(|i| InputAge {
                entry: i.entry.clone(),
                age_ns: i.age_ns.map(BigInt::from),
            })
            .collect()
    }

    /// Inputs delivered stale under `allow: hold`.
    #[napi(getter)]
    pub fn stale(&self) -> Vec<String> {
        self.meta
            .inputs
            .iter()
            .filter(|i| i.stale)
            .map(|i| i.entry.clone())
            .collect()
    }

    /// Window slots padded by `allow: warmup`.
    #[napi(getter)]
    pub fn padded(&self) -> Vec<InputPadding> {
        self.meta
            .inputs
            .iter()
            .filter(|i| i.padded > 0)
            .map(|i| InputPadding {
                entry: i.entry.clone(),
                slots: i.padded,
            })
            .collect()
    }

    /// Inputs bound to a constant (`allow: blank`).
    #[napi(getter)]
    pub fn blank(&self) -> Vec<String> {
        self.meta
            .inputs
            .iter()
            .filter(|i| i.blank)
            .map(|i| i.entry.clone())
            .collect()
    }

    /// The newest capture minus the oldest, across inputs.
    #[napi(getter)]
    pub fn skew_ns(&self) -> BigInt {
        BigInt::from(self.meta.skew_ns)
    }

    /// Per stream, the seq range used.
    #[napi(getter)]
    pub fn based_on(&self) -> Vec<StreamSeqs> {
        self.meta
            .based_on
            .iter()
            .map(|r| StreamSeqs {
                stream: r.channel.0,
                seq_first: seq(r.seq_first),
                seq_last: seq(r.seq_last),
            })
            .collect()
    }
}

/// The run-time options of a binding (§6.4).
#[napi(object)]
pub struct BindOptions {
    pub priority: Option<String>,
    pub under: Option<String>,
    pub at: Option<HashMap<String, String>>,
    pub positional: Option<Vec<String>>,
    pub allow: Option<Vec<String>>,
    pub captures: Option<HashMap<String, String>>,
    pub timeout_s: Option<f64>,
}

fn run_options(
    name: &str,
    robot: &str,
    opts: &BindOptions,
) -> std::result::Result<RunOptions, String> {
    let mut options = RunOptions::named(name).robot(robot)?;
    if let Some(under) = &opts.under {
        options = options.under(under)?;
    }
    // Sorted, as the Python binding's dicts are, so errors come in a stable
    // order.
    let sorted = |map: &Option<HashMap<String, String>>| {
        let mut pairs: Vec<(String, String)> =
            map.clone().unwrap_or_default().into_iter().collect();
        pairs.sort();
        pairs
    };
    for (entry, path) in sorted(&opts.at) {
        options = options.at(&entry, &path)?;
    }
    for entry in opts.positional.iter().flatten() {
        options = options.positional(entry);
    }
    for name in opts.allow.iter().flatten() {
        let conversion =
            Allow::from_name(name).ok_or_else(|| format!("allow: unknown conversion {name:?}"))?;
        options = options.allow(conversion);
    }
    for (capture, value) in sorted(&opts.captures) {
        options = options.capture(&capture, &value)?;
    }
    Ok(options)
}

/// A bound action: its entry, the part it drives (none when dropped) and its
/// horizon.
#[napi(object)]
pub struct ActionInfo {
    pub entry: String,
    pub part: Option<String>,
    pub horizon: u32,
}

/// An action entry's chunk seq, or none without the lease (or dropped).
#[napi(object)]
pub struct SentChunk {
    pub entry: String,
    pub seq: Option<i64>,
}

/// A lease the policy holds: the part and its token.
#[napi(object)]
pub struct HeldLease {
    pub path: String,
    pub token: BigInt,
}

/// Starts skipped by reason.
#[napi(object)]
pub struct SkipCount {
    pub reason: String,
    pub count: i64,
}

/// The start, skip and completion counters (§8.10), and the input the
/// starts are phased after.
#[napi(object)]
pub struct RunStats {
    /// The input whose publication the starts are phased just after.
    pub aligned_to: Option<String>,
    /// How far after it, in ms.
    pub phase_offset_ms: f64,
    pub starts: i64,
    pub completions: i64,
    pub skipped: Vec<SkipCount>,
    pub chunks_sent: i64,
    pub send_errors: i64,
    pub latency_p50_ms: Option<f64>,
    pub latency_p99_ms: Option<f64>,
    pub latency_max_ms: Option<f64>,
    pub drifting: bool,
}

/// An event the run loop reported.
#[napi(object)]
pub struct RunEventInfo {
    /// `rate_drift` or `inference_pending`.
    pub kind: String,
    pub message: String,
    pub measured_hz: Option<f64>,
    pub expected_hz: Option<f64>,
    pub cause: Option<String>,
    pub drifting: Option<bool>,
    pub pending_s: Option<f64>,
}

fn run_stats(controller: &Controller) -> RunStats {
    let stats = controller.stats();
    let ms = |d: Option<Duration>| d.map(|d| d.as_secs_f64() * 1e3);
    RunStats {
        aligned_to: controller.aligned_input().map(str::to_string),
        phase_offset_ms: controller.phase_offset().as_secs_f64() * 1e3,
        starts: seq(stats.starts),
        completions: seq(stats.completions),
        skipped: SkipReason::ALL
            .iter()
            .map(|reason| SkipCount {
                reason: reason.name().to_string(),
                count: seq(stats.skipped(*reason)),
            })
            .collect(),
        chunks_sent: seq(stats.chunks_sent),
        send_errors: seq(stats.send_errors),
        latency_p50_ms: ms(stats.latency_p50),
        latency_p99_ms: ms(stats.latency_p99),
        latency_max_ms: ms(stats.latency_max),
        drifting: stats.drifting,
    }
}

/// What JS answered for one start.
type Reply = std::result::Result<Vec<(String, Value)>, String>;

/// The start handed to JS: `predict` is called with it.
type PredictFn = ThreadsafeFunction<NativeObservation, (), NativeObservation, Status, false>;

/// `predict`, called from the run loop's thread: the observation goes to JS
/// and the thread waits for [`NativePolicy::reply`].
struct JsPredictor {
    predict: PredictFn,
    replies: mpsc::Receiver<Reply>,
    closing: Arc<AtomicBool>,
}

impl Predictor for JsPredictor {
    type Error = String;

    fn predict(
        &mut self,
        observation: Observation,
        snapshot_ns: i64,
    ) -> std::result::Result<Vec<(String, Value)>, String> {
        let start = NativeObservation::new(observation, snapshot_ns);
        let status = self
            .predict
            .call(start, ThreadsafeFunctionCallMode::NonBlocking);
        if status != Status::Ok {
            return Err(format!("predict could not be called: {status}"));
        }
        loop {
            match self.replies.recv_timeout(Duration::from_millis(20)) {
                Ok(reply) => return reply,
                Err(RecvTimeoutError::Timeout) if !self.closing.load(Ordering::Relaxed) => {}
                Err(_) => return Err("the policy closed while predict ran".into()),
            }
        }
    }
}

/// A policy bound to a robot: inputs, leases, chunks and the run loop.
#[napi]
pub struct NativePolicy {
    name: String,
    controller: Mutex<Option<Arc<Controller>>>,
    replies: Mutex<Option<SyncSender<Reply>>>,
    closing: Arc<AtomicBool>,
}

/// Attaches as `name` at `options.priority`, waits for `robot`'s parts,
/// binds `policy_json` (the policy map, §6) with the run-time options, and
/// declares the binding.
#[napi]
pub async fn bind_policy(
    address: String,
    name: String,
    policy_json: String,
    robot: String,
    options: BindOptions,
) -> Result<NativePolicy> {
    let address = self::address(&address)?;
    let wait = timeout(options.timeout_s.unwrap_or(10.0))?;
    let class = priority(options.priority.as_deref().unwrap_or("model"))?;
    let policy = Policy::parse(&policy_json)
        .map_err(|why| binding_refused(format!("binding refused: {name}: {why}")))?;
    let run_options = run_options(&name, &robot, &options)
        .map_err(|why| binding_refused(format!("binding refused: {why}")))?;
    let participant = address
        .join(&name, vec![Role::Controller], class, wait)
        .await
        .map_err(session_err)?;
    let config = ControllerConfig {
        bind_timeout: wait,
        ..ControllerConfig::default()
    };
    let participant = Arc::new(participant);
    let controller =
        match Controller::bind(participant.clone(), &policy, &run_options, config).await {
            Ok(controller) => controller,
            Err(err) => {
                // Refused: leave the session rather than linger.
                if let Ok(participant) = Arc::try_unwrap(participant) {
                    let _ = participant.detach().await;
                }
                return Err(controller_err(err));
            }
        };
    Ok(NativePolicy {
        name,
        controller: Mutex::new(Some(Arc::new(controller))),
        replies: Mutex::new(None),
        closing: Arc::new(AtomicBool::new(false)),
    })
}

impl NativePolicy {
    fn controller(&self) -> Result<Arc<Controller>> {
        self.controller
            .lock()
            .ok()
            .and_then(|c| c.clone())
            .ok_or_else(|| exception(format!("policy {} is closed", self.name)))
    }
}

#[napi]
impl NativePolicy {
    /// The policy's participant id.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// The binding table, as the cockpit shows it (§6.8).
    #[napi(getter)]
    pub fn table(&self) -> Result<String> {
        Ok(self.controller()?.binding().table())
    }

    /// The binding table as canonical JSON (what the fingerprint is over).
    #[napi(getter)]
    pub fn table_json(&self) -> Result<String> {
        Ok(self.controller()?.binding().table_text())
    }

    /// The binding fingerprint, 16 hex digits.
    #[napi(getter)]
    pub fn fingerprint(&self) -> Result<String> {
        Ok(self.controller()?.binding().fingerprint.to_string())
    }

    /// Warnings the binder accepted the binding with.
    #[napi(getter)]
    pub fn warnings(&self) -> Result<Vec<String>> {
        Ok(self.controller()?.binding().warnings.clone())
    }

    /// The replan rate R.
    #[napi(getter)]
    pub fn replan_hz(&self) -> Result<f64> {
        Ok(self.controller()?.binding().replan_hz)
    }

    /// Each bound action.
    #[napi]
    pub fn actions(&self) -> Result<Vec<ActionInfo>> {
        Ok(self
            .controller()?
            .binding()
            .actions
            .iter()
            .map(|a| ActionInfo {
                entry: a.entry.clone(),
                part: a.part.as_ref().map(|p| p.path.to_string()),
                horizon: a.horizon,
            })
            .collect())
    }

    /// A policy start now. Throws `StartSkipped` when an input is missing,
    /// stale (without `allow: hold`) or has a window gap.
    #[napi]
    pub fn observe(&self) -> Result<NativeObservation> {
        let (observation, at) = self.controller()?.observe().map_err(observe_err)?;
        Ok(NativeObservation::new(observation, at))
    }

    /// Sends one chunk per action entry, computed from `observation`, each
    /// `[horizon, ...]`.
    #[napi]
    pub fn send(
        &self,
        values: Vec<EntryValue>,
        observation: &NativeObservation,
    ) -> Result<Vec<SentChunk>> {
        let controller = self.controller()?;
        let values = values_in(&values)?;
        let sent = controller
            .send(&values, &observation.meta.based_on, observation.time_ns)
            .map_err(send_err)?;
        Ok(sent
            .into_iter()
            .map(|(entry, outcome)| SentChunk {
                entry,
                seq: match outcome {
                    Sent::Seq(s) => Some(seq(s)),
                    Sent::NoLease | Sent::Dropped => None,
                },
            })
            .collect())
    }

    /// Takes the binding's leases now, preempting a lower class; from then
    /// on they are kept.
    #[napi]
    pub async fn acquire(&self) -> Result<Vec<HeldLease>> {
        let controller = self.controller()?;
        let leases = controller.acquire().await.map_err(session_err)?;
        Ok(leases
            .iter()
            .map(|l| HeldLease {
                path: controller
                    .participant()
                    .part_by_id(l.channel)
                    .map_or_else(|_| l.channel.to_string(), |p| p.path.to_string()),
                token: BigInt::from(l.token.0),
            })
            .collect())
    }

    /// Gives the leases back; they are not re-requested until `acquire` or
    /// `run`.
    #[napi]
    pub async fn release(&self) -> Result<()> {
        let controller = self.controller()?;
        controller.release().await.map_err(session_err)
    }

    /// Whether this policy holds every lease its binding commands.
    #[napi]
    pub fn holds(&self) -> Result<bool> {
        Ok(self.controller()?.holds())
    }

    /// Runs the policy loop (§8.9) on a thread of its own until `stop()`,
    /// `duration_s`, or a failure, calling `predict(observation)` on the JS
    /// thread at the replan rate with one inference in flight; each call is
    /// answered with `reply`. If a reply is an error, the leases are
    /// released and the run rejects with code `Predict`. Resolves to the
    /// run's counters (taken as it returns, so a `close` racing the
    /// resolution cannot lose them).
    #[napi]
    pub async fn run(&self, predict: PredictFn, duration_s: Option<f64>) -> Result<RunStats> {
        let controller = self.controller()?;
        let duration = duration_s.map(timeout).transpose()?;
        let (tx, rx) = mpsc::sync_channel(1);
        {
            let mut slot = self.replies.lock().map_err(|_| exception("poisoned"))?;
            if slot.is_some() {
                return Err(exception(format!(
                    "policy {} is already running",
                    self.name
                )));
            }
            *slot = Some(tx);
        }
        let mut predictor = JsPredictor {
            predict,
            replies: rx,
            closing: self.closing.clone(),
        };
        let result = tokio::task::spawn_blocking(move || {
            controller
                .run(&mut predictor, duration)
                .map(|()| run_stats(&controller))
        })
        .await;
        if let Ok(mut slot) = self.replies.lock() {
            *slot = None;
        }
        match result {
            Ok(Ok(stats)) => Ok(stats),
            Ok(Err(RunError::Predict(message))) => Err(error("Predict", &[], message)),
            Ok(Err(RunError::Send(err))) => Err(send_err(err)),
            Err(join) => Err(exception(format!("the run loop failed: {join}"))),
        }
    }

    /// Answers the start `run` handed to `predict`: the actions, or (with
    /// `error`) why `predict` failed.
    #[napi]
    pub fn reply(&self, values: Option<Vec<EntryValue>>, error: Option<String>) -> Result<()> {
        let reply = match (values, error) {
            (_, Some(message)) => Err(message),
            (Some(values), None) => values_in(&values).map_err(|e| message_of(&e)),
            (None, None) => Err("predict returned nothing".into()),
        };
        let sender = self
            .replies
            .lock()
            .ok()
            .and_then(|s| s.clone())
            .ok_or_else(|| exception(format!("policy {} is not running", self.name)))?;
        sender
            .try_send(reply)
            .map_err(|_| exception("a reply with no start waiting for it"))
    }

    /// Asks a `run` in progress to return.
    #[napi]
    pub fn stop(&self) -> Result<()> {
        self.controller()?.stop();
        Ok(())
    }

    /// The start, skip and completion counters (§8.10).
    #[napi]
    pub fn stats(&self) -> Result<RunStats> {
        let controller = self.controller()?;
        Ok(run_stats(&controller))
    }

    /// Takes the events reported so far, oldest first.
    #[napi]
    pub fn events(&self) -> Result<Vec<RunEventInfo>> {
        Ok(self
            .controller()?
            .drain_events()
            .into_iter()
            .map(|event| {
                let mut info = RunEventInfo {
                    kind: String::new(),
                    message: event.to_string(),
                    measured_hz: None,
                    expected_hz: None,
                    cause: None,
                    drifting: None,
                    pending_s: None,
                };
                match event {
                    RunEvent::RateDrift {
                        measured_hz,
                        expected_hz,
                        cause,
                        drifting,
                    } => {
                        info.kind = "rate_drift".into();
                        info.measured_hz = Some(measured_hz);
                        info.expected_hz = Some(expected_hz);
                        info.cause = Some(cause.to_string());
                        info.drifting = Some(drifting);
                    }
                    RunEvent::InferencePending { pending } => {
                        info.kind = "inference_pending".into();
                        info.pending_s = Some(pending.as_secs_f64());
                    }
                }
                info
            })
            .collect())
    }

    /// Gives the leases back and detaches.
    #[napi]
    pub async fn close(&self) -> Result<()> {
        let taken = self.controller.lock().map(|mut c| c.take()).unwrap_or(None);
        let Some(mut controller) = taken else {
            return Ok(());
        };
        controller.stop();
        self.closing.store(true, Ordering::Relaxed);
        // A `run` returns within one idle slice.
        for _ in 0..50 {
            match Arc::try_unwrap(controller) {
                Ok(owned) => {
                    let participant = owned.close().await;
                    return match Arc::try_unwrap(participant) {
                        Ok(participant) => participant.detach().await.map_err(session_err),
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
}

/// A fixed-rate loop timer on `process.hrtime.bigint()`: `sleep()` resolves
/// when the next period starts (a coarse sleep, then a short spin, off the
/// JS thread). A tick more than a period late skips the missed ones rather
/// than bursting.
#[napi]
pub struct NativeRate {
    period_ns: i64,
    next_ns: Arc<AtomicU64>,
    missed: Arc<AtomicU64>,
}

#[napi]
impl NativeRate {
    #[napi(constructor)]
    pub fn new(hz: f64) -> Result<Self> {
        if !(hz.is_finite() && hz > 0.0) {
            return Err(invalid(format!("rate {hz} Hz is not positive")));
        }
        #[allow(clippy::cast_possible_truncation)]
        let period_ns = ((1e9 / hz).round() as i64).max(1);
        Ok(Self {
            period_ns,
            next_ns: Arc::new(AtomicU64::new(0)),
            missed: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Waits until the next period starts; resolves to how late this call
    /// was for it, in ns.
    #[napi]
    pub async fn sleep(&self) -> Result<i64> {
        let clock = SystemMonotonicClock;
        let now = clock.now_ns();
        let mut next = i64::try_from(self.next_ns.load(Ordering::Relaxed)).unwrap_or(0);
        if next == 0 {
            next = now;
        }
        if now > next + self.period_ns {
            let behind = (now - next) / self.period_ns;
            self.missed
                .fetch_add(behind.unsigned_abs(), Ordering::Relaxed);
            next += behind * self.period_ns;
        }
        let late = (now - next).max(0);
        self.next_ns.store(
            u64::try_from(next + self.period_ns).unwrap_or(0),
            Ordering::Relaxed,
        );
        tokio::task::spawn_blocking(move || sleep_until(|| clock.now_ns(), next))
            .await
            .map_err(|e| exception(e.to_string()))?;
        Ok(late)
    }

    /// Periods skipped because a call came more than a period late.
    #[napi(getter)]
    pub fn missed(&self) -> i64 {
        seq(self.missed.load(Ordering::Relaxed))
    }
}

fn record_err(err: RecordError) -> Error {
    match err {
        RecordError::Session(err) => session_err(err),
        other => exception(other.to_string()),
    }
}

/// A finished recording's counters.
#[napi(object)]
pub struct RecordStats {
    pub path: String,
    pub messages: i64,
    pub channels: u32,
    pub bytes: i64,
    pub log_time_clamped: i64,
}

/// An MCAP recording of a session (§13), as an observer participant.
#[napi]
pub struct NativeRecorder {
    path: String,
    recorder: Mutex<Option<Recorder>>,
}

/// Attaches as observer `name` and starts recording every stream, chunk and
/// authority change of the session into a new MCAP file at `path`.
#[napi]
pub async fn start_recorder(
    address: String,
    path: String,
    name: String,
    timeout_s: f64,
) -> Result<NativeRecorder> {
    let address = self::address(&address)?;
    let wait = timeout(timeout_s)?;
    let participant = address
        .join(&name, vec![Role::Observer], PriorityClass::Scripted, wait)
        .await
        .map_err(session_err)?;
    let recorder = Recorder::start(Arc::new(participant), &path)
        .await
        .map_err(record_err)?;
    Ok(NativeRecorder {
        path,
        recorder: Mutex::new(Some(recorder)),
    })
}

#[napi]
impl NativeRecorder {
    /// The MCAP file.
    #[napi(getter)]
    pub fn path(&self) -> String {
        self.path.clone()
    }

    /// Stops recording and finishes the file; none when already closed.
    #[napi]
    pub async fn close(&self) -> Result<Option<RecordStats>> {
        let taken = self.recorder.lock().map(|mut r| r.take()).unwrap_or(None);
        let Some(recorder) = taken else {
            return Ok(None);
        };
        let participant = recorder.participant().clone();
        let stats = recorder.finish().await.map_err(record_err)?;
        if let Ok(participant) = Arc::try_unwrap(participant) {
            participant.detach().await.map_err(session_err)?;
        }
        Ok(Some(RecordStats {
            path: stats.path.to_string_lossy().into_owned(),
            messages: seq(stats.messages),
            channels: stats.channels,
            bytes: seq(stats.bytes),
            log_time_clamped: seq(stats.log_time_clamped),
        }))
    }
}

/// One topic of a recording.
#[napi(object)]
pub struct TopicSummary {
    pub topic: String,
    pub schema: String,
    pub channels: u32,
    pub messages: i64,
    pub first_log_ns: BigInt,
    pub last_log_ns: BigInt,
    pub log_time_monotonic: bool,
    pub publish_time_monotonic: bool,
    pub undecodable: i64,
}

/// One lease change of a recording.
#[napi(object)]
pub struct LeaseChange {
    pub log_ns: BigInt,
    pub part: String,
    pub holder: Option<String>,
    pub priority: Option<String>,
    pub token: Option<BigInt>,
}

/// A recording read back.
#[napi(object)]
pub struct RecordingSummary {
    pub messages: i64,
    pub metadata: Vec<String>,
    pub topics: Vec<TopicSummary>,
    pub leases: Vec<LeaseChange>,
}

/// Reads a recording back (off the JS thread): messages per topic, time
/// ordering, the lease timeline.
#[napi]
pub async fn read_recording(path: String) -> Result<RecordingSummary> {
    let summary = tokio::task::spawn_blocking(move || rlmesh_record::summarize(&path))
        .await
        .map_err(|e| exception(e.to_string()))?
        .map_err(record_err)?;
    Ok(RecordingSummary {
        messages: seq(summary.messages),
        metadata: summary.metadata,
        topics: summary
            .topics
            .into_iter()
            .map(|t| TopicSummary {
                topic: t.topic,
                schema: t.schema,
                channels: t.channels,
                messages: seq(t.messages),
                first_log_ns: BigInt::from(t.first_log_ns),
                last_log_ns: BigInt::from(t.last_log_ns),
                log_time_monotonic: t.log_time_monotonic,
                publish_time_monotonic: t.publish_time_monotonic,
                undecodable: seq(t.undecodable),
            })
            .collect(),
        leases: summary
            .leases
            .into_iter()
            .map(|l| LeaseChange {
                log_ns: BigInt::from(l.log_ns),
                part: l.part,
                holder: l.holder,
                priority: l.priority,
                token: l.token.map(BigInt::from),
            })
            .collect(),
    })
}
