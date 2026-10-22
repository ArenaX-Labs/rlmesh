//! PyO3 bindings for real-time sessions (protocol v2, the v0.2 spec): an
//! embedded session host, a robot ([`rlmesh_session::edge::Robot`]), a bound
//! policy ([`rlmesh_session::Controller`]) and an MCAP recorder
//! ([`rlmesh_session::record::Recorder`]). The `rlmesh.v2` package wraps
//! them; every rule (packing, binding, the edge, leases, the run loop)
//! stays in the Rust crates.
//!
//! The GIL is released around every blocking wait and every network call.
//! Edge ticks and transport callbacks never take it: the robot's own loop
//! ticks its edge (`due`), and the run loop attaches only to call
//! `predict` (and, between starts, to check for signals).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pyo3::buffer::PyBuffer;
use pyo3::create_exception;
use pyo3::exceptions::{PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use rlmesh_session::address::Address;
use rlmesh_session::controller::sleep_until;
use rlmesh_session::edge::{Due, Rejection, Robot};
use rlmesh_session::record::{RecordError, Recorder};
use rlmesh_session::sdk::{self, BindOptions, BindPolicyError, EmbeddedHost, ErrorKind};
use rlmesh_session::spec::BindError;
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::wire::{PartSpec, PriorityClass};
use rlmesh_session::{
    Clock, Controller, ControllerError, Observation, ObserveError, Predictor, RunError, RunEvent,
    SendError, Sent, SessionError, SkipReason, Snapshot, SystemMonotonicClock,
};

use crate::client::{check_signals, signal_poll};
use crate::spaces::wrap_native_tensor;
use crate::types::RLMeshException;

create_exception!(
    _rlmesh,
    SessionException,
    RLMeshException,
    "A real-time session (protocol v2) failed."
);

create_exception!(
    _rlmesh,
    SessionRefused,
    SessionException,
    "The session host refused a verb; `status` names why (`LEASE_BUSY`, `NOT_READY`, ...)."
);

create_exception!(
    _rlmesh,
    BindingRefused,
    SessionException,
    "The binder refused a policy: the message is the refuse-and-explain text (§6.7); \
     `problems` lists `(code, \"input\" | \"action\", entry)` per problem."
);

create_exception!(
    _rlmesh,
    StartSkipped,
    SessionException,
    "A policy start was skipped (§8.8): `reason` is `waiting_for`, `stale_input` or \
     `window_gap`; `waiting`, `stale` and `gaps` name the entries."
);

/// Registers the exception types.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("SessionException", py.get_type::<SessionException>())?;
    m.add("SessionRefused", py.get_type::<SessionRefused>())?;
    m.add("BindingRefused", py.get_type::<BindingRefused>())?;
    m.add("StartSkipped", py.get_type::<StartSkipped>())?;
    m.add_class::<PyHost>()?;
    m.add_class::<PyRobot>()?;
    m.add_class::<PyDue>()?;
    m.add_class::<PyController>()?;
    m.add_class::<PyObservation>()?;
    m.add_class::<PyRate>()?;
    m.add_class::<PyRecorder>()?;
    m.add_function(wrap_pyfunction!(parse_address, m)?)?;
    m.add_function(wrap_pyfunction!(is_v2_address, m)?)?;
    m.add_function(wrap_pyfunction!(local_run_session, m)?)?;
    crate::lockstep::register(m)?;
    Ok(())
}

/// Parses an address: a participant address,
/// `[rlmesh://][host/][session/]participant`, or with `session=True` a
/// session address, `[rlmesh://][host/][session]`, as
/// `{"host", "port", "session", "participant"}` (`None` where it names
/// none). A malformed or old form (`lockstep://`, a tenant segment) is the
/// ValueError class, an upgrade's naming the new form.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def parse_address(text: str, session: bool = False) -> dict[str, typing.Any]: ...
"#
    )
)]
#[pyfunction]
#[pyo3(signature = (text, session=false))]
fn parse_address<'py>(py: Python<'py>, text: &str, session: bool) -> PyResult<Bound<'py, PyDict>> {
    let parsed = if session {
        Address::session(text)
    } else {
        Address::participant(text)
    }
    .map_err(address_err)?;
    let d = PyDict::new(py);
    d.set_item("host", parsed.host.as_ref().map(|h| h.name.clone()))?;
    d.set_item("port", parsed.host.as_ref().and_then(|h| h.port))?;
    d.set_item("session", parsed.session)?;
    d.set_item("participant", parsed.participant)?;
    Ok(d)
}

/// Whether `text` is a v2 session or participant address rather than a v1
/// endpoint (`host:port`, a port, `tcp://`, `unix://`, `https://`).
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def is_v2_address(text: str) -> bool: ...
"#
    )
)]
#[pyfunction]
fn is_v2_address(text: &str) -> bool {
    rlmesh_session::address::is_v2_address(text)
}

/// A fresh session on this process's shared in-process host, for a run's
/// local env.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def local_run_session() -> str: ...
"#
    )
)]
#[pyfunction]
fn local_run_session() -> PyResult<String> {
    rlmesh::lockstep::local_run_session().map_err(crate::lockstep::gym_err)
}

/// An address refused as the ValueError class, without the error's kind.
fn address_err(err: SessionError) -> PyErr {
    match err {
        SessionError::Config(why) => PyValueError::new_err(why),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// The runtime every session object of this process runs on.
pub(crate) fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .thread_name("rlmesh-session")
            .enable_all()
            .build()
            .expect("failed to build the rlmesh session runtime")
    })
}

/// Runs `work` on the session runtime with the GIL released, polling
/// Python signals so Ctrl-C interrupts a wait.
pub(crate) fn block_on<T>(
    py: Python<'_>,
    work: impl Future<Output = PyResult<T>> + Send,
) -> PyResult<T>
where
    T: Send,
{
    py.detach(|| {
        runtime().block_on(async move {
            let mut work = std::pin::pin!(work);
            let mut poll = signal_poll();
            loop {
                tokio::select! {
                    result = &mut work => return result,
                    _ = poll.tick() => check_signals()?,
                }
            }
        })
    })
}

pub(crate) fn session_err(err: SessionError) -> PyErr {
    match ErrorKind::of(&err) {
        ErrorKind::Refused(status) => Python::attach(|py| {
            let message = match &err {
                SessionError::Rejected(refusal) => refusal.message.clone(),
                other => other.to_string(),
            };
            let err = SessionRefused::new_err(message);
            if let Err(set) = err.value(py).setattr("status", status) {
                return set;
            }
            err
        }),
        ErrorKind::Timeout => PyTimeoutError::new_err(err.to_string()),
        ErrorKind::Value => PyValueError::new_err(err.to_string()),
        ErrorKind::Session => SessionException::new_err(err.to_string()),
    }
}

pub(crate) fn bind_err(err: BindError) -> PyErr {
    Python::attach(|py| {
        let problems = sdk::bind_problems(&err);
        let err_obj = BindingRefused::new_err(err.to_string());
        if let Err(set) = err_obj.value(py).setattr("problems", problems) {
            return set;
        }
        err_obj
    })
}

fn controller_err(err: ControllerError) -> PyErr {
    match err {
        ControllerError::Bind(err) => bind_err(err),
        ControllerError::Session(err) => session_err(err),
    }
}

fn bind_policy_err(err: BindPolicyError) -> PyErr {
    match err {
        BindPolicyError::Policy(_) => BindingRefused::new_err(err.to_string()),
        BindPolicyError::Controller(err) => controller_err(err),
    }
}

fn send_err(err: SendError) -> PyErr {
    match err {
        SendError::Value(message) => PyValueError::new_err(message),
        SendError::Session(err) => session_err(err),
    }
}

fn observe_err(err: ObserveError) -> PyErr {
    match err {
        ObserveError::Skipped(skip) => Python::attach(|py| {
            let reason = skip.reason();
            let err = StartSkipped::new_err(format!(
                "start skipped ({reason}): waiting for {:?}, stale {:?}, window gaps {:?}",
                skip.waiting, skip.stale, skip.gaps
            ));
            let value = err.value(py);
            let set = value
                .setattr("reason", reason.name())
                .and_then(|()| value.setattr("waiting", skip.waiting.clone()))
                .and_then(|()| value.setattr("stale", skip.stale.clone()))
                .and_then(|()| value.setattr("gaps", skip.gaps.clone()));
            match set {
                Ok(()) => err,
                Err(set) => set,
            }
        }),
        other => SessionException::new_err(other.to_string()),
    }
}

/// The session a session address (`[host/][session]`) names, placed by the
/// hosting rule: joined, or hosted in this process when no host answers.
/// Joining it puts capture times on `CLOCK_MONOTONIC`,
/// `time.monotonic_ns()`'s clock.
fn place(py: Python<'_>, text: &str, lease_grace: Duration) -> PyResult<EmbeddedHost> {
    let address = Address::session(text).map_err(address_err)?;
    block_on(py, async move {
        EmbeddedHost::start(&address, lease_grace)
            .await
            .map_err(session_err)
    })
}

use rlmesh_session::address::LEASE_GRACE;

pub(crate) fn timeout(seconds: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(seconds)
        .map_err(|_| PyValueError::new_err(format!("timeout {seconds} is not a duration")))
}

fn priority(name: &str) -> PyResult<PriorityClass> {
    sdk::priority(name).map_err(PyValueError::new_err)
}

/// A session host in this process, on local per-class TCP ports.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "V2Host", frozen)]
pub struct PyHost {
    host: EmbeddedHost,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyHost {
    /// Places the session `address` names (`":0"`: hosted here on an
    /// OS-assigned port).
    #[new]
    #[pyo3(signature = (address=":0", lease_grace_ms=250))]
    fn new(py: Python<'_>, address: &str, lease_grace_ms: u64) -> PyResult<Self> {
        let host = place(py, address, Duration::from_millis(lease_grace_ms))?;
        Ok(Self { host })
    }

    /// `{ip}:{port}[/{session}]`: what participants connect to.
    #[getter]
    fn address(&self) -> String {
        self.host.address().to_string()
    }

    /// Stops the host (participants see it leave).
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        block_on(py, async { self.host.close().await.map_err(session_err) })
    }
}

/// One field of a part, for the Python layer: `(name, dtype, shape)`.
type FieldInfo = (String, String, Vec<u32>);

fn fields_info(fields: &[rlmesh_session::wire::FieldSpec]) -> Vec<FieldInfo> {
    fields
        .iter()
        .map(|f| (f.name.clone(), f.dtype.name().to_string(), f.shape.clone()))
        .collect()
}

pub(crate) fn part_dict<'py>(py: Python<'py>, part: &PartSpec) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("path", part.path.as_str())?;
    d.set_item("kind", &part.kind)?;
    d.set_item("hz", part.hz)?;
    d.set_item("event", part.event)?;
    d.set_item("encoding", part.encoding.wire_name())?;
    d.set_item("fields", fields_info(&part.fields))?;
    match &part.command {
        Some(command) => {
            let c = PyDict::new(py);
            c.set_item("mode", &command.mode)?;
            c.set_item("fields", fields_info(&command.fields))?;
            c.set_item("group", command.group.as_deref())?;
            c.set_item("safe", &command.safe_behavior)?;
            c.set_item("control_hz", part.control_hz)?;
            d.set_item("command", c)?;
        }
        None => d.set_item("command", py.None())?,
    }
    Ok(d)
}

/// An `rlmesh_session::spec` tensor as a native `Tensor` (zero-copy when its buffer
/// is 8-byte aligned, §8.11; one copy otherwise).
pub(crate) fn tensor_to_py<'py>(py: Python<'py>, tensor: Tensor) -> PyResult<Bound<'py, PyAny>> {
    let dtype = rlmesh_spaces::DType::from_name(tensor.dtype.name()).ok_or_else(|| {
        SessionException::new_err(format!("no tensor dtype for {}", tensor.dtype.name()))
    })?;
    let shape: Vec<i64> = tensor.shape.iter().map(|d| i64::from(*d)).collect();
    let native = if tensor.data.as_ptr().addr().is_multiple_of(8) {
        rlmesh_spaces::Tensor::from_vec(tensor.data, shape, dtype)
    } else {
        rlmesh_spaces::Tensor::from_slice(&tensor.data, &shape, dtype)
    }
    .map_err(|e| SessionException::new_err(e.to_string()))?;
    wrap_native_tensor(py, native)
}

pub(crate) fn value_to_py<'py>(py: Python<'py>, value: Value) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Value::Tensor(tensor) => tensor_to_py(py, tensor),
        Value::Dict(fields) => {
            let d = PyDict::new(py);
            for (name, tensor) in fields {
                d.set_item(name, tensor_to_py(py, tensor)?)?;
            }
            Ok(d.into_any())
        }
        Value::Text(text) => Ok(text.into_pyobject(py)?.into_any()),
    }
}

/// One array from Python: `(dtype, shape, bytes-like)`.
pub(crate) fn tensor_from_py(entry: &str, obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let (dtype, shape, data): (String, Vec<u32>, PyBuffer<u8>) = obj.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{entry}: expected an array, got {}",
            obj.get_type()
                .name()
                .map_or_else(|_| "?".into(), |n| n.to_string())
        ))
    })?;
    let data = data.to_vec(obj.py())?;
    sdk::tensor(entry, &dtype, shape, data).map_err(PyValueError::new_err)
}

/// What `predict` returned, by action entry: an array, or a dict of arrays.
pub(crate) fn values_from_py(output: &Bound<'_, PyAny>) -> PyResult<Vec<(String, Value)>> {
    let dict = output
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("predict must return a dict keyed by action name"))?;
    let mut values = Vec::with_capacity(dict.len());
    for (key, item) in dict.iter() {
        let entry: String = key.extract()?;
        let value = match item.cast::<PyDict>() {
            Ok(fields) => Value::Dict(
                fields
                    .iter()
                    .map(|(name, array)| {
                        let name: String = name.extract()?;
                        let tensor = tensor_from_py(&format!("{entry}.{name}"), &array)?;
                        Ok((name, tensor))
                    })
                    .collect::<PyResult<_>>()?,
            ),
            Err(_) => Value::Tensor(tensor_from_py(&entry, &item)?),
        };
        values.push((entry, value));
    }
    Ok(values)
}

/// A robot: its map declared, a publisher per part and an edge per command
/// group.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "V2Robot", frozen)]
pub struct PyRobot {
    name: String,
    robot: Mutex<Option<Arc<Robot>>>,
    session: EmbeddedHost,
}

impl PyRobot {
    fn robot(&self) -> PyResult<Arc<Robot>> {
        self.robot
            .lock()
            .ok()
            .and_then(|r| r.clone())
            .ok_or_else(|| SessionException::new_err(format!("robot {} is closed", self.name)))
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PyRobot {
    /// Attaches as `name` (the robot's participant id), declares
    /// `robot_json` (the authored map, §3) and starts its publishers and
    /// edges.
    #[new]
    #[pyo3(signature = (address, name, robot_json, timeout_s=10.0))]
    fn new(
        py: Python<'_>,
        address: &str,
        name: &str,
        robot_json: String,
        timeout_s: f64,
    ) -> PyResult<Self> {
        let session = place(py, address, LEASE_GRACE)?;
        let address = session.address().clone();
        let wait = timeout(timeout_s)?;
        let robot = block_on(py, async move {
            sdk::connect_robot(&address, name, &robot_json, wait)
                .await
                .map_err(|err| match err {
                    rlmesh_session::edge::EdgeError::Session(err) => session_err(err),
                    other => SessionException::new_err(other.to_string()),
                })
        })?;
        Ok(Self {
            name: name.to_string(),
            robot: Mutex::new(Some(Arc::new(robot))),
            session,
        })
    }

    /// The robot's participant id.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// The robot's participant address, `{ip}:{port}[/{session}]/{name}`.
    #[getter]
    fn address(&self) -> String {
        self.session.address().participant(&self.name)
    }

    /// The parts as the host normalised them, in authored order.
    #[gen_stub(override_return_type(type_repr = "builtins.list[builtins.dict[builtins.str, typing.Any]]", imports = ("builtins", "typing")))]
    fn parts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let robot = self.robot()?;
        let parts = robot
            .parts()
            .iter()
            .map(|p| part_dict(py, p))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, parts)
    }

    /// Publishes one sample of `path` from its fields in declaration order
    /// (C-contiguous bytes-likes; a `str` for text), captured at
    /// `capture_ns` (`time.monotonic_ns()`; now by default). The values
    /// are copied before this returns. Returns the sample's seq, or `None`
    /// when coalesced (or handed to an image encoder).
    #[pyo3(signature = (path, fields, capture_ns=None))]
    fn publish(
        &self,
        py: Python<'_>,
        path: &str,
        #[gen_stub(override_type(
            type_repr = "collections.abc.Sequence[builtins.str | builtins.bytes | memoryview]",
            imports = ("collections.abc", "builtins")
        ))]
        fields: Vec<Bound<'_, PyAny>>,
        capture_ns: Option<i64>,
    ) -> PyResult<Option<u64>> {
        let robot = self.robot()?;
        let mut owned: Vec<Vec<u8>> = Vec::with_capacity(fields.len());
        for field in &fields {
            if let Ok(text) = field.extract::<String>() {
                owned.push(text.into_bytes());
            } else {
                let buffer = PyBuffer::<u8>::get(field).map_err(|_| {
                    PyValueError::new_err(format!(
                        "{path}: a field must be a C-contiguous bytes-like or a str"
                    ))
                })?;
                owned.push(buffer.to_vec(py)?);
            }
        }
        let capture = capture_ns.unwrap_or_else(|| robot.participant().clock().now_ns());
        py.detach(move || sdk::publish(&robot, path, capture, owned))
            .map_err(session_err)
    }

    /// One control tick of the edge serving `path` and its command group:
    /// what to dispatch now, or `None`. Call it once per tick per group.
    fn due(&self, py: Python<'_>, path: &str) -> PyResult<Option<PyDue>> {
        let robot = self.robot()?;
        let due = py.detach(|| robot.due(path)).map_err(session_err)?;
        due.map(|due| PyDue::new(py, &robot, due)).transpose()
    }

    /// The robot switched control modes after a handover of `path`'s
    /// group: the group may dispatch again.
    fn handover_done(&self, path: &str) -> PyResult<()> {
        self.robot()?.handover_done(path).map_err(session_err)
    }

    /// DISARMED or SAFE to ARMED, in a fresh epoch.
    fn arm(&self, py: Python<'_>) -> PyResult<()> {
        let robot = self.robot()?;
        block_on(py, async move {
            robot
                .participant()
                .arm()
                .await
                .map(|_| ())
                .map_err(session_err)
        })
    }

    /// Back to DISARMED.
    fn disarm(&self, py: Python<'_>) -> PyResult<()> {
        let robot = self.robot()?;
        block_on(py, async move {
            robot
                .participant()
                .disarm()
                .await
                .map(|_| ())
                .map_err(session_err)
        })
    }

    /// Emergency stop: latched until a local `clear`, then disarm and arm.
    fn estop(&self, py: Python<'_>, reason: String) -> PyResult<()> {
        let robot = self.robot()?;
        block_on(py, async move {
            robot
                .participant()
                .estop(reason)
                .await
                .map(|_| ())
                .map_err(session_err)
        })
    }

    /// Releases a latched e-stop (the embodiment's local clear).
    fn clear(&self, py: Python<'_>) -> PyResult<()> {
        let robot = self.robot()?;
        block_on(py, async move {
            robot
                .participant()
                .clear()
                .await
                .map(|_| ())
                .map_err(session_err)
        })
    }

    /// Tells the host the robot fell back on its own account (a device
    /// watchdog fired): the session goes SAFE. Never blocks.
    fn report_safe(&self, reason: String) -> PyResult<()> {
        let robot = self.robot()?;
        let participant = robot.participant();
        participant.report_safe(participant.authority_view().epoch(), reason);
        Ok(())
    }

    /// The session mode this robot sees (`disarmed`, `armed`, `active`,
    /// `safe`, `estop`).
    #[getter]
    fn mode(&self) -> PyResult<String> {
        Ok(self
            .robot()?
            .participant()
            .authority()
            .map_or_else(|| "disarmed".into(), |v| v.mode.to_string().to_lowercase()))
    }

    /// Now on this robot's clock (`time.monotonic_ns()`).
    fn now_ns(&self) -> PyResult<i64> {
        Ok(self.robot()?.participant().clock().now_ns())
    }

    /// The counters of the edge serving `path`.
    #[gen_stub(override_return_type(type_repr = "builtins.dict[builtins.str, typing.Any]", imports = ("builtins", "typing")))]
    fn edge_stats<'py>(&self, py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.robot()?.edge_stats(path).map_err(session_err)?;
        let d = PyDict::new(py);
        d.set_item("accepted", stats.accepted)?;
        d.set_item("rejected", stats.rejected_total())?;
        let rejected_by = PyDict::new(py);
        for reason in Rejection::ALL {
            let n = stats.rejected(reason);
            if n > 0 {
                rejected_by.set_item(sdk::rejection_name(reason), n)?;
            }
        }
        d.set_item("rejected_by", rejected_by)?;
        d.set_item("clamped", stats.clamped)?;
        d.set_item("handovers", stats.handovers)?;
        d.set_item("safe_reports", stats.safe_reports)?;
        let modes = PyDict::new(py);
        for mode in sdk::EDGE_MODES {
            let m = stats.mode(mode);
            modes.set_item(sdk::mode_name(mode), (m.ticks, m.entries))?;
        }
        d.set_item("modes", modes)?;
        Ok(d)
    }

    /// Detaches: the edges and publishers go, the host drops the parts.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let taken = self.robot.lock().map(|mut r| r.take()).unwrap_or(None);
        block_on(py, async move {
            if let Some(robot) = taken {
                sdk::close_robot(robot).await.map_err(session_err)?;
            }
            self.session.close().await.map_err(session_err)
        })
    }
}

/// What the robot does this control tick.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "V2Due", frozen)]
pub struct PyDue {
    /// The part the tick concerns.
    #[pyo3(get)]
    path: String,
    /// `driving`, `hold`, `safe`, `estop` or `idle`.
    #[pyo3(get)]
    mode: &'static str,
    fields: Option<Py<PyDict>>,
    /// The declared safe behaviour to run now, or `None`.
    #[pyo3(get)]
    safe: Option<String>,
    /// `(group, from_path, to_path)` on the tick a handover starts.
    #[pyo3(get)]
    handover: Option<(String, String, String)>,
    /// The lease token of the chunk the command came from.
    #[pyo3(get)]
    lease_token: Option<u64>,
    /// That chunk's seq under its token.
    #[pyo3(get)]
    chunk_seq: Option<u64>,
    /// The command's index in that chunk.
    #[pyo3(get)]
    index: Option<u32>,
    /// When the edge received the chunk (this robot's clock).
    #[pyo3(get)]
    received_ns: Option<i64>,
    /// When this command's slot starts (this robot's clock).
    #[pyo3(get)]
    slot_ns: Option<i64>,
    /// The inputs the chunk was computed from: `(path, seq_first,
    /// seq_last)`, with this robot's paths.
    #[pyo3(get)]
    based_on: Vec<(String, u64, u64)>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PyDue {
    /// The command fields by name (validated and clamped), or `None`.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "builtins.dict[builtins.str, Tensor] | None", imports = ("builtins")))]
    fn fields(&self, py: Python<'_>) -> Option<Py<PyDict>> {
        self.fields.as_ref().map(|f| f.clone_ref(py))
    }
}

impl PyDue {
    fn new(py: Python<'_>, robot: &Robot, due: Due) -> PyResult<Self> {
        let fields = match sdk::due_fields(robot, &due).map_err(SessionException::new_err)? {
            Some(fields) => {
                let d = PyDict::new(py);
                for (name, tensor) in fields {
                    d.set_item(name, tensor_to_py(py, tensor)?)?;
                }
                Some(d.unbind())
            }
            None => None,
        };
        let based_on = sdk::due_based_on(robot, &due);
        let Due {
            path,
            mode,
            action,
            safe,
            handover,
        } = due;
        let chunk = action.as_ref().map(|a| &a.chunk);
        Ok(Self {
            mode: sdk::mode_name(mode),
            fields,
            safe: safe.map(|s| s.to_string()),
            handover: handover.map(|h| (h.group, h.from, h.to)),
            lease_token: chunk.map(|c| c.lease_token.0),
            chunk_seq: chunk.map(|c| c.seq),
            index: action.as_ref().map(|a| a.index),
            received_ns: chunk.map(|c| c.receive_ns),
            slot_ns: action.as_ref().map(|a| a.slot_ns),
            based_on,
            path,
        })
    }
}

/// A policy start: each input as `predict` receives it, and its snapshot.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "V2Observation", frozen)]
pub struct PyObservation {
    values: Py<PyDict>,
    meta: Snapshot,
    /// The session time the snapshot was taken at.
    #[pyo3(get)]
    time_ns: i64,
}

impl PyObservation {
    fn new(py: Python<'_>, observation: Observation, time_ns: i64) -> PyResult<Self> {
        let values = PyDict::new(py);
        for (entry, value) in observation.values {
            values.set_item(entry, value_to_py(py, value)?)?;
        }
        Ok(Self {
            values: values.unbind(),
            meta: observation.meta,
            time_ns,
        })
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PyObservation {
    /// Inputs by entry: a `Tensor`, a dict of them, or a `str`.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "builtins.dict[builtins.str, typing.Any]", imports = ("builtins", "typing")))]
    fn values(&self, py: Python<'_>) -> Py<PyDict> {
        self.values.clone_ref(py)
    }

    /// Each input's conservative age in ns (`None`: blank, or no clock
    /// mapping).
    #[getter]
    fn ages(&self) -> BTreeMap<String, Option<i64>> {
        self.meta
            .inputs
            .iter()
            .map(|i| (i.entry.clone(), i.age_ns))
            .collect()
    }

    /// Inputs delivered stale under `allow: hold`.
    #[getter]
    fn stale(&self) -> Vec<String> {
        self.meta
            .inputs
            .iter()
            .filter(|i| i.stale)
            .map(|i| i.entry.clone())
            .collect()
    }

    /// Window slots padded by `allow: warmup`, by entry.
    #[getter]
    fn padded(&self) -> BTreeMap<String, u32> {
        self.meta
            .inputs
            .iter()
            .filter(|i| i.padded > 0)
            .map(|i| (i.entry.clone(), i.padded))
            .collect()
    }

    /// Inputs bound to a constant (`allow: blank`).
    #[getter]
    fn blank(&self) -> Vec<String> {
        self.meta
            .inputs
            .iter()
            .filter(|i| i.blank)
            .map(|i| i.entry.clone())
            .collect()
    }

    /// The newest capture minus the oldest, across inputs.
    #[getter]
    fn skew_ns(&self) -> i64 {
        self.meta.skew_ns
    }

    /// Per stream, the seq range used: `(stream id, seq_first, seq_last)`.
    #[getter]
    fn based_on(&self) -> Vec<(u32, u64, u64)> {
        self.meta
            .based_on
            .iter()
            .map(|r| (r.channel.0, r.seq_first, r.seq_last))
            .collect()
    }
}

/// `predict`, called from the run loop.
struct PyPredictor<'a> {
    predict: &'a Py<PyAny>,
}

impl Predictor for PyPredictor<'_> {
    type Error = PyErr;

    fn predict(
        &mut self,
        observation: Observation,
        snapshot_ns: i64,
    ) -> Result<Vec<(String, Value)>, PyErr> {
        Python::attach(|py| {
            let observation = Py::new(py, PyObservation::new(py, observation, snapshot_ns)?)?;
            let output = self.predict.call1(py, (observation,))?;
            values_from_py(output.bind(py))
        })
    }

    fn idle(&mut self) -> Result<(), PyErr> {
        check_signals()
    }
}

/// A policy bound to a robot: inputs, leases, chunks and the run loop.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "V2Controller", frozen)]
pub struct PyController {
    name: String,
    controller: Mutex<Option<Arc<Controller>>>,
    session: EmbeddedHost,
}

impl PyController {
    fn controller(&self) -> PyResult<Arc<Controller>> {
        self.controller
            .lock()
            .ok()
            .and_then(|c| c.clone())
            .ok_or_else(|| SessionException::new_err(format!("policy {} is closed", self.name)))
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PyController {
    /// Attaches as `name` at `priority`, waits for `robot`'s parts, binds
    /// `policy_json` (the policy map, §6) with the run-time options, and
    /// declares the binding. Raises `BindingRefused` with the binder's
    /// refuse-and-explain text.
    #[new]
    #[pyo3(signature = (
        address, name, policy_json, robot, *, priority="model", under=None,
        at=BTreeMap::new(), positional=Vec::new(), allow=Vec::new(),
        captures=BTreeMap::new(), timeout_s=10.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        address: &str,
        name: &str,
        policy_json: &str,
        robot: &str,
        priority: &str,
        under: Option<&str>,
        at: BTreeMap<String, String>,
        positional: Vec<String>,
        allow: Vec<String>,
        captures: BTreeMap<String, String>,
        timeout_s: f64,
    ) -> PyResult<Self> {
        let session = place(py, address, LEASE_GRACE)?;
        let address = session.address().clone();
        let wait = timeout(timeout_s)?;
        let class = self::priority(priority)?;
        let options = BindOptions {
            name,
            robot,
            under,
            at: at.into_iter().collect(),
            positional: &positional,
            allow: &allow,
            captures: captures.into_iter().collect(),
        }
        .run_options()
        .map_err(|why| BindingRefused::new_err(format!("binding refused: {why}")))?;
        let controller = block_on(py, async move {
            sdk::bind_policy(&address, name, policy_json, &options, class, wait)
                .await
                .map_err(bind_policy_err)
        })?;
        Ok(Self {
            name: name.to_string(),
            controller: Mutex::new(Some(Arc::new(controller))),
            session,
        })
    }

    /// The policy's participant id.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// The binding table, as the cockpit shows it (§6.8).
    #[getter]
    fn table(&self) -> PyResult<String> {
        Ok(self.controller()?.binding().table())
    }

    /// The binding table as canonical JSON (what the fingerprint is over).
    #[getter]
    fn table_json(&self) -> PyResult<String> {
        Ok(self.controller()?.binding().table_text())
    }

    /// The binding fingerprint, 16 hex digits.
    #[getter]
    fn fingerprint(&self) -> PyResult<String> {
        Ok(self.controller()?.binding().fingerprint.to_string())
    }

    /// Warnings the binder accepted the binding with.
    #[getter]
    fn warnings(&self) -> PyResult<Vec<String>> {
        Ok(self.controller()?.binding().warnings.clone())
    }

    /// The replan rate R.
    #[getter]
    fn replan_hz(&self) -> PyResult<f64> {
        Ok(self.controller()?.binding().replan_hz)
    }

    /// Each bound action: `(entry, part path or None, horizon)`.
    fn actions(&self) -> PyResult<Vec<(String, Option<String>, u32)>> {
        Ok(self
            .controller()?
            .binding()
            .actions
            .iter()
            .map(|a| {
                (
                    a.entry.clone(),
                    a.part.as_ref().map(|p| p.path.to_string()),
                    a.horizon,
                )
            })
            .collect())
    }

    /// A policy start now. Raises `StartSkipped` when an input is missing,
    /// stale (without `allow: hold`) or has a window gap.
    fn observe(&self, py: Python<'_>) -> PyResult<PyObservation> {
        let controller = self.controller()?;
        let (observation, at) = py.detach(|| controller.observe()).map_err(observe_err)?;
        PyObservation::new(py, observation, at)
    }

    /// Sends one chunk per action entry, computed from `observation`:
    /// `values` maps each action to `(dtype, shape, bytes)` or a dict of
    /// them, `[horizon, ...]`. Returns each entry's chunk seq, or `None`
    /// without the lease (or for a dropped entry).
    fn send(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "dict[builtins.str, typing.Any]", imports = ("builtins", "typing")))]
        values: &Bound<'_, PyAny>,
        observation: &PyObservation,
    ) -> PyResult<BTreeMap<String, Option<u64>>> {
        let controller = self.controller()?;
        let values = values_from_py(values)?;
        let based_on = observation.meta.based_on.clone();
        let at = observation.time_ns;
        let sent = py
            .detach(|| controller.send(&values, &based_on, at))
            .map_err(send_err)?;
        Ok(sent
            .into_iter()
            .map(|(entry, outcome)| {
                let seq = match outcome {
                    Sent::Seq(seq) => Some(seq),
                    Sent::NoLease | Sent::Dropped => None,
                };
                (entry, seq)
            })
            .collect())
    }

    /// Takes the binding's leases now, preempting a lower class; from then
    /// on they are kept. Raises `SessionRefused` (`LEASE_BUSY`,
    /// `GROUP_BUSY`, `NOT_READY`, ...) when they cannot be granted.
    fn acquire(&self, py: Python<'_>) -> PyResult<Vec<(String, u64)>> {
        let controller = self.controller()?;
        let leases = block_on(py, {
            let controller = controller.clone();
            async move { controller.acquire().await.map_err(session_err) }
        })?;
        Ok(leases
            .iter()
            .map(|l| {
                let path = controller
                    .participant()
                    .part_by_id(l.channel)
                    .map_or_else(|_| l.channel.to_string(), |p| p.path.to_string());
                (path, l.token.0)
            })
            .collect())
    }

    /// Gives the leases back; they are not re-requested until `acquire` or
    /// `run`.
    fn release(&self, py: Python<'_>) -> PyResult<()> {
        let controller = self.controller()?;
        block_on(py, async move {
            controller.release().await.map_err(session_err)
        })
    }

    /// Whether this policy holds every lease its binding commands.
    fn holds(&self) -> PyResult<bool> {
        Ok(self.controller()?.holds())
    }

    /// Runs the policy loop (§8.9) on this thread until `stop()`,
    /// `duration_s`, or a failure, calling `predict(observation)` at the
    /// replan rate with one inference in flight. If `predict` raises, the
    /// leases are released and the exception propagates.
    #[pyo3(signature = (predict, duration_s=None))]
    fn run(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(
            type_repr = "collections.abc.Callable[[V2Observation], dict[builtins.str, typing.Any]]",
            imports = ("collections.abc", "builtins", "typing")
        ))]
        predict: Py<PyAny>,
        duration_s: Option<f64>,
    ) -> PyResult<()> {
        let controller = self.controller()?;
        let duration = duration_s.map(timeout).transpose()?;
        let result = py.detach(|| {
            let mut predictor = PyPredictor { predict: &predict };
            controller.run(&mut predictor, duration)
        });
        match result {
            Ok(()) => Ok(()),
            Err(RunError::Predict(err)) => Err(err),
            Err(RunError::Send(err)) => Err(send_err(err)),
        }
    }

    /// Asks a `run` in progress to return.
    fn stop(&self) -> PyResult<()> {
        self.controller()?.stop();
        Ok(())
    }

    /// The start, skip and completion counters (§8.10), and the input the
    /// starts are phased after (`aligned_to`, `phase_offset_ms`).
    #[gen_stub(override_return_type(type_repr = "builtins.dict[builtins.str, typing.Any]", imports = ("builtins", "typing")))]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let controller = self.controller()?;
        let stats = controller.stats();
        let d = PyDict::new(py);
        d.set_item("aligned_to", controller.aligned_input())?;
        d.set_item(
            "phase_offset_ms",
            controller.phase_offset().as_secs_f64() * 1e3,
        )?;
        d.set_item("starts", stats.starts)?;
        d.set_item("completions", stats.completions)?;
        let skipped = PyDict::new(py);
        for reason in SkipReason::ALL {
            skipped.set_item(reason.name(), stats.skipped(reason))?;
        }
        d.set_item("skipped", skipped)?;
        d.set_item("chunks_sent", stats.chunks_sent)?;
        d.set_item("send_errors", stats.send_errors)?;
        let ms = |d: Option<Duration>| d.map(|d| d.as_secs_f64() * 1e3);
        d.set_item("latency_p50_ms", ms(stats.latency_p50))?;
        d.set_item("latency_p99_ms", ms(stats.latency_p99))?;
        d.set_item("latency_max_ms", ms(stats.latency_max))?;
        d.set_item("drifting", stats.drifting)?;
        Ok(d)
    }

    /// Takes the events reported so far (`rate_drift`, inference pending),
    /// oldest first, as dicts with a `kind` and a `message`.
    #[gen_stub(override_return_type(type_repr = "builtins.list[builtins.dict[builtins.str, typing.Any]]", imports = ("builtins", "typing")))]
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let events = self.controller()?.drain_events();
        let list = PyList::empty(py);
        for event in events {
            let d = PyDict::new(py);
            d.set_item("message", event.to_string())?;
            match event {
                RunEvent::RateDrift {
                    measured_hz,
                    expected_hz,
                    cause,
                    drifting,
                } => {
                    d.set_item("kind", "rate_drift")?;
                    d.set_item("measured_hz", measured_hz)?;
                    d.set_item("expected_hz", expected_hz)?;
                    d.set_item("cause", cause)?;
                    d.set_item("drifting", drifting)?;
                }
                RunEvent::InferencePending { pending } => {
                    d.set_item("kind", "inference_pending")?;
                    d.set_item("pending_s", pending.as_secs_f64())?;
                }
            }
            list.append(d)?;
        }
        Ok(list)
    }

    /// Gives the leases back and detaches.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let taken = self.controller.lock().map(|mut c| c.take()).unwrap_or(None);
        block_on(py, async move {
            if let Some(controller) = taken {
                sdk::close_policy(controller).await.map_err(session_err)?;
            }
            self.session.close().await.map_err(session_err)
        })
    }
}

/// A fixed-rate loop timer on `time.monotonic_ns()`: `sleep()` waits for
/// the next period (coarse sleep, then a short spin) with the GIL
/// released. A tick more than a period late skips the missed ones rather
/// than bursting.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "V2Rate", frozen)]
pub struct PyRate {
    rate: sdk::Rate,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyRate {
    #[new]
    fn new(hz: f64) -> PyResult<Self> {
        Ok(Self {
            rate: sdk::Rate::new(hz).map_err(PyValueError::new_err)?,
        })
    }

    /// Waits until the next period starts; returns how late this call was
    /// for it, in ns.
    fn sleep(&self, py: Python<'_>) -> i64 {
        let clock = SystemMonotonicClock;
        let (next, late) = self.rate.next(clock.now_ns());
        py.detach(|| sleep_until(|| clock.now_ns(), next));
        late
    }

    /// Periods skipped because a call came more than a period late.
    #[getter]
    fn missed(&self) -> u64 {
        self.rate.missed()
    }
}

fn record_err(err: RecordError) -> PyErr {
    match err {
        RecordError::Session(err) => session_err(err),
        other => SessionException::new_err(other.to_string()),
    }
}

/// An MCAP recording of a session (§13), as an observer participant.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "V2Recorder", frozen)]
pub struct PyRecorder {
    path: String,
    recorder: Mutex<Option<Recorder>>,
    session: EmbeddedHost,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PyRecorder {
    /// Attaches as observer `name` and starts recording every stream,
    /// chunk and authority change of the session into a new MCAP file at
    /// `path`.
    #[new]
    #[pyo3(signature = (address, path, name="recorder", timeout_s=10.0))]
    fn new(
        py: Python<'_>,
        address: &str,
        path: String,
        name: &str,
        timeout_s: f64,
    ) -> PyResult<Self> {
        let session = place(py, address, LEASE_GRACE)?;
        let address = session.address().clone();
        let wait = timeout(timeout_s)?;
        let recorder = block_on(py, async {
            sdk::start_recorder(&address, &path, name, wait)
                .await
                .map_err(record_err)
        })?;
        Ok(Self {
            path,
            recorder: Mutex::new(Some(recorder)),
            session,
        })
    }

    /// The MCAP file.
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    /// Stops recording and finishes the file: `messages`, `channels`,
    /// `bytes`, `log_time_clamped`. `None` when already closed.
    #[gen_stub(override_return_type(type_repr = "builtins.dict[builtins.str, typing.Any] | None", imports = ("builtins", "typing")))]
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let taken = self.recorder.lock().map(|mut r| r.take()).unwrap_or(None);
        let Some(recorder) = taken else {
            return Ok(None);
        };
        let stats = block_on(py, async move {
            let stats = sdk::finish_recorder(recorder).await.map_err(record_err)?;
            self.session.close().await.map_err(session_err)?;
            Ok::<_, PyErr>(stats)
        })?;
        let d = PyDict::new(py);
        d.set_item("path", stats.path.to_string_lossy())?;
        d.set_item("messages", stats.messages)?;
        d.set_item("channels", stats.channels)?;
        d.set_item("bytes", stats.bytes)?;
        d.set_item("log_time_clamped", stats.log_time_clamped)?;
        Ok(Some(d))
    }

    /// Reads a recording back: `messages`, `metadata` (record names),
    /// `topics` (per topic: `schema`, `channels`, `messages`,
    /// `first_log_ns`, `last_log_ns`, `log_time_monotonic`,
    /// `publish_time_monotonic`, `undecodable`) and `leases` (the lease
    /// timeline: `log_ns`, `part`, `holder`, `priority`, `token`).
    #[staticmethod]
    #[gen_stub(override_return_type(type_repr = "builtins.dict[builtins.str, typing.Any]", imports = ("builtins", "typing")))]
    fn read<'py>(py: Python<'py>, path: String) -> PyResult<Bound<'py, PyDict>> {
        let summary = py
            .detach(|| rlmesh_session::record::summarize(&path))
            .map_err(record_err)?;
        let d = PyDict::new(py);
        d.set_item("messages", summary.messages)?;
        d.set_item("metadata", summary.metadata)?;
        let topics = PyDict::new(py);
        for t in summary.topics {
            let row = PyDict::new(py);
            row.set_item("schema", t.schema)?;
            row.set_item("channels", t.channels)?;
            row.set_item("messages", t.messages)?;
            row.set_item("first_log_ns", t.first_log_ns)?;
            row.set_item("last_log_ns", t.last_log_ns)?;
            row.set_item("log_time_monotonic", t.log_time_monotonic)?;
            row.set_item("publish_time_monotonic", t.publish_time_monotonic)?;
            row.set_item("undecodable", t.undecodable)?;
            topics.set_item(t.topic, row)?;
        }
        d.set_item("topics", topics)?;
        let leases = PyList::empty(py);
        for l in summary.leases {
            let row = PyDict::new(py);
            row.set_item("log_ns", l.log_ns)?;
            row.set_item("part", l.part)?;
            row.set_item("holder", l.holder)?;
            row.set_item("priority", l.priority)?;
            row.set_item("token", l.token)?;
            leases.append(row)?;
        }
        d.set_item("leases", leases)?;
        Ok(d)
    }
}
