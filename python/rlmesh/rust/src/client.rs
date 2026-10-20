use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_methods_from_python, gen_stub_pyclass};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::inventory::submit;
use rlmesh::binding::env::{
    CoreConnect, EditionCell, EnvClientCore, VectorResetResult, VectorStepResult,
};
use rlmesh_spaces::spaces::SpaceSpec;

use crate::spaces::{
    ValueBackend, batched_space_values_to_py_neutral, env_contract_to_py, make_space,
    meta_map_to_pydict, py_any_to_batched_space_values_with_backend, py_any_to_meta_map,
    py_any_to_space_value_with_backend, space_value_to_py_neutral, tensor_from_shape,
};
use crate::telemetry::{ProfileCollector, init_tracing, profiling_enabled};
use crate::types::{space_value_size, to_py_err};

/// Interval for polling Python signals during blocked RPCs.
const SIGNAL_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The signal poll for a blocked call, first due one interval from now. A plain
/// `interval` ticks at once, and that tick waits for the timer wheel's next
/// millisecond: awaited before the call is first polled, it delayed every
/// request by about 1 ms.
pub(crate) fn signal_poll() -> tokio::time::Interval {
    tokio::time::interval_at(
        tokio::time::Instant::now() + SIGNAL_POLL_INTERVAL,
        SIGNAL_POLL_INTERVAL,
    )
}

/// Poll for a pending Python signal from a thread blocked in native code.
/// A daemon thread can outlive the interpreter; once it has finalized there is
/// no signal to see, and attaching would crash the exiting process.
pub(crate) fn check_signals() -> PyResult<()> {
    Python::try_attach(|py| py.check_signals()).unwrap_or(Ok(()))
}

/// pyo3 answers a re-entrant `&mut self` call with a bare
/// `RuntimeError: Already borrowed`; say what actually happened instead.
fn single_caller_error(class: &str) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "{class} is already in a call on another thread: an RLMesh client owns one wire session \
         and serves one caller at a time -- dial one client per thread"
    ))
}

/// Process-wide Tokio runtime shared by Python env clients.
fn shared_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build shared rlmesh client runtime")
    })
}

/// A client's selected workflow edition, read without borrowing the client,
/// so it never waits on (or fails against) a call in flight (RMT-P5).
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", frozen)]
pub struct PyEditionCell {
    cell: EditionCell,
}

#[pymethods]
impl PyEditionCell {
    /// The selection: the handshake's, or the last pin.
    fn get(&self) -> String {
        self.cell.get()
    }
}

/// Shared core for single and vector env facades: the facade's any-arity
/// client (timeouts, reserved info keys, the closed state and the bounded
/// close live there), driven with the GIL released and signals polled.
struct ClientCore {
    client: EnvClientCore,
    runtime: &'static tokio::runtime::Runtime,
    address: String,
    observation_space: SpaceSpec,
    action_space: SpaceSpec,
    profiler: Arc<ProfileCollector>,
    num_envs: usize,
}

impl ClientCore {
    #[allow(clippy::too_many_arguments)]
    fn connect(
        role: &'static str,
        public: &'static str,
        address: &str,
        connect_timeout_seconds: Option<f64>,
        request_timeout_seconds: Option<f64>,
        workflow_edition: Option<String>,
        api_key: Option<&str>,
        tls_ca_file: Option<&std::path::Path>,
    ) -> PyResult<Self> {
        init_tracing(role);
        let profiler = ProfileCollector::new(role);

        let connect_span = profiling_enabled()
            .then(|| tracing::info_span!("rlmesh.client.connect", address = address));
        let _connect_enter = connect_span.as_ref().map(|span| span.enter());
        let connect_guard = profiler.start("client.connect");

        let runtime = shared_runtime();

        rlmesh::ext::ConnectAddress::parse(address).map_err(to_py_err)?;
        let connect_timeout = rlmesh::binding::env::client_timeout_from_secs(
            connect_timeout_seconds,
            "connect_timeout_seconds",
        )
        .map_err(to_py_err)?;
        let request_timeout = rlmesh::binding::env::client_timeout_from_secs(
            request_timeout_seconds,
            "request_timeout_seconds",
        )
        .map_err(to_py_err)?;
        let declared = crate::lifecycle::checked_workflow_edition(workflow_edition)?;
        let auth =
            rlmesh::binding::client_auth(address, api_key, tls_ca_file, true).map_err(to_py_err)?;

        let mut client = Python::attach(|py| {
            py.detach(|| {
                runtime.block_on(EnvClientCore::connect_with(
                    address,
                    CoreConnect {
                        token: auth.token(),
                        trust: auth.trust().clone(),
                        declared: declared.as_deref(),
                        connect_timeout,
                        request_timeout,
                    },
                ))
            })
        })
        .map_err(to_py_err)?;
        if role == "env_client" {
            client.require_single().map_err(to_py_err)?;
        } else {
            client.require_vector().map_err(to_py_err)?;
        }
        client.set_name(public);
        let _ = connect_guard.finish(0);

        Ok(Self {
            address: client.address().to_string(),
            observation_space: client.observation_space().clone(),
            action_space: client.action_space().clone(),
            num_envs: client.num_envs(),
            client,
            runtime,
            profiler,
        })
    }

    /// Payload size when profiling is active.
    fn measure(&self, value: Option<&rlmesh_spaces::SpaceValue>) -> usize {
        if self.profiler.is_enabled() {
            value.map_or(0, space_value_size)
        } else {
            0
        }
    }

    fn span(&self, name: &'static str) -> Option<tracing::span::EnteredSpan> {
        profiling_enabled()
            .then(|| tracing::info_span!("rlmesh.client", op = name, address = %self.address))
            .map(|span| span.entered())
    }

    fn env_contract_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        env_contract_to_py(py, self.client.env_contract())
    }

    /// Run one RPC with the GIL released, polling Python signals so Ctrl-C
    /// abandons it (RMT-6). Deadlines are the facade client's.
    fn run_rpc<F, T>(&mut self, py: Python<'_>, make: F) -> PyResult<T>
    where
        F: for<'a> FnOnce(
                &'a mut EnvClientCore,
            )
                -> std::pin::Pin<Box<dyn Future<Output = rlmesh::Result<T>> + 'a>>
            + Send,
        T: Send,
    {
        let client = &mut self.client;
        let runtime = self.runtime;

        let outcome = py.detach(|| {
            runtime.block_on(async move {
                let mut rpc = make(client);

                let mut poll = signal_poll();

                loop {
                    tokio::select! {
                        result = &mut rpc => return RpcOutcome::Done(result),
                        _ = poll.tick() => {
                            // Re-acquire the GIL only to check for pending signals.
                            if let Err(err) = check_signals() {
                                return RpcOutcome::Signal(err);
                            }
                        }
                    }
                }
            })
        });

        match outcome {
            RpcOutcome::Done(result) => result.map_err(to_py_err),
            RpcOutcome::Signal(err) => Err(err),
        }
    }

    fn reset_rpc(
        &mut self,
        py: Python<'_>,
        seeds: Vec<i64>,
        options: Option<rlmesh_spaces::MetaMap>,
    ) -> PyResult<VectorResetResult> {
        self.run_rpc(py, move |client| {
            Box::pin(client.reset(rlmesh::binding::env::VectorResetRequest {
                seeds,
                options,
                timeout_ms: 0,
                // Whole-vector reset; partial reset is not exposed on this path.
                env_indices: Vec::new(),
            }))
        })
    }

    fn step_rpc(
        &mut self,
        py: Python<'_>,
        actions: Vec<rlmesh_spaces::SpaceValue>,
    ) -> PyResult<VectorStepResult> {
        self.run_rpc(py, move |client| {
            Box::pin(client.step(rlmesh::binding::env::VectorStepRequest {
                actions,
                timeout_ms: 0,
            }))
        })
    }

    fn render_frame(
        &mut self,
        py: Python<'_>,
        env_index: u64,
        rpc_phase: &'static str,
    ) -> PyResult<Option<Vec<u8>>> {
        let rpc_guard = self.profiler.start(rpc_phase);
        let frame = self.run_rpc(py, move |client| Box::pin(client.render(env_index)))?;
        let _ = rpc_guard.finish(frame.as_ref().map_or(0, Vec::len));
        Ok(frame)
    }

    fn close_rpc(&mut self, py: Python<'_>) -> PyResult<()> {
        // Bounded and detaching on a lost peer inside the facade (RMT-4).
        self.run_rpc(py, |client| Box::pin(client.close()))?;
        self.profiler.log_summary_once();
        Ok(())
    }

    fn shutdown_rpc(&mut self, py: Python<'_>, reason: String) -> PyResult<bool> {
        let accepted = self.run_rpc(py, move |client| {
            Box::pin(async move { client.shutdown(reason.as_str()).await })
        })?;
        self.profiler.log_summary_once();
        Ok(accepted)
    }
}

enum RpcOutcome<T> {
    Done(rlmesh::Result<T>),
    Signal(PyErr),
}

/// Emits a `PyEnvClient`/`PyVectorEnvClient` pyclass: the struct, the methods
/// shared by both facades (connect/handshake/spaces/render/close/shutdown), and
/// the `$extra` methods that differ (reset/step, plus `num_envs` on the vector
/// facade).
macro_rules! client_class {
    ($Class:ident, $role:literal, $public:literal, { $($extra:tt)* }) => {
        #[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
        #[pyclass(module = "rlmesh._rlmesh")]
        pub struct $Class {
            core: ClientCore,
        }

        #[pymethods]
        impl $Class {
            #[new]
            #[pyo3(signature = (address, *, connect_timeout_seconds=None, request_timeout_seconds=None, workflow_edition=None, api_key=None, tls_ca_file=None))]
            fn new(
                address: &str,
                connect_timeout_seconds: Option<f64>,
                request_timeout_seconds: Option<f64>,
                workflow_edition: Option<String>,
                api_key: Option<String>,
                tls_ca_file: Option<String>,
            ) -> PyResult<Self> {
                Ok(Self {
                    core: ClientCore::connect(
                        $role,
                        $public,
                        address,
                        connect_timeout_seconds,
                        request_timeout_seconds,
                        workflow_edition,
                        api_key.as_deref(),
                        tls_ca_file.as_deref().map(std::path::Path::new),
                    )?,
                })
            }

            fn address(&self) -> String {
                self.core.address.clone()
            }

            /// This connection's container id (UUIDv7) — a stable correlation
            /// identity, distinct from the human env name on the contract.
            fn env_id(&self) -> String {
                self.core.client.env_id().to_string()
            }

            /// The workflow edition this connection negotiated with the env.
            fn selected_workflow_edition(&self) -> String {
                self.core.client.selected_workflow_edition().to_string()
            }

            /// A handle reading the selected edition without borrowing this
            /// client, for accessors that overlap a call (RMT-P5).
            fn edition_cell(&self) -> PyEditionCell {
                PyEditionCell {
                    cell: self.core.client.edition_cell(),
                }
            }

            /// The env's handshake offer, `(CAN, WANT)`: the editions it
            /// advertised and the one it declared (`None` when it declared
            /// none). A served-model session hands this to `PyModelClient` so
            /// the three-way floor sees the env's real offer.
            fn session_offer(&self) -> (Vec<String>, Option<String>) {
                let offer = self.core.client.session_offer();
                (offer.editions.clone(), offer.preferred.clone())
            }

            /// Pin the env to the session edition a served-model session
            /// settled on (its three-way floor), sent as the first Join
            /// message. Call before the first reset.
            fn pin_workflow_edition(
                slf: &Bound<'_, Self>,
                selected_workflow_edition: &str,
            ) -> PyResult<()> {
                let mut this = Self::lock(slf)?;
                this.core
                    .client
                    .pin_workflow_edition(selected_workflow_edition);
                Ok(())
            }

            fn handshake(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let _span = self.core.span("handshake");
                self.core.env_contract_py(py)
            }

            fn observation_space(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                Ok(make_space(py, &self.core.observation_space)?
                    .into_any()
                    .unbind())
            }

            fn action_space(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                Ok(make_space(py, &self.core.action_space)?.into_any().unbind())
            }

            #[pyo3(signature = (env_index=0))]
            fn render(
                slf: &Bound<'_, Self>,
                py: Python<'_>,
                env_index: u64,
            ) -> PyResult<Py<PyAny>> {
                let mut this = Self::lock(slf)?;
                let _span = this.core.span("render");
                let frame = this.core.render_frame(py, env_index, "client.render.rpc")?;
                match frame {
                    Some(frame) => Ok(decode_render_frame(py, &frame)?.into_any().unbind()),
                    None => Ok(py.None()),
                }
            }

            fn close(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<()> {
                let mut this = Self::lock(slf)?;
                let _span = this.core.span("close");
                this.core.close_rpc(py)
            }

            #[pyo3(signature = (reason="owner shutdown"))]
            fn shutdown(slf: &Bound<'_, Self>, py: Python<'_>, reason: &str) -> PyResult<bool> {
                let mut this = Self::lock(slf)?;
                let _span = this.core.span("shutdown");
                this.core.shutdown_rpc(py, reason.to_string())
            }

            $($extra)*
        }

        impl $Class {
            /// Borrow the client for one call, or report the single-caller rule.
            fn lock<'py>(slf: &Bound<'py, Self>) -> PyResult<PyRefMut<'py, Self>> {
                slf.try_borrow_mut().map_err(|_| single_caller_error($public))
            }
        }

        impl Drop for $Class {
            fn drop(&mut self) {
                self.core.profiler.log_summary_once();
            }
        }
    };
}

client_class!(PyEnvClient, "env_client", "RemoteEnv", {
    #[pyo3(signature = (seeds=None, options=None))]
    fn reset(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        seeds: Option<Vec<i64>>,
        options: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let mut this = Self::lock(slf)?;
        let _span = this.core.span("reset");
        let total_guard = this.core.profiler.start("client.reset.total");

        let options = decode_options(py, options)?;
        let options_bytes = options.as_ref().map(|value| value.len()).unwrap_or(0);

        let rpc_guard = this.core.profiler.start("client.reset.rpc");
        let result = this
            .core
            .reset_rpc(py, seeds.unwrap_or_default(), options)?;
        let observation = result.observations.first().cloned();
        let obs_bytes_len = this.core.measure(observation.as_ref());
        let info_bytes_len = result.info.as_ref().map(|info| info.len()).unwrap_or(0);
        let _ = rpc_guard.finish(obs_bytes_len + info_bytes_len);

        let obs = match observation.as_ref() {
            Some(value) => space_value_to_py_neutral(py, value, &this.core.observation_space)?,
            None => py.None().bind(py).clone(),
        };
        let info = info_to_pydict(py, result.info.as_ref())?;

        let tuple = PyTuple::new(py, [obs.as_any(), info.as_any()])?;
        let _ = total_guard.finish(options_bytes + obs_bytes_len + info_bytes_len);
        Ok(tuple.into_any().unbind())
    }

    #[pyo3(signature = (actions))]
    fn step(slf: &Bound<'_, Self>, py: Python<'_>, actions: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let mut this = Self::lock(slf)?;
        let _span = this.core.span("step");
        let total_guard = this.core.profiler.start("client.step.total");

        let action = py_any_to_space_value_with_backend(
            py,
            actions.bind(py),
            &this.core.action_space,
            ValueBackend::Native,
        )?;
        let action_bytes_len = this.core.measure(Some(&action));

        let rpc_guard = this.core.profiler.start("client.step.rpc");
        let result = this.core.step_rpc(py, vec![action])?;
        let observation = result.observations.first().cloned();
        let reward = result.rewards.first().copied().unwrap_or_default();
        let terminated = result.terminated.first().copied().unwrap_or_default();
        let truncated = result.truncated.first().copied().unwrap_or_default();
        let obs_bytes_len = this.core.measure(observation.as_ref());
        let info_bytes_len = result.info.as_ref().map(|info| info.len()).unwrap_or(0);
        let _ = rpc_guard.finish(action_bytes_len + obs_bytes_len + info_bytes_len);

        let obs = match observation.as_ref() {
            Some(value) => space_value_to_py_neutral(py, value, &this.core.observation_space)?,
            None => py.None().bind(py).clone(),
        };
        let info = info_to_pydict(py, result.info.as_ref())?;

        let tuple = PyTuple::new(
            py,
            [
                obs.as_any(),
                reward.into_pyobject(py)?.as_any(),
                terminated.into_pyobject(py)?.as_any(),
                truncated.into_pyobject(py)?.as_any(),
                info.as_any(),
            ],
        )?;
        let _ = total_guard.finish(action_bytes_len + obs_bytes_len + info_bytes_len);
        Ok(tuple.into_any().unbind())
    }
});

client_class!(PyVectorEnvClient, "vector_env_client", "RemoteVectorEnv", {
    fn num_envs(&self) -> usize {
        self.core.num_envs
    }

    #[pyo3(signature = (seeds=None, options=None))]
    fn reset(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        seeds: Option<Vec<i64>>,
        options: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let mut this = Self::lock(slf)?;
        let _span = this.core.span("reset");
        let options = decode_options(py, options)?;

        let result = this
            .core
            .reset_rpc(py, seeds.unwrap_or_default(), options)?;

        let obs = batched_space_values_to_py_neutral(
            py,
            &result.observations,
            &this.core.observation_space,
        )?;
        let info = info_to_pydict(py, result.info.as_ref())?;
        Ok(PyTuple::new(py, [obs.as_any(), info.as_any()])?
            .into_any()
            .unbind())
    }

    #[pyo3(signature = (actions))]
    fn step(slf: &Bound<'_, Self>, py: Python<'_>, actions: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let mut this = Self::lock(slf)?;
        let _span = this.core.span("step");
        let batched_actions = py_any_to_batched_space_values_with_backend(
            py,
            actions.bind(py),
            &this.core.action_space,
            this.core.num_envs,
            ValueBackend::Native,
        )?;

        let result = this.core.step_rpc(py, batched_actions)?;

        let obs = batched_space_values_to_py_neutral(
            py,
            &result.observations,
            &this.core.observation_space,
        )?;
        let rewards = vector_f64_to_py(py, &result.rewards)?;
        let terminated = vector_bool_to_py(py, &result.terminated)?;
        let truncated = vector_bool_to_py(py, &result.truncated)?;
        let info = info_to_pydict(py, result.info.as_ref())?;

        Ok(PyTuple::new(
            py,
            [
                obs.as_any(),
                rewards.bind(py).as_any(),
                terminated.bind(py).as_any(),
                truncated.bind(py).as_any(),
                info.as_any(),
            ],
        )?
        .into_any()
        .unbind())
    }
});

#[cfg(feature = "stub-gen")]
submit! {
    gen_methods_from_python! {
        r#"
class PyEnvClient:
    def __init__(self, address: str, *, connect_timeout_seconds: float | None = None, request_timeout_seconds: float | None = None, workflow_edition: str | None = None, api_key: str | None = None, tls_ca_file: str | None = None) -> None: ...
    def address(self) -> str: ...
    def env_id(self) -> str: ...
    def selected_workflow_edition(self) -> str: ...
    def edition_cell(self) -> PyEditionCell: ...
    def session_offer(self) -> tuple[list[str], str | None]: ...
    def pin_workflow_edition(self, selected_workflow_edition: str) -> None: ...
    def handshake(self) -> EnvContract: ...
    def observation_space(self) -> Space: ...
    def action_space(self) -> Space: ...
    def reset(self, seeds: list[int] | None = None, options: dict[str, object] | None = None) -> tuple[Value, ResetInfo]: ...
    def step(self, actions: Value) -> tuple[Value, float, bool, bool, StepInfo]: ...
    def render(self, env_index: int = 0) -> Value | None: ...
    def close(self) -> None: ...
    def shutdown(self, reason: str = "owner shutdown") -> bool: ...
"#
    }
}

#[cfg(feature = "stub-gen")]
submit! {
    gen_methods_from_python! {
        r#"
class PyVectorEnvClient:
    def __init__(self, address: str, *, connect_timeout_seconds: float | None = None, request_timeout_seconds: float | None = None, workflow_edition: str | None = None, api_key: str | None = None, tls_ca_file: str | None = None) -> None: ...
    def address(self) -> str: ...
    def env_id(self) -> str: ...
    def selected_workflow_edition(self) -> str: ...
    def edition_cell(self) -> PyEditionCell: ...
    def session_offer(self) -> tuple[list[str], str | None]: ...
    def pin_workflow_edition(self, selected_workflow_edition: str) -> None: ...
    def handshake(self) -> EnvContract: ...
    def observation_space(self) -> Space: ...
    def action_space(self) -> Space: ...
    def num_envs(self) -> int: ...
    def reset(self, seeds: list[int] | None = None, options: dict[str, object] | None = None) -> tuple[Value, ResetInfo]: ...
    def step(self, actions: object) -> tuple[Value, Value, Value, Value, StepInfo]: ...
    def render(self, env_index: int = 0) -> Value | None: ...
    def close(self) -> None: ...
    def shutdown(self, reason: str = "owner shutdown") -> bool: ...
"#
    }
}

#[cfg(feature = "stub-gen")]
submit! {
    gen_methods_from_python! {
        r#"
class PyEditionCell:
    def get(self) -> str: ...
"#
    }
}

/// Python's `normalize_connect_address` over the facade's one
/// implementation (every language shares its rules and texts); the caller
/// has checked `port`'s type and range and `transport`'s spelling.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (address=None, *, host=None, port=None, path=None, transport=None))]
pub fn normalize_connect_address(
    address: Option<&str>,
    host: Option<&str>,
    port: Option<u16>,
    path: Option<&str>,
    transport: Option<&str>,
) -> PyResult<String> {
    let transport = match transport {
        None => None,
        Some("tcp") => Some(rlmesh::Transport::Tcp),
        Some("unix") => Some(rlmesh::Transport::Unix),
        Some(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "transport must be 'tcp', 'unix', or None",
            ));
        }
    };
    rlmesh::binding::env::normalize_connect_address(address, host, port, path, transport)
        .map_err(to_py_err)
}

/// Parse a client timeout given in seconds: `None` and zero are no timeout;
/// negative or non-finite is ValueError (RMT-4).
pub(crate) fn optional_timeout(
    value: Option<f64>,
    field: &'static str,
) -> PyResult<Option<Duration>> {
    rlmesh::binding::env::client_timeout_from_secs(value, field).map_err(to_py_err)
}

fn decode_options(
    py: Python<'_>,
    options: Option<Py<PyAny>>,
) -> PyResult<Option<rlmesh_spaces::MetaMap>> {
    match options {
        Some(options) => {
            let options_ref = options.bind(py);
            if options_ref.is_none() {
                Ok(None)
            } else {
                Ok(Some(py_any_to_meta_map(options_ref)?))
            }
        }
        None => Ok(None),
    }
}

fn info_to_pydict<'py>(
    py: Python<'py>,
    info: Option<&rlmesh_spaces::MetaMap>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    match info {
        Some(info) => meta_map_to_pydict(py, info),
        None => Ok(pyo3::types::PyDict::new(py)),
    }
}

/// A wire PNG frame as a uint8 tensor; an undecodable frame is
/// `ProtocolException` (RMT-P3).
fn decode_render_frame<'py>(py: Python<'py>, png: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    let (shape, raw) = rlmesh::binding::env::decode_render_frame(png).map_err(to_py_err)?;
    tensor_from_shape(py, raw, shape, "uint8")
}

fn vector_f64_to_py<'py>(py: Python<'py>, values: &[f64]) -> PyResult<Py<PyAny>> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    Ok(tensor_from_shape(py, bytes, vec![values.len()], "float64")?.unbind())
}

fn vector_bool_to_py<'py>(py: Python<'py>, values: &[bool]) -> PyResult<Py<PyAny>> {
    let bytes = values
        .iter()
        .map(|value| u8::from(*value))
        .collect::<Vec<_>>();
    Ok(tensor_from_shape(py, bytes, vec![values.len()], "bool")?.unbind())
}
