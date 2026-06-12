use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use image::ImageFormat;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyTuple};
use pyo3_stub_gen::derive::{gen_methods_from_python, gen_stub_pyclass};
use pyo3_stub_gen::inventory::submit;
use rlmesh::{ConnectAddress, RemoteEnv, ResetResult, StepResult};
use rlmesh_spaces::spaces::SpaceSpec;
use rlmesh_spaces::{RenderFrame as NativeRenderFrame, RenderRequest as NativeRenderRequest};

use crate::spaces::{
    ValueBackend, batched_space_values_to_py_neutral, env_contract_to_py, make_space,
    meta_map_to_pydict, py_any_to_batched_space_values_with_backend, py_any_to_meta_map,
    py_any_to_space_value_with_backend, space_value_to_py_neutral, tensor_from_shape,
};
use crate::telemetry::{ProfileCollector, init_tracing, profiling_enabled};
use crate::types::errors::to_py_err;
use crate::types::value_size::observation_size;

/// Interval at which blocked RPCs re-acquire the GIL to poll for pending Python
/// signals (e.g. KeyboardInterrupt). Short enough to feel responsive to Ctrl+C,
/// long enough that the overhead is negligible against an RPC round trip.
const SIGNAL_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// One process-wide multi-threaded runtime shared by every client, instead of
/// a full `Runtime` per client (review finding #72). A shared multi-thread
/// runtime keeps each gRPC connection's background I/O (HTTP/2 keepalive,
/// connection close) serviced even while a client is idle between calls — a
/// per-client current-thread runtime would only pump the connection inside
/// `block_on`, stalling the server's graceful drain on an idle client.
fn shared_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build shared rlmesh client runtime")
    })
}

/// Shared client core backing both single and vector env facades.
///
/// Owns the `RemoteEnv` directly (no Arc<Mutex>): every pymethod takes
/// `&mut self`, so exclusive access is guaranteed by the borrow checker and
/// `block_on` accepts non-'static futures — there is nothing to share and no
/// lock to hold across `.await`.
struct ClientCore {
    client: RemoteEnv,
    runtime: &'static tokio::runtime::Runtime,
    address: String,
    observation_space: SpaceSpec,
    action_space: SpaceSpec,
    profiler: Arc<ProfileCollector>,
    default_timeout: Option<Duration>,
    num_envs: usize,
}

impl ClientCore {
    fn connect(
        role: &'static str,
        address: &str,
        connect_timeout_seconds: Option<f64>,
        request_timeout_seconds: Option<f64>,
    ) -> PyResult<Self> {
        init_tracing(role);
        let profiler = ProfileCollector::new(role);

        let connect_span = profiling_enabled()
            .then(|| tracing::info_span!("rlmesh.client.connect", address = address));
        let _connect_enter = connect_span.as_ref().map(|span| span.enter());
        let connect_guard = profiler.start("client.connect");

        let runtime = shared_runtime();

        let connect_address = ConnectAddress::parse(address).map_err(to_py_err)?;
        let default_timeout = optional_timeout(request_timeout_seconds, "request_timeout_seconds")?;

        // Release the GIL for the network connect: a black-holed address can
        // park the OS TCP connect for minutes, and holding the GIL there would
        // freeze every other Python thread and make Ctrl+C undeliverable
        // (review finding #32).
        let client = Python::attach(|py| {
            py.detach(|| connect_remote_env(runtime, connect_address, connect_timeout_seconds))
        })?;

        let normalized_address = client.address().to_string();
        let observation_space = require_contract_space(
            client.env_contract().observation_space.clone(),
            "observation_space",
        )?;
        let action_space =
            require_contract_space(client.env_contract().action_space.clone(), "action_space")?;
        let num_envs = client.num_envs();
        let _ = connect_guard.finish(0);

        Ok(Self {
            client,
            runtime,
            address: normalized_address,
            observation_space,
            action_space,
            profiler,
            default_timeout,
            num_envs,
        })
    }

    /// Payload byte size, but only when profiling is active — the recursive
    /// value-tree walk is wasted work otherwise (review finding #112).
    fn measure(&self, value: Option<&rlmesh_spaces::SpaceValue>) -> usize {
        if self.profiler.is_enabled() {
            observation_size(value)
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

    /// Resolve a per-call timeout: an explicit kwarg overrides the client
    /// default; `None`/`0` means no timeout.
    fn resolve_timeout(&self, per_call: Option<f64>) -> PyResult<Option<Duration>> {
        match per_call {
            Some(value) => optional_timeout(Some(value), "timeout_seconds"),
            None => Ok(self.default_timeout),
        }
    }

    fn timeout_ms(timeout: Option<Duration>) -> i64 {
        timeout
            .map(|t| t.as_millis().min(i64::MAX as u128) as i64)
            .unwrap_or(0)
    }

    /// Drive a single RPC future to completion with the GIL released, enforcing
    /// an optional hard client-side deadline and polling for pending Python
    /// signals so a hung call stays interruptible via Ctrl+C (review finding #5).
    fn run_rpc<F, T>(&mut self, py: Python<'_>, timeout: Option<Duration>, make: F) -> PyResult<T>
    where
        F: for<'a> FnOnce(
                &'a mut RemoteEnv,
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

                let mut poll = tokio::time::interval(SIGNAL_POLL_INTERVAL);
                poll.tick().await; // first tick fires immediately; consume it.

                let deadline = timeout.map(|t| tokio::time::Instant::now() + t);

                loop {
                    tokio::select! {
                        result = &mut rpc => return RpcOutcome::Done(result),
                        _ = poll.tick() => {
                            // Re-acquire the GIL only to check for pending signals.
                            if let Err(err) = Python::attach(|py| py.check_signals()) {
                                return RpcOutcome::Signal(err);
                            }
                            if let Some(deadline) = deadline
                                && tokio::time::Instant::now() >= deadline
                            {
                                return RpcOutcome::TimedOut;
                            }
                        }
                    }
                }
            })
        });

        match outcome {
            RpcOutcome::Done(result) => result.map_err(to_py_err),
            RpcOutcome::Signal(err) => Err(err),
            RpcOutcome::TimedOut => Err(pyo3::exceptions::PyTimeoutError::new_err(format!(
                "remote environment call timed out after {:.3}s",
                timeout.expect("timeout set when TimedOut").as_secs_f64()
            ))),
        }
    }

    fn reset_rpc(
        &mut self,
        py: Python<'_>,
        seeds: Vec<i64>,
        options: Option<rlmesh_spaces::MetaMap>,
        timeout: Option<Duration>,
    ) -> PyResult<ResetResult> {
        let timeout_ms = Self::timeout_ms(timeout);
        self.run_rpc(py, timeout, move |client| {
            Box::pin(client.reset(rlmesh::ResetRequest {
                seeds,
                options,
                timeout_ms,
            }))
        })
    }

    fn step_rpc(
        &mut self,
        py: Python<'_>,
        actions: Vec<rlmesh_spaces::SpaceValue>,
        timeout: Option<Duration>,
    ) -> PyResult<StepResult> {
        let timeout_ms = Self::timeout_ms(timeout);
        self.run_rpc(py, timeout, move |client| {
            Box::pin(client.step(rlmesh::StepRequest {
                actions,
                timeout_ms,
            }))
        })
    }

    fn render_message(
        &mut self,
        py: Python<'_>,
        env_index: usize,
        rpc_phase: &'static str,
        timeout: Option<Duration>,
    ) -> PyResult<Option<NativeRenderFrame>> {
        let rpc_guard = self.profiler.start(rpc_phase);
        let timeout_ms = Self::timeout_ms(timeout);
        let result = self.run_rpc(py, timeout, move |client| {
            Box::pin(client.render(NativeRenderRequest {
                env_index: Some(env_index),
                timeout_ms,
            }))
        })?;

        let message = result.frame;
        let frame_bytes_len = message
            .as_ref()
            .map(|frame| frame.png_frame.len())
            .unwrap_or(0);
        let _ = rpc_guard.finish(frame_bytes_len);
        Ok(message)
    }

    fn close_rpc(&mut self, py: Python<'_>) -> PyResult<()> {
        self.run_rpc(py, self.default_timeout, |client| Box::pin(client.close()))?;
        self.profiler.log_summary_once();
        Ok(())
    }

    fn shutdown_rpc(&mut self, py: Python<'_>, reason: String) -> PyResult<bool> {
        let accepted = self.run_rpc(py, self.default_timeout, move |client| {
            Box::pin(client.shutdown(reason))
        })?;
        self.profiler.log_summary_once();
        Ok(accepted)
    }
}

enum RpcOutcome<T> {
    Done(rlmesh::Result<T>),
    Signal(PyErr),
    TimedOut,
}

#[gen_stub_pyclass]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyEnvClient {
    core: ClientCore,
}

#[pymethods]
impl PyEnvClient {
    #[new]
    #[pyo3(signature = (address, *, connect_timeout_seconds=None, request_timeout_seconds=None))]
    fn new(
        address: &str,
        connect_timeout_seconds: Option<f64>,
        request_timeout_seconds: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            core: ClientCore::connect(
                "env_client",
                address,
                connect_timeout_seconds,
                request_timeout_seconds,
            )?,
        })
    }

    fn address(&self) -> String {
        self.core.address.clone()
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

    #[pyo3(signature = (seeds=None, options=None, *, timeout_seconds=None))]
    fn reset(
        &mut self,
        py: Python<'_>,
        seeds: Option<Vec<i64>>,
        options: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let _span = self.core.span("reset");
        let total_guard = self.core.profiler.start("client.reset.total");
        let timeout = self.core.resolve_timeout(timeout_seconds)?;

        let options = decode_options(py, options)?;
        let options_bytes = options.as_ref().map(|value| value.len()).unwrap_or(0);

        let rpc_guard = self.core.profiler.start("client.reset.rpc");
        let result = self
            .core
            .reset_rpc(py, seeds.unwrap_or_default(), options, timeout)?;
        let observation = result.observations.first().cloned();
        let obs_bytes_len = self.core.measure(observation.as_ref());
        let info_bytes_len = result.info.as_ref().map(|info| info.len()).unwrap_or(0);
        let _ = rpc_guard.finish(obs_bytes_len + info_bytes_len);

        let obs = match observation.as_ref() {
            Some(value) => space_value_to_py_neutral(py, value, &self.core.observation_space)?,
            None => py.None().bind(py).clone(),
        };
        let info = info_to_pydict(py, result.info.as_ref())?;
        info.set_item("episode_ids", result.episode_ids.to_vec())?;

        let tuple = PyTuple::new(py, [obs.as_any(), info.as_any()])?;
        let _ = total_guard.finish(options_bytes + obs_bytes_len + info_bytes_len);
        Ok(tuple.into_any().unbind())
    }

    #[pyo3(signature = (actions, *, timeout_seconds=None))]
    fn step(
        &mut self,
        py: Python<'_>,
        actions: Py<PyAny>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let _span = self.core.span("step");
        let total_guard = self.core.profiler.start("client.step.total");
        let timeout = self.core.resolve_timeout(timeout_seconds)?;

        let action = py_any_to_space_value_with_backend(
            py,
            actions.bind(py),
            &self.core.action_space,
            ValueBackend::Native,
        )?;
        let action_bytes_len = self.core.measure(Some(&action));

        let rpc_guard = self.core.profiler.start("client.step.rpc");
        let result = self.core.step_rpc(py, vec![action], timeout)?;
        let observation = result.observations.first().cloned();
        let reward = result.rewards.first().copied().unwrap_or_default();
        let terminated = result.terminated.first().copied().unwrap_or_default();
        let truncated = result.truncated.first().copied().unwrap_or_default();
        let obs_bytes_len = self.core.measure(observation.as_ref());
        let info_bytes_len = result.info.as_ref().map(|info| info.len()).unwrap_or(0);
        let _ = rpc_guard.finish(action_bytes_len + obs_bytes_len + info_bytes_len);

        let obs = match observation.as_ref() {
            Some(value) => space_value_to_py_neutral(py, value, &self.core.observation_space)?,
            None => py.None().bind(py).clone(),
        };
        let info = info_to_pydict(py, result.info.as_ref())?;
        if terminated || truncated {
            info.set_item("completed_episodes", 1)?;
        }

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

    #[pyo3(signature = (env_index=0, *, timeout_seconds=None))]
    fn render(
        &mut self,
        py: Python<'_>,
        env_index: usize,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let _span = self.core.span("render");
        let timeout = self.core.resolve_timeout(timeout_seconds)?;
        let result = self
            .core
            .render_message(py, env_index, "client.render.rpc", timeout)?;
        match result {
            Some(message) => Ok(decode_render_frame(py, &message)?.into_any().unbind()),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (env_index=0, *, timeout_seconds=None))]
    fn render_packet(
        &mut self,
        py: Python<'_>,
        env_index: usize,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let _span = self.core.span("render_packet");
        let timeout = self.core.resolve_timeout(timeout_seconds)?;
        let result =
            self.core
                .render_message(py, env_index, "client.render_packet.rpc", timeout)?;
        match result {
            Some(frame) => Ok(PyBytes::new(py, render_packet(&frame)).into_any().unbind()),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (env_index=0, *, timeout_seconds=None))]
    fn render_bundle(
        &mut self,
        py: Python<'_>,
        env_index: usize,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let _span = self.core.span("render_bundle");
        let timeout = self.core.resolve_timeout(timeout_seconds)?;
        let result =
            self.core
                .render_message(py, env_index, "client.render_bundle.rpc", timeout)?;
        render_bundle_py(py, result)
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let _span = self.core.span("close");
        self.core.close_rpc(py)
    }

    #[pyo3(signature = (reason="owner shutdown"))]
    fn shutdown(&mut self, py: Python<'_>, reason: &str) -> PyResult<bool> {
        let _span = self.core.span("shutdown");
        self.core.shutdown_rpc(py, reason.to_string())
    }
}

impl Drop for PyEnvClient {
    fn drop(&mut self) {
        self.core.profiler.log_summary_once();
    }
}

#[gen_stub_pyclass]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyVectorEnvClient {
    core: ClientCore,
}

#[pymethods]
impl PyVectorEnvClient {
    #[new]
    #[pyo3(signature = (address, *, connect_timeout_seconds=None, request_timeout_seconds=None))]
    fn new(
        address: &str,
        connect_timeout_seconds: Option<f64>,
        request_timeout_seconds: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            core: ClientCore::connect(
                "vector_env_client",
                address,
                connect_timeout_seconds,
                request_timeout_seconds,
            )?,
        })
    }

    fn address(&self) -> String {
        self.core.address.clone()
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

    fn num_envs(&self) -> usize {
        self.core.num_envs
    }

    #[pyo3(signature = (seeds=None, options=None, *, timeout_seconds=None))]
    fn reset(
        &mut self,
        py: Python<'_>,
        seeds: Option<Vec<i64>>,
        options: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let _span = self.core.span("reset");
        let timeout = self.core.resolve_timeout(timeout_seconds)?;
        let options = decode_options(py, options)?;

        let result = self
            .core
            .reset_rpc(py, seeds.unwrap_or_default(), options, timeout)?;

        let obs = batched_space_values_to_py_neutral(
            py,
            &result.observations,
            &self.core.observation_space,
        )?;
        let info = info_to_pydict(py, result.info.as_ref())?;
        info.set_item("episode_ids", result.episode_ids)?;
        Ok(PyTuple::new(py, [obs.as_any(), info.as_any()])?
            .into_any()
            .unbind())
    }

    #[pyo3(signature = (actions, *, timeout_seconds=None))]
    fn step(
        &mut self,
        py: Python<'_>,
        actions: Py<PyAny>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let _span = self.core.span("step");
        let timeout = self.core.resolve_timeout(timeout_seconds)?;
        let batched_actions = py_any_to_batched_space_values_with_backend(
            py,
            actions.bind(py),
            &self.core.action_space,
            self.core.num_envs,
            ValueBackend::Native,
        )?;

        let result = self.core.step_rpc(py, batched_actions, timeout)?;

        let obs = batched_space_values_to_py_neutral(
            py,
            &result.observations,
            &self.core.observation_space,
        )?;
        let rewards = vector_f64_to_py(py, &result.rewards)?;
        let terminated = vector_bool_to_py(py, &result.terminated)?;
        let truncated = vector_bool_to_py(py, &result.truncated)?;
        let info = info_to_pydict(py, result.info.as_ref())?;
        info.set_item("episode_ids", result.episode_ids)?;
        info.set_item("completed_episodes", result.completed_episodes.len())?;

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

    #[pyo3(signature = (env_index=0, *, timeout_seconds=None))]
    fn render(
        &mut self,
        py: Python<'_>,
        env_index: usize,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let _span = self.core.span("render");
        let timeout = self.core.resolve_timeout(timeout_seconds)?;
        let result = self
            .core
            .render_message(py, env_index, "client.render.rpc", timeout)?;
        match result {
            Some(message) => Ok(decode_render_frame(py, &message)?.into_any().unbind()),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (env_index=0, *, timeout_seconds=None))]
    fn render_packet(
        &mut self,
        py: Python<'_>,
        env_index: usize,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let _span = self.core.span("render_packet");
        let timeout = self.core.resolve_timeout(timeout_seconds)?;
        let result =
            self.core
                .render_message(py, env_index, "client.render_packet.rpc", timeout)?;
        match result {
            Some(frame) => Ok(PyBytes::new(py, render_packet(&frame)).into_any().unbind()),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (env_index=0, *, timeout_seconds=None))]
    fn render_bundle(
        &mut self,
        py: Python<'_>,
        env_index: usize,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let _span = self.core.span("render_bundle");
        let timeout = self.core.resolve_timeout(timeout_seconds)?;
        let result =
            self.core
                .render_message(py, env_index, "client.render_bundle.rpc", timeout)?;
        render_bundle_py(py, result)
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let _span = self.core.span("close");
        self.core.close_rpc(py)
    }

    #[pyo3(signature = (reason="owner shutdown"))]
    fn shutdown(&mut self, py: Python<'_>, reason: &str) -> PyResult<bool> {
        let _span = self.core.span("shutdown");
        self.core.shutdown_rpc(py, reason.to_string())
    }
}

impl Drop for PyVectorEnvClient {
    fn drop(&mut self) {
        self.core.profiler.log_summary_once();
    }
}

submit! {
    gen_methods_from_python! {
        r#"
class PyEnvClient:
    def __init__(self, address: str, *, connect_timeout_seconds: float | None = None, request_timeout_seconds: float | None = None) -> None: ...
    def address(self) -> str: ...
    def handshake(self) -> EnvContract: ...
    def observation_space(self) -> Space: ...
    def action_space(self) -> Space: ...
    def reset(self, seeds: list[int] | None = None, options: dict[str, object] | None = None, *, timeout_seconds: float | None = None) -> tuple[Value, ResetInfo]: ...
    def step(self, actions: Value, *, timeout_seconds: float | None = None) -> tuple[Value, float, bool, bool, StepInfo]: ...
    def render(self, env_index: int = 0, *, timeout_seconds: float | None = None) -> Value | None: ...
    def render_packet(self, env_index: int = 0, *, timeout_seconds: float | None = None) -> bytes | None: ...
    def render_bundle(self, env_index: int = 0, *, timeout_seconds: float | None = None) -> tuple[Value | None, bytes | None]: ...
    def close(self) -> None: ...
    def shutdown(self, reason: str = "owner shutdown") -> bool: ...
"#
    }
}

submit! {
    gen_methods_from_python! {
        r#"
class PyVectorEnvClient:
    def __init__(self, address: str, *, connect_timeout_seconds: float | None = None, request_timeout_seconds: float | None = None) -> None: ...
    def address(self) -> str: ...
    def handshake(self) -> EnvContract: ...
    def observation_space(self) -> Space: ...
    def action_space(self) -> Space: ...
    def num_envs(self) -> int: ...
    def reset(self, seeds: list[int] | None = None, options: dict[str, object] | None = None, *, timeout_seconds: float | None = None) -> tuple[Value, ResetInfo]: ...
    def step(self, actions: object, *, timeout_seconds: float | None = None) -> tuple[Value, Value, Value, Value, StepInfo]: ...
    def render(self, env_index: int = 0, *, timeout_seconds: float | None = None) -> Value | None: ...
    def render_packet(self, env_index: int = 0, *, timeout_seconds: float | None = None) -> bytes | None: ...
    def render_bundle(self, env_index: int = 0, *, timeout_seconds: float | None = None) -> tuple[Value | None, bytes | None]: ...
    def close(self) -> None: ...
    def shutdown(self, reason: str = "owner shutdown") -> bool: ...
"#
    }
}

fn connect_remote_env(
    runtime: &tokio::runtime::Runtime,
    address: ConnectAddress,
    connect_timeout_seconds: Option<f64>,
) -> PyResult<RemoteEnv> {
    let timeout = optional_timeout(connect_timeout_seconds, "connect_timeout_seconds")?;
    let connect = RemoteEnv::connect_to(address);
    match timeout {
        Some(timeout) => runtime
            .block_on(async { tokio::time::timeout(timeout, connect).await })
            .map_err(|_| {
                pyo3::exceptions::PyTimeoutError::new_err(format!(
                    "remote environment connect timed out after {:.3}s",
                    timeout.as_secs_f64()
                ))
            })?
            .map_err(to_py_err),
        None => runtime.block_on(connect).map_err(to_py_err),
    }
}

fn optional_timeout(value: Option<f64>, field: &'static str) -> PyResult<Option<Duration>> {
    match value {
        // Treat 0 (and None) as "no timeout" to mirror the wire contract.
        None | Some(0.0) => Ok(None),
        Some(value) => Duration::try_from_secs_f64(value).map(Some).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{field} must be a non-negative finite float or None"
            ))
        }),
    }
}

fn require_contract_space(space: Option<SpaceSpec>, field: &'static str) -> PyResult<SpaceSpec> {
    space.ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "remote environment contract missing {field}"
        ))
    })
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

fn render_bundle_py(py: Python<'_>, result: Option<NativeRenderFrame>) -> PyResult<Py<PyAny>> {
    match result {
        Some(frame) => {
            let frame_value = decode_render_frame(py, &frame)?.into_any().unbind();
            let packet = PyBytes::new(py, render_packet(&frame));
            Ok(
                PyTuple::new(py, [frame_value.bind(py).as_any(), packet.as_any()])?
                    .into_any()
                    .unbind(),
            )
        }
        None => Ok(PyTuple::new(
            py,
            [py.None().bind(py).as_any(), py.None().bind(py).as_any()],
        )?
        .into_any()
        .unbind()),
    }
}

struct DecodedRenderFrame {
    shape: Vec<usize>,
    raw: Vec<u8>,
}

fn decode_render_frame<'py>(
    py: Python<'py>,
    frame: &NativeRenderFrame,
) -> PyResult<Bound<'py, PyAny>> {
    let frame = decode_render_bytes(frame).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    tensor_from_shape(py, frame.raw, frame.shape, "uint8")
}

fn decode_render_bytes(frame: &NativeRenderFrame) -> Result<DecodedRenderFrame, String> {
    use image::DynamicImage;

    let image = image::load_from_memory_with_format(&frame.png_frame, ImageFormat::Png)
        .map_err(|err| err.to_string())?;

    // Preserve the source channel count so a remote render() matches the
    // Gymnasium rgb_array contract: RGB -> (H, W, 3), grayscale -> (H, W), and
    // only genuinely 4-channel sources stay (H, W, 4). The server encodes the
    // PNG with the original ColorType, so the decoded color type is authoritative.
    let (width, height) = (image.width() as usize, image.height() as usize);
    let (channels, raw) = match image {
        DynamicImage::ImageLuma8(buf) => (1usize, buf.into_raw()),
        DynamicImage::ImageRgb8(buf) => (3, buf.into_raw()),
        DynamicImage::ImageRgba8(buf) => (4, buf.into_raw()),
        // Grayscale-with-alpha and any 16-bit/float PNG (not produced by the
        // server encoder) fall back to RGBA8 rather than failing.
        other => (4, other.to_rgba8().into_raw()),
    };

    let shape = if channels == 1 {
        vec![height, width]
    } else {
        vec![height, width, channels]
    };
    Ok(DecodedRenderFrame { shape, raw })
}

fn render_packet(frame: &NativeRenderFrame) -> &[u8] {
    frame.png_frame.as_slice()
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

#[cfg(test)]
mod tests {
    use super::{NativeRenderFrame, decode_render_bytes};
    use image::{ColorType, ImageEncoder, codecs::png::PngEncoder};

    fn png_frame(raw: &[u8], width: u32, height: u32, color: ColorType) -> NativeRenderFrame {
        let mut encoded = Vec::new();
        PngEncoder::new(&mut encoded)
            .write_image(raw, width, height, color.into())
            .unwrap();
        NativeRenderFrame { png_frame: encoded }
    }

    #[test]
    fn rgb_source_decodes_to_three_channels() {
        // 2x1 RGB image -> Gymnasium rgb_array contract is (H, W, 3).
        let frame = png_frame(&[1, 2, 3, 4, 5, 6], 2, 1, ColorType::Rgb8);
        let decoded = decode_render_bytes(&frame).unwrap();
        assert_eq!(decoded.shape, vec![1, 2, 3]);
        assert_eq!(decoded.raw, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn rgba_source_decodes_to_four_channels() {
        let frame = png_frame(&[1, 2, 3, 4, 5, 6, 7, 8], 2, 1, ColorType::Rgba8);
        let decoded = decode_render_bytes(&frame).unwrap();
        assert_eq!(decoded.shape, vec![1, 2, 4]);
        assert_eq!(decoded.raw, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn grayscale_source_decodes_to_two_dimensions() {
        let frame = png_frame(&[10, 20, 30, 40], 2, 2, ColorType::L8);
        let decoded = decode_render_bytes(&frame).unwrap();
        assert_eq!(decoded.shape, vec![2, 2]);
        assert_eq!(decoded.raw, vec![10, 20, 30, 40]);
    }
}
