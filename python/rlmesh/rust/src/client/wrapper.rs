use image::ImageFormat;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyTuple};
use pyo3_stub_gen::derive::{gen_methods_from_python, gen_stub_pyclass};
use pyo3_stub_gen::inventory::submit;
use rlmesh::{ConnectAddress, RemoteEnv};
use rlmesh_spaces::spaces::SpaceSpec;
use rlmesh_spaces::{RenderFrame as NativeRenderFrame, RenderRequest as NativeRenderRequest};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::spaces::{
    ValueBackend, batched_space_values_to_py_neutral, env_contract_to_py, make_space,
    meta_map_to_pydict, py_any_to_batched_space_values_with_backend, py_any_to_meta_map,
    py_any_to_space_value_with_backend, space_value_to_py_neutral, tensor_from_shape,
};
use crate::telemetry::{ProfileCollector, init_tracing};
use crate::types::errors::to_py_err;

#[gen_stub_pyclass]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyEnvClient {
    client: Arc<Mutex<RemoteEnv>>,
    runtime: tokio::runtime::Runtime,
    address: String,
    observation_space: SpaceSpec,
    action_space: SpaceSpec,
    handshake_complete: bool,
    profiler: Arc<ProfileCollector>,
}

#[allow(clippy::await_holding_lock)]
#[pymethods]
impl PyEnvClient {
    #[new]
    #[pyo3(signature = (address, *, connect_timeout_seconds=None))]
    fn new(address: &str, connect_timeout_seconds: Option<f64>) -> PyResult<Self> {
        init_tracing("env_client");
        let profiler = ProfileCollector::new("env_client");

        let connect_span = tracing::info_span!("rlmesh.client.connect", address = address);
        let _connect_enter = connect_span.enter();
        let connect_guard = profiler.start("client.connect");

        let runtime = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create runtime: {e}"))
        })?;

        let connect_address = ConnectAddress::parse(address).map_err(to_py_err)?;
        let client = connect_remote_env(&runtime, connect_address, connect_timeout_seconds)?;
        let normalized_address = client.address().to_string();
        let observation_space = require_contract_space(
            client.env_contract().observation_space.clone(),
            "observation_space",
        )?;
        let action_space =
            require_contract_space(client.env_contract().action_space.clone(), "action_space")?;
        let _ = connect_guard.finish(0);

        Ok(Self {
            client: Arc::new(Mutex::new(client)),
            runtime,
            address: normalized_address,
            observation_space,
            action_space,
            handshake_complete: true,
            profiler,
        })
    }

    fn address(&self) -> String {
        self.address.clone()
    }

    fn handshake(&mut self) -> PyResult<Py<PyAny>> {
        let span = tracing::info_span!("rlmesh.client.handshake", address = %self.address);
        let _enter = span.enter();
        let total_guard = self.profiler.start("client.handshake");
        let env_contract = self
            .client
            .lock()
            .expect("env client mutex poisoned")
            .env_contract()
            .clone();
        let _ = total_guard.finish(0);

        Python::attach(|py| env_contract_to_py(py, &env_contract))
    }

    fn observation_space(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if !self.handshake_complete {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "handshake() must be called before observation_space()",
            ));
        }
        Ok(make_space(py, &self.observation_space)?.into_any().unbind())
    }

    fn action_space(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if !self.handshake_complete {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "handshake() must be called before action_space()",
            ));
        }
        Ok(make_space(py, &self.action_space)?.into_any().unbind())
    }

    #[pyo3(signature = (seeds=None, options=None))]
    fn reset(
        &mut self,
        py: Python<'_>,
        seeds: Option<Vec<i64>>,
        options: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let span = tracing::info_span!("rlmesh.client.reset", address = %self.address);
        let _enter = span.enter();
        let total_guard = self.profiler.start("client.reset.total");
        let encode_guard = self.profiler.start("client.reset.encode_options");

        let client = Arc::clone(&self.client);
        let options = match options {
            Some(options) => {
                let options_ref = options.bind(py);
                if options_ref.is_none() {
                    None
                } else {
                    Some(py_any_to_meta_map(options_ref)?)
                }
            }
            None => None,
        };
        let options_bytes = options.as_ref().map(|value| value.len()).unwrap_or(0);
        let _ = encode_guard.finish(options_bytes);

        let rpc_guard = self.profiler.start("client.reset.rpc");
        let runtime = &self.runtime;
        let result = py
            .detach(|| {
                runtime.block_on(async move {
                    let mut guard = client.lock().expect("env client mutex poisoned");
                    guard
                        .reset(rlmesh::ResetRequest {
                            seeds: seeds.unwrap_or_default(),
                            options,
                            timeout_ms: 0,
                        })
                        .await
                })
            })
            .map_err(to_py_err)?;

        let observation = result.observations.first().cloned();
        let obs_bytes_len = observation_size(observation.as_ref());
        let info_bytes_len = result.info.as_ref().map(|info| info.len()).unwrap_or(0);
        let _ = rpc_guard.finish(obs_bytes_len + info_bytes_len);

        let decode_guard = self.profiler.start("client.reset.decode_obs");
        let obs = match observation.as_ref() {
            Some(value) => space_value_to_py_neutral(py, value, &self.observation_space)?,
            None => py.None().bind(py).clone(),
        };

        let info = match result.info.as_ref() {
            Some(info) => meta_map_to_pydict(py, info)?,
            None => pyo3::types::PyDict::new(py),
        };
        info.set_item("episode_ids", result.episode_ids.to_vec())?;

        let tuple = PyTuple::new(py, [obs.as_any(), info.as_any()])?;
        let _ = decode_guard.finish(obs_bytes_len);
        let _ = total_guard.finish(options_bytes + obs_bytes_len + info_bytes_len);
        Ok(tuple.into_any().unbind())
    }

    fn step(&mut self, py: Python<'_>, actions: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let span = tracing::info_span!("rlmesh.client.step", address = %self.address);
        let _enter = span.enter();
        let total_guard = self.profiler.start("client.step.total");

        let client = Arc::clone(&self.client);
        let action_space = self.action_space.clone();

        let encode_guard = self.profiler.start("client.step.encode_action");
        let actions_ref = actions.bind(py);
        let action = py_any_to_space_value_with_backend(
            py,
            actions_ref,
            &action_space,
            ValueBackend::Native,
        )?;
        let action_bytes_len = observation_size(Some(&action));
        let _ = encode_guard.finish(action_bytes_len);

        let rpc_guard = self.profiler.start("client.step.rpc");
        let runtime = &self.runtime;
        let result = py
            .detach(|| {
                runtime.block_on(async move {
                    let mut guard = client.lock().expect("env client mutex poisoned");
                    guard
                        .step(rlmesh::StepRequest {
                            actions: vec![action],
                            timeout_ms: 0,
                        })
                        .await
                })
            })
            .map_err(to_py_err)?;

        let observation = result.observations.first().cloned();
        let reward = result.rewards.first().copied().unwrap_or_default();
        let terminated = result.terminated.first().copied().unwrap_or_default();
        let truncated = result.truncated.first().copied().unwrap_or_default();
        let obs_bytes_len = observation_size(observation.as_ref());
        let info_bytes_len = result.info.as_ref().map(|info| info.len()).unwrap_or(0);
        let _ = rpc_guard.finish(action_bytes_len + obs_bytes_len + info_bytes_len);

        let decode_guard = self.profiler.start("client.step.decode_obs");
        let obs = match observation.as_ref() {
            Some(value) => space_value_to_py_neutral(py, value, &self.observation_space)?,
            None => py.None().bind(py).clone(),
        };

        let info = match result.info.as_ref() {
            Some(info) => meta_map_to_pydict(py, info)?,
            None => pyo3::types::PyDict::new(py),
        };
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
        let _ = decode_guard.finish(obs_bytes_len);
        let _ = total_guard.finish(action_bytes_len + obs_bytes_len + info_bytes_len);
        Ok(tuple.into_any().unbind())
    }

    #[pyo3(signature = (env_index=0))]
    fn render(&mut self, py: Python<'_>, env_index: usize) -> PyResult<Py<PyAny>> {
        let span = tracing::info_span!("rlmesh.client.render", address = %self.address);
        let _enter = span.enter();
        let total_guard = self.profiler.start("client.render.total");
        let result = self.render_message(py, env_index, "client.render.rpc")?;
        let frame_bytes_len = result
            .as_ref()
            .map(|frame| frame.png_frame.len())
            .unwrap_or(0);

        let decode_guard = self.profiler.start("client.render.decode_frame");
        let decoded = match result {
            Some(message) => decode_render_frame(py, &message)?.into_any().unbind(),
            None => py.None(),
        };
        let _ = decode_guard.finish(frame_bytes_len);
        let _ = total_guard.finish(frame_bytes_len);
        Ok(decoded)
    }

    #[pyo3(signature = (env_index=0))]
    fn render_packet(&mut self, py: Python<'_>, env_index: usize) -> PyResult<Py<PyAny>> {
        let span = tracing::info_span!("rlmesh.client.render_packet", address = %self.address);
        let _enter = span.enter();
        let total_guard = self.profiler.start("client.render_packet.total");
        let result = self.render_message(py, env_index, "client.render_packet.rpc")?;
        let frame_bytes_len = result
            .as_ref()
            .map(|frame| frame.png_frame.len())
            .unwrap_or(0);

        let packet = match result {
            Some(frame) => PyBytes::new(py, render_packet(&frame)).into_any().unbind(),
            None => py.None(),
        };
        let _ = total_guard.finish(frame_bytes_len);
        Ok(packet)
    }

    #[pyo3(signature = (env_index=0))]
    fn render_bundle(&mut self, py: Python<'_>, env_index: usize) -> PyResult<Py<PyAny>> {
        let span = tracing::info_span!("rlmesh.client.render_bundle", address = %self.address);
        let _enter = span.enter();
        let total_guard = self.profiler.start("client.render_bundle.total");
        let result = self.render_message(py, env_index, "client.render_bundle.rpc")?;
        let frame_bytes_len = result
            .as_ref()
            .map(|frame| frame.png_frame.len())
            .unwrap_or(0);

        let decode_guard = self.profiler.start("client.render_bundle.decode_frame");
        let bundle = match result {
            Some(frame) => {
                let frame_value = decode_render_frame(py, &frame)?.into_any().unbind();
                let packet = PyBytes::new(py, render_packet(&frame));
                PyTuple::new(py, [frame_value.bind(py).as_any(), packet.as_any()])?
                    .into_any()
                    .unbind()
            }
            None => PyTuple::new(
                py,
                [py.None().bind(py).as_any(), py.None().bind(py).as_any()],
            )?
            .into_any()
            .unbind(),
        };
        let _ = decode_guard.finish(frame_bytes_len);
        let _ = total_guard.finish(frame_bytes_len);
        Ok(bundle)
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let span = tracing::info_span!("rlmesh.client.close", address = %self.address);
        let _enter = span.enter();
        let client = Arc::clone(&self.client);
        let runtime = &self.runtime;

        py.detach(|| {
            runtime.block_on(async move {
                let mut guard = client.lock().expect("env client mutex poisoned");
                guard.close().await
            })
        })
        .map_err(to_py_err)?;

        self.profiler.log_summary_once();
        Ok(())
    }

    #[pyo3(signature = (reason="owner shutdown"))]
    fn shutdown(&mut self, py: Python<'_>, reason: &str) -> PyResult<bool> {
        let span = tracing::info_span!("rlmesh.client.shutdown", address = %self.address);
        let _enter = span.enter();
        let client = Arc::clone(&self.client);
        let reason = reason.to_string();
        let runtime = &self.runtime;

        let accepted = py
            .detach(|| {
                runtime.block_on(async move {
                    let mut guard = client.lock().expect("env client mutex poisoned");
                    guard.shutdown(reason).await
                })
            })
            .map_err(to_py_err)?;

        self.profiler.log_summary_once();
        Ok(accepted)
    }
}

impl Drop for PyEnvClient {
    fn drop(&mut self) {
        self.profiler.log_summary_once();
    }
}

fn connect_remote_env(
    runtime: &tokio::runtime::Runtime,
    address: ConnectAddress,
    connect_timeout_seconds: Option<f64>,
) -> PyResult<RemoteEnv> {
    let timeout = optional_connect_timeout(connect_timeout_seconds)?;
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

fn optional_connect_timeout(value: Option<f64>) -> PyResult<Option<Duration>> {
    value
        .map(|value| {
            Duration::try_from_secs_f64(value).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "connect_timeout_seconds must be a non-negative finite float or None",
                )
            })
        })
        .transpose()
}

fn require_contract_space(space: Option<SpaceSpec>, field: &'static str) -> PyResult<SpaceSpec> {
    space.ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "remote environment contract missing {field}"
        ))
    })
}

#[gen_stub_pyclass]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyVectorEnvClient {
    client: Arc<Mutex<RemoteEnv>>,
    runtime: tokio::runtime::Runtime,
    address: String,
    observation_space: SpaceSpec,
    action_space: SpaceSpec,
    handshake_complete: bool,
    profiler: Arc<ProfileCollector>,
    num_envs: usize,
}

#[allow(clippy::await_holding_lock)]
#[pymethods]
impl PyVectorEnvClient {
    #[new]
    #[pyo3(signature = (address, *, connect_timeout_seconds=None))]
    fn new(address: &str, connect_timeout_seconds: Option<f64>) -> PyResult<Self> {
        init_tracing("vector_env_client");
        let profiler = ProfileCollector::new("vector_env_client");
        let runtime = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create runtime: {e}"))
        })?;

        let connect_address = ConnectAddress::parse(address).map_err(to_py_err)?;
        let client = connect_remote_env(&runtime, connect_address, connect_timeout_seconds)?;
        let normalized_address = client.address().to_string();
        let observation_space = require_contract_space(
            client.env_contract().observation_space.clone(),
            "observation_space",
        )?;
        let action_space =
            require_contract_space(client.env_contract().action_space.clone(), "action_space")?;
        let num_envs = client.num_envs();

        Ok(Self {
            client: Arc::new(Mutex::new(client)),
            runtime,
            address: normalized_address,
            observation_space,
            action_space,
            handshake_complete: true,
            profiler,
            num_envs,
        })
    }

    fn address(&self) -> String {
        self.address.clone()
    }

    fn handshake(&mut self) -> PyResult<Py<PyAny>> {
        let env_contract = self
            .client
            .lock()
            .expect("env client mutex poisoned")
            .env_contract()
            .clone();
        Python::attach(|py| env_contract_to_py(py, &env_contract))
    }

    fn observation_space(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if !self.handshake_complete {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "handshake() must be called before observation_space()",
            ));
        }
        Ok(make_space(py, &self.observation_space)?.into_any().unbind())
    }

    fn action_space(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if !self.handshake_complete {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "handshake() must be called before action_space()",
            ));
        }
        Ok(make_space(py, &self.action_space)?.into_any().unbind())
    }

    fn num_envs(&self) -> usize {
        self.num_envs
    }

    #[pyo3(signature = (seeds=None, options=None))]
    fn reset(
        &mut self,
        py: Python<'_>,
        seeds: Option<Vec<i64>>,
        options: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let client = Arc::clone(&self.client);
        let options = match options {
            Some(options) => {
                let options_ref = options.bind(py);
                if options_ref.is_none() {
                    None
                } else {
                    Some(py_any_to_meta_map(options_ref)?)
                }
            }
            None => None,
        };

        let runtime = &self.runtime;
        let result = py
            .detach(|| {
                runtime.block_on(async move {
                    let mut guard = client.lock().expect("env client mutex poisoned");
                    guard
                        .reset(rlmesh::ResetRequest {
                            seeds: seeds.unwrap_or_default(),
                            options,
                            timeout_ms: 0,
                        })
                        .await
                })
            })
            .map_err(to_py_err)?;

        let obs =
            batched_space_values_to_py_neutral(py, &result.observations, &self.observation_space)?;
        let info = match result.info.as_ref() {
            Some(info) => meta_map_to_pydict(py, info)?,
            None => pyo3::types::PyDict::new(py),
        };
        info.set_item("episode_ids", result.episode_ids)?;
        Ok(PyTuple::new(py, [obs.as_any(), info.as_any()])?
            .into_any()
            .unbind())
    }

    fn step(&mut self, py: Python<'_>, actions: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let client = Arc::clone(&self.client);
        let action_space = self.action_space.clone();
        let obs_space = self.observation_space.clone();
        let num_envs = self.num_envs;
        let actions_ref = actions.bind(py);
        let batched_actions = py_any_to_batched_space_values_with_backend(
            py,
            actions_ref,
            &action_space,
            num_envs,
            ValueBackend::Native,
        )?;

        let runtime = &self.runtime;
        let result = py
            .detach(|| {
                runtime.block_on(async move {
                    let mut guard = client.lock().expect("env client mutex poisoned");
                    guard
                        .step(rlmesh::StepRequest {
                            actions: batched_actions,
                            timeout_ms: 0,
                        })
                        .await
                })
            })
            .map_err(to_py_err)?;

        let obs = batched_space_values_to_py_neutral(py, &result.observations, &obs_space)?;
        let rewards = vector_f64_to_py(py, &result.rewards)?;
        let terminated = vector_bool_to_py(py, &result.terminated)?;
        let truncated = vector_bool_to_py(py, &result.truncated)?;
        let info = match result.info.as_ref() {
            Some(info) => meta_map_to_pydict(py, info)?,
            None => pyo3::types::PyDict::new(py),
        };
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

    #[pyo3(signature = (env_index=0))]
    fn render(&mut self, py: Python<'_>, env_index: usize) -> PyResult<Py<PyAny>> {
        let result = self.render_message(py, env_index, "client.render.rpc")?;
        match result {
            Some(message) => Ok(decode_render_frame(py, &message)?.into_any().unbind()),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (env_index=0))]
    fn render_packet(&mut self, py: Python<'_>, env_index: usize) -> PyResult<Py<PyAny>> {
        let result = self.render_message(py, env_index, "client.render_packet.rpc")?;
        match result {
            Some(frame) => Ok(PyBytes::new(py, render_packet(&frame)).into_any().unbind()),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (env_index=0))]
    fn render_bundle(&mut self, py: Python<'_>, env_index: usize) -> PyResult<Py<PyAny>> {
        let result = self.render_message(py, env_index, "client.render_bundle.rpc")?;
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

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let client = Arc::clone(&self.client);
        let runtime = &self.runtime;
        py.detach(|| {
            runtime.block_on(async move {
                let mut guard = client.lock().expect("env client mutex poisoned");
                guard.close().await
            })
        })
        .map_err(to_py_err)?;
        self.profiler.log_summary_once();
        Ok(())
    }

    #[pyo3(signature = (reason="owner shutdown"))]
    fn shutdown(&mut self, py: Python<'_>, reason: &str) -> PyResult<bool> {
        let client = Arc::clone(&self.client);
        let reason = reason.to_string();
        let runtime = &self.runtime;
        let accepted = py
            .detach(|| {
                runtime.block_on(async move {
                    let mut guard = client.lock().expect("env client mutex poisoned");
                    guard.shutdown(reason).await
                })
            })
            .map_err(to_py_err)?;
        self.profiler.log_summary_once();
        Ok(accepted)
    }
}

impl Drop for PyVectorEnvClient {
    fn drop(&mut self) {
        self.profiler.log_summary_once();
    }
}

submit! {
    gen_methods_from_python! {
        r#"
class PyEnvClient:
    def __init__(self, address: str, *, connect_timeout_seconds: float | None = None) -> None: ...
    def address(self) -> str: ...
    def handshake(self) -> EnvContract: ...
    def observation_space(self) -> Space: ...
    def action_space(self) -> Space: ...
    def reset(self, seeds: list[int] | None = None, options: dict[str, object] | None = None) -> tuple[Value, ResetInfo]: ...
    def step(self, actions: Value) -> tuple[Value, float, bool, bool, StepInfo]: ...
    def render(self, env_index: int = 0) -> Value | None: ...
    def render_packet(self, env_index: int = 0) -> bytes | None: ...
    def render_bundle(self, env_index: int = 0) -> tuple[Value | None, bytes | None]: ...
    def close(self) -> None: ...
    def shutdown(self, reason: str = "owner shutdown") -> bool: ...
"#
    }
}

submit! {
    gen_methods_from_python! {
        r#"
class PyVectorEnvClient:
    def __init__(self, address: str, *, connect_timeout_seconds: float | None = None) -> None: ...
    def address(self) -> str: ...
    def handshake(self) -> EnvContract: ...
    def observation_space(self) -> Space: ...
    def action_space(self) -> Space: ...
    def num_envs(self) -> int: ...
    def reset(self, seeds: list[int] | None = None, options: dict[str, object] | None = None) -> tuple[Value, ResetInfo]: ...
    def step(self, actions: object) -> tuple[Value, Value, Value, Value, StepInfo]: ...
    def render(self, env_index: int = 0) -> Value | None: ...
    def render_packet(self, env_index: int = 0) -> bytes | None: ...
    def render_bundle(self, env_index: int = 0) -> tuple[Value | None, bytes | None]: ...
    def close(self) -> None: ...
    def shutdown(self, reason: str = "owner shutdown") -> bool: ...
"#
    }
}

impl PyVectorEnvClient {
    #[allow(clippy::await_holding_lock)]
    fn render_message(
        &mut self,
        py: Python<'_>,
        env_index: usize,
        rpc_phase: &'static str,
    ) -> PyResult<Option<NativeRenderFrame>> {
        let client = Arc::clone(&self.client);
        let rpc_guard = self.profiler.start(rpc_phase);
        let runtime = &self.runtime;
        let result = py
            .detach(|| {
                runtime.block_on(async move {
                    let mut guard = client.lock().expect("env client mutex poisoned");
                    guard
                        .render(NativeRenderRequest {
                            env_index: Some(env_index),
                            timeout_ms: 0,
                        })
                        .await
                })
            })
            .map_err(to_py_err)?;

        let message = result.frame;
        let frame_bytes_len = message
            .as_ref()
            .map(|frame| frame.png_frame.len())
            .unwrap_or(0);
        let _ = rpc_guard.finish(frame_bytes_len);
        Ok(message)
    }
}

impl PyEnvClient {
    #[allow(clippy::await_holding_lock)]
    fn render_message(
        &mut self,
        py: Python<'_>,
        env_index: usize,
        rpc_phase: &'static str,
    ) -> PyResult<Option<NativeRenderFrame>> {
        let client = Arc::clone(&self.client);

        let rpc_guard = self.profiler.start(rpc_phase);
        let runtime = &self.runtime;
        let result = py
            .detach(|| {
                runtime.block_on(async move {
                    let mut guard = client.lock().expect("env client mutex poisoned");
                    guard
                        .render(NativeRenderRequest {
                            env_index: Some(env_index),
                            timeout_ms: 0,
                        })
                        .await
                })
            })
            .map_err(to_py_err)?;

        let message = result.frame;
        let frame_bytes_len = message
            .as_ref()
            .map(|frame| frame.png_frame.len())
            .unwrap_or(0);
        let _ = rpc_guard.finish(frame_bytes_len);
        Ok(message)
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
    let image = image::load_from_memory_with_format(&frame.png_frame, ImageFormat::Png)
        .map_err(|err| err.to_string())?;
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(DecodedRenderFrame {
        shape: vec![height as usize, width as usize, 4],
        raw: rgba.into_raw(),
    })
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

fn observation_size(value: Option<&rlmesh_spaces::SpaceValue>) -> usize {
    value.map_or(0, space_value_size)
}

fn space_value_size(value: &rlmesh_spaces::SpaceValue) -> usize {
    match value {
        rlmesh_spaces::SpaceValue::Box(value) => value.nbytes(),
        rlmesh_spaces::SpaceValue::Discrete(_) => std::mem::size_of::<i64>(),
        rlmesh_spaces::SpaceValue::MultiBinary(values) => values.len(),
        rlmesh_spaces::SpaceValue::MultiDiscrete(values) => {
            values.len() * std::mem::size_of::<i64>()
        }
        rlmesh_spaces::SpaceValue::Text(value) => value.len(),
        rlmesh_spaces::SpaceValue::Dict(values) => values.values().map(space_value_size).sum(),
        rlmesh_spaces::SpaceValue::Tuple(values) => values.iter().map(space_value_size).sum(),
    }
}
