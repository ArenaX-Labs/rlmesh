use async_trait::async_trait;
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_methods_from_python, gen_stub_pyclass};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::inventory::submit;
use rlmesh::{
    BindAddress, ConnectAddress, Error as RLMeshError, ModelEpisodeEnd, ModelHandler,
    ModelLaneReset, ModelObservation, ModelRouteSetup, ModelWorker, RemoteModel, RunLocalOptions,
    ServeModelOptions,
};
use rlmesh_spaces::{EnvContract, SpaceValue, spaces::SpaceSpec};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::lifecycle::PyServeOptions;
use crate::spaces::{
    ValueBackend, batched_space_values_to_py_neutral, env_contract_to_py, extract_space_spec,
    make_space, py_any_to_batched_space_values_with_backend, py_any_to_meta_map,
    py_any_to_space_value_with_backend, space_value_to_py_neutral,
};
use crate::telemetry::{ProfileCollector, init_tracing};
use crate::types::to_py_err;

/// Process-wide multi-threaded runtime shared by Python model clients. The Join
/// response pump spawned during handshake lives here, so it must outlive any
/// single client; a process-wide runtime is simplest and matches the env client.
fn model_client_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build shared rlmesh model-client runtime")
    })
}

struct PyModelHandler {
    predict_fn: Py<PyAny>,
    configure_fn: Option<Py<PyAny>>,
    on_reset: Option<Py<PyAny>>,
    on_episode_end: Option<Py<PyAny>>,
    on_close: Option<Py<PyAny>>,
    profiler: Arc<ProfileCollector>,
    // route_key -> the Python adapter resolved at configure_route (a Python None
    // for a spec-less/NO_ADAPTER route). Shared with the PyRouteSetup so the
    // server resolves routes off the predict lock; predict only reads it.
    adapters: Adapters,
    // The route the server is currently processing, set by `enter_route` before
    // its per-lane resets fire; `on_lane_reset` carries only an env_index, so it
    // resolves the route's adapter through this.
    current_route: Option<String>,
}

/// route_key -> resolved Python adapter (or a Python None). Its own lock, held
/// only for the brief insert/read, so configuring a route never contends on the
/// predict-serialization lock.
type Adapters = Arc<Mutex<HashMap<String, Py<PyAny>>>>;

fn route_key(route: &rlmesh::ModelRouteContext) -> String {
    format!("{}:{}", route.session_id, route.route_id)
}

/// The per-route adapter resolver the served model exposes via
/// [`ModelHandler::route_setup`]; runs Python `resolve_from_contract` off the
/// predict lock and caches the result for `predict` to read.
struct PyRouteSetup {
    configure_fn: Py<PyAny>,
    adapters: Adapters,
}

#[async_trait]
impl ModelRouteSetup for PyRouteSetup {
    async fn configure_route(
        &self,
        route_key: &str,
        env_contract: &rlmesh::spaces::EnvContract,
    ) -> rlmesh::Result<()> {
        let configure_fn = Python::attach(|py| self.configure_fn.clone_ref(py));
        let contract = env_contract.clone();
        let adapter = tokio::task::spawn_blocking(move || {
            Python::attach(|py| -> PyResult<Py<PyAny>> {
                let contract_py = env_contract_to_py(py, &contract)?;
                configure_fn.call1(py, (contract_py,))
            })
        })
        .await
        .map_err(|err| RLMeshError::Internal(format!("configure task panicked: {err}")))?
        .map_err(|err| RLMeshError::Internal(err.to_string()))?;
        self.adapters
            .lock()
            .expect("adapters map poisoned")
            .insert(route_key.to_string(), adapter);
        Ok(())
    }

    async fn close_route(&self, route_key: &str) -> rlmesh::Result<()> {
        // Evict the route's cached adapter at CloseRoute so a long-lived served
        // model does not retain one adapter per session_id:route_id forever. The
        // map is shared with PyModelHandler, so predict/on_lane_reset stop
        // resolving this route after eviction.
        self.adapters
            .lock()
            .expect("adapters map poisoned")
            .remove(route_key);
        Ok(())
    }
}

impl PyModelHandler {
    async fn call_callback(
        callback: Option<Py<PyAny>>,
        profiler: Arc<ProfileCollector>,
        phase: &'static str,
    ) -> Result<(), RLMeshError> {
        let Some(callback) = callback else {
            return Ok(());
        };

        tokio::task::spawn_blocking(move || {
            Python::attach(|py| -> PyResult<()> {
                let guard = profiler.start(phase);
                callback.call0(py)?;
                let _ = guard.finish(0);
                Ok(())
            })
        })
        .await
        .map_err(|err| RLMeshError::Internal(format!("callback task panicked: {err}")))?
        .map_err(|err| RLMeshError::Internal(err.to_string()))
    }
}

#[async_trait]
impl ModelHandler for PyModelHandler {
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        let predict_fn = Python::attach(|py| self.predict_fn.clone_ref(py));
        // The adapter for this route was resolved at configure_route (or is a
        // Python None for a spec-less route / the never-configured run_local path).
        let adapter = self
            .adapters
            .lock()
            .expect("adapters map poisoned")
            .get(&route_key(&observation.route))
            .map(|adapter| Python::attach(|py| adapter.clone_ref(py)));
        let profiler = Arc::clone(&self.profiler);
        // Observation wire-byte volume for the predict spans (cheap: sums leaf
        // lengths, no decode/copy). Action encoding now happens in the Rust
        // worker (outside this fn), so only the obs side is measurable here; the
        // old `model.predict.encode_action` span moved with it.
        let obs_bytes_len = observation
            .observation
            .as_ref()
            .map(|leaves| leaves.iter().map(|leaf| leaf.len()).sum::<usize>())
            .unwrap_or(0);

        let predict_total_guard = profiler.start("model.predict.total");
        let actions = tokio::task::spawn_blocking(move || {
            Python::attach(|py| -> PyResult<Vec<SpaceValue>> {
                // Decode to typed lanes INSIDE the blocking task: it copies slab
                // bytes into per-lane Tensor storage, so running it on the async
                // gRPC worker would stall unrelated RPCs under a large/vector
                // observation. An absent observation yields no lanes and a Python
                // `None` (the spec-less / never-configured path).
                let lanes = if observation.observation.is_some() {
                    observation.decoded_lanes().map_err(|err| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "failed to decode model observation: {err}"
                        ))
                    })?
                } else {
                    Vec::new()
                };
                let observation_space = observation
                    .env_contract
                    .as_ref()
                    .and_then(|spec| spec.observation_space.as_ref());
                let obs = match (observation_space, lanes.len()) {
                    (_, 0) => py.None().bind(py).clone(),
                    (Some(space), 1) => space_value_to_py_neutral(py, &lanes[0], space)?,
                    (Some(space), _) => batched_space_values_to_py_neutral(py, &lanes, space)?,
                    (None, _) => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "model worker requires observation space metadata",
                        ));
                    }
                };

                let call_guard = profiler.start("model.predict.python_call");
                let adapter_arg = match adapter.as_ref() {
                    Some(adapter) => adapter.clone_ref(py),
                    None => py.None(),
                };
                let action = predict_fn.call1(py, (obs, adapter_arg))?;
                let _ = call_guard.finish(obs_bytes_len);

                let action_space = observation
                    .env_contract
                    .as_ref()
                    .and_then(|spec| spec.action_space.as_ref())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "model worker requires action space metadata",
                        )
                    })?;
                // One typed action per lane; the worker owns typed->wire encoding.
                // Single-env routes take a scalar action; vectorized routes take a
                // batched (N-stacked) action, mirroring the observation handling.
                if observation.num_envs == 1 {
                    let encoded = py_any_to_space_value_with_backend(
                        py,
                        action.bind(py),
                        action_space,
                        ValueBackend::Native,
                    )?;
                    Ok(vec![encoded])
                } else {
                    py_any_to_batched_space_values_with_backend(
                        py,
                        action.bind(py),
                        action_space,
                        observation.num_envs,
                        ValueBackend::Native,
                    )
                }
            })
        })
        .await
        .map_err(|err| RLMeshError::Internal(format!("predict task panicked: {err}")))?
        .map_err(|err| RLMeshError::Internal(err.to_string()))?;

        let _ = predict_total_guard.finish(obs_bytes_len);
        Ok(actions)
    }

    fn route_setup(&self) -> Option<Arc<dyn ModelRouteSetup>> {
        // Only a model with a configure_fn (the spec/NO_ADAPTER-aware resolver)
        // does per-route setup. The setup shares the adapters map so predict sees
        // what configure resolves, and runs off the predict lock.
        let configure_fn = self.configure_fn.as_ref()?;
        let configure_fn = Python::attach(|py| configure_fn.clone_ref(py));
        Some(Arc::new(PyRouteSetup {
            configure_fn,
            adapters: Arc::clone(&self.adapters),
        }))
    }

    async fn enter_route(&mut self, route_key: &str) -> rlmesh::Result<()> {
        self.current_route = Some(route_key.to_string());
        Ok(())
    }

    async fn on_lane_reset(&mut self, event: ModelLaneReset) -> rlmesh::Result<()> {
        // Reset only the lane whose episode rolled, mirroring the run(env) loop
        // but per-lane: a full reset() here would wipe the still-running lanes of
        // a vectorized route (see ModelHandler::on_reset). A None adapter is a
        // no-op. Single-env routes are a lane of one (env_index 0), so this still
        // fires at their initial reset and each autoreset.
        let Some(route_key) = self.current_route.as_deref() else {
            return Ok(());
        };
        let adapter = self
            .adapters
            .lock()
            .expect("adapters map poisoned")
            .get(route_key)
            .map(|adapter| Python::attach(|py| adapter.clone_ref(py)));
        if let Some(adapter) = adapter {
            let env_index = event.env_index;
            tokio::task::spawn_blocking(move || {
                Python::attach(|py| -> PyResult<()> {
                    if !adapter.is_none(py) {
                        adapter.call_method1(py, "reset", (env_index,))?;
                    }
                    Ok(())
                })
            })
            .await
            .map_err(|err| RLMeshError::Internal(format!("adapter reset task panicked: {err}")))?
            .map_err(|err| RLMeshError::Internal(err.to_string()))?;
        }
        Ok(())
    }

    async fn on_reset(&mut self, _observation: &ModelObservation) -> rlmesh::Result<()> {
        // Coarse "something reset" signal only: per-lane adapter buffers are
        // cleared in on_lane_reset, so this must NOT touch adapter state or it
        // would wipe still-running lanes on a vectorized route.
        let callback = Python::attach(|py| self.on_reset.as_ref().map(|cb| cb.clone_ref(py)));
        Self::call_callback(callback, Arc::clone(&self.profiler), "model.callback.reset").await
    }

    async fn on_episode_end(&mut self, _event: ModelEpisodeEnd) -> rlmesh::Result<()> {
        let callback = Python::attach(|py| self.on_episode_end.as_ref().map(|cb| cb.clone_ref(py)));
        Self::call_callback(
            callback,
            Arc::clone(&self.profiler),
            "model.callback.episode_end",
        )
        .await
    }

    async fn on_close(&mut self) -> rlmesh::Result<()> {
        let callback = Python::attach(|py| self.on_close.as_ref().map(|cb| cb.clone_ref(py)));
        Self::call_callback(callback, Arc::clone(&self.profiler), "model.callback.close").await
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyModel {
    predict_fn: Py<PyAny>,
    configure_fn: Option<Py<PyAny>>,
    on_reset: Option<Py<PyAny>>,
    on_episode_end: Option<Py<PyAny>>,
    on_close: Option<Py<PyAny>>,
    runtime: tokio::runtime::Runtime,
    profiler: Arc<ProfileCollector>,
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (predict_fn, configure_fn=None, on_reset=None, on_episode_end=None, on_close=None))]
    fn new(
        predict_fn: Py<PyAny>,
        configure_fn: Option<Py<PyAny>>,
        on_reset: Option<Py<PyAny>>,
        on_episode_end: Option<Py<PyAny>>,
        on_close: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        init_tracing("model_worker");
        let profiler = ProfileCollector::new("model_worker");

        let runtime = tokio::runtime::Runtime::new().map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to create tokio runtime: {err}"
            ))
        })?;

        Ok(Self {
            predict_fn,
            configure_fn,
            on_reset,
            on_episode_end,
            on_close,
            runtime,
            profiler,
        })
    }

    fn run_local(&self, py: Python<'_>, env_address: &str, token: &str) -> PyResult<()> {
        let run_span = tracing::info_span!("rlmesh.model.run_local", env_address = env_address);
        let _run_enter = run_span.enter();
        let total_guard = self.profiler.start("model.run_local.total");

        let env_address = ConnectAddress::parse(env_address).map_err(to_py_err)?;
        // `token` is retained for backward compatibility; env auth is configured
        // on the environment server.
        let _ = token;
        let profiler = Arc::clone(&self.profiler);
        let handler = Python::attach(|py| PyModelHandler {
            predict_fn: self.predict_fn.clone_ref(py),
            configure_fn: self.configure_fn.as_ref().map(|cb| cb.clone_ref(py)),
            on_reset: self.on_reset.as_ref().map(|cb| cb.clone_ref(py)),
            on_episode_end: self.on_episode_end.as_ref().map(|cb| cb.clone_ref(py)),
            on_close: self.on_close.as_ref().map(|cb| cb.clone_ref(py)),
            profiler: Arc::clone(&profiler),
            adapters: Arc::new(Mutex::new(HashMap::new())),
            current_route: None,
        });

        py.detach(|| {
            self.runtime.block_on(async move {
                ModelWorker::new(handler)
                    .run_local_async(RunLocalOptions::new(env_address))
                    .await
            })
        })
        .map_err(to_py_err)?;

        let _ = total_guard.finish(0);
        self.profiler.log_summary_once();
        Ok(())
    }

    fn run_local_for_episodes(
        &self,
        py: Python<'_>,
        env_address: &str,
        token: &str,
        max_episodes: u64,
    ) -> PyResult<()> {
        let run_span = tracing::info_span!(
            "rlmesh.model.run_local_for_episodes",
            env_address = env_address,
            max_episodes
        );
        let _run_enter = run_span.enter();
        let total_guard = self.profiler.start("model.run_local.total");

        let env_address = ConnectAddress::parse(env_address).map_err(to_py_err)?;
        // `token` retained for backward compatibility; run_local does not
        // authenticate against the env (see `run_local`).
        let _ = token;
        let profiler = Arc::clone(&self.profiler);
        let handler = Python::attach(|py| PyModelHandler {
            predict_fn: self.predict_fn.clone_ref(py),
            configure_fn: self.configure_fn.as_ref().map(|cb| cb.clone_ref(py)),
            on_reset: self.on_reset.as_ref().map(|cb| cb.clone_ref(py)),
            on_episode_end: self.on_episode_end.as_ref().map(|cb| cb.clone_ref(py)),
            on_close: self.on_close.as_ref().map(|cb| cb.clone_ref(py)),
            profiler: Arc::clone(&profiler),
            adapters: Arc::new(Mutex::new(HashMap::new())),
            current_route: None,
        });

        py.detach(|| {
            self.runtime.block_on(async move {
                ModelWorker::new(handler)
                    .run_local_async(RunLocalOptions::new(env_address).for_episodes(max_episodes))
                    .await
            })
        })
        .map_err(to_py_err)?;

        let _ = total_guard.finish(0);
        self.profiler.log_summary_once();
        Ok(())
    }

    #[pyo3(signature = (address, token, options=None))]
    fn serve(
        &self,
        py: Python<'_>,
        address: &str,
        token: &str,
        options: Option<PyServeOptions>,
    ) -> PyResult<()> {
        let run_span = tracing::info_span!("rlmesh.model.serve", address = address);
        let _run_enter = run_span.enter();
        let total_guard = self.profiler.start("model.serve.total");

        let address = BindAddress::parse(address).map_err(to_py_err)?;
        let token = token.to_string();
        let options = options.map(PyServeOptions::into_rust).unwrap_or_default();
        let profiler = Arc::clone(&self.profiler);
        let handler = Python::attach(|py| PyModelHandler {
            predict_fn: self.predict_fn.clone_ref(py),
            configure_fn: self.configure_fn.as_ref().map(|cb| cb.clone_ref(py)),
            on_reset: self.on_reset.as_ref().map(|cb| cb.clone_ref(py)),
            on_episode_end: self.on_episode_end.as_ref().map(|cb| cb.clone_ref(py)),
            on_close: self.on_close.as_ref().map(|cb| cb.clone_ref(py)),
            profiler: Arc::clone(&profiler),
            adapters: Arc::new(Mutex::new(HashMap::new())),
            current_route: None,
        });

        py.detach(|| {
            self.runtime.block_on(async move {
                ModelWorker::new(handler)
                    .serve_async(
                        ServeModelOptions::new(address)
                            .token(token)
                            .serve_options(options),
                    )
                    .await
            })
        })
        .map_err(to_py_err)?;

        let _ = total_guard.finish(0);
        self.profiler.log_summary_once();
        Ok(())
    }
}

#[cfg(feature = "stub-gen")]
submit! {
    gen_methods_from_python! {
        r#"
import collections.abc

class PyModel:
    def __init__(self, predict_fn: collections.abc.Callable[[Value, object], Value], configure_fn: collections.abc.Callable[[EnvContract], object] | None = None, on_reset: collections.abc.Callable[[], None] | None = None, on_episode_end: collections.abc.Callable[[], None] | None = None, on_close: collections.abc.Callable[[], None] | None = None) -> None: ...
    def run_local(self, env_address: str, token: str) -> None: ...
    def run_local_for_episodes(self, env_address: str, token: str, max_episodes: int) -> None: ...
    def serve(self, address: str, token: str, options: ServeOptions | None = None) -> None: ...
"#
    }
}

#[cfg(feature = "stub-gen")]
submit! {
    gen_methods_from_python! {
        r#"
class PyModelClient:
    def __init__(self, address: str, env_contract: EnvContract, token: str = "") -> None: ...
    def address(self) -> str: ...
    def observation_space(self) -> Space: ...
    def action_space(self) -> Space: ...
    def reset(self) -> None: ...
    def predict(self, observation: Value) -> Value: ...
    def close(self) -> None: ...
"#
    }
}

impl Drop for PyModel {
    fn drop(&mut self) {
        self.profiler.log_summary_once();
    }
}

/// Client handle to a model (policy) server: the model-side mirror of
/// `PyEnvClient`. Bound to one env contract (one route) for its lifetime; the
/// Python layer creates one per `RemoteModel.against(env)`.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyModelClient {
    inner: RemoteModel,
    runtime: &'static tokio::runtime::Runtime,
    observation_space: SpaceSpec,
    action_space: SpaceSpec,
    address: String,
}

#[pymethods]
impl PyModelClient {
    #[new]
    #[pyo3(signature = (address, env_contract, token=""))]
    fn new(
        py: Python<'_>,
        address: &str,
        env_contract: &Bound<'_, PyAny>,
        token: &str,
    ) -> PyResult<Self> {
        init_tracing("model_client");
        let contract = native_env_contract_from_py(env_contract)?;
        let observation_space = contract.observation_space.clone().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("env contract missing observation_space")
        })?;
        let action_space = contract.action_space.clone().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("env contract missing action_space")
        })?;
        let runtime = model_client_runtime();
        let address = address.to_string();
        let token = token.to_string();
        let inner = py
            .detach(|| {
                runtime.block_on(RemoteModel::connect_with_token(&address, &token, contract))
            })
            .map_err(to_py_err)?;
        let address = inner.address().to_string();
        Ok(Self {
            inner,
            runtime,
            observation_space,
            action_space,
            address,
        })
    }

    fn address(&self) -> String {
        self.address.clone()
    }

    fn observation_space(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(make_space(py, &self.observation_space)?.into_any().unbind())
    }

    fn action_space(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(make_space(py, &self.action_space)?.into_any().unbind())
    }

    /// Begin a new episode (next predict marks a reset boundary).
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn predict(&mut self, py: Python<'_>, observation: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let value = py_any_to_space_value_with_backend(
            py,
            observation.bind(py),
            &self.observation_space,
            ValueBackend::Native,
        )?;
        let runtime = self.runtime;
        let inner = &mut self.inner;
        let action = py
            .detach(|| runtime.block_on(inner.predict(value)))
            .map_err(to_py_err)?;
        Ok(space_value_to_py_neutral(py, &action, &self.action_space)?.unbind())
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let runtime = self.runtime;
        let inner = &mut self.inner;
        py.detach(|| runtime.block_on(inner.close()))
            .map_err(to_py_err)
    }
}

/// Reconstruct a native `EnvContract` from a Python contract object (the value
/// `RemoteEnv.env_contract` returns). Duck-typed via getattr because the pyclass
/// `PyEnvContract` is `skip_from_py_object` and its native `inner` cannot be
/// extracted directly. Carries `metadata` (the env's adapter tags) so the
/// served model can resolve its adapter from the route's contract.
fn native_env_contract_from_py(contract: &Bound<'_, PyAny>) -> PyResult<EnvContract> {
    let id: String = contract.getattr("id")?.extract()?;
    let num_envs: u32 = contract.getattr("num_envs")?.extract()?;
    let render_mode: String = contract
        .getattr("render_mode")?
        .extract::<Option<String>>()?
        .unwrap_or_default();
    let observation_space = extract_space_spec(&contract.getattr("observation_space")?)
        .ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err(
                "env contract observation_space is not an RLMesh space spec",
            )
        })?;
    let action_space = extract_space_spec(&contract.getattr("action_space")?).ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err(
            "env contract action_space is not an RLMesh space spec",
        )
    })?;
    let metadata_obj = contract.getattr("metadata")?;
    let metadata = if metadata_obj.is_none() {
        None
    } else {
        Some(py_any_to_meta_map(&metadata_obj)?)
    };
    Ok(EnvContract {
        id,
        action_space: Some(action_space),
        observation_space: Some(observation_space),
        metadata,
        render_mode,
        num_envs,
        // The driving loop is user-owned and single-env; autoreset mode does not
        // affect the client route, so the contract default is fine here.
        autoreset_mode: Default::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::{Adapters, PyRouteSetup};
    use pyo3::Python;
    use rlmesh::ModelRouteSetup;

    #[test]
    fn close_route_evicts_the_route_adapter() {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};

        Python::attach(|py| {
            let adapters: Adapters = Arc::new(Mutex::new(HashMap::new()));
            adapters
                .lock()
                .unwrap()
                .insert("session:route".to_string(), py.None());
            // close_route never touches configure_fn; any callable stands in.
            let configure_fn = py.eval(c"lambda *a: None", None, None).unwrap().unbind();
            let setup = PyRouteSetup {
                configure_fn,
                adapters: Arc::clone(&adapters),
            };

            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            py.detach(|| runtime.block_on(setup.close_route("session:route")))
                .unwrap();

            assert!(adapters.lock().unwrap().is_empty());
        });
    }
}
