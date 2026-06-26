use async_trait::async_trait;
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_methods_from_python, gen_stub_pyclass};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::inventory::submit;
use rlmesh::{
    AdaptedModelHandler, BindAddress, ConnectAddress, Error as RLMeshError, ModelObservation,
    ModelWorker, PredictFn, RemoteModel, RouteConfig, RouteResolver, RunLocalOptions,
    ServeModelOptions,
};
use rlmesh_adapters::v1::Value;
use rlmesh_spaces::{EnvContract, SpaceValue, spaces::SpaceSpec};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::adapters::{PyCustomTransform, PyEncodings, decode_value, encode_value};
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

/// The model's predict hole, backed by the Python predict callable and its
/// discovered lifecycle callbacks. The engine ([`AdaptedModelHandler`]) calls
/// these back from a blocking worker thread; the framework-bridge (numpy/torch)
/// round-trip lives in the Python `predict_neutral` callable, so this layer only
/// converts between the adapter [`Value`] model and the neutral Python tree.
struct PyPredict {
    predict_fn: Py<PyAny>,
    on_reset: Option<Py<PyAny>>,
    on_episode_end: Option<Py<PyAny>>,
    on_close: Option<Py<PyAny>>,
}

impl PyPredict {
    fn fire(callback: &Option<Py<PyAny>>) -> rlmesh::Result<()> {
        let Some(callback) = callback else {
            return Ok(());
        };
        Python::attach(|py| callback.call0(py))
            .map(|_| ())
            .map_err(|err| RLMeshError::Internal(err.to_string()))
    }
}

impl PredictFn for PyPredict {
    fn predict(&self, model_input: BTreeMap<String, Value>) -> rlmesh::Result<Value> {
        Python::attach(|py| -> PyResult<Value> {
            let input = pyo3::types::PyDict::new(py);
            for (key, value) in &model_input {
                input.set_item(key, encode_value(py, value)?)?;
            }
            let action = self.predict_fn.call1(py, (input,))?;
            decode_value(action.bind(py))
        })
        .map_err(|err| RLMeshError::Internal(err.to_string()))
    }

    fn predict_spec_less(&self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        // A spec-less route hands the raw observation straight to the model,
        // batched, preserving the pre-relocation path exactly (no adapter).
        let lanes = if observation.observation.is_some() {
            observation
                .decoded_lanes()
                .map_err(|err| RLMeshError::Internal(err.to_string()))?
        } else {
            Vec::new()
        };
        Python::attach(|py| -> PyResult<Vec<SpaceValue>> {
            let observation_space = observation
                .env_contract
                .as_ref()
                .and_then(|contract| contract.observation_space.as_ref());
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
            let action = self.predict_fn.call1(py, (obs,))?;
            let action_space = observation
                .env_contract
                .as_ref()
                .and_then(|contract| contract.action_space.as_ref())
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(
                        "model worker requires action space metadata",
                    )
                })?;
            if observation.num_envs == 1 {
                Ok(vec![py_any_to_space_value_with_backend(
                    py,
                    action.bind(py),
                    action_space,
                    ValueBackend::Native,
                )?])
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
        .map_err(|err| RLMeshError::Internal(err.to_string()))
    }

    fn on_reset(&self) -> rlmesh::Result<()> {
        Self::fire(&self.on_reset)
    }

    fn on_episode_end(&self) -> rlmesh::Result<()> {
        Self::fire(&self.on_episode_end)
    }

    fn on_close(&self) -> rlmesh::Result<()> {
        Self::fire(&self.on_close)
    }
}

/// The per-route resolver the served model exposes via `route_setup`. Runs the
/// Python `configure_fn` (which resolves the spec into a native plan + neutral
/// host holes) off the predict lock and hands the engine a [`RouteConfig`];
/// `None` is a spec-less / `NO_ADAPTER` route.
struct PyRouteResolver {
    configure_fn: Py<PyAny>,
}

#[async_trait]
impl RouteResolver for PyRouteResolver {
    async fn resolve(
        &self,
        _route_key: &str,
        env_contract: &EnvContract,
    ) -> rlmesh::Result<Option<RouteConfig>> {
        let configure_fn = Python::attach(|py| self.configure_fn.clone_ref(py));
        let contract = env_contract.clone();
        let observation_space = contract.observation_space.clone();
        let action_space = contract.action_space.clone();
        tokio::task::spawn_blocking(move || {
            Python::attach(|py| -> PyResult<Option<RouteConfig>> {
                let contract_py = env_contract_to_py(py, &contract)?;
                let resolved = configure_fn.call1(py, (contract_py,))?;
                let resolved = resolved.bind(py);
                if resolved.is_none() {
                    return Ok(None);
                }
                let served = resolved.cast::<pyo3::types::PyDict>()?;
                let plan_obj = served.get_item("plan")?.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("served route is missing 'plan'")
                })?;
                let plan = plan_obj
                    .cast::<crate::adapters::PyAdapterPlan>()
                    .map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(
                            "served route 'plan' is not an AdapterPlan",
                        )
                    })?;
                let adapter = plan.borrow().adapter().clone();

                let mut customs: HashMap<String, Py<PyAny>> = HashMap::new();
                if let Some(customs_obj) = served.get_item("customs")? {
                    for (key, value) in customs_obj.cast::<pyo3::types::PyDict>()?.iter() {
                        customs.insert(key.extract()?, value.unbind());
                    }
                }
                let obs_encodings = optional_callable(served.get_item("obs_encodings")?);
                let action_encodings = optional_callable(served.get_item("action_encodings")?);

                let observation_space = observation_space.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "env contract missing observation_space",
                    )
                })?;
                let action_space = action_space.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("env contract missing action_space")
                })?;

                Ok(Some(RouteConfig::new(
                    adapter,
                    observation_space,
                    action_space,
                    Box::new(PyCustomTransform::new(customs)),
                    Box::new(PyEncodings::new(obs_encodings, action_encodings)),
                )))
            })
        })
        .await
        .map_err(|err| RLMeshError::Internal(format!("configure task panicked: {err}")))?
        .map_err(|err| RLMeshError::Internal(err.to_string()))
    }
}

/// A present, non-`None` Python value as an owned callable handle.
fn optional_callable(value: Option<Bound<'_, PyAny>>) -> Option<Py<PyAny>> {
    match value {
        Some(value) if !value.is_none() => Some(value.unbind()),
        _ => None,
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

impl PyModel {
    /// Build the served engine handler: the predict hole plus, for a spec'd
    /// model, the route resolver. The engine owns the per-route adapter state and
    /// frame buffers; this layer only supplies the host (Python) callbacks.
    fn build_handler(&self) -> AdaptedModelHandler {
        let predict = Python::attach(|py| PyPredict {
            predict_fn: self.predict_fn.clone_ref(py),
            on_reset: self.on_reset.as_ref().map(|cb| cb.clone_ref(py)),
            on_episode_end: self.on_episode_end.as_ref().map(|cb| cb.clone_ref(py)),
            on_close: self.on_close.as_ref().map(|cb| cb.clone_ref(py)),
        });
        let resolver: Option<Arc<dyn RouteResolver>> =
            self.configure_fn.as_ref().map(|configure| {
                let configure_fn = Python::attach(|py| configure.clone_ref(py));
                Arc::new(PyRouteResolver { configure_fn }) as Arc<dyn RouteResolver>
            });
        AdaptedModelHandler::new(Arc::new(predict), resolver)
    }
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
        let handler = self.build_handler();

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
        let handler = self.build_handler();

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
        let handler = self.build_handler();

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
    def __init__(self, predict_fn: collections.abc.Callable[[Value], Value], configure_fn: collections.abc.Callable[[EnvContract], object] | None = None, on_reset: collections.abc.Callable[[], None] | None = None, on_episode_end: collections.abc.Callable[[], None] | None = None, on_close: collections.abc.Callable[[], None] | None = None) -> None: ...
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
/// Python layer creates one per `rlmesh.session(model, env)`.
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
