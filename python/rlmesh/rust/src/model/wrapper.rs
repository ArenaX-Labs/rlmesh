use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_methods_from_python, gen_stub_pyclass};
use pyo3_stub_gen::inventory::submit;
use rlmesh::spaces::BinaryPayload;
use rlmesh::{
    BindAddress, ConnectAddress, Error as RLMeshError, ModelEpisodeEnd, ModelHandler,
    ModelObservation, ModelWorker,
};
use rlmesh_grpc::wire::{binary_to_bytes, decode_batched_partial_values};
use rlmesh_spaces::{
    SpaceValue,
    spaces::{SpaceKind, SpaceSpec},
};
use std::sync::Arc;

use crate::lifecycle::PyServeOptions;
use crate::spaces::{
    ValueBackend, batched_space_values_to_py_neutral, encode_i64_sequence_bytes,
    py_any_to_space_value_with_backend, space_value_to_py_neutral,
};
use crate::telemetry::{ProfileCollector, init_tracing};
use crate::types::errors::to_py_err;

struct PyModelHandler {
    predict_fn: Py<PyAny>,
    on_reset: Option<Py<PyAny>>,
    on_episode_end: Option<Py<PyAny>>,
    on_close: Option<Py<PyAny>>,
    profiler: Arc<ProfileCollector>,
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
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<BinaryPayload> {
        let predict_fn = Python::attach(|py| self.predict_fn.clone_ref(py));
        let profiler = Arc::clone(&self.profiler);
        let observation_payload = observation.observation.clone();
        let env_contract = observation.env_contract.clone();
        let obs_bytes_len = observation_payload
            .as_ref()
            .map(|payload| payload.data.len())
            .unwrap_or(0);

        let predict_total_guard = profiler.start("model.predict.total");
        let action_bytes = tokio::task::spawn_blocking(move || {
            Python::attach(|py| -> PyResult<Vec<u8>> {
                let observation_space = env_contract
                    .as_ref()
                    .and_then(|spec| spec.observation_space.as_ref());
                let obs = neutral_observation(py, observation_payload.as_ref(), observation_space)?;

                let call_guard = profiler.start("model.predict.python_call");
                let action = predict_fn.call1(py, (obs,))?;
                let _ = call_guard.finish(obs_bytes_len);

                let encode_guard = profiler.start("model.predict.encode_action");
                let action_space = env_contract
                    .as_ref()
                    .and_then(|spec| spec.action_space.as_ref())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "model worker requires action space metadata",
                        )
                    })?;
                let encoded = py_any_to_space_value_with_backend(
                    py,
                    action.bind(py),
                    action_space,
                    ValueBackend::Native,
                )?;
                let bytes = space_value_to_raw_bytes(&encoded, action_space)?;
                let _ = encode_guard.finish(bytes.len());
                Ok(bytes)
            })
        })
        .await
        .map_err(|err| RLMeshError::Internal(format!("predict task panicked: {err}")))?
        .map_err(|err| RLMeshError::Internal(err.to_string()))?;

        let action_bytes_len = action_bytes.len();
        let _ = predict_total_guard.finish(obs_bytes_len + action_bytes_len);

        Ok(BinaryPayload { data: action_bytes })
    }

    async fn on_reset(&mut self, _observation: &ModelObservation) -> rlmesh::Result<()> {
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

#[gen_stub_pyclass]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyModel {
    predict_fn: Py<PyAny>,
    on_reset: Option<Py<PyAny>>,
    on_episode_end: Option<Py<PyAny>>,
    on_close: Option<Py<PyAny>>,
    runtime: tokio::runtime::Runtime,
    profiler: Arc<ProfileCollector>,
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (predict_fn, on_reset=None, on_episode_end=None, on_close=None))]
    fn new(
        predict_fn: Py<PyAny>,
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
        let token = token.to_string();
        let profiler = Arc::clone(&self.profiler);
        let handler = Python::attach(|py| PyModelHandler {
            predict_fn: self.predict_fn.clone_ref(py),
            on_reset: self.on_reset.as_ref().map(|cb| cb.clone_ref(py)),
            on_episode_end: self.on_episode_end.as_ref().map(|cb| cb.clone_ref(py)),
            on_close: self.on_close.as_ref().map(|cb| cb.clone_ref(py)),
            profiler: Arc::clone(&profiler),
        });

        py.detach(|| {
            self.runtime.block_on(async move {
                ModelWorker::new(handler)
                    .run_local_to_async(env_address, &token)
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
        let token = token.to_string();
        let profiler = Arc::clone(&self.profiler);
        let handler = Python::attach(|py| PyModelHandler {
            predict_fn: self.predict_fn.clone_ref(py),
            on_reset: self.on_reset.as_ref().map(|cb| cb.clone_ref(py)),
            on_episode_end: self.on_episode_end.as_ref().map(|cb| cb.clone_ref(py)),
            on_close: self.on_close.as_ref().map(|cb| cb.clone_ref(py)),
            profiler: Arc::clone(&profiler),
        });

        py.detach(|| {
            self.runtime.block_on(async move {
                ModelWorker::new(handler)
                    .run_local_to_async_for_episodes(env_address, &token, max_episodes)
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
        let options = options.map(PyServeOptions::to_rust).unwrap_or_default();
        let profiler = Arc::clone(&self.profiler);
        let handler = Python::attach(|py| PyModelHandler {
            predict_fn: self.predict_fn.clone_ref(py),
            on_reset: self.on_reset.as_ref().map(|cb| cb.clone_ref(py)),
            on_episode_end: self.on_episode_end.as_ref().map(|cb| cb.clone_ref(py)),
            on_close: self.on_close.as_ref().map(|cb| cb.clone_ref(py)),
            profiler: Arc::clone(&profiler),
        });

        py.detach(|| {
            self.runtime.block_on(async move {
                ModelWorker::new(handler)
                    .serve_to_async_with_options(address, &token, options)
                    .await
            })
        })
        .map_err(to_py_err)?;

        let _ = total_guard.finish(0);
        self.profiler.log_summary_once();
        Ok(())
    }
}

submit! {
    gen_methods_from_python! {
        r#"
import collections.abc

class PyModel:
    def __init__(self, predict_fn: collections.abc.Callable[[Value], Value], on_reset: collections.abc.Callable[[], None] | None = None, on_episode_end: collections.abc.Callable[[], None] | None = None, on_close: collections.abc.Callable[[], None] | None = None) -> None: ...
    def run_local(self, env_address: str, token: str) -> None: ...
    def run_local_for_episodes(self, env_address: str, token: str, max_episodes: int) -> None: ...
    def serve(self, address: str, token: str, options: ServeOptions | None = None) -> None: ...
"#
    }
}

impl Drop for PyModel {
    fn drop(&mut self) {
        self.profiler.log_summary_once();
    }
}

fn neutral_observation<'py>(
    py: Python<'py>,
    payload: Option<&BinaryPayload>,
    observation_space: Option<&SpaceSpec>,
) -> PyResult<Bound<'py, PyAny>> {
    let Some(payload) = payload else {
        return Ok(py.None().bind(py).clone());
    };

    let Some(observation_space) = observation_space else {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "model worker requires observation space metadata",
        ));
    };

    let payload = binary_to_bytes(payload);
    let values =
        decode_batched_partial_values(Some(&payload), observation_space).map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "failed to decode model observation payload against observation space: {err}"
            ))
        })?;
    if values.len() == 1 {
        space_value_to_py_neutral(py, &values[0], observation_space)
    } else {
        batched_space_values_to_py_neutral(py, &values, observation_space)
    }
}

fn space_value_to_raw_bytes(value: &SpaceValue, space: &SpaceSpec) -> PyResult<Vec<u8>> {
    match (space.spec.as_ref(), value) {
        (Some(SpaceKind::Box(_)), SpaceValue::Box(value)) => {
            Ok(value.to_contiguous_bytes().into_owned())
        }
        (Some(SpaceKind::Discrete(_)), SpaceValue::Discrete(value)) => {
            Ok(value.to_le_bytes().to_vec())
        }
        (Some(SpaceKind::MultiBinary(_)), SpaceValue::MultiBinary(values)) => {
            Ok(values.iter().map(|value| u8::from(*value)).collect())
        }
        (Some(SpaceKind::MultiDiscrete(_)), SpaceValue::MultiDiscrete(values)) => {
            encode_i64_sequence_bytes(values, space.dtype)
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "model worker only supports array-like action spaces",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::neutral_observation;
    use pyo3::Python;
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
    use rlmesh::spaces::BinaryPayload;
    use rlmesh_grpc::wire::encode_batched_partial_values;
    use rlmesh_spaces::SpaceValue;
    use rlmesh_spaces::spaces::{DictSpaceBuilder, TextBuilder};

    fn instruction_space() -> rlmesh_spaces::SpaceSpec {
        DictSpaceBuilder::new()
            .insert("instruction", TextBuilder::new(32).build().unwrap())
            .build()
            .unwrap()
    }

    fn instruction_value(value: &str) -> SpaceValue {
        SpaceValue::Dict(
            [(
                "instruction".to_string(),
                SpaceValue::Text(value.to_string()),
            )]
            .into_iter()
            .collect(),
        )
    }

    #[test]
    fn neutral_observation_decodes_one_lane_batched_partial_payload_as_single_value() {
        Python::attach(|py| {
            let space = instruction_space();
            let payload =
                encode_batched_partial_values(&[instruction_value("pick cup")], &space).unwrap();
            let payload = BinaryPayload { data: payload.data };

            let observation = neutral_observation(py, Some(&payload), Some(&space)).unwrap();
            let observation = observation.cast::<PyDict>().unwrap();

            assert_eq!(
                observation
                    .get_item("instruction")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "pick cup"
            );
        });
    }

    #[test]
    fn neutral_observation_decodes_multi_lane_payload_as_batched_value() {
        Python::attach(|py| {
            let space = instruction_space();
            let payload = encode_batched_partial_values(
                &[
                    instruction_value("pick cup"),
                    instruction_value("open drawer"),
                ],
                &space,
            )
            .unwrap();
            let payload = BinaryPayload { data: payload.data };

            let observation = neutral_observation(py, Some(&payload), Some(&space)).unwrap();
            let observation = observation.cast::<PyDict>().unwrap();
            let instructions = observation.get_item("instruction").unwrap().unwrap();

            assert_eq!(instructions.len().unwrap(), 2);
            assert_eq!(
                instructions
                    .get_item(1)
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "open drawer"
            );
        });
    }
}
