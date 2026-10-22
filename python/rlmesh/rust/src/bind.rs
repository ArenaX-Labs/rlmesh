//! `Model.bind(target)`: a model bound to one env, stepped by hand; and the
//! random baseline (`RANDOM_SAMPLE`) as a model a run or a binding drives.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rlmesh::binding::model::{Binding, ModelHandler, ModelObservation};
use rlmesh_spaces::SpaceValue;

use crate::model::{LastPyError, model_runtime};
use crate::spaces::{
    ValueBackend, env_contract_to_py, meta_map_to_pydict, py_any_to_meta_map,
    py_any_to_space_value_with_backend, space_value_to_py_neutral,
};
use crate::types::to_py_err;

/// The random baseline: samples the env's action space, reproducibly per
/// predict of a seeded episode.
#[derive(Default)]
pub(crate) struct RandomSample {
    /// Each live episode's predicts so far.
    predicts: HashMap<String, u64>,
}

#[async_trait]
impl ModelHandler for RandomSample {
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        let space = observation
            .env_contract
            .as_ref()
            .and_then(|contract| contract.action_space.clone())
            .ok_or_else(|| rlmesh::Error::Internal("the env has no action space".into()))?;
        observation
            .route
            .episodes
            .iter()
            .map(|episode| {
                let index = self.predicts.entry(episode.episode_id.clone()).or_default();
                let seed = match episode.seed {
                    Some(seed) => rlmesh::predict_seed(seed, *index) as u64,
                    None => uuid::Uuid::new_v4().as_u64_pair().0,
                };
                *index += 1;
                rlmesh_spaces::sample_seeded(&space, seed)
                    .map_err(|err| rlmesh::Error::Internal(err.to_string()))
            })
            .collect()
    }

    async fn reset_adapter(
        &mut self,
        _env_id: &str,
        episode_ids: Vec<String>,
    ) -> rlmesh::Result<()> {
        if episode_ids.is_empty() {
            self.predicts.clear();
        }
        for id in episode_ids {
            self.predicts.remove(&id);
        }
        Ok(())
    }
}

/// A model bound to one env lane. Values cross as neutral trees; the
/// Python `Binding` bridges them for the model's framework.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyBinding {
    inner: Arc<tokio::sync::Mutex<Option<Binding>>>,
    contract: Py<PyAny>,
    spaces: (rlmesh_spaces::SpaceSpec, rlmesh_spaces::SpaceSpec),
    /// The exception the model's predict last raised, re-raised as itself.
    last_error: LastPyError,
}

impl PyBinding {
    /// Binds `handler` to the env at `address`.
    pub(crate) fn connect(
        py: Python<'_>,
        handler: Box<dyn ModelHandler>,
        last_error: LastPyError,
        address: &str,
        execution_horizon: u32,
        workflow_edition: Option<String>,
    ) -> PyResult<Self> {
        let binding = py
            .detach(|| {
                model_runtime().block_on(Binding::connect(
                    handler,
                    address,
                    execution_horizon,
                    workflow_edition.as_deref(),
                ))
            })
            .map_err(to_py_err)?;
        let contract = binding.contract();
        let space = |space: Option<&rlmesh_spaces::SpaceSpec>| {
            space
                .cloned()
                .ok_or_else(|| PyRuntimeError::new_err("the env publishes no spaces"))
        };
        let spaces = (
            space(contract.observation_space.as_ref())?,
            space(contract.action_space.as_ref())?,
        );
        Ok(Self {
            contract: env_contract_to_py(py, contract)?,
            inner: Arc::new(tokio::sync::Mutex::new(Some(binding))),
            spaces,
            last_error,
        })
    }

    /// Runs `f` on the open binding, off the GIL. A pending signal
    /// (Ctrl-C) abandons the call and raises; an exception the model's
    /// predict raised is re-raised as itself.
    fn with<R: Send>(
        &self,
        py: Python<'_>,
        f: impl AsyncFnOnce(&mut Binding) -> rlmesh::Result<R> + Send,
    ) -> PyResult<R> {
        let inner = Arc::clone(&self.inner);
        self.last_error
            .lock()
            .expect("last_error lock poisoned")
            .take();
        let outcome = py.detach(|| {
            model_runtime().block_on(async move {
                let call = async {
                    let mut binding = inner.lock().await;
                    let binding = binding
                        .as_mut()
                        .ok_or_else(|| rlmesh::Error::Runtime("the binding is closed".into()))?;
                    f(binding).await
                };
                let mut call = std::pin::pin!(call);
                let mut poll = crate::client::signal_poll();
                loop {
                    tokio::select! {
                        result = &mut call => return Ok::<rlmesh::Result<R>, PyErr>(result),
                        _ = poll.tick() => crate::client::check_signals()?,
                    }
                }
            })
        })?;
        outcome.map_err(|err| {
            let raised = self
                .last_error
                .lock()
                .expect("last_error lock poisoned")
                .take();
            raised.unwrap_or_else(|| to_py_err(err))
        })
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PyBinding {
    #[getter]
    fn env_contract(&self, py: Python<'_>) -> Py<PyAny> {
        self.contract.clone_ref(py)
    }

    /// Starts the next episode: `(observation, info)`.
    #[pyo3(signature = (seed=None, options=None))]
    #[gen_stub(override_return_type(
        type_repr = "tuple[typing.Any, builtins.dict[builtins.str, typing.Any]]",
        imports = ("builtins", "typing")
    ))]
    fn reset<'py>(
        &self,
        py: Python<'py>,
        seed: Option<i64>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, pyo3::types::PyDict>)> {
        let options = options.map(|o| py_any_to_meta_map(&o)).transpose()?;
        let (observation, info) = self.with(py, async move |b: &mut Binding| {
            b.reset(seed, options).await
        })?;
        Ok((
            space_value_to_py_neutral(py, &observation, &self.spaces.0)?,
            match info {
                Some(info) => meta_map_to_pydict(py, &info)?,
                None => pyo3::types::PyDict::new(py),
            },
        ))
    }

    /// One step under `action` (the model's when `None`): `(observation,
    /// action, reward, terminated, truncated, info, predict_ms, step_ms)`.
    #[pyo3(signature = (action=None))]
    #[gen_stub(override_return_type(
        type_repr = "tuple[typing.Any, typing.Any, builtins.float, builtins.bool, builtins.bool, builtins.dict[builtins.str, typing.Any], builtins.float, builtins.float]",
        imports = ("builtins", "typing")
    ))]
    #[allow(clippy::type_complexity)] // the step tuple Python unpacks
    fn step<'py>(
        &self,
        py: Python<'py>,
        action: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        f64,
        bool,
        bool,
        Bound<'py, pyo3::types::PyDict>,
        f64,
        f64,
    )> {
        let action = action
            .map(|a| {
                py_any_to_space_value_with_backend(py, &a, &self.spaces.1, ValueBackend::Native)
            })
            .transpose()?;
        let step = self.with(py, async move |b: &mut Binding| b.step(action).await)?;
        Ok((
            space_value_to_py_neutral(py, &step.observation, &self.spaces.0)?,
            space_value_to_py_neutral(py, &step.action, &self.spaces.1)?,
            step.reward,
            step.terminated,
            step.truncated,
            match step.info {
                Some(info) => meta_map_to_pydict(py, &info)?,
                None => pyo3::types::PyDict::new(py),
            },
            step.predict.as_secs_f64() * 1000.0,
            step.env.as_secs_f64() * 1000.0,
        ))
    }

    /// The env's rendered frame (a uint8 HWC tensor), or `None`.
    fn render<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let frame = self.with(py, async |b: &mut Binding| b.render().await)?;
        frame
            .filter(|png| !png.is_empty())
            .map(|png| crate::client::decode_render_frame(py, &png))
            .transpose()
    }

    /// Ends the episode and releases the env (idempotent).
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        py.detach(|| {
            model_runtime().block_on(async move {
                match inner.lock().await.take() {
                    Some(binding) => binding.close().await,
                    None => Ok(()),
                }
            })
        })
        .map_err(to_py_err)
    }
}
