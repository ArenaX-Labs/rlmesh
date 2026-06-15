//! [`PyModelClient`]: the Python client side of a served model — the inverse of
//! [`PyEnvClient`](crate::client). It drives a `ModelService` endpoint (e.g. a
//! fast C/C++ model served via `rlmesh_model_serve`, or a Python model served
//! with `Model.serve`) so an eval loop can stay in Python while the model runs as
//! a served binary. Thin wrapper over [`rlmesh::RemoteModel`]: Python values in,
//! Python values out; the facade hides all wire/route/codec plumbing.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::gen_stub_pyclass;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::inventory::submit;
use rlmesh::{EnvContract, RemoteModel};
use rlmesh_spaces::SpaceSpec;

use crate::spaces::{
    ValueBackend, parse_space, py_any_to_space_value_with_backend, space_value_to_py_neutral,
};
use crate::types::to_py_err;

/// A client handle to a served model (single env / single route).
///
/// Construct with the env's observation/action spaces (the model needs them to
/// configure its route and frame values), then call
/// [`predict`](PyModelClient::predict) per step and
/// [`begin_episode`](PyModelClient::begin_episode) at each episode boundary.
/// Values cross via the dependency-free native backend, matching
/// `rlmesh.RemoteEnv`.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyModelClient {
    inner: RemoteModel,
    runtime: tokio::runtime::Runtime,
    observation_space: SpaceSpec,
    action_space: SpaceSpec,
}

#[pymethods]
impl PyModelClient {
    #[new]
    #[pyo3(signature = (address, observation_space, action_space, *, token=None))]
    fn new(
        py: Python<'_>,
        address: &str,
        observation_space: &Bound<'_, PyAny>,
        action_space: &Bound<'_, PyAny>,
        token: Option<&str>,
    ) -> PyResult<Self> {
        let observation_space = parse_space(observation_space)?;
        let action_space = parse_space(action_space)?;
        let contract = EnvContract {
            observation_space: Some(observation_space.clone()),
            action_space: Some(action_space.clone()),
            num_envs: 1,
            ..Default::default()
        };
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|err| PyRuntimeError::new_err(format!("failed to build runtime: {err}")))?;
        let address = address.to_string();
        let token = token.unwrap_or_default().to_string();
        let inner = py
            .detach(|| {
                runtime.block_on(RemoteModel::connect_with_token(&address, &token, contract))
            })
            .map_err(to_py_err)?;
        Ok(Self {
            inner,
            runtime,
            observation_space,
            action_space,
        })
    }

    /// Map one observation to the model's action.
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        observation: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let value = py_any_to_space_value_with_backend(
            py,
            observation,
            &self.observation_space,
            ValueBackend::Native,
        )?;
        let action = py
            .detach(|| self.runtime.block_on(self.inner.predict(value)))
            .map_err(to_py_err)?;
        space_value_to_py_neutral(py, &action, &self.action_space)
    }

    /// Mark the next [`predict`](PyModelClient::predict) as a new episode's first
    /// step (sets the served model's reset flag).
    fn begin_episode(&mut self) {
        self.inner.begin_episode();
    }

    /// Close this client session (does not stop the server).
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.runtime.block_on(self.inner.close()))
            .map_err(to_py_err)
    }

    /// Ask the server to shut down (honored only if it enabled remote shutdown).
    #[pyo3(signature = (reason="owner shutdown"))]
    fn shutdown(&mut self, py: Python<'_>, reason: &str) -> PyResult<bool> {
        py.detach(|| {
            self.runtime
                .block_on(self.inner.shutdown(reason.to_string()))
        })
        .map_err(to_py_err)
    }
}

#[cfg(feature = "stub-gen")]
submit! {
    pyo3_stub_gen::derive::gen_methods_from_python! {
        r#"
class PyModelClient:
    def __init__(self, address: str, observation_space: Space, action_space: Space, *, token: str | None = None) -> None: ...
    def predict(self, observation: Value) -> Value: ...
    def begin_episode(self) -> None: ...
    def close(self) -> None: ...
    def shutdown(self, reason: str = "owner shutdown") -> bool: ...
"#
    }
}
