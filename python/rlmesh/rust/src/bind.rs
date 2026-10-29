//! Python values over the shared facade's manual binding.
use crate::facade::{PyFacadeHooks, value_to_py};
use crate::model::LastPyError;
use crate::spaces::{
    ValueBackend, env_contract_to_py, py_any_to_meta_map, py_any_to_space_value_with_backend,
};
use crate::types::to_py_err;
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::{Arc, Mutex};

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyBinding {
    inner: Mutex<Option<rlmesh::Binding<'static>>>,
    contract: Py<PyAny>,
    action_space: rlmesh_spaces::SpaceSpec,
    last_error: LastPyError,
    done: Arc<std::sync::atomic::AtomicBool>,
    render: Arc<dyn Fn() -> rlmesh::Result<rlmesh::Value> + Send + Sync>,
}
impl PyBinding {
    pub(crate) fn connect(
        py: Python<'_>,
        model: rlmesh::binding::model::PreparedModel,
        last_error: LastPyError,
        address: &str,
        execution_horizon: u32,
        workflow_edition: Option<String>,
    ) -> PyResult<Self> {
        let binding = py
            .detach(|| {
                rlmesh::binding::model::bind_prepared(
                    model,
                    address.to_owned().into(),
                    rlmesh::BindOptions {
                        execution_horizon: Some(execution_horizon),
                        workflow_edition,
                        ..Default::default()
                    },
                )
            })
            .map_err(to_py_err)?;
        Self::from_binding(py, binding, last_error)
    }
    pub(crate) fn from_binding(
        py: Python<'_>,
        binding: rlmesh::Binding<'static>,
        last_error: LastPyError,
    ) -> PyResult<Self> {
        let contract =
            rlmesh::binding::env_contract_to_wire(binding.contract()).map_err(to_py_err)?;
        let action_space = contract.action_space.clone().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("the env publishes no action space")
        })?;
        let done = rlmesh::binding::model::binding_done_flag(&binding);
        let render = rlmesh::binding::model::binding_renderer(&binding).map_err(to_py_err)?;
        Ok(Self {
            done,
            render,
            contract: env_contract_to_py(py, &contract)?,
            action_space,
            inner: Mutex::new(Some(binding)),
            last_error,
        })
    }
    fn with<R: Send>(
        &self,
        py: Python<'_>,
        call: impl FnOnce(&mut rlmesh::Binding<'static>) -> rlmesh::Result<R> + Send,
    ) -> PyResult<R> {
        self.last_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let outcome = py.detach(|| {
            let mut guard = self.inner.try_lock().map_err(|_| {
                rlmesh::Error::Runtime("a binding operation is already in progress".into())
            })?;
            let binding = guard
                .as_mut()
                .ok_or_else(|| rlmesh::Error::Runtime("the binding is closed".into()))?;
            let cancel = rlmesh::binding::model::binding_cancellation(binding);
            let poll = crate::facade::signal_poll(self.last_error.clone(), move || cancel.cancel());
            rlmesh::binding::model::with_caller_poll(poll, || call(binding))
        });
        outcome.map_err(|err| {
            self.last_error
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .unwrap_or_else(|| to_py_err(err))
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
    #[getter]
    fn done(&self) -> bool {
        self.done.load(std::sync::atomic::Ordering::Acquire)
    }
    fn observe(&self, py: Python<'_>, relay: Py<PyAny>) -> PyResult<()> {
        let hooks = PyFacadeHooks::new(relay, self.last_error.clone());
        self.with(py, |b| {
            rlmesh::binding::model::observe_binding(b, Box::new(hooks.clone()), Box::new(hooks))
        })
    }
    #[pyo3(signature = (seed=None, options=None))]
    #[gen_stub(override_return_type(type_repr = "tuple[typing.Any, builtins.dict[builtins.str, typing.Any]]", imports = ("builtins", "typing")))]
    fn reset<'py>(
        &self,
        py: Python<'py>,
        seed: Option<i64>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, pyo3::types::PyDict>)> {
        let options = options
            .map(|v| py_any_to_meta_map(&v).map(rlmesh::binding::metadata_from_wire))
            .transpose()?;
        let (obs, info) = self.with(py, |b| b.reset(seed, options.as_ref()))?;
        Ok((
            value_to_py(py, &obs)?,
            value_to_py(py, &rlmesh::Value::Dict(info))?.cast_into()?,
        ))
    }
    #[pyo3(signature = (action=None))]
    #[gen_stub(override_return_type(type_repr = "tuple[typing.Any, builtins.float, builtins.bool, builtins.bool, builtins.dict[builtins.str, typing.Any]]", imports = ("builtins", "typing")))]
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &self,
        py: Python<'py>,
        action: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(
        Bound<'py, PyAny>,
        f64,
        bool,
        bool,
        Bound<'py, pyo3::types::PyDict>,
    )> {
        let action = action
            .map(|a| {
                py_any_to_space_value_with_backend(py, &a, &self.action_space, ValueBackend::Native)
                    .and_then(|a| {
                        rlmesh::binding::value_from_space(&self.action_space, a).map_err(to_py_err)
                    })
            })
            .transpose()?;
        let (obs, reward, terminated, truncated, info) = self.with(py, |b| b.step(action))?;
        Ok((
            value_to_py(py, &obs)?,
            reward,
            terminated,
            truncated,
            value_to_py(py, &rlmesh::Value::Dict(info))?.cast_into()?,
        ))
    }
    fn render<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let frame = py.detach(|| (self.render)()).map_err(to_py_err)?;
        if matches!(frame, rlmesh::Value::None) {
            Ok(None)
        } else {
            value_to_py(py, &frame).map(Some)
        }
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        // The core's close is idempotent. Preserve user hook exceptions even
        // when their text happens to match the closed-binding error.
        self.with(py, |b| b.close())
    }
}

impl Drop for PyBinding {
    fn drop(&mut self) {
        let mut binding = self
            .inner
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if binding.is_none() {
            return;
        }
        // The facade drains Python model/env callbacks when it closes. A
        // pyclass finalizer must release the GIL while those callbacks run.
        if Python::try_attach(|py| py.detach(|| drop(binding.take()))).is_none() {
            // Interpreter shutdown cannot legally execute Python callbacks.
            std::mem::forget(binding);
        }
    }
}
