use std::time::Duration;

use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rlmesh::ServeOptions;

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(
    module = "rlmesh._rlmesh",
    name = "ServeOptions",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyServeOptions {
    options: ServeOptions,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyServeOptions {
    #[new]
    #[pyo3(signature = (*, allow_remote_shutdown=false, idle_timeout_seconds=None, drain_timeout_seconds=None, close_timeout_seconds=None, token=None))]
    fn new(
        allow_remote_shutdown: bool,
        idle_timeout_seconds: Option<f64>,
        drain_timeout_seconds: Option<f64>,
        close_timeout_seconds: Option<f64>,
        token: Option<String>,
    ) -> PyResult<PyServeOptions> {
        Ok(PyServeOptions {
            options: ServeOptions {
                allow_remote_shutdown,
                idle_timeout: optional_duration("idle_timeout_seconds", idle_timeout_seconds)?,
                drain_timeout: optional_duration("drain_timeout_seconds", drain_timeout_seconds)?,
                close_timeout: optional_duration("close_timeout_seconds", close_timeout_seconds)?,
                token,
            },
        })
    }

    #[getter]
    fn allow_remote_shutdown(&self) -> bool {
        self.options.allow_remote_shutdown
    }

    #[getter]
    fn idle_timeout_seconds(&self) -> Option<f64> {
        self.options.idle_timeout.map(|value| value.as_secs_f64())
    }

    #[getter]
    fn drain_timeout_seconds(&self) -> Option<f64> {
        self.options.drain_timeout.map(|value| value.as_secs_f64())
    }

    #[getter]
    fn close_timeout_seconds(&self) -> Option<f64> {
        self.options.close_timeout.map(|value| value.as_secs_f64())
    }

    #[getter]
    fn token(&self) -> Option<String> {
        self.options.token.clone()
    }
}

impl PyServeOptions {
    pub(crate) fn into_rust(self) -> ServeOptions {
        self.options
    }
}

fn optional_duration(name: &str, value: Option<f64>) -> PyResult<Option<Duration>> {
    value.map(|value| duration(name, value)).transpose()
}

fn duration(name: &str, value: f64) -> PyResult<Duration> {
    if value <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be positive"
        )));
    }
    Ok(Duration::from_secs_f64(value))
}
