use std::time::Duration;

use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
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
    #[pyo3(signature = (*, close_timeout_seconds=None, workflow_edition=None))]
    fn new(
        close_timeout_seconds: Option<f64>,
        workflow_edition: Option<String>,
    ) -> PyResult<PyServeOptions> {
        Ok(PyServeOptions {
            options: ServeOptions {
                close_timeout: optional_duration("close_timeout_seconds", close_timeout_seconds)?,
                workflow_edition: checked_workflow_edition(workflow_edition)?,
            },
        })
    }

    #[getter]
    fn close_timeout_seconds(&self) -> Option<f64> {
        self.options.close_timeout.map(|value| value.as_secs_f64())
    }

    #[getter]
    fn workflow_edition(&self) -> Option<String> {
        self.options.workflow_edition.clone()
    }
}

/// The bare `YYYY.MM` workflow edition this build runs at when nothing declares
/// one: the value to paste into a declaration, the same on every build of a
/// release line.
#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn current_workflow_edition() -> &'static str {
    rlmesh::current_workflow_edition()
}

/// The identity of this build of the native core: package version, protocol
/// generation, the workflow edition it advertises, and how that edition was
/// spelled. Every field is fixed at compile time.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "BuildInfo", frozen)]
pub struct PyBuildInfo;

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyBuildInfo {
    /// The `rlmesh` package version the native core was built as.
    #[getter]
    fn version(&self) -> &'static str {
        rlmesh::build_info().version
    }

    /// The wire protocol generation this build speaks (`rlmesh-realtime-v1`).
    #[getter]
    fn protocol_generation(&self) -> &'static str {
        rlmesh::build_info().protocol_generation
    }

    /// The session wire revision this build speaks. A 0.2 pre-release's wire
    /// may break before 0.2.0 freezes it: two builds on different revisions
    /// cannot talk, and `rlmesh check-image` compares an image's with this.
    #[getter]
    fn session_wire_revision(&self) -> u32 {
        rlmesh::build_info().session_wire_revision
    }

    /// The bare `YYYY.MM` workflow edition this build runs at when nothing
    /// declares one; what `current_workflow_edition()` returns, on a release,
    /// prerelease and source build alike.
    #[getter]
    fn workflow_edition(&self) -> &'static str {
        rlmesh::build_info().workflow_edition
    }

    /// How the build was made: `release` (a release build), `package` (a
    /// published crate or a checkout without git), or `git` (a source build
    /// stamped from its commit).
    #[getter]
    fn build_source(&self) -> &'static str {
        rlmesh::build_info().build_source.as_str()
    }

    /// The short commit a source build was stamped from, or `None` when the
    /// build was not stamped from git.
    #[getter]
    fn git(&self) -> Option<&'static str> {
        rlmesh::build_info().git
    }

    fn __repr__(&self) -> String {
        rlmesh::build_info().to_string()
    }
}

/// This build's identity: the package version, the protocol generation and
/// session wire revision it speaks, and its workflow edition.
#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn build_info() -> PyBuildInfo {
    PyBuildInfo
}

/// Refuse a workflow edition this build does not know, naming it, the editions
/// this build knows, and the remedy; return it trimmed otherwise.
#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn validate_workflow_edition(edition: &str) -> PyResult<String> {
    rlmesh::binding::validate_workflow_edition(edition).map_err(crate::types::to_py_err)
}

/// The workflow edition a participant runs at: `RLMESH_WORKFLOW_EDITION`, then
/// its declaration `declared`, then `current_workflow_edition()`; validated and
/// trimmed.
#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (declared=None))]
pub fn resolve_workflow_edition(declared: Option<&str>) -> PyResult<String> {
    rlmesh::binding::resolve_workflow_edition(declared).map_err(crate::types::to_py_err)
}

/// Call `hook` with `{kind, participant, address, port, pid}` as each env or
/// model this process serves becomes able to answer (`None` clears it). The
/// address is redacted; `python -m rlmesh.serve` prints its serving line here.
#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (hook))]
pub fn set_ready_hook(hook: Option<Py<PyAny>>) {
    let hook = hook.map(|hook| {
        std::sync::Arc::new(move |ready: &rlmesh::binding::Ready| {
            Python::attach(|py| {
                let call = || -> PyResult<()> {
                    let info = pyo3::types::PyDict::new(py);
                    info.set_item("kind", ready.kind)?;
                    info.set_item("participant", &ready.participant)?;
                    info.set_item("address", &ready.address)?;
                    info.set_item("port", ready.port)?;
                    info.set_item("pid", ready.pid)?;
                    hook.call1(py, (info,))?;
                    Ok(())
                };
                let Err(err) = call() else { return };
                // The hook runs on the serve's (main) thread, where Python can raise SIGINT mid-hook: hand it to the serve's poll.
                if err.is_instance_of::<PyKeyboardInterrupt>(py) {
                    crate::client::defer_interrupt(err);
                } else {
                    err.write_unraisable(py, None);
                }
            });
        }) as std::sync::Arc<dyn Fn(&rlmesh::binding::Ready) + Send + Sync>
    });
    rlmesh::binding::set_ready_hook(hook);
}

/// Normalize an optional declared edition: blank declares nothing, anything
/// else must be an edition this build knows.
pub(crate) fn checked_workflow_edition(edition: Option<String>) -> PyResult<Option<String>> {
    match edition {
        Some(edition) if !edition.trim().is_empty() => {
            validate_workflow_edition(&edition).map(Some)
        }
        _ => Ok(None),
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
    Duration::try_from_secs_f64(value).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!("{name} must be a positive finite float"))
    })
}
