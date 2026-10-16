use std::time::Duration;

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
    #[pyo3(signature = (*, allow_remote_shutdown=false, idle_timeout_seconds=None, drain_timeout_seconds=None, close_timeout_seconds=None, workflow_edition=None, compress_responses=false))]
    fn new(
        allow_remote_shutdown: bool,
        idle_timeout_seconds: Option<f64>,
        drain_timeout_seconds: Option<f64>,
        close_timeout_seconds: Option<f64>,
        workflow_edition: Option<String>,
        compress_responses: bool,
    ) -> PyResult<PyServeOptions> {
        Ok(PyServeOptions {
            options: ServeOptions {
                allow_remote_shutdown,
                idle_timeout: optional_duration("idle_timeout_seconds", idle_timeout_seconds)?,
                drain_timeout: optional_duration("drain_timeout_seconds", drain_timeout_seconds)?,
                close_timeout: optional_duration("close_timeout_seconds", close_timeout_seconds)?,
                // The Python model client wrapper is inherently single-flight
                // (its `block_on` predict API serializes by construction), so the
                // server-side concurrency cap is left at the default here. A
                // future Python knob can surface it without a wire change.
                workflow_edition: checked_workflow_edition(workflow_edition)?,
                compress_responses,
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
    fn workflow_edition(&self) -> Option<String> {
        self.options.workflow_edition.clone()
    }

    /// Whether an env server compresses its responses (zstd, else gzip) for a
    /// client that accepts them. Off by default; it only pays on a slow link.
    #[getter]
    fn compress_responses(&self) -> bool {
        self.options.compress_responses
    }
}

/// The bare `YYYY.MM` workflow edition this build runs at — the value to paste
/// into a declaration, on every build.
///
/// A declaration names the contract a participant was authored against, and a
/// bare base selects whichever spelling of that base both sides offer: the
/// sealed name on a release, this build's cohort (`YYYY.MM-<cohort>`, what
/// `rlmesh.build_info().workflow_edition` reports) on a prerelease or local
/// build. That cohort spelling is also accepted as a declaration, pinning to
/// that exact moving build.
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

    /// The wire protocol generation this build speaks (`rlmesh-wire-v1`).
    #[getter]
    fn protocol_generation(&self) -> &'static str {
        rlmesh::build_info().protocol_generation
    }

    /// The exact workflow edition spelling this build advertises: the sealed
    /// `YYYY.MM` base on a release, `YYYY.MM-<cohort>` on a prerelease or
    /// source build.
    #[getter]
    fn workflow_edition(&self) -> &'static str {
        rlmesh::build_info().workflow_edition
    }

    /// The sealed `YYYY.MM` base of `workflow_edition`, the same on every build
    /// of the edition; what `current_workflow_edition()` returns.
    #[getter]
    fn workflow_edition_base(&self) -> &'static str {
        rlmesh::build_info().workflow_edition_base
    }

    /// Every workflow edition this build can drive a session at, newest first:
    /// the retained list a served peer offers on `supported_workflow_editions`
    /// in its handshake, so a describe envelope advertises exactly what the
    /// handshake will.
    #[getter]
    fn supported_workflow_editions(&self) -> Vec<String> {
        rlmesh::build_info()
            .supported_workflow_editions
            .iter()
            .map(|edition| (*edition).to_string())
            .collect()
    }

    /// The cohort that spells `workflow_edition`: `stable` on a sealed release,
    /// the prerelease version (`0.1.0-rc.12`) on a prerelease, `dev.<git>` on a
    /// source build.
    #[getter]
    fn build_cohort(&self) -> &'static str {
        rlmesh::build_info().build_cohort
    }

    /// Where the cohort came from: `release` (a release build), `package` (a
    /// published crate or a checkout without git), or `git` (a source build
    /// stamped from its commit).
    #[getter]
    fn build_source(&self) -> &'static str {
        rlmesh::build_info().build_source.as_str()
    }

    /// The git state a source build was stamped from — the short commit sha,
    /// suffixed `.dirty.<fingerprint>` when the tree had uncommitted changes —
    /// or `None` when the build was not stamped from git.
    #[getter]
    fn git(&self) -> Option<&'static str> {
        rlmesh::build_info().git
    }

    fn __repr__(&self) -> String {
        rlmesh::build_info().to_string()
    }
}

/// This build's identity: the package version, the protocol generation, and
/// the workflow edition it advertises with the cohort behind that spelling.
#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn build_info() -> PyBuildInfo {
    PyBuildInfo
}

/// Refuse a declared workflow edition this build cannot run a session at, naming
/// the value and the editions it offers; return it trimmed otherwise.
///
/// The single Python-side boundary for a declared edition: every surface that
/// takes one from a user (`ServeOptions`, `--workflow-edition`, `run`/`session`,
/// `RLMESH_WORKFLOW_EDITION`, a class declaration, `[tool.rlmesh]`) resolves
/// through it, so a name that cannot work is refused where it is typed rather
/// than as a negotiation failure on the first connection.
#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
pub fn validate_workflow_edition(edition: &str) -> PyResult<String> {
    rlmesh::binding::validate_workflow_edition(edition).map_err(crate::types::to_py_err)
}

/// The facade resolver's rungs (COR-6): the call's value,
/// `RLMESH_WORKFLOW_EDITION`, the served peer's option, the class
/// declaration. Returns the winner (validated and trimmed) and whether
/// resolution stopped at a set-but-blank variable; `(None, False)` leaves
/// Python's `pyproject.toml` rung and the float warning to the caller.
#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (*, call=None, option=None, declared=None))]
pub fn resolve_workflow_edition_rungs(
    call: Option<&str>,
    option: Option<&str>,
    declared: Option<&str>,
) -> PyResult<(Option<String>, bool)> {
    use rlmesh::binding::{EditionResolution, EditionSources};
    let resolution = rlmesh::binding::resolve_workflow_edition_rungs(EditionSources {
        call,
        option,
        declared,
    })
    .map_err(crate::types::to_py_err)?;
    Ok(match resolution {
        EditionResolution::Declared(edition) => (Some(edition), false),
        EditionResolution::FloatsDeliberately => (None, true),
        EditionResolution::Undeclared => (None, false),
    })
}

/// Normalize an optional declared edition: blank declares nothing, anything
/// else must be an edition this build retains.
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
