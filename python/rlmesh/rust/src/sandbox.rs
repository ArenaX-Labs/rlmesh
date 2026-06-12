use std::collections::BTreeMap;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rlmesh_sandbox::{EnvironmentSourceRef, SandboxOptions, VectorizationMode};

#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def sandbox_start_env(source: str, *, base_image: str | None = None, rlmesh_package: str | None = None, packages: list[str] | None = None, imports: list[str] | None = None, kwargs_json: str | None = None, num_envs: int = 1, vectorization_mode: str | None = None, trust_remote_code: bool = False, allow_unpinned_hf: bool = False) -> SandboxRunInfo: ...
"#
    )
)]
#[pyfunction]
#[pyo3(
    signature = (
        source,
        *,
        base_image = None,
        rlmesh_package = None,
        packages = None,
        imports = None,
        kwargs_json = None,
        num_envs = 1,
        vectorization_mode = None,
        trust_remote_code = false,
        allow_unpinned_hf = false
    )
)]
pub fn sandbox_start_env(
    py: Python<'_>,
    source: &str,
    base_image: Option<&str>,
    rlmesh_package: Option<&str>,
    packages: Option<Vec<String>>,
    imports: Option<Vec<String>>,
    kwargs_json: Option<&str>,
    num_envs: usize,
    vectorization_mode: Option<&str>,
    trust_remote_code: bool,
    allow_unpinned_hf: bool,
) -> PyResult<Py<PyAny>> {
    let source = source.to_string();
    let base_image = base_image.map(str::to_owned);
    let rlmesh_package = rlmesh_package.map(str::to_owned);
    let packages = packages.unwrap_or_default();
    let imports = imports.unwrap_or_default();
    let kwargs_json = kwargs_json.map(str::to_owned);
    let vectorization_mode = VectorizationMode::parse(vectorization_mode)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let started = py.detach(move || {
        let source_ref = EnvironmentSourceRef::parse(&source).map_err(|err| err.to_string())?;
        let run = rlmesh_sandbox::start_env(
            source_ref,
            SandboxOptions {
                base_image,
                rlmesh_package,
                packages,
                imports,
                kwargs: parse_kwargs_json(kwargs_json.as_deref()).map_err(|err| err.to_string())?,
                num_envs,
                vectorization_mode,
                trust_remote_code,
                allow_unpinned_hf,
            },
        )
        .map_err(|err| err.to_string())?;

        Ok::<SandboxStartResult, String>(SandboxStartResult {
            requested_source: run.requested_source,
            resolved_source: run.resolved_source,
            address: run.address,
            container_id: run.container_id,
        })
    });
    let started = started.map_err(PyRuntimeError::new_err)?;

    Python::attach(|py| {
        let info = PyDict::new(py);
        info.set_item("requested_source", started.requested_source)?;
        info.set_item("resolved_source", started.resolved_source)?;
        info.set_item("address", started.address)?;
        info.set_item("container_id", started.container_id)?;
        Ok(info.into_any().unbind())
    })
}

#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def sandbox_stop_env(*, container_id: str) -> None: ...
"#
    )
)]
#[pyfunction]
#[pyo3(signature = (*, container_id))]
pub fn sandbox_stop_env(py: Python<'_>, container_id: &str) -> PyResult<()> {
    let container_id = container_id.to_string();

    let result = py.detach(move || {
        rlmesh_sandbox::stop_container(&container_id).map_err(|err| err.to_string())?;
        Ok::<(), String>(())
    });
    result.map_err(PyRuntimeError::new_err)?;

    Ok(())
}

struct SandboxStartResult {
    requested_source: String,
    resolved_source: String,
    address: String,
    container_id: String,
}

fn parse_kwargs_json(raw: Option<&str>) -> PyResult<BTreeMap<String, serde_json::Value>> {
    let Some(raw) = raw else {
        return Ok(BTreeMap::new());
    };
    let value: serde_json::Value = serde_json::from_str(raw)
        .map_err(|err| PyValueError::new_err(format!("kwargs must be valid JSON: {err}")))?;
    let serde_json::Value::Object(object) = value else {
        return Err(PyValueError::new_err(
            "kwargs JSON must decode to an object",
        ));
    };
    Ok(object.into_iter().collect())
}
