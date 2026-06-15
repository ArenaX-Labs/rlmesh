use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rlmesh_sandbox::{
    EnvironmentSourceRef, RecipeProvenance, RecipeSourceRef, SandboxOptions, VectorizationMode,
};

#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def sandbox_start_env(source: str, *, base_image: str | None = None, rlmesh_package: str | None = None, packages: list[str] | None = None, imports: list[str] | None = None, kwargs_json: str | None = None, num_envs: int = 1, vectorization_mode: str | None = None, trust_remote_code: bool = False, allow_unpinned_hf: bool = False, recipe_json: str | None = None, recipe_provenance: str | None = None, context_root: str | None = None, mounts_json: str | None = None, build_memory: str | None = None) -> SandboxRunInfo: ...
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
        allow_unpinned_hf = false,
        recipe_json = None,
        recipe_provenance = None,
        context_root = None,
        mounts_json = None,
        build_memory = None
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
    recipe_json: Option<&str>,
    recipe_provenance: Option<&str>,
    context_root: Option<&str>,
    mounts_json: Option<&str>,
    build_memory: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let source = source.to_string();
    let base_image = base_image.map(str::to_owned);
    let rlmesh_package = rlmesh_package.map(str::to_owned);
    let packages = packages.unwrap_or_default();
    let imports = imports.unwrap_or_default();
    let kwargs_json = kwargs_json.map(str::to_owned);
    let recipe_json = recipe_json.map(str::to_owned);
    let recipe_provenance = recipe_provenance.map(str::to_owned);
    let context_root = context_root.map(PathBuf::from);
    let build_memory = build_memory.map(str::to_owned);
    let mounts = parse_mounts_json(mounts_json)?;
    let vectorization_mode = VectorizationMode::parse(vectorization_mode)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let started = py.detach(move || {
        let source_ref = match recipe_json.as_deref() {
            Some(document) => build_recipe_source(&source, document, recipe_provenance.as_deref())?,
            None => EnvironmentSourceRef::parse(&source).map_err(|err| err.to_string())?,
        };
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
                context_root,
                mounts,
                build_memory,
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

#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def sandbox_build_image(source: str, *, tag: str | None = None, base_image: str | None = None, rlmesh_package: str | None = None, packages: list[str] | None = None, imports: list[str] | None = None, trust_remote_code: bool = False, allow_unpinned_hf: bool = False, recipe_json: str | None = None, recipe_provenance: str | None = None, context_root: str | None = None, build_memory: str | None = None) -> SandboxBuildInfo: ...
"#
    )
)]
#[pyfunction]
#[pyo3(
    signature = (
        source,
        *,
        tag = None,
        base_image = None,
        rlmesh_package = None,
        packages = None,
        imports = None,
        trust_remote_code = false,
        allow_unpinned_hf = false,
        recipe_json = None,
        recipe_provenance = None,
        context_root = None,
        build_memory = None
    )
)]
pub fn sandbox_build_image(
    py: Python<'_>,
    source: &str,
    tag: Option<&str>,
    base_image: Option<&str>,
    rlmesh_package: Option<&str>,
    packages: Option<Vec<String>>,
    imports: Option<Vec<String>>,
    trust_remote_code: bool,
    allow_unpinned_hf: bool,
    recipe_json: Option<&str>,
    recipe_provenance: Option<&str>,
    context_root: Option<&str>,
    build_memory: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let source = source.to_string();
    let tag = tag.map(str::to_owned);
    let base_image = base_image.map(str::to_owned);
    let rlmesh_package = rlmesh_package.map(str::to_owned);
    let packages = packages.unwrap_or_default();
    let imports = imports.unwrap_or_default();
    let recipe_json = recipe_json.map(str::to_owned);
    let recipe_provenance = recipe_provenance.map(str::to_owned);
    let context_root = context_root.map(PathBuf::from);
    let build_memory = build_memory.map(str::to_owned);

    let built = py.detach(move || {
        let source_ref = match recipe_json.as_deref() {
            Some(document) => build_recipe_source(&source, document, recipe_provenance.as_deref())?,
            None => EnvironmentSourceRef::parse(&source).map_err(|err| err.to_string())?,
        };
        // num_envs/vectorization_mode/kwargs/mounts are runtime-only and excluded
        // from the build hash; defaults are fine for a build-only call.
        let result = rlmesh_sandbox::build_env(
            source_ref,
            SandboxOptions {
                base_image,
                rlmesh_package,
                packages,
                imports,
                kwargs: Default::default(),
                num_envs: 1,
                vectorization_mode: VectorizationMode::parse(None)
                    .map_err(|err| err.to_string())?,
                trust_remote_code,
                allow_unpinned_hf,
                context_root,
                mounts: Vec::new(),
                build_memory,
            },
            tag.as_deref(),
        )
        .map_err(|err| err.to_string())?;

        Ok::<SandboxBuildOutcome, String>(SandboxBuildOutcome {
            requested_source: result.requested_source,
            resolved_source: result.resolved_source,
            image: result.image,
            alias: result.alias,
            image_id: result.image_id,
        })
    });
    let built = built.map_err(PyRuntimeError::new_err)?;

    Python::attach(|py| {
        let info = PyDict::new(py);
        info.set_item("requested_source", built.requested_source)?;
        info.set_item("resolved_source", built.resolved_source)?;
        info.set_item("image", built.image)?;
        info.set_item("alias", built.alias)?;
        info.set_item("image_id", built.image_id)?;
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

struct SandboxBuildOutcome {
    requested_source: String,
    resolved_source: String,
    image: String,
    alias: Option<String>,
    image_id: String,
}

/// Build a recipe environment source from a JSON document handed in by the
/// Python registry layer (the name resolves to a recipe before this call). The
/// `provenance` defaults to `remote` -- the safe default -- so an unmarked
/// document is gated as untrusted.
fn build_recipe_source(
    source: &str,
    document_json: &str,
    provenance: Option<&str>,
) -> Result<EnvironmentSourceRef, String> {
    let document: serde_json::Value = serde_json::from_str(document_json)
        .map_err(|err| format!("recipe document must be valid JSON: {err}"))?;
    let name = document
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(source)
        .to_string();
    let provenance = match provenance.unwrap_or("remote") {
        "installed" => RecipeProvenance::Installed,
        "remote" => RecipeProvenance::Remote,
        other => {
            return Err(format!(
                "recipe_provenance must be 'installed' or 'remote', got {other:?}"
            ));
        }
    };
    Ok(EnvironmentSourceRef::Recipe(RecipeSourceRef {
        name,
        document,
        provenance,
    }))
}

/// Parse `[[host, target], ...]` artifact bind-mount pairs handed in by the
/// Python sandbox layer (already resolved to absolute host paths). An absent or
/// empty value means no mounts -- the gym/hf and no-artifact paths.
fn parse_mounts_json(raw: Option<&str>) -> PyResult<Vec<(String, String)>> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    serde_json::from_str::<Vec<(String, String)>>(raw).map_err(|err| {
        PyValueError::new_err(format!(
            "mounts must be a JSON array of [host, target] string pairs: {err}"
        ))
    })
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
