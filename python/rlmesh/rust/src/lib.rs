mod adapters;
mod client;
mod lifecycle;
mod model;
mod sandbox;
mod server;
mod spaces;
mod telemetry;
mod types;

#[cfg(feature = "viewer")]
use std::ffi::OsString;
#[cfg(feature = "stub-gen")]
use std::path::PathBuf;

#[cfg(feature = "viewer")]
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
#[cfg(all(feature = "viewer", feature = "stub-gen"))]
use pyo3_stub_gen::derive::gen_stub_pyfunction;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::gen_type_alias_from_python;

#[cfg(feature = "stub-gen")]
gen_type_alias_from_python!(
    "rlmesh._rlmesh",
    r#"
from typing import TypeAlias

PrimitiveValue: TypeAlias = None | bool | int | float | str | bytes
Value: TypeAlias = PrimitiveValue | Tensor | list["Value"] | tuple["Value", ...] | dict[str, "Value"]
"#
);

// Viewer support is feature-gated (on by default) so a lean wheel can opt out
// of linking egui/eframe via `--no-default-features`.
#[cfg(feature = "viewer")]
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def run_cli(args: list[str]) -> int: ...
"#
    )
)]
#[pyfunction]
fn run_cli(py: Python<'_>, args: Vec<String>) -> PyResult<i32> {
    py.detach(|| {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        runtime
            .block_on(rlmesh_cli::run_cli_with_args(
                args.into_iter().map(OsString::from).collect(),
            ))
            .map_err(|err| PyRuntimeError::new_err(format!("{err:#}")))
    })
}

#[pymodule]
#[pyo3(name = "_rlmesh")]
pub fn rlmesh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    types::register_exceptions(m)?;
    spaces::register_classes(m)?;
    m.add_class::<lifecycle::PyServeOptions>()?;

    m.add_class::<server::PyEnvServer>()?;
    m.add_class::<model::PyModel>()?;
    m.add_class::<client::PyEnvClient>()?;
    m.add_class::<client::PyVectorEnvClient>()?;
    #[cfg(feature = "viewer")]
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox::sandbox_start_env, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox::sandbox_stop_env, m)?)?;

    adapters::register_constants(m)?;
    m.add_class::<adapters::PyAdapterPlan>()?;
    m.add_function(wrap_pyfunction!(adapters::adapters_resolve, m)?)?;
    m.add_function(wrap_pyfunction!(adapters::adapters_join_check, m)?)?;

    Ok(())
}

#[cfg(feature = "stub-gen")]
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    pyo3_stub_gen::StubInfo::from_pyproject_toml(resolve_pyproject_toml(&manifest_dir))
}

#[cfg(feature = "stub-gen")]
fn resolve_pyproject_toml(manifest_dir: &std::path::Path) -> PathBuf {
    let mut candidates = vec![
        manifest_dir.join("../pyproject.toml"),
        PathBuf::from("python/rlmesh/pyproject.toml"),
        PathBuf::from("pyproject.toml"),
    ];

    if let Ok(current_dir) = std::env::current_dir() {
        candidates.push(current_dir.join("python/rlmesh/pyproject.toml"));
        candidates.push(current_dir.join("pyproject.toml"));
    }

    candidates
        .into_iter()
        .find(|path| path.exists())
        .unwrap_or_else(|| manifest_dir.join("../pyproject.toml"))
}
