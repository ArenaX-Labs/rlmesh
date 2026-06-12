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
use std::path::PathBuf;

#[cfg(feature = "viewer")]
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
#[cfg(feature = "viewer")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use pyo3_stub_gen::derive::gen_type_alias_from_python;

gen_type_alias_from_python!(
    "rlmesh._rlmesh",
    r#"
from typing import TypeAlias

PrimitiveValue: TypeAlias = None | bool | int | float | str | bytes
Value: TypeAlias = PrimitiveValue | Tensor | list["Value"] | tuple["Value", ...] | dict[str, "Value"]
"#
);

// The render viewer (and its egui/eframe/glow/wayland/x11 stack via rlmesh-cli)
// is only linked into the extension when the `viewer` feature is enabled, so
// default release wheels stay lean and headless (review finding #113). Builds
// that ship the viewer expose `run_cli`, which `python -m rlmesh viewer`
// drives; lean wheels simply omit it (the Python entrypoint degrades to an
// ImportError-guarded fallback).
#[cfg(feature = "viewer")]
#[gen_stub_pyfunction(
    module = "rlmesh._rlmesh",
    python = r#"
def run_cli(args: list[str]) -> int: ...
"#
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
    // Register custom exception types
    types::errors::register_exceptions(m)?;
    spaces::register_classes(m)?;
    m.add_class::<lifecycle::PyServeOptions>()?;

    // Add server classes
    m.add_class::<server::PyEnvServer>()?;

    // Add model classes
    m.add_class::<model::PyModel>()?;

    // Add client classes
    m.add_class::<client::PyEnvClient>()?;
    m.add_class::<client::PyVectorEnvClient>()?;
    #[cfg(feature = "viewer")]
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox::sandbox_start_env, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox::sandbox_stop_env, m)?)?;

    Ok(())
}

pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    pyo3_stub_gen::StubInfo::from_pyproject_toml(resolve_pyproject_toml(&manifest_dir))
}

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
