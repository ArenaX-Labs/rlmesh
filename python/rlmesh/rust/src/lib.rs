mod adapters;
mod client;
mod lifecycle;
mod model;
mod peer_info;
mod sandbox;
mod server;
mod spaces;
mod telemetry;
mod types;
mod viewer;

#[cfg(feature = "cli")]
use std::ffi::OsString;
#[cfg(feature = "stub-gen")]
use std::path::PathBuf;

#[cfg(feature = "cli")]
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
#[cfg(all(feature = "cli", feature = "stub-gen"))]
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

/// Run the embedded `rlmesh` CLI. Feature-gated (on by default) so a lean
/// wheel can opt out of linking `rlmesh-cli` via `--no-default-features`.
#[cfg(feature = "cli")]
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
    let sys = py.import("sys")?;
    let stdout_obj = sys.getattr("stdout")?.unbind();
    let stderr_obj = sys.getattr("stderr")?.unbind();
    let out_tty = stream_isatty(py, &stdout_obj);
    let err_tty = stream_isatty(py, &stderr_obj);

    py.detach(|| {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let mut stdout = PyStream { stream: stdout_obj };
        let mut stderr = PyStream { stream: stderr_obj };
        runtime
            .block_on(rlmesh_cli::run_cli_with_writers(
                args.into_iter().map(OsString::from).collect(),
                &mut stdout,
                &mut stderr,
                rlmesh_cli::Style::for_terminal(out_tty),
                rlmesh_cli::Style::for_terminal(err_tty),
            ))
            .map_err(|err| PyRuntimeError::new_err(format!("{err:#}")))
    })
}

/// Whether a Python file object is an interactive terminal, defaulting to
/// `false` when it does not implement `isatty` (e.g. a capture buffer).
#[cfg(feature = "cli")]
fn stream_isatty(py: Python<'_>, stream: &Py<pyo3::PyAny>) -> bool {
    stream
        .call_method0(py, "isatty")
        .and_then(|res| res.extract::<bool>(py))
        .unwrap_or(false)
}

/// A `std::io::Write` sink that forwards to a Python file object (`sys.stdout`
/// or `sys.stderr`), so the embedded CLI's output honors Python-level stream
/// redirection (`contextlib.redirect_stderr`, pytest `capsys`, Jupyter kernel
/// streams) instead of writing straight to the process file descriptor. Each
/// write briefly reattaches the interpreter; the CLI writes whole lines
/// infrequently, so per-write reattachment is not a hot path.
#[cfg(feature = "cli")]
struct PyStream {
    stream: Py<pyo3::PyAny>,
}

#[cfg(feature = "cli")]
impl std::io::Write for PyStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let text = String::from_utf8_lossy(buf);
        Python::attach(|py| {
            self.stream
                .call_method1(py, "write", (text.as_ref(),))
                .map_err(|err| std::io::Error::other(err.to_string()))
        })?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Python::attach(|py| {
            self.stream
                .call_method0(py, "flush")
                .map(|_| ())
                .map_err(|err| std::io::Error::other(err.to_string()))
        })
    }
}

#[pymodule]
#[pyo3(name = "_rlmesh")]
pub fn rlmesh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__build__", rlmesh_proto::CURRENT_WORKFLOW_EDITION)?;

    types::register_exceptions(m)?;
    spaces::register_classes(m)?;
    m.add_class::<lifecycle::PyServeOptions>()?;

    m.add_class::<server::PyEnvServer>()?;
    m.add_class::<server::PyVectorEnvServer>()?;
    m.add_class::<model::PyModel>()?;
    m.add_class::<model::PyModelClient>()?;
    m.add_class::<client::PyEnvClient>()?;
    m.add_class::<client::PyVectorEnvClient>()?;
    m.add_class::<viewer::PyViewer>()?;
    m.add_class::<viewer::PyVideoWriter>()?;
    #[cfg(feature = "cli")]
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    m.add_function(wrap_pyfunction!(peer_info::set_python_peer_info, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox::sandbox_start_env, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox::sandbox_stop_env, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox::sandbox_reap_orphans, m)?)?;

    adapters::register_constants(m)?;
    m.add_class::<adapters::PyAdvisory>()?;
    m.add_class::<adapters::PyAdapterPlan>()?;
    m.add_function(wrap_pyfunction!(adapters::adapters_resolve, m)?)?;
    m.add_function(wrap_pyfunction!(adapters::adapters_join_check, m)?)?;
    m.add_function(wrap_pyfunction!(adapters::adapters_spec_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(adapters::describe_envelope_normalize, m)?)?;

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
