use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const MODULE_NAME: &str = "rlmesh._rlmesh";
const CHECK_ARG: &str = "--check";
const SUPPLEMENTAL_EXPORTS: &[&str] = &[
    "RLMeshException",
    "ProtocolException",
    "EnvironmentException",
];
const SUPPLEMENTAL_STUBS: &str = r#"
__version__: builtins.str

class RLMeshException(builtins.RuntimeError): ...

class ProtocolException(RLMeshException): ...

class EnvironmentException(RLMeshException): ...

ResetInfo: TypeAlias = dict[str, object]

StepInfo: TypeAlias = dict[str, object]

class RenderBundle(TypedDict):
    frame: Value | None
    packet: bytes | None

class SandboxRunInfo(TypedDict):
    requested_source: str
    resolved_source: str
    address: str
    container_id: str

"#;
const TENSOR_BUFFER_STUB: &str =
    "    def __buffer__(self, flags: builtins.int, /) -> memoryview: ...\n";

fn main() -> pyo3_stub_gen::Result<()> {
    let check = env::args().skip(1).any(|arg| arg == CHECK_ARG);
    let stub = _rlmesh::stub_info()?;
    let module = stub.modules.get(MODULE_NAME).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("pyo3-stub-gen did not produce metadata for {MODULE_NAME}"),
        )
    })?;

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dest = manifest_dir.join("../src/rlmesh/_rlmesh.pyi");
    let content = finalize_stub(module.format_with_config(stub.config.use_type_statement));

    if check {
        check_stub_fresh(&dest, &content)?;
        check_no_stale_package_stub_dir(&manifest_dir)?;
    } else {
        fs::write(&dest, content)?;
        remove_stale_package_stub_dir(&manifest_dir)?;
    }
    Ok(())
}

fn finalize_stub(mut content: String) -> String {
    content = content.replace("# ruff: noqa: E501, F401, F403, F405", "# ruff: noqa");
    content = content.replace(
        "from typing import TypeAlias",
        "from typing import TypeAlias, TypedDict",
    );
    if !content.contains("TypedDict") {
        content = content.replace(
            "import typing\n",
            "import typing\nfrom typing import TypedDict\n",
        );
    }
    insert_supplemental_exports(&mut content);
    insert_supplemental_stubs(&mut content);
    insert_tensor_buffer_stub(&mut content);
    content
}

fn insert_supplemental_exports(content: &mut String) {
    let missing_exports = SUPPLEMENTAL_EXPORTS
        .iter()
        .filter(|name| !has_all_export(content, name))
        .map(|name| format!("    \"{name}\",\n"))
        .collect::<String>();
    if missing_exports.is_empty() {
        return;
    }

    if let Some(index) = content.find("__all__ = [\n") {
        content.insert_str(index + "__all__ = [\n".len(), &missing_exports);
    }
}

fn has_all_export(content: &str, name: &str) -> bool {
    let export = format!("\"{name}\",");
    content.lines().any(|line| line.trim() == export)
}

fn insert_supplemental_stubs(content: &mut String) {
    let stubs = SUPPLEMENTAL_STUBS.trim_start();
    if let Some(index) = content.find("\n@typing.final\nclass ") {
        content.insert_str(index + 1, stubs);
    } else if let Some(index) = content.find("\nclass ") {
        content.insert_str(index + 1, stubs);
    } else {
        if !content.ends_with('\n') {
            content.push('\n');
        }
        content.push_str(stubs);
    }
}

fn insert_tensor_buffer_stub(content: &mut String) {
    if content.contains("    def __buffer__(self, flags: builtins.int, /) -> memoryview: ...") {
        return;
    }

    if let Some(index) = content.find("    def __new__(cls, buffer: object,") {
        content.insert_str(index, TENSOR_BUFFER_STUB);
    }
}

fn check_stub_fresh(dest: &Path, expected: &str) -> io::Result<()> {
    let existing = fs::read_to_string(dest)?;
    if existing == expected {
        return Ok(());
    }

    Err(io::Error::other(format!(
        "{} is stale; run `cargo run --manifest-path python/rlmesh/rust/Cargo.toml --features stub-gen --bin stub_gen` (or `mise run stubs:generate`)",
        dest.display()
    )))
}

fn check_no_stale_package_stub_dir(manifest_dir: &Path) -> io::Result<()> {
    let stale_dir = package_stub_dir(manifest_dir);
    if stale_dir.exists() {
        return Err(io::Error::other(format!(
            "{} is stale; run `cargo run --manifest-path python/rlmesh/rust/Cargo.toml --features stub-gen --bin stub_gen` (or `mise run stubs:generate`)",
            stale_dir.display()
        )));
    }
    Ok(())
}

fn remove_stale_package_stub_dir(manifest_dir: &Path) -> io::Result<()> {
    let stale_dir = package_stub_dir(manifest_dir);
    if stale_dir.exists() {
        fs::remove_dir_all(stale_dir)?;
    }
    Ok(())
}

fn package_stub_dir(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join("../src/rlmesh/_rlmesh")
}
