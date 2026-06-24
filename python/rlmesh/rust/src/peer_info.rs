//! Bridge for the Python SDK to enrich the handshake [`PeerInfo`] with its real
//! runtime (language/version, OS, arch, framework versions).
//!
//! The Python package collects this best-effort at import (see
//! `rlmesh._peer_info.collect_peer_info`) and calls
//! [`_set_python_peer_info`](set_python_peer_info) once. It installs a
//! process-wide [`PeerInfoOverride`] that `rlmesh_proto::peer_info` merges over
//! the Rust-detected defaults for every handshake this (Python-hosted) process
//! performs. Purely additive diagnostics; PeerInfo never gates compatibility.
//!
//! The entry point is `_`-prefixed and intentionally not a public API symbol, so
//! it does not appear in the Python public API surface.

use std::collections::HashMap;

use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rlmesh_proto::{PeerInfoOverride, set_peer_info_override};

/// Install the process-wide handshake [`PeerInfoOverride`] from Python.
///
/// Every argument is optional and best-effort; empty/absent values fall back to
/// the Rust-detected defaults during the merge. The high-level package passes
/// the dict produced by `rlmesh._peer_info.collect_peer_info`.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def _set_python_peer_info(*, language: str | None = None, language_version: str | None = None, package_version: str | None = None, os: str | None = None, os_version: str | None = None, arch: str | None = None, framework_versions: dict[str, str] | None = None) -> None: ...
"#
    )
)]
#[pyfunction]
#[pyo3(name = "_set_python_peer_info")]
#[pyo3(
    signature = (
        *,
        language = None,
        language_version = None,
        package_version = None,
        os = None,
        os_version = None,
        arch = None,
        framework_versions = None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn set_python_peer_info(
    language: Option<String>,
    language_version: Option<String>,
    package_version: Option<String>,
    os: Option<String>,
    os_version: Option<String>,
    arch: Option<String>,
    framework_versions: Option<HashMap<String, String>>,
) {
    set_peer_info_override(PeerInfoOverride {
        language: language.unwrap_or_default(),
        language_version: language_version.unwrap_or_default(),
        package_version: package_version.unwrap_or_default(),
        os: os.unwrap_or_default(),
        os_version: os_version.unwrap_or_default(),
        arch: arch.unwrap_or_default(),
        framework_versions: framework_versions.unwrap_or_default(),
        extra: HashMap::new(),
    });
}
