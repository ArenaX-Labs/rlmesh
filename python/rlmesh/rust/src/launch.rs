//! The keys a launcher starts a session host under in another process, and
//! the keyed address it dials it by (`rlmesh.launch`).

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;

/// Fresh host keys: the `RLMESH_HOST_IDENTITY` value (a secret) and the
/// address key participants join by.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def launch_identity() -> tuple[str, str]: ...
"#
    )
)]
#[pyfunction]
pub fn launch_identity() -> PyResult<(String, String)> {
    let (identity, key) = rlmesh_session::address::launch_identity()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok((identity.to_string(), key.to_string()))
}

/// The address key of a host identity.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def identity_key(identity: str) -> str: ...
"#
    )
)]
#[pyfunction]
pub fn identity_key(identity: &str) -> PyResult<String> {
    rlmesh_session::address::identity_key(identity)
        .map(|key| key.to_string())
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// The keyed address of `session` (`host:port[/session]`) on a host run
/// under `identity`, or of `participant` in it.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def launched_address(identity: str, session: str, participant: str | None = None) -> str: ...
"#
    )
)]
#[pyfunction]
#[pyo3(signature = (identity, session, participant=None))]
pub fn launched_address(
    identity: &str,
    session: &str,
    participant: Option<&str>,
) -> PyResult<String> {
    let invalid = |err: rlmesh_session::SessionError| PyValueError::new_err(err.to_string());
    let mut address =
        rlmesh_session::address::launched_address(identity, session).map_err(invalid)?;
    address.participant = participant.map(str::to_string);
    let invite = address.invite();
    // Parsed back so a bad participant name is refused here, not at dial.
    match participant {
        Some(_) => rlmesh_session::address::Address::participant(&invite),
        None => rlmesh_session::address::Address::session(&invite),
    }
    .map_err(invalid)?;
    Ok(invite)
}
