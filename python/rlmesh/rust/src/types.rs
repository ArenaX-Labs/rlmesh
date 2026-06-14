//! Python exception types, error conversions, and payload-size accounting.

use pyo3::create_exception;
use pyo3::exceptions::{PyConnectionError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use rlmesh::{EnvironmentError, Error, ErrorCode};
use rlmesh_spaces::SpaceValue;

// Custom exception types for RLMesh-specific errors
create_exception!(
    _rlmesh,
    RLMeshException,
    PyRuntimeError,
    "Base exception for RLMesh errors"
);

create_exception!(
    _rlmesh,
    ProtocolException,
    RLMeshException,
    "Protocol-level errors (generation mismatch, invalid messages)"
);

create_exception!(
    _rlmesh,
    EnvironmentException,
    RLMeshException,
    "Environment-level errors (from the spaces environment)"
);

/// Convert rlmesh facade Error to Python exception.
///
/// Note: We can't implement `From<Error> for PyErr` due to orphan rules,
/// so we use this helper function instead.
pub fn to_py_err(err: Error) -> PyErr {
    match err {
        Error::Address(message) => PyValueError::new_err(format!("invalid address: {message}")),
        Error::Connection(message) => PyConnectionError::new_err(message),
        Error::Timeout(duration) => {
            PyTimeoutError::new_err(format!("operation timed out after {:?}", duration))
        }
        Error::Environment(e) => env_error_to_py(e),
        Error::Model(e) => PyRuntimeError::new_err(format!("model error: {}", e.message)),
        Error::Server(message) => PyRuntimeError::new_err(format!("server error: {message}")),
        Error::Internal(message) => PyRuntimeError::new_err(message),
        // rlmesh::Error is #[non_exhaustive].
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Convert environment error to Python exception.
fn env_error_to_py(err: EnvironmentError) -> PyErr {
    let msg = format!("[{}] {}", err.code, err.message);

    match err.code {
        ErrorCode::Timeout => PyTimeoutError::new_err(msg),
        ErrorCode::InvalidAction => PyValueError::new_err(msg),
        ErrorCode::NotReady => EnvironmentException::new_err(msg),
        ErrorCode::Busy => EnvironmentException::new_err(msg),
        ErrorCode::Internal => EnvironmentException::new_err(msg),
        ErrorCode::Crashed => EnvironmentException::new_err(msg),
        ErrorCode::Cancelled => PyRuntimeError::new_err(msg),
        ErrorCode::Closed => EnvironmentException::new_err(msg),
        ErrorCode::Unspecified => EnvironmentException::new_err(msg),
        // rlmesh::ErrorCode is #[non_exhaustive].
        _ => EnvironmentException::new_err(msg),
    }
}

/// Register exception types with the Python module.
pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("RLMeshException", m.py().get_type::<RLMeshException>())?;
    m.add("ProtocolException", m.py().get_type::<ProtocolException>())?;
    m.add(
        "EnvironmentException",
        m.py().get_type::<EnvironmentException>(),
    )?;
    Ok(())
}

/// Byte size of a single space value's payload (profiling).
pub(crate) fn space_value_size(value: &SpaceValue) -> usize {
    match value {
        SpaceValue::Box(value) => value.nbytes(),
        SpaceValue::Discrete(_) => std::mem::size_of::<i64>(),
        SpaceValue::MultiBinary(values) => values.len(),
        SpaceValue::MultiDiscrete(values) => values.len() * std::mem::size_of::<i64>(),
        SpaceValue::Text(value) => value.len(),
        SpaceValue::Dict(values) => values.values().map(space_value_size).sum(),
        SpaceValue::Tuple(values) => values.iter().map(space_value_size).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_error_code_formats_with_label() {
        assert_eq!(ErrorCode::Timeout.to_string(), "TIMEOUT");
        assert_eq!(ErrorCode::InvalidAction.to_string(), "INVALID_ACTION");
        assert_eq!(ErrorCode::Closed.to_string(), "CLOSED");
    }
}
