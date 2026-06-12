//! Python exception types and error conversions.

use pyo3::create_exception;
use pyo3::exceptions::{PyConnectionError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use rlmesh::{EnvironmentError, Error, ErrorCode};

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
    }
}

/// Convert environment error to Python exception.
fn env_error_to_py(err: EnvironmentError) -> PyErr {
    let msg = format!("[{}] {}", format_error_code(err.code), err.message);

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
    }
}
/// Format error code for display.
fn format_error_code(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::Unspecified => "UNSPECIFIED",
        ErrorCode::Timeout => "TIMEOUT",
        ErrorCode::InvalidAction => "INVALID_ACTION",
        ErrorCode::NotReady => "NOT_READY",
        ErrorCode::Busy => "BUSY",
        ErrorCode::Internal => "INTERNAL",
        ErrorCode::Crashed => "CRASHED",
        ErrorCode::Cancelled => "CANCELLED",
        ErrorCode::Closed => "CLOSED",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_formatting() {
        assert_eq!(format_error_code(ErrorCode::Timeout), "TIMEOUT");
        assert_eq!(
            format_error_code(ErrorCode::InvalidAction),
            "INVALID_ACTION"
        );
        assert_eq!(format_error_code(ErrorCode::Closed), "CLOSED");
    }
}
