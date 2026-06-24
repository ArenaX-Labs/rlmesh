use std::fmt;
use std::time::Duration;

use rlmesh_grpc::error::{EnvError, EnvErrorCode};

/// The result type used throughout this crate: `Result<T, `[`Error`]`>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Classifies an [`EnvironmentError`] reported by an environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorCode {
    /// No specific code was provided.
    Unspecified,
    /// The operation exceeded its deadline.
    Timeout,
    /// The submitted action was invalid for the action space.
    InvalidAction,
    /// The environment is not ready to serve the request yet.
    NotReady,
    /// The environment is busy with another request.
    Busy,
    /// An internal environment error.
    Internal,
    /// The environment process crashed.
    Crashed,
    /// The operation was cancelled.
    Cancelled,
    /// The environment (or session) is closed.
    Closed,
}

/// A failure reported by the environment when serving a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvironmentError {
    /// Machine-readable classification of the failure.
    pub code: ErrorCode,
    /// Human-readable description.
    pub message: String,
    /// Whether the operation may be retried.
    pub is_recoverable: bool,
}

/// A failure originating from a user-implemented model handler.
///
/// This is distinct from the transport/internal variants: it represents the
/// model *declining* a request (e.g. an invalid observation), so the runtime
/// can report and retry it appropriately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelError {
    /// Human-readable description of why the handler declined.
    pub message: String,
    /// Whether the caller may retry the request.
    pub is_recoverable: bool,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ModelError {}

/// The error type for every fallible operation in this crate.
///
/// The variants separate transport/setup faults ([`Error::Address`],
/// [`Error::Connection`], [`Error::Server`], [`Error::Internal`]) from the two
/// domain failures: [`Error::Environment`] (the env reported a failure) and
/// [`Error::Model`] (a user [`crate::ModelHandler`] declined a request). Use
/// [`Error::is_recoverable`] to decide whether to retry.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// An address or bind target could not be parsed.
    Address(String),
    /// A transport-level connection failure (treated as recoverable).
    Connection(String),
    /// An operation exceeded its deadline (treated as recoverable).
    Timeout(Duration),
    /// The environment reported a failure; see [`EnvironmentError`].
    Environment(EnvironmentError),
    /// A failure raised by a user-implemented [`crate::ModelHandler`].
    Model(ModelError),
    /// A server-side failure (e.g. bind failed, close hook failed).
    Server(String),
    /// An internal/protocol error that should not normally occur.
    Internal(String),
}

impl Error {
    /// Construct a non-recoverable model-handler error.
    ///
    /// Use this from a [`crate::ModelHandler::predict`] implementation to signal
    /// that the model declined the request (e.g. a malformed observation),
    /// rather than misreporting it as an internal/transport fault.
    pub fn model(message: impl Into<String>) -> Self {
        Self::Model(ModelError {
            message: message.into(),
            is_recoverable: false,
        })
    }

    /// Construct a recoverable model-handler error (the caller may retry).
    pub fn model_recoverable(message: impl Into<String>) -> Self {
        Self::Model(ModelError {
            message: message.into(),
            is_recoverable: true,
        })
    }

    /// Whether this error is recoverable (the operation may be retried).
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Timeout(_) => true,
            Self::Environment(error) => error.is_recoverable,
            Self::Model(error) => error.is_recoverable,
            Self::Connection(_) => true,
            _ => false,
        }
    }
}

/// Combine a primary result (carrying a value `T`) with a close-hook result. If
/// both fail, fold them into one [`Error::Internal`] prefixed with `ctx`
/// ("<ctx>: <a>; close hook failed: <b>").
pub(crate) fn join_results<T>(a: Result<T>, b: Result<()>, ctx: &str) -> Result<T> {
    match (a, b) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(err), Ok(())) => Err(err),
        (Ok(_), Err(err)) => Err(err),
        (Err(a_err), Err(b_err)) => Err(Error::Internal(format!(
            "{ctx}: {a_err}; close hook failed: {b_err}"
        ))),
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Unspecified => "UNSPECIFIED",
            Self::Timeout => "TIMEOUT",
            Self::InvalidAction => "INVALID_ACTION",
            Self::NotReady => "NOT_READY",
            Self::Busy => "BUSY",
            Self::Internal => "INTERNAL",
            Self::Crashed => "CRASHED",
            Self::Cancelled => "CANCELLED",
            Self::Closed => "CLOSED",
        };
        f.write_str(label)
    }
}

impl fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for EnvironmentError {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Address(message) => write!(f, "invalid address: {message}"),
            Self::Connection(message) => write!(f, "connection error: {message}"),
            Self::Timeout(duration) => write!(f, "timeout after {duration:?}"),
            Self::Environment(error) => write!(f, "environment error: {error}"),
            Self::Model(error) => write!(f, "model error: {error}"),
            Self::Server(message) => write!(f, "server error: {message}"),
            Self::Internal(message) => write!(f, "internal error: {message}"),
        }
    }
}

impl std::error::Error for Error {}

fn map_env_error_code(code: EnvErrorCode) -> ErrorCode {
    match code {
        EnvErrorCode::Unspecified => ErrorCode::Unspecified,
        EnvErrorCode::Timeout => ErrorCode::Timeout,
        EnvErrorCode::InvalidAction => ErrorCode::InvalidAction,
        EnvErrorCode::NotReady => ErrorCode::NotReady,
        EnvErrorCode::Busy => ErrorCode::Busy,
        EnvErrorCode::Internal => ErrorCode::Internal,
        EnvErrorCode::Crashed => ErrorCode::Crashed,
        EnvErrorCode::Cancelled => ErrorCode::Cancelled,
        EnvErrorCode::Closed => ErrorCode::Closed,
        // EnvErrorCode is #[non_exhaustive]; map unknown codes conservatively.
        _ => ErrorCode::Internal,
    }
}

impl From<EnvError> for EnvironmentError {
    fn from(value: EnvError) -> Self {
        Self {
            code: map_env_error_code(value.code),
            message: value.message,
            is_recoverable: value.is_recoverable,
        }
    }
}

impl From<rlmesh_grpc::error::ProtocolError> for Error {
    fn from(value: rlmesh_grpc::error::ProtocolError) -> Self {
        Self::Internal(value.to_string())
    }
}

impl From<rlmesh_grpc::error::Error> for Error {
    fn from(value: rlmesh_grpc::error::Error) -> Self {
        match value {
            rlmesh_grpc::error::Error::Transport(error) => match error {
                rlmesh_grpc::error::TransportError::InvalidAddress(message) => {
                    Self::Address(message)
                }
                rlmesh_grpc::error::TransportError::BindFailed(message) => Self::Server(message),
                rlmesh_grpc::error::TransportError::ConnectFailed(message) => {
                    Self::Connection(message)
                }
                rlmesh_grpc::error::TransportError::ConnectionClosed => {
                    Self::Connection("connection closed".to_string())
                }
                rlmesh_grpc::error::TransportError::Io(error) => {
                    Self::Connection(error.to_string())
                }
                rlmesh_grpc::error::TransportError::MessageTooLarge { size, max } => {
                    Self::Connection(format!("message too large: {size} > {max}"))
                }
                rlmesh_grpc::error::TransportError::Unavailable(message) => {
                    Self::Connection(message)
                }
                rlmesh_grpc::error::TransportError::Status { code, message } => {
                    Self::Connection(format!("{code:?}: {message}"))
                }
                // TransportError is #[non_exhaustive].
                other => Self::Connection(other.to_string()),
            },
            rlmesh_grpc::error::Error::Protocol(error) => Self::Internal(error.to_string()),
            rlmesh_grpc::error::Error::Environment(error) => {
                Self::Environment(EnvironmentError::from(error))
            }
            rlmesh_grpc::error::Error::Model(error) => Self::Model(ModelError {
                message: error.message,
                is_recoverable: error.is_recoverable,
            }),
            rlmesh_grpc::error::Error::Timeout(duration) => Self::Timeout(duration),
            // A cancelled connect/handshake is a connection-establishment
            // failure (e.g. the server's port is published but it isn't serving
            // yet during boot), not an internal fault -- surface it as Connection
            // so dial-retry treats it as transient.
            rlmesh_grpc::error::Error::Cancelled(message) => Self::Connection(message),
            rlmesh_grpc::error::Error::Client(error) => Self::Connection(error.to_string()),
            // rlmesh_grpc::error::Error is #[non_exhaustive].
            other => Self::Internal(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grpc_model_error_maps_to_model_variant_preserving_recoverability() {
        let grpc = rlmesh_grpc::error::Error::Model(rlmesh_grpc::error::ModelError {
            code: rlmesh_grpc::error::ModelErrorCode::Internal,
            message: "handler declined".to_string(),
            is_recoverable: true,
            debug_info: None,
        });
        match Error::from(grpc) {
            Error::Model(model) => {
                assert_eq!(model.message, "handler declined");
                assert!(model.is_recoverable);
            }
            other => panic!("expected Error::Model, got {other:?}"),
        }
    }

    #[test]
    fn model_error_constructors_set_recoverability() {
        assert!(!Error::model("nope").is_recoverable());
        assert!(Error::model_recoverable("retry").is_recoverable());
    }
}
