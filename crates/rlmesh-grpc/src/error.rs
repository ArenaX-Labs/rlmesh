//! Error types for concrete RLMesh gRPC clients and servers.

use std::time::Duration;
use thiserror::Error;

/// Top-level error type for rlmesh-grpc operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Transport-level error (connection, I/O, etc.)
    #[error("transport error: {0}")]
    Transport(#[from] TransportError),

    /// Protocol-level error (encoding, framing, etc.)
    #[error("protocol error: {0}")]
    Protocol(#[from] ProtocolError),

    /// Environment error (from the environment itself)
    #[error("environment error: {0}")]
    Environment(#[from] EnvError),

    /// Model error (from a served model handler)
    #[error("model error: {0}")]
    Model(#[from] ModelError),

    /// Operation timed out
    #[error("timeout after {0:?}")]
    Timeout(Duration),

    /// Operation was cancelled
    #[error("cancelled: {0}")]
    Cancelled(String),

    /// Client error
    #[error("client error: {0}")]
    Client(#[from] ClientError),
}

impl Error {
    /// Check if this error is recoverable (can retry).
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Timeout(_) => true,
            Self::Environment(error) => error.is_recoverable,
            Self::Model(error) => error.is_recoverable,
            Self::Transport(TransportError::Unavailable(_)) => true,
            Self::Transport(TransportError::Status {
                code: tonic::Code::DeadlineExceeded,
                ..
            }) => true,
            Self::Transport(TransportError::Io(_)) => false,
            Self::Transport(TransportError::ConnectionClosed) => false,
            _ => false,
        }
    }
}

/// Map a `tonic::Status` from an established connection into a structured
/// [`enum@Error`], preserving the gRPC status code so callers can distinguish a
/// retryable condition (e.g. `Unavailable`) from a permanent one (e.g.
/// `Unimplemented`) instead of seeing every failure as `failed to connect`.
pub fn status_to_grpc_error(status: tonic::Status) -> Error {
    use tonic::Code;

    let code = status.code();
    let message = status.message().to_string();
    match code {
        Code::Unavailable | Code::ResourceExhausted | Code::Aborted => {
            Error::Transport(TransportError::Unavailable(message))
        }
        Code::Cancelled => Error::Cancelled(message),
        // Everything else keeps the structured code. DeadlineExceeded stays
        // recoverable without fabricating a Timeout(0ns) duration.
        _ => Error::Transport(TransportError::Status { code, message }),
    }
}

/// Transport-level errors.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TransportError {
    /// I/O error
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Connection was closed
    #[error("connection closed")]
    ConnectionClosed,

    /// Failed to bind to address
    #[error("failed to bind: {0}")]
    BindFailed(String),

    /// Failed to connect
    #[error("failed to connect: {0}")]
    ConnectFailed(String),

    /// Invalid address format
    #[error("invalid address: {0}")]
    InvalidAddress(String),

    /// Message too large
    #[error("message too large: {size} > {max}")]
    MessageTooLarge { size: usize, max: usize },

    /// Server is temporarily unavailable (retryable).
    #[error("server unavailable: {0}")]
    Unavailable(String),

    /// A gRPC status returned on an established connection, preserving its code.
    #[error("grpc status {code:?}: {message}")]
    Status {
        /// The gRPC status code.
        code: tonic::Code,
        /// The status message.
        message: String,
    },
}

/// Protocol-level errors.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProtocolError {
    /// Failed to encode message
    #[error("encode error: {0}")]
    EncodeError(String),

    /// Failed to decode message
    #[error("decode error: {0}")]
    DecodeError(String),

    /// Handshake failed
    #[error("handshake failed: {0}")]
    HandshakeFailed(String),

    /// Unexpected message
    #[error("unexpected message: expected {expected}, got {actual}")]
    UnexpectedMessage { expected: String, actual: String },
}

/// Environment errors (from the environment itself).
#[derive(Debug, Error)]
pub struct EnvError {
    /// Error code
    pub code: EnvErrorCode,
    /// Human-readable message
    pub message: String,
    /// Whether this error is recoverable
    pub is_recoverable: bool,
    /// Debug information
    pub debug_info: Option<String>,
}

impl std::fmt::Display for EnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.code, self.message)
    }
}

/// Environment error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EnvErrorCode {
    /// Unspecified error
    Unspecified,
    /// Operation timed out
    Timeout,
    /// Invalid action
    InvalidAction,
    /// Environment not ready (needs reset)
    NotReady,
    /// Environment busy with another operation
    Busy,
    /// Internal error
    Internal,
    /// Environment crashed
    Crashed,
    /// Operation was cancelled
    Cancelled,
    /// Environment was closed
    Closed,
}

impl EnvError {
    /// Create a new environment error.
    pub fn new(code: EnvErrorCode, message: impl Into<String>) -> Self {
        let is_recoverable = matches!(
            code,
            EnvErrorCode::Timeout
                | EnvErrorCode::InvalidAction
                | EnvErrorCode::NotReady
                | EnvErrorCode::Busy
        );
        Self {
            code,
            message: message.into(),
            is_recoverable,
            debug_info: None,
        }
    }
}

impl From<rlmesh_proto::env::v1::EnvErrorCode> for EnvErrorCode {
    fn from(code: rlmesh_proto::env::v1::EnvErrorCode) -> Self {
        use rlmesh_proto::env::v1::EnvErrorCode as ProtoCode;
        match code {
            ProtoCode::Unspecified => EnvErrorCode::Unspecified,
            ProtoCode::Timeout => EnvErrorCode::Timeout,
            ProtoCode::InvalidAction => EnvErrorCode::InvalidAction,
            ProtoCode::NotReady => EnvErrorCode::NotReady,
            ProtoCode::Busy => EnvErrorCode::Busy,
            ProtoCode::Internal => EnvErrorCode::Internal,
            ProtoCode::Crashed => EnvErrorCode::Crashed,
            ProtoCode::Cancelled => EnvErrorCode::Cancelled,
            ProtoCode::Closed => EnvErrorCode::Closed,
        }
    }
}

impl From<EnvErrorCode> for rlmesh_proto::env::v1::EnvErrorCode {
    fn from(code: EnvErrorCode) -> Self {
        use rlmesh_proto::env::v1::EnvErrorCode as ProtoCode;
        match code {
            EnvErrorCode::Unspecified => ProtoCode::Unspecified,
            EnvErrorCode::Timeout => ProtoCode::Timeout,
            EnvErrorCode::InvalidAction => ProtoCode::InvalidAction,
            EnvErrorCode::NotReady => ProtoCode::NotReady,
            EnvErrorCode::Busy => ProtoCode::Busy,
            EnvErrorCode::Internal => ProtoCode::Internal,
            EnvErrorCode::Crashed => ProtoCode::Crashed,
            EnvErrorCode::Cancelled => ProtoCode::Cancelled,
            EnvErrorCode::Closed => ProtoCode::Closed,
        }
    }
}

/// Model errors (from a served model handler).
#[derive(Debug, Error)]
pub struct ModelError {
    /// Error code
    pub code: ModelErrorCode,
    /// Human-readable message
    pub message: String,
    /// Whether this error is recoverable
    pub is_recoverable: bool,
    /// Debug information
    pub debug_info: Option<String>,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.code, self.message)
    }
}

/// Model error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelErrorCode {
    /// Unspecified error
    Unspecified,
    /// The request was invalid
    InvalidRequest,
    /// The route was not configured
    NotConfigured,
    /// The model is busy with another operation
    Busy,
    /// Internal error
    Internal,
    /// Operation was cancelled
    Cancelled,
    /// The model/route was closed
    Closed,
}

impl From<rlmesh_proto::model::v1::ModelErrorCode> for ModelErrorCode {
    fn from(code: rlmesh_proto::model::v1::ModelErrorCode) -> Self {
        use rlmesh_proto::model::v1::ModelErrorCode as ProtoCode;
        match code {
            ProtoCode::Unspecified => ModelErrorCode::Unspecified,
            ProtoCode::InvalidRequest => ModelErrorCode::InvalidRequest,
            ProtoCode::NotConfigured => ModelErrorCode::NotConfigured,
            ProtoCode::Busy => ModelErrorCode::Busy,
            ProtoCode::Internal => ModelErrorCode::Internal,
            ProtoCode::Cancelled => ModelErrorCode::Cancelled,
            ProtoCode::Closed => ModelErrorCode::Closed,
        }
    }
}

/// Client-specific errors.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ClientError {
    /// Not connected
    #[error("not connected")]
    NotConnected,

    /// Handshake not completed
    #[error("handshake not completed")]
    NotHandshaked,
}

/// Result type alias for rlmesh-grpc operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod status_mapping_tests {
    use super::*;
    use tonic::{Code, Status};

    #[test]
    fn unavailable_status_maps_to_recoverable_transport_error() {
        let error = status_to_grpc_error(Status::new(Code::Unavailable, "try later"));
        assert!(error.is_recoverable(), "Unavailable must be retryable");
        match error {
            Error::Transport(TransportError::Unavailable(message)) => {
                assert_eq!(message, "try later");
            }
            other => panic!("expected Unavailable transport error, got {other:?}"),
        }
    }

    #[test]
    fn unimplemented_status_preserves_code_and_is_not_recoverable() {
        let error = status_to_grpc_error(Status::new(Code::Unimplemented, "no such method"));
        assert!(!error.is_recoverable(), "Unimplemented must be permanent");
        match error {
            Error::Transport(TransportError::Status { code, message }) => {
                assert_eq!(code, Code::Unimplemented);
                assert_eq!(message, "no such method");
            }
            other => panic!("expected structured Status error, got {other:?}"),
        }
        // It must NOT be misreported as a connect failure.
        let error = status_to_grpc_error(Status::new(Code::Unimplemented, "x"));
        assert!(!error.to_string().contains("failed to connect"));
    }

    #[test]
    fn deadline_exceeded_status_keeps_code_and_recoverability() {
        let error = status_to_grpc_error(Status::new(Code::DeadlineExceeded, "slow"));
        assert!(matches!(
            error,
            Error::Transport(TransportError::Status {
                code: Code::DeadlineExceeded,
                ..
            })
        ));
        assert!(error.is_recoverable());
        // The message must not fabricate a zero duration.
        assert!(!error.to_string().contains("0ns"));
        assert!(error.to_string().contains("slow"));
    }
}
