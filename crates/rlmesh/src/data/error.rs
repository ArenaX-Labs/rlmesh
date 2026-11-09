//! Internal driver and environment error types.
use thiserror::Error;

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

    /// An integer value did not fit its declared dtype on encode.
    #[error("out of range: {0}")]
    OutOfRange(String),

    /// A leaf or batch-slab byte length did not match the spec.
    #[error("length mismatch: {0}")]
    LengthMismatch(String),

    /// Handshake failed
    #[error("handshake failed: {0}")]
    HandshakeFailed(String),

    /// Unexpected message
    #[error("unexpected message: expected {expected}, got {actual}")]
    UnexpectedMessage { expected: String, actual: String },
}

impl From<rlmesh_spaces::leaves::LeafError> for ProtocolError {
    fn from(error: rlmesh_spaces::leaves::LeafError) -> Self {
        use rlmesh_spaces::leaves::LeafError;
        match error {
            LeafError::Encode(message) => Self::EncodeError(message),
            LeafError::Decode(message) => Self::DecodeError(message),
            LeafError::OutOfRange(message) => Self::OutOfRange(message),
            LeafError::LengthMismatch(message) => Self::LengthMismatch(message),
            other => Self::DecodeError(other.to_string()),
        }
    }
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
    /// Well-formed request for a feature this peer does not support yet
    Unsupported,
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
                | EnvErrorCode::Unsupported
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
            ProtoCode::Unsupported => EnvErrorCode::Unsupported,
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
            EnvErrorCode::Unsupported => ProtoCode::Unsupported,
            EnvErrorCode::Internal => ProtoCode::Internal,
            EnvErrorCode::Crashed => ProtoCode::Crashed,
            EnvErrorCode::Cancelled => ProtoCode::Cancelled,
            EnvErrorCode::Closed => ProtoCode::Closed,
        }
    }
}
