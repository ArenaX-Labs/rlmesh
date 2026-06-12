//! Apply-time error type.

/// A resolved plan could not be applied to a concrete observation or
/// action value (missing keys, malformed shapes, unsupported dtypes).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ApplyError {
    /// Human-readable description of the failure.
    pub message: String,
}

impl ApplyError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
