//! Resolution and apply error types.

pub use rlmesh_session_core::spec::role::{Error as AdapterResolutionError, ErrorCode};

/// A resolved plan could not be applied to a concrete observation or
/// action value (missing keys, malformed shapes, unsupported dtypes).
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct ApplyError {
    /// Original host callback failure, preserved across the native apply seam.
    #[source]
    pub source: Option<std::sync::Arc<dyn std::error::Error + Send + Sync>>,
    /// Human-readable description of the failure.
    pub message: String,
}

impl ApplyError {
    /// Construct an apply failure. Public so other-language bindings can build
    /// one when implementing [`CustomTransform`](crate::apply::CustomTransform) /
    /// [`EncodingTransform`](crate::stateful::EncodingTransform) host holes.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }
}

impl PartialEq for ApplyError {
    fn eq(&self, other: &Self) -> bool {
        self.message == other.message
            && self.source.as_ref().map(ToString::to_string)
                == other.source.as_ref().map(ToString::to_string)
    }
}
impl Eq for ApplyError {}
