//! Resolution error type.

/// A model input or action component has no usable counterpart in the env
/// spec (or a spec declares something definitionally impossible).
///
/// Messages are part of the cross-implementation contract: the conformance
/// vectors pin substrings, so they must match the reference implementation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct AdapterResolutionError {
    /// Human-readable description of the failed pairing.
    pub message: String,
}

impl AdapterResolutionError {
    pub(crate) fn new(message: String) -> Self {
        Self { message }
    }
}
