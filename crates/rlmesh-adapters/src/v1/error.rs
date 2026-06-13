//! Resolution error type.

/// A model input or action component has no usable counterpart in the env
/// spec (or a spec declares something definitionally impossible).
///
/// The message is a human-readable *reference snapshot*: the conformance
/// vectors pin substrings so implementations stay consistent, but it is not a
/// stable cross-language contract. Structural callers (and other-language
/// bindings) should categorize on the error rather than parse this text;
/// join-time failures already carry the typed [`JoinError`](super::JoinError).
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
