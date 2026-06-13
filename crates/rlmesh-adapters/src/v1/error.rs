//! Resolution error type.

/// A stable category for a resolution failure — the cross-language surface.
///
/// Unlike the human-readable [message](AdapterResolutionError::message), which
/// is a snapshot, the code is meant to be matched on by callers and by
/// other-language bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorCode {
    /// The env annotations could not be joined against the spaces.
    InvalidAnnotation,
    /// Two features, inputs, or action components claim the same role or key.
    Duplicate,
    /// A model input or action component found no env counterpart by role.
    MissingRole,
    /// A declared width or dim disagrees (action width, encoding dims, ...).
    DimMismatch,
    /// Two rotation encodings cannot be converted between.
    EncodingMismatch,
    /// A slice index or dim falls outside the source feature's width.
    SliceOutOfRange,
    /// A required sizing hint is absent (an optional component's zero-fill).
    MissingWidth,
    /// An option value is not supported (e.g. a resample algorithm).
    Unsupported,
    /// A custom-input entrypoint was referenced without trust.
    UntrustedEntrypoint,
}

/// A model input or action component has no usable counterpart in the env
/// spec (or a spec declares something definitionally impossible).
///
/// The [`message`](Self::message) is a human-readable *reference snapshot*: the
/// conformance vectors pin substrings so implementations stay consistent, but
/// it is not a stable cross-language contract. Match on [`code`](Self::code)
/// instead (other-language bindings should too); join-time failures
/// additionally carry the typed [`JoinError`](super::JoinError).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct AdapterResolutionError {
    /// Stable category for structural matching.
    pub code: ErrorCode,
    /// Human-readable description of the failed pairing.
    pub message: String,
}

impl AdapterResolutionError {
    pub(crate) fn new(code: ErrorCode, message: String) -> Self {
        Self { code, message }
    }
}
