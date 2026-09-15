//! Resolution error type.

/// A stable category for a resolution failure — the cross-language surface.
///
/// Unlike the human-readable [message](AdapterResolutionError::message), which
/// is a snapshot, the code is meant to be matched on by callers and by
/// other-language bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorCode {
    /// The env tags could not be joined against the spaces.
    InvalidTag,
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
    /// The two sides disagree about a geometry attribute, or one of them
    /// declares a value outside its vocabulary: an absolute pose in a frame the
    /// model was not trained on (`frame`), or a delta integrated against the
    /// other reference pose (`reference`). One code covers both attributes --
    /// they are one rule with two vocabularies, and a caller matching "the
    /// declared geometry disagrees" should not have to match two names; the
    /// message says which attribute and which role.
    FrameMismatch,
    /// A model image asserts a `render` size the camera it bound does not
    /// render at — the env's camera dial was never moved to the resolution the
    /// model was trained on.
    RenderMismatch,
    /// A leaf declares a *kind* this core does not define: a model input of an
    /// unrecognized kind, or a model input whose role the env offers only as an
    /// unrecognized observation kind. The fix is to upgrade the runtime, not to
    /// edit the spec — distinct from [`MissingRole`](Self::MissingRole), which
    /// asks the author to add a role.
    UnsupportedKind,
    /// The two sides' axis labels cannot be aligned: the model names labels
    /// and the env leaf declares none, or the model names a label the env
    /// lacks, or a model subset would leave env axes undriven on an actuator
    /// that is not `optional`. Structural, never a caution: a joint vector
    /// bound on hope is the bug labels exist to catch.
    LabelMismatch,
    /// The two sides disagree about where a state leaf's numbers come from
    /// (`sensed`, `estimated`, `privileged`), or one of them declares a value
    /// outside that vocabulary: a checkpoint trained on privileged simulator
    /// state bound to an estimate on the robot is the pairing this refuses.
    ProvenanceMismatch,
    /// A seeker that pins no provenance found the role under several: a sim
    /// that publishes its truth beside its estimate leaves nothing to guess
    /// through, so the model must declare `provenance=`.
    Ambiguous,
}

/// A model input or action component has no usable counterpart in the env
/// spec (or a spec declares something definitionally impossible).
///
/// The [`message`](Self::message) is a human-readable *reference snapshot*: the
/// conformance vectors pin substrings so implementations stay consistent, but
/// it is not a stable cross-language contract. Match on [`code`](Self::code)
/// instead (other-language bindings should too); join-time failures
/// additionally carry the typed [`JoinError`](crate::join::JoinError).
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

/// A resolved plan could not be applied to a concrete observation or
/// action value (missing keys, malformed shapes, unsupported dtypes).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ApplyError {
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
        }
    }
}
