#[allow(unused_imports)]
use crate::prelude::*;

use core::fmt;

/// A stable category for a resolution failure — the cross-language surface.
///
/// Unlike the human-readable [message](Error::message), which is a snapshot,
/// the code is meant to be matched on by callers and by other-language
/// bindings.
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
    /// The two sides disagree about a geometry attribute (`frame` or `reference`), or one declares a
    /// value outside its vocabulary.
    FrameMismatch,
    /// A model image asserts a `render` size the bound camera does not render at.
    RenderMismatch,
    /// A leaf declares a kind this core does not define; the fix is to upgrade the runtime, unlike
    /// [`MissingRole`](Self::MissingRole).
    UnsupportedKind,
    /// The two sides' axis labels cannot be aligned: one names labels and the other none, or the sets differ.
    /// Always an error, never a caution.
    LabelMismatch,
    /// The two sides disagree about where a state leaf's numbers come from (`sensed`, `estimated`,
    /// `privileged`), or one declares a value outside that vocabulary.
    ProvenanceMismatch,
    /// A seeker that pins no provenance found the role under several; the model must declare `provenance=`.
    Ambiguous,
    /// An env observation tag declares a role under `action/`; a model reads its own previous command
    /// with `source="action"`.
    ActionRoleOnObservation,
    /// The two sides declare a different gripper `polarity` or delta `unit` and the model spec allows no
    /// conversion (`flip_gripper`, `convert_units`).
    ConventionMismatch,
}

/// A model input or action component has no usable counterpart in the env spec, or a spec declares
/// something impossible.
///
/// The [`message`](Self::message) wording is not a stable contract; match on [`code`](Self::code) instead.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct Error {
    /// Stable category for structural matching.
    pub code: ErrorCode,
    /// Human-readable description of the failed pairing.
    pub message: String,
}

impl Error {
    /// A failure of `code`.
    pub fn new(code: ErrorCode, message: String) -> Self {
        Self { code, message }
    }
}

/// How much an advisory should worry an automated consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    /// A benign note: an authoring nudge or an explicitly-requested lossy step.
    Info,
    /// The binding substituted or fabricated data the model consumes; the run proceeds.
    Caution,
}

impl Severity {
    /// The stable lowercase name (`"info"` / `"caution"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Caution => "caution",
        }
    }
}

/// One non-fatal note raised while lowering or binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Advisory {
    /// Its tier.
    pub severity: Severity,
    /// What it says.
    pub message: String,
}

impl Advisory {
    /// A benign note (see [`Severity::Info`]).
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Info,
            message: message.into(),
        }
    }

    /// A data-substitution/fabrication note (see [`Severity::Caution`]).
    pub fn caution(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Caution,
            message: message.into(),
        }
    }
}

impl fmt::Display for Advisory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
