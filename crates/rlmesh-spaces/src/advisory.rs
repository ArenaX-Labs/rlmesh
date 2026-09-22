//! Non-fatal adapter advisories with a severity tier.
//!
//! Advisories surface degradations the adapter tolerated instead of failing
//! on: the consumer (an authoring warning, a serve-time log, an eval harness)
//! decides what to do with them. The severity split exists so a harness can
//! hard-fail on the dangerous tier without policing every note: `Caution`
//! marks the adapter *substituting or fabricating* model-visible data (a
//! role-rebound camera, a zero-filled frame), `Info` marks benign hints and
//! lossy-but-requested steps (layout nudges, no-op ranges, crop/letterbox).

use std::fmt;

/// How much an advisory should worry an automated consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvisorySeverity {
    /// A benign note: an authoring nudge or an explicitly-requested lossy step.
    Info,
    /// The adapter substituted or fabricated data the model consumes; the run
    /// proceeds, but the model may not be seeing what its author expects.
    Caution,
}

impl AdvisorySeverity {
    /// The stable lowercase wire/API name (`"info"` / `"caution"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Caution => "caution",
        }
    }
}

/// One non-fatal note raised at join, resolve, or plan derivation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Advisory {
    pub severity: AdvisorySeverity,
    pub message: String,
}

impl Advisory {
    /// A benign note (see [`AdvisorySeverity::Info`]).
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            severity: AdvisorySeverity::Info,
            message: message.into(),
        }
    }

    /// A data-substitution/fabrication note (see [`AdvisorySeverity::Caution`]).
    pub fn caution(message: impl Into<String>) -> Self {
        Self {
            severity: AdvisorySeverity::Caution,
            message: message.into(),
        }
    }
}

impl fmt::Display for Advisory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
