//! The check-only attributes a state leaf can carry: `frame`, `reference` and `provenance`.
//!
//! - **`frame`** qualifies an absolute pose (`proprio/eef_*`, `action/eef_*`): the coordinate frame
//!   its components are expressed in.
//! - **`reference`** qualifies a delta command (`action/delta_eef_*`): the pose the controller
//!   integrates it against, the measured pose (`current`) or the last commanded target (`target`).
//!   A delta may also carry a `frame`; declared frames must agree.
//! - **`provenance`** says where a number came from: `sensed`, `estimated` or `privileged`
//!   (simulator truth). It stops a checkpoint trained on privileged state binding an estimate.
//!   An env declares one value per leaf; a model declares one, or the set it accepts.
//!
//! The three share one mechanism ([`Attr`]). A value outside a vocabulary parses and round-trips
//! verbatim but is rejected at resolve.

#[allow(unused_imports)]
use crate::prelude::*;

use serde::{Deserialize, Serialize};

/// The recognized reference frames: the `Frame` vocabulary
/// (`Literal["world", "robot_base", "tool"]` on the Python side). `tool` is
/// the frame attached to the end effector (the flange or TCP), where a
/// wrist-mounted force/torque sensor reads.
pub const FRAMES: [&str; 3] = ["world", "robot_base", "tool"];

/// The recognized delta references: what an env's Cartesian controller
/// integrates a delta against, and what a policy was trained against.
pub const REFERENCES: [&str; 2] = ["current", "target"];

/// The recognized provenances: where a state leaf's numbers come from.
pub const PROVENANCES: [&str; 3] = ["sensed", "estimated", "privileged"];

/// A provenance value, closed: the model side declares an
/// `AcceptSet` of these (a bare string when one), so an
/// unknown value round-trips and fails at resolve like an unknown encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// A physical sensor, or its simulated equivalent.
    Sensed,
    /// A state estimator's output.
    Estimated,
    /// Simulator truth with no hardware counterpart.
    Privileged,
}

impl Provenance {
    /// Every value, for consumers exporting the vocabulary.
    pub const ALL: [Self; 3] = [Self::Sensed, Self::Estimated, Self::Privileged];

    /// Wire/display name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sensed => "sensed",
            Self::Estimated => "estimated",
            Self::Privileged => "privileged",
        }
    }
}

impl Provenance {
    /// The provenance `name` names.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|value| value.as_str() == name)
    }
}

/// A declared geometry value, held as a tolerant string (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FrameRef(String);

impl FrameRef {
    /// The declared value, verbatim.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for FrameRef {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl core::fmt::Display for FrameRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Which attribute a rule is about — the only axis on which `frame`,
/// `reference` and `provenance` differ. The rules themselves are one
/// implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attr {
    /// The frame an absolute pose is expressed in.
    Frame,
    /// The pose a delta is integrated against.
    Reference,
    /// Where a state leaf's numbers come from.
    Provenance,
}

impl Attr {
    /// The wire/field name, used verbatim in every message.
    pub fn name(self) -> &'static str {
        match self {
            Attr::Frame => "frame",
            Attr::Reference => "reference",
            Attr::Provenance => "provenance",
        }
    }

    /// The recognized values for this attribute.
    pub fn vocabulary(self) -> &'static [&'static str] {
        match self {
            Attr::Frame => &FRAMES,
            Attr::Reference => &REFERENCES,
            Attr::Provenance => &PROVENANCES,
        }
    }

    /// The `describe` sigil: `@robot_base` for a frame, `~target` for a
    /// reference, `#sensed` for a provenance (after any `#part`).
    pub fn sigil(self) -> char {
        match self {
            Attr::Frame => '@',
            Attr::Reference => '~',
            Attr::Provenance => '#',
        }
    }

    /// Whether `value` is in this attribute's vocabulary.
    pub fn recognizes(self, value: &FrameRef) -> bool {
        self.vocabulary().contains(&value.as_str())
    }
}

/// Whether a registered role's values are expressed in a reference frame.
///
/// Validation only, and only at the publish gate's opt-in require-frames tier
/// (`rlmesh-adapters`' `FramePolicy::Require`): nothing here supplies a frame, and a role
/// that is `Frameless` still has any declared `frame` checked for agreement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameLaw {
    /// No frame is required: a scalar, a joint vector, or a delta (qualified by [`ReferenceLaw`]).
    Frameless,
    /// An absolute Cartesian pose: its numbers mean nothing without their frame.
    Framed,
}

/// Whether a registered role is a delta integrated against a reference pose —
/// the delta-role mirror of [`FrameLaw`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceLaw {
    /// Not a delta: nothing is integrated, so there is no reference pose.
    Unreferenced,
    /// A per-step delta, qualified by the reference pose the controller adds it to.
    Referenced,
}

#[cfg(test)]
mod tests {
    use super::{Attr, FrameRef, PROVENANCES, Provenance};

    #[test]
    fn provenance_is_a_closed_vocabulary() {
        assert_eq!(
            Provenance::ALL.map(Provenance::as_str),
            PROVENANCES,
            "the enum and the vocabulary list drift"
        );
        for value in PROVENANCES {
            assert!(Attr::Provenance.recognizes(&FrameRef::from(value)));
        }
        assert!(!Attr::Provenance.recognizes(&FrameRef::from("guessed")));
    }
}
