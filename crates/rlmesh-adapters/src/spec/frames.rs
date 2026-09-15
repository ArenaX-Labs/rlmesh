//! The check-only attributes a state leaf can carry: `frame`, `reference`
//! and `provenance`.
//!
//! The first two answer "relative to what?" for a quantity whose numbers are
//! otherwise indistinguishable:
//!
//! - **`frame`** qualifies an *absolute* pose (`proprio/eef_*`,
//!   `action/eef_*`): the coordinate frame its components are expressed in.
//! - **`reference`** qualifies a *delta* command (`action/delta_eef_*`): the
//!   pose the env's controller integrates the delta against — the measured
//!   pose (`current`) or the last commanded target (`target`). A delta carries
//!   no `frame`: it is expressed in the controller's own frame by definition.
//!
//! **`provenance`** answers "where did the number come from?": a physical
//! sensor or its simulated equivalent (`sensed`), a state estimator
//! (`estimated`), or simulator truth with no hardware counterpart
//! (`privileged`). It is what stops a checkpoint trained on privileged sim
//! state from binding an estimate on the robot without anyone noticing. The
//! env declares one value per leaf and may publish a role under several; a
//! model declares one, or the set it accepts.
//!
//! They are one mechanism with three vocabularies ([`Attr`]), so every rule —
//! the tolerant string codec, the resolve-time agreement check, the publish
//! gate, the describe suffix — is written once and parameterized.
//!
//! **Tolerant vocabulary.** A value outside the vocabulary parses and
//! round-trips verbatim (a newer peer's frame survives relay) but is rejected
//! at *resolve*, where a frame this core cannot reason about is a geometry it
//! cannot verify. Same shape as an unrecognized rotation encoding.

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

impl super::accept_set::WireVocab for Provenance {
    fn from_wire(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|value| value.as_str() == name)
    }

    fn as_wire(self) -> &'static str {
        self.as_str()
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

impl std::fmt::Display for FrameRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Which attribute a rule is about — the only axis on which `frame`,
/// `reference` and `provenance` differ. The rules themselves are one
/// implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attr {
    Frame,
    Reference,
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
/// Validation only, and only at the opt-in [`FramePolicy::Require`](super::strict::FramePolicy::Require) tier
/// (`crate::spec::FramePolicy`): nothing here supplies a frame, and a role
/// that is `Frameless` still has any declared `frame` checked for agreement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameLaw {
    /// No frame applies: a scalar (gripper width), a joint vector (joint space
    /// has no Cartesian frame), or a delta (expressed in the controller's own
    /// frame, and qualified by [`ReferenceLaw`] instead).
    Frameless,
    /// An absolute Cartesian pose component: its numbers mean nothing without
    /// the frame they are expressed in.
    Framed,
}

/// Whether a registered role is a delta integrated against a reference pose —
/// the delta-role mirror of [`FrameLaw`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceLaw {
    /// Not a delta: nothing is integrated, so there is no reference pose.
    Unreferenced,
    /// A per-step delta: the env's controller adds it to a reference pose, and
    /// a policy trained against the other one drifts silently.
    Referenced,
}

#[cfg(test)]
mod tests {
    use super::{Attr, FrameRef, PROVENANCES, Provenance};
    use crate::spec::AcceptSet;

    #[test]
    fn provenance_is_a_closed_vocabulary_the_accept_set_tolerates() {
        assert_eq!(
            Provenance::ALL.map(Provenance::as_str),
            PROVENANCES,
            "the enum and the vocabulary list drift"
        );
        for value in PROVENANCES {
            assert!(Attr::Provenance.recognizes(&FrameRef::from(value)));
        }
        assert!(!Attr::Provenance.recognizes(&FrameRef::from("guessed")));
        // A single value is a bare string on the wire; an unknown one rides
        // along and is simply not a known entry.
        let set: AcceptSet<Provenance> = serde_json::from_str(r#""estimated""#).unwrap();
        assert_eq!(set.first_known(), Some(Provenance::Estimated));
        assert_eq!(serde_json::to_string(&set).unwrap(), r#""estimated""#);
        let set: AcceptSet<Provenance> =
            serde_json::from_str(r#"["privileged", "estimated", "guessed"]"#).unwrap();
        assert_eq!(set.known().count(), 2);
        assert_eq!(set.wire_names(), vec!["privileged", "estimated", "guessed"]);
    }
}
