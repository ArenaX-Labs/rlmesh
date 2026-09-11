//! The two geometry attributes a pose-shaped role can carry: `frame` and
//! `reference`.
//!
//! Both answer "relative to what?" for a quantity whose numbers are otherwise
//! indistinguishable:
//!
//! - **`frame`** qualifies an *absolute* pose (`proprio/eef_*`,
//!   `action/eef_*`): the coordinate frame its components are expressed in.
//! - **`reference`** qualifies a *delta* command (`action/delta_eef_*`): the
//!   pose the env's controller integrates the delta against — the measured
//!   pose (`current`) or the last commanded target (`target`). A delta carries
//!   no `frame`: it is expressed in the controller's own frame by definition.
//!
//! They are one mechanism with two vocabularies ([`Attr`]), so every rule —
//! the tolerant string codec, the resolve-time agreement check, the publish
//! gate, the describe suffix — is written once and parameterized.
//!
//! **Tolerant vocabulary.** A value outside the vocabulary parses and
//! round-trips verbatim (a newer peer's frame survives relay) but is rejected
//! at *resolve*, where a frame this core cannot reason about is a geometry it
//! cannot verify. Same shape as an unrecognized rotation encoding.

use serde::{Deserialize, Serialize};

/// The recognized reference frames: the `Frame` vocabulary
/// (`Literal["world", "robot_base"]` on the Python side).
pub const FRAMES: [&str; 2] = ["world", "robot_base"];

/// The recognized delta references: what an env's Cartesian controller
/// integrates a delta against, and what a policy was trained against.
pub const REFERENCES: [&str; 2] = ["current", "target"];

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

/// Which geometry attribute a rule is about — the only axis on which `frame`
/// and `reference` differ. The rules themselves are one implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attr {
    Frame,
    Reference,
}

impl Attr {
    /// The wire/field name, used verbatim in every message.
    pub fn name(self) -> &'static str {
        match self {
            Attr::Frame => "frame",
            Attr::Reference => "reference",
        }
    }

    /// The recognized values for this attribute.
    pub fn vocabulary(self) -> &'static [&'static str] {
        match self {
            Attr::Frame => &FRAMES,
            Attr::Reference => &REFERENCES,
        }
    }

    /// The `describe` sigil: `@robot_base` for a frame, `~target` for a
    /// reference.
    pub fn sigil(self) -> char {
        match self {
            Attr::Frame => '@',
            Attr::Reference => '~',
        }
    }

    /// Whether `value` is in this attribute's vocabulary.
    pub fn recognizes(self, value: &FrameRef) -> bool {
        self.vocabulary().contains(&value.as_str())
    }
}

/// Whether a registered role's values are expressed in a reference frame.
///
/// Validation only, and only at the opt-in [`FramePolicy::Require`] tier
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
