//! Rotation encoding vocabulary and per-encoding dimensions.

use serde::{Deserialize, Serialize};

/// Rotation representation of a state feature or action component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RotationEncoding {
    QuatXyzw,
    QuatWxyz,
    AxisAngle,
    Rot6d,
}

impl RotationEncoding {
    /// Every encoding, for consumers exporting the vocabulary.
    pub const ALL: [Self; 4] = [Self::QuatXyzw, Self::QuatWxyz, Self::AxisAngle, Self::Rot6d];

    /// Definitional width of this encoding (the `ROTATION_DIMS` law).
    pub const fn dims(self) -> u32 {
        match self {
            Self::QuatXyzw | Self::QuatWxyz => 4,
            Self::AxisAngle => 3,
            Self::Rot6d => 6,
        }
    }

    /// Wire/display name (matches the JSON form).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QuatXyzw => "quat_xyzw",
            Self::QuatWxyz => "quat_wxyz",
            Self::AxisAngle => "axis_angle",
            Self::Rot6d => "rot6d",
        }
    }
}
