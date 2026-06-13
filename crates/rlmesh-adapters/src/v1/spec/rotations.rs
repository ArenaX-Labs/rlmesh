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
    /// Roll-pitch-yaw `[roll, pitch, yaw]` (radians), extrinsic XYZ:
    /// `R = Rz(yaw) * Ry(pitch) * Rx(roll)` (the ROS / scipy lowercase
    /// `'xyz'` convention; pitch is recovered in `[-pi/2, pi/2]`). Other
    /// Euler conventions are not built in -- use a custom input for them.
    EulerXyz,
}

impl RotationEncoding {
    /// Every encoding, for consumers exporting the vocabulary.
    pub const ALL: [Self; 5] = [
        Self::QuatXyzw,
        Self::QuatWxyz,
        Self::AxisAngle,
        Self::Rot6d,
        Self::EulerXyz,
    ];

    /// Definitional width of this encoding (the `ROTATION_DIMS` law).
    pub const fn dims(self) -> u32 {
        match self {
            Self::QuatXyzw | Self::QuatWxyz => 4,
            Self::AxisAngle | Self::EulerXyz => 3,
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
            Self::EulerXyz => "euler_xyz",
        }
    }
}
