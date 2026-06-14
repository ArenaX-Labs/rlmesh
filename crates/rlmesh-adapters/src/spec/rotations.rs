//! Rotation encoding vocabulary and per-encoding dimensions.

use serde::{Deserialize, Serialize};

/// Rotation representation of a state feature or action component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RotationEncoding {
    QuatXyzw,
    QuatWxyz,
    AxisAngle,
    /// The standard 6D rotation (Zhou et al. 2019): the rotation matrix's first
    /// two columns concatenated, `[m00, m10, m20, m01, m11, m21]`.
    Rot6d,
    /// The same first-two-columns 6D rotation, but flattened row-major over the
    /// `(3, 2)` column block: `[m00, m01, m10, m11, m20, m21]`. A non-standard
    /// interleaving some checkpoints (e.g. X-VLA proprio) were trained on; kept
    /// explicit so [`Rot6d`](Self::Rot6d) can stay the standard convention.
    // serde's snake_case would yield `rot6d_row_major`; pin it to match
    // `as_str` and the Python `RotationEncoding` literal.
    #[serde(rename = "rot6d_rowmajor")]
    Rot6dRowMajor,
    /// Roll-pitch-yaw `[roll, pitch, yaw]` (radians), extrinsic XYZ:
    /// `R = Rz(yaw) * Ry(pitch) * Rx(roll)` (the ROS / scipy lowercase
    /// `'xyz'` convention; pitch is recovered in `[-pi/2, pi/2]`). Other
    /// Euler conventions are not built in -- use a custom input for them.
    EulerXyz,
}

impl RotationEncoding {
    /// Every encoding, for consumers exporting the vocabulary.
    pub const ALL: [Self; 6] = [
        Self::QuatXyzw,
        Self::QuatWxyz,
        Self::AxisAngle,
        Self::Rot6d,
        Self::Rot6dRowMajor,
        Self::EulerXyz,
    ];

    /// Definitional width of this encoding (the `ROTATION_DIMS` law).
    pub const fn dims(self) -> u32 {
        match self {
            Self::QuatXyzw | Self::QuatWxyz => 4,
            Self::AxisAngle | Self::EulerXyz => 3,
            Self::Rot6d | Self::Rot6dRowMajor => 6,
        }
    }

    /// Wire/display name (matches the JSON form).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QuatXyzw => "quat_xyzw",
            Self::QuatWxyz => "quat_wxyz",
            Self::AxisAngle => "axis_angle",
            Self::Rot6d => "rot6d",
            Self::Rot6dRowMajor => "rot6d_rowmajor",
            Self::EulerXyz => "euler_xyz",
        }
    }
}
