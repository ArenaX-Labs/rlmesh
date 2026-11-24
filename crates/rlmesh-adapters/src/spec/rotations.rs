use serde::{Deserialize, Serialize};

/// Rotation representation of a state feature or action component.
///
/// The vocabulary is closed on both sides. On the observation side an unknown value parses,
/// round-trips and fails at resolve naming it; on the action side
/// [`ActionEncoding`](super::ActionEncoding) rejects an unknown value at parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RotationEncoding {
    QuatXyzw,
    QuatWxyz,
    AxisAngle,
    /// The standard 6D rotation (Zhou et al. 2019): the rotation matrix's first
    /// two columns concatenated, `[m00, m10, m20, m01, m11, m21]`.
    Rot6d,
    #[serde(rename = "rot6d_rowmajor")]
    Rot6dRowMajor,
    /// Roll-pitch-yaw `[roll, pitch, yaw]` (radians), extrinsic XYZ:
    /// `R = Rz(yaw) * Ry(pitch) * Rx(roll)` (the ROS / scipy lowercase
    /// `'xyz'` convention; pitch is recovered in `[-pi/2, pi/2]`).
    EulerXyz,
    /// Projected gravity: the gravity direction expressed in the body frame,
    /// `R_world_from_base^T · (0, 0, -1)`, the 3-vector every locomotion
    /// checkpoint reads instead of an orientation. An identity orientation yields `(0, 0, -1)`.
    ///
    /// A sink, not a rotation: any encoding converts into it, but it carries only a direction,
    /// so nothing converts out of it. A `gravity_xyz` source binds only a reader that wants
    /// `gravity_xyz`, a `post_rotate` cannot decode it, and no action component can use it.
    GravityXyz,
}

impl RotationEncoding {
    /// Every encoding, for consumers exporting the vocabulary.
    pub const ALL: [Self; 7] = [
        Self::QuatXyzw,
        Self::QuatWxyz,
        Self::AxisAngle,
        Self::Rot6d,
        Self::Rot6dRowMajor,
        Self::EulerXyz,
        Self::GravityXyz,
    ];

    /// Definitional width of this encoding (the `ROTATION_DIMS` law).
    pub const fn dims(self) -> u32 {
        match self {
            Self::QuatXyzw | Self::QuatWxyz => 4,
            Self::AxisAngle | Self::EulerXyz | Self::GravityXyz => 3,
            Self::Rot6d | Self::Rot6dRowMajor => 6,
        }
    }

    /// Whether this encoding is a sink: a direction a rotation projects to,
    /// which nothing decodes back into a rotation (the direction law).
    pub const fn is_sink(self) -> bool {
        matches!(self, Self::GravityXyz)
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
            Self::GravityXyz => "gravity_xyz",
        }
    }
}

impl crate::spec::accept_set::WireVocab for RotationEncoding {
    fn from_wire(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|encoding| encoding.as_str() == name)
    }

    fn as_wire(self) -> &'static str {
        self.as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::RotationEncoding;

    #[test]
    fn known_vocab_roundtrips_and_serde_matches_as_str() {
        for encoding in RotationEncoding::ALL {
            let json = serde_json::to_string(&encoding).expect("serialize");
            assert_eq!(json.trim_matches('"'), encoding.as_str());
            let back: RotationEncoding = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, encoding);
        }
    }

    #[test]
    fn unknown_encoding_is_rejected() {
        assert!(serde_json::from_str::<RotationEncoding>("\"rot10d\"").is_err());
    }

    #[test]
    fn gravity_is_the_only_sink_and_is_three_wide() {
        assert_eq!(RotationEncoding::GravityXyz.dims(), 3);
        let sinks: Vec<_> = RotationEncoding::ALL
            .into_iter()
            .filter(|encoding| encoding.is_sink())
            .collect();
        assert_eq!(sinks, vec![RotationEncoding::GravityXyz]);
    }
}
