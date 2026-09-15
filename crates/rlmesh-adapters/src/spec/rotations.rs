//! Rotation encoding vocabulary and per-encoding dimensions.

use serde::{Deserialize, Serialize};

/// Rotation representation of a state feature or action component.
///
/// **Adding a value.** The vocabulary is closed on both sides, but the two
/// sides tolerate a new value differently, and that decides what a new value
/// costs. On the observation side the field is an [`AcceptSet`](super::AcceptSet):
/// an old core parses a value it does not know as an unknown entry, round-trips
/// it, and fails at *resolve* naming it, so an observation-side value (such as
/// [`GravityXyz`](Self::GravityXyz)) mints no edition and no key bump. On the
/// action side [`ActionEncoding`](super::ActionEncoding) is rigid (a single
/// value, rejected at parse when unknown), so a value a controller can take
/// would need an edition before an old core could read a spec that names it.
/// The TS/FE binding models this field as an open `string`, so it degrades
/// gracefully regardless of the Rust representation.
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
    /// Projected gravity: the gravity direction expressed in the body frame,
    /// `R_world_from_base^T · (0, 0, -1)`, the 3-vector every locomotion
    /// checkpoint reads instead of an orientation. An identity orientation
    /// yields `(0, 0, -1)`; a robot on its back yields `(0, 0, 1)`.
    ///
    /// **A sink, not a rotation.** Any rotation encoding converts *into* it,
    /// but it carries only a direction (the yaw is gone), so nothing converts
    /// *out of* it: a source declared `gravity_xyz` binds only a reader that
    /// wants `gravity_xyz`, a `post_rotate` cannot decode it, and an action
    /// component cannot be expressed in it.
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
            // The serde wire string must equal as_str() for every variant: this
            // pins the frozen vocabulary and catches a rename that drifts the
            // serde form away from as_str() (and the Python literal).
            assert_eq!(json.trim_matches('"'), encoding.as_str());
            let back: RotationEncoding = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, encoding);
        }
    }

    #[test]
    fn unknown_encoding_is_rejected() {
        // The bare enum rejects an unrecognized value at parse; the
        // observation-side `AcceptSet` is what tolerates one until resolve.
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
