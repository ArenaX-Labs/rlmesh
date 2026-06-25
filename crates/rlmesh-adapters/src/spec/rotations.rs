//! Rotation encoding vocabulary and per-encoding dimensions.

use serde::{Deserialize, Serialize};

/// Rotation representation of a state feature or action component.
///
/// **Frozen v1 vocabulary.** The known encodings below are part of the v1 wire
/// contract; an unknown encoding string is rejected (today at parse time).
/// Adding an encoding is a v2 key bump (see the version dispatch in
/// [`crate::v1`]) with v1 still readable. Graceful per-field degradation
/// (parse-now / reject-at-resolve via an `Unknown(String)` arm) is an
/// *additive, non-wire-breaking* reader refinement that may land post-freeze —
/// it does not change what a valid v1 document looks like, so it is not gated
/// on the freeze. The TS/FE binding already models this field as an open
/// `string`, so it degrades gracefully regardless of the Rust representation.
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
        // Frozen vocabulary: an unrecognized encoding is rejected (today at
        // parse). If graceful degradation lands later this becomes a
        // resolve-time rejection — update this test deliberately then.
        assert!(serde_json::from_str::<RotationEncoding>("\"rot10d\"").is_err());
    }
}
