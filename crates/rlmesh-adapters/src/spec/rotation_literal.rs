//! A literal rotation constant authored on a model state part.

use serde::{Deserialize, Serialize};

use crate::spec::RotationEncoding;

/// Tolerance on an authored rotation's orthonormality (and on `|det - 1|`).
const ORTHONORMAL_TOL: f64 = 1e-4;

/// A fixed rotation authored on a model state part's `post_rotate`.
///
/// `R_out = R_in @ R(post_rotate)`: the rigid re-frame a checkpoint was trained
/// against (a tool/grip offset), right-multiplied onto the env's rotation after
/// it is decoded and before it is re-encoded into the model's encoding. The
/// authored value must already *be* a rotation -- it is validated orthonormal
/// (with `|det - 1| <= 1e-4`) at the codec rather than silently re-orthonormalized.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RotationLiteralWire")]
pub struct RotationLiteral {
    pub encoding: RotationEncoding,
    pub value: Vec<f64>,
}

/// Wire form of [`RotationLiteral`], validated via [`TryFrom`].
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RotationLiteralWire {
    encoding: RotationEncoding,
    value: Vec<f64>,
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

/// A matrix from two authored columns, third = `c0 x c1`. Unlike the apply-side
/// 6D decode this does *not* Gram-Schmidt: the authored basis is checked, not
/// repaired.
fn from_columns(c0: [f32; 3], c1: [f32; 3]) -> [[f32; 3]; 3] {
    let c2 = cross(c0, c1);
    [
        [c0[0], c1[0], c2[0]],
        [c0[1], c1[1], c2[1]],
        [c0[2], c1[2], c2[2]],
    ]
}

impl RotationLiteral {
    /// The 3x3 rotation this literal denotes.
    pub(crate) fn matrix(&self) -> [[f32; 3]; 3] {
        let value: Vec<f32> = self.value.iter().map(|&x| x as f32).collect();
        match self.encoding {
            RotationEncoding::Rot6d => from_columns(
                [value[0], value[1], value[2]],
                [value[3], value[4], value[5]],
            ),
            RotationEncoding::Rot6dRowMajor => from_columns(
                [value[0], value[2], value[4]],
                [value[1], value[3], value[5]],
            ),
            other => crate::apply::geometry::to_matrix(&value, other),
        }
    }
}

impl TryFrom<RotationLiteralWire> for RotationLiteral {
    type Error = String;

    fn try_from(wire: RotationLiteralWire) -> Result<Self, Self::Error> {
        // A sink encoding is a direction, not a rotation: there is no matrix
        // to right-multiply (the direction law).
        if wire.encoding.is_sink() {
            return Err(format!(
                "post_rotate: encoding {:?} is a direction, not a rotation; only a rotation \
                 encoding converts into it, and none out of it",
                wire.encoding.as_str()
            ));
        }
        let expected = wire.encoding.dims() as usize;
        if wire.value.len() != expected {
            return Err(format!(
                "post_rotate: encoding {:?} needs {expected} values, got {}",
                wire.encoding.as_str(),
                wire.value.len()
            ));
        }
        if let Some(bad) = wire.value.iter().find(|value| !value.is_finite()) {
            return Err(format!("post_rotate: value {bad} must be finite"));
        }
        let literal = RotationLiteral {
            encoding: wire.encoding,
            value: wire.value,
        };
        let matrix = literal.matrix();
        let entry = |row: usize, col: usize| f64::from(matrix[row][col]);
        // R^T R = I (columns orthonormal) and det = +1 (right-handed).
        for left in 0..3 {
            for right in 0..3 {
                let dot: f64 = (0..3).map(|row| entry(row, left) * entry(row, right)).sum();
                let want = if left == right { 1.0 } else { 0.0 };
                if (dot - want).abs() > ORTHONORMAL_TOL {
                    return Err(format!(
                        "post_rotate: {:?} value {:?} is not a rotation (its columns are \
                         not orthonormal)",
                        literal.encoding.as_str(),
                        literal.value
                    ));
                }
            }
        }
        let det = entry(0, 0) * (entry(1, 1) * entry(2, 2) - entry(1, 2) * entry(2, 1))
            - entry(0, 1) * (entry(1, 0) * entry(2, 2) - entry(1, 2) * entry(2, 0))
            + entry(0, 2) * (entry(1, 0) * entry(2, 1) - entry(1, 1) * entry(2, 0));
        if (det - 1.0).abs() > ORTHONORMAL_TOL {
            return Err(format!(
                "post_rotate: {:?} value {:?} has determinant {det} (a rotation has 1; \
                 a reflection is not supported)",
                literal.encoding.as_str(),
                literal.value
            ));
        }
        Ok(literal)
    }
}

#[cfg(test)]
mod tests {
    use super::RotationLiteral;

    #[test]
    fn accepts_a_rot6d_quarter_turn_and_round_trips() {
        // Rz(-90 deg) as its first two columns: the LIBERO hand->grip offset.
        let literal: RotationLiteral = serde_json::from_str(
            r#"{"encoding": "rot6d", "value": [0.0, -1.0, 0.0, 1.0, 0.0, 0.0]}"#,
        )
        .expect("parse");
        let matrix = literal.matrix();
        assert_eq!(matrix[0], [0.0, 1.0, 0.0]);
        assert_eq!(matrix[1], [-1.0, 0.0, 0.0]);
        assert_eq!(matrix[2], [0.0, 0.0, 1.0]);
        let json = serde_json::to_string(&literal).expect("serialize");
        assert_eq!(
            json,
            r#"{"encoding":"rot6d","value":[0.0,-1.0,0.0,1.0,0.0,0.0]}"#
        );
    }

    #[test]
    fn rejects_a_non_orthonormal_basis() {
        let err = serde_json::from_str::<RotationLiteral>(
            r#"{"encoding": "rot6d", "value": [1.0, 0.0, 0.0, 0.5, 1.0, 0.0]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("orthonormal"), "got: {err}");
    }

    #[test]
    fn rejects_the_gravity_sink() {
        let err = serde_json::from_str::<RotationLiteral>(
            r#"{"encoding": "gravity_xyz", "value": [0.0, 0.0, -1.0]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("not a rotation"), "got: {err}");
    }

    #[test]
    fn rejects_a_wrong_width_value() {
        let err = serde_json::from_str::<RotationLiteral>(
            r#"{"encoding": "rot6d", "value": [1.0, 0.0, 0.0]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("needs 6 values"), "got: {err}");
    }
}
