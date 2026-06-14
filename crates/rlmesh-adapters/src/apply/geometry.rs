//! Rotation encoding conversions used by resolved adapters.

use crate::error::ApplyError;
use crate::spec::RotationEncoding;

const EPS: f64 = 1e-8;

type Matrix = [[f32; 3]; 3];

fn norm(values: &[f32]) -> f64 {
    f64::from(values.iter().map(|&x| x * x).sum::<f32>().sqrt())
}

fn eye() -> Matrix {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Reorder a quaternion to xyzw if it is wxyz.
fn as_quat_xyzw(quat: &[f32], encoding: RotationEncoding) -> [f32; 4] {
    if encoding == RotationEncoding::QuatWxyz {
        [quat[1], quat[2], quat[3], quat[0]]
    } else {
        [quat[0], quat[1], quat[2], quat[3]]
    }
}

/// Convert an xyzw quaternion to an axis-angle 3-vector.
fn quat_xyzw_to_axis_angle(quat: [f32; 4]) -> Vec<f32> {
    let quat_norm = norm(&quat);
    if quat_norm <= EPS {
        return vec![0.0; 3];
    }
    let scale = quat_norm as f32;
    let unit: Vec<f32> = quat.iter().map(|&x| x / scale).collect();
    let (xyz, w) = (&unit[..3], f64::from(unit[3]));
    let sin_half = norm(xyz);
    if sin_half <= EPS {
        return vec![0.0; 3];
    }
    let angle = 2.0 * sin_half.atan2(w);
    let factor = (angle / sin_half) as f32;
    xyz.iter().map(|&x| x * factor).collect()
}

/// Convert a rotation matrix to an xyzw quaternion (float64 internally,
/// like the reference).
fn matrix_to_quat_xyzw(matrix: &Matrix) -> Vec<f32> {
    let m = |row: usize, col: usize| f64::from(matrix[row][col]);
    let trace = m(0, 0) + m(1, 1) + m(2, 2);
    let (x, y, z, w);
    if trace > 0.0 {
        let s = 2.0 * (trace + 1.0).sqrt();
        w = 0.25 * s;
        x = (m(2, 1) - m(1, 2)) / s;
        y = (m(0, 2) - m(2, 0)) / s;
        z = (m(1, 0) - m(0, 1)) / s;
    } else if m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2) {
        let s = 2.0 * (1.0 + m(0, 0) - m(1, 1) - m(2, 2)).sqrt();
        w = (m(2, 1) - m(1, 2)) / s;
        x = 0.25 * s;
        y = (m(0, 1) + m(1, 0)) / s;
        z = (m(0, 2) + m(2, 0)) / s;
    } else if m(1, 1) > m(2, 2) {
        let s = 2.0 * (1.0 + m(1, 1) - m(0, 0) - m(2, 2)).sqrt();
        w = (m(0, 2) - m(2, 0)) / s;
        x = (m(0, 1) + m(1, 0)) / s;
        y = 0.25 * s;
        z = (m(1, 2) + m(2, 1)) / s;
    } else {
        let s = 2.0 * (1.0 + m(2, 2) - m(0, 0) - m(1, 1)).sqrt();
        w = (m(1, 0) - m(0, 1)) / s;
        x = (m(0, 2) + m(2, 0)) / s;
        y = (m(1, 2) + m(2, 1)) / s;
        z = 0.25 * s;
    }
    vec![x as f32, y as f32, z as f32, w as f32]
}

/// Reconstruct a rotation matrix from a 6D rotation's two column vectors via
/// Gram-Schmidt: `a1` -> `b1`, `a2` orthonormalized against `b1` -> `b2`, and
/// `b3 = b1 x b2`. Shared by both 6D orderings, which differ only in how `a1`
/// and `a2` are read out of the flat vector.
fn rot6d_basis_to_matrix(a1: [f32; 3], a2: [f32; 3]) -> Matrix {
    let b1_scale = (norm(&a1) + EPS) as f32;
    let b1 = a1.map(|x| x / b1_scale);
    let dot: f32 = b1.iter().zip(&a2).map(|(&l, &r)| l * r).sum();
    let mut b2 = [
        a2[0] - dot * b1[0],
        a2[1] - dot * b1[1],
        a2[2] - dot * b1[2],
    ];
    let b2_scale = (norm(&b2) + EPS) as f32;
    for entry in &mut b2 {
        *entry /= b2_scale;
    }
    let b3 = [
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    ];
    // Columns are b1, b2, b3.
    [
        [b1[0], b2[0], b3[0]],
        [b1[1], b2[1], b3[1]],
        [b1[2], b2[2], b3[2]],
    ]
}

/// Convert a rotation vector in any supported encoding to a matrix.
fn to_matrix(value: &[f32], encoding: RotationEncoding) -> Matrix {
    match encoding {
        RotationEncoding::QuatXyzw | RotationEncoding::QuatWxyz => {
            let quat = as_quat_xyzw(value, encoding);
            let quat_norm = norm(&quat);
            if quat_norm <= EPS {
                return eye();
            }
            let scale = quat_norm as f32;
            let [x, y, z, w] = quat.map(|component| f64::from(component / scale));
            let (xx, yy, zz) = (x * x, y * y, z * z);
            let (xy, xz, yz) = (x * y, x * z, y * z);
            let (wx, wy, wz) = (w * x, w * y, w * z);
            let entries = [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ];
            entries.map(|row| row.map(|entry| entry as f32))
        }
        RotationEncoding::AxisAngle => {
            let angle = norm(value);
            if angle <= EPS {
                return eye();
            }
            let scale = angle as f32;
            let (x, y, z) = (value[0] / scale, value[1] / scale, value[2] / scale);
            let skew: Matrix = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]];
            let (sin, cos) = (angle.sin() as f32, (1.0 - angle.cos()) as f32);
            let mut out = eye();
            for row in 0..3 {
                for col in 0..3 {
                    let skew_sq: f32 = (0..3)
                        .map(|inner| skew[row][inner] * skew[inner][col])
                        .sum();
                    out[row][col] += sin * skew[row][col] + cos * skew_sq;
                }
            }
            out
        }
        RotationEncoding::Rot6d => {
            // Standard: a1 = first column, a2 = second column.
            rot6d_basis_to_matrix(
                [value[0], value[1], value[2]],
                [value[3], value[4], value[5]],
            )
        }
        RotationEncoding::Rot6dRowMajor => {
            // Row-major flatten of the (3, 2) column block: de-interleave the
            // two columns before the same Gram-Schmidt as `Rot6d`.
            rot6d_basis_to_matrix(
                [value[0], value[2], value[4]],
                [value[1], value[3], value[5]],
            )
        }
        RotationEncoding::EulerXyz => {
            // [roll, pitch, yaw], extrinsic XYZ: R = Rz(yaw) Ry(pitch) Rx(roll).
            let (roll, pitch, yaw) = (
                f64::from(value[0]),
                f64::from(value[1]),
                f64::from(value[2]),
            );
            let (sr, cr) = (roll.sin(), roll.cos());
            let (sp, cp) = (pitch.sin(), pitch.cos());
            let (sy, cy) = (yaw.sin(), yaw.cos());
            let entries = [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ];
            entries.map(|row| row.map(|entry| entry as f32))
        }
    }
}

/// Convert a rotation matrix to a vector in any supported encoding.
fn matrix_to(matrix: &Matrix, encoding: RotationEncoding) -> Vec<f32> {
    match encoding {
        RotationEncoding::AxisAngle => {
            let trace = f64::from(matrix[0][0] + matrix[1][1] + matrix[2][2]);
            let theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
            if theta.abs() < EPS {
                return vec![0.0; 3];
            }
            let sin_theta = theta.sin();
            if sin_theta.abs() < EPS {
                // θ ≈ π: the antisymmetric part of R vanishes (a 180° rotation
                // matrix is symmetric), so `matrix[2][1] - matrix[1][2]` etc.
                // are all ~0 and the axis is lost. Recover it through the
                // quaternion path, which handles this case via the
                // largest-diagonal branch.
                let quat = matrix_to_quat_xyzw(matrix);
                return quat_xyzw_to_axis_angle([quat[0], quat[1], quat[2], quat[3]]);
            }
            let axis = [
                matrix[2][1] - matrix[1][2],
                matrix[0][2] - matrix[2][0],
                matrix[1][0] - matrix[0][1],
            ];
            let factor = (theta / (2.0 * sin_theta + EPS)) as f32;
            axis.iter().map(|&x| x * factor).collect()
        }
        RotationEncoding::Rot6d => {
            // Standard: the first two columns concatenated (col0 then col1).
            vec![
                matrix[0][0],
                matrix[1][0],
                matrix[2][0],
                matrix[0][1],
                matrix[1][1],
                matrix[2][1],
            ]
        }
        RotationEncoding::Rot6dRowMajor => {
            // The same two columns flattened row-major over the (3, 2) block.
            vec![
                matrix[0][0],
                matrix[0][1],
                matrix[1][0],
                matrix[1][1],
                matrix[2][0],
                matrix[2][1],
            ]
        }
        RotationEncoding::QuatXyzw | RotationEncoding::QuatWxyz => {
            let quat = matrix_to_quat_xyzw(matrix);
            if encoding == RotationEncoding::QuatWxyz {
                vec![quat[3], quat[0], quat[1], quat[2]]
            } else {
                quat
            }
        }
        RotationEncoding::EulerXyz => {
            // Inverse of the extrinsic-XYZ matrix; pitch is recovered in
            // [-pi/2, pi/2]. At gimbal lock (cos(pitch) ~ 0) roll and yaw
            // couple, so yaw is pinned to 0.
            let m = |row: usize, col: usize| f64::from(matrix[row][col]);
            let sin_pitch = (-m(2, 0)).clamp(-1.0, 1.0);
            let cos_pitch = (1.0 - sin_pitch * sin_pitch).max(0.0).sqrt();
            let pitch = sin_pitch.asin();
            let (roll, yaw) = if cos_pitch > EPS {
                (m(2, 1).atan2(m(2, 2)), m(1, 0).atan2(m(0, 0)))
            } else {
                // With yaw pinned to 0, the combined roll+/-yaw angle is read
                // off the top-left 2x2. Its sign tracks sin(pitch): +pi/2 gives
                // sin(roll-yaw), -pi/2 gives -sin(roll+yaw), so scale m[0][1] by
                // sin_pitch to recover an angle that reconstructs the matrix.
                ((sin_pitch * m(0, 1)).atan2(m(1, 1)), 0.0)
            };
            vec![roll as f32, pitch as f32, yaw as f32]
        }
    }
}

/// Convert a flat rotation vector between encodings (float32 output).
pub fn convert_rotation(
    value: &[f32],
    source: RotationEncoding,
    target: RotationEncoding,
) -> Result<Vec<f32>, ApplyError> {
    let expected = source.dims() as usize;
    if value.len() != expected {
        return Err(ApplyError::new(format!(
            "expected {expected}-element {} rotation, got shape ({},)",
            source.as_str(),
            value.len()
        )));
    }
    if source == target {
        return Ok(value.to_vec());
    }
    if matches!(
        source,
        RotationEncoding::QuatXyzw | RotationEncoding::QuatWxyz
    ) && target == RotationEncoding::AxisAngle
    {
        return Ok(quat_xyzw_to_axis_angle(as_quat_xyzw(value, source)));
    }
    Ok(matrix_to(&to_matrix(value, source), target))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_to_axis_angle_recovers_180_degree_rotation() {
        // rot6d (first two columns) of a 180° rotation about x:
        // R = diag(1, -1, -1). The matrix is symmetric, so the antisymmetric
        // recovery yields a zero axis — the pre-fix bug returned ~[0, 0, 0].
        let rot6d = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0];
        let axis_angle =
            convert_rotation(&rot6d, RotationEncoding::Rot6d, RotationEncoding::AxisAngle)
                .expect("convert");
        let pi = std::f32::consts::PI;
        assert!(
            (axis_angle[0].abs() - pi).abs() < 1e-4,
            "expected |angle| ~ pi about x, got {axis_angle:?}"
        );
        assert!(
            axis_angle[1].abs() < 1e-4 && axis_angle[2].abs() < 1e-4,
            "{axis_angle:?}"
        );
    }

    #[test]
    fn near_pi_branch_only_engages_close_to_pi() {
        // A general (non-degenerate) rotation must still resolve to a non-zero
        // axis-angle — the near-pi branch must not swallow ordinary angles.
        let rot6d = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]; // 90° about z
        let axis_angle =
            convert_rotation(&rot6d, RotationEncoding::Rot6d, RotationEncoding::AxisAngle)
                .expect("convert");
        let angle: f32 = axis_angle.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (angle - std::f32::consts::FRAC_PI_2).abs() < 1e-3,
            "{axis_angle:?}"
        );
    }

    #[test]
    fn euler_xyz_round_trips_through_the_matrix() {
        use RotationEncoding::{EulerXyz, QuatXyzw};
        // Non-degenerate roll/pitch/yaw recover exactly through a quaternion.
        let euler = vec![0.3_f32, -0.4, 1.1];
        let quat = convert_rotation(&euler, EulerXyz, QuatXyzw).expect("to quat");
        let back = convert_rotation(&quat, QuatXyzw, EulerXyz).expect("from quat");
        for (expected, actual) in euler.iter().zip(&back) {
            assert!(
                (expected - actual).abs() < 1e-4,
                "euler round-trip {expected} vs {actual}"
            );
        }
    }

    #[test]
    fn euler_xyz_axes_match_the_convention() {
        use RotationEncoding::{AxisAngle, EulerXyz};
        let half_pi = std::f32::consts::FRAC_PI_2;
        // Pure yaw (rotation about z) -> axis-angle about z.
        let yaw = convert_rotation(&[0.0, 0.0, half_pi], EulerXyz, AxisAngle).expect("yaw");
        assert!(yaw[0].abs() < 1e-4 && yaw[1].abs() < 1e-4);
        assert!((yaw[2] - half_pi).abs() < 1e-4, "{yaw:?}");
        // Pure roll (rotation about x) -> axis-angle about x.
        let roll = convert_rotation(&[half_pi, 0.0, 0.0], EulerXyz, AxisAngle).expect("roll");
        assert!(roll[1].abs() < 1e-4 && roll[2].abs() < 1e-4);
        assert!((roll[0] - half_pi).abs() < 1e-4, "{roll:?}");
    }

    #[test]
    fn euler_xyz_recovers_at_gimbal_lock() {
        use RotationEncoding::{EulerXyz, QuatXyzw};
        let half_pi = std::f32::consts::FRAC_PI_2;
        // At pitch = +/-pi/2, roll and yaw couple, so the recovered euler need
        // not match componentwise -- but it must encode the *same* rotation.
        // (Pre-fix, the -pi/2 case flipped the combined angle's sign.)
        for pitch in [half_pi, -half_pi] {
            let euler = vec![0.3_f32, pitch, 1.1];
            let quat = convert_rotation(&euler, EulerXyz, QuatXyzw).expect("to quat");
            let back = convert_rotation(&quat, QuatXyzw, EulerXyz).expect("from quat");
            let requat = convert_rotation(&back, EulerXyz, QuatXyzw).expect("re-quat");
            // q and -q are the same rotation.
            let agree = quat.iter().zip(&requat).all(|(a, b)| (a - b).abs() < 1e-4)
                || quat.iter().zip(&requat).all(|(a, b)| (a + b).abs() < 1e-4);
            assert!(agree, "gimbal pitch {pitch}: {quat:?} vs {requat:?}");
        }
    }

    #[test]
    fn rot6d_is_standard_column_concat_and_round_trips() {
        use RotationEncoding::{AxisAngle, QuatXyzw, Rot6d};
        // 90° about z: R = [[0,-1,0],[1,0,0],[0,0,1]]. Standard rot6d is the
        // first two columns concatenated: col0=[0,1,0], col1=[-1,0,0].
        let half = std::f32::consts::FRAC_PI_2 / 2.0;
        let quat = [0.0, 0.0, half.sin(), half.cos()];
        let rot6d = convert_rotation(&quat, QuatXyzw, Rot6d).expect("to rot6d");
        let expected = [0.0_f32, 1.0, 0.0, -1.0, 0.0, 0.0];
        for (got, want) in rot6d.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "{rot6d:?} vs {expected:?}");
        }
        // Encode and decode now agree: the round-trip recovers the rotation.
        let axis_angle = [0.3_f32, -0.4, 1.1];
        let r6d = convert_rotation(&axis_angle, AxisAngle, Rot6d).expect("to rot6d");
        let back = convert_rotation(&r6d, Rot6d, AxisAngle).expect("from rot6d");
        for (expected, actual) in axis_angle.iter().zip(&back) {
            assert!((expected - actual).abs() < 1e-4, "{expected} vs {actual}");
        }
    }

    #[test]
    fn rot6d_rowmajor_is_the_interleaving_and_round_trips() {
        use RotationEncoding::{AxisAngle, QuatXyzw, Rot6d, Rot6dRowMajor};
        let half = std::f32::consts::FRAC_PI_2 / 2.0;
        let quat = [0.0, 0.0, half.sin(), half.cos()]; // 90° about z
        let standard = convert_rotation(&quat, QuatXyzw, Rot6d).expect("std");
        let rowmajor = convert_rotation(&quat, QuatXyzw, Rot6dRowMajor).expect("row");
        // Row-major is the row-wise interleaving of the standard column-concat:
        // [c0_0, c1_0, c0_1, c1_1, c0_2, c1_2] vs [c0_0, c0_1, c0_2, c1_0, ...].
        let interleaved = [
            standard[0],
            standard[3],
            standard[1],
            standard[4],
            standard[2],
            standard[5],
        ];
        for (got, want) in rowmajor.iter().zip(&interleaved) {
            assert!((got - want).abs() < 1e-5, "{rowmajor:?} vs {interleaved:?}");
        }
        // The row-major encoding is self-consistent (its own encode/decode pair).
        let axis_angle = [0.3_f32, -0.4, 1.1];
        let r6d = convert_rotation(&axis_angle, AxisAngle, Rot6dRowMajor).expect("to");
        let back = convert_rotation(&r6d, Rot6dRowMajor, AxisAngle).expect("from");
        for (expected, actual) in axis_angle.iter().zip(&back) {
            assert!((expected - actual).abs() < 1e-4, "{expected} vs {actual}");
        }
    }
}
