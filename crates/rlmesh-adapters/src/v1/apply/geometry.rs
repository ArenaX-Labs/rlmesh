//! Rotation encoding conversions used by resolved adapters.

use super::super::spec::RotationEncoding;
use super::error::ApplyError;

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
            let (a1, a2) = (&value[..3], &value[3..]);
            let b1_scale = (norm(a1) + EPS) as f32;
            let b1: Vec<f32> = a1.iter().map(|&x| x / b1_scale).collect();
            let dot: f32 = b1.iter().zip(a2).map(|(&l, &r)| l * r).sum();
            let mut b2: Vec<f32> = a2.iter().zip(&b1).map(|(&a, &b)| a - dot * b).collect();
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
            // First two columns, flattened row-major.
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
}
