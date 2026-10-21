//! Rotation encodings (§5.3).
//!
//! `quat_xyzw` is canonical (decided Q2). `quat_xyzw ↔ quat_wxyz` is a
//! component permutation that keeps the sign and is automatic; every other
//! re-encoding needs `allow: rotation`. `gravity_xyz` is a target only: it
//! discards yaw, so nothing converts from it.

use std::fmt;

/// A rotation encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rotation {
    /// Unit quaternion, scalar last (canonical).
    QuatXyzw,
    /// Unit quaternion, scalar first (MuJoCo, Isaac, Unitree).
    QuatWxyz,
    /// Rotation vector: axis × angle (rad), norm ≤ π. Canonical for deltas.
    AxisAngle,
    /// The first two rotation-matrix columns, column-major.
    Rot6d,
    /// The same block, row-major.
    Rot6dRowMajor,
    /// Row-major 3×3.
    Matrix,
    /// Roll, pitch, yaw in rad, extrinsic XYZ.
    EulerXyz,
    /// `Rᵀ · (0, 0, -1)` for an orientation `R`. Target only.
    GravityXyz,
}

impl Rotation {
    /// Every encoding.
    pub const ALL: [Self; 8] = [
        Self::QuatXyzw,
        Self::QuatWxyz,
        Self::AxisAngle,
        Self::Rot6d,
        Self::Rot6dRowMajor,
        Self::Matrix,
        Self::EulerXyz,
        Self::GravityXyz,
    ];

    /// Its name in declarations and on the wire.
    pub fn name(self) -> &'static str {
        match self {
            Self::QuatXyzw => "quat_xyzw",
            Self::QuatWxyz => "quat_wxyz",
            Self::AxisAngle => "axis_angle",
            Self::Rot6d => "rot6d",
            Self::Rot6dRowMajor => "rot6d_rowmajor",
            Self::Matrix => "matrix",
            Self::EulerXyz => "euler_xyz",
            Self::GravityXyz => "gravity_xyz",
        }
    }

    /// The encoding `name` names.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.name() == name)
    }

    /// Elements per rotation.
    pub fn width(self) -> u32 {
        match self {
            Self::QuatXyzw | Self::QuatWxyz => 4,
            Self::AxisAngle | Self::EulerXyz | Self::GravityXyz => 3,
            Self::Rot6d | Self::Rot6dRowMajor => 6,
            Self::Matrix => 9,
        }
    }

    /// Whether it is a quaternion.
    pub fn is_quaternion(self) -> bool {
        matches!(self, Self::QuatXyzw | Self::QuatWxyz)
    }

    /// Whether a source may publish it (everything but `gravity_xyz`).
    pub fn is_source(self) -> bool {
        self != Self::GravityXyz
    }
}

impl fmt::Display for Rotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How one encoding becomes another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationChange {
    /// The same encoding.
    Same,
    /// Quaternion component order: automatic.
    Auto,
    /// Any other re-encoding: needs `allow: rotation`.
    Allowed,
    /// Nothing converts from `gravity_xyz`.
    Incompatible,
}

/// The change from `from` to `to`.
pub fn change(from: Rotation, to: Rotation) -> RotationChange {
    if from == to {
        RotationChange::Same
    } else if !from.is_source() {
        RotationChange::Incompatible
    } else if from.is_quaternion() && to.is_quaternion() {
        RotationChange::Auto
    } else {
        RotationChange::Allowed
    }
}

/// A 3×3 rotation matrix, row-major.
type Matrix = [[f64; 3]; 3];

/// `value` (one rotation in `from`) as a matrix.
fn to_matrix(value: &[f64], from: Rotation) -> Matrix {
    match from {
        Rotation::QuatXyzw | Rotation::QuatWxyz => {
            let (x, y, z, w) = if from == Rotation::QuatXyzw {
                (value[0], value[1], value[2], value[3])
            } else {
                (value[1], value[2], value[3], value[0])
            };
            let n = (x * x + y * y + z * z + w * w).sqrt();
            let (x, y, z, w) = if n > 0.0 {
                (x / n, y / n, z / n, w / n)
            } else {
                (0.0, 0.0, 0.0, 1.0)
            };
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ]
        }
        Rotation::AxisAngle => {
            let (rx, ry, rz) = (value[0], value[1], value[2]);
            let angle = (rx * rx + ry * ry + rz * rz).sqrt();
            if angle < 1e-12 {
                return identity();
            }
            let (x, y, z) = (rx / angle, ry / angle, rz / angle);
            let (s, c) = angle.sin_cos();
            let t = 1.0 - c;
            [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ]
        }
        Rotation::Rot6d | Rotation::Rot6dRowMajor => {
            // Column-major: [a1, a2] are the first two columns. Row-major
            // stores the same 3×2 block row by row.
            let (a1, a2) = if from == Rotation::Rot6d {
                (
                    [value[0], value[1], value[2]],
                    [value[3], value[4], value[5]],
                )
            } else {
                (
                    [value[0], value[2], value[4]],
                    [value[1], value[3], value[5]],
                )
            };
            gram_schmidt(a1, a2)
        }
        Rotation::Matrix => [
            [value[0], value[1], value[2]],
            [value[3], value[4], value[5]],
            [value[6], value[7], value[8]],
        ],
        Rotation::EulerXyz => {
            let (sr, cr) = value[0].sin_cos();
            let (sp, cp) = value[1].sin_cos();
            let (sy, cy) = value[2].sin_cos();
            // Extrinsic XYZ: R = Rz(yaw) · Ry(pitch) · Rx(roll).
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        }
        Rotation::GravityXyz => identity(),
    }
}

fn identity() -> Matrix {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn gram_schmidt(a1: [f64; 3], a2: [f64; 3]) -> Matrix {
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let scale = |v: [f64; 3], s: f64| [v[0] * s, v[1] * s, v[2] * s];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let n1 = norm(a1);
    if n1 == 0.0 {
        return identity();
    }
    let b1 = scale(a1, 1.0 / n1);
    let d = dot(b1, a2);
    let u2 = [a2[0] - d * b1[0], a2[1] - d * b1[1], a2[2] - d * b1[2]];
    let n2 = norm(u2);
    if n2 == 0.0 {
        return identity();
    }
    let b2 = scale(u2, 1.0 / n2);
    let b3 = [
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    ];
    // Columns b1, b2, b3.
    [
        [b1[0], b2[0], b3[0]],
        [b1[1], b2[1], b3[1]],
        [b1[2], b2[2], b3[2]],
    ]
}

/// `m` as a unit quaternion `(x, y, z, w)` with `w ≥ 0` (§5.3: a quaternion
/// produced from a sign-free encoding has w ≥ 0).
fn matrix_to_quat(m: &Matrix) -> [f64; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let (x, y, z, w) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            0.25 * s,
        )
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        (
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        )
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        (
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        )
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        (
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s,
        )
    };
    let n = (x * x + y * y + z * z + w * w).sqrt();
    let sign = if w < 0.0 { -1.0 } else { 1.0 };
    [sign * x / n, sign * y / n, sign * z / n, sign * w / n]
}

/// `m` in `to`.
fn from_matrix(m: &Matrix, to: Rotation, out: &mut Vec<f64>) {
    match to {
        Rotation::QuatXyzw => out.extend(matrix_to_quat(m)),
        Rotation::QuatWxyz => {
            let [x, y, z, w] = matrix_to_quat(m);
            out.extend([w, x, y, z]);
        }
        Rotation::AxisAngle => {
            let [x, y, z, w] = matrix_to_quat(m);
            let s = (x * x + y * y + z * z).sqrt();
            if s < 1e-12 {
                out.extend([0.0, 0.0, 0.0]);
            } else {
                // w ≥ 0, so the angle is in [0, π].
                let angle = 2.0 * s.atan2(w);
                out.extend([x / s * angle, y / s * angle, z / s * angle]);
            }
        }
        Rotation::Rot6d => out.extend([m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1]]),
        Rotation::Rot6dRowMajor => {
            out.extend([m[0][0], m[0][1], m[1][0], m[1][1], m[2][0], m[2][1]])
        }
        Rotation::Matrix => out.extend(m.iter().flatten()),
        Rotation::EulerXyz => {
            let pitch = (-m[2][0]).clamp(-1.0, 1.0).asin();
            let (roll, yaw) = if m[2][0].abs() < 1.0 - 1e-12 {
                (m[2][1].atan2(m[2][2]), m[1][0].atan2(m[0][0]))
            } else {
                // Gimbal lock: put the whole remaining rotation in yaw.
                (0.0, (-m[0][1]).atan2(m[1][1]))
            };
            out.extend([roll, pitch, yaw]);
        }
        // Rᵀ · (0, 0, -1) is minus the third row of R.
        Rotation::GravityXyz => out.extend([-m[2][0], -m[2][1], -m[2][2]]),
    }
}

/// Re-encodes a run of rotations in `from` (a whole number of them) into
/// `to`, appending to `out`. A quaternion order swap is a pure permutation
/// that keeps the sign; every other change goes through a rotation matrix.
pub fn convert(values: &[f64], from: Rotation, to: Rotation, out: &mut Vec<f64>) {
    let width = from.width() as usize;
    for value in values.chunks_exact(width) {
        match (from, to) {
            _ if from == to => out.extend_from_slice(value),
            (Rotation::QuatXyzw, Rotation::QuatWxyz) => {
                out.extend([value[3], value[0], value[1], value[2]]);
            }
            (Rotation::QuatWxyz, Rotation::QuatXyzw) => {
                out.extend([value[1], value[2], value[3], value[0]]);
            }
            _ => from_matrix(&to_matrix(value, from), to, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn quaternion_order_is_automatic_and_keeps_the_sign() {
        assert_eq!(
            change(Rotation::QuatWxyz, Rotation::QuatXyzw),
            RotationChange::Auto
        );
        assert_eq!(
            change(Rotation::QuatWxyz, Rotation::GravityXyz),
            RotationChange::Allowed
        );
        assert_eq!(
            change(Rotation::GravityXyz, Rotation::Matrix),
            RotationChange::Incompatible
        );
        let mut out = Vec::new();
        convert(
            &[-1.0, 0.0, 0.0, 0.0],
            Rotation::QuatWxyz,
            Rotation::QuatXyzw,
            &mut out,
        );
        assert_eq!(out, [0.0, 0.0, 0.0, -1.0], "sign kept");
    }

    #[test]
    fn every_encoding_round_trips_through_quat_xyzw() {
        // 90° about z, then 30° about x.
        let half = std::f64::consts::FRAC_PI_4;
        let q = [0.0, 0.0, half.sin(), half.cos()];
        for to in Rotation::ALL.into_iter().filter(|r| r.is_source()) {
            let mut encoded = Vec::new();
            convert(&q, Rotation::QuatXyzw, to, &mut encoded);
            assert_eq!(encoded.len(), to.width() as usize);
            let mut back = Vec::new();
            convert(&encoded, to, Rotation::QuatXyzw, &mut back);
            assert!(close(&back, &q), "{to}: {back:?}");
        }
    }

    #[test]
    fn gravity_is_down_in_the_sensor_frame() {
        let mut out = Vec::new();
        convert(
            &[0.0, 0.0, 0.0, 1.0],
            Rotation::QuatXyzw,
            Rotation::GravityXyz,
            &mut out,
        );
        assert!(close(&out, &[0.0, 0.0, -1.0]));
        // Rolled 90° about x: gravity points along -y... in the sensor frame
        // it is R^T (0, 0, -1) = (0, -1, 0).
        let half = std::f64::consts::FRAC_PI_4;
        out.clear();
        convert(
            &[half.sin(), 0.0, 0.0, half.cos()],
            Rotation::QuatXyzw,
            Rotation::GravityXyz,
            &mut out,
        );
        assert!(close(&out, &[0.0, -1.0, 0.0]), "{out:?}");
    }

    #[test]
    fn axis_angle_of_a_negative_w_quaternion_stays_within_pi() {
        let mut out = Vec::new();
        convert(
            &[0.0, 0.0, 0.1, -0.995],
            Rotation::QuatXyzw,
            Rotation::AxisAngle,
            &mut out,
        );
        let norm = out.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm <= std::f64::consts::PI);
    }
}
