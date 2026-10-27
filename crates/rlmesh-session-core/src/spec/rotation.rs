//! Rotation encodings (§5.3) and the one codec between them.
//!
//! `quat_xyzw` is canonical (decided Q2). `quat_xyzw ↔ quat_wxyz` is a
//! component permutation that keeps the sign and is automatic; every other
//! re-encoding needs `allow: rotation`.
//! `gravity_xyz` is a target only: it discards yaw, so nothing converts
//! from it. [`convert`] states the sign conventions.

#[allow(unused_imports)]
use crate::prelude::*;

use core::fmt;

/// A rotation encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rotation {
    /// Unit quaternion, scalar last (canonical).
    QuatXyzw,
    /// Unit quaternion, scalar first (MuJoCo, Isaac, Unitree).
    QuatWxyz,
    /// Rotation vector: axis × angle (rad). Canonical for deltas. Norm at most
    /// π, except converted from a quaternion with w < 0 (see [`convert`]).
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
pub type Matrix = [[f64; 3]; 3];

/// Below this a quaternion, rotation vector or half-angle sine is zero.
const EPS: f64 = 1e-8;
/// `cos(pitch)` below which roll and yaw are one coupled angle. Sized for
/// `f32` sources, whose rounding noise reaches ~1e-7.
const GIMBAL_EPS: f64 = 1e-4;

/// The identity rotation.
pub fn identity() -> Matrix {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// A quaternion `(x, y, z, w)` from `value` in `from` (a quaternion
/// encoding).
fn as_xyzw(value: &[f64], from: Rotation) -> [f64; 4] {
    if from == Rotation::QuatWxyz {
        [value[1], value[2], value[3], value[0]]
    } else {
        [value[0], value[1], value[2], value[3]]
    }
}

/// `q`'s own rotation vector, `2·atan2(|v|, w)·v/|v|`: the hemisphere `q`
/// is in is kept, so the norm is in `[0, 2π)`.
fn quat_to_axis_angle(q: [f64; 4]) -> [f64; 3] {
    let n = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    if n <= EPS {
        return [0.0; 3];
    }
    let [x, y, z, w] = q.map(|c| c / n);
    let s = (x * x + y * y + z * z).sqrt();
    if s <= EPS {
        return [0.0; 3];
    }
    let k = 2.0 * s.atan2(w) / s;
    [x * k, y * k, z * k]
}

/// `value` (one rotation in `from`) as a matrix. `gravity_xyz` is no source
/// ([`Rotation::is_source`]); it reads as the identity.
pub fn to_matrix(value: &[f64], from: Rotation) -> Matrix {
    match from {
        Rotation::QuatXyzw | Rotation::QuatWxyz => {
            let n = as_xyzw(value, from)
                .iter()
                .map(|c| c * c)
                .sum::<f64>()
                .sqrt();
            if n <= EPS {
                return identity();
            }
            let [x, y, z, w] = as_xyzw(value, from).map(|c| c / n);
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
            if angle <= EPS {
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

/// Gram-Schmidt on two columns: `a1` normalised, `a2` orthonormalised
/// against it, the third their cross product. A degenerate pair is the
/// identity.
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

/// `m` as a quaternion `(x, y, z, w)` by the largest-diagonal method, with
/// the sign that method gives (0.1's).
fn matrix_to_quat(m: &Matrix) -> [f64; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            0.25 * s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s,
        ]
    }
}

/// `q` in the quaternion encoding `to`.
fn put_quat(q: [f64; 4], to: Rotation, out: &mut Vec<f64>) {
    let [x, y, z, w] = q;
    if to == Rotation::QuatWxyz {
        out.extend([w, x, y, z]);
    } else {
        out.extend([x, y, z, w]);
    }
}

/// `m` in `to`.
fn from_matrix(m: &Matrix, to: Rotation, out: &mut Vec<f64>) {
    match to {
        Rotation::QuatXyzw | Rotation::QuatWxyz => put_quat(matrix_to_quat(m), to, out),
        // Through the quaternion: the angle read off the trace loses
        // precision near 0 and π. Taken with w ≥ 0, the norm is at most π.
        Rotation::AxisAngle => {
            let q = matrix_to_quat(m);
            let q = if q[3] < 0.0 { q.map(|c| -c) } else { q };
            out.extend(quat_to_axis_angle(q));
        }
        Rotation::Rot6d => out.extend([m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1]]),
        Rotation::Rot6dRowMajor => {
            out.extend([m[0][0], m[0][1], m[1][0], m[1][1], m[2][0], m[2][1]])
        }
        Rotation::Matrix => out.extend(m.iter().flatten()),
        Rotation::EulerXyz => {
            // Pitch in [-π/2, π/2]. cos(pitch) comes from the entries roll
            // and yaw are read from, not sqrt(1 - sin²): at an f32 gimbal
            // lock those entries are already zero while the latter is ~3e-4.
            let sin_pitch = (-m[2][0]).clamp(-1.0, 1.0);
            let cos_pitch = m[2][1].hypot(m[2][2]);
            let pitch = sin_pitch.atan2(cos_pitch);
            let (roll, yaw) = if cos_pitch > GIMBAL_EPS {
                (m[2][1].atan2(m[2][2]), m[1][0].atan2(m[0][0]))
            } else {
                // Gimbal lock: yaw is 0 (scipy's and transforms3d's choice)
                // and roll carries the coupled angle, read off the top-left
                // 2×2 with the sign of sin(pitch).
                ((sin_pitch * m[0][1]).atan2(m[1][1]), 0.0)
            };
            out.extend([roll, pitch, yaw]);
        }
        // Rᵀ · (0, 0, -1) is minus the third row of R.
        Rotation::GravityXyz => out.extend([-m[2][0], -m[2][1], -m[2][2]]),
    }
}

/// `left · right`.
fn mat_mul(left: &Matrix, right: &Matrix) -> Matrix {
    let mut out = [[0.0; 3]; 3];
    for (row, entry) in out.iter_mut().enumerate() {
        for (col, cell) in entry.iter_mut().enumerate() {
            *cell = (0..3).map(|k| left[row][k] * right[k][col]).sum();
        }
    }
    out
}

/// Re-encodes a run of rotations in `from` (a whole number of them) into
/// `to`, appending to `out`.
///
/// The sign conventions, one rule each:
///
/// - The other quaternion order is a permutation: it keeps the sign (and the
///   norm). A quaternion from any other encoding has the sign the
///   largest-diagonal method gives, as in 0.1.
/// - `axis_angle` from a quaternion is that quaternion's own rotation vector
///   (robosuite's `quat2axisangle`, which LIBERO checkpoints read): its
///   hemisphere is kept, so the norm may exceed π. From any other encoding
///   the norm is at most π.
/// - `euler_xyz` at gimbal lock has yaw 0.
pub fn convert(values: &[f64], from: Rotation, to: Rotation, out: &mut Vec<f64>) {
    convert_with(values, from, to, None, out);
}

/// [`convert`] with a fixed rotation right-multiplied onto each value first
/// (`R_out = R_in · post`). With a `post` every value goes through the
/// matrix, including a conversion to its own encoding.
pub fn convert_with(
    values: &[f64],
    from: Rotation,
    to: Rotation,
    post: Option<&Matrix>,
    out: &mut Vec<f64>,
) {
    let width = from.width() as usize;
    for value in values.chunks_exact(width) {
        match post {
            Some(post) => from_matrix(&mat_mul(&to_matrix(value, from), post), to, out),
            None if from == to => out.extend_from_slice(value),
            None if from.is_quaternion() && to.is_quaternion() => {
                put_quat(as_xyzw(value, from), to, out);
            }
            None if from.is_quaternion() && to == Rotation::AxisAngle => {
                out.extend(quat_to_axis_angle(as_xyzw(value, from)));
            }
            None => from_matrix(&to_matrix(value, from), to, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    fn one(value: &[f64], from: Rotation, to: Rotation) -> Vec<f64> {
        let mut out = Vec::new();
        convert(value, from, to, &mut out);
        out
    }

    fn distance(a: &Matrix, b: &Matrix) -> f64 {
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .map(|(l, r)| (l - r) * (l - r))
            .sum::<f64>()
            .sqrt()
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
    fn a_quaternion_from_another_encoding_has_the_largest_diagonal_sign() {
        use Rotation::{EulerXyz, QuatXyzw};
        // Rx(-170°): the x-diagonal branch, whose w carries sin(-170°) < 0.
        let angle = -170_f64.to_radians();
        let q = one(&[angle, 0.0, 0.0], EulerXyz, QuatXyzw);
        let h = angle / 2.0;
        assert!(close(&q, &[-h.sin(), 0.0, 0.0, -h.cos()]), "{q:?}");
        assert!(q[3] < 0.0, "{q:?}");
    }

    #[test]
    fn every_encoding_round_trips_through_quat_xyzw() {
        let q = [0.0, 0.0, FRAC_PI_4.sin(), FRAC_PI_4.cos()];
        for to in Rotation::ALL.into_iter().filter(|r| r.is_source()) {
            let encoded = one(&q, Rotation::QuatXyzw, to);
            assert_eq!(encoded.len(), to.width() as usize);
            let back = one(&encoded, to, Rotation::QuatXyzw);
            assert!(close(&back, &q), "{to}: {back:?}");
        }
    }

    #[test]
    fn gravity_is_down_in_the_sensor_frame() {
        let down = one(
            &[0.0, 0.0, 0.0, 1.0],
            Rotation::QuatXyzw,
            Rotation::GravityXyz,
        );
        assert!(close(&down, &[0.0, 0.0, -1.0]));
        let rolled = one(
            &[FRAC_PI_4.sin(), 0.0, 0.0, FRAC_PI_4.cos()],
            Rotation::QuatXyzw,
            Rotation::GravityXyz,
        );
        assert!(close(&rolled, &[0.0, -1.0, 0.0]), "{rolled:?}");
        let nose = one(
            &[0.0, FRAC_PI_2, 0.0],
            Rotation::EulerXyz,
            Rotation::GravityXyz,
        );
        assert!(close(&nose, &[1.0, 0.0, 0.0]), "{nose:?}");
    }

    #[test]
    fn axis_angle_keeps_a_quaternions_hemisphere_and_is_minimal_otherwise() {
        use Rotation::{AxisAngle, Matrix, QuatXyzw};
        // w < 0: 2·atan2(0.1, -0.995) about +z, not 0.2 about -z.
        let q = [0.0, 0.0, 0.1, -0.995];
        let kept = one(&q, QuatXyzw, AxisAngle);
        assert!(kept[2] > PI && kept[2] < 2.0 * PI, "{kept:?}");
        // The same rotation from its matrix is the minimal vector.
        let m: Vec<f64> = to_matrix(&q, QuatXyzw).iter().flatten().copied().collect();
        let minimal = one(&m, Matrix, AxisAngle);
        assert!(minimal[2] < 0.0 && minimal[2].abs() <= PI, "{minimal:?}");
        assert!(
            distance(
                &to_matrix(&kept, AxisAngle),
                &to_matrix(&minimal, AxisAngle)
            ) < 1e-9
        );
    }

    #[test]
    fn matrix_to_axis_angle_is_stable_near_zero_and_pi() {
        use Rotation::{AxisAngle, EulerXyz, Rot6d};
        for offset in [1e-4, 3e-4, 4e-4, 1e-3, 1e-2, 0.5] {
            for angle in [offset, PI - offset] {
                let got = one(&[angle, 0.0, 0.0], EulerXyz, AxisAngle);
                assert!(
                    (got[0] - angle).abs() < 1e-9 && got[1].abs() < 1e-9 && got[2].abs() < 1e-9,
                    "Rx({angle}) -> {got:?}"
                );
            }
        }
        // 180° about x: a symmetric matrix, whose antisymmetric part is zero.
        let half_turn = one(&[1.0, 0.0, 0.0, 0.0, -1.0, 0.0], Rot6d, AxisAngle);
        assert!((half_turn[0].abs() - PI).abs() < 1e-9, "{half_turn:?}");
    }

    #[test]
    fn euler_xyz_pins_yaw_at_gimbal_lock_and_reconstructs_the_rotation() {
        use Rotation::{EulerXyz, QuatXyzw};
        for pitch in [FRAC_PI_2, -FRAC_PI_2] {
            let euler = [0.3, pitch, 1.1];
            let q = one(&euler, EulerXyz, QuatXyzw);
            let back = one(&q, QuatXyzw, EulerXyz);
            assert_eq!(back[2], 0.0, "yaw pinned: {back:?}");
            let d = distance(&to_matrix(&euler, EulerXyz), &to_matrix(&back, EulerXyz));
            assert!(d < 1e-9, "pitch {pitch}: {back:?} is {d} away");
        }
        // An f32 quaternion of [0.3, -π/2, 1.1]: -m20 rounds to 0.99999994,
        // so sqrt(1 - s²) claims cos(pitch) ~3e-4 while roll's and yaw's
        // entries are zero.
        let q = [0.455_530_7_f32, -0.540_825_07, 0.455_530_7, 0.540_825_07].map(f64::from);
        let euler = one(&q, QuatXyzw, EulerXyz);
        let d = distance(&to_matrix(&q, QuatXyzw), &to_matrix(&euler, EulerXyz));
        assert!(d < 1e-5, "{euler:?} is {d} away");
    }

    #[test]
    fn a_post_rotation_right_multiplies_even_onto_the_same_encoding() {
        use Rotation::{QuatXyzw, Rot6d};
        // Rz(-90°).
        let post = to_matrix(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0], Rot6d);
        let mut out = Vec::new();
        convert_with(
            &[0.0, 0.0, 0.0, 1.0],
            QuatXyzw,
            Rot6d,
            Some(&post),
            &mut out,
        );
        assert!(close(&out, &[0.0, -1.0, 0.0, 1.0, 0.0, 0.0]), "{out:?}");
        out.clear();
        convert_with(
            &[0.0, 0.0, 0.0, 1.0],
            QuatXyzw,
            QuatXyzw,
            Some(&post),
            &mut out,
        );
        let h = FRAC_PI_4;
        assert!(close(&out, &[0.0, 0.0, -h.sin(), h.cos()]), "{out:?}");
    }

    #[test]
    fn rot6d_layouts_are_the_column_block_and_its_row_major_flatten() {
        use Rotation::{QuatXyzw, Rot6d, Rot6dRowMajor};
        let q = [0.0, 0.0, FRAC_PI_4.sin(), FRAC_PI_4.cos()]; // 90° about z
        assert!(close(
            &one(&q, QuatXyzw, Rot6d),
            &[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
        ));
        assert!(close(
            &one(&q, QuatXyzw, Rot6dRowMajor),
            &[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]
        ));
    }
}
