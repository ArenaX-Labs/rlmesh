//! Rotation re-encodings on `f32` values, through the shared codec
//! ([`rotation`]).

use crate::spec::apply::ApplyError;
use crate::spec::role::apply::err;
use crate::spec::rotation::{self, Matrix, Rotation};

/// A source must be a rotation of its declared width: a sink encoding carries
/// a direction, not a rotation, so nothing decodes out of it (the direction
/// law; the binder refuses the pairing first, this is the apply-side guard).
fn check_source(value: &[f32], source: Rotation) -> Result<(), ApplyError> {
    if !source.is_source() {
        return Err(err(format!(
            "encoding {} is a sink (a direction, not a rotation) and cannot be converted \
             out of; only a rotation encoding converts into it",
            source.name()
        )));
    }
    let expected = source.width() as usize;
    if value.len() != expected {
        return Err(err(format!(
            "expected {expected}-element {} rotation, got shape ({},)",
            source.name(),
            value.len()
        )));
    }
    Ok(())
}

/// Convert a flat rotation vector between encodings, right-multiplying a fixed
/// `post` rotation onto it first (`R_out = R_in @ post`).
pub fn convert_rotation_with(
    value: &[f32],
    source: Rotation,
    target: Rotation,
    post: Option<&Matrix>,
) -> Result<Vec<f32>, ApplyError> {
    if source == target && post.is_none() {
        return Ok(value.to_vec());
    }
    check_source(value, source)?;
    let input: Vec<f64> = value.iter().map(|&x| f64::from(x)).collect();
    let mut out = Vec::with_capacity(target.width() as usize);
    rotation::convert_with(&input, source, target, post, &mut out);
    Ok(out.into_iter().map(|x| x as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_xyz_never_converts_out() {
        use Rotation::{GravityXyz, QuatXyzw};
        let same =
            convert_rotation_with(&[0.0, 0.0, -1.0], GravityXyz, GravityXyz, None).expect("ok");
        assert_eq!(same, vec![0.0, 0.0, -1.0]);
        let err = convert_rotation_with(&[0.0, 0.0, -1.0], GravityXyz, QuatXyzw, None).unwrap_err();
        assert!(err.to_string().contains("is a sink"), "{err}");
        // A roll of π about x: columns (1, 0, 0), (0, -1, 0).
        let roll = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        let err = convert_rotation_with(&[0.0, 0.0, -1.0], GravityXyz, GravityXyz, Some(&roll))
            .unwrap_err();
        assert!(err.to_string().contains("is a sink"), "{err}");
    }

    #[test]
    fn post_rotation_survives_a_no_op_conversion() {
        use Rotation::{QuatXyzw, Rot6d};
        // A quarter turn about z: columns (0, -1, 0), (1, 0, 0).
        let post = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let identity = [0.0_f32, 0.0, 0.0, 1.0];
        let rotated = convert_rotation_with(&identity, QuatXyzw, Rot6d, Some(&post)).expect("post");
        for (got, want) in rotated.iter().zip(&[0.0_f32, -1.0, 0.0, 1.0, 0.0, 0.0]) {
            assert!((got - want).abs() < 1e-6, "{rotated:?}");
        }
    }

    #[test]
    fn euler_xyz_survives_float32_quaternions_across_gimbal_lock() {
        use Rotation::{EulerXyz, QuatXyzw};
        let to_matrix = |v: &[f32], e| {
            let v: Vec<f64> = v.iter().map(|&x| f64::from(x)).collect();
            rotation::to_matrix(&v, e)
        };
        let half_pi = std::f32::consts::FRAC_PI_2;
        for pitch in [half_pi, -half_pi, half_pi - 1e-4, -half_pi + 1e-4] {
            for step in 0..64 {
                let euler = [-3.0 + step as f32 * 0.1, pitch, 3.0 - step as f32 * 0.07];
                let quat = convert_rotation_with(&euler, EulerXyz, QuatXyzw, None).expect("ok");
                let back = convert_rotation_with(&quat, QuatXyzw, EulerXyz, None).expect("ok");
                let (a, b) = (to_matrix(&quat, QuatXyzw), to_matrix(&back, EulerXyz));
                let distance = a
                    .iter()
                    .flatten()
                    .zip(b.iter().flatten())
                    .map(|(l, r)| (l - r) * (l - r))
                    .sum::<f64>()
                    .sqrt();
                assert!(distance < 5e-4, "{euler:?} -> {back:?} is {distance} away");
            }
        }
    }
}
