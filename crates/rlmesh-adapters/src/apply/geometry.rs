//! Rotation encoding conversions used by resolved adapters: the shared codec
//! (`rlmesh_session_core::spec::rotation`) on `f32` values.

use rlmesh_session_core::spec::rotation::{self, Rotation};

use crate::error::ApplyError;
use crate::spec::{RotationEncoding, RotationLiteral};

/// The shared codec's encoding for a v1 one (the same names).
pub(crate) fn shared(encoding: RotationEncoding) -> Rotation {
    match encoding {
        RotationEncoding::QuatXyzw => Rotation::QuatXyzw,
        RotationEncoding::QuatWxyz => Rotation::QuatWxyz,
        RotationEncoding::AxisAngle => Rotation::AxisAngle,
        RotationEncoding::Rot6d => Rotation::Rot6d,
        RotationEncoding::Rot6dRowMajor => Rotation::Rot6dRowMajor,
        RotationEncoding::EulerXyz => Rotation::EulerXyz,
        RotationEncoding::GravityXyz => Rotation::GravityXyz,
    }
}

/// A source must be a rotation of its declared width: a sink encoding carries
/// a direction, not a rotation, so nothing decodes out of it (the direction
/// law; the resolver refuses the pairing first, this is the apply-side guard).
fn check_source(value: &[f32], source: RotationEncoding) -> Result<(), ApplyError> {
    if source.is_sink() {
        return Err(ApplyError::new(format!(
            "encoding {} is a sink (a direction, not a rotation) and cannot be converted \
             out of; only a rotation encoding converts into it",
            source.as_str()
        )));
    }
    let expected = source.dims() as usize;
    if value.len() != expected {
        return Err(ApplyError::new(format!(
            "expected {expected}-element {} rotation, got shape ({},)",
            source.as_str(),
            value.len()
        )));
    }
    Ok(())
}

/// Convert a flat rotation vector between encodings, right-multiplying a fixed
/// `post` rotation onto it first (`R_out = R_in @ R(post)`).
pub fn convert_rotation_with(
    value: &[f32],
    source: RotationEncoding,
    target: RotationEncoding,
    post: Option<&RotationLiteral>,
) -> Result<Vec<f32>, ApplyError> {
    if source == target && post.is_none() {
        return Ok(value.to_vec());
    }
    check_source(value, source)?;
    let input: Vec<f64> = value.iter().map(|&x| f64::from(x)).collect();
    let post = post.map(RotationLiteral::matrix);
    let mut out = Vec::with_capacity(target.dims() as usize);
    rotation::convert_with(
        &input,
        shared(source),
        shared(target),
        post.as_ref(),
        &mut out,
    );
    Ok(out.into_iter().map(|x| x as f32).collect())
}

/// Convert a flat rotation vector between encodings (float32 output).
pub fn convert_rotation(
    value: &[f32],
    source: RotationEncoding,
    target: RotationEncoding,
) -> Result<Vec<f32>, ApplyError> {
    convert_rotation_with(value, source, target, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_xyz_never_converts_out() {
        use RotationEncoding::{GravityXyz, QuatXyzw};
        let same = convert_rotation(&[0.0, 0.0, -1.0], GravityXyz, GravityXyz).expect("ok");
        assert_eq!(same, vec![0.0, 0.0, -1.0]);
        let err = convert_rotation(&[0.0, 0.0, -1.0], GravityXyz, QuatXyzw).unwrap_err();
        assert!(err.to_string().contains("is a sink"), "{err}");
        let roll: RotationLiteral = serde_json::from_str(
            r#"{"encoding": "rot6d", "value": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0]}"#,
        )
        .expect("literal");
        let err = convert_rotation_with(&[0.0, 0.0, -1.0], GravityXyz, GravityXyz, Some(&roll))
            .unwrap_err();
        assert!(err.to_string().contains("is a sink"), "{err}");
    }

    #[test]
    fn post_rotation_survives_a_no_op_conversion() {
        use RotationEncoding::{QuatXyzw, Rot6d};
        let post: RotationLiteral = serde_json::from_str(
            r#"{"encoding": "rot6d", "value": [0.0, -1.0, 0.0, 1.0, 0.0, 0.0]}"#,
        )
        .expect("literal");
        let identity = [0.0_f32, 0.0, 0.0, 1.0];
        let rotated = convert_rotation_with(&identity, QuatXyzw, Rot6d, Some(&post)).expect("post");
        for (got, want) in rotated.iter().zip(&[0.0_f32, -1.0, 0.0, 1.0, 0.0, 0.0]) {
            assert!((got - want).abs() < 1e-6, "{rotated:?}");
        }
    }

    #[test]
    fn euler_xyz_survives_float32_quaternions_across_gimbal_lock() {
        use RotationEncoding::{EulerXyz, QuatXyzw};
        let to_matrix = |v: &[f32], e| {
            let v: Vec<f64> = v.iter().map(|&x| f64::from(x)).collect();
            rotation::to_matrix(&v, shared(e))
        };
        let half_pi = std::f32::consts::FRAC_PI_2;
        for pitch in [half_pi, -half_pi, half_pi - 1e-4, -half_pi + 1e-4] {
            for step in 0..64 {
                let euler = [-3.0 + step as f32 * 0.1, pitch, 3.0 - step as f32 * 0.07];
                let quat = convert_rotation(&euler, EulerXyz, QuatXyzw).expect("ok");
                let back = convert_rotation(&quat, QuatXyzw, EulerXyz).expect("ok");
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
