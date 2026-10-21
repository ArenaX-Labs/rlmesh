//! Command validation and limits (§9.6), on packed actions (§10.6).
//!
//! Every action of a chunk is checked before anything is spliced:
//!
//! - float values are finite and booleans are 0 or 1;
//! - rotation fields are valid: a quaternion's norm is within 1 ± 1e-3 (it is
//!   renormalised; never replaced with identity), a rotation vector's norm
//!   is at most 2π (one over π is wrapped to the same rotation at most π),
//!   matrices and 6D rotations are orthonormal within 1e-3.
//!
//! Then each element is clamped to its command field's `limits`, a
//! `max_norm` scales a delta vector down to the bound, and a `binary` part
//! snaps every value to 0 or 1 at 0.5. A chunk with any invalid action is
//! rejected whole. The clamped value is what the edge dispatches and what
//! `_applied` shows. Actions are only copied when one of them changes.

use crate::spec::{Rotation, convert};
use crate::wire::{CommandSpec, DType, Layout};
use bytes::Bytes;

/// Tolerance on unit norms and orthogonality.
const UNIT_TOLERANCE: f64 = 1e-3;
/// Slack on the rotation-vector bounds, for float rounding.
const PI_SLACK: f64 = 1e-6;

/// One command field, prepared.
#[derive(Clone, Debug)]
struct Field {
    dtype: DType,
    range: std::ops::Range<usize>,
    /// A validated rotation encoding and its width.
    rotation: Option<(Rotation, usize)>,
    /// Per element `(low, high)`; one pair broadcasts.
    bounds: Vec<(f64, f64)>,
    max_norm: Option<f64>,
}

/// What [`CommandCheck::check`] found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checked {
    /// The actions, clamped and renormalised (the input when unchanged).
    pub actions: Bytes,
    /// Actions a limit, a norm bound or the binary snap changed.
    pub clamped: u32,
}

/// The validation and limits of one commandable part.
#[derive(Clone, Debug)]
pub struct CommandCheck {
    fields: Vec<Field>,
    stride: usize,
    binary: bool,
    /// Whether any action may need rewriting.
    rewrites: bool,
}

impl CommandCheck {
    /// Prepares the checks of `command`; `None` for a text command (the
    /// host never declares one).
    pub fn new(command: &CommandSpec) -> Option<Self> {
        let layout = command.layout().ok()?;
        let stride = layout.stride()?;
        let fields: Vec<Field> = command
            .fields
            .iter()
            .enumerate()
            .map(|(i, spec)| {
                let limit = command.limit(&spec.name);
                Field {
                    dtype: spec.dtype,
                    range: layout_field(&layout, i),
                    rotation: Rotation::from_name(&spec.rotation)
                        .filter(|r| validated(*r) && matches!(spec.dtype, DType::F32 | DType::F64))
                        .map(|r| (r, r.width() as usize)),
                    bounds: limit
                        .map(|l| l.low.iter().copied().zip(l.high.iter().copied()).collect())
                        .unwrap_or_default(),
                    max_norm: limit.and_then(|l| l.max_norm),
                }
            })
            .collect();
        let rewrites = command.binary
            || fields.iter().any(|f| {
                !f.bounds.is_empty()
                    || f.max_norm.is_some()
                    || f.rotation.is_some_and(|(r, _)| r.is_quaternion())
            });
        Some(Self {
            fields,
            stride,
            binary: command.binary,
            rewrites,
        })
    }

    /// One action's packed size.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Validates `count` packed actions and applies the limits; `Err` when
    /// any action is invalid (the chunk is rejected whole).
    pub fn check(&self, actions: &Bytes, count: usize) -> Result<Checked, String> {
        if count == 0 || Some(actions.len()) != count.checked_mul(self.stride) {
            return Err(format!(
                "{} bytes for {count} actions of {}",
                actions.len(),
                self.stride
            ));
        }
        if !self.rewrites {
            for action in actions.chunks_exact(self.stride) {
                self.validate(action)?;
            }
            return Ok(Checked {
                actions: actions.clone(),
                clamped: 0,
            });
        }
        let mut out = actions.to_vec();
        let mut clamped = 0;
        let mut changed = false;
        for action in out.chunks_exact_mut(self.stride) {
            self.validate(action)?;
            let (normalised, limited) = self.apply(action);
            changed |= normalised || limited;
            clamped += u32::from(limited);
        }
        Ok(Checked {
            actions: if changed {
                Bytes::from(out)
            } else {
                actions.clone()
            },
            clamped,
        })
    }

    fn validate(&self, action: &[u8]) -> Result<(), String> {
        for field in &self.fields {
            let bytes = &action[field.range.clone()];
            match field.dtype {
                DType::Bool if bytes.iter().any(|b| *b > 1) => {
                    return Err("a bool other than 0 or 1".into());
                }
                // Exponent all ones is inf or NaN.
                DType::F16
                    if bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .any(|b| u16::from_le_bytes(*b) & 0x7C00 == 0x7C00) =>
                {
                    return Err("a non-finite value".into());
                }
                DType::F32 | DType::F64 => {
                    let values = read(field.dtype, bytes);
                    if values.iter().any(|v| !v.is_finite()) {
                        return Err("a non-finite value".into());
                    }
                    if let Some((rotation, width)) = field.rotation {
                        for group in values.chunks_exact(width) {
                            check_rotation(rotation, group)?;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Renormalises quaternions, wraps rotation vectors and applies limits, norm bounds and the
    /// binary snap in place: `(renormalised, limited)`.
    fn apply(&self, action: &mut [u8]) -> (bool, bool) {
        let (mut normalised, mut limited) = (false, false);
        for field in &self.fields {
            if !matches!(field.dtype, DType::F32 | DType::F64) {
                continue;
            }
            let bytes = &mut action[field.range.clone()];
            let mut values = read(field.dtype, bytes);
            let before = values.clone();
            if let Some((rotation, width)) = field.rotation {
                for group in values.chunks_exact_mut(width) {
                    normalised |= canonical(rotation, group);
                }
            }
            for (i, v) in values.iter_mut().enumerate() {
                let bound = match field.bounds.as_slice() {
                    [] => None,
                    [one] => Some(*one),
                    many => many.get(i).copied(),
                };
                if let Some((low, high)) = bound {
                    let c = v.clamp(low, high);
                    limited |= c != *v;
                    *v = c;
                }
            }
            if let Some(max) = field.max_norm {
                let n = norm(&values);
                if n > max {
                    values.iter_mut().for_each(|v| *v *= max / n);
                    limited = true;
                }
            }
            if self.binary {
                for v in &mut values {
                    let snapped = convert::binary(*v, 0.0, 1.0);
                    limited |= snapped != *v;
                    *v = snapped;
                }
            }
            if values != before {
                write(field.dtype, bytes, &values);
            }
        }
        (normalised, limited)
    }
}

fn layout_field(layout: &Layout, i: usize) -> std::ops::Range<usize> {
    layout.field(i).unwrap_or(0..0)
}

fn read(dtype: DType, bytes: &[u8]) -> Vec<f64> {
    match dtype {
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f64::from(f32::from_le_bytes(*b)))
            .collect(),
        DType::F64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect(),
        _ => Vec::new(),
    }
}

// Values are clamped or normalised copies of f32 inputs, so narrowing is
// exact or the nearest f32.
#[allow(clippy::cast_possible_truncation)]
fn write(dtype: DType, bytes: &mut [u8], values: &[f64]) {
    match dtype {
        DType::F32 => {
            for (b, v) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                *b = (*v as f32).to_le_bytes();
            }
        }
        DType::F64 => {
            for (b, v) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(values) {
                *b = v.to_le_bytes();
            }
        }
        _ => {}
    }
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Whether `vectors` are unit length and mutually orthogonal, within the
/// tolerance.
fn orthonormal(vectors: &[[f64; 3]]) -> bool {
    vectors.iter().enumerate().all(|(i, a)| {
        (dot(*a, *a).sqrt() - 1.0).abs() <= UNIT_TOLERANCE
            && vectors[i + 1..]
                .iter()
                .all(|b| dot(*a, *b).abs() <= UNIT_TOLERANCE)
    })
}

/// Whether the edge validates `rotation` (Euler angles take any value and
/// gravity is no command).
fn validated(rotation: Rotation) -> bool {
    !matches!(rotation, Rotation::EulerXyz | Rotation::GravityXyz)
}

/// Renormalises a quaternion, and wraps a rotation vector over π to the
/// same rotation the other way round (θ − 2π along its axis), as robosuite's
/// `quat2axisangle` can emit up to 2π. Whether it changed anything.
fn canonical(rotation: Rotation, v: &mut [f64]) -> bool {
    let n = norm(v);
    let scale = match rotation {
        _ if rotation.is_quaternion() && (n - 1.0).abs() > f64::EPSILON => 1.0 / n,
        Rotation::AxisAngle if n > std::f64::consts::PI => (n - std::f64::consts::TAU) / n,
        _ => return false,
    };
    v.iter_mut().for_each(|x| *x *= scale);
    true
}

fn check_rotation(rotation: Rotation, v: &[f64]) -> Result<(), String> {
    let ok = match rotation {
        Rotation::QuatXyzw | Rotation::QuatWxyz => (norm(v) - 1.0).abs() <= UNIT_TOLERANCE,
        Rotation::AxisAngle => norm(v) <= std::f64::consts::TAU + PI_SLACK,
        Rotation::Matrix => {
            orthonormal(&[[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]])
        }
        Rotation::Rot6d => orthonormal(&[[v[0], v[1], v[2]], [v[3], v[4], v[5]]]),
        Rotation::Rot6dRowMajor => orthonormal(&[[v[0], v[2], v[4]], [v[1], v[3], v[5]]]),
        Rotation::EulerXyz | Rotation::GravityXyz => true,
    };
    if ok {
        Ok(())
    } else {
        Err(format!("an invalid {rotation} rotation {v:?}"))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::wire::{FieldLimit, FieldSpec, MergePolicy};

    use super::*;

    fn command(fields: Vec<FieldSpec>, limits: Vec<FieldLimit>, binary: bool) -> CommandSpec {
        CommandSpec {
            mode: "delta".into(),
            fields,
            limits,
            safe_behavior: "stop".into(),
            hold_max: Duration::from_millis(100),
            max_action_age: Duration::from_millis(200),
            merge: MergePolicy::Replace,
            reference: "current".into(),
            group: None,
            binary,
            hold_last: false,
        }
    }

    fn f32s(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn rot(name: &str, width: u32, encoding: &str) -> FieldSpec {
        let mut field = FieldSpec::new(name, DType::F32, &[width]);
        field.rotation = encoding.into();
        field
    }

    #[test]
    fn limits_clamp_and_norm_bounds_scale_without_touching_valid_actions() {
        let check = CommandCheck::new(&command(
            vec![
                FieldSpec::new("pos", DType::F32, &[3]),
                FieldSpec::new("dpos", DType::F32, &[3]),
            ],
            vec![
                FieldLimit {
                    field: "pos".into(),
                    low: vec![-1.0, -2.0, -3.0],
                    high: vec![1.0, 2.0, 3.0],
                    max_norm: None,
                },
                FieldLimit {
                    field: "dpos".into(),
                    low: Vec::new(),
                    high: Vec::new(),
                    max_norm: Some(0.1),
                },
            ],
            false,
        ))
        .expect("check");
        assert_eq!(
            check.stride(),
            32,
            "two 12-byte fields at 8-aligned offsets"
        );
        let mut action = f32s(&[5.0, -5.0, 0.5]);
        action.extend([0; 4]);
        action.extend(f32s(&[0.3, 0.4, 0.0]));
        action.extend([0; 4]);
        let mut ok = f32s(&[0.1, 0.2, 0.3]);
        ok.extend([0; 4]);
        ok.extend(f32s(&[0.01, 0.0, 0.0]));
        ok.extend([0; 4]);
        let both = Bytes::from([action, ok.clone()].concat());
        let checked = check.check(&both, 2).expect("valid");
        assert_eq!(checked.clamped, 1, "one action of two");
        assert_eq!(values(&checked.actions[0..12]), vec![1.0, -2.0, 0.5]);
        let scaled = values(&checked.actions[16..28]);
        assert!((scaled[0] - 0.06).abs() < 1e-6 && (scaled[1] - 0.08).abs() < 1e-6);
        assert_eq!(
            &checked.actions[32..],
            &ok[..],
            "the valid action is untouched"
        );
        assert!(check.check(&both, 3).is_err(), "a length mismatch");
    }

    #[test]
    fn rotations_are_validated_and_quaternions_renormalised() {
        let check = CommandCheck::new(&command(
            vec![rot("q", 4, "quat_xyzw"), rot("r", 3, "axis_angle")],
            Vec::new(),
            false,
        ))
        .expect("check");
        let action = |q: [f32; 4], r: [f32; 3]| {
            let mut a = f32s(&q);
            a.extend(f32s(&r));
            a.extend([0; 4]);
            Bytes::from(a)
        };
        let nearly = check
            .check(&action([0.0, 0.0, 0.0, 1.0005], [0.0, 0.0, 3.0]), 1)
            .expect("within tolerance");
        let q = values(&nearly.actions[0..16]);
        assert!((q[3] - 1.0).abs() < 1e-6, "renormalised: {q:?}");
        assert_eq!(nearly.clamped, 0, "renormalising is not clamping");
        let wrapped = check
            .check(&action([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 6.083]), 1)
            .expect("a rotation vector up to 2π");
        let r = values(&wrapped.actions[16..28]);
        assert!(
            (f64::from(r[2]) - (6.083 - std::f64::consts::TAU)).abs() < 1e-5,
            "wrapped to the same rotation: {r:?}"
        );
        for bad in [
            action([0.0, 0.0, 0.0, 1.01], [0.0; 3]),
            action([0.0; 4], [0.0; 3]),
            action([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 6.3]),
            action([0.0, 0.0, 0.0, f32::NAN], [0.0; 3]),
        ] {
            assert!(check.check(&bad, 1).is_err(), "{bad:?}");
        }
        let six = CommandCheck::new(&command(
            vec![rot("m", 6, "rot6d"), rot("n", 6, "rot6d_rowmajor")],
            Vec::new(),
            false,
        ))
        .expect("check");
        // The identity's first two columns, column- and row-major.
        let mut good = f32s(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        good.extend(f32s(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
        assert!(six.check(&Bytes::from(good), 1).is_ok());
        let mut skew = f32s(&[1.0, 0.0, 0.0, 0.5, 1.0, 0.0]);
        skew.extend(f32s(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
        assert!(six.check(&Bytes::from(skew), 1).is_err());
        let matrix = CommandCheck::new(&command(
            vec![{
                let mut m = FieldSpec::new("m", DType::F64, &[3, 3]);
                m.rotation = "matrix".into();
                m
            }],
            Vec::new(),
            false,
        ))
        .expect("check");
        let identity: Vec<u8> = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert!(matrix.check(&Bytes::from(identity), 1).is_ok());
    }

    #[test]
    fn binary_parts_snap_and_bools_must_be_zero_or_one() {
        let check = CommandCheck::new(&command(
            vec![
                FieldSpec::new("pos", DType::F32, &[1]),
                FieldSpec::new("on", DType::Bool, &[]),
            ],
            Vec::new(),
            true,
        ))
        .expect("check");
        let action = |pos: f32, on: u8| {
            let mut a = f32s(&[pos]);
            a.extend([0; 4]);
            a.push(on);
            a.extend([0; 7]);
            Bytes::from(a)
        };
        let checked = check.check(&action(0.7, 1), 1).expect("valid");
        assert_eq!(values(&checked.actions[0..4]), vec![1.0]);
        assert_eq!(checked.clamped, 1);
        assert!(check.check(&action(0.2, 2), 1).is_err());
        let plain = CommandCheck::new(&command(
            vec![FieldSpec::new("v", DType::F32, &[6])],
            Vec::new(),
            false,
        ))
        .expect("check");
        let zeros = Bytes::from(vec![0_u8; 24]);
        let checked = plain.check(&zeros, 1).expect("valid");
        assert_eq!(checked.actions.as_ptr(), zeros.as_ptr(), "no copy");
    }
}
