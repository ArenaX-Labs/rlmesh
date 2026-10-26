//! Running a Gym binding's plans: one env camera frame to a model image,
//! a model state's sources to its vector, and a model action to the env's.
//!
//! The host walks its observation tree and hands each plan the values it
//! reads; this module does the conversions. State and action values are
//! carried in `f32` between steps, each step computed in `f64` (rotations,
//! affines) or `f32` (range maps) as v1 adapters always have, so a binding
//! is byte-identical to the engine it replaced. Casts are NumPy `astype`
//! with integer targets refusing non-integral or out-of-range values.
//!
//! The one codec step, the JPEG round trip, is the host's
//! ([`Codec`]): this crate carries no image codec.

mod action;
mod geometry;
mod image;
mod state;

use crate::spec::apply::{ApplyError, Tensor, encode};
use crate::wire::part::DType;

pub use geometry::convert_rotation_with;
pub use image::Codec;

/// A dense little-endian, C-contiguous array the host reads out of its
/// observation.
#[derive(Clone, Copy, Debug)]
pub struct Array<'a> {
    /// Its element type.
    pub dtype: DType,
    /// Its shape.
    pub shape: &'a [usize],
    /// `product(shape) × dtype size` bytes.
    pub data: &'a [u8],
}

pub(crate) fn err(message: impl Into<String>) -> ApplyError {
    ApplyError(message.into())
}

/// `values` cast to `target` as NumPy's `astype`, except that an integer
/// target refuses a non-integral or out-of-range value rather than
/// truncating or wrapping it.
pub(crate) fn cast(values: &[f64], shape: Vec<u32>, target: DType) -> Result<Tensor, ApplyError> {
    if !target.is_float() {
        let (low, high) = int_range(target);
        for &value in values {
            if value.fract() != 0.0 {
                return Err(err(format!(
                    "cannot cast non-integral value {value} to {}",
                    target.name()
                )));
            }
            if !(low..=high).contains(&value) {
                return Err(err(format!(
                    "value {} is out of range for {}",
                    value as i64,
                    target.name()
                )));
            }
        }
    }
    Ok(Tensor {
        dtype: target,
        shape,
        data: encode(target, values),
    })
}

/// An integer dtype's exact range (`u64`'s signed-representable half).
fn int_range(dtype: DType) -> (f64, f64) {
    match dtype {
        DType::Bool => (0.0, 1.0),
        DType::U8 => (0.0, f64::from(u8::MAX)),
        DType::I8 => (f64::from(i8::MIN), f64::from(i8::MAX)),
        DType::U16 => (0.0, f64::from(u16::MAX)),
        DType::I16 => (f64::from(i16::MIN), f64::from(i16::MAX)),
        DType::U32 => (0.0, f64::from(u32::MAX)),
        DType::I32 => (f64::from(i32::MIN), f64::from(i32::MAX)),
        DType::I64 => (i64::MIN as f64, i64::MAX as f64),
        DType::U64 => (0.0, i64::MAX as f64),
        _ => (f64::NEG_INFINITY, f64::INFINITY),
    }
}

/// `values` as an `f32` vector, one element each.
pub(crate) fn f32_tensor(shape: Vec<u32>, values: &[f32]) -> Tensor {
    Tensor {
        dtype: DType::F32,
        shape,
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}
