//! The value model the apply engine operates on.
//!
//! Dense arrays are [`rlmesh_spaces::Tensor`], the repo-wide tensor type;
//! the free functions here give the apply kernels typed (`f32`/`u8`) views,
//! a dtype-generic element gather, and `astype`-style casts — all through the
//! shared [`scalar`](rlmesh_spaces::scalar) byte codec, so no parallel array
//! or dtype representation lives in this crate.
//!
//! [`Value`] is a thin model-IO *payload* envelope around a `Tensor`. It is
//! deliberately NOT [`rlmesh_spaces::SpaceValue`]: a model input can be a
//! list of scalars (not a tensor) or a bare number — distinctions a
//! space-typed value does not carry.

use std::collections::BTreeMap;

use rlmesh_spaces::scalar::{Scalar, check_int_in_dtype_range, decode_scalars, encode_scalars};
use rlmesh_spaces::{DType, Tensor, dtype_size};

use crate::error::ApplyError;

/// A value flowing through the apply engine.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Tensor(Tensor),
    Text(String),
    Bytes(Vec<u8>),
    Number(f64),
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

/// Decode a tensor's elements into `f64` (dtype-aware, exact within `f64`).
pub fn to_f64_vec(tensor: &Tensor) -> Vec<f64> {
    let dtype = tensor.dtype();
    let bytes = tensor.to_contiguous_bytes();
    decode_scalars(&bytes, dtype)
        .expect("a constructed tensor has a concrete dtype and whole elements")
        .iter()
        .map(|scalar| scalar.to_f64(dtype))
        .collect()
}

/// Flat `f32` view of a tensor (NumPy `astype(float32).reshape(-1)`).
///
/// The state/action kernels compute in `f32` to match the reference; integer
/// magnitudes beyond `f32`'s exact range therefore lose precision here. That
/// is a deliberate engine-wide choice, not a per-call rounding bug.
pub fn to_f32_vec(tensor: &Tensor) -> Vec<f32> {
    to_f64_vec(tensor)
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

/// Build a float32 tensor from `f32` data (little-endian, repo convention).
pub fn tensor_from_f32(shape: Vec<i64>, data: &[f32]) -> Tensor {
    let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
    Tensor::from_vec(bytes, shape, DType::Float32).expect("f32 byte length matches shape")
}

/// Build a uint8 tensor from pixel bytes.
pub fn tensor_from_u8(shape: Vec<i64>, data: Vec<u8>) -> Tensor {
    Tensor::from_vec(data, shape, DType::Uint8).expect("u8 byte length matches shape")
}

/// Whether `dtype` is an integer family.
fn is_integer(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Bool
            | DType::Uint8
            | DType::Int8
            | DType::Int16
            | DType::Uint16
            | DType::Int32
            | DType::Uint32
            | DType::Int64
            | DType::Uint64
    )
}

/// Convert any tensor to 8-bit pixels without truncation. A float image is
/// mapped from its declared source range `src_range` into the pipeline's
/// canonical `[0, 255]`: a unit `[0, 1]` image scales by 255, a byte-range
/// `[0, 255]` image passes through unchanged. With no declared range (an
/// unbounded space) a float image is assumed normalized `[0, 1]`. Integer
/// dtypes carry raw pixel magnitudes and are only rounded and clamped.
pub fn to_u8_pixels(tensor: &Tensor, src_range: Option<(f64, f64)>) -> Vec<u8> {
    let dtype = tensor.dtype();
    if dtype == DType::Uint8 {
        return tensor.to_contiguous_bytes().into_owned();
    }
    let float = !is_integer(dtype);
    to_f64_vec(tensor)
        .into_iter()
        .map(|value| {
            let value = if float {
                match src_range {
                    Some((low, high)) if high > low => (value - low) / (high - low) * 255.0,
                    _ => value * 255.0,
                }
            } else {
                value
            };
            value.round_ties_even().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// The pixel bytes of a uint8 tensor (the resize kernels operate on these).
pub fn u8_pixels(tensor: &Tensor) -> Result<Vec<u8>, ApplyError> {
    if tensor.dtype() != DType::Uint8 {
        return Err(ApplyError::new(
            "resize expects uint8 image data".to_owned(),
        ));
    }
    Ok(tensor.to_contiguous_bytes().into_owned())
}

/// Reorder a tensor's elements so `out[i] = src[indices[i]]`, dtype-generic
/// (moves whole `itemsize`-byte elements).
pub fn gather(tensor: &Tensor, indices: &[usize], out_shape: Vec<i64>) -> Tensor {
    let dtype = tensor.dtype();
    let size = dtype_size(dtype);
    let bytes = tensor.to_contiguous_bytes();
    let mut out = Vec::with_capacity(indices.len() * size);
    for &index in indices {
        out.extend_from_slice(&bytes[index * size..(index + 1) * size]);
    }
    Tensor::from_vec(out, out_shape, dtype).expect("gather preserves byte length")
}

/// Cast a tensor to `target` (NumPy `astype` for floats). Integer targets
/// reject non-integral or out-of-range values rather than silently
/// truncating or wrapping them.
pub fn cast(tensor: &Tensor, target: DType) -> Result<Tensor, ApplyError> {
    if tensor.dtype() == target {
        return Ok(tensor.clone());
    }
    let values = to_f64_vec(tensor);
    let scalars: Vec<Scalar> = if is_integer(target) {
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            if value.fract() != 0.0 {
                return Err(ApplyError::new(format!(
                    "cannot cast non-integral value {value} to {}",
                    target.name()
                )));
            }
            let integral = value as i64;
            check_int_in_dtype_range(integral, target)
                .map_err(|err| ApplyError::new(err.to_string()))?;
            out.push(Scalar::Int(integral));
        }
        out
    } else {
        values.into_iter().map(Scalar::Float).collect()
    };
    let bytes = encode_scalars(&scalars, target).map_err(|err| ApplyError::new(err.to_string()))?;
    Tensor::from_vec(bytes, tensor.shape().to_vec(), target)
        .map_err(|err| ApplyError::new(err.to_string()))
}

/// Convert a shape to the tensor `i64` form.
pub fn shape_i64(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|&dim| dim as i64).collect()
}

/// Convert a tensor's shape to `usize` (for indexing arithmetic).
pub fn shape_usize(tensor: &Tensor) -> Vec<usize> {
    tensor
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).unwrap_or(0))
        .collect()
}
