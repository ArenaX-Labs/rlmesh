//! The host value model a binding's plans are run on.
//!
//! Dense arrays are [`rlmesh_spaces::Tensor`]; the free functions here give typed
//! (`f32`/`f64`) views and `astype`-style casts through the shared
//! [`scalar`](rlmesh_spaces::scalar) byte codec. [`Value`] is a model-IO payload, not a
//! [`rlmesh_spaces::SpaceValue`]: a model input can be a list of scalars or a bare number.

use std::collections::BTreeMap;

use rlmesh_spaces::scalar::{Scalar, check_int_in_dtype_range, decode_scalars, encode_scalars};
use rlmesh_spaces::{DType, Tensor};

use crate::error::ApplyError;

/// A value flowing through the apply engine.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A null host value, retained by custom inputs.
    None,
    /// A boolean host value, retained without coercion to a float.
    Bool(bool),
    /// An exact signed integer host value.
    Int(i64),
    /// A host tuple; unlike a list, it is one chunk frame.
    Tuple(Vec<Value>),
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
/// Integer magnitudes beyond `f32`'s exact range lose precision.
pub fn to_f32_vec(tensor: &Tensor) -> Vec<f32> {
    let dtype = tensor.dtype();
    let bytes = tensor.to_contiguous_bytes();
    decode_scalars(&bytes, dtype)
        .expect("a constructed tensor has a concrete dtype and whole elements")
        .iter()
        .map(|scalar| scalar.to_f64(dtype) as f32)
        .collect()
}

/// Build a float32 tensor from `f32` data (little-endian, repo convention).
pub fn tensor_from_f32(shape: Vec<i64>, data: &[f32]) -> Tensor {
    let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
    Tensor::from_vec(bytes, shape, DType::Float32).expect("f32 byte length matches shape")
}

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

/// Cast a tensor to `target` (NumPy `astype` for floats).
pub fn cast(tensor: &Tensor, target: DType) -> Result<Tensor, ApplyError> {
    if tensor.dtype() == target {
        return Ok(tensor.clone());
    }
    encode_f64_to(to_f64_vec(tensor), tensor.shape().to_vec(), target)
}

/// Encode already-decoded `f64` values into a tensor of `target`, applying the
/// same NumPy-`astype` semantics as [`cast`]: integer targets reject
/// non-integral or out-of-range values rather than truncating or wrapping.
pub fn encode_f64_to(
    values: Vec<f64>,
    shape: Vec<i64>,
    target: DType,
) -> Result<Tensor, ApplyError> {
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
    Tensor::from_vec(bytes, shape, target).map_err(|err| ApplyError::new(err.to_string()))
}

/// Convert a tensor's shape to `usize` (for indexing arithmetic).
pub fn shape_usize(tensor: &Tensor) -> Vec<usize> {
    tensor
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).unwrap_or(0))
        .collect()
}
