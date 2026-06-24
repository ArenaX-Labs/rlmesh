use rlmesh_spaces as native;
use rlmesh_spaces::Scalar;

use crate::error::ProtocolError;

/// Convert a decoded float into an integer, rejecting fractional and
/// non-finite values instead of silently truncating them or mapping NaN to 0.
pub(super) fn float_to_int(value: f64) -> Result<i64, ProtocolError> {
    if !value.is_finite() {
        return Err(ProtocolError::DecodeError(format!(
            "non-finite value {value} is not a valid integer"
        )));
    }
    if value.fract() != 0.0 {
        return Err(ProtocolError::DecodeError(format!(
            "fractional value {value} is not a valid integer"
        )));
    }
    // `i64::MAX as f64` rounds up to 2^63, which `as i64` would saturate to
    // i64::MAX; the exclusive upper bound must therefore be 2^63 itself.
    if value < i64::MIN as f64 || value >= -(i64::MIN as f64) {
        return Err(ProtocolError::DecodeError(format!(
            "value {value} is out of range for a 64-bit integer"
        )));
    }
    Ok(value as i64)
}

pub(super) fn encode_int_sequence(
    values: &[i64],
    dtype: native::DType,
) -> Result<Vec<u8>, ProtocolError> {
    let scalars = values
        .iter()
        .copied()
        .map(ScalarValue::Int)
        .collect::<Vec<_>>();
    encode_scalars(&scalars, dtype)
}

pub(super) fn decode_int_sequence(
    bytes: &[u8],
    dtype: native::DType,
) -> Result<Vec<i64>, ProtocolError> {
    decode_scalars(bytes, dtype)?
        .into_iter()
        .map(|value| match value {
            ScalarValue::Bool(value) => Ok(i64::from(value)),
            ScalarValue::Int(value) => Ok(value),
            ScalarValue::Float(value) => float_to_int(value),
        })
        .collect()
}

pub(super) fn encode_scalars(
    values: &[ScalarValue],
    dtype: native::DType,
) -> Result<Vec<u8>, ProtocolError> {
    let scalars = values
        .iter()
        .map(|value| match value {
            ScalarValue::Bool(value) => Scalar::Bool(*value),
            ScalarValue::Int(value) => Scalar::Int(*value),
            ScalarValue::Float(value) => Scalar::Float(*value),
        })
        .collect::<Vec<_>>();
    native::encode_scalars(&scalars, dtype).map_err(|_| {
        ProtocolError::EncodeError(format!("unsupported scalar encoding for dtype {dtype:?}"))
    })
}

pub(super) fn decode_scalars(
    bytes: &[u8],
    dtype: native::DType,
) -> Result<Vec<ScalarValue>, ProtocolError> {
    if dtype == native::DType::Unspecified {
        return Err(ProtocolError::DecodeError(
            "cannot decode scalar bytes with unspecified dtype".to_string(),
        ));
    }
    // No pre-trim: the leaf codec enforces exact length before decoding, so
    // trailing bytes are a hard error, not silently dropped.
    native::decode_scalars(bytes, dtype)
        .map_err(|err| ProtocolError::DecodeError(err.to_string()))
        .map(|scalars| {
            scalars
                .into_iter()
                .map(|scalar| match scalar {
                    Scalar::Bool(value) => ScalarValue::Bool(value),
                    Scalar::Int(value) => ScalarValue::Int(value),
                    Scalar::Float(value) => ScalarValue::Float(value),
                })
                .collect()
        })
}

#[derive(Debug, Clone)]
pub(super) enum ScalarValue {
    Bool(bool),
    Int(i64),
    Float(f64),
}
