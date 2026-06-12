use prost_types::{Value, value};
use rlmesh_spaces as native;
use rlmesh_spaces::Scalar;

use crate::error::ProtocolError;

use super::proto_value::expect_list_value;

/// Largest integer magnitude that survives an f64 round-trip exactly (2^53).
const MAX_EXACT_F64_INT: i64 = 1 << 53;

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
    if value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(ProtocolError::DecodeError(format!(
            "value {value} is out of range for a 64-bit integer"
        )));
    }
    Ok(value as i64)
}

/// Convert an integer into the f64 used by proto Value, rejecting magnitudes
/// that would lose precision in the round-trip (|value| > 2^53).
pub(super) fn int_to_proto_f64(value: i64) -> Result<f64, ProtocolError> {
    if value.abs() > MAX_EXACT_F64_INT {
        return Err(ProtocolError::EncodeError(format!(
            "integer {value} exceeds the exact-float range (2^53) and cannot be encoded as a \
             JSON number without precision loss"
        )));
    }
    Ok(value as f64)
}

pub(super) fn encode_scalar(value: i64, dtype: native::DType) -> Result<Vec<u8>, ProtocolError> {
    encode_scalars(&[ScalarValue::Int(value)], dtype)
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

pub(super) fn decode_scalar(bytes: &[u8], dtype: native::DType) -> Result<i64, ProtocolError> {
    let values = decode_scalars(bytes, dtype)?;
    let Some(value) = values.first() else {
        return Err(ProtocolError::DecodeError(
            "expected one scalar value".to_string(),
        ));
    };
    Ok(match value {
        ScalarValue::Bool(value) => i64::from(*value),
        ScalarValue::Int(value) => *value,
        ScalarValue::Float(value) => float_to_int(*value)?,
        ScalarValue::String(_) => {
            return Err(ProtocolError::DecodeError(
                "scalar text is not valid for integer decode".to_string(),
            ));
        }
    })
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
            ScalarValue::String(_) => Err(ProtocolError::DecodeError(
                "text is not valid in integer sequence".to_string(),
            )),
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
            ScalarValue::Bool(value) => Ok(Scalar::Bool(*value)),
            ScalarValue::Int(value) => Ok(Scalar::Int(*value)),
            ScalarValue::Float(value) => Ok(Scalar::Float(*value)),
            ScalarValue::String(_) => Err(ProtocolError::EncodeError(format!(
                "unsupported scalar encoding for dtype {dtype:?}"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
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
    // The wire codec has always silently dropped trailing bytes that do not
    // form a whole element; keep that leniency for compatibility.
    let whole = bytes.len() - bytes.len() % native::dtype_size(dtype);
    native::decode_scalars(&bytes[..whole], dtype)
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

pub(super) fn decode_proto_array(value: &Value) -> Result<Vec<ScalarValue>, ProtocolError> {
    expect_list_value(value, "array")?
        .values
        .iter()
        .map(|value| match &value.kind {
            Some(value::Kind::BoolValue(flag)) => Ok(ScalarValue::Bool(*flag)),
            Some(value::Kind::NumberValue(number)) => {
                if number.is_finite() && number.fract() == 0.0 {
                    Ok(ScalarValue::Int(float_to_int(*number)?))
                } else {
                    Ok(ScalarValue::Float(*number))
                }
            }
            Some(value::Kind::StringValue(text)) => Ok(ScalarValue::String(text.clone())),
            _ => Err(ProtocolError::DecodeError(
                "array payload contained non-scalar value".to_string(),
            )),
        })
        .collect()
}

pub(super) fn scalar_to_proto_value(value: impl Into<ScalarValue>) -> Result<Value, ProtocolError> {
    Ok(match value.into() {
        ScalarValue::Bool(value) => Value {
            kind: Some(value::Kind::BoolValue(value)),
        },
        ScalarValue::Int(value) => Value {
            kind: Some(value::Kind::NumberValue(int_to_proto_f64(value)?)),
        },
        ScalarValue::Float(value) => Value {
            kind: Some(value::Kind::NumberValue(value)),
        },
        ScalarValue::String(value) => Value {
            kind: Some(value::Kind::StringValue(value)),
        },
    })
}

#[derive(Debug, Clone)]
pub(super) enum ScalarValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

impl From<i64> for ScalarValue {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}
