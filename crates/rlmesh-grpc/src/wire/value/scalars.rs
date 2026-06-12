use half::{bf16, f16};
use prost_types::{Value, value};
use rlmesh_spaces::v1 as native;

use crate::error::ProtocolError;

use super::proto_value::expect_list_value;

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
        ScalarValue::Float(value) => *value as i64,
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
            ScalarValue::Float(value) => Ok(value as i64),
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
    let mut encoded = Vec::new();
    for value in values {
        match (dtype, value) {
            (native::DType::Bool, ScalarValue::Bool(value)) => encoded.push(u8::from(*value)),
            (native::DType::Bool, ScalarValue::Int(value)) => encoded.push(u8::from(*value != 0)),
            (native::DType::Bool, ScalarValue::Float(value)) => {
                encoded.push(u8::from(*value != 0.0))
            }
            (native::DType::Uint8, ScalarValue::Int(value)) => encoded.push(*value as u8),
            (native::DType::Int32, ScalarValue::Int(value)) => {
                encoded.extend_from_slice(&(*value as i32).to_le_bytes())
            }
            (native::DType::Int64, ScalarValue::Int(value)) => {
                encoded.extend_from_slice(&value.to_le_bytes())
            }
            (native::DType::Int8, ScalarValue::Int(value)) => {
                encoded.extend_from_slice(&(*value as i8).to_le_bytes())
            }
            (native::DType::Int16, ScalarValue::Int(value)) => {
                encoded.extend_from_slice(&(*value as i16).to_le_bytes())
            }
            (native::DType::Uint16, ScalarValue::Int(value)) => {
                encoded.extend_from_slice(&(*value as u16).to_le_bytes())
            }
            (native::DType::Uint32, ScalarValue::Int(value)) => {
                encoded.extend_from_slice(&(*value as u32).to_le_bytes())
            }
            (native::DType::Uint64, ScalarValue::Int(value)) => {
                encoded.extend_from_slice(&(*value as u64).to_le_bytes())
            }
            (native::DType::Float16, ScalarValue::Float(value)) => {
                encoded.extend_from_slice(&f16::from_f64(*value).to_le_bytes())
            }
            (native::DType::Bfloat16, ScalarValue::Float(value)) => {
                encoded.extend_from_slice(&bf16::from_f64(*value).to_le_bytes())
            }
            (native::DType::Bfloat16, ScalarValue::Bool(value)) => encoded
                .extend_from_slice(&bf16::from_f32(if *value { 1.0 } else { 0.0 }).to_le_bytes()),
            (native::DType::Float16, ScalarValue::Bool(value)) => encoded
                .extend_from_slice(&f16::from_f32(if *value { 1.0 } else { 0.0 }).to_le_bytes()),
            (native::DType::Float32, ScalarValue::Float(value)) => {
                encoded.extend_from_slice(&(*value as f32).to_le_bytes())
            }
            (native::DType::Float32, ScalarValue::Bool(value)) => {
                encoded.extend_from_slice(&(if *value { 1.0_f32 } else { 0.0_f32 }).to_le_bytes())
            }
            (native::DType::Float64, ScalarValue::Float(value)) => {
                encoded.extend_from_slice(&value.to_le_bytes())
            }
            (native::DType::Float64, ScalarValue::Bool(value)) => {
                encoded.extend_from_slice(&(if *value { 1.0_f64 } else { 0.0_f64 }).to_le_bytes())
            }
            (native::DType::Float16, ScalarValue::Int(value))
            | (native::DType::Bfloat16, ScalarValue::Int(value))
            | (native::DType::Float32, ScalarValue::Int(value))
            | (native::DType::Float64, ScalarValue::Int(value)) => encoded.extend_from_slice(
                &encode_scalars(&[ScalarValue::Float(*value as f64)], dtype)?,
            ),
            (native::DType::Unspecified, _) | (_, ScalarValue::String(_)) => {
                return Err(ProtocolError::EncodeError(format!(
                    "unsupported scalar encoding for dtype {dtype:?}"
                )));
            }
            (native::DType::Uint8, ScalarValue::Bool(value)) => encoded.push(u8::from(*value)),
            (native::DType::Int32, ScalarValue::Bool(value))
            | (native::DType::Int64, ScalarValue::Bool(value))
            | (native::DType::Int8, ScalarValue::Bool(value))
            | (native::DType::Int16, ScalarValue::Bool(value))
            | (native::DType::Uint16, ScalarValue::Bool(value))
            | (native::DType::Uint32, ScalarValue::Bool(value))
            | (native::DType::Uint64, ScalarValue::Bool(value)) => encoded.extend_from_slice(
                &encode_scalars(&[ScalarValue::Int(i64::from(*value))], dtype)?,
            ),
            (native::DType::Uint8, ScalarValue::Float(value)) => encoded.push(*value as u8),
            (native::DType::Int32, ScalarValue::Float(value))
            | (native::DType::Int64, ScalarValue::Float(value))
            | (native::DType::Int8, ScalarValue::Float(value))
            | (native::DType::Int16, ScalarValue::Float(value))
            | (native::DType::Uint16, ScalarValue::Float(value))
            | (native::DType::Uint32, ScalarValue::Float(value))
            | (native::DType::Uint64, ScalarValue::Float(value)) => encoded
                .extend_from_slice(&encode_scalars(&[ScalarValue::Int(*value as i64)], dtype)?),
        }
    }
    Ok(encoded)
}

pub(super) fn decode_scalars(
    bytes: &[u8],
    dtype: native::DType,
) -> Result<Vec<ScalarValue>, ProtocolError> {
    match dtype {
        native::DType::Bool => Ok(bytes
            .iter()
            .map(|value| ScalarValue::Bool(*value != 0))
            .collect()),
        native::DType::Uint8 => Ok(bytes
            .iter()
            .map(|value| ScalarValue::Int(*value as i64))
            .collect()),
        native::DType::Int32 => bytes
            .chunks_exact(4)
            .map(|chunk| {
                Ok(ScalarValue::Int(i32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact yields fixed-size chunks"),
                ) as i64))
            })
            .collect(),
        native::DType::Int64 => bytes
            .chunks_exact(8)
            .map(|chunk| {
                Ok(ScalarValue::Int(i64::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact yields fixed-size chunks"),
                )))
            })
            .collect(),
        native::DType::Float16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                Ok(ScalarValue::Float(
                    f16::from_le_bytes(
                        chunk
                            .try_into()
                            .expect("chunks_exact yields fixed-size chunks"),
                    )
                    .to_f64(),
                ))
            })
            .collect(),
        native::DType::Float32 => bytes
            .chunks_exact(4)
            .map(|chunk| {
                Ok(ScalarValue::Float(f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact yields fixed-size chunks"),
                ) as f64))
            })
            .collect(),
        native::DType::Float64 => bytes
            .chunks_exact(8)
            .map(|chunk| {
                Ok(ScalarValue::Float(f64::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact yields fixed-size chunks"),
                )))
            })
            .collect(),
        native::DType::Int8 => Ok(bytes
            .iter()
            .map(|value| ScalarValue::Int(*value as i8 as i64))
            .collect()),
        native::DType::Int16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                Ok(ScalarValue::Int(i16::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact yields fixed-size chunks"),
                ) as i64))
            })
            .collect(),
        native::DType::Uint16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                Ok(ScalarValue::Int(u16::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact yields fixed-size chunks"),
                ) as i64))
            })
            .collect(),
        native::DType::Uint32 => bytes
            .chunks_exact(4)
            .map(|chunk| {
                Ok(ScalarValue::Int(u32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact yields fixed-size chunks"),
                ) as i64))
            })
            .collect(),
        // u64 values above i64::MAX wrap; ScalarValue has no unsigned variant.
        native::DType::Uint64 => bytes
            .chunks_exact(8)
            .map(|chunk| {
                Ok(ScalarValue::Int(u64::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact yields fixed-size chunks"),
                ) as i64))
            })
            .collect(),
        native::DType::Bfloat16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                Ok(ScalarValue::Float(
                    bf16::from_le_bytes(
                        chunk
                            .try_into()
                            .expect("chunks_exact yields fixed-size chunks"),
                    )
                    .to_f64(),
                ))
            })
            .collect(),
        native::DType::Unspecified => Err(ProtocolError::DecodeError(
            "cannot decode scalar bytes with unspecified dtype".to_string(),
        )),
    }
}

pub(super) fn decode_proto_array(value: &Value) -> Result<Vec<ScalarValue>, ProtocolError> {
    expect_list_value(value, "array")?
        .values
        .iter()
        .map(|value| match &value.kind {
            Some(value::Kind::BoolValue(flag)) => Ok(ScalarValue::Bool(*flag)),
            Some(value::Kind::NumberValue(number)) => {
                if number.fract() == 0.0 {
                    Ok(ScalarValue::Int(*number as i64))
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

pub(super) fn scalar_to_proto_value(value: impl Into<ScalarValue>) -> Value {
    match value.into() {
        ScalarValue::Bool(value) => Value {
            kind: Some(value::Kind::BoolValue(value)),
        },
        ScalarValue::Int(value) => Value {
            kind: Some(value::Kind::NumberValue(value as f64)),
        },
        ScalarValue::Float(value) => Value {
            kind: Some(value::Kind::NumberValue(value)),
        },
        ScalarValue::String(value) => Value {
            kind: Some(value::Kind::StringValue(value)),
        },
    }
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
