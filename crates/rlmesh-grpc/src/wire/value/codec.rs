use std::borrow::Cow;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use prost_types::{ListValue, Struct, Value, value};
use rlmesh_spaces as native;

use crate::error::ProtocolError;

use super::proto_value::{
    decode_proto_value, encode_proto_value, expect_list_value, expect_struct_value,
};
use super::scalars::{
    decode_int_sequence, decode_proto_array, decode_scalar, encode_int_sequence, encode_scalar,
    encode_scalars, scalar_to_proto_value,
};

pub(super) fn encode_space_value<'v>(
    value: &'v native::SpaceValue,
    space: &native::SpaceSpec,
) -> Result<Cow<'v, [u8]>, ProtocolError> {
    match (space.spec.as_ref(), value) {
        (Some(native::SpaceKind::Box(_)), native::SpaceValue::Box(value)) => {
            Ok(value.to_contiguous_bytes())
        }
        (Some(native::SpaceKind::Discrete(_)), native::SpaceValue::Discrete(value)) => {
            Ok(Cow::Owned(encode_scalar(*value, space.dtype)?))
        }
        (Some(native::SpaceKind::MultiBinary(_)), native::SpaceValue::MultiBinary(values)) => Ok(
            Cow::Owned(values.iter().map(|value| u8::from(*value)).collect()),
        ),
        (Some(native::SpaceKind::MultiDiscrete(_)), native::SpaceValue::MultiDiscrete(values)) => {
            Ok(Cow::Owned(encode_int_sequence(values, space.dtype)?))
        }
        (Some(native::SpaceKind::Text(_)), native::SpaceValue::Text(value)) => {
            Ok(Cow::Owned(encode_proto_value(&Value {
                kind: Some(value::Kind::StringValue(value.clone())),
            })))
        }
        (Some(native::SpaceKind::Dict(spec)), native::SpaceValue::Dict(values)) => {
            let fields = spec
                .keys
                .iter()
                .zip(spec.spaces.iter())
                .map(|(key, space)| {
                    let value = values.get(key).ok_or_else(|| {
                        ProtocolError::EncodeError(format!("missing dict key '{key}'"))
                    })?;
                    Ok((key.clone(), encode_value_for_space(value, space)?))
                })
                .collect::<Result<_, ProtocolError>>()?;
            Ok(Cow::Owned(encode_proto_value(&Value {
                kind: Some(value::Kind::StructValue(Struct { fields })),
            })))
        }
        (Some(native::SpaceKind::Tuple(spec)), native::SpaceValue::Tuple(values)) => {
            let values = values
                .iter()
                .zip(spec.spaces.iter())
                .map(|(value, space)| encode_value_for_space(value, space))
                .collect::<Result<_, _>>()?;
            Ok(Cow::Owned(encode_proto_value(&Value {
                kind: Some(value::Kind::ListValue(ListValue { values })),
            })))
        }
        _ => Err(ProtocolError::EncodeError(format!(
            "value kind did not match space {:?}",
            space.space_type()
        ))),
    }
}

pub(super) fn decode_space_value(
    bytes: &[u8],
    space: &native::SpaceSpec,
) -> Result<native::SpaceValue, ProtocolError> {
    match space.spec.as_ref() {
        Some(native::SpaceKind::Box(_)) => Ok(native::SpaceValue::Box(
            native::Tensor::from_slice(bytes, &space.shape, space.dtype)
                .map_err(|err| ProtocolError::DecodeError(format!("invalid box payload: {err}")))?,
        )),
        Some(native::SpaceKind::Discrete(_)) => Ok(native::SpaceValue::Discrete(decode_scalar(
            bytes,
            space.dtype,
        )?)),
        Some(native::SpaceKind::MultiBinary(_)) => Ok(native::SpaceValue::MultiBinary(
            bytes.iter().map(|value| *value != 0).collect(),
        )),
        Some(native::SpaceKind::MultiDiscrete(_)) => Ok(native::SpaceValue::MultiDiscrete(
            decode_int_sequence(bytes, space.dtype)?,
        )),
        Some(native::SpaceKind::Text(_)) => match decode_proto_value(bytes)?.kind {
            Some(value::Kind::StringValue(value)) => Ok(native::SpaceValue::Text(value)),
            _ => Err(ProtocolError::DecodeError(
                "text transport payload was not a string".to_string(),
            )),
        },
        Some(native::SpaceKind::Dict(spec)) => {
            let value = decode_proto_value(bytes)?;
            let struct_value = expect_struct_value(&value, "dict")?;
            let mut result = std::collections::BTreeMap::new();
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                let child_value = struct_value.fields.get(key).ok_or_else(|| {
                    ProtocolError::DecodeError(format!("missing dict field '{key}'"))
                })?;
                result.insert(
                    key.clone(),
                    decode_value_for_space(child_value, child_space)?,
                );
            }
            Ok(native::SpaceValue::Dict(result))
        }
        Some(native::SpaceKind::Tuple(spec)) => {
            let value = decode_proto_value(bytes)?;
            let list = expect_list_value(&value, "tuple")?;
            if list.values.len() != spec.spaces.len() {
                return Err(ProtocolError::DecodeError(format!(
                    "tuple payload length {} did not match tuple space length {}",
                    list.values.len(),
                    spec.spaces.len()
                )));
            }
            Ok(native::SpaceValue::Tuple(
                list.values
                    .iter()
                    .zip(spec.spaces.iter())
                    .map(|(value, space)| decode_value_for_space(value, space))
                    .collect::<Result<_, _>>()?,
            ))
        }
        None => Err(ProtocolError::DecodeError(
            "space spec is missing kind".to_string(),
        )),
    }
}

pub fn encode_space_value_bytes(
    value: &native::SpaceValue,
    space: &native::SpaceSpec,
) -> Result<Vec<u8>, ProtocolError> {
    Ok(encode_space_value(value, space)?.into_owned())
}

pub fn decode_space_value_bytes(
    bytes: &[u8],
    space: &native::SpaceSpec,
) -> Result<native::SpaceValue, ProtocolError> {
    decode_space_value(bytes, space)
}

pub(super) fn encode_value_for_space(
    value: &native::SpaceValue,
    space: &native::SpaceSpec,
) -> Result<Value, ProtocolError> {
    match value {
        native::SpaceValue::Box(boxed) => Ok(Value {
            kind: Some(value::Kind::StringValue(
                BASE64.encode(boxed.to_contiguous_bytes()),
            )),
        }),
        native::SpaceValue::Discrete(value) => Ok(Value {
            kind: Some(value::Kind::NumberValue(*value as f64)),
        }),
        native::SpaceValue::MultiBinary(values) => Ok(Value {
            kind: Some(value::Kind::ListValue(ListValue {
                values: values
                    .iter()
                    .map(|value| Value {
                        kind: Some(value::Kind::BoolValue(*value)),
                    })
                    .collect(),
            })),
        }),
        native::SpaceValue::MultiDiscrete(values) => Ok(Value {
            kind: Some(value::Kind::ListValue(ListValue {
                values: values
                    .iter()
                    .map(|value| scalar_to_proto_value(*value))
                    .collect(),
            })),
        }),
        native::SpaceValue::Text(value) => Ok(Value {
            kind: Some(value::Kind::StringValue(value.clone())),
        }),
        native::SpaceValue::Dict(_) | native::SpaceValue::Tuple(_) => {
            let encoded = encode_space_value(value, space)?;
            decode_proto_value(&encoded)
        }
    }
}

pub(super) fn decode_value_for_space(
    value: &Value,
    space: &native::SpaceSpec,
) -> Result<native::SpaceValue, ProtocolError> {
    match space.spec.as_ref() {
        Some(native::SpaceKind::Box(_)) => {
            let data = match value.kind.as_ref() {
                Some(value::Kind::StringValue(encoded)) => {
                    BASE64.decode(encoded.as_bytes()).map_err(|err| {
                        ProtocolError::DecodeError(format!("invalid base64 box payload: {err}"))
                    })?
                }
                _ => {
                    let scalars = decode_proto_array(value)?;
                    encode_scalars(&scalars, space.dtype)?
                }
            };
            Ok(native::SpaceValue::Box(
                native::Tensor::from_vec(data, space.shape.clone(), space.dtype).map_err(
                    |err| ProtocolError::DecodeError(format!("invalid box payload: {err}")),
                )?,
            ))
        }
        Some(native::SpaceKind::Discrete(_)) => match value.kind {
            Some(value::Kind::NumberValue(number)) => {
                Ok(native::SpaceValue::Discrete(number as i64))
            }
            Some(value::Kind::BoolValue(flag)) => Ok(native::SpaceValue::Discrete(i64::from(flag))),
            _ => Err(ProtocolError::DecodeError(
                "discrete payload was not numeric".to_string(),
            )),
        },
        Some(native::SpaceKind::MultiBinary(_)) => Ok(native::SpaceValue::MultiBinary(
            expect_list_value(value, "multibinary")?
                .values
                .iter()
                .map(|value| match value.kind {
                    Some(value::Kind::BoolValue(flag)) => Ok(flag),
                    Some(value::Kind::NumberValue(number)) => Ok(number != 0.0),
                    _ => Err(ProtocolError::DecodeError(
                        "multibinary element was not bool-like".to_string(),
                    )),
                })
                .collect::<Result<_, _>>()?,
        )),
        Some(native::SpaceKind::MultiDiscrete(_)) => Ok(native::SpaceValue::MultiDiscrete(
            expect_list_value(value, "multidiscrete")?
                .values
                .iter()
                .map(|value| match value.kind {
                    Some(value::Kind::NumberValue(number)) => Ok(number as i64),
                    _ => Err(ProtocolError::DecodeError(
                        "multidiscrete element was not numeric".to_string(),
                    )),
                })
                .collect::<Result<_, _>>()?,
        )),
        Some(native::SpaceKind::Text(_)) => match &value.kind {
            Some(value::Kind::StringValue(text)) => Ok(native::SpaceValue::Text(text.clone())),
            _ => Err(ProtocolError::DecodeError(
                "text payload was not a string".to_string(),
            )),
        },
        _ => {
            let encoded = encode_proto_value(value);
            decode_space_value(&encoded, space)
        }
    }
}
