use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use half::f16;
use prost::Message;
use prost_types::{ListValue, Struct, Value, value};
use rlmesh_proto::common::v1::MessageBytes;
use rlmesh_proto::env::v1::{
    RenderRequest, RenderResponse, ResetRequest, ResetResponse, StepRequest, StepResponse,
};
use rlmesh_proto::spaces::v1::SpaceValue;
use rlmesh_spaces::v1 as native;

use crate::error::ProtocolError;
use crate::wire::spaces::{meta_map_from_struct, meta_map_to_struct};

pub fn reset_request_to_proto(
    request: &native::ResetRequest,
) -> Result<ResetRequest, ProtocolError> {
    Ok(ResetRequest {
        seeds: request.seed.into_iter().collect(),
        options: request.options.as_ref().map(meta_map_to_struct),
        timeout_ms: request.timeout_ms,
    })
}

pub fn reset_result_from_proto(
    response: ResetResponse,
    observation_space: &native::SpaceSpec,
) -> Result<native::ResetResult, ProtocolError> {
    Ok(native::ResetResult {
        observation: decode_value(response.observation.as_ref(), observation_space)?,
        info: response.infos.map(meta_map_from_struct),
        episode_id: response.episode_ids.into_iter().next(),
    })
}

pub fn step_request_to_proto(
    request: &native::StepRequest,
    action_space: &native::SpaceSpec,
) -> Result<StepRequest, ProtocolError> {
    Ok(StepRequest {
        action: request
            .action
            .as_ref()
            .map(|value| encode_value(value, action_space))
            .transpose()?,
        timeout_ms: request.timeout_ms,
    })
}

pub fn step_result_from_proto(
    response: StepResponse,
    observation_space: &native::SpaceSpec,
) -> Result<native::StepResult, ProtocolError> {
    Ok(native::StepResult {
        observation: decode_value(response.observation.as_ref(), observation_space)?,
        reward: response.rewards.first().copied().unwrap_or_default(),
        terminated: response
            .terminated_mask
            .first()
            .copied()
            .unwrap_or_default()
            != 0,
        truncated: response.truncated_mask.first().copied().unwrap_or_default() != 0,
        info: response.infos.map(meta_map_from_struct),
    })
}

pub fn render_request_to_proto(request: &native::RenderRequest) -> RenderRequest {
    RenderRequest {
        mask: request.env_index.map(render_mask).unwrap_or_default(),
        timeout_ms: request.timeout_ms,
    }
}

pub fn render_result_from_proto(
    response: RenderResponse,
) -> Result<native::RenderResult, ProtocolError> {
    Ok(native::RenderResult {
        frame: response
            .png_frame
            .map(|png_frame| native::RenderFrame { png_frame }),
    })
}

fn render_mask(env_index: usize) -> Vec<u8> {
    let mut mask = vec![0_u8; env_index + 1];
    if let Some(byte) = mask.get_mut(env_index) {
        *byte = 1;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::{
        binary_to_bytes, decode_batch_bytes, decode_batched_partial_values, decode_value_bytes,
        decode_value_for_space, encode_batch_bytes, encode_batched_partial_values,
        encode_value_bytes, optional_bytes_to_binary, render_request_to_proto,
    };
    use prost_types::{ListValue, Value, value};
    use rlmesh_proto::common::v1::MessageBytes;
    use rlmesh_spaces::v1::spaces::{BoxSpaceBuilder, DictSpaceBuilder, DiscreteBuilder};
    use rlmesh_spaces::v1::{BinaryPayload, BoxValue, DType, RenderRequest, SpaceValue};

    #[test]
    fn render_request_without_env_index_uses_empty_mask() {
        let request = RenderRequest {
            env_index: None,
            timeout_ms: 17,
        };

        let proto = render_request_to_proto(&request);
        assert_eq!(proto.mask, Vec::<u8>::new());
        assert_eq!(proto.timeout_ms, 17);
    }

    #[test]
    fn render_request_with_env_index_maps_to_single_bit_mask() {
        let request = RenderRequest {
            env_index: Some(2),
            timeout_ms: 0,
        };

        let proto = render_request_to_proto(&request);
        assert_eq!(proto.mask, vec![0, 0, 1]);
    }

    #[test]
    fn batched_dict_values_roundtrip_through_wire_helpers() {
        let space = DictSpaceBuilder::new()
            .insert("choice", DiscreteBuilder::new(4).build().unwrap())
            .build()
            .unwrap();
        let values = vec![
            SpaceValue::Dict(
                [("choice".to_string(), SpaceValue::Discrete(1))]
                    .into_iter()
                    .collect(),
            ),
            SpaceValue::Dict(
                [("choice".to_string(), SpaceValue::Discrete(3))]
                    .into_iter()
                    .collect(),
            ),
        ];

        let payload = encode_batch_bytes(&values, &space).unwrap();
        let decoded = decode_batch_bytes(Some(&payload), &space).unwrap();

        assert_eq!(decoded, values);
    }

    #[test]
    fn batched_partial_box_values_use_raw_concatenated_payload() {
        let space = BoxSpaceBuilder::scalar(0.0, 255.0, vec![2])
            .dtype(DType::Uint8)
            .build()
            .unwrap();
        let values = vec![
            SpaceValue::Box(BoxValue {
                data: vec![1, 2],
                shape: vec![2],
                dtype: DType::Uint8,
            }),
            SpaceValue::Box(BoxValue {
                data: vec![3, 4],
                shape: vec![2],
                dtype: DType::Uint8,
            }),
        ];

        let payload = encode_batched_partial_values(&values, &space).unwrap();
        let decoded = decode_batched_partial_values(Some(&payload), &space).unwrap();

        assert_eq!(payload.data, vec![1, 2, 3, 4]);
        assert_eq!(decoded, values);
    }

    #[test]
    fn batched_partial_raw_decode_rejects_misaligned_payload() {
        let space = BoxSpaceBuilder::scalar(0.0, 255.0, vec![2])
            .dtype(DType::Uint8)
            .build()
            .unwrap();
        let payload = MessageBytes {
            data: vec![1, 2, 3],
        };

        let error = decode_batched_partial_values(Some(&payload), &space).unwrap_err();

        assert!(error.to_string().contains("is not divisible"));
    }

    #[test]
    fn binary_payload_roundtrips_through_message_bytes() {
        let payload = BinaryPayload {
            data: vec![1, 2, 3, 4],
        };

        let encoded = binary_to_bytes(&payload);
        let decoded = optional_bytes_to_binary(Some(&encoded))
            .unwrap()
            .expect("payload present");

        assert_eq!(decoded, payload);
    }

    #[test]
    fn nested_image_box_roundtrips_and_stays_compact() {
        let space = DictSpaceBuilder::new()
            .insert(
                "image",
                BoxSpaceBuilder::scalar(0.0, 255.0, vec![16, 16, 3])
                    .dtype(DType::Uint8)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();
        let raw: Vec<u8> = (0..16 * 16 * 3).map(|i| (i % 256) as u8).collect();
        let value = SpaceValue::Dict(
            [(
                "image".to_string(),
                SpaceValue::Box(BoxValue {
                    data: raw.clone(),
                    shape: vec![16, 16, 3],
                    dtype: DType::Uint8,
                }),
            )]
            .into_iter()
            .collect(),
        );

        let payload = encode_value_bytes(&value, &space).unwrap();
        let decoded = decode_value_bytes(Some(&payload), &space).unwrap().unwrap();

        assert_eq!(decoded, value);
        assert!(payload.data.len() < raw.len() * 2);
    }

    #[test]
    fn legacy_scalar_list_box_payload_still_decodes() {
        let space = BoxSpaceBuilder::scalar(0.0, 255.0, vec![3])
            .dtype(DType::Uint8)
            .build()
            .unwrap();
        let value = Value {
            kind: Some(value::Kind::ListValue(ListValue {
                values: vec![1.0, 2.0, 3.0]
                    .into_iter()
                    .map(|number| Value {
                        kind: Some(value::Kind::NumberValue(number)),
                    })
                    .collect(),
            })),
        };

        let decoded = decode_value_for_space(&value, &space).unwrap();

        assert_eq!(
            decoded,
            SpaceValue::Box(BoxValue {
                data: vec![1, 2, 3],
                shape: vec![3],
                dtype: DType::Uint8,
            })
        );
    }
}

pub fn reset_result_to_proto(
    result: &native::ResetResult,
    observation_space: &native::SpaceSpec,
) -> Result<ResetResponse, ProtocolError> {
    Ok(ResetResponse {
        observation: result
            .observation
            .as_ref()
            .map(|value| encode_value(value, observation_space))
            .transpose()?,
        infos: result.info.as_ref().map(meta_map_to_struct),
        episode_ids: result.episode_id.iter().cloned().collect(),
    })
}

pub fn step_result_to_proto(
    result: &native::StepResult,
    observation_space: &native::SpaceSpec,
) -> Result<StepResponse, ProtocolError> {
    Ok(StepResponse {
        observation: result
            .observation
            .as_ref()
            .map(|value| encode_value(value, observation_space))
            .transpose()?,
        rewards: vec![result.reward],
        terminated_mask: vec![u8::from(result.terminated)],
        truncated_mask: vec![u8::from(result.truncated)],
        infos: result.info.as_ref().map(meta_map_to_struct),
        completed_episodes: vec![],
        episode_ids: vec![],
    })
}

pub fn render_result_to_proto(result: &native::RenderResult) -> RenderResponse {
    RenderResponse {
        png_frame: result.frame.as_ref().map(|frame| frame.png_frame.clone()),
    }
}

pub fn encode_value_bytes(
    value: &native::SpaceValue,
    space: &native::SpaceSpec,
) -> Result<MessageBytes, ProtocolError> {
    Ok(MessageBytes {
        data: encode_space_value(value, space)?,
    })
}

pub fn bytes_value(value: MessageBytes) -> SpaceValue {
    SpaceValue { bytes: Some(value) }
}

pub fn encode_value(
    value: &native::SpaceValue,
    space: &native::SpaceSpec,
) -> Result<SpaceValue, ProtocolError> {
    Ok(bytes_value(encode_value_bytes(value, space)?))
}

pub fn value_bytes(payload: Option<&SpaceValue>) -> Result<Option<MessageBytes>, ProtocolError> {
    let Some(payload) = payload else {
        return Ok(None);
    };
    Ok(payload.bytes.clone())
}

pub fn value_bytes_ref(
    payload: Option<&SpaceValue>,
) -> Result<Option<&MessageBytes>, ProtocolError> {
    let Some(payload) = payload else {
        return Ok(None);
    };
    Ok(payload.bytes.as_ref())
}

pub fn decode_value_bytes(
    payload: Option<&MessageBytes>,
    space: &native::SpaceSpec,
) -> Result<Option<native::SpaceValue>, ProtocolError> {
    let Some(bytes) = payload else {
        return Ok(None);
    };
    Ok(Some(decode_space_value(&bytes.data, space)?))
}

pub fn decode_value(
    payload: Option<&SpaceValue>,
    space: &native::SpaceSpec,
) -> Result<Option<native::SpaceValue>, ProtocolError> {
    decode_value_bytes(value_bytes_ref(payload)?, space)
}

pub fn encode_batch_bytes(
    values: &[native::SpaceValue],
    space: &native::SpaceSpec,
) -> Result<MessageBytes, ProtocolError> {
    let values = values
        .iter()
        .map(|value| encode_value_for_space(value, space))
        .collect::<Result<_, _>>()?;

    Ok(MessageBytes {
        data: encode_proto_value(&Value {
            kind: Some(value::Kind::ListValue(ListValue { values })),
        }),
    })
}

pub fn binary_to_bytes(value: &native::BinaryPayload) -> MessageBytes {
    MessageBytes {
        data: value.data.clone(),
    }
}

pub fn bytes_to_binary(value: MessageBytes) -> Result<native::BinaryPayload, ProtocolError> {
    Ok(native::BinaryPayload { data: value.data })
}

pub fn decode_batch_bytes(
    payload: Option<&MessageBytes>,
    space: &native::SpaceSpec,
) -> Result<Vec<native::SpaceValue>, ProtocolError> {
    let Some(bytes) = payload else {
        return Ok(vec![]);
    };
    let decoded = decode_proto_value(&bytes.data)?;
    let values = expect_list_value(&decoded, "batched space value")?;
    values
        .values
        .iter()
        .map(|value| decode_value_for_space(value, space))
        .collect()
}

#[doc(hidden)]
pub fn encode_batched_partial_values(
    values: &[native::SpaceValue],
    space: &native::SpaceSpec,
) -> Result<MessageBytes, ProtocolError> {
    if uses_raw_batch_encoding(space) {
        let raw = values
            .iter()
            .map(|value| encode_space_value(value, space))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        Ok(binary_to_bytes(&native::BinaryPayload { data: raw }))
    } else {
        encode_batch_bytes(values, space)
    }
}

#[doc(hidden)]
pub fn decode_batched_partial_values(
    payload: Option<&MessageBytes>,
    space: &native::SpaceSpec,
) -> Result<Vec<native::SpaceValue>, ProtocolError> {
    if uses_raw_batch_encoding(space) {
        decode_raw_batched_values(payload.map(|value| value.data.as_slice()), space)
    } else {
        decode_batch_bytes(payload, space)
    }
}

pub fn optional_bytes_to_binary(
    payload: Option<&MessageBytes>,
) -> Result<Option<native::BinaryPayload>, ProtocolError> {
    let Some(data) = payload.cloned() else {
        return Ok(None);
    };
    Ok(Some(bytes_to_binary(data)?))
}

fn encode_space_value(
    value: &native::SpaceValue,
    space: &native::SpaceSpec,
) -> Result<Vec<u8>, ProtocolError> {
    match (space.spec.as_ref(), value) {
        (Some(native::space_spec::Spec::Box(_)), native::SpaceValue::Box(value)) => {
            Ok(value.data.clone())
        }
        (Some(native::space_spec::Spec::Discrete(_)), native::SpaceValue::Discrete(value)) => {
            encode_scalar(*value, space.dtype)
        }
        (
            Some(native::space_spec::Spec::MultiBinary(_)),
            native::SpaceValue::MultiBinary(values),
        ) => Ok(values.iter().map(|value| u8::from(*value)).collect()),
        (
            Some(native::space_spec::Spec::MultiDiscrete(_)),
            native::SpaceValue::MultiDiscrete(values),
        ) => encode_int_sequence(values, space.dtype),
        (Some(native::space_spec::Spec::Text(_)), native::SpaceValue::Text(value)) => {
            Ok(encode_proto_value(&Value {
                kind: Some(value::Kind::StringValue(value.clone())),
            }))
        }
        (Some(native::space_spec::Spec::Dict(spec)), native::SpaceValue::Dict(values)) => {
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
            Ok(encode_proto_value(&Value {
                kind: Some(value::Kind::StructValue(Struct { fields })),
            }))
        }
        (Some(native::space_spec::Spec::Tuple(spec)), native::SpaceValue::Tuple(values)) => {
            let values = values
                .iter()
                .zip(spec.spaces.iter())
                .map(|(value, space)| encode_value_for_space(value, space))
                .collect::<Result<_, _>>()?;
            Ok(encode_proto_value(&Value {
                kind: Some(value::Kind::ListValue(ListValue { values })),
            }))
        }
        _ => Err(ProtocolError::EncodeError(format!(
            "value kind did not match space {:?}",
            space.space_type()
        ))),
    }
}

fn decode_space_value(
    bytes: &[u8],
    space: &native::SpaceSpec,
) -> Result<native::SpaceValue, ProtocolError> {
    match space.spec.as_ref() {
        Some(native::space_spec::Spec::Box(_)) => Ok(native::SpaceValue::Box(native::BoxValue {
            data: bytes.to_vec(),
            shape: space.shape.clone(),
            dtype: space.dtype,
        })),
        Some(native::space_spec::Spec::Discrete(_)) => Ok(native::SpaceValue::Discrete(
            decode_scalar(bytes, space.dtype)?,
        )),
        Some(native::space_spec::Spec::MultiBinary(_)) => Ok(native::SpaceValue::MultiBinary(
            bytes.iter().map(|value| *value != 0).collect(),
        )),
        Some(native::space_spec::Spec::MultiDiscrete(_)) => Ok(native::SpaceValue::MultiDiscrete(
            decode_int_sequence(bytes, space.dtype)?,
        )),
        Some(native::space_spec::Spec::Text(_)) => match decode_proto_value(bytes)?.kind {
            Some(value::Kind::StringValue(value)) => Ok(native::SpaceValue::Text(value)),
            _ => Err(ProtocolError::DecodeError(
                "text transport payload was not a string".to_string(),
            )),
        },
        Some(native::space_spec::Spec::Dict(spec)) => {
            let value = decode_proto_value(bytes)?;
            let struct_value = expect_struct_value(&value, "dict")?;
            let mut decoded = native::MetaMap::new();
            let mut result = std::collections::BTreeMap::new();
            let _ = &mut decoded;
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
        Some(native::space_spec::Spec::Tuple(spec)) => {
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
    encode_space_value(value, space)
}

pub fn decode_space_value_bytes(
    bytes: &[u8],
    space: &native::SpaceSpec,
) -> Result<native::SpaceValue, ProtocolError> {
    decode_space_value(bytes, space)
}

fn decode_raw_batched_values(
    raw: Option<&[u8]>,
    space: &native::SpaceSpec,
) -> Result<Vec<native::SpaceValue>, ProtocolError> {
    let Some(raw) = raw else {
        return Ok(vec![]);
    };

    let bytes_per_value = raw_value_size(space).ok_or_else(|| {
        ProtocolError::DecodeError("space does not support fixed-size raw env batching".to_string())
    })?;

    if bytes_per_value == 0 {
        return Err(ProtocolError::DecodeError(
            "space has zero-sized raw encoding".to_string(),
        ));
    }

    if !raw.len().is_multiple_of(bytes_per_value) {
        return Err(ProtocolError::DecodeError(format!(
            "batched raw payload length {} is not divisible by per-env size {bytes_per_value}",
            raw.len()
        )));
    }

    raw.chunks(bytes_per_value)
        .map(|chunk| decode_space_value(chunk, space))
        .collect()
}

fn uses_raw_batch_encoding(space: &native::SpaceSpec) -> bool {
    matches!(
        space.spec.as_ref(),
        Some(native::space_spec::Spec::Box(_))
            | Some(native::space_spec::Spec::Discrete(_))
            | Some(native::space_spec::Spec::MultiBinary(_))
            | Some(native::space_spec::Spec::MultiDiscrete(_))
    )
}

fn raw_value_size(space: &native::SpaceSpec) -> Option<usize> {
    if !uses_raw_batch_encoding(space) {
        return None;
    }

    let item_count = match space.spec.as_ref() {
        Some(native::space_spec::Spec::Discrete(_)) => 1,
        _ => {
            if space.shape.is_empty() {
                return None;
            }
            space.shape.iter().try_fold(1usize, |acc, dim| {
                usize::try_from(*dim)
                    .ok()
                    .and_then(|dim| acc.checked_mul(dim))
            })?
        }
    };

    let dtype_size = native::dtype_size(space.dtype);
    (dtype_size > 0).then_some(item_count * dtype_size)
}

fn encode_value_for_space(
    value: &native::SpaceValue,
    space: &native::SpaceSpec,
) -> Result<Value, ProtocolError> {
    match value {
        native::SpaceValue::Box(boxed) => Ok(Value {
            kind: Some(value::Kind::StringValue(BASE64.encode(&boxed.data))),
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

fn decode_value_for_space(
    value: &Value,
    space: &native::SpaceSpec,
) -> Result<native::SpaceValue, ProtocolError> {
    match space.spec.as_ref() {
        Some(native::space_spec::Spec::Box(_)) => {
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
            Ok(native::SpaceValue::Box(native::BoxValue {
                data,
                shape: space.shape.clone(),
                dtype: space.dtype,
            }))
        }
        Some(native::space_spec::Spec::Discrete(_)) => match value.kind {
            Some(value::Kind::NumberValue(number)) => {
                Ok(native::SpaceValue::Discrete(number as i64))
            }
            Some(value::Kind::BoolValue(flag)) => Ok(native::SpaceValue::Discrete(i64::from(flag))),
            _ => Err(ProtocolError::DecodeError(
                "discrete payload was not numeric".to_string(),
            )),
        },
        Some(native::space_spec::Spec::MultiBinary(_)) => Ok(native::SpaceValue::MultiBinary(
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
        Some(native::space_spec::Spec::MultiDiscrete(_)) => Ok(native::SpaceValue::MultiDiscrete(
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
        Some(native::space_spec::Spec::Text(_)) => match &value.kind {
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

fn encode_scalar(value: i64, dtype: native::DType) -> Result<Vec<u8>, ProtocolError> {
    encode_scalars(&[ScalarValue::Int(value)], dtype)
}

fn encode_int_sequence(values: &[i64], dtype: native::DType) -> Result<Vec<u8>, ProtocolError> {
    let scalars = values
        .iter()
        .copied()
        .map(ScalarValue::Int)
        .collect::<Vec<_>>();
    encode_scalars(&scalars, dtype)
}

fn decode_scalar(bytes: &[u8], dtype: native::DType) -> Result<i64, ProtocolError> {
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

fn decode_int_sequence(bytes: &[u8], dtype: native::DType) -> Result<Vec<i64>, ProtocolError> {
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

fn encode_scalars(values: &[ScalarValue], dtype: native::DType) -> Result<Vec<u8>, ProtocolError> {
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
            (native::DType::Float16, ScalarValue::Float(value)) => {
                encoded.extend_from_slice(&f16::from_f64(*value).to_le_bytes())
            }
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
            | (native::DType::Int64, ScalarValue::Bool(value)) => encoded.extend_from_slice(
                &encode_scalars(&[ScalarValue::Int(i64::from(*value))], dtype)?,
            ),
            (native::DType::Uint8, ScalarValue::Float(value)) => encoded.push(*value as u8),
            (native::DType::Int32, ScalarValue::Float(value))
            | (native::DType::Int64, ScalarValue::Float(value)) => encoded
                .extend_from_slice(&encode_scalars(&[ScalarValue::Int(*value as i64)], dtype)?),
        }
    }
    Ok(encoded)
}

fn decode_scalars(bytes: &[u8], dtype: native::DType) -> Result<Vec<ScalarValue>, ProtocolError> {
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
        native::DType::Unspecified => Err(ProtocolError::DecodeError(
            "cannot decode scalar bytes with unspecified dtype".to_string(),
        )),
    }
}

fn decode_proto_array(value: &Value) -> Result<Vec<ScalarValue>, ProtocolError> {
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

fn expect_struct_value<'a>(value: &'a Value, label: &str) -> Result<&'a Struct, ProtocolError> {
    match &value.kind {
        Some(value::Kind::StructValue(struct_value)) => Ok(struct_value),
        _ => Err(ProtocolError::DecodeError(format!(
            "{label} transport payload was not a struct"
        ))),
    }
}

fn expect_list_value<'a>(value: &'a Value, label: &str) -> Result<&'a ListValue, ProtocolError> {
    match &value.kind {
        Some(value::Kind::ListValue(list_value)) => Ok(list_value),
        _ => Err(ProtocolError::DecodeError(format!(
            "{label} transport payload was not a list"
        ))),
    }
}

fn scalar_to_proto_value(value: impl Into<ScalarValue>) -> Value {
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

fn encode_proto_value(value: &Value) -> Vec<u8> {
    value.encode_to_vec()
}

fn decode_proto_value(bytes: &[u8]) -> Result<Value, ProtocolError> {
    Value::decode(bytes)
        .map_err(|err| ProtocolError::DecodeError(format!("failed to decode value payload: {err}")))
}

#[derive(Debug, Clone)]
enum ScalarValue {
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
