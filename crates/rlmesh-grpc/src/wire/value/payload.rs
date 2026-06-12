use rlmesh_proto::common::v1::MessageBytes;
use rlmesh_proto::spaces::v1::SpaceValue;
use rlmesh_spaces as native;

use crate::error::ProtocolError;

use super::codec::{decode_space_value, encode_space_value};

pub fn encode_value_bytes(
    value: &native::SpaceValue,
    space: &native::SpaceSpec,
) -> Result<MessageBytes, ProtocolError> {
    Ok(MessageBytes {
        data: encode_space_value(value, space)?.into_owned(),
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

pub fn binary_to_bytes(value: &native::BinaryPayload) -> MessageBytes {
    MessageBytes {
        data: value.data.clone(),
    }
}

pub fn bytes_to_binary(value: MessageBytes) -> Result<native::BinaryPayload, ProtocolError> {
    Ok(native::BinaryPayload { data: value.data })
}

pub fn optional_bytes_to_binary(
    payload: Option<&MessageBytes>,
) -> Result<Option<native::BinaryPayload>, ProtocolError> {
    let Some(data) = payload.cloned() else {
        return Ok(None);
    };
    Ok(Some(bytes_to_binary(data)?))
}
