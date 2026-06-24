use prost::bytes::Bytes;
use rlmesh_proto::spaces::v1::SpaceValue;
use rlmesh_spaces as native;

use crate::error::ProtocolError;

use super::leaves::{decode_leaves, encode_leaves};

/// Wrap leaf byte slabs into a wire [`SpaceValue`].
pub fn leaves_value(leaves: Vec<Bytes>) -> SpaceValue {
    SpaceValue { leaves }
}

/// The leaf slabs of a wire value, if present.
pub fn value_leaves(payload: Option<&SpaceValue>) -> Option<&[Bytes]> {
    payload.map(|payload| payload.leaves.as_slice())
}

/// Encode a single typed value into a wire [`SpaceValue`].
pub fn encode_value(
    value: &native::SpaceValue,
    space: &native::SpaceSpec,
) -> Result<SpaceValue, ProtocolError> {
    Ok(leaves_value(encode_leaves(value, space)?))
}

/// Decode a wire [`SpaceValue`] back to a typed value (`None` when absent).
pub fn decode_value(
    payload: Option<&SpaceValue>,
    space: &native::SpaceSpec,
) -> Result<Option<native::SpaceValue>, ProtocolError> {
    match value_leaves(payload) {
        Some(leaves) => Ok(Some(decode_leaves(leaves, space)?)),
        None => Ok(None),
    }
}

pub fn binary_to_bytes(value: &native::BinaryPayload) -> Bytes {
    Bytes::from(value.data.clone())
}

pub fn bytes_to_binary(value: Bytes) -> Result<native::BinaryPayload, ProtocolError> {
    Ok(native::BinaryPayload {
        data: value.to_vec(),
    })
}

pub fn optional_bytes_to_binary(
    payload: Option<&Bytes>,
) -> Result<Option<native::BinaryPayload>, ProtocolError> {
    let Some(data) = payload else {
        return Ok(None);
    };
    Ok(Some(bytes_to_binary(data.clone())?))
}
