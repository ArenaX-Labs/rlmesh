use prost::Message;
use rlmesh_proto::common::v1::MessageBytes;
use rlmesh_proto::spaces::v1::ValueList;
use rlmesh_spaces as native;

use crate::error::ProtocolError;

use super::codec::{decode_space_value, decode_value_node, encode_space_value, encode_value_node};

pub fn encode_batch_bytes(
    values: &[native::SpaceValue],
    space: &native::SpaceSpec,
) -> Result<MessageBytes, ProtocolError> {
    let items = values
        .iter()
        .map(|value| encode_value_node(value, space))
        .collect::<Result<_, _>>()?;

    Ok(MessageBytes {
        data: ValueList { items }.encode_to_vec(),
    })
}

pub fn decode_batch_bytes(
    payload: Option<&MessageBytes>,
    space: &native::SpaceSpec,
) -> Result<Vec<native::SpaceValue>, ProtocolError> {
    let Some(bytes) = payload else {
        return Ok(vec![]);
    };
    let list = ValueList::decode(bytes.data.as_slice()).map_err(|err| {
        ProtocolError::DecodeError(format!("failed to decode batched space value: {err}"))
    })?;
    list.items
        .iter()
        .map(|node| decode_value_node(node, space))
        .collect()
}

#[doc(hidden)]
pub fn encode_batched_partial_values(
    values: &[native::SpaceValue],
    space: &native::SpaceSpec,
) -> Result<MessageBytes, ProtocolError> {
    if uses_raw_batch_encoding(space) {
        let mut raw = Vec::new();
        for value in values {
            raw.extend_from_slice(&encode_space_value(value, space)?);
        }
        Ok(MessageBytes { data: raw })
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
        Some(native::SpaceKind::Box(_))
            | Some(native::SpaceKind::Discrete(_))
            | Some(native::SpaceKind::MultiBinary(_))
            | Some(native::SpaceKind::MultiDiscrete(_))
    )
}

fn raw_value_size(space: &native::SpaceSpec) -> Option<usize> {
    if !uses_raw_batch_encoding(space) {
        return None;
    }

    let item_count = match space.spec.as_ref() {
        Some(native::SpaceKind::Discrete(_)) => 1,
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
