use std::borrow::Cow;

use prost::Message;
use prost::bytes::Bytes;
use prost_types::{Value, value};
use rlmesh_proto::spaces::v1::space_value_node::Kind as NodeKind;
use rlmesh_proto::spaces::v1::{SpaceValueNode, ValueList, ValueMap};
use rlmesh_spaces as native;

use crate::error::ProtocolError;

use super::scalars::{decode_int_sequence, decode_scalar, encode_int_sequence, encode_scalar};

fn encode_proto_value(value: &Value) -> Vec<u8> {
    value.encode_to_vec()
}

fn decode_proto_value(bytes: &[u8]) -> Result<Value, ProtocolError> {
    Value::decode(bytes)
        .map_err(|err| ProtocolError::DecodeError(format!("failed to decode value payload: {err}")))
}

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
        (Some(native::SpaceKind::Dict(_)), native::SpaceValue::Dict(_))
        | (Some(native::SpaceKind::Tuple(_)), native::SpaceValue::Tuple(_)) => {
            Ok(Cow::Owned(encode_value_node(value, space)?.encode_to_vec()))
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
        Some(native::SpaceKind::Dict(_)) | Some(native::SpaceKind::Tuple(_)) => {
            let node = SpaceValueNode::decode(bytes).map_err(|err| {
                ProtocolError::DecodeError(format!("failed to decode composite value: {err}"))
            })?;
            decode_value_node(&node, space)
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

/// Encode one space value as a recursive `SpaceValueNode`. Leaf arms carry the
/// same exact raw encoding used at the top level (raw little-endian bytes for
/// tensor-shaped leaves, exact int64 for Discrete) — no base64, no f64.
pub(super) fn encode_value_node(
    value: &native::SpaceValue,
    space: &native::SpaceSpec,
) -> Result<SpaceValueNode, ProtocolError> {
    let kind = match value {
        native::SpaceValue::Box(tensor) => {
            if !matches!(space.spec.as_ref(), Some(native::SpaceKind::Box(_))) {
                return Err(ProtocolError::EncodeError(format!(
                    "value kind did not match space {:?}",
                    space.space_type()
                )));
            }
            NodeKind::Tensor(tensor_wire_bytes(tensor))
        }
        native::SpaceValue::MultiBinary(_) | native::SpaceValue::MultiDiscrete(_) => {
            // These arms always encode into a fresh Vec, so into_owned() is a
            // move and Bytes::from is refcount-only — no copy.
            NodeKind::Multi(encode_space_value(value, space)?.into_owned().into())
        }
        native::SpaceValue::Discrete(value) => NodeKind::Discrete(*value),
        native::SpaceValue::Text(value) => NodeKind::Text(value.clone()),
        native::SpaceValue::Dict(values) => {
            let Some(native::SpaceKind::Dict(spec)) = space.spec.as_ref() else {
                return Err(ProtocolError::EncodeError(
                    "dict value did not match dict space".to_string(),
                ));
            };
            let mut entries = std::collections::HashMap::with_capacity(spec.keys.len());
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                let child = values.get(key).ok_or_else(|| {
                    ProtocolError::EncodeError(format!("missing dict key '{key}'"))
                })?;
                entries.insert(key.clone(), encode_value_node(child, child_space)?);
            }
            NodeKind::Dict(ValueMap { entries })
        }
        native::SpaceValue::Tuple(values) => {
            let Some(native::SpaceKind::Tuple(spec)) = space.spec.as_ref() else {
                return Err(ProtocolError::EncodeError(
                    "tuple value did not match tuple space".to_string(),
                ));
            };
            let items = values
                .iter()
                .zip(spec.spaces.iter())
                .map(|(value, child_space)| encode_value_node(value, child_space))
                .collect::<Result<_, _>>()?;
            NodeKind::Tuple(ValueList { items })
        }
    };
    Ok(SpaceValueNode { kind: Some(kind) })
}

/// The tensor's element bytes as a wire-ready [`Bytes`].
///
/// A contiguous tensor shares its refcounted [`Storage`](native::Storage)
/// with the message — no element bytes are copied until the node tree is
/// serialized. Non-contiguous layouts gather into a fresh buffer, which
/// `Bytes` then adopts without a further copy.
fn tensor_wire_bytes(tensor: &native::Tensor) -> Bytes {
    match tensor.to_contiguous_bytes() {
        Cow::Borrowed(_) => {
            let start = tensor.byte_offset();
            Bytes::from_owner(SharedStorage(tensor.storage().clone()))
                .slice(start..start + tensor.nbytes())
        }
        Cow::Owned(gathered) => Bytes::from(gathered),
    }
}

/// Adapter giving [`Bytes::from_owner`] a view of a tensor's refcounted
/// storage, keeping the allocation alive for the message's lifetime.
struct SharedStorage(native::Storage);

impl AsRef<[u8]> for SharedStorage {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

/// Decode a recursive `SpaceValueNode` against its space spec.
pub(super) fn decode_value_node(
    node: &SpaceValueNode,
    space: &native::SpaceSpec,
) -> Result<native::SpaceValue, ProtocolError> {
    match (space.spec.as_ref(), node.kind.as_ref()) {
        (Some(native::SpaceKind::Box(_)), Some(NodeKind::Tensor(raw))) => {
            Ok(native::SpaceValue::Box(
                native::Tensor::from_slice(raw, &space.shape, space.dtype).map_err(|err| {
                    ProtocolError::DecodeError(format!("invalid box payload: {err}"))
                })?,
            ))
        }
        (Some(native::SpaceKind::Discrete(_)), Some(NodeKind::Discrete(value))) => {
            Ok(native::SpaceValue::Discrete(*value))
        }
        (Some(native::SpaceKind::MultiBinary(_)), Some(NodeKind::Multi(raw))) => Ok(
            native::SpaceValue::MultiBinary(raw.iter().map(|value| *value != 0).collect()),
        ),
        (Some(native::SpaceKind::MultiDiscrete(_)), Some(NodeKind::Multi(raw))) => Ok(
            native::SpaceValue::MultiDiscrete(decode_int_sequence(raw, space.dtype)?),
        ),
        (Some(native::SpaceKind::Text(_)), Some(NodeKind::Text(text))) => {
            Ok(native::SpaceValue::Text(text.clone()))
        }
        (Some(native::SpaceKind::Dict(spec)), Some(NodeKind::Dict(map))) => {
            let mut result = std::collections::BTreeMap::new();
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                let child = map.entries.get(key).ok_or_else(|| {
                    ProtocolError::DecodeError(format!("missing dict field '{key}'"))
                })?;
                result.insert(key.clone(), decode_value_node(child, child_space)?);
            }
            Ok(native::SpaceValue::Dict(result))
        }
        (Some(native::SpaceKind::Tuple(spec)), Some(NodeKind::Tuple(list))) => {
            if list.items.len() != spec.spaces.len() {
                return Err(ProtocolError::DecodeError(format!(
                    "tuple payload length {} did not match tuple space length {}",
                    list.items.len(),
                    spec.spaces.len()
                )));
            }
            Ok(native::SpaceValue::Tuple(
                list.items
                    .iter()
                    .zip(spec.spaces.iter())
                    .map(|(node, child_space)| decode_value_node(node, child_space))
                    .collect::<Result<_, _>>()?,
            ))
        }
        _ => Err(ProtocolError::DecodeError(format!(
            "composite value node did not match space {:?}",
            space.space_type()
        ))),
    }
}
