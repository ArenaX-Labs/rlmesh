use prost::Message;
use prost::bytes::Bytes;
use rlmesh_proto::spaces::v1::SpaceValue;
use rlmesh_spaces as native;

use crate::error::ProtocolError;

use super::leaves::{decode_leaves, encode_leaves};

/// Wrap leaf byte slabs into a wire [`SpaceValue`].
pub fn leaves_value(leaves: Vec<Bytes>) -> SpaceValue {
    SpaceValue { leaves }
}

/// Serialize per-leaf wire bytes into one self-delimiting blob (an encoded
/// wire [`SpaceValue`]): the blessed way to persist a hook payload as a
/// single artifact. Plain concatenation loses the leaf boundaries — a
/// multi-leaf value can no longer be split, and a variable-length leaf
/// (`Text`) makes the fused bytes ambiguous. Inverse of [`leaves_from_blob`].
pub fn leaves_to_blob(leaves: &[Bytes]) -> Vec<u8> {
    leaves_value(leaves.to_vec()).encode_to_vec()
}

/// Split a [`leaves_to_blob`] blob back into per-leaf wire bytes, ready for
/// [`decode_leaves`] against the matching space spec.
pub fn leaves_from_blob(blob: &[u8]) -> Result<Vec<Bytes>, ProtocolError> {
    Ok(SpaceValue::decode(blob)
        .map_err(|e| ProtocolError::DecodeError(e.to_string()))?
        .leaves)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Boundaries survive the blob round-trip even when a variable-length
    /// leaf makes plain concatenation ambiguous, and an empty leaf stays
    /// distinguishable from an absent one.
    #[test]
    fn leaves_blob_round_trips_boundaries() {
        let leaves = vec![
            Bytes::from_static(b"pick up the red block"),
            Bytes::from_static(&[0u8; 28]),
            Bytes::new(),
        ];
        let restored = leaves_from_blob(&leaves_to_blob(&leaves)).unwrap();
        assert_eq!(restored, leaves);

        assert!(leaves_from_blob(&leaves_to_blob(&[])).unwrap().is_empty());
        assert!(leaves_from_blob(&[0xff, 0xff]).is_err());
    }
}
