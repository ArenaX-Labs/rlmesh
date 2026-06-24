use rlmesh_proto::spaces::v1::SpaceValue;
use rlmesh_spaces as native;

use crate::error::ProtocolError;

use super::leaves::{decode_leaf_slab, encode_leaf_slab};
use super::payload::{leaves_value, value_leaves};

/// Encode `values.len()` lanes into one batched wire [`SpaceValue`] (row-major
/// `(N, *shape)` slab per leaf).
#[doc(hidden)]
pub fn encode_batched_partial_values(
    values: &[native::SpaceValue],
    space: &native::SpaceSpec,
) -> Result<SpaceValue, ProtocolError> {
    Ok(leaves_value(encode_leaf_slab(values, space)?))
}

/// Decode a batched wire value into `n` per-lane typed values. **`n` is an
/// authoritative input — the lane count is never recovered by slab division.**
#[doc(hidden)]
pub fn decode_batched_partial_values(
    payload: Option<&SpaceValue>,
    space: &native::SpaceSpec,
    n: usize,
) -> Result<Vec<native::SpaceValue>, ProtocolError> {
    match value_leaves(payload) {
        Some(leaves) => decode_leaf_slab(leaves, space, n),
        None => Ok(Vec::new()),
    }
}
