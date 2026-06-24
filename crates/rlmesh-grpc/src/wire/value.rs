//! Conversions between native space values and their wire encodings.
//!
//! Submodules split the codec by concern: `interaction` adapts
//! reset/step/render requests and results, `payload` wraps values in
//! transport messages, `codec` encodes single space values, `batch` handles
//! batched payloads, and `scalars` packs scalar element bytes.

mod batch;
mod codec;
mod interaction;
mod leaves;
mod payload;
mod scalars;

#[cfg(test)]
mod proptests;
#[cfg(test)]
mod tests;

pub use batch::{decode_batched_partial_values, encode_batched_partial_values};
pub use interaction::{
    render_request_to_proto, render_result_from_proto, render_result_to_proto,
    reset_request_to_proto, reset_result_from_proto, reset_result_to_proto, step_request_to_proto,
    step_result_from_proto, step_result_to_proto,
};
pub use leaves::{decode_leaf_slab, decode_leaves, encode_leaf_slab, encode_leaves};
pub use payload::{
    binary_to_bytes, bytes_to_binary, decode_value, encode_value, leaves_value,
    optional_bytes_to_binary, value_leaves,
};
