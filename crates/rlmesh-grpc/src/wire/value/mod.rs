//! Conversions between native space values and their wire encodings.
//!
//! Submodules split the codec by concern: `interaction` adapts
//! reset/step/render requests and results, `payload` wraps values in
//! transport messages, `codec` encodes single space values, `batch` handles
//! batched payloads, and `scalars` packs scalar element bytes.

mod batch;
mod codec;
mod interaction;
mod payload;
mod proto_value;
mod scalars;

#[cfg(test)]
mod tests;

pub use batch::{
    decode_batch_bytes, decode_batched_partial_values, encode_batch_bytes,
    encode_batched_partial_values,
};
pub use codec::{decode_space_value_bytes, encode_space_value_bytes};
pub use interaction::{
    render_request_to_proto, render_result_from_proto, render_result_to_proto,
    reset_request_to_proto, reset_result_from_proto, reset_result_to_proto, step_request_to_proto,
    step_result_from_proto, step_result_to_proto,
};
pub use payload::{
    binary_to_bytes, bytes_to_binary, bytes_value, decode_value, decode_value_bytes, encode_value,
    encode_value_bytes, optional_bytes_to_binary, value_bytes, value_bytes_ref,
};
