//! Conversions between native space values and their wire encodings.

mod batch;
mod interaction;
mod payload;

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
pub use payload::{
    binary_to_bytes, bytes_to_binary, decode_value, encode_value, leaves_from_blob, leaves_to_blob,
    leaves_value, optional_bytes_to_binary, value_leaves,
};
