pub mod spaces;
pub mod value;

/// Re-exported so consumers can name the leaf byte type without a direct `prost`
/// dependency (the wire value's leaves are `bytes::Bytes`).
pub use prost::bytes::Bytes;

pub use spaces::{
    env_contract_from_proto, env_contract_to_proto, env_spec_from_proto, env_spec_to_proto,
    meta_map_from_proto, meta_map_to_proto, space_spec_from_proto, space_spec_to_proto,
};
pub use value::{
    binary_to_bytes, decode_batched_partial_values, decode_value, encode_batched_partial_values,
    encode_value, leaves_value, optional_bytes_to_binary, render_request_to_proto,
    render_result_from_proto, render_result_to_proto, reset_request_to_proto,
    reset_result_from_proto, reset_result_to_proto, step_request_to_proto, step_result_from_proto,
    step_result_to_proto, value_leaves,
};
