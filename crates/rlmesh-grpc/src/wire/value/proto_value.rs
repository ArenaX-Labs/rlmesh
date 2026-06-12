use prost::Message;
use prost_types::Value;

use crate::error::ProtocolError;

pub(super) fn encode_proto_value(value: &Value) -> Vec<u8> {
    value.encode_to_vec()
}

pub(super) fn decode_proto_value(bytes: &[u8]) -> Result<Value, ProtocolError> {
    Value::decode(bytes)
        .map_err(|err| ProtocolError::DecodeError(format!("failed to decode value payload: {err}")))
}
