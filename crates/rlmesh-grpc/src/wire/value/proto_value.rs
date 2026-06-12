use prost::Message;
use prost_types::{ListValue, Struct, Value, value};

use crate::error::ProtocolError;

pub(super) fn expect_struct_value<'a>(
    value: &'a Value,
    label: &str,
) -> Result<&'a Struct, ProtocolError> {
    match &value.kind {
        Some(value::Kind::StructValue(struct_value)) => Ok(struct_value),
        _ => Err(ProtocolError::DecodeError(format!(
            "{label} transport payload was not a struct"
        ))),
    }
}

pub(super) fn expect_list_value<'a>(
    value: &'a Value,
    label: &str,
) -> Result<&'a ListValue, ProtocolError> {
    match &value.kind {
        Some(value::Kind::ListValue(list_value)) => Ok(list_value),
        _ => Err(ProtocolError::DecodeError(format!(
            "{label} transport payload was not a list"
        ))),
    }
}

pub(super) fn encode_proto_value(value: &Value) -> Vec<u8> {
    value.encode_to_vec()
}

pub(super) fn decode_proto_value(bytes: &[u8]) -> Result<Value, ProtocolError> {
    Value::decode(bytes)
        .map_err(|err| ProtocolError::DecodeError(format!("failed to decode value payload: {err}")))
}
