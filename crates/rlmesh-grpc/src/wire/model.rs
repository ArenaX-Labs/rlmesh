use rlmesh_proto::common::v1::MessageBytes;
use rlmesh_proto::spaces::v1::SpaceValue;

pub fn bytes_value(value: MessageBytes) -> SpaceValue {
    SpaceValue { bytes: Some(value) }
}

pub fn value_bytes(payload: Option<&SpaceValue>) -> Option<MessageBytes> {
    let payload = payload?;
    payload.bytes.clone()
}

pub fn value_bytes_ref(payload: Option<&SpaceValue>) -> Option<&MessageBytes> {
    let payload = payload?;
    payload.bytes.as_ref()
}
