use std::collections::HashSet;

use rlmesh_proto::model::v1::PredictContext;

use crate::error::{Error as GrpcError, ProtocolError};

pub(super) fn route_request_id<F>(context: Option<&PredictContext>, fallback: F) -> String
where
    F: FnOnce() -> String,
{
    context
        .map(|context| context.request_id.clone())
        .filter(|request_id| !request_id.is_empty())
        .unwrap_or_else(fallback)
}

pub(super) fn validate_route(context: &PredictContext) -> Result<(), GrpcError> {
    if context.route_id.is_empty() {
        return Err(decode_error("model route_id is empty"));
    }
    if context.request_id.is_empty() {
        return Err(decode_error("model request_id is empty"));
    }
    Ok(())
}

pub(super) fn validate_predict_route(context: &PredictContext) -> Result<(), GrpcError> {
    validate_route(context)?;
    if context.slots.is_empty() {
        return Err(decode_error("model route must include at least one slot"));
    }

    let mut env_indexes = HashSet::new();
    for (index, slot) in context.slots.iter().enumerate() {
        if slot.episode_id.is_empty() {
            return Err(decode_error(format!(
                "model route slot {index} missing episode_id"
            )));
        }
        if slot.env_index < 0 {
            return Err(decode_error(format!(
                "model route slot {index} has negative env_index {}",
                slot.env_index
            )));
        }
        if !env_indexes.insert(slot.env_index) {
            return Err(decode_error(format!(
                "model route has duplicate env_index {}",
                slot.env_index
            )));
        }
    }

    Ok(())
}

pub(super) fn decode_error(message: impl Into<String>) -> GrpcError {
    GrpcError::Protocol(ProtocolError::DecodeError(message.into()))
}
