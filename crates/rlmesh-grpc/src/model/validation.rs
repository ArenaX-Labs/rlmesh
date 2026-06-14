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

#[cfg(test)]
mod tests {
    use rlmesh_proto::model::v1::{PredictContext, PredictSlot};

    use super::validate_predict_route;

    fn route_context() -> PredictContext {
        PredictContext {
            session_id: "session-1".to_string(),
            route_id: "route-1".to_string(),
            request_id: "request-1".to_string(),
            slots: vec![PredictSlot {
                episode_id: "episode-1".to_string(),
                env_index: 0,
                step: 3,
                reset: false,
            }],
        }
    }

    #[test]
    fn model_predict_request_accepts_route_context() {
        validate_predict_route(&route_context()).unwrap();
    }

    #[test]
    fn model_predict_request_rejects_empty_route_slots() {
        let mut route = route_context();
        route.slots.clear();

        let err = validate_predict_route(&route).unwrap_err();

        assert!(err.to_string().contains("at least one slot"));
    }

    #[test]
    fn model_predict_request_rejects_incomplete_slots() {
        let mut route = route_context();
        route.slots[0].episode_id.clear();

        let err = validate_predict_route(&route).unwrap_err();

        assert!(err.to_string().contains("missing episode_id"));
    }
}
