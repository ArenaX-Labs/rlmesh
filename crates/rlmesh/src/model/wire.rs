use std::collections::HashSet;
use std::time::Instant;

use rlmesh_grpc::wire::{binary_to_bytes, bytes_value, optional_bytes_to_binary, value_bytes};
use rlmesh_proto::core::v1::{OperationMetric, OperationTelemetry, operation_metric};
use rlmesh_proto::model::v1::{
    ModelError, ModelErrorCode, PredictContext, PredictRequest, PredictResponse, PredictSlot,
    join_request, join_response,
};

use super::types::{ModelObservation, ModelRouteContext, ModelRouteSlot};
use crate::{Error, Result, spaces};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ModelAction {
    pub(super) action: Option<spaces::BinaryPayload>,
    pub(super) route: ModelRouteContext,
    pub(super) telemetry: Option<OperationTelemetry>,
}

pub(super) fn model_error(message: impl Into<String>) -> join_response::Kind {
    join_response::Kind::Error(ModelError {
        code: ModelErrorCode::InvalidRequest as i32,
        message: message.into(),
        is_recoverable: false,
        debug_info: String::new(),
    })
}

/// Map a facade [`Error`] returned by a handler onto the wire model-error,
/// preserving the handler-fault vs internal-fault distinction and the
/// recoverable flag so the caller can react appropriately.
pub(super) fn model_error_from_error(error: &Error) -> join_response::Kind {
    let (code, is_recoverable) = match error {
        Error::Model(model) => (ModelErrorCode::Internal, model.is_recoverable),
        _ => (ModelErrorCode::Internal, false),
    };
    join_response::Kind::Error(ModelError {
        code: code as i32,
        message: error.to_string(),
        is_recoverable,
        debug_info: String::new(),
    })
}

pub(super) fn model_join_request_operation(kind: Option<&join_request::Kind>) -> &'static str {
    match kind {
        Some(join_request::Kind::ConfigureRoute(_)) => "model.configure_route",
        Some(join_request::Kind::Predict(_)) => "model.predict",
        Some(join_request::Kind::CloseRoute(_)) => "model.close_route",
        Some(join_request::Kind::Close(_)) => "model.close",
        None => "model.unknown",
    }
}

pub(super) fn model_operation_telemetry(
    operation: &str,
    started_at: Instant,
) -> OperationTelemetry {
    OperationTelemetry {
        operation: operation.to_string(),
        component_id: String::new(),
        metrics: vec![OperationMetric {
            name: "endpoint.total".to_string(),
            labels: Default::default(),
            value: Some(operation_metric::Value::DurationNs(
                started_at.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
            )),
        }],
    }
}

pub(super) fn model_observation_from_endpoint_request(
    request: PredictRequest,
) -> Result<ModelObservation> {
    let route = model_route_from_proto(request.context)?;
    let num_envs = route.slots.len();
    validate_predict_route(&route)?;

    Ok(ModelObservation {
        observation: optional_bytes_to_binary(value_bytes(request.observation.as_ref())?.as_ref())?,
        reset: route.primary_reset(),
        num_envs,
        env_contract: None,
        route,
    })
}

pub(super) fn model_action_to_endpoint_response(action: ModelAction) -> PredictResponse {
    let _telemetry = action.telemetry;
    PredictResponse {
        context: Some((&action.route).into()),
        action: action.action.as_ref().map(binary_to_bytes).map(bytes_value),
    }
}

fn model_route_from_proto(route: Option<PredictContext>) -> Result<ModelRouteContext> {
    route
        .map(ModelRouteContext::from)
        .ok_or_else(|| Error::Internal("model request missing route context".to_string()))
}

fn validate_predict_route(route: &ModelRouteContext) -> Result<()> {
    if route.route_id.is_empty() {
        return Err(Error::Internal("model route_id is empty".to_string()));
    }
    if route.request_id.is_empty() {
        return Err(Error::Internal("model request_id is empty".to_string()));
    }
    if route.slots.is_empty() {
        return Err(Error::Internal(
            "model route must include at least one slot".to_string(),
        ));
    }

    let mut env_indexes = HashSet::new();
    for (index, slot) in route.slots.iter().enumerate() {
        if slot.episode_id.is_empty() {
            return Err(Error::Internal(format!(
                "model route slot {index} missing episode_id"
            )));
        }
        if slot.env_index < 0 {
            return Err(Error::Internal(format!(
                "model route slot {index} has negative env_index {}",
                slot.env_index
            )));
        }
        if !env_indexes.insert(slot.env_index) {
            return Err(Error::Internal(format!(
                "model route has duplicate env_index {}",
                slot.env_index
            )));
        }
    }

    Ok(())
}

impl From<PredictContext> for ModelRouteContext {
    fn from(value: PredictContext) -> Self {
        Self {
            session_id: value.session_id,
            route_id: value.route_id,
            request_id: value.request_id,
            slots: value.slots.into_iter().map(ModelRouteSlot::from).collect(),
        }
    }
}

impl From<&ModelRouteContext> for PredictContext {
    fn from(value: &ModelRouteContext) -> Self {
        Self {
            session_id: value.session_id.clone(),
            route_id: value.route_id.clone(),
            request_id: value.request_id.clone(),
            slots: value.slots.iter().map(PredictSlot::from).collect(),
        }
    }
}

impl From<PredictSlot> for ModelRouteSlot {
    fn from(value: PredictSlot) -> Self {
        Self {
            episode_id: value.episode_id,
            env_index: value.env_index,
            step: value.step,
            reset: value.reset,
        }
    }
}

impl From<&ModelRouteSlot> for PredictSlot {
    fn from(value: &ModelRouteSlot) -> Self {
        Self {
            episode_id: value.episode_id.clone(),
            env_index: value.env_index,
            step: value.step,
            reset: value.reset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unwrap_error(kind: join_response::Kind) -> ModelError {
        match kind {
            join_response::Kind::Error(error) => error,
            other => panic!("expected model error, got {other:?}"),
        }
    }

    #[test]
    fn handler_model_error_preserves_recoverability_on_the_wire() {
        // A recoverable handler decline must surface as a recoverable model
        // error, not a non-recoverable internal/transport fault.
        let recoverable = unwrap_error(model_error_from_error(&Error::model_recoverable(
            "retry me",
        )));
        assert!(recoverable.is_recoverable);
        assert_eq!(recoverable.code, ModelErrorCode::Internal as i32);
        assert!(recoverable.message.contains("retry me"));

        let permanent = unwrap_error(model_error_from_error(&Error::model("bad observation")));
        assert!(!permanent.is_recoverable);

        // A genuine internal fault is never reported as recoverable.
        let internal = unwrap_error(model_error_from_error(&Error::Internal("boom".to_string())));
        assert!(!internal.is_recoverable);
    }
}
