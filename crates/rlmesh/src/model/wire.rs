use std::collections::HashSet;
use std::time::Instant;

use rlmesh_grpc::wire::value_leaves;
use rlmesh_proto::model::v1::{
    ModelError, ModelErrorCode, PredictContext, PredictRequest, PredictResponse, PredictSlot,
    join_response,
};
use rlmesh_proto::spaces::v1::SpaceValue;

use super::types::{ModelObservation, ModelRouteContext, ModelRouteSlot};
use crate::{Error, Result, spaces};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ModelAction {
    /// Finished wire value built by the worker (codec already ran).
    pub(super) action: Option<SpaceValue>,
    pub(super) route: ModelRouteContext,
}

pub(super) fn model_error(message: impl Into<String>) -> join_response::Kind {
    join_response::Kind::Error(ModelError {
        code: ModelErrorCode::InvalidRequest as i32,
        message: message.into(),
        is_recoverable: false,
        debug_info: String::new(),
    })
}

/// Build the wire [`ModelError`] for a facade [`Error`] returned by a handler,
/// preserving the handler-fault vs internal-fault distinction and the
/// recoverable flag. Used directly by the grouped-predict path (which carries a
/// `ModelError` per group inside `GroupedPredictResult`) and wrapped by
/// [`model_error_from_error`] for the single-predict `JoinResponse.error` arm.
pub(super) fn model_error_value(error: &Error) -> ModelError {
    let (code, is_recoverable) = match error {
        Error::Model(model) => (ModelErrorCode::Internal, model.is_recoverable),
        _ => (ModelErrorCode::Internal, false),
    };
    ModelError {
        code: code as i32,
        message: error.to_string(),
        is_recoverable,
        debug_info: String::new(),
    }
}

/// Map a facade [`Error`] returned by a handler onto the wire model-error,
/// preserving the handler-fault vs internal-fault distinction and the
/// recoverable flag so the caller can react appropriately.
pub(super) fn model_error_from_error(error: &Error) -> join_response::Kind {
    join_response::Kind::Error(model_error_value(error))
}

/// Endpoint-local op duration in nanoseconds for the per-step
/// `JoinResponse.endpoint_total_ns` scalar. Replaces the old nested per-step
/// telemetry message construction (and with it the dead `labels` map and the
/// always-empty `component_id` — the runtime attributes by connection).
/// Saturates at `u64::MAX`.
pub(super) fn model_endpoint_total_ns(started_at: Instant) -> u64 {
    started_at.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

pub(super) fn model_observation_from_endpoint_request(
    request: PredictRequest,
) -> Result<ModelObservation> {
    let route = model_route_from_proto(request.context)?;
    let num_envs = route.slots.len();
    validate_predict_route(&route)?;

    Ok(ModelObservation {
        observation: value_leaves(request.observation.as_ref()).map(<[_]>::to_vec),
        reset: route.primary_reset(),
        num_envs,
        env_contract: None,
        route,
    })
}

pub(super) fn model_action_to_endpoint_response(action: ModelAction) -> PredictResponse {
    PredictResponse {
        context: Some((&action.route).into()),
        action: action.action,
    }
}

/// Structurally validate each per-lane action against the route's action space
/// before the codec encodes it. The spec-directed codec would otherwise silently
/// drop or reinterpret a mismatched typed action (extra Dict keys are skipped by
/// the spec-key walk; a wrong-dtype Box leaf is emitted as raw bytes and read
/// back at the spec dtype), and that value/spec mismatch never reaches the env's
/// own validation. Range deviations (Box bounds) pass through — those are the
/// env's validation policy to decide.
pub(super) fn check_actions_conform(
    action_space: &spaces::SpaceSpec,
    actions: &[spaces::SpaceValue],
) -> Result<()> {
    for (lane, action) in actions.iter().enumerate() {
        if let spaces::Conformance::Structural(err) = spaces::conform(action_space, action) {
            return Err(Error::model(format!(
                "model action for lane {lane} does not match the action space: {err}"
            )));
        }
    }
    Ok(())
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
        // Wire env_index is uint32, so a slot's native i32 is always >= 0 here.
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
            // Proto env_index is uint32; native is i32. `step` is int64 on both
            // sides, so it carries across with no conversion.
            env_index: i32::try_from(value.env_index).unwrap_or(i32::MAX),
            step: value.step,
            reset: value.reset,
        }
    }
}

impl From<&ModelRouteSlot> for PredictSlot {
    fn from(value: &ModelRouteSlot) -> Self {
        Self {
            episode_id: value.episode_id.clone(),
            // Native env_index is i32 (>=0); proto field is uint32. `step` is
            // int64 on both sides, so it carries across with no conversion.
            env_index: value.env_index.max(0) as u32,
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
    fn check_actions_conform_rejects_structural_mismatch() {
        let space = spaces::spaces::BoxSpaceBuilder::scalar(0.0, 1.0, vec![1])
            .dtype(spaces::DType::Uint8)
            .build()
            .unwrap();
        let boxed = |data: Vec<u8>, shape: Vec<i64>| {
            spaces::SpaceValue::Box(
                spaces::Tensor::from_vec(data, shape, spaces::DType::Uint8).unwrap(),
            )
        };

        // Matching kind/shape/dtype passes (one lane and many).
        assert!(check_actions_conform(&space, &[boxed(vec![0], vec![1])]).is_ok());
        assert!(
            check_actions_conform(&space, &[boxed(vec![0], vec![1]), boxed(vec![1], vec![1])])
                .is_ok()
        );

        // A Discrete value for a Box space is a structural mismatch.
        assert!(check_actions_conform(&space, &[spaces::SpaceValue::Discrete(0)]).is_err());
        // A wrong-shape Box would otherwise mis-encode -> rejected.
        assert!(check_actions_conform(&space, &[boxed(vec![0, 1], vec![2])]).is_err());
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
