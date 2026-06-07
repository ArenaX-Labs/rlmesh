use rlmesh_proto::model::v1::{PredictContext, PredictSlot};

use super::validation::{decode_error, validate_predict_route};

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
    let route = route_context();

    validate_predict_route(&route).unwrap();
}

#[test]
fn model_predict_request_requires_route_context() {
    let err = decode_error("predict missing route context");

    assert!(err.to_string().contains("missing route context"));
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
