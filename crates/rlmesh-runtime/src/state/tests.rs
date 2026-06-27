use crate::spec::{RuntimeLimits, RuntimeSessionSpec};

use super::*;

#[test]
fn route_state_tracks_slot_episode_records() {
    let mut spec = test_session_spec();
    spec.num_envs = 2;
    let mut state = RouteState::new(&spec);

    let started = state.start_episodes(vec!["env-ep-a".to_string(), "env-ep-b".to_string()], false);

    assert_eq!(started.len(), 2);
    assert_eq!(state.snapshot().episode_ids, ["env-ep-a", "env-ep-b"]);
    assert_eq!(
        state.snapshot().episode_record_ids,
        ["ep-000001", "ep-000002"]
    );
    assert_eq!(state.episode_ids(), ["env-ep-a", "env-ep-b"]);
}

#[test]
fn route_state_generates_monotonic_request_ids() {
    let state_spec = test_session_spec();
    let mut state = RouteState::new(&state_spec);

    assert_eq!(state.next_request_id("reset"), "test-env:reset:000001");
    assert_eq!(state.next_request_id("step"), "test-env:step:000002");
}

#[test]
fn request_ids_do_not_collide_across_sibling_envs() {
    let mut spec_a = test_session_spec();
    spec_a.env_id = "env-a".to_string();
    let mut spec_b = test_session_spec();
    spec_b.env_id = "env-b".to_string();
    assert_eq!(spec_a.session_id, spec_b.session_id);

    let mut state_a = RouteState::new(&spec_a);
    let mut state_b = RouteState::new(&spec_b);

    let id_a = state_a.next_request_id("reset");
    let id_b = state_b.next_request_id("reset");

    assert_ne!(
        id_a, id_b,
        "sibling envs in the same session must not share request IDs"
    );
    assert_eq!(id_a, "env-a:reset:000001");
    assert_eq!(id_b, "env-b:reset:000001");
}

fn test_session_spec() -> RuntimeSessionSpec {
    RuntimeSessionSpec {
        session_id: "session".to_string(),
        env_id: "test-env".to_string(),
        env_component_id: "env".to_string(),
        model_component_id: "model".to_string(),
        workflow_edition: rlmesh_proto::CURRENT_WORKFLOW_EDITION.to_string(),
        env_contract: Default::default(),
        num_envs: 1,
        base_seed: None,
        max_episodes: Some(1),
        close_env_on_end: true,
        limits: RuntimeLimits::default(),
    }
}
