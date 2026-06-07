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
    assert_eq!(state.slots()[0].env_index, 0);
    assert_eq!(state.slots()[1].env_index, 1);
    assert_eq!(state.slots()[0].episode_id, "env-ep-a");
    assert_eq!(state.slots()[1].episode_id, "env-ep-b");
}

#[test]
fn route_state_generates_monotonic_request_ids() {
    let state_spec = test_session_spec();
    let mut state = RouteState::new(&state_spec);

    assert_eq!(state.next_request_id("reset"), "session:reset:000001");
    assert_eq!(state.next_request_id("step"), "session:step:000002");
}

fn test_session_spec() -> RuntimeSessionSpec {
    RuntimeSessionSpec {
        session_id: "session".to_string(),
        route_id: "route".to_string(),
        env_component_id: "env".to_string(),
        model_component_id: "model".to_string(),
        env_id: "test-env".to_string(),
        env_contract: Default::default(),
        num_envs: 1,
        max_episodes: Some(1),
        close_env_on_end: true,
        limits: RuntimeLimits::default(),
    }
}
