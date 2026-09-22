use crate::spec::{RuntimeLimits, RuntimeSessionSpec};

use super::*;

#[test]
fn route_state_tracks_slot_episode_records() {
    let mut spec = test_session_spec();
    spec.num_envs = 2;
    let mut state = RouteState::new(&spec);

    let started = state.start_episodes_at(
        &[0, 1],
        vec!["env-ep-a".to_string(), "env-ep-b".to_string()],
        false,
        &[0, 1],
    );

    assert_eq!(started.len(), 2);
    assert_eq!(
        state.snapshot_at(&[0, 1]).episode_ids,
        ["env-ep-a", "env-ep-b"]
    );
    assert_eq!(
        state.snapshot_at(&[0, 1]).episode_record_ids,
        ["ep-000001", "ep-000002"]
    );
    assert_eq!(state.episode_ids_at(&[0, 1]), ["env-ep-a", "env-ep-b"]);
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

#[test]
fn predict_request_includes_seed_metadata_aligned_to_episode_ids() {
    let mut spec = test_session_spec();
    spec.num_envs = 2;
    let mut state = RouteState::new(&spec);
    let episode_ids = vec!["env-ep-a".to_string(), "env-ep-b".to_string()];

    state.start_episodes_at(&[0, 1], episode_ids.clone(), false, &[0, 1]);
    state.note_episode_seeds(&episode_ids, &[7]);

    let request = state.predict_request_at(&[0, 1], None, RequestPhase::ResetObservation);

    assert_eq!(request.episode_info.len(), 2);
    assert_eq!(request.episode_info[0].episode_id, "env-ep-a");
    assert_eq!(request.episode_info[0].seed, Some(7));
    assert_eq!(request.episode_info[1].episode_id, "env-ep-b");
    assert_eq!(request.episode_info[1].seed, None);
}

#[test]
fn episode_seed_survives_completion_until_the_lane_rolls() {
    let mut state = RouteState::new(&test_session_spec());
    state.start_episodes_at(&[0], vec!["ep-a".to_string()], false, &[0]);
    state.note_episode_seeds(&["ep-a".to_string()], &[7]);

    // Completion emit reads the seed non-destructively: the same iteration
    // still builds the terminal predict for this episode id.
    state.complete_episode("ep-a");
    let request = state.predict_request_at(&[0], None, RequestPhase::StepObservation);
    assert_eq!(request.episode_info[0].seed, Some(7));

    // The lane rolling to a fresh id is what retires the seed.
    state.observe_episode_ids_at(&[0], vec!["ep-b".to_string()], &[Some(1)]);
    assert_eq!(state.seed_for_episode("ep-a"), None);
}

#[test]
fn trial_cursor_walks_forward_across_whole_and_partial_resets() {
    let mut spec = test_session_spec();
    spec.num_envs = 2;
    let mut state = RouteState::new(&spec);

    // A whole-vector reset opens the first window of num_envs ordinals, aligned
    // to the lanes it restarts.
    assert_eq!(state.claim_trial_indices(100, 2), vec![100, 101]);
    // A partial reset claims one ordinal per restarted lane from the same
    // cursor, so no two episodes on the route share an ordinal.
    assert_eq!(state.claim_trial_indices(100, 1), vec![102]);
    // The cursor only ever walks forward, so the ordinals a route hands out are
    // distinct and contiguous from its base (the window rule's other half -- a
    // window is the max_episodes budget -- is pinned in tests/driver.rs).
    assert_eq!(state.claim_trial_indices(100, 2), vec![103, 104]);
}

#[test]
fn episode_trial_survives_completion_until_the_lane_rolls() {
    let mut state = RouteState::new(&test_session_spec());
    state.start_episodes_at(&[0], vec!["ep-a".to_string()], false, &[0]);
    state.note_episode_trials(&["ep-a".to_string()], &[4]);

    state.complete_episode("ep-a");
    assert_eq!(state.trial_for_episode("ep-a"), Some(4));

    state.observe_episode_ids_at(&[0], vec!["ep-b".to_string()], &[Some(1)]);
    assert_eq!(state.trial_for_episode("ep-a"), None);
}

fn test_session_spec() -> RuntimeSessionSpec {
    RuntimeSessionSpec {
        session_id: "session".to_string(),
        env_id: "test-env".to_string(),
        env_component_id: "env".to_string(),
        model_component_id: "model".to_string(),
        workflow_edition: rlmesh_proto::Edition::E2026_06,
        env_contract: Default::default(),
        num_envs: 1,
        episode_seeds: Vec::new(),
        base_seed: None,
        max_episodes: Some(1),
        trial_index_base: None,
        max_episode_steps: None,
        max_episode_seconds: None,
        close_env_on_end: true,
        subset_step: false,
        limits: RuntimeLimits::default(),
        env_ceiling: None,
        model_ceiling: None,
    }
}
