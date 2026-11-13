//! The basic route loop: episodes, ids, chunk replay, predict timeouts,
//! seeds, RPC error sources, cancellation reasons and observation events.

use super::*;

#[tokio::test]
async fn driver_runs_one_episode_and_closes_terminal_route() {
    let env = TestEnv::default();
    let model = TestModel::default();
    let hooks = Arc::new(RecordingHooks::default());

    let report = new_driver(
        one_episode_spec(),
        env.clone(),
        model.clone(),
        hooks.clone(),
    )
    .run()
    .await
    .unwrap();

    assert_eq!(report.total_steps, 1);
    assert_eq!(report.total_episodes, 1);
    // A driver-owned reset completes the episode right at its terminal step:
    // no observation follows under its id, and the completion is its last event.
    let completed = hooks.completed_ids();
    assert_eq!(completed.len(), 1);
    assert_eq!(
        hooks.events_for(&completed[0]),
        ["started", "observation", "step", "completed"]
    );
    // Telemetry flows end-to-end: the session snapshot carries per-op rows.
    let predict_rpc = report
        .telemetry
        .rows
        .iter()
        .find(|row| row.source.op == "model.predict" && row.metric.name == "rpc.total")
        .expect("model.predict rpc.total recorded");
    assert_eq!(predict_rpc.count, 1);
    // The request/response byte sizes are recorded alongside latency.
    for metric in ["request.bytes", "response.bytes"] {
        assert!(
            report
                .telemetry
                .rows
                .iter()
                .any(|row| row.source.op == "model.predict" && row.metric.name == metric),
            "model.predict {metric} recorded",
        );
    }
    // Each completed loop iteration records its wall clock, and it bounds the
    // per-op time it contains.
    let round = report
        .telemetry
        .rows
        .iter()
        .find(|row| row.source.op == "runner.round" && row.metric.name == "rpc.total")
        .expect("runner.round rpc.total recorded");
    assert_eq!(round.count, 1);
    assert!(
        round.avg >= predict_rpc.avg,
        "round wall ({}) must cover the predict RPC it contains ({})",
        round.avg,
        predict_rpc.avg,
    );
    assert!(env.closed.load(Ordering::SeqCst));
    assert!(model.closed.load(Ordering::SeqCst));
    assert_eq!(hooks.actions.load(Ordering::SeqCst), 1);
    assert_eq!(
        *hooks.step_infos.lock().unwrap(),
        vec![Some(info_map("phase", "step"))]
    );
    // The terminal step says so itself, on the step event.
    assert_eq!(
        *hooks.step_flags.lock().unwrap(),
        vec![(vec![true], vec![false], vec![false])]
    );
    assert_eq!(
        hooks
            .emitted_observations
            .lock()
            .unwrap()
            .iter()
            .map(|emitted| emitted.infos.clone())
            .collect::<Vec<_>>(),
        vec![Some(info_map("phase", "reset"))]
    );
    assert_eq!(hooks.ended.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn runtime_mints_uuidv7_episode_id_and_emits_reset_adapter_on_episode_end() {
    // R1 + R2: the runtime is the sole id authority — it mints a fresh UUIDv7
    // episode id per reset (the env merely adopts + echoes it) — and fires an
    // explicit ResetAdapter to the model when the episode ends, so the model
    // evicts that episode's state without any position-diffing.
    let env = TestEnv::default();
    let model = TestModel::default();
    let report = new_driver(
        one_episode_spec(),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();
    assert_eq!(report.total_episodes, 1);

    let resets = model
        .reset_adapters
        .lock()
        .expect("reset_adapter recorder lock poisoned");
    assert_eq!(
        resets.len(),
        1,
        "ResetAdapter fires exactly once, on the single episode's end"
    );
    assert_eq!(resets[0].len(), 1, "one ended episode id was evicted");
    let id = &resets[0][0];
    // A runtime-minted UUIDv7 in hyphenated string form (8-4-4-4-12), not the
    // old env-minted placeholder. Version nibble is 7.
    assert_eq!(id.len(), 36, "episode id must be a UUID string, got {id:?}");
    assert_eq!(id.matches('-').count(), 4, "UUID has four hyphens: {id:?}");
    assert_eq!(
        id.as_bytes()[14],
        b'7',
        "episode id must be a UUIDv7 (version nibble 7): {id:?}"
    );
}

#[tokio::test]
async fn runtime_replays_buffered_chunk_frames_and_skips_predict() {
    // Runtime-owned action chunking: a model returns its ordered frames in
    // `actions` (frame 0 plus `replay_frames` future frames); the driver applies
    // one per step WITHOUT re-calling the model and re-plans only when the buffer
    // drains. With chunk size 3 (2 replay frames) over a 6-step episode, predict
    // fires on steps 1 and 4 only, while env.step and the observation ledger fire
    // every step — the invariant the managed perturbation hooks depend on.
    let env = TestEnv {
        terminal_after: 6,
        ..Default::default()
    };
    let model = TestModel {
        replay_frames: 2,
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let spec = one_episode_spec();

    let report = new_driver(spec, env.clone(), model.clone(), hooks.clone())
        .run()
        .await
        .unwrap();

    assert_eq!(report.total_steps, 6, "env advanced a full 6-step episode");
    assert_eq!(report.total_episodes, 1);
    // Chunk size 3 => predict on steps 1 and 4 only (4 of 6 steps are replays).
    assert_eq!(
        model.predicts.load(Ordering::SeqCst),
        2,
        "model.predict fires once per chunk (every 3 steps), not every step",
    );
    assert_eq!(
        env.step_count.load(Ordering::SeqCst),
        6,
        "env stepped every step, replay or not",
    );
    // The observation ledger stays intact: one observation emitted per step input
    // (reset obs + 5 step obs), even on the 4 replay steps the model never saw.
    assert_eq!(
        hooks.emitted_observations.lock().unwrap().len(),
        6,
        "observation emitted every step, including replay steps",
    );
    // action_received fires every step (action perturbations apply to replays too).
    assert_eq!(hooks.actions.load(Ordering::SeqCst), 6);
}

#[tokio::test]
async fn model_predict_timeout_fails_session() {
    let env = TestEnv::default();
    let model = TestModel {
        predict_delay: Some(Duration::from_millis(50)),
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = one_episode_spec();
    spec.limits.model_predict_timeout = Duration::from_millis(5);

    let error = new_driver(spec, env, model, hooks.clone())
        .run()
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        RuntimeError::OperationTimeout {
            operation: "model.predict",
            ..
        }
    ));
    assert_eq!(hooks.failed.load(Ordering::SeqCst), 1);
    // The durable session telemetry is delivered even when the run errors out:
    // the epilogue pushes the final Session snapshot on every exit path. With the
    // 1s default window the ticker never fired in this ~5ms run, so EXACTLY one
    // Session push arrives — from the failure epilogue, not the ticker — and it
    // carries the reset rows recorded before the predict timed out.
    assert_eq!(
        hooks.telemetry_sessions.load(Ordering::SeqCst),
        1,
        "exactly one final session push on an aborted run (epilogue, not ticker)",
    );
    assert!(
        hooks.telemetry_session_rows.load(Ordering::SeqCst) >= 1,
        "final session snapshot must carry the pre-timeout reset telemetry",
    );
}

#[tokio::test]
async fn zero_telemetry_window_disables_streaming_but_still_delivers_final() {
    let env = TestEnv {
        terminal_after: 1,
        ..Default::default()
    };
    let model = TestModel::default();
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = one_episode_spec();
    spec.limits.telemetry_window = Duration::ZERO;

    let report = new_driver(spec, env, model, hooks.clone())
        .run()
        .await
        .unwrap();

    // A zero window disables the background ticker (no live Window stream)...
    assert_eq!(hooks.telemetry_windows.load(Ordering::SeqCst), 0);
    // ...but the final durable Session push still fires exactly once at session
    // end, carrying real rows...
    assert_eq!(hooks.telemetry_sessions.load(Ordering::SeqCst), 1);
    assert!(hooks.telemetry_session_rows.load(Ordering::SeqCst) >= 1);
    // ...and the pull report still carries the per-op session total (pinned to
    // model.predict so the mandatory env.reset row alone cannot satisfy it).
    assert!(
        report
            .telemetry
            .rows
            .iter()
            .any(|row| row.source.op == "model.predict" && row.metric.name == "rpc.total"),
    );
}

#[tokio::test]
async fn driver_continues_after_non_terminal_step() {
    let env = TestEnv {
        terminal_after: 2,
        ..Default::default()
    };
    let model = TestModel::default();

    let report = new_driver(
        one_episode_spec(),
        env,
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    assert_eq!(report.total_steps, 2);
    assert_eq!(report.total_episodes, 1);
    assert_eq!(model.predicts.load(Ordering::SeqCst), 2);
}

/// The reset seeds a route threads, one inner list per reset.
async fn threaded_seeds(spec: RuntimeSessionSpec, hooks: Arc<RecordingHooks>) -> Vec<Vec<i64>> {
    let env = TestEnv::default();
    new_driver(spec, env.clone(), TestModel::default(), hooks)
        .run()
        .await
        .expect("the seeded route completes");
    env.reset_seeds
        .lock()
        .expect("reset seed recorder lock poisoned")
        .clone()
}

#[tokio::test]
async fn base_seed_episodes_are_paired_across_models_and_shards() {
    let mut spec = one_episode_spec();
    spec.base_seed = Some(1234);
    spec.max_episodes = Some(2);

    let hooks = Arc::new(RecordingHooks::default());
    let seeds = threaded_seeds(spec.clone(), hooks.clone()).await;
    let derived: Vec<Vec<i64>> = (0..2)
        .map(|slot| {
            vec![rlmesh::binding::runtime::episode_seed(
                1234,
                "TestEnv-v0",
                slot,
            )]
        })
        .collect();
    assert_eq!(seeds, derived);
    // Both episode events carry the seed the episode was reset with.
    let seeded: Vec<Option<i64>> = seeds.iter().map(|seeds| Some(seeds[0])).collect();
    assert_eq!(*hooks.started_seeds.lock().unwrap(), seeded);
    assert_eq!(*hooks.completed_seeds.lock().unwrap(), seeded);

    // Another model in another session draws the same episodes.
    let other_model = RuntimeSessionSpec {
        session_id: "session-2".to_string(),
        model_component_id: "model-2".to_string(),
        ..spec.clone()
    };
    assert_eq!(threaded_seeds(other_model, Arc::default()).await, seeds);

    // Two one-episode shards walk the same slots as the two-episode route.
    let mut sharded = Vec::new();
    for base in 0..2 {
        let shard = RuntimeSessionSpec {
            max_episodes: Some(1),
            trial_index_base: Some(base),
            ..spec.clone()
        };
        sharded.extend(threaded_seeds(shard, Arc::default()).await);
    }
    assert_eq!(sharded, seeds);
}

#[test]
fn env_rpc_preserves_recoverability_and_source() {
    let recoverable =
        RuntimeError::env_rpc_with_recoverability("env.step", 7, true, FakeTransportError);
    assert!(recoverable.is_recoverable());

    let fatal = RuntimeError::env_rpc("env.reset", 0, FakeTransportError);
    assert!(!fatal.is_recoverable());

    // The structured source is preserved and downcastable, not flattened to a
    // string.
    use std::error::Error;
    let source = recoverable.source().expect("EnvRpc carries a source");
    assert!(source.downcast_ref::<FakeTransportError>().is_some());
}

#[test]
fn model_rpc_preserves_source() {
    let error = RuntimeError::model_rpc("local-model", FakeTransportError);
    assert!(!error.is_recoverable());

    let recoverable =
        RuntimeError::model_rpc_with_recoverability("endpoint-a", true, FakeTransportError);
    assert!(recoverable.is_recoverable());

    use std::error::Error;
    assert!(
        error
            .source()
            .and_then(|source| source.downcast_ref::<FakeTransportError>())
            .is_some()
    );
}

#[tokio::test]
async fn cancellation_reason_is_threaded_from_caller() {
    use tokio_util::sync::CancellationToken;

    let env = TestEnv::default();
    let model = TestModel::default();
    let cancellation = CancellationToken::new();
    // Pre-cancel so the first cancellation check trips during the session.
    cancellation.cancel();

    let error = new_driver(
        one_episode_spec(),
        env,
        model,
        Arc::new(RecordingHooks::default()),
    )
    .run_with_cancellation_reason(cancellation, "operator requested shutdown")
    .await
    .unwrap_err();

    let RuntimeError::RouteCancelled { reason, .. } = error else {
        panic!("expected RouteCancelled, got {error:?}");
    };
    assert_eq!(reason, "operator requested shutdown");
}

#[tokio::test]
async fn default_cancellation_reason_does_not_claim_sibling_failure() {
    use tokio_util::sync::CancellationToken;

    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let error = new_driver(
        one_episode_spec(),
        TestEnv::default(),
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run_with_cancellation(cancellation)
    .await
    .unwrap_err();

    let RuntimeError::RouteCancelled { reason, .. } = error else {
        panic!("expected RouteCancelled, got {error:?}");
    };
    assert!(
        !reason.contains("sibling"),
        "default reason should not fabricate a sibling-route failure: {reason}"
    );
}

#[tokio::test]
async fn observation_emitted_carries_what_the_model_receives() {
    const MARKER: u8 = 0xAB;
    let env = TestEnv {
        terminal_after: 2,
        ..Default::default()
    };
    let model = TestModel::default();
    let hooks = Arc::new(RecordingHooks {
        observation_marker: Some(MARKER),
        ..Default::default()
    });

    new_driver(one_episode_spec(), env, model.clone(), hooks.clone())
        .run()
        .await
        .unwrap();

    let emitted = hooks
        .emitted_observations
        .lock()
        .expect("emitted observation recorder lock poisoned")
        .clone();

    // Model saw: initial reset observation + the step observation for the
    // non-terminal step. (The terminal step's observation is never sent.)
    assert_eq!(emitted.len(), 2, "emitted: {emitted:?}");
    // Each carries the transformed payload and, beside it, the env's.
    for emitted in &emitted {
        let raw = emitted.raw_observation.as_ref().expect("a transform ran");
        assert_eq!(emitted.observation, [&[MARKER][..], raw].concat());
    }
    // The model received exactly the transformed observations.
    let seen = model
        .seen_observations
        .lock()
        .expect("model observation recorder lock poisoned")
        .clone();
    assert_eq!(
        seen,
        emitted
            .iter()
            .map(|emitted| emitted.observation.clone())
            .collect::<Vec<_>>()
    );
}
