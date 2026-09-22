//! Behavioral fingerprint for workflow edition 2026.06: these lifecycle
//! assertions are the edition contract (episode accounting, per-lane autoreset,
//! request/response ordering). Changing observable behavior here changes the
//! edition; see docs/editions/2026.06.md. The fakes they drive live in
//! `common/mod.rs`, shared with the byte-exact golden records in
//! `edition_2026_06.rs`.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::{FutureExt, StreamExt};
use rlmesh_proto::env::v1::{
    EpisodeMetadata, ResetRequest, ResetResponse, StepRequest, StepResponse,
};
use rlmesh_proto::spaces::v1::{MetaValue, meta_value};
use rlmesh_runtime::{
    EndpointPhases, RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep, RuntimeError,
    RuntimeSessionSpec,
};

mod common;

use common::*;

#[tokio::test]
async fn driver_runs_one_episode_and_closes_terminal_route() {
    let env = TestEnv::default();
    let model = TestModel::default();
    let hooks = Arc::new(RecordingHooks::default());

    let report = RuntimeDriver::new(
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
    let report = RuntimeDriver::new(
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

    let report = RuntimeDriver::new(spec, env.clone(), model.clone(), hooks.clone())
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

    let error = RuntimeDriver::new(spec, env, model, hooks.clone())
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

    let report = RuntimeDriver::new(spec, env, model, hooks.clone())
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

    let report = RuntimeDriver::new(
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

// Real-time (not start_paused, which needs tokio's test-util feature): the
// 100ms stall vs 10ms cadence is a wide enough margin to be robust.
#[tokio::test]
async fn telemetry_ticks_on_wall_clock_during_a_stalled_step() {
    let env = TestEnv {
        terminal_after: 1,
        ..Default::default()
    };
    let model = TestModel {
        predict_delay: Some(Duration::from_millis(100)),
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());

    let mut spec = one_episode_spec();
    spec.limits.telemetry_window = Duration::from_millis(10);

    let report = RuntimeDriver::new(spec, env, model, hooks.clone())
        .run()
        .await
        .unwrap();

    // The lone predict stalls 100ms; with a 10ms wall-clock cadence the
    // background ticker emits a (non-empty) Window snapshot *during* the stall —
    // the old step-gated path could not, being parked inside the await. The
    // ticker streams only Window deltas; the cumulative Session total is pushed
    // once by the epilogue, never per tick.
    assert!(
        hooks.telemetry_windows.load(Ordering::SeqCst) >= 1,
        "expected a wall-clock window snapshot during the stalled predict",
    );
    assert_eq!(
        hooks.telemetry_sessions.load(Ordering::SeqCst),
        1,
        "exactly one final Session push (epilogue) — the ticker emits no sessions",
    );
    assert!(hooks.telemetry_session_rows.load(Ordering::SeqCst) >= 1);
    // The final pull still carries the per-op session total.
    assert!(
        report
            .telemetry
            .rows
            .iter()
            .any(|row| row.source.op == "model.predict" && row.metric.name == "rpc.total"),
    );
}

#[tokio::test]
async fn fatal_transform_hook_failure_closes_route() {
    let env = TestEnv::default();
    let model = TestModel::default();
    let hooks = Arc::new(RecordingHooks {
        fail_action_transform: true,
        ..Default::default()
    });

    let error = RuntimeDriver::new(one_episode_spec(), env.clone(), model.clone(), hooks)
        .run()
        .await
        .unwrap_err();

    assert!(matches!(error, RuntimeError::Hook(_)));
    assert!(env.closed.load(Ordering::SeqCst));
    assert!(model.closed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn driver_threads_deterministic_reset_seeds() {
    let mut spec = one_episode_spec();
    spec.base_seed = Some(1234);
    spec.max_episodes = Some(2);

    let first_env = TestEnv::default();
    let first_model = TestModel::default();
    let first_hooks = Arc::new(RecordingHooks::default());
    RuntimeDriver::new(
        spec.clone(),
        first_env.clone(),
        first_model,
        first_hooks.clone(),
    )
    .run()
    .await
    .unwrap();

    let second_env = TestEnv::default();
    let second_model = TestModel::default();
    RuntimeDriver::new(
        spec,
        second_env.clone(),
        second_model,
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    let first_seeds = first_env
        .reset_seeds
        .lock()
        .expect("reset seed recorder lock poisoned")
        .clone();
    let second_seeds = second_env
        .reset_seeds
        .lock()
        .expect("reset seed recorder lock poisoned")
        .clone();

    assert_eq!(first_seeds, second_seeds);
    assert_eq!(first_seeds.len(), 2);
    assert_eq!(first_seeds[0].len(), 1);
    assert_eq!(first_seeds[1].len(), 1);
    assert_ne!(first_seeds[0], first_seeds[1]);
    // Both episode events carry the seed the episode was reset with.
    let seeded: Vec<Option<i64>> = first_seeds.iter().map(|seeds| Some(seeds[0])).collect();
    assert_eq!(*first_hooks.started_seeds.lock().unwrap(), seeded);
    assert_eq!(*first_hooks.completed_seeds.lock().unwrap(), seeded);
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

    let error = RuntimeDriver::new(
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

    let error = RuntimeDriver::new(
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
async fn observation_emitted_always_carries_transformed_payload() {
    let env = TestEnv {
        terminal_after: 2,
        ..Default::default()
    };
    let model = TestModel::default();
    const MARKER: u8 = 0xAB;
    let hooks = Arc::new(RecordingHooks {
        observation_marker: Some(MARKER),
        ..Default::default()
    });

    RuntimeDriver::new(one_episode_spec(), env, model.clone(), hooks.clone())
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
    // Every emitted observation must be the transformed payload the model
    // actually received, i.e. carry the marker byte.
    for emitted in &emitted {
        let (bytes, raw) = (&emitted.observation, &emitted.raw_observation);
        assert_eq!(
            bytes.first().copied(),
            Some(MARKER),
            "observation_emitted exposed pre-transform bytes: {bytes:?}"
        );
        assert_ne!(bytes, raw);
        assert_eq!(
            &bytes[1..],
            raw.as_slice(),
            "raw_observation must be the pre-transform bytes"
        );
    }
    // The model received exactly these transformed observations.
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

#[tokio::test]
async fn shutdown_enforces_service_close_timeout_on_hung_model() {
    let env = TestEnv::default();
    let model = TestModel {
        release_adapter_hangs: true,
        ..Default::default()
    };
    let mut spec = one_episode_spec();
    spec.limits.service_close_timeout = Duration::from_millis(50);

    // Without driver-side timeout enforcement, run() would hang forever in
    // shutdown on the hung close_route. The driver must give up after
    // service_close_timeout and complete the session.
    let report = tokio::time::timeout(
        Duration::from_secs(5),
        RuntimeDriver::new(
            spec,
            env,
            model.clone(),
            Arc::new(RecordingHooks::default()),
        )
        .run(),
    )
    .await
    .expect("driver hung in shutdown despite service_close_timeout")
    .expect("session should complete");

    assert_eq!(report.total_episodes, 1);
    // The hung close never set `closed`, confirming the driver abandoned it.
    assert!(!model.closed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn hung_eviction_yields_to_cancellation() {
    use tokio_util::sync::CancellationToken;

    // The first episode's eviction goes out at the top of the next loop
    // iteration (the second episode's reset is in flight) and never returns.
    // Cancelling the route must interrupt that await: the run ends as
    // cancelled and still releases the model, instead of sitting on the
    // eviction until the model answers.
    let env = TestEnv::default();
    let model = TestModel {
        reset_adapter_hangs: true,
        ..Default::default()
    };
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(2);
    let cancellation = CancellationToken::new();
    // Bounded so a driver that never evicts at all fails this test instead of
    // spinning here forever.
    let cancel_once_evicting = tokio::time::timeout(Duration::from_secs(5), async {
        while model
            .reset_adapters
            .lock()
            .expect("reset_adapter recorder lock poisoned")
            .is_empty()
        {
            tokio::task::yield_now().await;
        }
        cancellation.cancel();
    });
    let (result, _) = tokio::join!(
        tokio::time::timeout(
            Duration::from_secs(5),
            RuntimeDriver::new(
                spec,
                env,
                model.clone(),
                Arc::new(RecordingHooks::default()),
            )
            .run_with_cancellation(cancellation.clone()),
        ),
        cancel_once_evicting,
    );
    let error = result
        .expect("cancellation must interrupt the hung eviction")
        .unwrap_err();

    assert!(
        matches!(error, RuntimeError::RouteCancelled { .. }),
        "expected RouteCancelled, got {error:?}"
    );
    // The abandoned eviction is the only one attempted (the model never
    // predicted on the second episode, so teardown has nothing to end), and
    // the release still went out after it.
    assert_eq!(
        model
            .reset_adapters
            .lock()
            .expect("reset_adapter recorder lock poisoned")
            .len(),
        1
    );
    assert!(model.closed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn failed_route_ends_its_live_episode_before_release() {
    // Episode 1 runs two steps and completes; episode 2's first predict fails
    // the route. The model has predicted on episode 2 and must hear its end
    // before the release (a context-aware model keeps per-episode state until
    // then), and episode 1, already evicted at completion, is not ended twice.
    let env = TestEnv {
        terminal_after: 2,
        ..Default::default()
    };
    let model = TestModel {
        fail_predict_at: Some(3),
        ..Default::default()
    };
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(2);

    let error = RuntimeDriver::new(
        spec,
        env,
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap_err();
    assert!(
        matches!(error, RuntimeError::ModelRpc { .. }),
        "expected the failed predict, got {error:?}"
    );

    let lifecycle = model
        .lifecycle
        .lock()
        .expect("lifecycle lock poisoned")
        .clone();
    let first = lifecycle[0].1.clone();
    let second = lifecycle[3].1.clone();
    assert_ne!(first, second);
    assert_eq!(
        lifecycle,
        [
            ("predict", first.clone()),
            ("predict", first.clone()),
            ("evict", first),
            ("predict", second.clone()),
            ("evict", second),
        ]
    );
    assert!(model.closed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn peer_reported_phases_land_in_the_session_snapshot() {
    let env = TestEnv {
        endpoint_total_ns: Some(9_000_000),
        phases: EndpointPhases {
            decode_ns: 1_000_000,
            user_ns: 6_000_000,
            encode_ns: 2_000_000,
            ..EndpointPhases::default()
        },
        ..TestEnv::default()
    };
    let model = TestModel {
        endpoint_total_ns: Some(5_000_000),
        phases: EndpointPhases {
            decode_ns: 1_000_000,
            user_ns: 3_000_000,
            encode_ns: 1_000_000,
            queue_ns: 4_000_000,
            in_flight: 7,
            adapter_ns: 2_000_000,
            held_episodes: Some(5),
            held_state_bytes: Some(6_000_000),
            lane_skew_ns: None,
        },
        ..TestModel::default()
    };

    let report = RuntimeDriver::new(
        one_episode_spec(),
        env,
        model,
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    let avg = |op: &str, metric: &str| {
        report
            .telemetry
            .rows
            .iter()
            .find(|row| row.source.op == op && row.metric.name == metric)
            .unwrap_or_else(|| panic!("{op} {metric} recorded"))
            .avg
    };

    // Durations are reported in ms, so the split reads back as the peer stamped it.
    assert_eq!(avg("env.step", "endpoint.decode"), 1.0);
    assert_eq!(avg("env.step", "endpoint.user"), 6.0);
    assert_eq!(avg("env.step", "endpoint.encode"), 2.0);
    assert_eq!(avg("model.predict", "endpoint.total"), 5.0);
    assert_eq!(avg("model.predict", "endpoint.user"), 3.0);
    assert_eq!(avg("model.predict", "endpoint.queue"), 4.0);
    // The adapter's share of the handler's own work: forward = user - adapter.
    assert_eq!(avg("model.predict", "predict.adapter"), 2.0);
    // Held state gauges: an episode count and raw bytes, not durations.
    assert_eq!(avg("model.predict", "held.episodes"), 5.0);
    assert_eq!(avg("model.predict", "held.bytes"), 6_000_000.0);
    // Slot depth is a count, not a duration.
    assert_eq!(avg("model.predict", "predict.in_flight"), 7.0);
}

#[tokio::test]
async fn a_straggler_lane_records_its_skew_over_the_median_lane() {
    // A vector env timing its own lanes stamps one dispersion scalar per step,
    // so a single slow lane among healthy ones is visible without a series per
    // lane. The env is otherwise silent: skew does not depend on a phase split.
    let env = TestEnv {
        phases: EndpointPhases {
            lane_skew_ns: Some(8_000_000),
            ..EndpointPhases::default()
        },
        ..TestEnv::default()
    };

    let report = RuntimeDriver::new(
        one_episode_spec(),
        env,
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    let skew = |op: &str| {
        report
            .telemetry
            .rows
            .iter()
            .find(|row| row.metric.name == "lane.skew" && row.source.op == op)
    };
    assert_eq!(skew("env.step").expect("env.step lane.skew").avg, 8.0);
    // Only the env reports lanes.
    assert!(skew("model.predict").is_none());
}

#[tokio::test]
async fn a_measured_zero_gauge_is_a_sample_not_an_absence() {
    // An even vector has zero skew and an engine that just evicted holds zero
    // episodes: both are real observations the percentiles need, unlike a peer
    // that never measures (which records nothing, see the test below).
    let env = TestEnv {
        phases: EndpointPhases {
            lane_skew_ns: Some(0),
            ..EndpointPhases::default()
        },
        ..TestEnv::default()
    };
    let model = TestModel {
        phases: EndpointPhases {
            held_episodes: Some(0),
            held_state_bytes: Some(0),
            ..EndpointPhases::default()
        },
        ..TestModel::default()
    };
    let report = RuntimeDriver::new(
        one_episode_spec(),
        env,
        model,
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    let row = |op: &str, metric: &str| {
        report
            .telemetry
            .rows
            .iter()
            .find(|row| row.source.op == op && row.metric.name == metric)
            .unwrap_or_else(|| panic!("{op} {metric} recorded"))
    };
    assert_eq!(row("env.step", "lane.skew").avg, 0.0);
    assert!(row("env.step", "lane.skew").count > 0);
    assert_eq!(row("model.predict", "held.episodes").avg, 0.0);
    assert_eq!(row("model.predict", "held.bytes").avg, 0.0);
}

#[tokio::test]
async fn a_peer_that_reports_no_phases_records_only_the_total() {
    // An older peer stamps `endpoint_total_ns` and nothing else: the phase fields
    // arrive absent, decode to zero, and record no rows at all.
    let env = TestEnv {
        endpoint_total_ns: Some(9_000_000),
        ..TestEnv::default()
    };
    let report = RuntimeDriver::new(
        one_episode_spec(),
        env,
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    let recorded = |metric: &str| {
        report
            .telemetry
            .rows
            .iter()
            .any(|row| row.metric.name == metric)
    };
    assert!(recorded("endpoint.total"));
    for metric in [
        "endpoint.decode",
        "endpoint.user",
        "endpoint.encode",
        "endpoint.queue",
        "predict.in_flight",
        "predict.adapter",
        "held.episodes",
        "held.bytes",
        "lane.skew",
    ] {
        assert!(!recorded(metric), "{metric} must stay unrecorded");
    }
}

#[tokio::test]
async fn next_step_vector_env_completes_lanes_independently_without_whole_vector_reset() {
    // The headline regression. With num_envs=4, NEXT_STEP, and variable-length
    // episodes, the driver must never issue a reset after the cold start; the env
    // auto-resets each lane itself. Previously any single lane completing fired a
    // whole-vector reset that cut every other lane's episode short.
    let env = VectorTestEnv::new(vec![2, 3, 2, 4]);
    let model = TestModel::default();
    let report = RuntimeDriver::new(
        vector_spec(4, 8),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    // Exactly one env.reset over the whole run: the initial cold start.
    assert_eq!(
        env.reset_seeds
            .lock()
            .expect("reset seed recorder lock poisoned")
            .len(),
        1,
        "driver must reset only once (cold start), never on a lane completion"
    );
    // Lanes completed episodes independently until the episode budget.
    assert!(
        report.total_episodes >= 8,
        "expected >= 8 episodes across lanes, got {}",
        report.total_episodes
    );
    // One predict per driver step: no stalled lanes.
    assert_eq!(
        model.predicts.load(Ordering::SeqCst) as i64,
        report.total_steps
    );
}

#[tokio::test]
async fn next_step_roll_attributes_reset_infos_to_the_new_episode() {
    // Both lanes complete at step 1 and roll at step 2. The step-2 response is
    // the new episodes' reset observation + infos: those infos belong on the
    // post-roll observation event (the new episodes' first observation), not on
    // the step event still attributed to the old episodes.
    let env = VectorTestEnv::new(vec![1, 1]);
    let hooks = Arc::new(RecordingHooks::default());
    RuntimeDriver::new(vector_spec(2, 4), env, TestModel::default(), hooks.clone())
        .run()
        .await
        .unwrap();

    assert_eq!(
        *hooks.step_infos.lock().unwrap(),
        vec![
            Some(info_map("phase", "step")),
            None,
            Some(info_map("phase", "step")),
        ]
    );
    let emitted = hooks.emitted_observations.lock().unwrap();
    assert_eq!(
        emitted.iter().map(|e| e.infos.clone()).collect::<Vec<_>>(),
        vec![None, None, Some(info_map("phase", "roll"))]
    );
    assert!(
        emitted[2]
            .episode_ids
            .iter()
            .all(|id| !emitted[1].episode_ids.contains(id)),
        "post-roll observation carries the new episode ids"
    );
}

#[tokio::test]
async fn chunking_does_not_break_autoreset_eviction() {
    // Chunking × NEXT_STEP autoreset. Chunk replay skips most predict calls, but
    // env.step + completion detection + ResetAdapter eviction still run every
    // step — so each episode end is evicted exactly once, with a runtime-minted
    // UUIDv7 id, even while the model is mid-chunk and not being re-called.
    let env = VectorTestEnv::new(vec![2, 3]);
    let model = TestModel {
        replay_frames: 4, // chunk of 5; most steps replay without a predict
        ..Default::default()
    };
    let report = RuntimeDriver::new(
        vector_spec(2, 4),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    assert!(report.total_episodes >= 4);
    // Predict was skipped on most steps (chunk replay), proving the eviction
    // below fired independently of the predict cadence.
    assert!(
        (model.predicts.load(Ordering::SeqCst) as i64) < report.total_steps,
        "chunk replay must skip some predict calls"
    );
    // Exactly one ResetAdapter eviction per completed episode, each a UUIDv7.
    let evicted: Vec<String> = model
        .reset_adapters
        .lock()
        .expect("reset_adapter recorder lock poisoned")
        .iter()
        .flatten()
        .cloned()
        .collect();
    // Every episode the model predicted on is evicted exactly once: each
    // completed episode at its end, and the lane the budget cut mid-episode at
    // teardown, so nothing is left behind in the model.
    let predicted: std::collections::HashSet<String> = model
        .lifecycle
        .lock()
        .expect("lifecycle lock poisoned")
        .iter()
        .filter(|(kind, _)| *kind == "predict")
        .flat_map(|(_, ids)| ids.iter().cloned())
        .collect();
    let distinct: std::collections::HashSet<String> = evicted.iter().cloned().collect();
    assert_eq!(
        evicted.len(),
        distinct.len(),
        "an episode evicted twice: {evicted:?}"
    );
    assert_eq!(distinct, predicted, "one eviction per predicted episode");
    assert!(evicted.len() as i64 >= report.total_episodes);
    for id in &evicted {
        assert_eq!(id.len(), 36, "evicted a UUID id, got {id:?}");
        assert_eq!(id.as_bytes()[14], b'7', "UUIDv7 version nibble: {id:?}");
    }
}

#[tokio::test]
async fn next_step_autoreset_never_predicts_on_an_evicted_episode() {
    // NEXT_STEP × lockstep vector: a lane that ends at t is predicted on once
    // more (the terminal observation feeds the autoreset step, whose action
    // the env discards) before its slot rolls at t+1. The model's lifecycle
    // for that id must still be predict* → evict, never evict → predict: an
    // eviction-then-predict would re-seed the ended episode's state and leak
    // the entry. Uneven lane lengths make the rolls land on different steps.
    let env = VectorTestEnv::new(vec![1, 3]);
    let model = TestModel::default();
    let report = RuntimeDriver::new(
        vector_spec(2, 4),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();
    assert!(report.total_episodes >= 4);

    let lifecycle = model
        .lifecycle
        .lock()
        .expect("lifecycle lock poisoned")
        .clone();
    let mut evicted: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (kind, ids) in &lifecycle {
        match *kind {
            "evict" => evicted.extend(ids.iter().cloned()),
            _ => {
                for id in ids {
                    assert!(
                        !evicted.contains(id),
                        "predict on {id} after its eviction; lifecycle: {lifecycle:?}"
                    );
                }
            }
        }
    }
    // Every episode the model predicted on was evicted — the completed ones at
    // their end, the lane the budget cut mid-episode at teardown — so nothing
    // is left behind in the model.
    let predicted: std::collections::HashSet<String> = lifecycle
        .iter()
        .filter(|(kind, _)| *kind == "predict")
        .flat_map(|(_, ids)| ids.iter().cloned())
        .collect();
    assert_eq!(evicted, predicted);
    assert!(evicted.len() as i64 >= report.total_episodes);
}

#[tokio::test]
async fn autoreset_roll_keeps_a_sibling_lane_evictable_at_teardown() {
    // NEXT_STEP × lockstep vector, lanes of 2, 3 and 7 steps, budget 2. The
    // lane-0 roll and the lane-1 completion that spends the budget land on the
    // same step, so lane 2's episode is live at teardown and its last predict
    // was before that roll. It must still be ended on the model: a roll must
    // not clear a sibling lane's model-side episode state.
    let env = VectorTestEnv::new(vec![2, 3, 7]);
    let model = TestModel::default();
    let report = RuntimeDriver::new(
        vector_spec(3, 2),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();
    assert_eq!(report.total_episodes, 2);

    let lifecycle = model
        .lifecycle
        .lock()
        .expect("lifecycle lock poisoned")
        .clone();
    let predicted: std::collections::HashSet<String> = lifecycle
        .iter()
        .filter(|(kind, _)| *kind == "predict")
        .flat_map(|(_, ids)| ids.iter().cloned())
        .filter(|id| !id.is_empty())
        .collect();
    let evicted: Vec<String> = model
        .reset_adapters
        .lock()
        .expect("reset_adapter recorder lock poisoned")
        .iter()
        .flatten()
        .cloned()
        .collect();
    let distinct: std::collections::HashSet<String> = evicted.iter().cloned().collect();
    assert_eq!(
        evicted.len(),
        distinct.len(),
        "an episode evicted twice: {evicted:?}"
    );
    assert_eq!(
        distinct, predicted,
        "every predicted episode ends on the model exactly once; lifecycle: {lifecycle:?}"
    );
}

#[tokio::test]
async fn next_step_episode_completed_is_the_last_hook_event_under_its_id() {
    // NEXT_STEP × lockstep vector, lanes of 1 and 3 steps, budget 4. Lane 0
    // ends at steps 1, 3 and 5; lane 1 at step 3. The step-5 completion hits
    // the budget, so that episode's roll never lands. From a hook's side an
    // ended id is still observed once more (the terminal observation feeding
    // the autoreset step) and stepped once more (the autoreset step itself),
    // both under the old id; `episode_completed` follows those, as the last
    // event the hook sees under that id, mirroring the model's
    // predict* → evict lifecycle. The budget-ended episode completes with no
    // roll behind it. Every episode is completed exactly once and summarized.
    let env = VectorTestEnv::new(vec![1, 3]);
    let hooks = Arc::new(RecordingHooks::default());
    let report = RuntimeDriver::new(vector_spec(2, 4), env, TestModel::default(), hooks.clone())
        .run()
        .await
        .unwrap();
    assert_eq!(report.total_episodes, 4);

    let completed = hooks.completed_ids();
    let unique: std::collections::HashSet<&String> = completed.iter().collect();
    assert_eq!(
        completed.len(),
        4,
        "one completion per episode: {completed:?}"
    );
    assert_eq!(unique.len(), 4, "no id completed twice: {completed:?}");
    assert_eq!(report.episodes.len(), 4, "one summary per episode");
    let events = hooks.events.lock().unwrap().clone();
    for id in &completed {
        let last = events
            .iter()
            .rposition(|(_, ids)| ids.contains(id))
            .expect("completed id was seen");
        assert_eq!(
            events[last].0, "completed",
            "episode_completed must be the last event under {id}; events: {events:?}"
        );
    }
    // A mid-run boundary (lane 0's first episode): its terminal step, the
    // post-end observation, the autoreset step, then the completion.
    assert_eq!(
        hooks.events_for(&completed[0]),
        [
            "started",
            "observation",
            "step",
            "observation",
            "step",
            "completed"
        ],
        "events: {events:?}"
    );
    // The budget-ended episode (lane 0's third, the last completion): the
    // group idles right after its terminal step, so nothing is observed
    // between that step and its completion, and it is the run's last event.
    assert_eq!(
        hooks.events_for(&completed[3]),
        ["started", "observation", "step", "completed"],
        "events: {events:?}"
    );
    assert_eq!(events.last().map(|(kind, _)| *kind), Some("completed"));
}

#[tokio::test]
async fn a_history_route_sees_every_env_step_exactly_once_in_order() {
    // The exactly-once invariant behind observation history: across an
    // episode, the union of every predict's history rows and its own
    // observation is the env-step sequence, in order, each step once, with no
    // gap within or between requests. A prefetch lead moves each re-plan
    // earlier (it fires with `lead` frames still buffered, from the latest
    // observation, which then rides as the request's own instead of a row),
    // so the rows per request shrink and the steps after the episode's last
    // predict have no request left to ride — but nothing is delivered twice.
    // Chunks of 4 over a 9-step episode (steps 0..=8 observed; the terminal
    // observation ends the episode without a predict):
    //   lead 0: predicts at 0, 4, 8 — every replayed step rides the next one.
    //   lead 1: the second predict fires while frame 4 is still buffered, from
    //           step 2 (step 1 rides); the third from step 6 (3, 4, 5 ride);
    //           the chunk from 6 covers 7, 8 and the episode ends.
    //   lead 2: the second fires from step 1 with the first chunk barely
    //           started (no row yet); its chunk plays 5..=8, so the third
    //           fires from step 5 (2, 3, 4 ride) and steps 6..=8 replay.
    let expected: [Vec<LedgerEntry>; 3] = [
        vec![
            (vec![], (0, 1)),
            (vec![(1, 1), (2, 2), (3, 3)], (4, 4)),
            (vec![(5, 5), (6, 6), (7, 7)], (8, 8)),
        ],
        vec![
            (vec![], (0, 1)),
            (vec![(1, 1)], (2, 2)),
            (vec![(3, 3), (4, 4), (5, 5)], (6, 6)),
        ],
        vec![
            (vec![], (0, 1)),
            (vec![], (1, 1)),
            (vec![(2, 2), (3, 3), (4, 4)], (5, 5)),
        ],
    ];
    for (lead, expected) in (0u32..).zip(expected) {
        let env = TestEnv {
            terminal_after: 9,
            ..Default::default()
        };
        let model = TestModel {
            replay_frames: 3, // chunks of 4
            wants_history: true,
            ..Default::default()
        };
        let report = RuntimeDriver::new(
            one_episode_spec(),
            env.clone(),
            model.clone(),
            Arc::new(RecordingHooks::default()),
        )
        .with_prefetch(lead)
        .run()
        .await
        .unwrap();
        assert_eq!(report.total_episodes, 1);
        assert_eq!(report.total_steps, 9);

        // Reset observation carries byte 1 at step 0; step n carries byte n.
        let ledger = model.ledger.lock().unwrap().clone();
        assert_eq!(ledger, expected, "lead {lead}");
        // Each step once, consecutive within and across requests, from step 0:
        // the union is a prefix of the episode, and the whole of it without a
        // lead.
        let union = delivered_steps(&ledger);
        let last = *union.last().unwrap();
        assert_eq!(union, (0..=last).collect::<Vec<_>>(), "lead {lead}");
        assert_eq!(last, [8, 6, 5][lead as usize], "lead {lead}");
        assert_eq!(model.predicts.load(Ordering::SeqCst), 3, "lead {lead}");
    }
}

#[tokio::test]
async fn a_prefetch_across_a_reset_delivers_the_reset_observation_once() {
    // Driver-owned resets, chunks of 3, lead 1, two 3-step episodes. The
    // prefetch fired at each episode's tail is conditioned on step 1 and
    // lands stale, AFTER the next episode's reset observation, which was
    // buffered as a row while it was in flight. The re-arm must promote that
    // row to the fresh request's own — not send it as a row AND as the own.
    let env = TestEnv {
        terminal_after: 3,
        ..Default::default()
    };
    let model = TestModel {
        replay_frames: 2,
        wants_history: true,
        // Slower than the env: the stale result lands after the reset.
        predict_delay: Some(Duration::from_millis(20)),
        ..Default::default()
    };
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(2);
    let report = RuntimeDriver::new(
        spec,
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .with_prefetch(1)
    .run()
    .await
    .unwrap();
    assert_eq!(report.total_episodes, 2);
    assert_eq!(report.total_steps, 6);

    // The group's step counter runs across episodes: the second episode's
    // reset observation is step 3. Its first predict is the re-arm from that
    // observation, carrying no row (the first episode's rows were dropped
    // with its reset; the reset observation itself is the own). Step 2 and
    // step 5 landed after their episode's last predict fired, so no request
    // was left to carry them.
    let episodes = ledger_per_episode(&model);
    assert_eq!(episodes.len(), 2);
    assert_ne!(episodes[0].0, episodes[1].0);
    assert_eq!(episodes[0].1, vec![(vec![], (0, 1)), (vec![], (1, 1))]);
    assert_eq!(episodes[1].1, vec![(vec![], (3, 1)), (vec![], (4, 1))]);
    assert_eq!(delivered_steps(&episodes[0].1), vec![0, 1]);
    assert_eq!(delivered_steps(&episodes[1].1), vec![3, 4]);
}

#[tokio::test]
async fn a_prefetch_across_a_next_step_roll_delivers_the_terminal_step_once() {
    // NEXT_STEP autoreset, one lane, chunks of 3, lead 1, two 3-step episodes.
    // The prefetch fired from step 1 lands stale after the terminal
    // observation (step 3) was buffered as a row behind step 2. The re-arm
    // promotes step 3 to the own of the predict the ended episode still gets
    // (the autoreset step's action), and step 2 rides it as the row it is —
    // so the ended episode's every step reached the model exactly once,
    // under its own id. The roll observation (step 4) starts the next
    // episode's ledger.
    let env = VectorTestEnv::new(vec![3]);
    let model = TestModel {
        replay_frames: 2,
        wants_history: true,
        predict_delay: Some(Duration::from_millis(20)),
        ..Default::default()
    };
    let report = RuntimeDriver::new(
        vector_spec(1, 2),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .with_prefetch(1)
    .run()
    .await
    .unwrap();
    assert_eq!(report.total_episodes, 2);

    let episodes = ledger_per_episode(&model);
    assert_eq!(episodes.len(), 2);
    assert_ne!(episodes[0].0, episodes[1].0);
    assert_eq!(
        episodes[0].1,
        vec![(vec![], (0, 0)), (vec![], (1, 0)), (vec![(2, 0)], (3, 0))]
    );
    assert_eq!(delivered_steps(&episodes[0].1), vec![0, 1, 2, 3]);
    assert_eq!(episodes[1].1, vec![(vec![], (4, 0))]);
}

#[tokio::test]
async fn a_route_without_history_carries_no_rows_and_no_step() {
    let model = TestModel {
        replay_frames: 3,
        ..Default::default()
    };
    RuntimeDriver::new(
        one_episode_spec(),
        TestEnv {
            terminal_after: 9,
            ..Default::default()
        },
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();
    let ledger = model.ledger.lock().unwrap().clone();
    assert!(!ledger.is_empty());
    for (rows, own) in ledger {
        assert!(rows.is_empty());
        assert_eq!(own.0, -1, "no step stamped without history");
    }
}

#[tokio::test]
async fn a_vector_route_losing_a_chunk_mid_episode_trips_the_wire_once() {
    // The residual detector for chunked vector routes. `run_local` and the managed
    // runner refuse the pairing outright, but a host driving RuntimeDriver directly
    // can still reach it — and the loss is silent: the replay buffer is whole-batch,
    // so ONE lane's episode end throws away every lane's remaining frames. Warn
    // once per session, and only when frames were actually discarded.
    let logs = chunked_run_logs(vec![2, 3]).await;
    assert_eq!(
        logs.matches("chunk replay is whole-batch").count(),
        1,
        "the tripwire fires once per session, not once per episode: {logs}"
    );
}

#[tokio::test]
async fn a_single_lane_chunked_route_never_trips_the_wire() {
    // The supported shape: one lane, chunking on, episodes ending mid-chunk. No
    // lane can lose another lane's frames, so nothing is warned about.
    let logs = chunked_run_logs(vec![2]).await;
    assert!(
        !logs.contains("chunk replay is whole-batch"),
        "a single-lane chunked route is the supported shape: {logs}"
    );
}

#[tokio::test]
async fn prefetch_overlaps_predict_with_replay_and_keeps_the_ledger() {
    // Async-inference mode: with chunk size 3 and lead 1, the next chunk's
    // predict fires while the current chunk still has a replay frame left, so
    // the loop never stalls on inference — but every step/observation/action
    // hook still fires exactly as in the synchronous loop.
    let env = TestEnv {
        terminal_after: 6,
        ..Default::default()
    };
    let model = TestModel {
        replay_frames: 2,
        // Slower than an env step: without prefetch this stalls each refill.
        predict_delay: Some(Duration::from_millis(20)),
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let spec = one_episode_spec();

    let report = RuntimeDriver::new(spec, env.clone(), model.clone(), hooks.clone())
        .with_prefetch(1)
        .run()
        .await
        .unwrap();

    assert_eq!(report.total_steps, 6, "env advanced a full 6-step episode");
    assert_eq!(report.total_episodes, 1);
    assert_eq!(
        env.step_count.load(Ordering::SeqCst),
        6,
        "env stepped every step, replay or not",
    );
    assert_eq!(
        hooks.emitted_observations.lock().unwrap().len(),
        6,
        "observation emitted every step, including replay steps",
    );
    assert_eq!(hooks.actions.load(Ordering::SeqCst), 6);
    // Chunked cadence holds: the model plans once per chunk boundary. Prefetch
    // may run one extra speculative predict at the episode tail (fired before
    // the terminal step landed), never more.
    let predicts = model.predicts.load(Ordering::SeqCst);
    assert!(
        (2..=3).contains(&predicts),
        "expected 2 chunk predicts (+ at most 1 speculative tail), got {predicts}",
    );
}

#[tokio::test]
async fn prefetch_discards_the_stale_chunk_across_episode_boundaries() {
    // Two 3-step episodes with chunk size 3, lead 1: the prefetch fired near
    // each episode's tail is conditioned on a pre-terminal observation and must
    // be discarded — the new episode re-plans from its own reset observation,
    // and no replayed action from the old chunk leaks across the boundary.
    let env = TestEnv {
        terminal_after: 3,
        ..Default::default()
    };
    // The stale prediction must land AFTER the reset observation: that is the
    // ordering in which the driver has to re-plan from the saved observation
    // itself, or the group stalls with nothing in flight.
    let model = TestModel {
        replay_frames: 2,
        predict_delay: Some(Duration::from_millis(20)),
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(2);

    let report = RuntimeDriver::new(spec, env.clone(), model.clone(), hooks.clone())
        .with_prefetch(1)
        .run()
        .await
        .unwrap();

    assert_eq!(report.total_episodes, 2);
    assert_eq!(report.total_steps, 6);
    // Each episode re-planned from its own first observation: the fresh predict
    // after each boundary saw a reset observation, not a stale step one. The
    // reset observation payload is the env's reset marker (see TestEnv::reset),
    // recorded as the first seen observation of each episode's first predict.
    let seen = model.seen_observations.lock().unwrap();
    assert!(
        seen.len() >= 2,
        "at least one fresh plan per episode, got {}",
        seen.len()
    );
}

#[tokio::test]
async fn declared_env_receives_the_trial_ordinal_on_every_reset() {
    let env = TestEnv::default();
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = trial_spec(3, &["trial_index"]);
    spec.trial_index_base = Some(100);

    let report = RuntimeDriver::new(spec, env.clone(), TestModel::default(), hooks.clone())
        .run()
        .await
        .unwrap();

    // One reset per episode (driver-owned resets), each carrying the next ordinal.
    let options = env.reset_options.lock().unwrap().clone();
    assert_eq!(
        options.iter().map(recorded_trial).collect::<Vec<_>>(),
        vec![Some(100), Some(101), Some(102)],
    );
    // The window rule: a shard walks exactly its `max_episodes` ordinals, so the
    // next shard's base is base + M.
    assert_eq!(report.total_episodes, 3);
    assert_eq!(
        report
            .episodes
            .iter()
            .map(|episode| episode.trial_index)
            .collect::<Vec<_>>(),
        vec![Some(100), Some(101), Some(102)],
    );
    assert_eq!(
        *hooks.started_trials.lock().unwrap(),
        vec![Some(100), Some(101), Some(102)],
    );
    assert_eq!(
        *hooks.completed_trials.lock().unwrap(),
        vec![Some(100), Some(101), Some(102)],
    );
}

#[tokio::test]
async fn undeclared_env_is_not_sent_the_ordinal_but_the_report_still_carries_it() {
    let env = TestEnv::default();
    let mut spec = trial_spec(2, &["something_else"]);
    spec.trial_index_base = Some(7);

    let report = RuntimeDriver::new(
        spec,
        env.clone(),
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    assert!(
        env.reset_options
            .lock()
            .unwrap()
            .iter()
            .all(|options| options.is_none()),
        "an env that declared no trial_index must not receive one",
    );
    assert_eq!(
        report
            .episodes
            .iter()
            .map(|episode| episode.trial_index)
            .collect::<Vec<_>>(),
        vec![Some(7), Some(8)],
        "the ordinal is minted either way, so a coverage audit can read the sweep",
    );
}

#[tokio::test]
async fn no_trial_base_walks_the_ordinals_from_zero() {
    // The ordinal is on by default: with `trial_index_base` unset a declaring env
    // receives 0, 1, 2 ... and the report / hooks carry the same sweep.
    let env = TestEnv::default();
    let hooks = Arc::new(RecordingHooks::default());
    let spec = trial_spec(3, &["trial_index"]);
    assert_eq!(spec.trial_index_base, None);

    let report = RuntimeDriver::new(spec, env.clone(), TestModel::default(), hooks.clone())
        .run()
        .await
        .unwrap();

    let options = env.reset_options.lock().unwrap().clone();
    assert_eq!(
        options.iter().map(recorded_trial).collect::<Vec<_>>(),
        vec![Some(0), Some(1), Some(2)],
    );
    assert_eq!(
        report
            .episodes
            .iter()
            .map(|episode| episode.trial_index)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1), Some(2)],
    );
    assert_eq!(
        *hooks.started_trials.lock().unwrap(),
        vec![Some(0), Some(1), Some(2)],
    );
    assert_eq!(
        *hooks.completed_trials.lock().unwrap(),
        vec![Some(0), Some(1), Some(2)],
    );
}

#[tokio::test]
async fn no_trial_base_still_withholds_the_ordinal_from_an_undeclared_env() {
    // Minted by default, delivered only on declaration: a plain env keeps its
    // options-less reset, while the report still records the sweep.
    let env = TestEnv::default();
    let report = RuntimeDriver::new(
        trial_spec(2, &[]),
        env.clone(),
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    assert!(
        env.reset_options
            .lock()
            .unwrap()
            .iter()
            .all(|options| options.is_none()),
        "an env that declared no trial_index must not receive one",
    );
    assert_eq!(
        report
            .episodes
            .iter()
            .map(|episode| episode.trial_index)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1)],
    );
}

#[tokio::test]
async fn next_step_autoreset_mints_no_ordinal() {
    // Under NEXT_STEP the env restarts its own lanes: the default base validates
    // (an explicit 0 as well, the Python surface always passes an integer) but
    // no ordinal is minted, so neither the declaring env nor the report sees one.
    for base in [None, Some(0)] {
        let env = VectorTestEnv::new(vec![2, 3]);
        let hooks = Arc::new(RecordingHooks::default());
        let mut spec = vector_spec(2, 4);
        spec.trial_index_base = base;
        declare_reset_options(&mut spec, &["trial_index"]);

        let report = RuntimeDriver::new(spec, env.clone(), TestModel::default(), hooks.clone())
            .run()
            .await
            .unwrap();

        assert!(report.total_episodes >= 4);
        assert!(
            env.reset_options
                .lock()
                .unwrap()
                .iter()
                .all(|options| options.is_none()),
            "base {base:?}: no ordinal reaches an env that owns its resets",
        );
        assert!(
            report
                .episodes
                .iter()
                .all(|episode| episode.trial_index.is_none()),
            "base {base:?}: no ordinal is reported",
        );
        assert!(
            hooks
                .started_trials
                .lock()
                .unwrap()
                .iter()
                .all(Option::is_none)
        );
    }
}

// ---------------------------------------------------------------------------
// Lane sessions: one driver per lane over a shared, lane-capable env.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_abandoned_predict_at_route_end_sends_no_evict() {
    // One lane, a two-frame chunk with lead 1, so the episode's last step runs
    // from replay while the prefetch is in flight: that prefetch is gated for
    // good, and the ended episode's eviction is held behind it. Cancelling
    // leaves the driver draining a predict that never lands, and the endpoint
    // that just missed the close deadline must get no evict -- the RPC carries
    // no deadline of its own, so sending it would hang shutdown forever.
    let (_open, gate) = oneshot::channel();
    let model = TestModel {
        replay_frames: 1,
        gate: Some(gate.shared()),
        gate_at: 1,
        reset_adapter_hangs: true,
        ..TestModel::default()
    };
    let env = TestEnv {
        terminal_after: 2,
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = one_episode_spec();
    spec.limits.service_close_timeout = Duration::from_millis(50);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let run = RuntimeDriver::new(spec, env, model.clone(), hooks.clone())
        .with_prefetch(1)
        .run_with_cancellation(cancellation.clone());
    let cancel = async {
        tokio::time::timeout(Duration::from_secs(5), async {
            while hooks
                .completed_seeds
                .lock()
                .expect("completed seed recorder lock poisoned")
                .is_empty()
            {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("the episode never completed");
        cancellation.cancel();
    };
    let (result, ()) =
        tokio::time::timeout(Duration::from_secs(5), async { tokio::join!(run, cancel) })
            .await
            .expect("driver hung in shutdown on an evict after an abandoned predict");
    assert!(result.is_err(), "the run was cancelled");
    assert!(
        model
            .reset_adapters
            .lock()
            .expect("reset_adapter recorder lock poisoned")
            .is_empty(),
        "no evict goes to an endpoint whose predict missed the close deadline"
    );
}

#[tokio::test]
async fn a_gated_lane_predict_never_holds_a_sibling_lane() {
    // Two lanes on one endpoint, two-step episodes, a budget of four. The
    // first predict to arrive is held at a gate; the sibling lane must keep
    // stepping and predicting behind it rather than wait at a grouped
    // barrier, and the run still scores every budgeted slot once the gate
    // opens.
    let (open, gate) = oneshot::channel();
    let (arrivals, mut arrived) = mpsc::unbounded();
    let model = TestModel {
        gate: Some(gate.shared()),
        arrivals: Some(arrivals),
        ..TestModel::default()
    };
    let env = LaneTestEnv::new(vec![(2, Duration::ZERO), (2, Duration::ZERO)]);
    let run = RuntimeDriver::new(
        lane_spec(2, 4, (100..106).collect()),
        env,
        model,
        Arc::new(RecordingHooks::default()),
    )
    .run();
    let sibling = async {
        // One predict is gated; the sibling lane lands two more (its next
        // step's, then its next episode's) while it is held.
        tokio::time::timeout(Duration::from_secs(5), async {
            for _ in 0..3 {
                arrived.next().await.expect("model dropped");
            }
        })
        .await
        .expect("the sibling lane's predicts stalled behind the gated one");
        open.send(()).expect("gated predict dropped");
    };
    let (report, ()) = tokio::join!(run, sibling);
    assert_eq!(report.unwrap().total_episodes, 4);
}

#[tokio::test]
async fn lane_sessions_score_exactly_the_budgeted_slots_whatever_the_lane_timing() {
    // Three lanes: a 2-step lane, a slow 5-step lane, and a 1-step sprinter.
    // Budget 7 episodes with 10 explicit seeds. The sprinter takes most slots
    // and lane order is timing-dependent, but the scored set is slots 0..7 with
    // seeds 100..107, every slot exactly once, indexed by slot.
    // The sprinter's whole episode is one in-memory round trip; the other
    // lanes' steps take tens of milliseconds, so the slot assignment is not a
    // coin flip under a loaded runner.
    let env = LaneTestEnv::new(vec![
        (2, Duration::from_millis(20)),
        (5, Duration::from_millis(40)),
        (1, Duration::ZERO),
    ]);
    let model = TestModel::default();
    let report = RuntimeDriver::new(
        lane_spec(3, 7, (100..110).collect()),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    assert_eq!(report.total_episodes, 7);
    // The report lists episodes in completion order; by slot they are exactly
    // 1..=7 with seeds 100..107.
    let mut by_slot = report.episodes.clone();
    by_slot.sort_by_key(|e| e.episode_index);
    let indices: Vec<i64> = by_slot.iter().map(|e| e.episode_index).collect();
    assert_eq!(indices, (1..=7).collect::<Vec<_>>(), "one record per slot");
    let seeds: Vec<i64> = by_slot.iter().map(|e| e.seed.unwrap()).collect();
    assert_eq!(
        seeds,
        (100..107).collect::<Vec<_>>(),
        "seed is fixed by slot"
    );
    let lanes_used: std::collections::BTreeSet<i32> =
        report.episodes.iter().map(|e| e.env_index).collect();
    assert_eq!(
        lanes_used,
        [0, 1, 2].into_iter().collect(),
        "every lane ran"
    );
    // Lanes 0 and 1 hold slots 0 and 1 for at least 40ms; the sprinter takes
    // every other slot: 2..7, five episodes.
    let sprinter = report.episodes.iter().filter(|e| e.env_index == 2).count();
    assert_eq!(sprinter, 5, "the fastest lane takes the remaining slots");
    assert!(
        sprinter >= 3,
        "the fastest lane takes the most slots, got {sprinter}"
    );

    // The env saw exactly the seven seeded lane resets and one close; the
    // release handle released once after every lane finished.
    let state = env.inner.lock().expect("lane env state lock poisoned");
    let mut seen: Vec<i64> = state.resets.iter().filter_map(|(_, s)| *s).collect();
    seen.sort_unstable();
    assert_eq!(seen, (100..107).collect::<Vec<_>>());
    assert_eq!(env.closed.load(Ordering::SeqCst), 1);
    assert!(model.closed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn lane_sessions_idle_surplus_lanes_when_the_budget_is_smaller() {
    // Four lanes, two episodes: two lanes run one episode each and the other
    // two never reset (their first slot claim is already past the budget).
    let env = LaneTestEnv::new(vec![(1, Duration::ZERO); 4]);
    let hooks = Arc::new(RecordingHooks::default());
    let report = RuntimeDriver::new(
        lane_spec(4, 2, Vec::new()),
        env.clone(),
        TestModel::default(),
        hooks.clone(),
    )
    .run()
    .await
    .unwrap();
    assert_eq!(report.total_episodes, 2);
    // Driver-owned lane resets complete each episode at its terminal step, as
    // the last event under its id.
    let completed = hooks.completed_ids();
    assert_eq!(completed.len(), 2);
    for id in &completed {
        assert_eq!(
            hooks.events_for(id),
            ["started", "observation", "step", "completed"]
        );
    }
    assert_eq!(
        env.inner
            .lock()
            .expect("lane env state lock poisoned")
            .resets
            .len(),
        2
    );
}

// ---------------------------------------------------------------------------
// The episode ledger under a runtime-side truncation: the runtime owns episode
// ids (R1), so a peer-reported completion naming a superseded id is an echo.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_runtime_truncated_episode_is_not_double_counted_when_the_env_echoes_it() {
    // max_episode_steps=4 over 3 episodes on an env that never terminates. The
    // driver truncates and resets mid-episode, so the env buffers each
    // interrupted episode and reports it one step into the NEXT episode. Trusting
    // that echo counted every episode twice, evicted the live episode's adapter
    // state, and reset the fresh episode after a single step -- so the run
    // degenerated into 1-step episodes.
    let env = EchoingEnv::default();
    let hooks = Arc::new(RecordingHooks::default());
    let spec = RuntimeSessionSpec {
        max_episodes: Some(3),
        max_episode_steps: Some(4),
        ..one_episode_spec()
    };
    let report = RuntimeDriver::new(spec, env.clone(), TestModel::default(), hooks.clone())
        .run()
        .await
        .unwrap();

    let state = env.inner.lock().expect("echoing env lock poisoned");
    assert_eq!(state.steps, 12, "3 episodes of exactly max_episode_steps=4");
    assert_eq!(
        state.resets, 3,
        "one reset per episode, cold start included"
    );
    drop(state);

    assert_eq!(
        report.total_episodes, 3,
        "an echoed id must not count again"
    );
    assert_eq!(report.total_steps, 12);
    assert_eq!(report.episodes.len(), 3, "one summary per episode");
    assert_eq!(
        report
            .episodes
            .iter()
            .map(|episode| episode.step_count)
            .collect::<Vec<_>>(),
        vec![4, 4, 4],
        "every episode runs the full cap, not one step"
    );
    assert!(
        report.episodes.iter().all(|episode| episode.truncated),
        "the runtime truncated every one of them"
    );
    assert_eq!(
        report
            .episodes
            .iter()
            .map(|episode| episode.episode_index)
            .collect::<Vec<_>>(),
        vec![1, 2, 3],
        "distinct, contiguous completion indices"
    );
    let completed = hooks.completed_ids();
    assert_eq!(completed.len(), 3);
    assert_eq!(
        completed
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3,
        "three distinct episode ids completed: {completed:?}"
    );
}

/// A lane-capable env under NEXT_STEP autoreset: every op names exactly one
/// lane, each lane ends its episode on a fixed cadence and rolls itself on the
/// following step. No driver reset ever runs after the cold start, so the
/// route's episode budget is the only thing that can end the run.
#[derive(Clone)]
struct NextStepLaneEnv {
    inner: Arc<Mutex<NextStepLaneState>>,
    length: usize,
}

struct NextStepLaneState {
    resets: usize,
    step: Vec<usize>,
    ids: Vec<String>,
    pending_autoreset: Vec<bool>,
}

impl NextStepLaneEnv {
    fn new(num_envs: usize, length: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(NextStepLaneState {
                resets: 0,
                step: vec![0; num_envs],
                ids: vec![String::new(); num_envs],
                pending_autoreset: vec![false; num_envs],
            })),
            length,
        }
    }
}

#[async_trait]
impl RuntimeEnv for NextStepLaneEnv {
    async fn reset(&mut self, request: ResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        assert_eq!(request.env_indices.len(), 1, "lane resets name one lane");
        let lane = request.env_indices[0] as usize;
        let mut state = self.inner.lock().expect("lane env lock poisoned");
        state.resets += 1;
        state.step[lane] = 0;
        state.pending_autoreset[lane] = false;
        state.ids[lane] = request.episode_ids.first().cloned().unwrap_or_default();
        Ok(RuntimeEnvReset {
            response: ResetResponse {
                observation: Some(leaves_value(payload([0]))),
                infos: None,
            },
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn step(&mut self, request: StepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        assert_eq!(request.env_indices.len(), 1, "lane steps name one lane");
        let lane = request.env_indices[0] as usize;
        let mut state = self.inner.lock().expect("lane env lock poisoned");
        if state.pending_autoreset[lane] {
            // t+1: the env rolls this lane itself and returns the new episode's
            // reset observation, adopting the id the runtime pushed down.
            state.pending_autoreset[lane] = false;
            state.step[lane] = 0;
            state.ids[lane] = request.episode_ids.first().cloned().unwrap_or_default();
            return Ok(RuntimeEnvStep {
                response: StepResponse {
                    observation: Some(leaves_value(payload([0]))),
                    rewards: vec![0.0],
                    terminated_mask: vec![0],
                    truncated_mask: vec![0],
                    infos: None,
                    completed_episodes: Vec::new(),
                    env_indices: vec![lane as u32],
                },
                endpoint_total_ns: None,
                phases: EndpointPhases::default(),
            });
        }
        state.step[lane] += 1;
        let done = state.step[lane] >= self.length;
        let completed_episodes = done
            .then(|| EpisodeMetadata {
                episode_id: state.ids[lane].clone(),
                env_index: lane as u32,
                step_count: state.step[lane] as i64,
                cumulative_reward: state.step[lane] as f64,
                terminated: true,
                ..Default::default()
            })
            .into_iter()
            .collect();
        state.pending_autoreset[lane] = done;
        Ok(RuntimeEnvStep {
            response: StepResponse {
                observation: Some(leaves_value(payload([0]))),
                rewards: vec![1.0],
                terminated_mask: vec![u8::from(done)],
                truncated_mask: vec![0],
                infos: None,
                completed_episodes,
                env_indices: vec![lane as u32],
            },
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}

#[tokio::test]
async fn a_next_step_lane_session_stops_at_the_episode_budget() {
    // A lane group's budget is normally the slot counter, claimed at reset -- but
    // under NEXT_STEP the env owns the rolls and no reset ever runs, so nothing
    // checked `max_episodes` and the run never terminated.
    let num_envs = 2;
    let budget = 4;
    let env = NextStepLaneEnv::new(num_envs, 2);
    let hooks = Arc::new(RecordingHooks::default());
    let spec = RuntimeSessionSpec {
        subset_step: true,
        ..vector_spec(num_envs, budget)
    };
    let report = tokio::time::timeout(
        Duration::from_secs(10),
        RuntimeDriver::new(spec, env.clone(), TestModel::default(), hooks.clone()).run(),
    )
    .await
    .expect("a lane session under NEXT_STEP must stop at its episode budget")
    .unwrap();

    assert_eq!(
        env.inner.lock().expect("lane env lock poisoned").resets,
        num_envs,
        "one cold-start reset per lane, and never again under NEXT_STEP"
    );
    assert!(
        (budget as i64..=budget as i64 + num_envs as i64).contains(&report.total_episodes),
        "the budget bounds the run (one lane may overshoot by the episode it \
         was already finishing), got {}",
        report.total_episodes
    );
    assert_eq!(report.episodes.len() as i64, report.total_episodes);
    let completed = hooks.completed_ids();
    assert_eq!(
        completed
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        completed.len(),
        "every completion names a distinct episode id: {completed:?}"
    );
}

#[tokio::test]
async fn a_leftover_chunk_frame_never_crosses_an_episode_boundary() {
    // One-step episodes with a chunk of 3: every predict leaves two buffered
    // frames behind when its episode ends. Each new episode must re-plan from
    // its own reset observation rather than replay them, so the model sees
    // exactly one predict per episode, each carrying the reset observation
    // (byte 1) and no step number (no history was negotiated).
    let env = TestEnv {
        terminal_after: 1,
        ..Default::default()
    };
    let model = TestModel {
        replay_frames: 2,
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(3);

    let report = RuntimeDriver::new(spec, env, model.clone(), hooks)
        .run()
        .await
        .unwrap();

    assert_eq!(report.total_episodes, 3);
    assert_eq!(report.total_steps, 3);
    let ledger = model.ledger.lock().expect("ledger poisoned").clone();
    assert_eq!(ledger, vec![(vec![], (-1, 1)); 3]);
}

#[tokio::test]
async fn a_leftover_chunk_frame_never_crosses_a_next_step_autoreset() {
    // Same invariant when the ENV owns the reset (NEXT_STEP): the driver never
    // issues a reset after the cold start, so the reset-path discard cannot
    // save it — the episode-completion path must drop the buffered frames.
    // Single lane, one-step episodes, chunk of 3.
    let env = VectorTestEnv::new(vec![1]);
    let model = TestModel {
        replay_frames: 2,
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());

    let report = RuntimeDriver::new(vector_spec(1, 3), env, model.clone(), hooks)
        .run()
        .await
        .unwrap();

    assert_eq!(report.total_episodes, 3);
    let predicts = model.predicts.load(Ordering::SeqCst);
    assert_eq!(predicts, 3, "one fresh plan per episode, got {predicts}");
}
#[tokio::test]
async fn the_driver_sends_the_shared_reset_options_encoding_on_the_wire() {
    // The literal encoding, not a restatement of the fn the driver calls: a
    // single-lane reset carries the bare integer and a multi-lane reset the
    // list in lane order, so the managed prober can be held to the same bytes.
    let env = TestEnv::default();
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = trial_spec(2, &["trial_index"]);
    spec.trial_index_base = Some(41);

    RuntimeDriver::new(spec, env.clone(), TestModel::default(), hooks)
        .run()
        .await
        .unwrap();

    let options = env.reset_options.lock().unwrap().clone();
    let trials = options
        .iter()
        .map(|options| {
            options
                .as_ref()
                .expect("a declaring env receives options")
                .entries
                .get("trial_index")
                .and_then(|value| value.kind.clone())
                .expect("trial_index")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        trials,
        vec![meta_value::Kind::Integer(41), meta_value::Kind::Integer(42),],
        "a single-lane reset sends the bare integer",
    );
}

#[test]
fn a_multi_lane_reset_sends_the_trial_ordinals_as_a_list_in_lane_order() {
    let mut spec = trial_spec(1, &["trial_index"]);
    declare_reset_options(&mut spec, &["trial_index"]);

    let options = rlmesh_runtime::reset_options_for(
        &spec.env_contract,
        rlmesh_runtime::TRIAL_INDEX_OPTION,
        &[41, 42],
    )
    .expect("a declaring env receives options");

    assert_eq!(
        options
            .entries
            .get("trial_index")
            .and_then(|v| v.kind.clone()),
        Some(meta_value::Kind::List(rlmesh_proto::spaces::v1::MetaList {
            items: [41, 42]
                .into_iter()
                .map(|trial| MetaValue {
                    kind: Some(meta_value::Kind::Integer(trial)),
                })
                .collect(),
        })),
    );
}
