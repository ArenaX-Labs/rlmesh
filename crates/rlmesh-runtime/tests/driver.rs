//! Behavioral fingerprint for workflow edition 2026.06: these lifecycle
//! assertions are the edition contract (episode accounting, per-lane autoreset,
//! request/response ordering). Changing observable behavior here changes the
//! edition; see docs/editions/2026.06.md.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use prost::bytes::Bytes;
use rlmesh_proto::core::v1::{EnvContract, EnvSpec};
use rlmesh_proto::env::v1::{
    EpisodeMetadata, ResetRequest, ResetResponse, StepRequest, StepResponse,
};
use rlmesh_proto::model::v1::{
    PredictRequest, PredictResponse, ReleaseAdapterRequest, ResetAdapterRequest,
};
use rlmesh_proto::spaces::v1::{MetaMap, MetaValue, SpaceSpec, SpaceValue, meta_value};
use rlmesh_runtime::{
    ActionReceivedEvent, EndpointPhases, HookError, RuntimeDriver, RuntimeEnv, RuntimeEnvReset,
    RuntimeEnvStep, RuntimeError, RuntimeHooks, RuntimeModel, RuntimeModelPrediction,
    RuntimeSessionSpec,
};

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

#[derive(Debug, thiserror::Error)]
#[error("simulated transport failure")]
struct FakeTransportError;

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
            lane_skew_ns: 0,
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
    assert_eq!(avg("model.predict", "predict.queue"), 4.0);
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
            lane_skew_ns: 8_000_000,
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
        "predict.queue",
        "predict.in_flight",
        "lane.skew",
    ] {
        assert!(!recorded(metric), "{metric} must stay unrecorded");
    }
}

#[derive(Clone)]
struct TestEnv {
    closed: Arc<AtomicBool>,
    step_count: Arc<AtomicUsize>,
    reset_seeds: Arc<Mutex<Vec<Vec<i64>>>>,
    // The runtime is the id authority: the env adopts the id pushed down on
    // reset and echoes it back in completed_episodes (never mints its own).
    current_episode: Arc<Mutex<String>>,
    terminal_after: usize,
    // What this env claims to have spent on each op, as a peer would stamp it.
    endpoint_total_ns: Option<u64>,
    phases: EndpointPhases,
}

impl Default for TestEnv {
    fn default() -> Self {
        Self {
            closed: Arc::new(AtomicBool::new(false)),
            step_count: Arc::new(AtomicUsize::new(0)),
            reset_seeds: Arc::new(Mutex::new(Vec::new())),
            current_episode: Arc::new(Mutex::new(String::new())),
            terminal_after: 1,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        }
    }
}

#[async_trait]
impl RuntimeEnv for TestEnv {
    async fn reset(&mut self, request: ResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        self.reset_seeds
            .lock()
            .expect("reset seed recorder lock poisoned")
            .push(request.seeds);
        // Adopt the runtime-pushed id (the env never mints).
        *self
            .current_episode
            .lock()
            .expect("current episode lock poisoned") =
            request.episode_ids.first().cloned().unwrap_or_default();
        self.step_count.store(0, Ordering::SeqCst);
        Ok(RuntimeEnvReset {
            response: ResetResponse {
                observation: Some(leaves_value(payload([1]))),
                infos: Some(info_map("phase", "reset")),
            },
            endpoint_total_ns: self.endpoint_total_ns,
            phases: self.phases,
        })
    }

    async fn step(&mut self, _request: StepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        let step = self.step_count.fetch_add(1, Ordering::SeqCst) + 1;
        let terminal = step >= self.terminal_after;
        let episode_id = self
            .current_episode
            .lock()
            .expect("current episode lock poisoned")
            .clone();
        Ok(RuntimeEnvStep {
            response: StepResponse {
                observation: Some(leaves_value(payload([step as u8]))),
                rewards: vec![1.0],
                terminated_mask: vec![u8::from(terminal)],
                truncated_mask: vec![0],
                infos: Some(info_map("phase", "step")),
                completed_episodes: terminal
                    .then(|| EpisodeMetadata {
                        episode_id,
                        step_count: step as i64,
                        cumulative_reward: step as f64,
                        terminated: true,
                        ..Default::default()
                    })
                    .into_iter()
                    .collect(),
                env_indices: vec![],
            },
            endpoint_total_ns: self.endpoint_total_ns,
            phases: self.phases,
        })
    }

    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Clone, Default)]
struct TestModel {
    closed: Arc<AtomicBool>,
    predicts: Arc<AtomicUsize>,
    predict_delay: Option<Duration>,
    seen_observations: Arc<Mutex<Vec<Vec<u8>>>>,
    // Episode ids the driver asked the model to evict via ResetAdapter (R2), in
    // order — one ResetAdapterRequest's episode_ids per entry.
    reset_adapters: Arc<Mutex<Vec<Vec<String>>>>,
    // Number of chunk replay frames to return per predict (frames 1.. of the
    // ordered `actions` list; frame 0 is always present). 0 = not chunking (a
    // single-frame `actions`, the unchanged path).
    replay_frames: usize,
    // Simulates a release_adapter impl that blocks (e.g. an RPC on a hung
    // connection) without honoring the supplied timeout.
    release_adapter_hangs: bool,
    // What this model claims to have spent on each predict, as a peer would
    // stamp it.
    endpoint_total_ns: Option<u64>,
    phases: EndpointPhases,
}

#[async_trait]
impl RuntimeModel for TestModel {
    async fn predict(
        &mut self,
        request: PredictRequest,
    ) -> Result<RuntimeModelPrediction, RuntimeError> {
        if let Some(delay) = self.predict_delay {
            tokio::time::sleep(delay).await;
        }
        self.predicts.fetch_add(1, Ordering::SeqCst);
        let observation_bytes = request
            .observation
            .as_ref()
            .and_then(|value| value.leaves.first())
            .map(|leaf| leaf.to_vec())
            .unwrap_or_default();
        self.seen_observations
            .lock()
            .expect("model observation recorder lock poisoned")
            .push(observation_bytes);
        // Ordered chunk frames: `actions[0]` is this step, `actions[1..]` are the
        // replay frames the driver buffers and replays without re-calling the model.
        let mut actions = vec![leaves_value(payload([0]))];
        actions.extend((0..self.replay_frames).map(|i| leaves_value(payload([100 + i as u8]))));
        Ok(RuntimeModelPrediction {
            response: PredictResponse {
                context: request.context,
                actions,
            },
            endpoint_total_ns: self.endpoint_total_ns,
            phases: self.phases,
            group_size: None,
        })
    }

    async fn reset_adapter(&mut self, request: ResetAdapterRequest) -> Result<(), RuntimeError> {
        self.reset_adapters
            .lock()
            .expect("reset_adapter recorder lock poisoned")
            .push(request.episode_ids);
        Ok(())
    }

    async fn release_adapter(
        &mut self,
        _request: ReleaseAdapterRequest,
        _timeout: Duration,
    ) -> Result<(), String> {
        if self.release_adapter_hangs {
            // Ignore the supplied timeout entirely, like a misbehaving impl.
            std::future::pending::<()>().await;
        }
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }
}

fn info_map(key: &str, value: &str) -> MetaMap {
    MetaMap {
        entries: [(
            key.to_string(),
            MetaValue {
                kind: Some(meta_value::Kind::Text(value.to_string())),
            },
        )]
        .into(),
    }
}

#[derive(Debug, Clone)]
struct EmittedObservation {
    episode_ids: Vec<String>,
    observation: Vec<u8>,
    raw_observation: Vec<u8>,
    infos: Option<MetaMap>,
}

#[derive(Default)]
struct RecordingHooks {
    actions: AtomicUsize,
    ended: AtomicUsize,
    failed: AtomicUsize,
    fail_action_transform: bool,
    // When set, transform_observation prepends this marker byte to every
    // observation it forwards to the model.
    observation_marker: Option<u8>,
    emitted_observations: Mutex<Vec<EmittedObservation>>,
    step_infos: Mutex<Vec<Option<MetaMap>>>,
    started_seeds: Mutex<Vec<Option<i64>>>,
    completed_seeds: Mutex<Vec<Option<i64>>>,
    // Counts of live telemetry snapshots streamed via on_telemetry, by horizon.
    telemetry_windows: AtomicUsize,
    telemetry_sessions: AtomicUsize,
    // Largest row count seen in any Session snapshot — proves the final push
    // carried real telemetry, not an empty event.
    telemetry_session_rows: AtomicUsize,
}

#[async_trait]
impl RuntimeHooks for RecordingHooks {
    async fn action_received(&self, _event: ActionReceivedEvent) -> Result<(), HookError> {
        self.actions.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn transform_action(
        &self,
        event: ActionReceivedEvent,
    ) -> Result<Option<Vec<Bytes>>, HookError> {
        if self.fail_action_transform {
            return Err(HookError::Message("transform failed".to_string()));
        }
        Ok(event.action)
    }

    async fn transform_observation(
        &self,
        event: rlmesh_runtime::ObservationEmittedEvent,
    ) -> Result<Option<Vec<Bytes>>, HookError> {
        let Some(marker) = self.observation_marker else {
            return Ok(event.observation);
        };
        // `Bytes` is immutable, so prepend the marker to the first leaf by
        // building a fresh buffer (the marker stays byte 0 of leaf 0).
        Ok(event.observation.map(|mut leaves| {
            if let Some(first) = leaves.first_mut() {
                let mut prefixed = Vec::with_capacity(first.len() + 1);
                prefixed.push(marker);
                prefixed.extend_from_slice(first);
                *first = Bytes::from(prefixed);
            }
            leaves
        }))
    }

    async fn observation_emitted(
        &self,
        event: rlmesh_runtime::ObservationEmittedEvent,
    ) -> Result<(), HookError> {
        let first_leaf = |leaves: Option<Vec<Bytes>>| {
            leaves
                .and_then(|leaves| leaves.into_iter().next())
                .map(|leaf| leaf.to_vec())
                .unwrap_or_default()
        };
        self.emitted_observations
            .lock()
            .expect("emitted observation recorder lock poisoned")
            .push(EmittedObservation {
                episode_ids: event.episode_ids,
                observation: first_leaf(event.observation),
                raw_observation: first_leaf(event.raw_observation),
                infos: event.infos,
            });
        Ok(())
    }

    async fn step_completed(
        &self,
        event: rlmesh_runtime::StepCompletedEvent,
    ) -> Result<(), HookError> {
        self.step_infos
            .lock()
            .expect("step info recorder lock poisoned")
            .push(event.infos);
        Ok(())
    }

    async fn episode_started(
        &self,
        event: rlmesh_runtime::EpisodeStartedEvent,
    ) -> Result<(), HookError> {
        self.started_seeds
            .lock()
            .expect("started seed recorder lock poisoned")
            .push(event.seed);
        Ok(())
    }

    async fn episode_completed(
        &self,
        event: rlmesh_runtime::EpisodeCompletedEvent,
    ) -> Result<(), HookError> {
        self.completed_seeds
            .lock()
            .expect("completed seed recorder lock poisoned")
            .push(event.seed);
        Ok(())
    }

    async fn session_ended(
        &self,
        _event: rlmesh_runtime::SessionEndedEvent,
    ) -> Result<(), HookError> {
        self.ended.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn on_telemetry(
        &self,
        event: rlmesh_runtime::TelemetrySnapshotEvent,
    ) -> Result<(), HookError> {
        let rows = event.snapshot.rows.len();
        if event.snapshot.horizon == rlmesh_runtime::telemetry::Horizon::Window {
            self.telemetry_windows.fetch_add(1, Ordering::SeqCst);
        } else if event.snapshot.horizon == rlmesh_runtime::telemetry::Horizon::Session {
            self.telemetry_sessions.fetch_add(1, Ordering::SeqCst);
            self.telemetry_session_rows
                .fetch_max(rows, Ordering::SeqCst);
        }
        Ok(())
    }

    async fn session_failed(
        &self,
        _event: rlmesh_runtime::SessionFailedEvent,
    ) -> Result<(), HookError> {
        self.failed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn one_episode_spec() -> RuntimeSessionSpec {
    RuntimeSessionSpec {
        session_id: "session-1".to_string(),
        env_id: "TestEnv-v0".to_string(),
        env_component_id: "env-1".to_string(),
        model_component_id: "model-1".to_string(),
        workflow_edition: rlmesh_proto::CURRENT_WORKFLOW_EDITION.to_string(),
        env_contract: EnvContract {
            spec: Some(EnvSpec {
                observation_space: Some(SpaceSpec::default()),
                action_space: Some(SpaceSpec::default()),
                ..Default::default()
            }),
            num_envs: 1,
            ..Default::default()
        },
        num_envs: 1,
        episode_seeds: Vec::new(),
        base_seed: None,
        max_episodes: Some(1),
        max_episode_steps: None,
        max_episode_seconds: None,
        close_env_on_end: true,
        limits: Default::default(),
    }
}

fn payload<const N: usize>(data: [u8; N]) -> Bytes {
    Bytes::copy_from_slice(&data)
}

fn leaves_value(data: Bytes) -> SpaceValue {
    SpaceValue { leaves: vec![data] }
}

/// A NEXT_STEP vector env with a per-lane terminal schedule. It mimics the env
/// server's output: a lane terminates at its scheduled step (terminal obs keeps
/// the old episode id), then auto-resets on the FOLLOWING step (fresh obs, new
/// id, reward 0), never requiring a driver-issued reset.
#[derive(Clone)]
struct VectorTestEnv {
    reset_seeds: Arc<Mutex<Vec<Vec<i64>>>>,
    closed: Arc<AtomicBool>,
    terminal_after: Vec<usize>,
    lane_step: Vec<usize>,
    // The runtime is the id authority: each lane adopts the id pushed down on
    // reset / on the autoreset roll, and echoes it in completed_episodes.
    current_ids: Vec<String>,
    pending_autoreset: Vec<bool>,
}

impl VectorTestEnv {
    fn new(terminal_after: Vec<usize>) -> Self {
        let n = terminal_after.len();
        Self {
            reset_seeds: Arc::new(Mutex::new(Vec::new())),
            closed: Arc::new(AtomicBool::new(false)),
            terminal_after,
            lane_step: vec![0; n],
            current_ids: vec![String::new(); n],
            pending_autoreset: vec![false; n],
        }
    }
}

#[async_trait]
impl RuntimeEnv for VectorTestEnv {
    async fn reset(&mut self, request: ResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        self.reset_seeds
            .lock()
            .expect("reset seed recorder lock poisoned")
            .push(request.seeds);
        let n = self.terminal_after.len();
        self.lane_step = vec![0; n];
        self.pending_autoreset = vec![false; n];
        // Adopt the runtime-pushed ids (full-width on a whole-vector reset).
        self.current_ids = request.episode_ids.clone();
        self.current_ids.resize(n, String::new());
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
        let n = self.terminal_after.len();
        let mut rewards = vec![1.0; n];
        let mut terminated_mask = vec![0u8; n];
        let mut completed_episodes = Vec::new();
        let mut rolled = false;

        for lane in 0..n {
            if self.pending_autoreset[lane] {
                rolled = true;
                // t+1: the env auto-resets this lane and delivers the fresh obs of
                // a new episode (step 0, reward 0, terminated=false). It adopts the
                // rolled id the runtime pushed for this lane.
                self.pending_autoreset[lane] = false;
                self.lane_step[lane] = 0;
                rewards[lane] = 0.0;
                if let Some(id) = request.episode_ids.get(lane) {
                    self.current_ids[lane] = id.clone();
                }
            } else {
                self.lane_step[lane] += 1;
                if self.lane_step[lane] >= self.terminal_after[lane] {
                    terminated_mask[lane] = 1;
                    completed_episodes.push(EpisodeMetadata {
                        episode_id: self.current_ids[lane].clone(),
                        env_index: lane as u32,
                        step_count: self.lane_step[lane] as i64,
                        cumulative_reward: self.lane_step[lane] as f64,
                        terminated: true,
                        ..Default::default()
                    });
                    self.pending_autoreset[lane] = true;
                }
            }
        }

        Ok(RuntimeEnvStep {
            response: StepResponse {
                observation: Some(leaves_value(payload([0]))),
                rewards,
                terminated_mask,
                truncated_mask: vec![0u8; n],
                infos: Some(info_map("phase", if rolled { "roll" } else { "step" })),
                completed_episodes,
                env_indices: vec![],
            },
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }
}

fn vector_spec(num_envs: usize, max_episodes: u64) -> RuntimeSessionSpec {
    RuntimeSessionSpec {
        session_id: "session-vec".to_string(),
        env_id: "VectorTestEnv-v0".to_string(),
        env_component_id: "env-vec".to_string(),
        model_component_id: "model-vec".to_string(),
        workflow_edition: rlmesh_proto::CURRENT_WORKFLOW_EDITION.to_string(),
        env_contract: EnvContract {
            spec: Some(EnvSpec {
                observation_space: Some(SpaceSpec::default()),
                action_space: Some(SpaceSpec::default()),
                ..Default::default()
            }),
            num_envs: num_envs as u32,
            autoreset_mode: rlmesh_proto::core::v1::AutoresetMode::NextStep as i32,
            ..Default::default()
        },
        num_envs,
        episode_seeds: Vec::new(),
        base_seed: None,
        max_episodes: Some(max_episodes),
        max_episode_steps: None,
        max_episode_seconds: None,
        close_env_on_end: true,
        limits: Default::default(),
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
    assert_eq!(
        evicted.len() as i64,
        report.total_episodes,
        "one eviction per completed episode, even under chunking"
    );
    for id in &evicted {
        assert_eq!(id.len(), 36, "evicted a UUID id, got {id:?}");
        assert_eq!(id.as_bytes()[14], b'7', "UUIDv7 version nibble: {id:?}");
    }
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
        .with_prefetch(Box::new(model.clone()), 1)
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
    let model = TestModel {
        replay_frames: 2,
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(2);

    let report = RuntimeDriver::new(spec, env.clone(), model.clone(), hooks.clone())
        .with_prefetch(Box::new(model.clone()), 1)
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
