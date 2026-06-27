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
use rlmesh_proto::model::v1::{CloseRouteRequest, PredictRequest, PredictResponse};
use rlmesh_proto::spaces::v1::{SpaceSpec, SpaceValue};
use rlmesh_runtime::{
    ActionReceivedEvent, HookError, RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep,
    RuntimeError, RuntimeHooks, RuntimeModel, RuntimeModelPrediction, RuntimeSessionSpec,
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
    assert!(env.closed.load(Ordering::SeqCst));
    assert!(model.closed.load(Ordering::SeqCst));
    assert_eq!(hooks.actions.load(Ordering::SeqCst), 1);
    assert_eq!(hooks.ended.load(Ordering::SeqCst), 1);
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
    RuntimeDriver::new(
        spec.clone(),
        first_env.clone(),
        first_model,
        Arc::new(RecordingHooks::default()),
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
    for (_, bytes) in &emitted {
        assert_eq!(
            bytes.first().copied(),
            Some(MARKER),
            "observation_emitted exposed pre-transform bytes: {bytes:?}"
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
            .map(|(_, bytes)| bytes.clone())
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn shutdown_enforces_service_close_timeout_on_hung_model() {
    let env = TestEnv::default();
    let model = TestModel {
        close_route_hangs: true,
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

#[derive(Clone)]
struct TestEnv {
    closed: Arc<AtomicBool>,
    step_count: Arc<AtomicUsize>,
    reset_seeds: Arc<Mutex<Vec<Vec<i64>>>>,
    terminal_after: usize,
}

impl Default for TestEnv {
    fn default() -> Self {
        Self {
            closed: Arc::new(AtomicBool::new(false)),
            step_count: Arc::new(AtomicUsize::new(0)),
            reset_seeds: Arc::new(Mutex::new(Vec::new())),
            terminal_after: 1,
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
        self.step_count.store(0, Ordering::SeqCst);
        Ok(RuntimeEnvReset {
            response: ResetResponse {
                observation: Some(leaves_value(payload([1]))),
                infos: None,
                episode_ids: vec!["episode-1".to_string()],
            },
            endpoint_total_ns: None,
        })
    }

    async fn step(&mut self, _request: StepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        let step = self.step_count.fetch_add(1, Ordering::SeqCst) + 1;
        let terminal = step >= self.terminal_after;
        Ok(RuntimeEnvStep {
            response: StepResponse {
                observation: Some(leaves_value(payload([step as u8]))),
                rewards: vec![1.0],
                terminated_mask: vec![u8::from(terminal)],
                truncated_mask: vec![0],
                infos: None,
                completed_episodes: terminal
                    .then(|| EpisodeMetadata {
                        episode_id: "episode-1".to_string(),
                        step_count: step as i64,
                        cumulative_reward: step as f64,
                        terminated: true,
                        ..Default::default()
                    })
                    .into_iter()
                    .collect(),
                episode_ids: vec![],
                env_indices: vec![],
            },
            endpoint_total_ns: None,
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
    // Number of chunk replay frames to return per predict (frames 1.. of the
    // ordered `actions` list; frame 0 is always present). 0 = not chunking (a
    // single-frame `actions`, the unchanged path).
    replay_frames: usize,
    // Simulates a close_route impl that blocks (e.g. an RPC on a hung
    // connection) without honoring the supplied timeout.
    close_route_hangs: bool,
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
            endpoint_total_ns: None,
        })
    }

    async fn close_route(
        &mut self,
        _request: CloseRouteRequest,
        _timeout: Duration,
    ) -> Result<(), String> {
        if self.close_route_hangs {
            // Ignore the supplied timeout entirely, like a misbehaving impl.
            std::future::pending::<()>().await;
        }
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }
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
    // Records (is_reset, observation_bytes) for every observation_emitted hook.
    emitted_observations: Mutex<Vec<(bool, Vec<u8>)>>,
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
        let bytes = event
            .observation
            .and_then(|leaves| leaves.into_iter().next())
            .map(|leaf| leaf.to_vec())
            .unwrap_or_default();
        self.emitted_observations
            .lock()
            .expect("emitted observation recorder lock poisoned")
            .push((event.is_reset, bytes));
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
        route_id: "route-1".to_string(),
        env_component_id: "env-1".to_string(),
        model_component_id: "model-1".to_string(),
        env_id: "TestEnv-v0".to_string(),
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
        base_seed: None,
        max_episodes: Some(1),
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
    lane_episode: Vec<usize>,
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
            lane_episode: vec![0; n],
            pending_autoreset: vec![false; n],
        }
    }

    fn episode_id(lane: usize, episode: usize) -> String {
        format!("ep-{lane}-{episode}")
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
        self.lane_episode = vec![0; n];
        self.pending_autoreset = vec![false; n];
        let episode_ids = (0..n).map(|lane| Self::episode_id(lane, 0)).collect();
        Ok(RuntimeEnvReset {
            response: ResetResponse {
                observation: Some(leaves_value(payload([0]))),
                infos: None,
                episode_ids,
            },
            endpoint_total_ns: None,
        })
    }

    async fn step(&mut self, _request: StepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        let n = self.terminal_after.len();
        let mut rewards = vec![1.0; n];
        let mut terminated_mask = vec![0u8; n];
        let mut completed_episodes = Vec::new();
        let mut episode_ids = vec![String::new(); n];

        for lane in 0..n {
            if self.pending_autoreset[lane] {
                // t+1: the env auto-resets this lane and delivers the fresh obs of
                // a new episode (step 0, reward 0, terminated=false).
                self.pending_autoreset[lane] = false;
                self.lane_episode[lane] += 1;
                self.lane_step[lane] = 0;
                rewards[lane] = 0.0;
                episode_ids[lane] = Self::episode_id(lane, self.lane_episode[lane]);
            } else {
                self.lane_step[lane] += 1;
                let id = Self::episode_id(lane, self.lane_episode[lane]);
                if self.lane_step[lane] >= self.terminal_after[lane] {
                    terminated_mask[lane] = 1;
                    completed_episodes.push(EpisodeMetadata {
                        episode_id: id.clone(),
                        env_index: lane as u32,
                        step_count: self.lane_step[lane] as i64,
                        cumulative_reward: self.lane_step[lane] as f64,
                        terminated: true,
                        ..Default::default()
                    });
                    self.pending_autoreset[lane] = true;
                }
                // The terminal obs keeps the old id; an ongoing step keeps the
                // active id. Neither is a roll.
                episode_ids[lane] = id;
            }
        }

        Ok(RuntimeEnvStep {
            response: StepResponse {
                observation: Some(leaves_value(payload([0]))),
                rewards,
                terminated_mask,
                truncated_mask: vec![0u8; n],
                infos: None,
                completed_episodes,
                episode_ids,
                env_indices: vec![],
            },
            endpoint_total_ns: None,
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
        route_id: "route-vec".to_string(),
        env_component_id: "env-vec".to_string(),
        model_component_id: "model-vec".to_string(),
        env_id: "VectorTestEnv-v0".to_string(),
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
        base_seed: None,
        max_episodes: Some(max_episodes),
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
