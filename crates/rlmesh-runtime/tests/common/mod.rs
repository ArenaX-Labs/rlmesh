//! Shared fake env/model/hook harness for the `rlmesh-runtime` integration
//! tests. `driver.rs` asserts the runtime's behavior against it; the sealed
//! edition fingerprint in `edition_2026_06.rs` drives the same fakes, so both
//! test binaries observe one stimulus, not two drifting copies.
#![allow(dead_code)]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::future::Shared;
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

#[derive(Debug, thiserror::Error)]
#[error("simulated transport failure")]
pub struct FakeTransportError;

#[derive(Clone)]
pub struct TestEnv {
    pub closed: Arc<AtomicBool>,
    pub step_count: Arc<AtomicUsize>,
    pub reset_seeds: Arc<Mutex<Vec<Vec<i64>>>>,
    pub reset_options: Arc<Mutex<Vec<Option<MetaMap>>>>,
    // The runtime is the id authority: the env adopts the id pushed down on
    // reset and echoes it back in completed_episodes (never mints its own).
    pub current_episode: Arc<Mutex<String>>,
    pub terminal_after: usize,
    // What this env reports as the terminal episode's `final_info`. Gymnasium's
    // `is_success`/`success` live there, and the runtime derives the ledger's
    // `success` from them; `None` leaves that outcome unreported.
    pub final_info: Option<MetaMap>,
    // What this env claims to have spent on each op, as a peer would stamp it.
    pub endpoint_total_ns: Option<u64>,
    pub phases: EndpointPhases,
}

impl Default for TestEnv {
    fn default() -> Self {
        Self {
            closed: Arc::new(AtomicBool::new(false)),
            step_count: Arc::new(AtomicUsize::new(0)),
            reset_seeds: Arc::new(Mutex::new(Vec::new())),
            reset_options: Arc::new(Mutex::new(Vec::new())),
            current_episode: Arc::new(Mutex::new(String::new())),
            terminal_after: 1,
            final_info: None,
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
        self.reset_options
            .lock()
            .expect("reset option recorder lock poisoned")
            .push(request.options);
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
                        final_info: self.final_info.clone(),
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

/// Model calls in arrival order: `("predict" | "evict", episode ids)`.
pub type Lifecycle = Arc<Mutex<Vec<(&'static str, Vec<String>)>>>;
/// One predict request's history rows and its own observation, each as
/// `(step, first observation byte)`.
pub type LedgerEntry = (Vec<(i64, u8)>, (i64, u8));
/// One entry per predict request, in arrival order.
pub type Ledger = Arc<Mutex<Vec<LedgerEntry>>>;

#[derive(Clone, Default)]
pub struct TestModel {
    pub closed: Arc<AtomicBool>,
    pub predicts: Arc<AtomicUsize>,
    pub predict_delay: Option<Duration>,
    pub seen_observations: Arc<Mutex<Vec<Vec<u8>>>>,
    // Episode ids the driver asked the model to evict via ResetAdapter (R2), in
    // order — one ResetAdapterRequest's episode_ids per entry.
    pub reset_adapters: Arc<Mutex<Vec<Vec<String>>>>,
    // Every model call in arrival order: ("predict", the request's episode ids)
    // or ("evict", the ResetAdapter's ids) — the model-side episode lifecycle.
    pub lifecycle: Lifecycle,
    // Number of chunk replay frames to return per predict (frames 1.. of the
    // ordered `actions` list; frame 0 is always present). 0 = not chunking (a
    // single-frame `actions`, the unchanged path).
    pub replay_frames: usize,
    // The route negotiated observation history: the driver then carries every
    // replayed step as a row on the next predict.
    pub wants_history: bool,
    // Per predict request: the history rows' `(step, first observation byte)`
    // and the request's own, in arrival order.
    pub ledger: Ledger,
    // Simulates a release_adapter impl that blocks (e.g. an RPC on a hung
    // connection) without honoring the supplied timeout.
    pub release_adapter_hangs: bool,
    // Simulates a reset_adapter (evict) RPC the model never answers: the call
    // is recorded, then pends forever.
    pub reset_adapter_hangs: bool,
    // The 1-based predict call that fails with a fatal model RPC error (after
    // its lifecycle entry is recorded), failing the route mid-episode.
    pub fail_predict_at: Option<usize>,
    // What this model claims to have spent on each predict, as a peer would
    // stamp it.
    pub endpoint_total_ns: Option<u64>,
    pub phases: EndpointPhases,
    // Holds the `gate_at`th predict to arrive (0-based) until the gate opens;
    // every other predict returns at once.
    pub gate: Option<Shared<oneshot::Receiver<()>>>,
    pub gate_at: usize,
    pub arrived: Arc<AtomicUsize>,
    // One `()` per predict as it arrives, ahead of any delay or gate.
    pub arrivals: Option<mpsc::UnboundedSender<()>>,
}

#[async_trait]
impl RuntimeModel for TestModel {
    async fn predict(
        &self,
        request: PredictRequest,
    ) -> Result<RuntimeModelPrediction, RuntimeError> {
        if let Some(arrivals) = &self.arrivals {
            let _ = arrivals.unbounded_send(());
        }
        if let Some(gate) = &self.gate
            && self.arrived.fetch_add(1, Ordering::SeqCst) == self.gate_at
        {
            let _ = gate.clone().await;
        }
        if let Some(delay) = self.predict_delay {
            tokio::time::sleep(delay).await;
        }
        let call = self.predicts.fetch_add(1, Ordering::SeqCst) + 1;
        self.lifecycle
            .lock()
            .expect("lifecycle lock poisoned")
            .push((
                "predict",
                request
                    .episode_info
                    .iter()
                    .map(|e| e.episode_id.clone())
                    .collect(),
            ));
        if self.fail_predict_at == Some(call) {
            return Err(RuntimeError::model_rpc("model-1", FakeTransportError));
        }
        let first_byte = |value: Option<&SpaceValue>| {
            value
                .and_then(|value| value.leaves.first())
                .and_then(|leaf| leaf.first().copied())
                .unwrap_or(0)
        };
        self.ledger.lock().expect("ledger poisoned").push((
            request
                .history
                .iter()
                .map(|row| (row.step, first_byte(row.observation.as_ref())))
                .collect(),
            (
                request.step.unwrap_or(-1),
                first_byte(request.observation.as_ref()),
            ),
        ));
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

    fn wants_history(&self) -> bool {
        self.wants_history
    }

    async fn reset_adapter(&self, request: ResetAdapterRequest) -> Result<(), RuntimeError> {
        self.lifecycle
            .lock()
            .expect("lifecycle lock poisoned")
            .push(("evict", request.episode_ids.clone()));
        self.reset_adapters
            .lock()
            .expect("reset_adapter recorder lock poisoned")
            .push(request.episode_ids);
        if self.reset_adapter_hangs {
            // The call is on the record; now ignore any deadline entirely,
            // like an evict RPC on a hung connection.
            std::future::pending::<()>().await;
        }
        Ok(())
    }

    async fn release_adapter(
        &self,
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

/// A one-entry `MetaMap`, the shape both step infos and an episode's
/// `final_info` take in these fakes.
pub fn meta_map(key: &str, kind: meta_value::Kind) -> MetaMap {
    MetaMap {
        entries: [(key.to_string(), MetaValue { kind: Some(kind) })].into(),
    }
}

pub fn info_map(key: &str, value: &str) -> MetaMap {
    meta_map(key, meta_value::Kind::Text(value.to_string()))
}

#[derive(Debug, Clone)]
pub struct EmittedObservation {
    pub episode_ids: Vec<String>,
    pub observation: Vec<u8>,
    pub raw_observation: Vec<u8>,
    pub infos: Option<MetaMap>,
}

#[derive(Default)]
pub struct RecordingHooks {
    pub actions: AtomicUsize,
    pub ended: AtomicUsize,
    pub failed: AtomicUsize,
    pub fail_action_transform: bool,
    // When set, transform_observation prepends this marker byte to every
    // observation it forwards to the model.
    pub observation_marker: Option<u8>,
    pub emitted_observations: Mutex<Vec<EmittedObservation>>,
    pub step_infos: Mutex<Vec<Option<MetaMap>>>,
    pub started_seeds: Mutex<Vec<Option<i64>>>,
    pub completed_seeds: Mutex<Vec<Option<i64>>>,
    pub started_trials: Mutex<Vec<Option<u64>>>,
    pub completed_trials: Mutex<Vec<Option<u64>>>,
    // Every per-episode hook event in arrival order, as (kind, the episode ids
    // it names): the hook-side episode lifecycle.
    pub events: Mutex<Vec<(&'static str, Vec<String>)>>,
    // Counts of live telemetry snapshots streamed via on_telemetry, by horizon.
    pub telemetry_windows: AtomicUsize,
    pub telemetry_sessions: AtomicUsize,
    // Largest row count seen in any Session snapshot — proves the final push
    // carried real telemetry, not an empty event.
    pub telemetry_session_rows: AtomicUsize,
}

impl RecordingHooks {
    pub fn note_event(&self, kind: &'static str, ids: Vec<String>) {
        self.events
            .lock()
            .expect("event recorder lock poisoned")
            .push((kind, ids));
    }

    /// The kinds of the events that named `episode_id`, in arrival order.
    pub fn events_for(&self, episode_id: &str) -> Vec<&'static str> {
        self.events
            .lock()
            .expect("event recorder lock poisoned")
            .iter()
            .filter(|(_, ids)| ids.iter().any(|id| id == episode_id))
            .map(|(kind, _)| *kind)
            .collect()
    }

    /// Every episode id an `episode_completed` event named, in arrival order.
    pub fn completed_ids(&self) -> Vec<String> {
        self.events
            .lock()
            .expect("event recorder lock poisoned")
            .iter()
            .filter(|(kind, _)| *kind == "completed")
            .flat_map(|(_, ids)| ids.clone())
            .collect()
    }
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
        self.note_event("observation", event.episode_ids.clone());
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
        self.note_event("step", vec![event.episode_id]);
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
        self.note_event("started", vec![event.episode_id]);
        self.started_seeds
            .lock()
            .expect("started seed recorder lock poisoned")
            .push(event.seed);
        self.started_trials
            .lock()
            .expect("started trial recorder lock poisoned")
            .push(event.trial_index);
        Ok(())
    }

    async fn episode_completed(
        &self,
        event: rlmesh_runtime::EpisodeCompletedEvent,
    ) -> Result<(), HookError> {
        self.note_event("completed", vec![event.episode_id]);
        self.completed_seeds
            .lock()
            .expect("completed seed recorder lock poisoned")
            .push(event.seed);
        self.completed_trials
            .lock()
            .expect("completed trial recorder lock poisoned")
            .push(event.trial_index);
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

pub fn one_episode_spec() -> RuntimeSessionSpec {
    RuntimeSessionSpec {
        session_id: "session-1".to_string(),
        env_id: "TestEnv-v0".to_string(),
        env_component_id: "env-1".to_string(),
        model_component_id: "model-1".to_string(),
        workflow_edition: rlmesh_proto::Edition::E2026_06,
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
        trial_index_base: None,
        max_episode_steps: None,
        max_episode_seconds: None,
        close_env_on_end: true,
        subset_step: false,
        limits: Default::default(),
        env_ceiling: None,
        model_ceiling: None,
    }
}

pub fn payload<const N: usize>(data: [u8; N]) -> Bytes {
    Bytes::copy_from_slice(&data)
}

pub fn leaves_value(data: Bytes) -> SpaceValue {
    SpaceValue { leaves: vec![data] }
}

/// A NEXT_STEP vector env with a per-lane terminal schedule. It mimics the env
/// server's output: a lane terminates at its scheduled step (terminal obs keeps
/// the old episode id), then auto-resets on the FOLLOWING step (fresh obs, new
/// id, reward 0), never requiring a driver-issued reset.
#[derive(Clone)]
pub struct VectorTestEnv {
    pub reset_seeds: Arc<Mutex<Vec<Vec<i64>>>>,
    pub reset_options: Arc<Mutex<Vec<Option<MetaMap>>>>,
    pub closed: Arc<AtomicBool>,
    pub terminal_after: Vec<usize>,
    pub lane_step: Vec<usize>,
    // The runtime is the id authority: each lane adopts the id pushed down on
    // reset / on the autoreset roll, and echoes it in completed_episodes.
    pub current_ids: Vec<String>,
    pub pending_autoreset: Vec<bool>,
}

impl VectorTestEnv {
    pub fn new(terminal_after: Vec<usize>) -> Self {
        let n = terminal_after.len();
        Self {
            reset_seeds: Arc::new(Mutex::new(Vec::new())),
            reset_options: Arc::new(Mutex::new(Vec::new())),
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
        self.reset_options
            .lock()
            .expect("reset option recorder lock poisoned")
            .push(request.options);
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
                        // Lane 0 reports the outcome as Gymnasium's numeric
                        // `success: 0`, so the ledger pins a derived `false`
                        // (and, from the other lanes, an unreported outcome).
                        final_info: (lane == 0)
                            .then(|| meta_map("success", meta_value::Kind::Integer(0))),
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

pub fn vector_spec(num_envs: usize, max_episodes: u64) -> RuntimeSessionSpec {
    RuntimeSessionSpec {
        session_id: "session-vec".to_string(),
        env_id: "VectorTestEnv-v0".to_string(),
        env_component_id: "env-vec".to_string(),
        model_component_id: "model-vec".to_string(),
        workflow_edition: rlmesh_proto::Edition::E2026_06,
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
        trial_index_base: None,
        max_episode_steps: None,
        max_episode_seconds: None,
        close_env_on_end: true,
        subset_step: false,
        limits: Default::default(),
        env_ceiling: None,
        model_ceiling: None,
    }
}

/// Every predict's history rows followed by its own observation, as steps,
/// in arrival order: the sequence of steps the model was handed.
pub fn delivered_steps(ledger: &[LedgerEntry]) -> Vec<i64> {
    ledger
        .iter()
        .flat_map(|(rows, own)| rows.iter().map(|row| row.0).chain([own.0]))
        .collect()
}

/// The ledger split per episode, by the episode id each predict named, in
/// first-predict order.
pub fn ledger_per_episode(model: &TestModel) -> Vec<(String, Vec<LedgerEntry>)> {
    let ledger = model.ledger.lock().expect("ledger poisoned").clone();
    let predicts: Vec<String> = model
        .lifecycle
        .lock()
        .expect("lifecycle lock poisoned")
        .iter()
        .filter(|(call, _)| *call == "predict")
        .map(|(_, ids)| ids.join(","))
        .collect();
    assert_eq!(predicts.len(), ledger.len());
    let mut episodes: Vec<(String, Vec<_>)> = Vec::new();
    for (id, entry) in predicts.into_iter().zip(ledger) {
        match episodes.iter_mut().find(|(known, _)| *known == id) {
            Some((_, entries)) => entries.push(entry),
            None => episodes.push((id, vec![entry])),
        }
    }
    episodes
}

/// A `tracing` writer that appends everything into a shared buffer, so a test can
/// assert on a `warn!` that has no other observable effect.
#[derive(Clone)]
pub struct LogCapture(pub Arc<Mutex<Vec<u8>>>);

impl std::io::Write for LogCapture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .expect("log buffer poisoned")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogCapture {
    type Writer = LogCapture;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// Drive a chunking run over `lanes` and hand back everything it logged.
pub async fn chunked_run_logs(lanes: Vec<usize>) -> String {
    let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(LogCapture(Arc::clone(&buffer)))
        .with_ansi(false)
        .finish();
    {
        let _guard = tracing::subscriber::set_default(subscriber);
        RuntimeDriver::new(
            vector_spec(lanes.len(), 4),
            VectorTestEnv::new(lanes.clone()),
            TestModel {
                replay_frames: 4,
                ..Default::default()
            },
            Arc::new(RecordingHooks::default()),
        )
        .run()
        .await
        .expect("the chunked run completes");
    }
    String::from_utf8(buffer.lock().expect("log buffer poisoned").clone())
        .expect("captured logs are utf-8")
}

/// A spec for `episodes` back-to-back single-lane episodes whose env declares
/// `reset_options = declared`.
pub fn trial_spec(episodes: u64, declared: &[&str]) -> RuntimeSessionSpec {
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(episodes);
    declare_reset_options(&mut spec, declared);
    spec
}

/// Stamp `reset_options = declared` onto the spec's env contract metadata, the
/// way `EnvFactory.make()` publishes the declaration.
pub fn declare_reset_options(spec: &mut RuntimeSessionSpec, declared: &[&str]) {
    if declared.is_empty() {
        return;
    }
    let options = MetaValue {
        kind: Some(meta_value::Kind::List(rlmesh_proto::spaces::v1::MetaList {
            items: declared
                .iter()
                .map(|key| MetaValue {
                    kind: Some(meta_value::Kind::Text((*key).to_string())),
                })
                .collect(),
        })),
    };
    spec.env_contract.spec.as_mut().expect("env spec").metadata = Some(MetaMap {
        entries: [(rlmesh_runtime::ENV_RESET_OPTIONS_KEY.to_string(), options)].into(),
    });
}

/// The `trial_index` carried by one recorded `ResetRequest.options`, as the
/// integer a single-lane reset sends.
pub fn recorded_trial(options: &Option<MetaMap>) -> Option<i64> {
    match options
        .as_ref()?
        .entries
        .get("trial_index")?
        .kind
        .as_ref()?
    {
        meta_value::Kind::Integer(value) => Some(*value),
        _ => None,
    }
}

/// A lane-capable env handle: clones share one endpoint, every op names one
/// lane, and each lane has its own episode length and step latency so lanes
/// finish episodes in a timing-dependent order.
#[derive(Clone)]
pub struct LaneTestEnv {
    pub inner: Arc<Mutex<LaneTestState>>,
    pub lanes: Vec<(usize, Duration)>,
    pub closed: Arc<AtomicUsize>,
}

#[derive(Default)]
pub struct LaneTestState {
    /// `(lane, seed)` per reset, in the order the env saw them.
    pub resets: Vec<(u32, Option<i64>)>,
    pub step: Vec<usize>,
    pub ids: Vec<String>,
}

impl LaneTestEnv {
    pub fn new(lanes: Vec<(usize, Duration)>) -> Self {
        let n = lanes.len();
        Self {
            inner: Arc::new(Mutex::new(LaneTestState {
                resets: Vec::new(),
                step: vec![0; n],
                ids: vec![String::new(); n],
            })),
            lanes,
            closed: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl RuntimeEnv for LaneTestEnv {
    async fn reset(&mut self, request: ResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        assert_eq!(
            request.env_indices.len(),
            1,
            "lane resets name exactly one lane"
        );
        let lane = request.env_indices[0] as usize;
        let mut state = self.inner.lock().expect("lane env state lock poisoned");
        state
            .resets
            .push((lane as u32, request.seeds.first().copied()));
        state.step[lane] = 0;
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
        assert_eq!(
            request.env_indices.len(),
            1,
            "lane steps name exactly one lane"
        );
        let lane = request.env_indices[0] as usize;
        let (length, latency) = self.lanes[lane];
        tokio::time::sleep(latency).await;
        let mut state = self.inner.lock().expect("lane env state lock poisoned");
        state.step[lane] += 1;
        let done = state.step[lane] >= length;
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
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

pub fn lane_spec(num_envs: usize, max_episodes: u64, seeds: Vec<i64>) -> RuntimeSessionSpec {
    RuntimeSessionSpec {
        env_contract: EnvContract {
            autoreset_mode: rlmesh_proto::core::v1::AutoresetMode::Disabled as i32,
            ..vector_spec(num_envs, max_episodes).env_contract
        },
        episode_seeds: seeds,
        subset_step: true,
        ..vector_spec(num_envs, max_episodes)
    }
}

/// A single-lane env that never terminates on its own and mimics the env
/// server's interrupted-episode buffer: a reset arriving while an episode is
/// still active buffers that episode, and the NEXT `StepResponse` reports it in
/// `completed_episodes`. Every runtime-side truncation resets mid-episode, so
/// every truncated episode comes back as a stale echo one step later.
#[derive(Clone, Default)]
pub struct EchoingEnv {
    pub inner: Arc<Mutex<EchoingEnvState>>,
}

#[derive(Default)]
pub struct EchoingEnvState {
    pub resets: usize,
    pub steps: usize,
    pub step_in_episode: i64,
    pub current_id: String,
    pub interrupted: Vec<EpisodeMetadata>,
}

#[async_trait]
impl RuntimeEnv for EchoingEnv {
    async fn reset(&mut self, request: ResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        let mut state = self.inner.lock().expect("echoing env lock poisoned");
        if state.step_in_episode > 0 {
            let interrupted = EpisodeMetadata {
                episode_id: state.current_id.clone(),
                env_index: 0,
                step_count: state.step_in_episode,
                cumulative_reward: state.step_in_episode as f64,
                truncated: true,
                ..Default::default()
            };
            state.interrupted.push(interrupted);
        }
        state.resets += 1;
        state.step_in_episode = 0;
        state.current_id = request.episode_ids.first().cloned().unwrap_or_default();
        Ok(RuntimeEnvReset {
            response: ResetResponse {
                observation: Some(leaves_value(payload([0]))),
                infos: None,
            },
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn step(&mut self, _request: StepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        let mut state = self.inner.lock().expect("echoing env lock poisoned");
        state.steps += 1;
        state.step_in_episode += 1;
        let completed_episodes = std::mem::take(&mut state.interrupted);
        Ok(RuntimeEnvStep {
            response: StepResponse {
                observation: Some(leaves_value(payload([0]))),
                rewards: vec![1.0],
                terminated_mask: vec![0],
                truncated_mask: vec![0],
                infos: None,
                completed_episodes,
                env_indices: vec![],
            },
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}
