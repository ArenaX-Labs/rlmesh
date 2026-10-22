//! Shared fake env/model/hook harness for the runtime driver's integration
//! tests. `runtime_driver.rs` asserts the runtime's behavior against it; the sealed
//! edition fingerprint in `runtime_edition_2026_06.rs` drives the same fakes, so both
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
use rlmesh::binding::model::{ModelHandler, ModelObservation, PredictFrames};
use rlmesh::binding::runtime::{
    ActionReceivedEvent, EndpointPhases, EnvResetRequest, EnvStepRequest, HookError, RuntimeDriver,
    RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep, RuntimeError, RuntimeHooks, RuntimeSessionSpec,
};
use rlmesh_proto::core::v1::{EnvContract, EnvSpec};
use rlmesh_proto::spaces::v1::{MetaMap, MetaValue, SpaceSpec, meta_value};
use rlmesh_spaces::{DType, SpaceValue, Tensor};

#[derive(Debug, thiserror::Error)]
#[error("simulated transport failure")]
pub struct FakeTransportError;

#[derive(Clone)]
pub struct TestEnv {
    pub closed: Arc<AtomicBool>,
    pub step_count: Arc<AtomicUsize>,
    pub reset_seeds: Arc<Mutex<Vec<Vec<i64>>>>,
    pub reset_options: Arc<Mutex<Vec<Option<rlmesh_spaces::MetaMap>>>>,
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
            terminal_after: 1,
            final_info: None,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        }
    }
}

#[async_trait]
impl RuntimeEnv for TestEnv {
    async fn reset(&mut self, request: EnvResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        self.reset_seeds
            .lock()
            .expect("reset seed recorder lock poisoned")
            .push(request.seeds);
        self.reset_options
            .lock()
            .expect("reset option recorder lock poisoned")
            .push(request.options);
        self.step_count.store(0, Ordering::SeqCst);
        Ok(RuntimeEnvReset {
            observation: vec![payload([1])],
            infos: Some(info_map("phase", "reset")),
            endpoint_total_ns: self.endpoint_total_ns,
            phases: self.phases,
        })
    }

    async fn step(&mut self, _request: EnvStepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        let step = self.step_count.fetch_add(1, Ordering::SeqCst) + 1;
        let terminal = step >= self.terminal_after;
        // The terminal step's info carries the episode's `final_info`, as a
        // Gymnasium env reports its outcome.
        let mut infos = info_map("phase", "step");
        if terminal && let Some(final_info) = &self.final_info {
            infos.entries.insert(
                "final_info".to_string(),
                MetaValue {
                    kind: Some(meta_value::Kind::Map(final_info.clone())),
                },
            );
        }
        Ok(RuntimeEnvStep {
            observation: vec![payload([step as u8])],
            rewards: vec![1.0],
            terminated: vec![terminal],
            truncated: vec![false],
            infos: Some(infos),
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
    // Set when the driver released the route at route end.
    pub closed: Arc<AtomicBool>,
    // Simulates a route release that blocks (an RPC on a hung connection)
    // without honoring any timeout.
    pub release_adapter_hangs: bool,
    pub predicts: Arc<AtomicUsize>,
    pub predict_delay: Option<Duration>,
    pub seen_observations: Arc<Mutex<Vec<Vec<u8>>>>,
    // Episode ids the driver asked the model to evict (R2), in order: one
    // `reset_adapter` call's episode ids per entry.
    pub reset_adapters: Arc<Mutex<Vec<Vec<String>>>>,
    // Every model call in arrival order: ("predict", the observation's episode
    // ids) or ("evict", the ids it ended) — the model-side episode lifecycle.
    pub lifecycle: Lifecycle,
    // Number of chunk replay frames to return per predict (frames after frame
    // 0, which is always present). 0 = not chunking.
    pub replay_frames: usize,
    // The route negotiated observation history: the driver then carries every
    // replayed step as a row on the next predict.
    pub wants_history: bool,
    // Per predict: the history rows' `(step, first observation byte)` and the
    // observation's own, in arrival order.
    pub ledger: Ledger,
    // Simulates an evict the model never answers: the call is recorded, then
    // pends forever.
    pub reset_adapter_hangs: bool,
    // The 1-based predict call that fails with a fatal model error (after its
    // lifecycle entry is recorded), failing the route mid-episode.
    pub fail_predict_at: Option<usize>,
    // What this model reports holding, and the adapter share of each predict.
    pub held: Option<(u64, u64)>,
    pub adapter_ns: u64,
    // Holds the `gate_at`th predict to arrive (0-based) until the gate opens;
    // every other predict returns at once.
    pub gate: Option<Shared<oneshot::Receiver<()>>>,
    pub gate_at: usize,
    pub arrived: Arc<AtomicUsize>,
    // One `()` per predict as it arrives, ahead of any delay or gate.
    pub arrivals: Option<mpsc::UnboundedSender<()>>,
}

impl TestModel {
    fn predict_one(&self, observation: &ModelObservation) -> rlmesh::Result<PredictFrames> {
        let call = self.predicts.fetch_add(1, Ordering::SeqCst) + 1;
        self.lifecycle
            .lock()
            .expect("lifecycle lock poisoned")
            .push(("predict", observation.episode_ids()));
        if self.fail_predict_at == Some(call) {
            return Err(rlmesh::Error::model(FakeTransportError.to_string()));
        }
        let first_byte = |leaves: Option<&Vec<Bytes>>| {
            leaves
                .and_then(|leaves| leaves.first())
                .and_then(|leaf| leaf.first().copied())
                .unwrap_or(0)
        };
        self.ledger.lock().expect("ledger poisoned").push((
            observation
                .history
                .iter()
                .map(|row| (row.step, first_byte(row.observation.as_ref())))
                .collect(),
            (
                observation.step.unwrap_or(-1),
                first_byte(observation.observation.as_ref()),
            ),
        ));
        let observation_bytes = observation
            .observation
            .as_ref()
            .and_then(|leaves| leaves.first())
            .map(|leaf| leaf.to_vec())
            .unwrap_or_default();
        self.seen_observations
            .lock()
            .expect("model observation recorder lock poisoned")
            .push(observation_bytes);
        // Frame 0 is this step; the replay frames are buffered and replayed
        // without re-calling the model.
        let frame = |byte: u8| vec![action(byte); observation.num_envs];
        Ok(PredictFrames {
            actions: frame(0),
            replay: (0..self.replay_frames)
                .map(|i| frame(100 + i as u8))
                .collect(),
        })
    }
}

#[async_trait]
impl ModelHandler for TestModel {
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        Ok(self.predict_chunked(observation).await?.actions)
    }

    async fn predict_chunked(
        &mut self,
        observation: ModelObservation,
    ) -> rlmesh::Result<PredictFrames> {
        self.predict_grouped(vec![observation])
            .await
            .pop()
            .expect("one group, one result")
    }

    async fn predict_grouped(
        &mut self,
        observations: Vec<ModelObservation>,
    ) -> Vec<rlmesh::Result<PredictFrames>> {
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
        observations.iter().map(|o| self.predict_one(o)).collect()
    }

    fn held_state(&self) -> Option<rlmesh::binding::model::HeldState> {
        self.held
            .map(|(episodes, bytes)| rlmesh::binding::model::HeldState { episodes, bytes })
    }

    fn take_adapter_ns(&mut self) -> u64 {
        self.adapter_ns
    }

    fn route_setup(&self) -> Option<Arc<dyn rlmesh::binding::model::ModelRouteSetup>> {
        Some(Arc::new(TestRoutes {
            closed: Arc::clone(&self.closed),
            hangs: self.release_adapter_hangs,
        }))
    }

    async fn reset_adapter(
        &mut self,
        _env_id: &str,
        episode_ids: Vec<String>,
    ) -> rlmesh::Result<()> {
        self.lifecycle
            .lock()
            .expect("lifecycle lock poisoned")
            .push(("evict", episode_ids.clone()));
        self.reset_adapters
            .lock()
            .expect("reset_adapter recorder lock poisoned")
            .push(episode_ids);
        if self.reset_adapter_hangs {
            // The call is on the record; now ignore any deadline entirely,
            // like an evict on a hung connection.
            std::future::pending::<()>().await;
        }
        Ok(())
    }
}

/// The test model's route setup: records the release at route end.
struct TestRoutes {
    closed: Arc<AtomicBool>,
    hangs: bool,
}

#[async_trait]
impl rlmesh::binding::model::ModelRouteSetup for TestRoutes {
    async fn resolve_adapter(
        &self,
        _env_id: &str,
        _env_contract: &rlmesh_spaces::EnvContract,
        _options: rlmesh::binding::model::ResolveOptions,
    ) -> rlmesh::Result<rlmesh::binding::model::RouteNeeds> {
        Ok(Default::default())
    }

    async fn release_adapter(&self, _env_id: &str) -> rlmesh::Result<()> {
        if self.hangs {
            // Ignore any deadline entirely, like a misbehaving release.
            std::future::pending::<()>().await;
        }
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }
}

/// The fakes' action: one `uint8 [1]` box.
pub fn action(byte: u8) -> SpaceValue {
    SpaceValue::Box(Tensor::from_vec(vec![byte], vec![1], DType::Uint8).expect("a uint8 box"))
}

/// The native contract the fakes' spaces type: `uint8 [1]` boxes.
pub fn contract(spec: &RuntimeSessionSpec) -> rlmesh_spaces::EnvContract {
    let space = rlmesh_spaces::spaces::BoxSpaceBuilder::scalar(0.0, 255.0, vec![1])
        .dtype(DType::Uint8)
        .build()
        .expect("a uint8 box space");
    rlmesh_spaces::EnvContract {
        id: spec.env_id.clone(),
        action_space: Some(space.clone()),
        observation_space: Some(space),
        metadata: None,
        render_mode: String::new(),
        num_envs: spec.num_envs as u32,
        autoreset_mode: Default::default(),
    }
}

/// A driver of `model` against `env` under `spec`.
pub fn new_driver<E>(
    spec: RuntimeSessionSpec,
    env: E,
    model: TestModel,
    hooks: Arc<dyn RuntimeHooks>,
) -> RuntimeDriver<E, TestModel>
where
    E: RuntimeEnv + Clone + 'static,
{
    let contract = contract(&spec);
    let history = model.wants_history;
    RuntimeDriver::new(spec, env, model, contract, hooks).with_history(history)
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
    pub infos: Option<MetaMap>,
}

/// One step event's (terminated, truncated) per lane and its autoreset-roll mark.
pub type StepFlags = (Vec<bool>, Vec<bool>, Vec<bool>);

#[derive(Default)]
pub struct RecordingHooks {
    pub actions: AtomicUsize,
    pub ended: AtomicUsize,
    pub failed: AtomicUsize,
    pub emitted_observations: Mutex<Vec<EmittedObservation>>,
    pub step_infos: Mutex<Vec<Option<MetaMap>>>,
    // Per step event: (terminated, truncated, autoreset_roll), each per lane.
    pub step_flags: Mutex<Vec<StepFlags>>,
    /// `(predict_ms, step_ms)` of every `step_completed` event, in order.
    pub step_timings: Mutex<Vec<(f64, f64)>>,
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

    async fn observation_emitted(
        &self,
        event: rlmesh::binding::runtime::ObservationEmittedEvent,
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
                infos: event.infos,
            });
        Ok(())
    }

    async fn step_completed(
        &self,
        event: rlmesh::binding::runtime::StepCompletedEvent,
    ) -> Result<(), HookError> {
        self.note_event("step", vec![event.episode_id]);
        self.step_infos
            .lock()
            .expect("step info recorder lock poisoned")
            .push(event.infos);
        self.step_flags
            .lock()
            .expect("step flag recorder lock poisoned")
            .push((event.terminated, event.truncated, event.autoreset_roll));
        self.step_timings
            .lock()
            .expect("step timing recorder lock poisoned")
            .push((event.predict_ms, event.step_ms));
        Ok(())
    }

    async fn episode_started(
        &self,
        event: rlmesh::binding::runtime::EpisodeStartedEvent,
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
        event: rlmesh::binding::runtime::EpisodeCompletedEvent,
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
        _event: rlmesh::binding::runtime::SessionEndedEvent,
    ) -> Result<(), HookError> {
        self.ended.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn on_telemetry(
        &self,
        event: rlmesh::binding::runtime::TelemetrySnapshotEvent,
    ) -> Result<(), HookError> {
        let rows = event.snapshot.rows.len();
        if event.snapshot.horizon == rlmesh::binding::runtime::telemetry::Horizon::Window {
            self.telemetry_windows.fetch_add(1, Ordering::SeqCst);
        } else if event.snapshot.horizon == rlmesh::binding::runtime::telemetry::Horizon::Session {
            self.telemetry_sessions.fetch_add(1, Ordering::SeqCst);
            self.telemetry_session_rows
                .fetch_max(rows, Ordering::SeqCst);
        }
        Ok(())
    }

    async fn session_failed(
        &self,
        _event: rlmesh::binding::runtime::SessionFailedEvent,
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
    }
}

pub fn payload<const N: usize>(data: [u8; N]) -> Bytes {
    Bytes::copy_from_slice(&data)
}

/// A NEXT_STEP vector env with a per-lane terminal schedule. It mimics the env
/// server's output: a lane terminates at its scheduled step (terminal obs keeps
/// the old episode id), then auto-resets on the FOLLOWING step (fresh obs, new
/// id, reward 0), never requiring a driver-issued reset.
#[derive(Clone)]
pub struct VectorTestEnv {
    pub reset_seeds: Arc<Mutex<Vec<Vec<i64>>>>,
    pub reset_options: Arc<Mutex<Vec<Option<rlmesh_spaces::MetaMap>>>>,
    pub closed: Arc<AtomicBool>,
    pub terminal_after: Vec<usize>,
    pub lane_step: Vec<usize>,
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
            pending_autoreset: vec![false; n],
        }
    }
}

#[async_trait]
impl RuntimeEnv for VectorTestEnv {
    async fn reset(&mut self, request: EnvResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
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
        Ok(RuntimeEnvReset {
            observation: vec![payload([0])],
            infos: None,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn step(&mut self, _request: EnvStepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        let n = self.terminal_after.len();
        let mut rewards = vec![1.0; n];
        let mut terminated = vec![false; n];
        let mut final_info = false;
        let mut rolled = false;

        for lane in 0..n {
            if self.pending_autoreset[lane] {
                rolled = true;
                // t+1: the env auto-resets this lane and delivers the fresh obs of
                // a new episode (step 0, reward 0, terminated=false).
                self.pending_autoreset[lane] = false;
                self.lane_step[lane] = 0;
                rewards[lane] = 0.0;
            } else {
                self.lane_step[lane] += 1;
                if self.lane_step[lane] >= self.terminal_after[lane] {
                    terminated[lane] = true;
                    // Lane 0 reports the outcome as Gymnasium's numeric
                    // `success: 0`, so the ledger pins a derived `false`
                    // (and, from the other lanes, an unreported outcome).
                    final_info |= lane == 0;
                    self.pending_autoreset[lane] = true;
                }
            }
        }

        let infos = vector_infos(if rolled { "roll" } else { "step" }, n, final_info);
        Ok(RuntimeEnvStep {
            observation: vec![payload([0])],
            rewards,
            terminated,
            truncated: vec![false; n],
            infos: Some(infos),
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }
}

/// A [`VectorTestEnv`] step's infos: its `phase` and, when lane 0 ended,
/// lane 0's final info (Gymnasium's numeric `success: 0`) in the vector
/// layout, masked to lane 0.
pub fn vector_infos(phase: &str, lanes: usize, lane0_ended: bool) -> MetaMap {
    let mut infos = info_map("phase", phase);
    if lane0_ended {
        let per_lane = |item: &dyn Fn(usize) -> meta_value::Kind| MetaValue {
            kind: Some(meta_value::Kind::List(rlmesh_proto::spaces::v1::MetaList {
                items: (0..lanes)
                    .map(|lane| MetaValue {
                        kind: Some(item(lane)),
                    })
                    .collect(),
            })),
        };
        infos.entries.insert(
            "final_info".to_string(),
            per_lane(&|lane| {
                meta_value::Kind::Map(if lane == 0 {
                    meta_map("success", meta_value::Kind::Integer(0))
                } else {
                    MetaMap::default()
                })
            }),
        );
        infos.entries.insert(
            "_final_info".to_string(),
            per_lane(&|lane| meta_value::Kind::Bool(lane == 0)),
        );
    }
    infos
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

/// The one subscriber this test binary logs to, installed on first use. A
/// thread-scoped `set_default` per capture raced tracing's callsite-interest
/// cache against every other test's first log line and could capture nothing
/// at all under load; a global subscriber is asked once and stays asked.
fn captured_logs() -> &'static Arc<Mutex<Vec<u8>>> {
    static LOGS: std::sync::OnceLock<Arc<Mutex<Vec<u8>>>> = std::sync::OnceLock::new();
    LOGS.get_or_init(|| {
        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(LogCapture(Arc::clone(&buffer)))
            .with_ansi(false)
            .with_thread_ids(true)
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("this binary installs no other global subscriber");
        buffer
    })
}

/// Drive a chunking run over `lanes` and hand back everything it logged.
pub async fn chunked_run_logs(lanes: Vec<usize>) -> String {
    // The tripwire is once per session, so two captures must not interleave.
    static ONE_CAPTURE_AT_A_TIME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let _serial = ONE_CAPTURE_AT_A_TIME.lock().await;
    let logs = captured_logs();
    // A callsite another test registered while the subscriber was being
    // installed may still carry the no-op dispatcher's interest.
    tracing::callsite::rebuild_interest_cache();
    let start = logs.lock().expect("log buffer poisoned").len();
    new_driver(
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
    let captured = logs.lock().expect("log buffer poisoned")[start..].to_vec();
    // Every other test in the binary logs to the same subscriber; the run above
    // is single-threaded, so its lines are the ones stamped with this thread.
    // Compared numerically: the formatter zero-pads (`ThreadId(02)`), Debug
    // does not.
    let this_thread = thread_number(&format!("{:?}", std::thread::current().id()));
    String::from_utf8(captured)
        .expect("captured logs are utf-8")
        .lines()
        .filter(|line| line.split("ThreadId(").nth(1).map(thread_number) == Some(this_thread))
        .map(|line| format!("{line}\n"))
        .collect()
}

fn thread_number(stamp: &str) -> u64 {
    stamp
        .trim_start_matches("ThreadId(")
        .split(')')
        .next()
        .and_then(|digits| digits.parse().ok())
        .expect("a ThreadId(N) stamp")
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
        entries: [(
            rlmesh::binding::runtime::ENV_RESET_OPTIONS_KEY.to_string(),
            options,
        )]
        .into(),
    });
}

/// The `trial_index` carried by one recorded `ResetRequest.options`, as the
/// integer a single-lane reset sends.
pub fn recorded_trial(options: &Option<rlmesh_spaces::MetaMap>) -> Option<i64> {
    match options.as_ref()?.get("trial_index")? {
        rlmesh_spaces::MetaValue::Int(value) => Some(*value),
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
}

impl LaneTestEnv {
    pub fn new(lanes: Vec<(usize, Duration)>) -> Self {
        let n = lanes.len();
        Self {
            inner: Arc::new(Mutex::new(LaneTestState {
                resets: Vec::new(),
                step: vec![0; n],
            })),
            lanes,
            closed: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl RuntimeEnv for LaneTestEnv {
    async fn reset(&mut self, request: EnvResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
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
        Ok(RuntimeEnvReset {
            observation: vec![payload([0])],
            infos: None,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn step(&mut self, request: EnvStepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
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
        Ok(RuntimeEnvStep {
            observation: vec![payload([0])],
            rewards: vec![1.0],
            terminated: vec![done],
            truncated: vec![false],
            infos: None,
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

/// A single-lane env that never terminates on its own: only the runtime's
/// caps end its episodes.
#[derive(Clone, Default)]
pub struct EndlessEnv {
    pub inner: Arc<Mutex<EndlessEnvState>>,
}

#[derive(Default)]
pub struct EndlessEnvState {
    pub resets: usize,
    pub steps: usize,
}

#[async_trait]
impl RuntimeEnv for EndlessEnv {
    async fn reset(&mut self, _request: EnvResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        self.inner.lock().expect("endless env lock poisoned").resets += 1;
        Ok(RuntimeEnvReset {
            observation: vec![payload([0])],
            infos: None,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn step(&mut self, _request: EnvStepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        self.inner.lock().expect("endless env lock poisoned").steps += 1;
        Ok(RuntimeEnvStep {
            observation: vec![payload([0])],
            rewards: vec![1.0],
            terminated: vec![false],
            truncated: vec![false],
            infos: None,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}
