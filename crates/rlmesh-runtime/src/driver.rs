//! The single-route run loop.
//!
//! Drives one ready model/env session: reset, then predict/step until a step or
//! episode limit, cancellation, or failure. Records per-op telemetry and fans
//! every state change out to the session's
//! [`RuntimeHooks`](crate::hooks::RuntimeHooks).

use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use prost::{Message, bytes::Bytes};
use rlmesh_proto::core::v1::AutoresetMode;
use rlmesh_proto::env::v1::{
    EpisodeMetadata, ResetRequest, ResetResponse, StepRequest, StepResponse,
};
use rlmesh_proto::model::v1::{
    PredictRequest, PredictResponse, ReleaseAdapterRequest, ResetAdapterRequest,
};
use rlmesh_proto::spaces::v1::SpaceValue;
use tokio_util::sync::CancellationToken;

use crate::hooks::{
    ActionReceivedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, LogEvent, LogLevel,
    ObservationEmittedEvent, RuntimeEnvContext, RuntimeHooks, SessionEndedEvent,
    SessionFailedEvent, SessionStartedEvent, StepCompletedEvent, TelemetrySnapshotEvent,
};
use crate::spec::{RuntimeReport, RuntimeSessionSpec};
use crate::state::{RequestPhase, RouteSnapshot, RouteState, StartedEpisode};
use crate::telemetry::{Aggregator, Horizon, Sample, Source, metrics};

mod error;

pub use error::RuntimeError;

/// Sends `$event` to the best-effort hook `$method`, logging any failure and
/// keeping the route moving.
macro_rules! fan_out_event {
    ($self:ident, $method:ident, $event:expr) => {
        if let Err(err) = $self.hooks.$method($event).await {
            tracing::warn!(
                concat!("runtime hook ", stringify!($method), " failed: {}"),
                err
            );
        }
    };
}

/// The env's reset reply plus the endpoint-local op duration the peer stamped.
pub struct RuntimeEnvReset {
    pub response: ResetResponse,
    /// Endpoint-local op duration (ns) from `JoinResponse.endpoint_total_ns`
    /// (replaces the old nested per-step telemetry message).
    pub endpoint_total_ns: Option<u64>,
}

/// The env's step reply plus the endpoint-local op duration the peer stamped.
pub struct RuntimeEnvStep {
    pub response: StepResponse,
    /// Endpoint-local op duration (ns) from `JoinResponse.endpoint_total_ns`.
    pub endpoint_total_ns: Option<u64>,
}

/// The model's predict reply plus the endpoint-local op duration the peer stamped.
pub struct RuntimeModelPrediction {
    pub response: PredictResponse,
    /// Endpoint-local op duration (ns) from `JoinResponse.endpoint_total_ns`.
    pub endpoint_total_ns: Option<u64>,
}

/// The environment side of a route: reset and step over the wire, plus close.
#[async_trait]
pub trait RuntimeEnv: Send {
    /// Reset the requested lanes and return their initial observation.
    async fn reset(&mut self, request: ResetRequest) -> Result<RuntimeEnvReset, RuntimeError>;

    /// Advance the lanes one step under `request.action`.
    async fn step(&mut self, request: StepRequest) -> Result<RuntimeEnvStep, RuntimeError>;

    /// Release the env endpoint within `timeout`. Default no-op for an endpoint
    /// the run does not own.
    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}

/// The model side of a route: predict, plus the per-episode adapter-state
/// lifecycle (evict on episode end, release at session end).
#[async_trait]
pub trait RuntimeModel: Send {
    /// Predict the ordered action frames for the batched observation (frame 0 is
    /// this step; any further frames replay open-loop before the next call).
    async fn predict(
        &mut self,
        request: PredictRequest,
    ) -> Result<RuntimeModelPrediction, RuntimeError>;

    /// Evict the model's per-episode adapter state (frame-stack buffers) for the
    /// ended episodes. Best-effort GC, not a correctness gate: because episode
    /// ids never repeat (UUIDv7), a dropped ResetAdapter only leaks memory and
    /// can never alias a new episode. Default no-op for impls that hold no
    /// per-episode state.
    async fn reset_adapter(&mut self, _request: ResetAdapterRequest) -> Result<(), RuntimeError> {
        Ok(())
    }

    /// Release the model endpoint within `timeout`. Default no-op.
    async fn release_adapter(
        &mut self,
        _request: ReleaseAdapterRequest,
        _timeout: Duration,
    ) -> Result<(), String> {
        Ok(())
    }
}

/// Default reason attributed to a cancellation when the caller does not supply
/// one via [`RuntimeDriver::run_with_cancellation_reason`].
const DEFAULT_CANCELLATION_REASON: &str = "cancelled by caller";

// Telemetry sources for the three driver ops. `component` is a coarse class
// label — the serial single-route driver has one model + one env, and `op`
// already distinguishes them (see telemetry::Source).
const SRC_PREDICT: Source = Source {
    op: "model.predict",
    component: "model",
};
const SRC_STEP: Source = Source {
    op: "env.step",
    component: "env",
};
const SRC_RESET: Source = Source {
    op: "env.reset",
    component: "env",
};

/// Drives one ready model/env session through its `reset -> predict -> step`
/// loop. Inert until a `run*` method is awaited.
#[must_use = "a RuntimeDriver does nothing until one of its run methods is awaited"]
pub struct RuntimeDriver<E, M> {
    spec: RuntimeSessionSpec,
    env: E,
    model: M,
    hooks: Arc<dyn RuntimeHooks>,
    cancellation_reason: String,
    /// Action/observation space specs shared into every per-step hook event.
    /// Populated once after [`validate`](RuntimeSessionSpec::validate) so the
    /// hot path clones an `Arc` instead of deep-copying the spec each step.
    action_space: Arc<rlmesh_proto::spaces::v1::SpaceSpec>,
    observation_space: Arc<rlmesh_proto::spaces::v1::SpaceSpec>,
}

impl<E, M> RuntimeDriver<E, M>
where
    E: RuntimeEnv,
    M: RuntimeModel,
{
    pub fn new(spec: RuntimeSessionSpec, env: E, model: M, hooks: Arc<dyn RuntimeHooks>) -> Self {
        Self {
            spec,
            env,
            model,
            hooks,
            cancellation_reason: DEFAULT_CANCELLATION_REASON.to_string(),
            // Filled from the validated spec at run time; default until then.
            action_space: Arc::default(),
            observation_space: Arc::default(),
        }
    }

    fn reset_seeds(&self, reset_generation: u64) -> Vec<i64> {
        self.seeds_for(reset_generation, 0..self.spec.num_envs)
    }

    /// Deterministic seeds for a partial (`reset_subset`) reset, positionally
    /// aligned to `env_indices`. Empty when no base seed is configured.
    fn reset_subset_seeds(&self, reset_generation: u64, env_indices: &[u32]) -> Vec<i64> {
        self.seeds_for(
            reset_generation,
            env_indices.iter().map(|&index| index as usize),
        )
    }

    /// Deterministic per-lane reset seeds for `env_indices`, positionally
    /// aligned. Empty when no base seed is configured.
    fn seeds_for(
        &self,
        reset_generation: u64,
        env_indices: impl Iterator<Item = usize>,
    ) -> Vec<i64> {
        let Some(base_seed) = self.spec.base_seed else {
            return Vec::new();
        };
        env_indices
            .map(|env_index| {
                deterministic_reset_seed(
                    base_seed,
                    &self.spec.session_id,
                    reset_generation,
                    env_index,
                )
            })
            .collect()
    }

    /// Per-lane autoreset convention declared by the served env's contract.
    /// `UNSPECIFIED` is treated as `DISABLED` (explicit reset only).
    fn autoreset_mode(&self) -> AutoresetMode {
        // Unknown modes are rejected at RuntimeSessionSpec::validate() (run before
        // the loop); a value that still fails to decode falls back to the safe
        // explicit-reset DISABLED rather than silently aliasing a newer mode.
        AutoresetMode::try_from(self.spec.env_contract.autoreset_mode)
            .unwrap_or(AutoresetMode::Disabled)
    }

    pub async fn run(self) -> Result<RuntimeReport, RuntimeError> {
        self.run_with_cancellation(CancellationToken::new()).await
    }

    pub async fn run_with_cancellation(
        self,
        cancellation: CancellationToken,
    ) -> Result<RuntimeReport, RuntimeError> {
        self.run_with_cancellation_reason(cancellation, DEFAULT_CANCELLATION_REASON)
            .await
    }

    /// Runs the session, attributing any cancellation of `cancellation` to
    /// `reason`.
    ///
    /// The reason is carried into [`RuntimeError::RouteCancelled`], the
    /// `session_failed` hook event, and the `CloseRoute` reason, so callers
    /// (e.g. an owner that cancels for Ctrl+C, a deadline, or a sibling-route
    /// failure) can supply an accurate cause instead of a hardcoded one.
    pub async fn run_with_cancellation_reason(
        mut self,
        cancellation: CancellationToken,
        reason: impl Into<String>,
    ) -> Result<RuntimeReport, RuntimeError> {
        self.cancellation_reason = reason.into();
        self.spec.validate().map_err(RuntimeError::InvalidSpec)?;
        // validate() confirmed both spaces are present; cache them as shared
        // Arcs so per-step hook events clone a pointer, not the whole spec.
        self.action_space = Arc::new(self.spec.action_space_validated().clone());
        self.observation_space = Arc::new(self.spec.observation_space_validated().clone());
        let mut state = RouteState::new(&self.spec);
        // Telemetry lives here, not in run_loop, so the final Session snapshot is
        // delivered on EVERY exit (including aborts). The background ticker only
        // ever pushes Window snapshots (the live tier); the cumulative Session
        // total is pushed once below and returned on the report (the durable
        // tier), so a late ticker tick cannot race or supersede it.
        let telemetry = Arc::new(Mutex::new(Aggregator::default()));
        // A zero window disables live streaming (it would otherwise be a 1ms hot
        // loop); the final session push below still fires.
        let ticker = (!self.spec.limits.telemetry_window.is_zero()).then(|| {
            TelemetryTicker::spawn(
                Arc::clone(&telemetry),
                Arc::clone(&self.hooks),
                self.spec.limits.telemetry_window,
                state.session_id().to_string(),
                state.env_context(),
            )
        });
        let result = self.run_loop(&mut state, &cancellation, &telemetry).await;
        // Stop the ticker (it only emits Window snapshots, so it cannot contend
        // this Session push), then deliver the durable session total exactly once
        // on every exit path.
        drop(ticker);
        let final_snapshot = lock_agg(&telemetry).snapshot(Horizon::Session);
        fan_out_event!(
            self,
            on_telemetry,
            TelemetrySnapshotEvent {
                session_id: state.session_id().to_string(),
                route: state.env_context(),
                snapshot: final_snapshot,
            }
        );
        if let Err(error) = &result {
            self.shutdown_after_failure(&mut state, error).await;
        }
        result
    }

    /// Session/route-level span (enabling-only): lets a closed-side OTel
    /// subscriber attach to `rlmesh.route` later; inert under the default
    /// subscriber. Created once per session, not per step; `skip_all` records only
    /// the cheap ids. Any future per-step span MUST be trace-level + target-gated.
    #[tracing::instrument(
        name = "rlmesh.route",
        level = "info",
        skip_all,
        fields(
            session_id = %state.session_id(),
            env_id = %self.spec.env_id,
            num_envs = self.spec.num_envs,
        ),
    )]
    async fn run_loop(
        &mut self,
        state: &mut RouteState,
        cancellation: &CancellationToken,
        telemetry: &Arc<Mutex<Aggregator>>,
    ) -> Result<RuntimeReport, RuntimeError> {
        fan_out_event!(
            self,
            session_started,
            SessionStartedEvent {
                session_id: state.session_id().to_string(),
                route: state.env_context(),
                env_id: self.spec.env_id.clone(),
            }
        );

        let mut reset_generation = 0_u64;
        let reset_timeout = self.spec.limits.env_reset_timeout;
        // Spec timeout getter returns a clamped-non-negative i64; proto field is uint64.
        let reset_timeout_ms = self.spec.limits.env_reset_timeout_ms().max(0) as u64;
        let reset_seeds = self.reset_seeds(reset_generation);
        // The runtime is the sole id authority (R1): mint a fresh UUIDv7 per lane
        // and push them DOWN so the env tags its episodes with our ids; we never
        // read ids back from the env.
        let initial_episode_ids = mint_episode_ids(self.spec.num_envs);
        let reset_request = ResetRequest {
            seeds: reset_seeds,
            options: None,
            timeout_ms: reset_timeout_ms,
            env_indices: Vec::new(),
            episode_ids: initial_episode_ids.clone(),
        };
        let reset_request_bytes = reset_request.encoded_len() as u64;
        // Time only the RPC (after building the request), matching the predict /
        // step / in-loop-reset sites so rpc.total is consistent across ops.
        let reset_started = Instant::now();
        let reset_ok = await_runtime_operation(
            cancellation,
            reset_timeout,
            RuntimeError::operation_timeout(
                state.env_id(),
                state.env_component_id(),
                "env.reset",
                0,
                reset_timeout,
            ),
            self.cancelled_error(state, 0),
            self.env.reset(reset_request),
        )
        .await?;
        let reset_latency = reset_started.elapsed();
        record_op(
            telemetry,
            SRC_RESET,
            reset_latency,
            reset_ok.endpoint_total_ns,
            reset_request_bytes,
            reset_ok.response.encoded_len() as u64,
        );
        fan_out_event!(
            self,
            log,
            LogEvent {
                session_id: state.session_id().to_string(),
                route: state.env_context(),
                level: LogLevel::Info,
                message: format!(
                    "env reset complete in {:.0}ms ({} episode(s) ready)",
                    reset_latency.as_secs_f64() * 1000.0,
                    initial_episode_ids.len()
                ),
                source: Some("runtime".to_string()),
            }
        );

        let reset_observation = value_leaves(reset_ok.response.observation.as_ref())?;
        let started_episodes = state.start_episodes(initial_episode_ids, false);
        self.invoke_started_episodes(state, started_episodes).await;

        // Runtime-side mirror of the env's NEXT_STEP autoreset expectation: maps a
        // lane that completed this step to the fresh UUIDv7 we minted for its next
        // episode. Set on completion (step t), consumed on the autoreset roll
        // (step t+1) — both the down-push to the env and our own slot roll read
        // from here, so the env tags the rolled episode with exactly our id.
        let mut pending_roll: std::collections::HashMap<u32, String> =
            std::collections::HashMap::new();

        let mut reset_msg =
            state.predict_request(reset_observation.clone(), RequestPhase::ResetObservation);
        let mut reset_event =
            self.observation_event(state, state.snapshot(), true, reset_observation.clone());
        let transformed_reset_observation = self
            .invoke_transform_observation(reset_event.clone())
            .await?;
        reset_event.observation = transformed_reset_observation.clone();
        reset_msg.observation = transformed_reset_observation.map(leaves_value);
        fan_out_event!(self, observation_emitted, reset_event);

        let mut pending_observation_msg = reset_msg;

        // Runtime-owned action-chunk replay buffer. A predict returns its ordered
        // action frames in `PredictResponse.actions` (frame 0 = this step, frames
        // 1.. = open-loop replay); the driver pushes every frame here (whole-batch
        // frames, one `SpaceValue`'s leaves per step, covering every lane), pops one
        // per step, and re-calls the model only when the buffer drains. A
        // non-chunking predict returns exactly one frame, so the predict-every-step
        // path is unchanged.
        let mut replay_buffer: std::collections::VecDeque<Vec<Bytes>> =
            std::collections::VecDeque::new();

        loop {
            if cancellation.is_cancelled() {
                return Err(self.cancelled_error(state, state.snapshot().step));
            }

            let predict_snapshot = state.snapshot();
            // Re-call model.predict only when the replay buffer is empty; otherwise
            // replay a buffered chunk frame (no RPC, no obs re-encode). A predict
            // returns its ordered frames in `actions` (frame 0 = this step, frames
            // 1.. = open-loop replay): push every frame, then pop one. env.step +
            // the observation transform/emit below run every iteration regardless,
            // so a replay step still feeds the observation ledger.
            if replay_buffer.is_empty() {
                let predict_timeout = self.spec.limits.model_predict_timeout;
                let expected_context = pending_observation_msg.context.clone();
                let predict_request_bytes = pending_observation_msg.encoded_len() as u64;
                let predict_started = Instant::now();
                let action_msg = await_runtime_operation(
                    cancellation,
                    predict_timeout,
                    RuntimeError::operation_timeout(
                        state.env_id(),
                        state.model_component_id(),
                        "model.predict",
                        predict_snapshot.step,
                        predict_timeout,
                    ),
                    self.cancelled_error(state, predict_snapshot.step),
                    self.model.predict(pending_observation_msg),
                )
                .await?;
                let predict_rpc = predict_started.elapsed();
                if action_msg.response.context != expected_context {
                    let request_id = expected_context
                        .as_ref()
                        .map(|context| context.request_id.clone())
                        .unwrap_or_default();
                    return Err(RuntimeError::ModelRouteMismatch {
                        component_id: state.model_component_id().to_string(),
                        request_id,
                    });
                }
                record_op(
                    telemetry,
                    SRC_PREDICT,
                    predict_rpc,
                    action_msg.endpoint_total_ns,
                    predict_request_bytes,
                    action_msg.response.encoded_len() as u64,
                );
                if action_msg.response.actions.is_empty() {
                    return Err(RuntimeError::Protocol(format!(
                        "model endpoint {} returned a predict response with no actions",
                        state.model_component_id()
                    )));
                }
                // Push every ordered frame (frame 0 = this step, frames 1.. =
                // open-loop replay); the pop below applies frame 0 now and the rest
                // on the following steps without re-calling the model.
                for frame in &action_msg.response.actions {
                    if let Some(leaves) = value_leaves(Some(frame))? {
                        replay_buffer.push_back(leaves);
                    }
                }
            }
            let model_action = replay_buffer
                .pop_front()
                .expect("replay buffer is non-empty after a refill");

            let action_step = predict_snapshot.step + 1;
            let mut action_event = ActionReceivedEvent {
                session_id: state.session_id().to_string(),
                route: state.env_context(),
                episode_id: predict_snapshot.episode_id.clone(),
                episode_record_id: predict_snapshot.episode_record_id.clone(),
                episode_ids: predict_snapshot.episode_ids.clone(),
                episode_record_ids: predict_snapshot.episode_record_ids.clone(),
                step: action_step,
                env_index: predict_snapshot.env_index,
                action_space: Arc::clone(&self.action_space),
                action: Some(model_action),
            };
            action_event.action = self.invoke_transform_action(action_event.clone()).await?;
            fan_out_event!(self, action_received, action_event.clone());

            let step_timeout = self.spec.limits.env_step_timeout;
            // Spec timeout getter returns a clamped-non-negative i64; proto field is uint64.
            let step_timeout_ms = self.spec.limits.env_step_timeout_ms().max(0) as u64;
            // Down-push the authoritative per-lane ids. For a NEXT_STEP autoreset
            // roll (a lane in `pending_roll`), substitute the freshly minted id so
            // the env tags its rolled episode with our id; the slot itself rolls to
            // the same id below, after the step.
            let step_episode_ids = episode_ids_with_roll(state.episode_ids(), &pending_roll);
            let step_request = StepRequest {
                action: action_event.action.map(leaves_value),
                timeout_ms: step_timeout_ms,
                env_indices: Vec::new(),
                episode_ids: step_episode_ids,
            };
            let step_request_bytes = step_request.encoded_len() as u64;
            let step_started = Instant::now();
            let step_ok = await_runtime_operation(
                cancellation,
                step_timeout,
                RuntimeError::operation_timeout(
                    state.env_id(),
                    state.env_component_id(),
                    "env.step",
                    action_step,
                    step_timeout,
                ),
                self.cancelled_error(state, action_step),
                self.env.step(step_request),
            )
            .await?;
            let step_rpc = step_started.elapsed();
            record_op(
                telemetry,
                SRC_STEP,
                step_rpc,
                step_ok.endpoint_total_ns,
                step_request_bytes,
                step_ok.response.encoded_len() as u64,
            );
            let step_observation = value_leaves(step_ok.response.observation.as_ref())?;

            state.record_step();
            let step_snapshot = state.snapshot();
            fan_out_event!(
                self,
                step_completed,
                StepCompletedEvent {
                    session_id: state.session_id().to_string(),
                    route: state.env_context(),
                    episode_id: step_snapshot.episode_id.clone(),
                    episode_record_id: step_snapshot.episode_record_id.clone(),
                    step: step_snapshot.step,
                    env_index: step_snapshot.env_index,
                    rewards: step_ok.response.rewards.clone(),
                }
            );

            // Apply any NEXT_STEP autoreset roll the env just performed. The env
            // rolled the lanes in `pending_roll` (from the previous step's
            // completions) using the ids we pushed above; mirror that here by
            // rolling our own slots to the same ids. The env no longer mints or
            // returns ids — the runtime is authoritative.
            if !pending_roll.is_empty() {
                let roll_ids = episode_ids_with_roll(state.episode_ids(), &pending_roll);
                pending_roll.clear();
                let started_episodes = state.observe_episode_ids(roll_ids);
                self.invoke_started_episodes(state, started_episodes).await;
            }
            // The observation_emitted hook fires once per observation actually
            // sent to the model, post-transform, below (or at the initial
            // reset). Emitting the raw step observation here would expose
            // pre-transform bytes and, when the episode completes, an
            // observation the model never sees.

            self.emit_completed_episodes(state, &step_ok.response.completed_episodes)
                .await;
            // Tell the model to evict the ended episodes' frame-stack buffers
            // (best-effort GC; ids never repeat so a miss only leaks memory).
            self.emit_reset_adapter(state, &step_ok.response.completed_episodes)
                .await;

            // A lane that completed this step gets a fresh episode, so its buffered
            // future actions are stale. The replay buffer holds whole-batch frames
            // and cannot be partially invalidated, so flush it and re-plan on the
            // next step (receding horizon on reset). No-op when not chunking.
            if !step_ok.response.completed_episodes.is_empty() {
                replay_buffer.clear();
            }

            // Under NEXT_STEP, a lane that completed this step (t) autoresets at
            // t+1. Mint its next id now and stash it; the next step's down-push and
            // slot roll both consume it. Mirrors the env's `expect_autoreset`.
            if matches!(
                self.autoreset_mode(),
                AutoresetMode::NextStep | AutoresetMode::SameStep
            ) {
                for completed in &step_ok.response.completed_episodes {
                    pending_roll
                        .entry(completed.env_index)
                        .or_insert_with(mint_episode_id);
                }
            }

            // Under NEXT_STEP, the final episode completes at its done step `t`
            // and this early-return fires before the `t+1` roll. So the
            // model-side `on_episode_end` for the final episode is not delivered
            // at `t+1`; instead it fires via the close-time `finish_lifecycle`
            // sweep during shutdown. Asymmetric versus mid-run episodes, but the
            // callback is not lost.
            if self
                .spec
                .max_episodes
                .is_some_and(|limit| state.total_episodes() >= limit as i64)
            {
                let release_request = state.release_adapter_request("completed requested episodes");
                self.shutdown_terminal_route(
                    state,
                    "completed requested episodes",
                    release_request,
                )
                .await;
                // The single final session push is delivered by the epilogue in
                // run_with_cancellation_reason (on every exit path); here we only
                // capture the durable pull snapshot for the returned report.
                let telemetry_snapshot = lock_agg(telemetry).snapshot(Horizon::Session);
                fan_out_event!(
                    self,
                    session_ended,
                    SessionEndedEvent {
                        session_id: state.session_id().to_string(),
                        route: state.env_context(),
                        reason: "completed requested episodes".to_string(),
                        total_steps: state.total_steps(),
                        total_episodes: state.total_episodes(),
                    }
                );
                return Ok(RuntimeReport {
                    session_id: state.session_id().to_string(),
                    env_id: self.spec.env_id.clone(),
                    total_steps: state.total_steps(),
                    total_episodes: state.total_episodes(),
                    telemetry: telemetry_snapshot,
                });
            }

            // Mode-aware next observation. The reflexive "any lane completed =>
            // reset the whole vector" trigger is gone. That was the category
            // error that cut healthy lanes short.
            let (next_obs, phase, is_reset_msg) = match self.autoreset_mode() {
                // NEXT_STEP (and the unreachable SAME_STEP): the env auto-resets a
                // done lane itself and the rolled episode ids already arrived via
                // observe_episode_ids above. The driver is purely observational;
                // it never resets on the hot path.
                AutoresetMode::NextStep | AutoresetMode::SameStep => (
                    step_observation.clone(),
                    RequestPhase::StepObservation,
                    false,
                ),
                // DISABLED (and the single-env default): the env does not
                // autoreset, so restart the lanes that just completed. When every
                // lane completed this is a whole-vector reset (also the num_envs==1
                // path); a strict subset uses a per-lane seeded reset_subset, the
                // controlled / reproducible path.
                AutoresetMode::Unspecified | AutoresetMode::Disabled => {
                    // Proto env_index is uint32; thread it straight into the
                    // uint32 ResetRequest.env_indices without a round-trip.
                    let mut done_lanes: Vec<u32> = step_ok
                        .response
                        .completed_episodes
                        .iter()
                        .map(|metadata| metadata.env_index)
                        .collect();
                    // completed_episodes can carry duplicate env_index entries
                    // (e.g. drained interrupted episodes), which would inflate
                    // the lane count and misfire the whole_vector decision below.
                    // Dedupe so the count reflects distinct lanes. Sorting is
                    // safe: reset_subset_seeds is derived FROM done_lanes (so
                    // seeds stay positionally aligned) and env_indices is
                    // done_lanes.clone().
                    done_lanes.sort_unstable();
                    done_lanes.dedup();
                    if done_lanes.is_empty() {
                        (
                            step_observation.clone(),
                            RequestPhase::StepObservation,
                            false,
                        )
                    } else {
                        reset_generation += 1;
                        let step = state.snapshot().step;
                        let reset_timeout = self.spec.limits.env_reset_timeout;
                        // Spec timeout getter returns a clamped-non-negative i64; proto field is uint64.
                        let reset_timeout_ms =
                            self.spec.limits.env_reset_timeout_ms().max(0) as u64;
                        let whole_vector = done_lanes.len() == self.spec.num_envs;
                        // Mint the authoritative ids for the lanes this reset
                        // restarts (full-width for a whole reset, aligned to
                        // done_lanes for a partial one) and push them DOWN.
                        let (reset_seeds, env_indices, reset_episode_ids) = if whole_vector {
                            (
                                self.reset_seeds(reset_generation),
                                Vec::new(),
                                mint_episode_ids(self.spec.num_envs),
                            )
                        } else {
                            (
                                self.reset_subset_seeds(reset_generation, &done_lanes),
                                done_lanes.clone(),
                                mint_episode_ids(done_lanes.len()),
                            )
                        };
                        let reset_request = ResetRequest {
                            seeds: reset_seeds,
                            options: None,
                            timeout_ms: reset_timeout_ms,
                            env_indices,
                            episode_ids: reset_episode_ids.clone(),
                        };
                        let reset_request_bytes = reset_request.encoded_len() as u64;
                        let inloop_reset_started = Instant::now();
                        let reset_ok = await_runtime_operation(
                            cancellation,
                            reset_timeout,
                            RuntimeError::operation_timeout(
                                state.env_id(),
                                state.env_component_id(),
                                "env.reset",
                                step,
                                reset_timeout,
                            ),
                            self.cancelled_error(state, step),
                            self.env.reset(reset_request),
                        )
                        .await?;
                        record_op(
                            telemetry,
                            SRC_RESET,
                            inloop_reset_started.elapsed(),
                            reset_ok.endpoint_total_ns,
                            reset_request_bytes,
                            reset_ok.response.encoded_len() as u64,
                        );
                        let next_obs = value_leaves(reset_ok.response.observation.as_ref())?;
                        // Whole-vector reset starts every lane; a partial reset
                        // rolls only the lanes it restarted (build a full-width id
                        // vector: current ids with the reset lanes replaced).
                        let started_episodes = if whole_vector {
                            state.start_episodes(reset_episode_ids, true)
                        } else {
                            let mut full = state.episode_ids();
                            for (lane, id) in done_lanes.iter().zip(reset_episode_ids) {
                                if let Some(slot) = full.get_mut(*lane as usize) {
                                    *slot = id;
                                }
                            }
                            state.observe_episode_ids(full)
                        };
                        self.invoke_started_episodes(state, started_episodes).await;
                        (next_obs, RequestPhase::ResetObservation, true)
                    }
                }
            };

            let mut obs_msg = state.predict_request(next_obs.clone(), phase);
            let mut outgoing_observation_event =
                self.observation_event(state, state.snapshot(), is_reset_msg, next_obs);
            let transformed_observation = self
                .invoke_transform_observation(outgoing_observation_event.clone())
                .await?;
            outgoing_observation_event.observation = transformed_observation.clone();
            obs_msg.observation = transformed_observation.map(leaves_value);
            // Emit the transformed observation actually sent to the model, for
            // both step and reset observations, so hooks always see the same
            // payload model.predict receives.
            fan_out_event!(self, observation_emitted, outgoing_observation_event);

            pending_observation_msg = obs_msg;
        }
    }

    async fn shutdown_after_failure(&mut self, state: &mut RouteState, error: &RuntimeError) {
        let reason = error.to_string();
        let request = state.release_adapter_request(reason.clone());
        self.shutdown_terminal_route(state, &reason, request).await;

        if let Err(err) = self
            .hooks
            .session_failed(SessionFailedEvent {
                session_id: state.session_id().to_string(),
                route: state.env_context(),
                reason,
            })
            .await
        {
            tracing::warn!("runtime hook session_failed failed: {err}");
        }
    }

    async fn shutdown_terminal_route(
        &mut self,
        state: &RouteState,
        reason: &str,
        request: ReleaseAdapterRequest,
    ) {
        let timeout = self.spec.limits.service_close_timeout;
        // The timeout is also forwarded to the impls, but the driver enforces
        // it independently: a close impl that blocks (e.g. an RPC on a hung
        // connection) without honoring the deadline must not be able to hang
        // run()/run_with_cancellation() forever during shutdown.
        let model_close =
            tokio::time::timeout(timeout, self.model.release_adapter(request, timeout));
        if self.spec.close_env_on_end {
            let env_close = tokio::time::timeout(timeout, self.env.close(timeout));
            let (env_result, model_result) = tokio::join!(env_close, model_close);
            match env_result {
                Ok(Err(err)) => {
                    tracing::warn!(error = %err, "environment close failed during route shutdown");
                }
                Err(_) => {
                    tracing::warn!(
                        timeout_ms = timeout.as_millis(),
                        "environment close timed out during route shutdown; abandoning close"
                    );
                }
                Ok(Ok(())) => {}
            }
            log_model_close_result(model_result, reason, timeout);
            return;
        }

        tracing::debug!(
            env_id = %state.env_id(),
            reason,
            "skipping environment close for adapter; endpoint remains owned by the run"
        );
        log_model_close_result(model_close.await, reason, timeout);
    }

    fn cancelled_error(&self, state: &RouteState, step: i64) -> RuntimeError {
        RuntimeError::route_cancelled(state.env_id(), step, self.cancellation_reason.as_str())
    }

    async fn invoke_started_episodes(&self, state: &RouteState, episodes: Vec<StartedEpisode>) {
        for episode in episodes {
            let record = &episode.record;
            fan_out_event!(
                self,
                episode_started,
                EpisodeStartedEvent {
                    session_id: state.session_id().to_string(),
                    route: state.env_context(),
                    episode_id: episode.episode_id.clone(),
                    episode_record_id: record.record_id.clone(),
                    episode_index: record.index,
                    env_index: record.env_index,
                    started_from_auto_reset: record.started_from_auto_reset,
                }
            );
        }
    }

    async fn emit_completed_episodes(&self, state: &mut RouteState, episodes: &[EpisodeMetadata]) {
        for completed in episodes {
            let record = state.complete_episode(&completed.episode_id);
            let episode_record_id = record
                .as_ref()
                .map(|record| record.record_id.clone())
                .unwrap_or_default();
            // Proto env_index/duration_ms are uint32/uint64; events are i32/i64.
            let env_index = i32::try_from(completed.env_index).unwrap_or(i32::MAX);
            fan_out_event!(
                self,
                episode_completed,
                EpisodeCompletedEvent {
                    session_id: state.session_id().to_string(),
                    route: state.env_context(),
                    episode_id: completed.episode_id.clone(),
                    episode_record_id,
                    episode_index: record.as_ref().map_or(0, |record| record.index),
                    env_index,
                    step_count: completed.step_count,
                    cumulative_reward: completed.cumulative_reward,
                    terminated: completed.terminated,
                    truncated: completed.truncated,
                    duration_ms: i64::try_from(completed.duration_ms).unwrap_or(i64::MAX),
                    final_info: completed.final_info.clone(),
                }
            );
        }
    }

    /// Tell the model to evict the ended episodes' per-episode adapter state.
    /// Best-effort GC (R2): a failure is logged and the route keeps moving — a
    /// missed evict only leaks model memory, never corrupts state (ids never
    /// repeat). Skips the call when nothing completed.
    ///
    /// The id to evict is resolved POSITIONALLY from the runtime's own slot by
    /// `env_index` (decision A) — the env's `completed_episodes[].episode_id`
    /// echo is never trusted as the authority. At the completion step the slot
    /// still holds the completing id (the autoreset roll lands at t+1), so this
    /// is the id the model lazily seeded and must drop.
    async fn emit_reset_adapter(&mut self, state: &mut RouteState, episodes: &[EpisodeMetadata]) {
        let slot_ids = state.episode_ids();
        let episode_ids: Vec<String> = episodes
            .iter()
            .filter_map(|completed| slot_ids.get(completed.env_index as usize).cloned())
            .filter(|id| !id.is_empty())
            .collect();
        if episode_ids.is_empty() {
            return;
        }
        let request = state.reset_adapter_request(episode_ids);
        if let Err(err) = self.model.reset_adapter(request).await {
            tracing::warn!("model reset_adapter (evict) failed: {err}");
        }
    }

    async fn invoke_transform_action(
        &self,
        event: ActionReceivedEvent,
    ) -> Result<Option<Vec<Bytes>>, RuntimeError> {
        match self.hooks.transform_action(event).await {
            Ok(action) => Ok(action),
            Err(err) => {
                tracing::warn!("runtime hook transform_action failed: {err}");
                Err(RuntimeError::Hook(err))
            }
        }
    }

    async fn invoke_transform_observation(
        &self,
        event: ObservationEmittedEvent,
    ) -> Result<Option<Vec<Bytes>>, RuntimeError> {
        match self.hooks.transform_observation(event).await {
            Ok(observation) => Ok(observation),
            Err(err) => {
                tracing::warn!("runtime hook transform_observation failed: {err}");
                Err(RuntimeError::Hook(err))
            }
        }
    }

    fn observation_event(
        &self,
        state: &RouteState,
        snapshot: RouteSnapshot,
        is_reset: bool,
        observation: Option<Vec<Bytes>>,
    ) -> ObservationEmittedEvent {
        ObservationEmittedEvent {
            session_id: state.session_id().to_string(),
            route: state.env_context(),
            episode_id: snapshot.episode_id,
            episode_record_id: snapshot.episode_record_id,
            episode_ids: snapshot.episode_ids,
            episode_record_ids: snapshot.episode_record_ids,
            step: snapshot.step,
            env_index: snapshot.env_index,
            is_reset,
            num_envs: self.spec.num_envs as u32,
            observation_space: Arc::clone(&self.observation_space),
            observation,
        }
    }
}

/// The per-lane reset seed, derived purely from reproducible inputs: the user's
/// base_seed, the session id, the reset generation, and the lane index. The
/// container env_id is deliberately NOT mixed in — it is a per-attach random
/// UUIDv7, so including it would make a base_seed non-reproducible across runs.
fn deterministic_reset_seed(
    base_seed: i64,
    session_id: &str,
    reset_generation: u64,
    env_index: usize,
) -> i64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    fn update(mut hash: u64, bytes: &[u8]) -> u64 {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    let mut hash = FNV_OFFSET;
    hash = update(hash, &base_seed.to_le_bytes());
    hash = update(hash, &[0xff]);
    hash = update(hash, session_id.as_bytes());
    hash = update(hash, &[0xfd]);
    hash = update(hash, &reset_generation.to_le_bytes());
    hash = update(hash, &[0xfc]);
    hash = update(hash, &(env_index as u64).to_le_bytes());
    (hash & i64::MAX as u64) as i64
}

async fn await_runtime_operation<T, F>(
    cancellation: &CancellationToken,
    timeout: Duration,
    timeout_error: RuntimeError,
    cancelled_error: RuntimeError,
    operation: F,
) -> Result<T, RuntimeError>
where
    F: Future<Output = Result<T, RuntimeError>>,
{
    tokio::select! {
        _ = cancellation.cancelled() => Err(cancelled_error),
        result = tokio::time::timeout(timeout, operation) => match result {
            Ok(result) => result,
            Err(_) => Err(timeout_error),
        },
    }
}

fn log_model_close_result(
    result: Result<Result<(), String>, tokio::time::error::Elapsed>,
    reason: &str,
    timeout: Duration,
) {
    match result {
        Ok(Err(err)) => {
            tracing::warn!(
                error = %err,
                reason,
                "model route close failed during route shutdown; relying on owner shutdown"
            );
        }
        Err(_) => {
            tracing::warn!(
                timeout_ms = timeout.as_millis(),
                reason,
                "model route close timed out during route shutdown; relying on owner shutdown"
            );
        }
        Ok(Ok(())) => {}
    }
}

fn leaves_value(leaves: Vec<Bytes>) -> SpaceValue {
    SpaceValue { leaves }
}

/// Mint one authoritative episode id. UUIDv7 is time-ordered (sortable by
/// creation) and never repeats, so a missed ResetAdapter can only leak memory —
/// never alias a fresh episode.
fn mint_episode_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

/// Mint `count` fresh episode ids (one per lane being started).
fn mint_episode_ids(count: usize) -> Vec<String> {
    (0..count).map(|_| mint_episode_id()).collect()
}

/// Current per-lane ids with the pending NEXT_STEP autoreset rolls substituted
/// in. Lanes not rolling keep their current id; a rolling lane takes its freshly
/// minted next id. Used for both the env down-push and our own slot roll so they
/// stay byte-identical.
fn episode_ids_with_roll(
    mut ids: Vec<String>,
    pending_roll: &std::collections::HashMap<u32, String>,
) -> Vec<String> {
    // Consume the already-allocated id vector and roll in place; an empty
    // `pending_roll` (the steady-state common case) is then zero-copy.
    for (env_index, new_id) in pending_roll {
        if let Some(slot) = ids.get_mut(*env_index as usize) {
            *slot = new_id.clone();
        }
    }
    ids
}

/// The relay is content-blind: it carries the peer's leaf vector through
/// unchanged (structure/dtype live in the route spec, never inline). Kept
/// returning `Result` so the existing `?` call sites are untouched.
fn value_leaves(payload: Option<&SpaceValue>) -> Result<Option<Vec<Bytes>>, RuntimeError> {
    Ok(payload.map(|payload| payload.leaves.clone()))
}

/// Locks the telemetry aggregator, recovering from a poisoned mutex instead of
/// panicking. Telemetry is best-effort and must never take down the route, so a
/// panic under the guard degrades telemetry rather than killing the session.
fn lock_agg(telemetry: &Mutex<Aggregator>) -> MutexGuard<'_, Aggregator> {
    telemetry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Record the four per-op telemetry samples — RPC latency, the optional
/// endpoint-local duration the peer stamped, and request + response wire bytes —
/// under a single lock. Every driver op (predict, step, reset) records this same
/// shape, so they all route through here.
fn record_op(
    telemetry: &Mutex<Aggregator>,
    src: Source,
    rpc: Duration,
    endpoint_total_ns: Option<u64>,
    request_bytes: u64,
    response_bytes: u64,
) {
    let mut agg = lock_agg(telemetry);
    agg.record(Sample::dur(src, metrics::RPC_TOTAL, rpc));
    if let Some(ns) = endpoint_total_ns {
        agg.record(Sample::dur(
            src,
            metrics::ENDPOINT_TOTAL,
            Duration::from_nanos(ns),
        ));
    }
    agg.record(Sample::bytes(src, metrics::REQUEST_BYTES, request_bytes));
    agg.record(Sample::bytes(src, metrics::RESPONSE_BYTES, response_bytes));
}

/// Background wall-clock telemetry emitter. On a fixed real-time cadence it
/// snapshots the aggregator's Window horizon and pushes it to the hooks — so live
/// Window deltas keep arriving even while the run loop is parked in a stalled
/// predict/step/reset (which a step-gated path cannot see). It does NOT push
/// Session snapshots: the cumulative session total is the durable tier, delivered
/// once by the run epilogue and on `RuntimeReport.telemetry`. Empty windows (no
/// samples since the last flush) are skipped. Aborts when the returned handle is
/// dropped; because it only ever emits Window snapshots, a late tick can never
/// race the epilogue's authoritative Session push.
struct TelemetryTicker {
    handle: tokio::task::JoinHandle<()>,
}

impl TelemetryTicker {
    fn spawn(
        telemetry: Arc<Mutex<Aggregator>>,
        hooks: Arc<dyn RuntimeHooks>,
        window: Duration,
        session_id: String,
        route: RuntimeEnvContext,
    ) -> Self {
        // The caller skips spawning for a zero window (disabled live streaming).
        // Defensive floor for any sub-ms value: interval panics on a zero period.
        let period = window.max(Duration::from_millis(1));
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(period);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            ticker.tick().await; // the first tick is immediate; skip it
            loop {
                ticker.tick().await;
                // Snapshot + clear the Window horizon under a scoped lock; the
                // guard is NEVER held across the await below (keeps the std Mutex
                // sound + the task future Send).
                let window_snap = {
                    let mut agg = lock_agg(&telemetry);
                    let snap = agg.snapshot(Horizon::Window);
                    agg.flush_window();
                    snap
                };
                // Nothing recorded this window — skip the push rather than emit an
                // empty snapshot to consumers.
                if window_snap.rows.is_empty() {
                    continue;
                }
                // Tag the snapshot with the route/session it belongs to (one
                // shared hooks instance serves all concurrent routes).
                let window_event = TelemetrySnapshotEvent {
                    session_id: session_id.clone(),
                    route: route.clone(),
                    snapshot: window_snap,
                };
                if let Err(err) = hooks.on_telemetry(window_event).await {
                    tracing::warn!("runtime hook on_telemetry (window) failed: {err}");
                }
            }
        });
        Self { handle }
    }
}

impl Drop for TelemetryTicker {
    fn drop(&mut self) {
        self.handle.abort();
    }
}
