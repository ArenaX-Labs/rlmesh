//! The single-route run loop.
//!
//! One driver drives one route. The route's lanes form *groups*: one group per
//! lane for a lane endpoint (the env advertised `subset_step`), one group for
//! the whole vector otherwise. Every group is its own `reset -> predict ->
//! step` episode loop, advanced as its own ops complete, so a slow step or
//! reset in one group never stalls another. Groups waiting for a prediction
//! are planned by the [`PredictScheduler`]: a model that fuses predicts gets
//! the plan as one grouped predict, any other gets one predict per group, and
//! every result is applied as it lands while later groups go out behind it.
//!
//! Records per-op telemetry and fans every state change out to the session's
//! [`RuntimeHooks`](crate::run::runtime::hooks::RuntimeHooks).

use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{FuturesUnordered, StreamExt};
use rlmesh_proto::EndpointPhases;
use rlmesh_proto::core::v1::AutoresetMode;
use rlmesh_proto::env::v1::EpisodeMetadata;
use rlmesh_proto::spaces::v1::MetaMap;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::model::handler::{ModelHandler, ModelResetOptions, PredictFrames};
use crate::model::types::{EpisodeInfo, HistoryFrame, ModelObservation};
use crate::run::runtime::hooks::{
    ActionReceivedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, HookError, LogEvent, LogLevel,
    ObservationEmittedEvent, RunDirective, RuntimeEnvContext, RuntimeHooks, SessionEndedEvent,
    SessionFailedEvent, SessionStartedEvent, StepCompletedEvent, TelemetrySnapshotEvent,
};
use crate::run::runtime::seeding::episode_seed;
use crate::run::runtime::spec::{
    ENV_RESET_OPTIONS_KEY, RuntimeReport, RuntimeSessionSpec, reset_options_for,
};
use crate::run::runtime::state::{RequestPhase, RouteSnapshot, RouteState, StartedEpisode};
use crate::run::runtime::telemetry::{Aggregator, Horizon, Sample, Source, metrics};

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

/// A reset of the requested lanes (all of them when `env_indices` is
/// empty): their seeds and options, positional to the lanes.
pub struct EnvResetRequest {
    pub env_indices: Vec<u32>,
    pub seeds: Vec<i64>,
    pub options: Option<rlmesh_spaces::MetaMap>,
}

/// One step of the requested lanes (all of them when `env_indices` is
/// empty): one action per lane as leaf slabs.
pub struct EnvStepRequest {
    pub env_indices: Vec<u32>,
    pub action: Vec<Bytes>,
}

/// The env's answer to a reset: the lanes' observation as leaf slabs, and
/// what the env measured answering.
pub struct RuntimeEnvReset {
    pub observation: Vec<Bytes>,
    pub infos: Option<MetaMap>,
    pub endpoint_total_ns: Option<u64>,
    pub phases: EndpointPhases,
}

/// The env's answer to a step: the lanes' next observation, one reward and
/// done flags per lane, and what the env measured answering.
pub struct RuntimeEnvStep {
    pub observation: Vec<Bytes>,
    pub rewards: Vec<f64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub infos: Option<MetaMap>,
    pub endpoint_total_ns: Option<u64>,
    pub phases: EndpointPhases,
}

/// What a peer reported about its own handling of one op, as recorded beside the
/// driver's own RPC timing.
pub(crate) struct PeerReport {
    pub(crate) endpoint_total_ns: Option<u64>,
    pub(crate) phases: EndpointPhases,
    pub(crate) group_size: Option<u64>,
}

/// The environment side of a route: reset and step, plus close.
///
/// The driver holds one clone per group (a lane endpoint is driven with one
/// request per lane in flight), so an implementation is a handle onto the
/// env session: clones share the session and multiplex their requests.
#[async_trait]
pub trait RuntimeEnv: Send {
    /// Reset the requested lanes and return their initial observation.
    async fn reset(&mut self, request: EnvResetRequest) -> Result<RuntimeEnvReset, RuntimeError>;

    /// Advance the lanes one step under `request.action`.
    async fn step(&mut self, request: EnvStepRequest) -> Result<RuntimeEnvStep, RuntimeError>;

    /// Release the env endpoint within `timeout`. Default no-op for an endpoint
    /// the run does not own.
    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}

/// One group's prediction: its ordered action frames as leaf slabs (frame 0
/// is this step; any further frames replay open-loop before the next call),
/// and what the model measured.
struct Prediction {
    frames: Vec<Vec<Bytes>>,
    endpoint_total_ns: Option<u64>,
    phases: EndpointPhases,
}

/// Decides which groups waiting for a prediction go out next (as one grouped
/// predict for a model that fuses them, one predict each otherwise). Consulted
/// whenever at least one group is waiting and the model can take more; `busy`
/// is how many groups are still stepping or resetting. An
/// empty plan waits for the next event (a step or reset completing) — the
/// driver forces a full plan when nothing else is in flight, so a scheduler
/// cannot stall the route.
pub trait PredictScheduler: Send + Sync {
    fn plan(&mut self, waiting: &[usize], busy: usize) -> Vec<usize>;
}

/// The default policy: predict for every waiting group right away. Groups
/// that become ready while the model is busy form the next batch.
pub struct EagerScheduler;

impl PredictScheduler for EagerScheduler {
    fn plan(&mut self, waiting: &[usize], _busy: usize) -> Vec<usize> {
        waiting.to_vec()
    }
}

/// Default reason attributed to a cancellation when the caller does not supply
/// one via [`RuntimeDriver::run_with_cancellation_reason`].
const DEFAULT_CANCELLATION_REASON: &str = "cancelled by caller";

/// The env-reported task outcome from an episode's final-step info, read under
/// the `keys` the session's edition governs
/// ([`EditionDefaults::success_info_keys`](rlmesh_proto::EditionDefaults::success_info_keys)),
/// in priority order: the first key holding a bool, integer, or number decides
/// (numbers by truthiness, `1`/`1.0` → true), and keys of any other kind are
/// skipped; `None` when no key decides. The Python Session's `_episode_success`
/// applies the same rule so the two loops report identical success.
fn success_from_final_info(
    keys: &[&str],
    final_info: Option<&rlmesh_proto::spaces::v1::MetaMap>,
) -> Option<bool> {
    use rlmesh_proto::spaces::v1::meta_value::Kind;
    let entries = &final_info?.entries;
    keys.iter()
        .find_map(|key| match entries.get(*key)?.kind.as_ref()? {
            Kind::Bool(value) => Some(*value),
            Kind::Integer(value) => Some(*value != 0),
            Kind::Number(value) => Some(*value != 0.0),
            _ => None,
        })
}

// Telemetry sources for the driver ops. `component` is a coarse class label —
// the driver has one model + one env, and `op` already distinguishes them.
const SRC_PREDICT: Source = Source {
    op: "model.predict",
    component: "model",
};
const SRC_RESET_ADAPTER: Source = Source {
    op: "model.reset_adapter",
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
// Recorded only for a transform that replaced its payload, so a run whose
// hooks transform nothing records nothing.
const SRC_TRANSFORM_OBS: Source = Source {
    op: "runner.transform_observation",
    component: "runner",
};
const SRC_TRANSFORM_ACTION: Source = Source {
    op: "runner.transform_action",
    component: "runner",
};
// Wall clock of one group's full predict -> step -> transform cycle: from one
// step completing (or the reset) to the next step completing. Consumers
// subtract the per-op rows to get the driver's own residual.
const SRC_ROUND: Source = Source {
    op: "runner.round",
    component: "runner",
};

/// Drives one ready model/env session through its `reset -> predict -> step`
/// loop. Inert until a `run*` method is awaited.
#[must_use = "a RuntimeDriver does nothing until one of its run methods is awaited"]
pub struct RuntimeDriver<E, M> {
    spec: RuntimeSessionSpec,
    /// The env session handle; cloned once per group.
    env: E,
    /// The model, shared into every predict in flight: each grouped predict
    /// is one `predict_grouped` call under its lock.
    model: Arc<tokio::sync::Mutex<M>>,
    /// The env contract the model's observations carry, and its action
    /// space, which every predicted frame is encoded against.
    contract: Arc<rlmesh_spaces::EnvContract>,
    /// Async-inference mode: with this many replay frames (or fewer) left, a
    /// group asks for its next chunk while the current one still executes.
    /// The chunk is conditioned on an observation up to `prefetch_lead` steps
    /// stale — deployment-realistic async semantics, not the benchmark loop.
    /// 0 = predict only when a group has no frame to play.
    prefetch_lead: u32,
    /// The model's route negotiated observation history at resolve
    /// (`RouteNeeds::history`): the driver then carries every env step it
    /// executed from a replayed chunk, without predicting, as a history row
    /// on the next predict, and stamps observations and rows with the group's
    /// step counter.
    deliver_history: bool,
    scheduler: Box<dyn PredictScheduler>,
    hooks: Arc<dyn RuntimeHooks>,
    cancellation_reason: String,
    /// Action/observation space specs shared into every per-step hook event.
    /// Populated once after [`validate`](RuntimeSessionSpec::validate) so the
    /// hot path clones an `Arc` instead of deep-copying the spec each step.
    action_space: Arc<rlmesh_proto::spaces::v1::SpaceSpec>,
    observation_space: Arc<rlmesh_proto::spaces::v1::SpaceSpec>,
    /// Episode ids whose adapter state is evicted on the next loop turn
    /// (evictions never wait on a predict).
    pending_evictions: Vec<String>,
    /// `(group, episode id)` evictions queued while that group had a predict
    /// in flight under the id; each joins `pending_evictions` once that
    /// predict lands, so the model never sees an evict before the last predict
    /// under the id.
    held_evictions: Vec<(usize, String)>,
    /// The session set a non-default `trial_index_base` for an env whose
    /// contract does not declare the reset option, so the ordinal was withheld
    /// from `ResetRequest.options`. Latched so the warning fires once per
    /// session, not once per reset.
    trial_options_warned: AtomicBool,
    /// A lane's episode ended mid-chunk on the whole-vector group, discarding
    /// every lane's buffered frames; warned once per session.
    vector_replay_warned: bool,
    /// The options the model adds to the resets that start its episodes.
    reset_options: Option<Arc<dyn ModelResetOptions>>,
    /// The canonical index and space of the observation leaf the env tags
    /// `text/instruction`, which `transform_instruction` rewrites; `None`
    /// when the env tags none.
    instruction_leaf: Option<(usize, rlmesh_spaces::SpaceSpec)>,
}

/// Where a group is in its env lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvPhase {
    /// Holds an observation; may step as soon as it has an action frame.
    Ready,
    /// An env.step is in flight.
    Stepping,
    /// An env.reset is in flight.
    Resetting,
    /// Out of episode budget; never touched again.
    Idle,
}

/// Most observation-history rows a group buffers between predicts: one per
/// replayed frame, and a chunk never legitimately exceeds the runtime's
/// execution-horizon bound (`MAX_EXECUTION_HORIZON`, 1024, in the adapters
/// crate). Beyond it the endpoint is over-producing and the next request would
/// only fail at the message-size cap anyway; fail here with the cause named.
const HISTORY_BACKLOG_CAP: usize = 1023;

/// Where a group is with respect to the model.
enum PredictState {
    /// Nothing outstanding.
    None,
    /// Wants a prediction for this observation; the scheduler picks it up.
    Wanted(ModelObservation),
    /// Part of the grouped predict in flight. `stale` marks a prediction
    /// conditioned on an observation from before an episode boundary; its
    /// result is discarded (one wasted forward per boundary).
    InFlight { stale: bool },
    /// Frames that arrived while the current replay was still playing.
    Ready(VecDeque<Vec<Bytes>>),
}

/// One independently driven set of lanes.
struct Group<E> {
    /// Env lane indices (wire `env_index`).
    lanes: Vec<u32>,
    /// The lanes' slot positions in the route state.
    positions: Vec<usize>,
    /// One lane of a lane endpoint (as opposed to the whole vector).
    lane_group: bool,
    /// Requests name no lanes (the whole vector) rather than `lanes`.
    whole: bool,
    phase: EnvPhase,
    predict: PredictState,
    /// The group's env handle; taken while an op is in flight.
    env: Option<E>,
    /// Runtime-owned action-chunk replay buffer: frame 0 of a predict applies
    /// now, frames 1.. replay open-loop on the following steps without
    /// re-calling the model. A non-chunking predict yields one frame.
    replay: VecDeque<Vec<Bytes>>,
    /// The latest observation request built for this group (the one a
    /// prefetch would predict from).
    obs_msg: Option<ModelObservation>,
    /// History routes: the observations of the steps this group executed from
    /// its replay buffer since its last predict, oldest first, carried on the
    /// next predict so the model-side windows see every step exactly once.
    history: Vec<HistoryFrame>,
    /// This group's env-step counter (one per `env.step`), stamped on predict
    /// requests and history rows so the model can hold them to consecutive
    /// steps. Consecutive observations of one episode carry consecutive values.
    steps: i64,
    /// NEXT_STEP autoreset: lanes that completed this step. Set on
    /// completion (step t), consumed on the autoreset roll (step t+1).
    pending_roll: HashSet<u32>,
    /// NEXT_STEP autoreset: the `episode_completed` events of the lanes in
    /// `pending_roll`, held until the roll so a hook sees the ended id's last
    /// observation and step before its completion, as the model does.
    pending_completed: Vec<EpisodeCompletedEvent>,
    /// The ids and slots of the episodes a pending reset starts.
    pending_start: Option<(Vec<String>, Vec<u64>)>,
    round_started: Instant,
    /// Wall time of the predict(s) that landed since the group's last step,
    /// charged to its next step (see `StepCompletedEvent::predict_ms`).
    pending_predict: Duration,
    /// Instruction routes: the episodes started since the group's last
    /// observation, whose instruction `transform_instruction` is asked for
    /// on it.
    instruction_starts: Vec<EpisodeStartedEvent>,
    /// Instruction routes, per lane: the episode id and the instruction text
    /// that replaces the env's in every observation of that episode.
    instructions: Vec<Option<(String, String)>>,
}

impl<E> Group<E> {
    fn width(&self) -> usize {
        self.lanes.len()
    }

    fn busy(&self) -> bool {
        matches!(self.phase, EnvPhase::Stepping | EnvPhase::Resetting)
    }
}

/// A finished env op, handed back with the group's env handle.
enum EnvOutcome<E> {
    Reset {
        group: usize,
        env: E,
        initial: bool,
        request_bytes: u64,
        rpc: Duration,
        result: Result<RuntimeEnvReset, RuntimeError>,
    },
    Step {
        group: usize,
        env: E,
        request_bytes: u64,
        rpc: Duration,
        result: Result<RuntimeEnvStep, RuntimeError>,
    },
}

/// One predict in flight. Not spawned: it shares the model handle for its
/// duration and is polled by the route loop, so a borrowed model handle (the
/// local runner's) needs no `'static` lifetime.
type PredictFuture<'m> = Pin<Box<dyn Future<Output = PredictOutcome> + Send + 'm>>;

/// A finished predict, one result per request it carried.
struct PredictOutcome {
    /// `(group, request bytes)` per request, in order.
    requests: Vec<(usize, u64)>,
    rpc: Duration,
    result: Result<Vec<Result<Prediction, RuntimeError>>, RuntimeError>,
}

impl<E, M> RuntimeDriver<E, M>
where
    E: RuntimeEnv + Clone + 'static,
    M: ModelHandler,
{
    /// A driver of `model` against `env`, whose contract is `contract`.
    pub fn new(
        spec: RuntimeSessionSpec,
        env: E,
        model: M,
        contract: rlmesh_spaces::EnvContract,
        hooks: Arc<dyn RuntimeHooks>,
    ) -> Self {
        Self {
            spec,
            env,
            reset_options: model.reset_options(),
            model: Arc::new(tokio::sync::Mutex::new(model)),
            contract: Arc::new(contract),
            prefetch_lead: 0,
            deliver_history: false,
            scheduler: Box::new(EagerScheduler),
            hooks,
            cancellation_reason: DEFAULT_CANCELLATION_REASON.to_string(),
            // Filled from the validated spec at run time; default until then.
            action_space: Arc::default(),
            observation_space: Arc::default(),
            pending_evictions: Vec::new(),
            held_evictions: Vec::new(),
            trial_options_warned: AtomicBool::new(false),
            vector_replay_warned: false,
            instruction_leaf: None,
        }
    }

    /// Mint the trial ordinals for the lanes a reset restarts, positionally
    /// aligned to them. Always minted under driver-owned resets (off the
    /// session's `trial_index_base`, 0 by default), so the events and summaries
    /// carry the sweep even when the env never asked for the option. Empty under
    /// `NEXT_STEP` autoreset: the env restarts its own lanes there, so the one
    /// cold-start reset this driver issues would leave every later episode
    /// unsequenced.
    fn planned_trial_indices(&self, state: &mut RouteState, lanes: usize) -> Vec<u64> {
        if self.driver_owns_resets() {
            state.claim_trial_indices(self.spec.trial_index_base(), lanes)
        } else {
            Vec::new()
        }
    }

    /// The `ResetRequest.options` map carrying `trial_index`, or `None`.
    ///
    /// Delivered only to an env whose contract metadata declares the key under
    /// [`ENV_RESET_OPTIONS_KEY`]: an env that forwards `options` blindly into a
    /// third-party `reset` must never receive a reserved key it cannot interpret.
    /// Withholding it from a non-declaring env is silent under the default base
    /// (the env never opted in, and the ordinal is minted regardless) and warns
    /// once when the session set a non-default base, since that caller expected
    /// the sweep to reach the env. A single lane sends the bare integer; a
    /// multi-lane reset sends the list, in the same lane order as `seeds` and
    /// `episode_ids`.
    fn trial_options(&self, trials: &[u64]) -> Option<rlmesh_spaces::MetaMap> {
        let option_key = self.spec.edition_defaults().trial_index_option_key;
        let options = reset_options_for(&self.spec.env_contract, option_key, trials);
        if options.is_none()
            && !trials.is_empty()
            && self.spec.trial_index_base() != 0
            && !self.trial_options_warned.swap(true, Ordering::Relaxed)
        {
            tracing::warn!(
                env_id = %self.spec.env_id,
                key = ENV_RESET_OPTIONS_KEY,
                option = option_key,
                "trial_index_base is set but the env contract declares no such reset \
                 option; the ordinal is recorded on the episode events and summaries \
                 but not delivered to the env",
            );
        }
        options
    }

    /// Enable async inference: a group asks for its next action chunk while
    /// `lead` (or fewer) replay frames of the current one remain, so the
    /// forward overlaps the env steps. The chunk sees an observation up to
    /// `lead` steps stale, so results are NOT comparable to the synchronous
    /// loop — callers must label runs accordingly. `lead == 0` leaves it off.
    pub fn with_prefetch(mut self, lead: u32) -> Self {
        self.prefetch_lead = lead;
        self
    }

    /// Replace the predict scheduling policy (default: [`EagerScheduler`]).
    pub fn with_scheduler(mut self, scheduler: Box<dyn PredictScheduler>) -> Self {
        self.scheduler = scheduler;
        self
    }

    /// Deliver observation history (see `deliver_history`): the model's
    /// route asked for it at resolve.
    pub fn with_history(mut self, deliver: bool) -> Self {
        self.deliver_history = deliver;
        self
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

    /// Whether this session's edition puts lane restarts on the runtime (it
    /// issues the `Reset`s) rather than on the env (which rolls its own lanes).
    fn driver_owns_resets(&self) -> bool {
        self.spec
            .edition_defaults()
            .driver_owned_reset_modes
            .contains(&self.autoreset_mode())
    }

    /// The route's groups: one per lane for a lane endpoint, else the whole
    /// vector as one group.
    fn groups(&self) -> Vec<Group<E>> {
        let num_envs = self.spec.num_envs.max(1);
        let partitions: Vec<Vec<u32>> = if self.spec.subset_step && num_envs > 1 {
            (0..num_envs as u32).map(|lane| vec![lane]).collect()
        } else {
            vec![(0..num_envs as u32).collect()]
        };
        let lane_group = self.spec.subset_step && num_envs > 1;
        partitions
            .into_iter()
            .map(|lanes| Group {
                positions: lanes.iter().map(|&lane| lane as usize).collect(),
                instructions: vec![None; lanes.len()],
                whole: !lane_group,
                lane_group,
                lanes,
                phase: EnvPhase::Ready,
                predict: PredictState::None,
                env: Some(self.env.clone()),
                replay: VecDeque::new(),
                obs_msg: None,
                history: Vec::new(),
                steps: 0,
                pending_roll: HashSet::new(),
                pending_completed: Vec::new(),
                pending_start: None,
                round_started: Instant::now(),
                pending_predict: Duration::ZERO,
                instruction_starts: Vec::new(),
            })
            .collect()
    }

    /// Reset seeds for the episodes a group starts at `slots` (aligned to the
    /// group's lanes): explicit `episode_seeds` indexed by slot when
    /// configured, else the `base_seed` derivation, else unseeded (empty).
    fn seeds_for(&self, group: &Group<E>, slots: &[u64]) -> Vec<i64> {
        if !self.spec.episode_seeds.is_empty() {
            let seeds: Vec<Option<i64>> = slots
                .iter()
                .map(|&slot| {
                    self.spec
                        .episode_seeds
                        .get(usize::try_from(slot).unwrap_or(usize::MAX))
                        .copied()
                })
                .collect();
            // ResetRequest.seeds is positional and all-or-nothing: a batch the
            // list cannot fully cover runs unseeded.
            return if seeds.iter().all(Option::is_some) {
                seeds.into_iter().flatten().collect()
            } else {
                if seeds.iter().any(Option::is_some) {
                    tracing::warn!(
                        lanes = group.width(),
                        "episode_seeds cannot cover this reset batch; it runs unseeded"
                    );
                }
                Vec::new()
            };
        }
        let Some(base_seed) = self.spec.base_seed else {
            return Vec::new();
        };
        // An episode's seed slot is its trial ordinal, `trial_index_base` plus
        // its route slot, so shards that split one total partition one slot
        // range. A lockstep vector's slots are consecutive from the first lane;
        // a surplus lane past the budget takes the next, never-scored slot.
        let Some(&first) = slots.first() else {
            return Vec::new();
        };
        let first = self.spec.trial_index_base().saturating_add(first);
        (0..group.width() as u64)
            .map(|lane| episode_seed(base_seed, &self.spec.seed_key, first.saturating_add(lane)))
            .collect()
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
    /// `session_failed` hook event, and the `ReleaseAdapter` reason, so callers
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
        self.instruction_leaf = instruction_leaf(&self.contract);
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
        let mut env_ops: JoinSet<EnvOutcome<E>> = JoinSet::new();
        let mut predicts: FuturesUnordered<PredictFuture<'_>> = FuturesUnordered::new();
        let result = self
            .run_loop(
                &mut state,
                &cancellation,
                &telemetry,
                &mut env_ops,
                &mut predicts,
            )
            .await;
        // Whatever ended the loop, let the predicts still in flight land
        // before the route releases the model; a hung predict is abandoned
        // after the close timeout.
        env_ops.abort_all();
        let mut abandoned = false;
        if !predicts.is_empty() {
            let drain = async { while predicts.next().await.is_some() {} };
            if tokio::time::timeout(self.spec.limits.service_close_timeout, drain)
                .await
                .is_err()
            {
                abandoned = true;
                tracing::warn!("predict still in flight at route end; abandoned");
            }
        }
        drop(predicts);
        // A route ending mid-episode (a failure, a cancellation, a sibling
        // lane cut by the episode budget) still has episodes the model has
        // predicted on and never heard the end of; its per-episode state (a
        // Python model's episode store) ends through the same hook a completed
        // episode's does, so end them here, once, before the release. Together
        // with the evictions held behind those predicts and any queued since
        // the last loop turn, they go out on the shutdown budget, cancelled or
        // not — but never to an endpoint that just missed the close deadline:
        // it is provably hung, so the flush would only burn its own deadline,
        // and the deadline-bounded release below frees that state anyway.
        if abandoned {
            self.pending_evictions.clear();
            self.held_evictions.clear();
        } else {
            self.pending_evictions
                .extend(self.held_evictions.drain(..).map(|(_, id)| id));
            self.pending_evictions.extend(state.end_live_episodes());
            self.flush_evictions(
                &mut state,
                &telemetry,
                None,
                self.spec.limits.service_close_timeout,
            )
            .await;
        }
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
                snapshot: final_snapshot.clone(),
            }
        );
        match result {
            Ok(reason) => {
                self.shutdown_terminal_route(&state, reason).await;
                fan_out_event!(
                    self,
                    session_ended,
                    SessionEndedEvent {
                        session_id: state.session_id().to_string(),
                        route: state.env_context(),
                        reason: reason.to_string(),
                        total_steps: state.total_steps(),
                        total_episodes: state.total_episodes(),
                    }
                );
                Ok(RuntimeReport {
                    session_id: state.session_id().to_string(),
                    env_id: self.spec.env_id.clone(),
                    total_steps: state.total_steps(),
                    total_episodes: state.total_episodes(),
                    episodes: state.take_episode_summaries(),
                    telemetry: final_snapshot,
                    advisories: Vec::new(),
                })
            }
            Err(error) => {
                self.shutdown_after_failure(&mut state, &error).await;
                Err(error)
            }
        }
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
            lanes = self.spec.subset_step,
        ),
    )]
    async fn run_loop<'m>(
        &mut self,
        state: &mut RouteState,
        cancellation: &CancellationToken,
        telemetry: &Arc<Mutex<Aggregator>>,
        env_ops: &mut JoinSet<EnvOutcome<E>>,
        predicts: &mut FuturesUnordered<PredictFuture<'m>>,
    ) -> Result<&'static str, RuntimeError>
    where
        M: 'm,
    {
        fan_out_event!(
            self,
            session_started,
            SessionStartedEvent {
                session_id: state.session_id().to_string(),
                route: state.env_context(),
                env_id: self.spec.env_id.clone(),
                env_contract: Arc::new(self.spec.env_contract.clone()),
            }
        );

        let mut groups = self.groups();
        let group_count = groups.len();
        for gid in 0..group_count {
            self.begin_reset(gid, &mut groups, state, env_ops, true);
        }

        loop {
            if cancellation.is_cancelled() {
                return Err(self.cancelled_error(state, &groups));
            }
            if groups.iter().all(|group| group.phase == EnvPhase::Idle)
                && env_ops.is_empty()
                && predicts.is_empty()
            {
                return Ok("completed requested episodes");
            }
            // A predict future parked in `predicts` is not polled while this
            // await runs, and a local model's eviction shares its handler lock
            // with predict: flushing here would wait out the predict timeout.
            if predicts.is_empty() {
                self.flush_evictions(
                    state,
                    telemetry,
                    Some(cancellation),
                    self.spec.limits.model_predict_timeout,
                )
                .await;
            }
            self.dispatch_steps(&mut groups, state, env_ops, telemetry)
                .await?;
            // With nothing else in flight, a waiting group must be predicted now
            // or the route would sit forever; the scheduler's plan is advisory
            // only while something else can wake the loop.
            let force = env_ops.is_empty() && predicts.is_empty();
            self.dispatch_predict(&mut groups, state, predicts, force);
            if env_ops.is_empty() && predicts.is_empty() {
                // Nothing in flight and no group could be advanced: a live group
                // with neither a frame nor an observation, which the transitions
                // above never produce.
                return Err(RuntimeError::Protocol(format!(
                    "route {} stalled with no operation in flight",
                    state.env_id()
                )));
            }

            tokio::select! {
                _ = cancellation.cancelled() => {
                    return Err(self.cancelled_error(state, &groups));
                }
                Some(joined) = env_ops.join_next() => {
                    let outcome = joined.map_err(|error| RuntimeError::Protocol(format!(
                        "env operation task failed: {error}"
                    )))?;
                    self.on_env_outcome(outcome, &mut groups, state, env_ops, telemetry)
                        .await?;
                }
                Some(outcome) = predicts.next(), if !predicts.is_empty() => {
                    self.on_predict_outcome(outcome, &mut groups, state, telemetry)?;
                }
            }
        }
    }

    /// Start (or restart) every lane of a group: claim the episodes' slots,
    /// pick their seeds, mint their ids, and put the reset in flight. A group
    /// with no slot left goes idle instead; a lockstep vector wider than the
    /// remaining budget starts its extra lanes surplus (never scored).
    fn begin_reset(
        &mut self,
        gid: usize,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        env_ops: &mut JoinSet<EnvOutcome<E>>,
        initial: bool,
    ) {
        let width = groups[gid].width();
        let slots = if groups[gid].lane_group {
            state.claim_slots(width, true).unwrap_or_default()
        } else {
            state.claim_slots_upto(width)
        };
        if slots.is_empty() {
            groups[gid].phase = EnvPhase::Idle;
            groups[gid].predict = PredictState::None;
            return;
        }
        let seeds = self.seeds_for(&groups[gid], &slots);
        // The runtime is the sole id authority (R1): a fresh UUIDv7 per lane.
        let episode_ids = mint_episode_ids(width);
        for surplus in &episode_ids[slots.len()..] {
            state.mark_surplus(surplus);
        }
        state.note_episode_seeds(&episode_ids, &seeds);
        // Who the model's reset options are for (none asked without a source).
        let starts: Vec<EpisodeInfo> = match self.reset_options {
            Some(_) => episode_ids
                .iter()
                .enumerate()
                .map(|(lane, episode_id)| EpisodeInfo {
                    episode_id: episode_id.clone(),
                    seed: seeds.get(lane).copied(),
                    env_id: state.env_id().to_string(),
                })
                .collect(),
            None => Vec::new(),
        };
        let trials = self.planned_trial_indices(state, width);
        state.note_episode_trials(&episode_ids, &trials);
        let group = &mut groups[gid];
        group.pending_start = Some((episode_ids.clone(), slots));
        group.replay.clear();
        // The rows belonged to the episodes this reset ends; their windows are
        // evicted, and the new episodes start from their reset observation.
        group.history.clear();
        group.pending_roll.clear();
        let reset_options = self.reset_options.clone();
        let mut request = EnvResetRequest {
            seeds,
            options: self.trial_options(&trials),
            env_indices: if group.whole {
                Vec::new()
            } else {
                group.lanes.clone()
            },
        };
        let request_bytes = (request.seeds.len() * 8) as u64;
        let timeout = self.spec.limits.env_reset_timeout;
        let timeout_error = RuntimeError::operation_timeout(
            state.env_id(),
            state.env_component_id(),
            "env.reset",
            0,
            timeout,
        );
        let mut env = group
            .env
            .take()
            .expect("group env handle present while ready");
        group.phase = EnvPhase::Resetting;
        env_ops.spawn(async move {
            let started = Instant::now();
            let result = match model_reset_options(reset_options, &starts, &trials).await {
                Ok(options) => {
                    request.options = merge_options(options, request.options.take());
                    match tokio::time::timeout(timeout, env.reset(request)).await {
                        Ok(result) => result,
                        Err(_) => Err(timeout_error),
                    }
                }
                Err(err) => Err(err),
            };
            EnvOutcome::Reset {
                group: gid,
                env,
                initial,
                request_bytes,
                rpc: started.elapsed(),
                result,
            }
        });
    }

    /// Put a step in flight for every ready group that has an action frame:
    /// the next replay frame, or a chunk that arrived while the last one
    /// played.
    #[allow(clippy::needless_range_loop)]
    async fn dispatch_steps(
        &mut self,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        env_ops: &mut JoinSet<EnvOutcome<E>>,
        telemetry: &Arc<Mutex<Aggregator>>,
    ) -> Result<(), RuntimeError> {
        for gid in 0..groups.len() {
            if groups[gid].phase != EnvPhase::Ready {
                continue;
            }
            if groups[gid].replay.is_empty() {
                match std::mem::replace(&mut groups[gid].predict, PredictState::None) {
                    PredictState::Ready(frames) => groups[gid].replay = frames,
                    other => {
                        groups[gid].predict = other;
                        continue;
                    }
                }
            }
            let Some(model_action) = groups[gid].replay.pop_front() else {
                continue;
            };
            // Async-inference mode: with `prefetch_lead` (or fewer) replay frames
            // left, ask for the next chunk from the latest observation now. On a
            // history route that observation is the backlog's trailing row (it
            // landed while a frame covered its step); `latest_request` promotes
            // it to the request's own and the rows before it ride along.
            if self.prefetch_lead > 0
                && groups[gid].replay.len() <= self.prefetch_lead as usize
                && matches!(groups[gid].predict, PredictState::None)
                && let Some(msg) = self.latest_request(&mut groups[gid], telemetry)
            {
                groups[gid].predict = PredictState::Wanted(msg);
            }

            let group = &mut groups[gid];
            let snapshot = state.snapshot_at(&group.positions);
            let context = state.group_context(&group.positions, group.lane_group);
            let action_step = snapshot.step + 1;
            let mut action_event = ActionReceivedEvent {
                session_id: state.session_id().to_string(),
                route: context,
                episode_id: snapshot.episode_id.clone(),
                episode_record_id: snapshot.episode_record_id.clone(),
                episode_ids: snapshot.episode_ids.clone(),
                episode_record_ids: snapshot.episode_record_ids.clone(),
                step: action_step,
                env_index: snapshot.env_index,
                action_space: Arc::clone(&self.action_space),
                action: Some(model_action),
                raw_action: None,
            };
            let started = Instant::now();
            let transformed = self
                .hooks
                .transform_action(&action_event)
                .await
                .map_err(|err| transform_failed("transform_action", err))?;
            if let Some(action) = transformed {
                record_transform(telemetry, SRC_TRANSFORM_ACTION, started);
                action_event.raw_action = action_event.action.replace(action);
            }
            let action = action_event.action.clone().unwrap_or_default();
            fan_out_event!(self, action_received, action_event);

            let group = &mut groups[gid];
            let request_bytes = leaf_bytes(&action);
            let request = EnvStepRequest {
                action,
                env_indices: if group.whole {
                    Vec::new()
                } else {
                    group.lanes.clone()
                },
            };
            let timeout = self.spec.limits.env_step_timeout;
            let timeout_error = RuntimeError::operation_timeout(
                state.env_id(),
                state.env_component_id(),
                "env.step",
                action_step,
                timeout,
            );
            let mut env = group
                .env
                .take()
                .expect("group env handle present while ready");
            group.phase = EnvPhase::Stepping;
            env_ops.spawn(async move {
                let started = Instant::now();
                let result = match tokio::time::timeout(timeout, env.step(request)).await {
                    Ok(result) => result.map_err(|err| err.at_step(action_step)),
                    Err(_) => Err(timeout_error),
                };
                EnvOutcome::Step {
                    group: gid,
                    env,
                    request_bytes,
                    rpc: started.elapsed(),
                    result,
                }
            });
        }
        Ok(())
    }

    /// Put the groups waiting for a prediction in flight, per the scheduler:
    /// a model that fuses predicts takes the plan as one grouped predict and
    /// the next forms behind it; any other takes one predict per group, in
    /// flight together, each landing on its own.
    fn dispatch_predict<'m>(
        &mut self,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        predicts: &mut FuturesUnordered<PredictFuture<'m>>,
        force: bool,
    ) where
        M: 'm,
    {
        // A model fuses its waiting groups into one `predict_grouped`: one
        // grouped predict is in flight at a time, and the groups that become
        // ready behind it form the next.
        if !predicts.is_empty() {
            return;
        }
        let waiting: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter(|(_, group)| matches!(group.predict, PredictState::Wanted(_)))
            .map(|(gid, _)| gid)
            .collect();
        if waiting.is_empty() {
            return;
        }
        let busy = groups.iter().filter(|group| group.busy()).count();
        let mut chosen = self.scheduler.plan(&waiting, busy);
        chosen.retain(|gid| waiting.contains(gid));
        if chosen.is_empty() {
            if !force {
                return;
            }
            chosen = waiting;
        }
        predicts.push(self.predict_for(groups, state, chosen));
    }

    /// One predict call carrying the `chosen` groups' requests, sharing the
    /// model handle for its duration.
    fn predict_for<'m>(
        &mut self,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        chosen: Vec<usize>,
    ) -> PredictFuture<'m>
    where
        M: 'm,
    {
        let mut requests = Vec::with_capacity(chosen.len());
        let mut metas = Vec::with_capacity(chosen.len());
        let mut step = 0;
        for gid in chosen {
            let PredictState::Wanted(msg) = std::mem::replace(
                &mut groups[gid].predict,
                PredictState::InFlight { stale: false },
            ) else {
                unreachable!("only waiting groups are planned")
            };
            step = step.max(state.snapshot_at(&groups[gid].positions).step);
            state.mark_predicted(&groups[gid].positions);
            metas.push((gid, observation_bytes(&msg)));
            requests.push(msg);
        }
        let timeout = self.spec.limits.model_predict_timeout;
        let timeout_error = RuntimeError::operation_timeout(
            state.env_id(),
            state.model_component_id(),
            "model.predict",
            step,
            timeout,
        );
        let model = Arc::clone(&self.model);
        let contract = Arc::clone(&self.contract);
        Box::pin(async move {
            let started = Instant::now();
            let predicted = predict_grouped(&model, &contract, requests);
            let result = match tokio::time::timeout(timeout, predicted).await {
                Ok(results) => Ok(results),
                Err(_) => Err(timeout_error),
            };
            PredictOutcome {
                requests: metas,
                rpc: started.elapsed(),
                result,
            }
        })
    }

    /// Apply a finished predict: every group in it gets its frames, or has
    /// them discarded if its episode ended in the meantime, and the evictions
    /// held behind it go out.
    fn on_predict_outcome(
        &mut self,
        outcome: PredictOutcome,
        groups: &mut [Group<E>],
        state: &RouteState,
        telemetry: &Arc<Mutex<Aggregator>>,
    ) -> Result<(), RuntimeError> {
        let results = outcome.result?;
        if results.len() != outcome.requests.len() {
            return Err(RuntimeError::Protocol(format!(
                "model endpoint {} answered {} of {} grouped predicts",
                state.model_component_id(),
                results.len(),
                outcome.requests.len()
            )));
        }
        let group_count = outcome.requests.len() as u64;
        let rpc = outcome.rpc;
        let mut recorded = false;
        for ((gid, request_bytes), result) in outcome.requests.into_iter().zip(results) {
            let prediction = result?;
            // One grouped call is one predict op: record its RPC once, with the
            // peer's report from the first reply and the fused width.
            if !recorded {
                recorded = true;
                record_op(
                    telemetry,
                    SRC_PREDICT,
                    outcome.rpc,
                    PeerReport {
                        endpoint_total_ns: prediction.endpoint_total_ns,
                        phases: prediction.phases,
                        group_size: (group_count > 1).then_some(group_count),
                    },
                    request_bytes,
                    prediction
                        .frames
                        .iter()
                        .map(|frame| leaf_bytes(frame))
                        .sum(),
                );
            }
            if prediction.frames.is_empty() {
                return Err(RuntimeError::Protocol(format!(
                    "model endpoint {} returned a predict response with no actions",
                    state.model_component_id()
                )));
            }
            // Ordered frames: frame 0 = this step, frames 1.. = open-loop replay.
            let frames = VecDeque::from(prediction.frames);
            let group = &mut groups[gid];
            group.pending_predict += rpc;
            group.predict = match std::mem::replace(&mut group.predict, PredictState::None) {
                // Conditioned on an observation from before an episode boundary:
                // the chunk must not leak into the new episode. If the next
                // observation already landed while this was in flight, nothing
                // else will re-plan from it: re-arm here or the group stalls.
                //
                // History routes deliver that observation exactly once either
                // way. It landed with this predict in flight, so `observe`
                // buffered it as a row, and `latest_request` promotes it off the
                // backlog to the re-armed request's own. Under driver-owned
                // resets it is the reset observation: `begin_reset` cleared the
                // ended episode's rows before it, so it is the backlog's only
                // row and the request carries none. Under NEXT_STEP it is the
                // terminal observation, predicted on under the ended id before
                // its eviction at t+1; the ended episode's earlier rows are
                // still buffered (completion does not clear them) and ride that
                // request, where they belong. If instead the stale result lands
                // before the observation (a reset still in flight), nothing is
                // re-armed and `observe` re-plans from it as its own.
                PredictState::InFlight { stale: true } => {
                    if group.phase == EnvPhase::Ready && group.replay.is_empty() {
                        self.latest_request(group, telemetry)
                            .map_or(PredictState::None, PredictState::Wanted)
                    } else {
                        PredictState::None
                    }
                }
                PredictState::InFlight { stale: false } => {
                    if group.replay.is_empty() {
                        group.replay = frames;
                        PredictState::None
                    } else {
                        PredictState::Ready(frames)
                    }
                }
                other => other,
            };
            self.pending_evictions.extend(
                self.held_evictions
                    .extract_if(.., |(held, _)| *held == gid)
                    .map(|(_, id)| id),
            );
        }
        Ok(())
    }

    async fn on_env_outcome(
        &mut self,
        outcome: EnvOutcome<E>,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        env_ops: &mut JoinSet<EnvOutcome<E>>,
        telemetry: &Arc<Mutex<Aggregator>>,
    ) -> Result<(), RuntimeError> {
        match outcome {
            EnvOutcome::Reset {
                group: gid,
                env,
                initial,
                request_bytes,
                rpc,
                result,
            } => {
                groups[gid].env = Some(env);
                let reset = result?;
                record_op(
                    telemetry,
                    SRC_RESET,
                    rpc,
                    PeerReport {
                        endpoint_total_ns: reset.endpoint_total_ns,
                        phases: reset.phases,
                        group_size: None,
                    },
                    request_bytes,
                    leaf_bytes(&reset.observation),
                );
                let group = &mut groups[gid];
                let (episode_ids, slots) = group
                    .pending_start
                    .take()
                    .expect("a reset in flight has its episodes staged");
                let context = state.group_context(&group.positions, group.lane_group);
                if initial {
                    fan_out_event!(
                        self,
                        log,
                        LogEvent {
                            session_id: state.session_id().to_string(),
                            route: context.clone(),
                            level: LogLevel::Info,
                            message: format!(
                                "env reset complete in {:.0}ms ({} episode(s) ready)",
                                rpc.as_secs_f64() * 1000.0,
                                episode_ids.len()
                            ),
                            source: Some("runtime".to_string()),
                        }
                    );
                }
                let started =
                    state.start_episodes_at(&groups[gid].positions, episode_ids, !initial, &slots);
                let started = self.invoke_started_episodes(state, &context, started).await;
                groups[gid].instruction_starts.extend(started);
                groups[gid].round_started = Instant::now();
                self.observe(
                    gid,
                    groups,
                    state,
                    telemetry,
                    Some(reset.observation),
                    reset.infos,
                    RequestPhase::ResetObservation,
                    true,
                )
                .await
            }
            EnvOutcome::Step {
                group: gid,
                env,
                request_bytes,
                rpc,
                result,
            } => {
                groups[gid].env = Some(env);
                let step = result?;
                record_op(
                    telemetry,
                    SRC_STEP,
                    rpc,
                    PeerReport {
                        endpoint_total_ns: step.endpoint_total_ns,
                        phases: step.phases,
                        group_size: None,
                    },
                    request_bytes,
                    leaf_bytes(&step.observation),
                );
                self.on_step(gid, groups, state, env_ops, telemetry, step, rpc)
                    .await
            }
        }
    }

    /// Apply one group's step: account the step, complete and (per autoreset
    /// mode) restart its finished lanes, then observe the next observation.
    #[allow(clippy::too_many_arguments)]
    async fn on_step(
        &mut self,
        gid: usize,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        env_ops: &mut JoinSet<EnvOutcome<E>>,
        telemetry: &Arc<Mutex<Aggregator>>,
        response: RuntimeEnvStep,
        rpc: Duration,
    ) -> Result<(), RuntimeError> {
        let positions = groups[gid].positions.clone();
        let lane_group = groups[gid].lane_group;
        let context = state.group_context(&positions, lane_group);
        lock_agg(telemetry).record(Sample::dur(
            SRC_ROUND,
            metrics::RPC_TOTAL,
            groups[gid].round_started.elapsed(),
        ));
        groups[gid].round_started = Instant::now();
        groups[gid].phase = EnvPhase::Ready;
        groups[gid].steps += 1;

        let predict = std::mem::take(&mut groups[gid].pending_predict);
        state.record_step_at(&positions, &response.rewards, rpc, predict);
        let snapshot = state.snapshot_at(&positions);
        // The lanes the env ended this step, from its own flags and this
        // driver's accounting of each lane's episode. A lane rolling under
        // NEXT_STEP autoreset answers its new episode's first observation,
        // never a completion.
        let mut completed_episodes = self.env_completions(state, &groups[gid], &response);

        // Then the lanes the step or time cap ends, skipping those the env
        // just ended, and every live lane an observer ends early. Both run
        // ahead of the step event so it can carry the terminal flags; a cap
        // is never configured under NEXT_STEP, so nothing here precedes a
        // roll it should follow.
        let directive = self.hooks.directive();
        let capped = self.capped_completions(
            state,
            &positions,
            &completed_episodes,
            response.infos.as_ref(),
            directive != RunDirective::Continue && self.driver_owns_resets(),
        );
        completed_episodes.extend(capped);
        let lane_ids = state.episode_ids_at(&positions);
        let lane_flags = |ended: fn(&EpisodeMetadata) -> bool| -> Vec<bool> {
            lane_ids
                .iter()
                .map(|id| {
                    completed_episodes
                        .iter()
                        .any(|m| m.episode_id == *id && ended(m))
                })
                .collect()
        };
        let terminated = lane_flags(|m| m.terminated);
        let truncated = lane_flags(|m| m.truncated);
        let autoreset_roll: Vec<bool> = groups[gid]
            .lanes
            .iter()
            .map(|lane| groups[gid].pending_roll.contains(lane))
            .collect();
        // On an autoreset roll this response carries the NEW episode's reset
        // observation + infos, which belong on its observation event, not on
        // the old episode's step.
        let rolled = !groups[gid].pending_roll.is_empty();
        fan_out_event!(
            self,
            step_completed,
            StepCompletedEvent {
                session_id: state.session_id().to_string(),
                route: context.clone(),
                episode_id: snapshot.episode_id.clone(),
                episode_record_id: snapshot.episode_record_id.clone(),
                step: snapshot.step,
                env_index: snapshot.env_index,
                rewards: response.rewards.clone(),
                infos: if rolled { None } else { response.infos.clone() },
                terminated,
                truncated,
                autoreset_roll,
                predict_ms: predict.as_secs_f64() * 1e3,
                step_ms: rpc.as_secs_f64() * 1e3,
            }
        );

        // Apply any NEXT_STEP autoreset roll the env just performed: roll our
        // slots to freshly minted ids, each on a fresh route-global slot.
        if rolled {
            // The step event above was the ended ids' last hook event, so their
            // completions go out now, ahead of the new episodes' start.
            let completions = std::mem::take(&mut groups[gid].pending_completed);
            self.fan_out_completions(completions).await;
            let pending_roll = std::mem::take(&mut groups[gid].pending_roll);
            // The ended ids leave their slots now, so this is when the model
            // drops them (see `queue_evictions` for why not at completion).
            self.queue_evictions(gid, &groups[gid], state, pending_roll.iter().copied());
            let roll_ids = episode_ids_with_roll(
                state.episode_ids_at(&positions),
                &groups[gid].lanes,
                &pending_roll,
            );
            let rolling: Vec<Option<u64>> = groups[gid]
                .lanes
                .iter()
                .map(|lane| {
                    pending_roll
                        .contains(lane)
                        .then(|| state.claim_slots(1, true))
                        .flatten()
                        .and_then(|slots| slots.first().copied())
                })
                .collect();
            // A rolled lane the budget no longer covers runs surplus.
            for ((lane, id), slot) in groups[gid].lanes.iter().zip(&roll_ids).zip(&rolling) {
                if pending_roll.contains(lane) && slot.is_none() {
                    state.mark_surplus(id);
                }
            }
            let started = state.observe_episode_ids_at(&positions, roll_ids, &rolling);
            let started = self.invoke_started_episodes(state, &context, started).await;
            groups[gid].instruction_starts.extend(started);
        }

        let completions = self.complete_episodes(state, &context, &completed_episodes);
        // Tell the model to evict the ended episodes' state (best-effort GC;
        // ids never repeat so a miss only leaks memory). Under NEXT_STEP the
        // ended id is still observed and predicted on once more (below) and
        // stepped at t+1, so its `episode_completed` event and its eviction
        // both wait for the roll at t+1.
        if self.driver_owns_resets() {
            self.fan_out_completions(completions).await;
            self.queue_evictions(
                gid,
                &groups[gid],
                state,
                completed_episodes.iter().map(|c| c.env_index),
            );
        } else {
            groups[gid].pending_completed.extend(completions);
        }

        if !completed_episodes.is_empty() {
            // A lane that completed gets a fresh episode, so buffered future
            // actions are stale: flush and re-plan (receding horizon on reset).
            // A prediction in flight was conditioned on the ended episode; its
            // chunk must not leak into the new one.
            let group = &mut groups[gid];
            if !group.replay.is_empty() && group.width() > 1 && !self.vector_replay_warned {
                // Tripwire for chunk replay on a lockstep vector group, once per
                // session: the buffer is whole-batch, so ONE lane's episode end
                // throws away every lane's remaining frames.
                self.vector_replay_warned = true;
                tracing::warn!(
                    num_envs = group.width(),
                    discarded_frames = group.replay.len(),
                    "a lane's episode ended mid-chunk on a lockstep vector group: chunk replay \
                     is whole-batch, so every lane's buffered frames were discarded and the \
                     group re-plans. Serve lanes, or use an execution horizon of 1.",
                );
            }
            group.replay.clear();
            group.predict = match std::mem::replace(&mut group.predict, PredictState::None) {
                PredictState::InFlight { .. } => PredictState::InFlight { stale: true },
                _ => PredictState::None,
            };
            // Under NEXT_STEP, a lane that completed this step (t) autoresets
            // at t+1, which rolls its slot.
            if !self.driver_owns_resets() {
                group
                    .pending_roll
                    .extend(completed_episodes.iter().map(|c| c.env_index));
            }
        }

        // The edition's default step bound truncates under driver-owned resets
        // (`capped_completions`). Under NEXT_STEP the env owns its lanes, so a
        // scored episode it never ends fails the run there instead of hanging.
        if !self.driver_owns_resets() {
            let bound = self.spec.edition_defaults().default_max_episode_steps;
            let endless = state.slots_at(&positions).into_iter().find(|slot| {
                slot.step >= bound
                    && slot.episode.as_ref().is_some_and(|episode| {
                        !state.is_surplus(&episode.episode_id)
                            && !completed_episodes
                                .iter()
                                .any(|ended| ended.episode_id == episode.episode_id)
                    })
            });
            if let Some(slot) = endless {
                return Err(RuntimeError::EnvRpc {
                    operation: "step",
                    step: slot.step,
                    message: format!(
                        "lane {}'s episode ran {bound} steps without the env ending it; under \
                         NEXT_STEP autoreset the env owns its episodes, so the runtime cannot \
                         truncate one: give the env a step limit (e.g. Gymnasium's TimeLimit)",
                        slot.env_index
                    ),
                    recoverable: false,
                    source: None,
                });
            }
        }

        // A group runs until every scored episode completed -- the budget bounds
        // episode starts (slot claims), so `total_episodes` reaching it means
        // nothing scored is live anywhere (surplus lanes the env rolled past the
        // budget are stepped, never counted). The one exception is a lane group
        // whose resets the DRIVER owns: that budget is the slot counter, checked
        // at reset by `claim_slots`. A group left holding only surplus lanes
        // (a lane group rolled past the budget under NEXT_STEP) stops now, and
        // every group stops once an observer quits.
        let budget_spent = self
            .spec
            .max_episodes
            .is_some_and(|limit| state.total_episodes() >= limit as i64);
        if directive == RunDirective::Stop
            || ((!lane_group || !self.driver_owns_resets()) && budget_spent)
            || state.all_surplus_at(&positions)
        {
            // The group never steps again, so the ended lanes' deferred
            // completions and evictions (NEXT_STEP, see `queue_evictions`) go
            // out now: the route-end flush sends the evictions before the
            // model is released.
            let completions = std::mem::take(&mut groups[gid].pending_completed);
            self.fan_out_completions(completions).await;
            let pending_roll = std::mem::take(&mut groups[gid].pending_roll);
            self.queue_evictions(gid, &groups[gid], state, pending_roll.iter().copied());
            groups[gid].phase = EnvPhase::Idle;
            groups[gid].predict = PredictState::None;
            return Ok(());
        }

        // Driver-owned resets: a finished lane restarts. Groups are reset whole
        // (a lane group is one lane; a whole-vector group under DISABLED is
        // num_envs == 1, see RuntimeSessionSpec::validate).
        if self.driver_owns_resets() {
            let done: HashSet<u32> = completed_episodes
                .iter()
                .map(|metadata| metadata.env_index)
                .collect();
            if !done.is_empty() {
                self.begin_reset(gid, groups, state, env_ops, false);
                return Ok(());
            }
        }

        self.observe(
            gid,
            groups,
            state,
            telemetry,
            Some(response.observation),
            if rolled { response.infos } else { None },
            RequestPhase::StepObservation,
            false,
        )
        .await
    }

    /// Hand a group its next observation: build the predict request, run the
    /// observation transform, emit it, and ask for a prediction unless replay
    /// frames still cover the next step.
    #[allow(clippy::too_many_arguments)]
    async fn observe(
        &mut self,
        gid: usize,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        telemetry: &Arc<Mutex<Aggregator>>,
        observation: Option<Vec<Bytes>>,
        infos: Option<rlmesh_proto::spaces::v1::MetaMap>,
        phase: RequestPhase,
        is_reset: bool,
    ) -> Result<(), RuntimeError> {
        let positions = groups[gid].positions.clone();
        let context = state.group_context(&positions, groups[gid].lane_group);
        let mut event = self.observation_event(
            state,
            context,
            state.snapshot_at(&positions),
            is_reset,
            observation,
            infos,
            groups[gid].width(),
        );
        let started = Instant::now();
        self.transform_instruction(&mut groups[gid], &mut event)
            .await?;
        let transformed = self
            .hooks
            .transform_observation(&event)
            .await
            .map_err(|err| transform_failed("transform_observation", err))?;
        if let Some(observation) = transformed {
            let raw = event.observation.replace(observation);
            event
                .raw_observation
                .get_or_insert_with(|| raw.unwrap_or_default());
        }
        if event.raw_observation.is_some() {
            record_transform(telemetry, SRC_TRANSFORM_OBS, started);
        }
        let mut msg =
            state.predict_request_at(&positions, event.observation.clone(), phase, &self.contract);
        // Hooks see the observation the model receives, for both step and
        // reset observations, and the env's when a transform replaced it.
        fan_out_event!(self, observation_emitted, event);

        let group = &mut groups[gid];
        // The observation re-plans now only if nothing covers the next step: no
        // replay frame left, and no chunk in flight or waiting to be played
        // (with a prefetch lead the replay can be spent while the prefetched
        // chunk is still on its way, so the replay alone does not decide).
        let replans = group.replay.is_empty() && matches!(group.predict, PredictState::None);
        if self.deliver_history {
            // Exactly-once: an observation is either the predict's own (it
            // re-plans now, carrying the backlog as its rows) or a history row
            // (a buffered or arriving frame will act on it). Never both, never
            // neither. A row stays the group's latest observation: should an
            // async prefetch or a stale-result re-arm re-plan from it later,
            // `latest_request` promotes it off the backlog to that request's
            // own, so it still leaves the group once.
            msg.step = Some(group.steps);
            if replans {
                msg.history = std::mem::take(&mut group.history);
                record_history_rows(telemetry, &msg);
            } else {
                if group.history.len() >= HISTORY_BACKLOG_CAP {
                    return Err(RuntimeError::Protocol(format!(
                        "route {} buffered {} observation-history rows without a predict: the \
                         model endpoint returned a chunk longer than any execution horizon \
                         the runtime supports",
                        state.env_id(),
                        group.history.len()
                    )));
                }
                group.history.push(HistoryFrame {
                    observation: msg.observation.clone(),
                    episodes: msg.route.episodes.clone(),
                    step: group.steps,
                });
            }
        }
        group.obs_msg = Some(msg.clone());
        group.phase = EnvPhase::Ready;
        if replans {
            group.predict = PredictState::Wanted(msg);
        }
        Ok(())
    }

    /// The request a re-plan from the group's latest observation sends: the
    /// prefetch arm in `dispatch_steps` and the stale-result re-arm in
    /// `on_predict_outcome` both predict from `obs_msg` rather than from a
    /// fresh observation. On a history route that observation was buffered as
    /// a row when it landed (nothing re-planned then), so it is popped off the
    /// backlog as the request's own and the rows before it ride along: every
    /// step still leaves the group exactly once. If it is not the backlog's
    /// trailing row it already left as an earlier request's own (a lead of at
    /// least the chunk length re-arms before the next observation lands);
    /// re-sending it would deliver that step twice, so wait for the next
    /// observation instead. Routes without history re-send it as before.
    fn latest_request(
        &self,
        group: &mut Group<E>,
        telemetry: &Arc<Mutex<Aggregator>>,
    ) -> Option<ModelObservation> {
        let mut msg = group.obs_msg.clone()?;
        if self.deliver_history {
            if group.history.last().map(|row| row.step) != msg.step {
                return None;
            }
            group.history.pop();
            msg.history = std::mem::take(&mut group.history);
            record_history_rows(telemetry, &msg);
        }
        Some(msg)
    }

    async fn shutdown_after_failure(&mut self, state: &mut RouteState, error: &RuntimeError) {
        let reason = error.to_string();
        self.shutdown_terminal_route(state, &reason).await;

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

    /// Releases the model's route and, when the session owns it, closes the
    /// env, each within the close timeout: the driver enforces it, so a
    /// close that blocks (an RPC on a hung connection) cannot hang the run's
    /// shutdown.
    async fn shutdown_terminal_route(&mut self, state: &RouteState, reason: &str) {
        let timeout = self.spec.limits.service_close_timeout;
        let released = release(&self.model, state.env_id().to_string());
        match tokio::time::timeout(timeout, released).await {
            Ok(Err(err)) => tracing::warn!(
                error = %err,
                reason,
                "model route release failed during route shutdown"
            ),
            Err(_) => tracing::warn!(
                timeout_ms = timeout.as_millis(),
                reason,
                "model route release timed out during route shutdown; abandoning it"
            ),
            Ok(Ok(())) => {}
        }
        if !self.spec.close_env_on_end {
            tracing::debug!(
                env_id = %state.env_id(),
                reason,
                "skipping environment close; the endpoint remains owned by the run"
            );
            return;
        }
        match tokio::time::timeout(timeout, self.env.close(timeout)).await {
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
    }

    fn cancelled_error(&self, state: &RouteState, groups: &[Group<E>]) -> RuntimeError {
        let step = groups
            .iter()
            .map(|group| state.snapshot_at(&group.positions).step)
            .max()
            .unwrap_or(0);
        RuntimeError::route_cancelled(state.env_id(), step, self.cancellation_reason.as_str())
    }

    /// Fan out the scored episodes' starts; on an instruction route, hand
    /// them back for `transform_instruction` at their first observation.
    async fn invoke_started_episodes(
        &self,
        state: &RouteState,
        context: &RuntimeEnvContext,
        episodes: Vec<StartedEpisode>,
    ) -> Vec<EpisodeStartedEvent> {
        let mut instruction_starts = Vec::new();
        for episode in episodes {
            if state.is_surplus(&episode.episode_id) {
                continue;
            }
            let record = &episode.record;
            let event = EpisodeStartedEvent {
                session_id: state.session_id().to_string(),
                route: context.clone(),
                episode_id: episode.episode_id.clone(),
                episode_record_id: record.record_id.clone(),
                episode_index: record.index,
                env_index: record.env_index,
                started_from_auto_reset: record.started_from_auto_reset,
                seed: state.seed_for_episode(&episode.episode_id),
                trial_index: state.trial_for_episode(&episode.episode_id),
            };
            if self.instruction_leaf.is_some() {
                instruction_starts.push(event.clone());
            }
            fan_out_event!(self, episode_started, event);
        }
        instruction_starts
    }

    /// The completions of the lanes the env ended this step: each built
    /// from the lane's episode as this driver accounted it (steps, reward,
    /// start time) and the lane's share of this step's info.
    fn env_completions(
        &self,
        state: &RouteState,
        group: &Group<E>,
        response: &RuntimeEnvStep,
    ) -> Vec<EpisodeMetadata> {
        let now_ns = crate::run::runtime::state::now_unix_ns();
        let width = group.lanes.len();
        state
            .slots_at(&group.positions)
            .into_iter()
            .zip(&group.lanes)
            .enumerate()
            .filter_map(|(lane, (slot, env_index))| {
                let terminated = response.terminated.get(lane).copied().unwrap_or(false);
                let truncated = response.truncated.get(lane).copied().unwrap_or(false);
                if !(terminated || truncated) || group.pending_roll.contains(env_index) {
                    return None;
                }
                let episode = slot.episode.as_ref()?;
                Some(EpisodeMetadata {
                    episode_id: episode.episode_id.clone(),
                    seed: None,
                    env_index: *env_index,
                    step_count: slot.step,
                    cumulative_reward: slot.cumulative_reward,
                    terminated,
                    truncated,
                    start_timestamp_ns: slot.started_at_ns,
                    end_timestamp_ns: now_ns,
                    final_info: rlmesh_proto::lane_final_info(response.infos.as_ref(), lane, width),
                })
            })
            .collect()
    }

    /// Runtime-truncated completions for the group's lanes at the step/time cap
    /// this step, excluding lanes the env itself just completed. Built from the
    /// driver's own per-slot accounting (steps, accumulated reward, episode
    /// start time) and this step's `infos`, which carry each lane's last
    /// reported outcome; `validate()` guarantees driver-owned resets (autoreset
    /// `DISABLED`) whenever a cap is configured, so the reset path restarts
    /// these lanes exactly like env-reported completions. `ended` caps every
    /// live lane (an observer ended the episode).
    fn capped_completions(
        &self,
        state: &RouteState,
        positions: &[usize],
        env_completed: &[EpisodeMetadata],
        infos: Option<&rlmesh_proto::spaces::v1::MetaMap>,
        ended: bool,
    ) -> Vec<EpisodeMetadata> {
        let step_cap = self.spec.max_episode_steps.or_else(|| {
            self.driver_owns_resets()
                .then_some(self.spec.edition_defaults().default_max_episode_steps)
        });
        let time_cap = self.spec.max_episode_seconds;
        if step_cap.is_none() && time_cap.is_none() && !ended {
            return Vec::new();
        }
        let env_done: Vec<u32> = env_completed
            .iter()
            .map(|metadata| metadata.env_index)
            .collect();
        let now_ns = crate::run::runtime::state::now_unix_ns();
        state
            .slots_at(positions)
            .into_iter()
            .enumerate()
            .filter_map(|(lane, slot)| {
                let episode = slot.episode.as_ref()?;
                let env_index = u32::try_from(slot.env_index).ok()?;
                if env_done.contains(&env_index) {
                    return None;
                }
                let steps_capped = step_cap.is_some_and(|cap| slot.step >= cap);
                let elapsed_seconds = (now_ns - slot.started_at_ns).max(0) as f64 / 1e9;
                let time_capped = time_cap.is_some_and(|cap| elapsed_seconds >= cap);
                (ended || steps_capped || time_capped).then(|| EpisodeMetadata {
                    episode_id: episode.episode_id.clone(),
                    seed: None,
                    env_index,
                    step_count: slot.step,
                    cumulative_reward: slot.cumulative_reward,
                    terminated: false,
                    truncated: true,
                    start_timestamp_ns: slot.started_at_ns,
                    end_timestamp_ns: now_ns,
                    final_info: rlmesh_proto::lane_final_info(infos, lane, positions.len()),
                })
            })
            .collect()
    }

    /// Complete each episode: registry + summary (bounded runs only), and
    /// build its `episode_completed` hook event for the caller to fan out once
    /// the ended id's last event is out (see `Group::pending_completed`).
    /// Summaries are recorded only when `max_episodes` is set — the report is
    /// drained once at route end, so an unbounded (`max_episodes: None`)
    /// session would otherwise accumulate one entry per episode for its whole
    /// lifetime with no reader.
    fn complete_episodes(
        &self,
        state: &mut RouteState,
        context: &RuntimeEnvContext,
        episodes: &[EpisodeMetadata],
    ) -> Vec<EpisodeCompletedEvent> {
        let mut events = Vec::with_capacity(episodes.len());
        for completed in episodes {
            if state.is_surplus(&completed.episode_id) {
                continue;
            }
            let (predict_ms, step_ms) = state.slot_timings_ms(completed.env_index);
            let record = state.complete_episode(&completed.episode_id);
            let episode_record_id = record
                .as_ref()
                .map(|record| record.record_id.clone())
                .unwrap_or_default();
            // Proto env_index is uint32; events are i32/i64.
            let env_index = i32::try_from(completed.env_index).unwrap_or(i32::MAX);
            let seed = state.seed_for_episode(&completed.episode_id);
            let trial_index = state.trial_for_episode(&completed.episode_id);
            if self.spec.max_episodes.is_some() {
                state.record_episode_summary(crate::run::runtime::spec::EpisodeSummary {
                    episode_index: record.as_ref().map_or(0, |record| record.index),
                    env_index,
                    seed,
                    trial_index,
                    step_count: completed.step_count,
                    cumulative_reward: completed.cumulative_reward,
                    terminated: completed.terminated,
                    truncated: completed.truncated,
                    duration_ms: (completed.end_timestamp_ns - completed.start_timestamp_ns).max(0)
                        / 1_000_000,
                    success: success_from_final_info(
                        self.spec.edition_defaults().success_info_keys,
                        completed.final_info.as_ref(),
                    ),
                    predict_ms,
                    step_ms,
                });
            }
            events.push(EpisodeCompletedEvent {
                session_id: state.session_id().to_string(),
                route: context.clone(),
                episode_id: completed.episode_id.clone(),
                episode_record_id,
                episode_index: record.as_ref().map_or(0, |record| record.index),
                env_index,
                step_count: completed.step_count,
                cumulative_reward: completed.cumulative_reward,
                terminated: completed.terminated,
                truncated: completed.truncated,
                duration_ms: (completed.end_timestamp_ns - completed.start_timestamp_ns).max(0)
                    / 1_000_000,
                success: success_from_final_info(
                    self.spec.edition_defaults().success_info_keys,
                    completed.final_info.as_ref(),
                ),
                final_info: completed.final_info.clone(),
                seed,
                trial_index,
                predict_ms,
                step_ms,
            });
        }
        events
    }

    /// `episode_completed` is the last hook event under its episode id.
    async fn fan_out_completions(&self, events: Vec<EpisodeCompletedEvent>) {
        for event in events {
            fan_out_event!(self, episode_completed, event);
        }
    }

    /// Queue the ended episodes' model-side state for eviction. The id to
    /// evict is resolved POSITIONALLY from the runtime's own slot by
    /// `env_index` — the env's `completed_episodes[].episode_id` echo is never
    /// trusted as the authority — so this must run while the slot still holds
    /// the ended id. Under driver-owned resets that is the completion step.
    /// Under NEXT_STEP the lockstep group predicts once more on the terminal
    /// observation (the autoreset step's action, which the env discards for
    /// that lane) and that predict has to land on the ended id — evicting first
    /// would make the model re-seed the ended episode, and re-tagging it with
    /// the new id would leak the old episode's last frame into the new one —
    /// so the eviction waits for the roll at t+1, just before the slot moves
    /// on. Either way the model sees its `on_episode_end` after the last
    /// predict under that id and never a predict after it. Sent by
    /// [`flush_evictions`](Self::flush_evictions) on the next loop turn — or,
    /// while the group still has a predict in flight (a prefetch conditioned
    /// on the ended id), held until that predict lands. Each id is ended on
    /// the slot once, so the teardown sweep does not end it again.
    fn queue_evictions(
        &mut self,
        gid: usize,
        group: &Group<E>,
        state: &mut RouteState,
        env_indices: impl Iterator<Item = u32>,
    ) {
        let ids = env_indices.filter_map(|env_index| state.end_episode_at(env_index));
        if matches!(group.predict, PredictState::InFlight { .. }) {
            self.held_evictions.extend(ids.map(|id| (gid, id)));
        } else {
            self.pending_evictions.extend(ids);
        }
    }

    /// Best-effort GC (R2): a failure is logged and the route keeps moving — a
    /// missed evict only leaks model memory, never corrupts state (ids never
    /// repeat). Bounded by `deadline` and by the route's cancellation, so a
    /// model that keeps its stream open but never answers cannot hold the
    /// route (or its teardown) on an eviction; the abandoned ids are dropped,
    /// not retried — the model ends whatever it still holds at release.
    async fn flush_evictions(
        &mut self,
        state: &mut RouteState,
        telemetry: &Mutex<Aggregator>,
        cancellation: Option<&CancellationToken>,
        deadline: Duration,
    ) {
        if self.pending_evictions.is_empty() {
            return;
        }
        let episode_ids = std::mem::take(&mut self.pending_evictions);
        let episodes = episode_ids.len();
        let evicted = evict(&self.model, state.env_id().to_string(), episode_ids);
        let cancelled = async {
            match cancellation {
                Some(cancellation) => cancellation.cancelled().await,
                None => std::future::pending().await,
            }
        };
        // The eviction's wall time is a telemetry row of its own: an
        // answered RPC (ok or failed) records `rpc.total` under
        // `model.reset_adapter` next to `model.predict`, so a model that
        // serializes its evictions behind its forwards shows up in the
        // windows instead of only in the inter-episode gap.
        let started = Instant::now();
        tokio::select! {
            biased;
            () = cancelled => {
                tracing::warn!(episodes, "model reset_adapter (evict) abandoned: route cancelled");
            }
            result = tokio::time::timeout(deadline, evicted) => match result {
                Ok(result) => {
                    lock_agg(telemetry).record(Sample::dur(
                        SRC_RESET_ADAPTER,
                        metrics::RPC_TOTAL,
                        started.elapsed(),
                    ));
                    if let Err(err) = result {
                        tracing::warn!("model reset_adapter (evict) failed: {err}");
                    }
                }
                Err(_) => tracing::warn!(
                    episodes,
                    timeout_ms = deadline.as_millis(),
                    "model reset_adapter (evict) timed out; abandoning"
                ),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn observation_event(
        &self,
        state: &RouteState,
        route: RuntimeEnvContext,
        snapshot: RouteSnapshot,
        is_reset: bool,
        observation: Option<Vec<Bytes>>,
        infos: Option<rlmesh_proto::spaces::v1::MetaMap>,
        width: usize,
    ) -> ObservationEmittedEvent {
        ObservationEmittedEvent {
            session_id: state.session_id().to_string(),
            route,
            episode_id: snapshot.episode_id,
            episode_record_id: snapshot.episode_record_id,
            episode_ids: snapshot.episode_ids,
            episode_record_ids: snapshot.episode_record_ids,
            step: snapshot.step,
            env_index: snapshot.env_index,
            is_reset,
            num_envs: width as u32,
            observation_space: Arc::clone(&self.observation_space),
            observation,
            raw_observation: None,
            infos,
        }
    }

    /// Instruction routes: ask `transform_instruction` for the episodes that
    /// started since the group's last observation, then put each lane's
    /// replacement into its instruction leaf while that episode runs.
    async fn transform_instruction(
        &self,
        group: &mut Group<E>,
        event: &mut ObservationEmittedEvent,
    ) -> Result<(), RuntimeError> {
        let Some((leaf, space)) = &self.instruction_leaf else {
            return Ok(());
        };
        let Some(leaves) = event.observation.as_ref() else {
            return Ok(());
        };
        let width = group.width();
        let decode = |leaves: &[Bytes]| {
            let slab = leaves.get(*leaf..=*leaf).ok_or_else(|| {
                RuntimeError::Protocol(format!("observation has no instruction leaf {leaf}"))
            })?;
            rlmesh_spaces::leaves::decode_leaf_slab(slab, space, width)
                .map_err(|err| RuntimeError::Protocol(format!("instruction leaf: {err}")))
        };
        let mut texts = None;
        for episode in std::mem::take(&mut group.instruction_starts) {
            let Some(lane) = group
                .lanes
                .iter()
                .position(|&lane| i64::from(lane) == i64::from(episode.env_index))
            else {
                continue;
            };
            if texts.is_none() {
                texts = Some(decode(leaves)?);
            }
            let current = match texts.as_ref().and_then(|texts| texts.get(lane)) {
                Some(rlmesh_spaces::SpaceValue::Text(text)) => text.as_str(),
                _ => "",
            };
            group.instructions[lane] = self
                .hooks
                .transform_instruction(&episode, current)
                .await
                .map_err(|err| transform_failed("transform_instruction", err))?
                .map(|text| (episode.episode_id, text));
        }
        let live = |lane: usize| {
            group.instructions[lane]
                .as_ref()
                .filter(|(id, _)| event.episode_ids.get(lane) == Some(id))
                .map(|(_, text)| text.clone())
        };
        if !(0..width).any(|lane| live(lane).is_some()) {
            return Ok(());
        }
        let mut texts = match texts {
            Some(texts) => texts,
            None => decode(leaves)?,
        };
        for (lane, value) in texts.iter_mut().enumerate() {
            if let Some(text) = live(lane) {
                *value = rlmesh_spaces::SpaceValue::Text(text);
            }
        }
        let slab = rlmesh_spaces::leaves::encode_leaf_slab(&texts, space)
            .map_err(|err| RuntimeError::Protocol(format!("instruction leaf: {err}")))?
            .pop()
            .unwrap_or_default();
        let mut rewritten = leaves.clone();
        rewritten[*leaf] = slab;
        event.raw_observation = event.observation.replace(rewritten);
        Ok(())
    }
}

/// A fatal transform's failure, naming the hook.
fn transform_failed(hook: &str, err: HookError) -> RuntimeError {
    tracing::warn!("runtime hook {hook} failed: {err}");
    RuntimeError::Hook(HookError::Message(format!("{hook}: {err}")))
}

fn record_transform(telemetry: &Mutex<Aggregator>, source: Source, started: Instant) {
    lock_agg(telemetry).record(Sample::dur(source, metrics::RPC_TOTAL, started.elapsed()));
}

/// The canonical leaf index and space of the Text observation leaf the env
/// tags `text/instruction`, walking its tag tree beside the space.
fn instruction_leaf(
    contract: &rlmesh_spaces::EnvContract,
) -> Option<(usize, rlmesh_spaces::SpaceSpec)> {
    use crate::adapters::{EnvTags, INSTRUCTION, ObsLeaf, ObsNode};
    use rlmesh_spaces::SpaceKind;
    fn leaves(space: &rlmesh_spaces::SpaceSpec) -> usize {
        rlmesh_spaces::leaf_specs(space).len()
    }
    fn walk<'a>(
        node: Option<&ObsNode>,
        space: &'a rlmesh_spaces::SpaceSpec,
        offset: &mut usize,
    ) -> Option<&'a rlmesh_spaces::SpaceSpec> {
        match (node, space.spec.as_ref()) {
            (Some(ObsNode::Dict(nodes)), Some(SpaceKind::Dict(dict))) => {
                dict.keys.iter().zip(&dict.spaces).find_map(|(key, child)| {
                    let node = nodes.iter().find(|(k, _)| k == key).map(|(_, n)| n);
                    walk(node, child, offset)
                })
            }
            (Some(ObsNode::Tuple(nodes)), Some(SpaceKind::Tuple(tuple))) => tuple
                .spaces
                .iter()
                .enumerate()
                .find_map(|(i, child)| walk(nodes.get(i), child, offset)),
            (Some(ObsNode::Leaf(ObsLeaf::TextTag(tag))), Some(SpaceKind::Text(_)))
                if tag.role == INSTRUCTION =>
            {
                Some(space)
            }
            _ => {
                *offset += leaves(space);
                None
            }
        }
    }
    let facade = crate::specs::EnvContract::from_wire(contract).ok()?;
    let tags = EnvTags::from_metadata(facade.metadata()?).ok()??;
    let space = contract.observation_space.as_ref()?;
    let mut offset = 0;
    let found = walk(Some(tags.observation()), space, &mut offset)?;
    Some((offset, found.clone()))
}

/// The bytes of a payload's leaf slabs.
fn leaf_bytes(leaves: &[Bytes]) -> u64 {
    leaves.iter().map(|leaf| leaf.len() as u64).sum()
}

/// The bytes an observation carries to the model, history rows included.
fn observation_bytes(observation: &ModelObservation) -> u64 {
    observation.observation.as_deref().map_or(0, leaf_bytes)
        + observation
            .history
            .iter()
            .map(|row| row.observation.as_deref().map_or(0, leaf_bytes))
            .sum::<u64>()
}

/// A model error as the driver's, keeping its recoverable flag.
/// The options the model adds for the lanes `starts` a reset starts as
/// `trials`: one lane's map, or for several lanes each key as a per-lane list
/// (`None` where a lane did not set it).
async fn model_reset_options(
    source: Option<Arc<dyn ModelResetOptions>>,
    starts: &[EpisodeInfo],
    trials: &[u64],
) -> Result<Option<rlmesh_spaces::MetaMap>, RuntimeError> {
    use rlmesh_spaces::MetaValue;
    let Some(source) = source else {
        return Ok(None);
    };
    let mut lanes = Vec::with_capacity(starts.len());
    for (lane, episode) in starts.iter().enumerate() {
        let options = source
            .reset_options(episode, trials.get(lane).copied())
            .await
            .map_err(model_error)?;
        lanes.push(options.unwrap_or_default());
    }
    if let [only] = lanes.as_mut_slice() {
        return Ok(Some(std::mem::take(only)));
    }
    let keys: std::collections::BTreeSet<&String> = lanes.iter().flat_map(|l| l.keys()).collect();
    Ok(Some(
        keys.into_iter()
            .map(|key| {
                let values = lanes
                    .iter()
                    .map(|lane| lane.get(key).cloned().unwrap_or(MetaValue::Null))
                    .collect();
                (key.clone(), MetaValue::List(values))
            })
            .collect(),
    ))
}

/// The model's reset options under the runtime's own reserved ones.
fn merge_options(
    model: Option<rlmesh_spaces::MetaMap>,
    reserved: Option<rlmesh_spaces::MetaMap>,
) -> Option<rlmesh_spaces::MetaMap> {
    let mut options = model.unwrap_or_default();
    options.extend(reserved.unwrap_or_default());
    (!options.is_empty()).then_some(options)
}

fn model_error(error: crate::Error) -> RuntimeError {
    RuntimeError::model_rpc_with_recoverability("model", error.is_transient(), error)
}

/// One grouped predict: every observation in one `predict_grouped` call, the
/// model's fusion seam, then each group's frames encoded against the action
/// space. One group's failure is its own.
async fn predict_grouped<M: ModelHandler>(
    model: &tokio::sync::Mutex<M>,
    contract: &rlmesh_spaces::EnvContract,
    observations: Vec<ModelObservation>,
) -> Vec<Result<Prediction, RuntimeError>> {
    let started = Instant::now();
    let Some(action_space) = contract.action_space.as_ref() else {
        return observations
            .iter()
            .map(|_| {
                Err(model_error(crate::Error::model(
                    "the env contract has no action space",
                )))
            })
            .collect();
    };
    let rows: Vec<usize> = observations.iter().map(|o| o.num_envs).collect();
    let mut handler = model.lock().await;
    let call_started = Instant::now();
    let results = handler.predict_grouped(observations).await;
    let user_ns = rlmesh_proto::elapsed_ns(call_started);
    // Drain the adapter share even for a failed forward, or its time leaks
    // into the next predict's.
    let adapter_ns = handler.take_adapter_ns();
    let held = handler.held_state();
    // A served model's own timing replaces this call's: its user time is its
    // share of a forward it may have fused with other routes' requests, and
    // the call's wall time is the op's RPC.
    let (endpoint_total_ns, endpoint) = match handler.take_endpoint_timing() {
        Some((total, phases)) => (total, Some(phases)),
        None => (None, None),
    };
    drop(handler);
    let encode_started = Instant::now();
    let mut results = results.into_iter();
    rows.into_iter()
        .map(|rows| {
            let PredictFrames { actions, replay } = results
                .next()
                .ok_or_else(|| {
                    model_error(crate::Error::model(
                        "predict_grouped returned fewer results than groups",
                    ))
                })?
                .map_err(model_error)?;
            let frames = std::iter::once(actions)
                .chain(replay)
                .map(|frame| {
                    if frame.len() != rows {
                        return Err(model_error(crate::Error::model(format!(
                            "predict returned {} actions for {rows} lanes",
                            frame.len()
                        ))));
                    }
                    crate::model::types::check_actions_conform(action_space, &frame)
                        .map_err(model_error)?;
                    rlmesh_spaces::leaves::encode_leaf_slab(&frame, action_space)
                        .map_err(|err| model_error(crate::Error::model(err.to_string())))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let phases = endpoint.unwrap_or(EndpointPhases {
                user_ns,
                encode_ns: rlmesh_proto::elapsed_ns(encode_started),
                ..EndpointPhases::default()
            });
            Ok(Prediction {
                frames,
                endpoint_total_ns: endpoint_total_ns
                    .or_else(|| Some(rlmesh_proto::elapsed_ns(started))),
                phases: EndpointPhases {
                    adapter_ns,
                    held_episodes: held.map(|held| held.episodes.min(u64::from(u32::MAX)) as u32),
                    held_state_bytes: held.map(|held| held.bytes),
                    ..phases
                },
            })
        })
        .collect()
}

/// Drops the model's route state at route end.
async fn release<M: ModelHandler>(
    model: &tokio::sync::Mutex<M>,
    env_id: String,
) -> Result<(), RuntimeError> {
    let route_setup = model.lock().await.route_setup();
    match route_setup {
        Some(route_setup) => route_setup
            .release_adapter(&env_id)
            .await
            .map_err(model_error),
        None => Ok(()),
    }
}

/// Ends `episode_ids` on the model: the route's own state first, then the
/// model's hook, as a served model does.
async fn evict<M: ModelHandler>(
    model: &tokio::sync::Mutex<M>,
    env_id: String,
    episode_ids: Vec<String>,
) -> Result<(), RuntimeError> {
    let mut handler = model.lock().await;
    if let Some(route_setup) = handler.route_setup() {
        route_setup
            .reset_adapter(&env_id, &episode_ids)
            .await
            .map_err(model_error)?;
    }
    handler
        .reset_adapter(&env_id, episode_ids)
        .await
        .map_err(model_error)
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

/// Current per-lane ids with the NEXT_STEP autoreset rolls substituted in:
/// lanes not rolling keep their current id, a rolling lane takes a freshly
/// minted one. `ids` is aligned to `lanes` (the group's own lanes, in order),
/// while `pending_roll` holds route-global `env_index`es, so a roll is placed
/// at the lane's POSITION in the group.
fn episode_ids_with_roll(
    mut ids: Vec<String>,
    lanes: &[u32],
    pending_roll: &HashSet<u32>,
) -> Vec<String> {
    for (id, lane) in ids.iter_mut().zip(lanes) {
        if pending_roll.contains(lane) {
            *id = mint_episode_id();
        }
    }
    ids
}

/// Locks the telemetry aggregator, recovering from a poisoned mutex instead of
/// panicking. Telemetry is best-effort and must never take down the route, so a
/// panic under the guard degrades telemetry rather than killing the session.
fn lock_agg(telemetry: &Mutex<Aggregator>) -> MutexGuard<'_, Aggregator> {
    telemetry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Account the observation-history rows riding a predict request as a
/// `history.rows` sample, if it carries any.
fn record_history_rows(telemetry: &Arc<Mutex<Aggregator>>, msg: &ModelObservation) {
    if !msg.history.is_empty() {
        lock_agg(telemetry).record(Sample::count(
            SRC_PREDICT,
            metrics::HISTORY_ROWS,
            msg.history.len() as u64,
        ));
    }
}

fn record_op(
    telemetry: &Mutex<Aggregator>,
    src: Source,
    rpc: Duration,
    peer: PeerReport,
    request_bytes: u64,
    response_bytes: u64,
) {
    let mut agg = lock_agg(telemetry);
    agg.record(Sample::dur(src, metrics::RPC_TOTAL, rpc));
    if let Some(ns) = peer.endpoint_total_ns {
        agg.record(Sample::dur(
            src,
            metrics::ENDPOINT_TOTAL,
            Duration::from_nanos(ns),
        ));
    }
    for (metric, ns) in [
        (metrics::ENDPOINT_DECODE, peer.phases.decode_ns),
        (metrics::ENDPOINT_USER, peer.phases.user_ns),
        (metrics::ENDPOINT_ENCODE, peer.phases.encode_ns),
        (metrics::ENDPOINT_QUEUE, peer.phases.queue_ns),
        (metrics::PREDICT_ADAPTER, peer.phases.adapter_ns),
    ] {
        if ns != 0 {
            agg.record(Sample::dur(src, metric, Duration::from_nanos(ns)));
        }
    }
    // Gauges: a measured zero is a sample (an even vector, an emptied engine),
    // only an unmeasuring peer records nothing.
    if let Some(ns) = peer.phases.lane_skew_ns {
        agg.record(Sample::dur(
            src,
            metrics::LANE_SKEW,
            Duration::from_nanos(ns),
        ));
    }
    if peer.phases.in_flight != 0 {
        agg.record(Sample::count(
            src,
            metrics::PREDICT_IN_FLIGHT,
            u64::from(peer.phases.in_flight),
        ));
    }
    if let Some(episodes) = peer.phases.held_episodes {
        agg.record(Sample::count(
            src,
            metrics::HELD_EPISODES,
            u64::from(episodes),
        ));
    }
    if let Some(bytes) = peer.phases.held_state_bytes {
        agg.record(Sample::bytes(src, metrics::HELD_BYTES, bytes));
    }
    agg.record(Sample::bytes(src, metrics::REQUEST_BYTES, request_bytes));
    agg.record(Sample::bytes(src, metrics::RESPONSE_BYTES, response_bytes));
    if let Some(group) = peer.group_size {
        agg.record(Sample::count(src, metrics::GROUP_SIZE, group));
    }
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

#[cfg(test)]
mod tests {
    use super::success_from_final_info;
    use rlmesh_proto::spaces::v1::{MetaMap, MetaValue, meta_value::Kind};

    const KEYS: &[&str] = &["is_success", "success", "task_success"];

    fn info(entries: &[(&str, Option<Kind>)]) -> MetaMap {
        MetaMap {
            entries: entries
                .iter()
                .map(|(key, kind)| (key.to_string(), MetaValue { kind: kind.clone() }))
                .collect(),
        }
    }

    #[test]
    fn success_skips_keys_whose_kind_carries_no_outcome() {
        // A string (even "False"), a null, or a list never decides; the next
        // key in priority order does.
        let skipped = info(&[
            ("is_success", Some(Kind::Text("False".into()))),
            ("success", None),
            ("task_success", Some(Kind::Integer(1))),
        ]);
        assert_eq!(success_from_final_info(KEYS, Some(&skipped)), Some(true));

        let undecided = info(&[("is_success", Some(Kind::Text("True".into())))]);
        assert_eq!(success_from_final_info(KEYS, Some(&undecided)), None);
        assert_eq!(success_from_final_info(KEYS, None), None);
    }

    #[test]
    fn success_takes_the_first_deciding_key_by_priority() {
        let both = info(&[
            ("is_success", Some(Kind::Number(0.0))),
            ("success", Some(Kind::Bool(true))),
        ]);
        assert_eq!(success_from_final_info(KEYS, Some(&both)), Some(false));
    }
}
