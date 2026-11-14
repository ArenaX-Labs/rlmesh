//! The single-route run loop.
//!
//! One driver drives one route. The route's lanes form *groups*: one group per
//! lane for a lane endpoint (the env advertised `subset_step`), one group for
//! the whole vector otherwise. Every group is its own `reset -> predict ->
//! step` episode loop, advanced as its own ops complete, so a slow step or
//! reset in one group never stalls another. Groups waiting for a prediction
//! are planned by the [`PredictScheduler`]: the plan goes to the model as one
//! grouped predict, fused or not (the handler serves one call at a time), its
//! results are applied when it lands, and the groups ready behind it form the
//! next. Env steps and resets stay concurrent across groups.
//!
//! Records per-op telemetry and fans every state change out to the session's
//! [`RuntimeHooks`](crate::run::runtime::hooks::RuntimeHooks).

use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use rlmesh_proto::EndpointPhases;
use rlmesh_proto::core::v1::AutoresetMode;
use rlmesh_proto::spaces::v1::MetaMap;

use crate::model::handler::{ModelHandler, ModelResetOptions};
use crate::model::types::{HistoryFrame, ModelObservation};
use crate::run::runtime::hooks::{EpisodeCompletedEvent, EpisodeStartedEvent, RuntimeHooks};
use crate::run::runtime::seeding::episode_seed;
use crate::run::runtime::spec::{ENV_RESET_OPTIONS_KEY, RuntimeSessionSpec, reset_options_for};
use crate::run::runtime::state::RouteState;

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

mod env_ops;
mod lifecycle;
mod model_calls;
mod observe;
mod predict;
mod run;
mod telemetry;

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
    /// Lanes in the forward the handler answered in, as it reported them.
    group_size: Option<u64>,
}

/// Decides which groups waiting for a prediction go out next, as one grouped
/// predict. Consulted
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
}
