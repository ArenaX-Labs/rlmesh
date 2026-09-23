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
//! [`RuntimeHooks`](crate::hooks::RuntimeHooks).

use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::stream::{FuturesUnordered, StreamExt};
use prost::{Message, bytes::Bytes};
use rlmesh_proto::EndpointPhases;
use rlmesh_proto::core::v1::AutoresetMode;
use rlmesh_proto::env::v1::{
    EpisodeMetadata, ResetRequest, ResetResponse, StepRequest, StepResponse,
};
use rlmesh_proto::model::v1::{
    AdapterContext, ObservationHistoryFrame, PredictRequest, PredictResponse,
    ReleaseAdapterRequest, ResetAdapterRequest,
};
use rlmesh_proto::spaces::v1::{MetaMap, SpaceSpec, SpaceValue, TupleSpec, space_spec};
use rlmesh_spaces::Advisory;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::hooks::{
    ActionReceivedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, Leg, LogEvent, LogLevel,
    ObservationEmittedEvent, PayloadFacts, RefusingRelayPolicy, RelayAdvisoryEvent, RelayDecision,
    RelayPolicy, RuntimeEnvContext, RuntimeHooks, SessionEndedEvent, SessionFailedEvent,
    SessionStartedEvent, StepCompletedEvent, TelemetrySnapshotEvent,
};
use crate::spec::{ENV_RESET_OPTIONS_KEY, RuntimeReport, RuntimeSessionSpec, reset_options_for};
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
    /// The peer's split of that duration; all-zero from a peer that sends none.
    pub phases: EndpointPhases,
}

/// The env's step reply plus the endpoint-local op duration the peer stamped.
pub struct RuntimeEnvStep {
    pub response: StepResponse,
    /// Endpoint-local op duration (ns) from `JoinResponse.endpoint_total_ns`.
    pub endpoint_total_ns: Option<u64>,
    /// The peer's split of that duration; all-zero from a peer that sends none.
    pub phases: EndpointPhases,
}

/// The model's predict reply plus the endpoint-local op duration the peer stamped.
pub struct RuntimeModelPrediction {
    pub response: PredictResponse,
    /// Endpoint-local op duration (ns) from `JoinResponse.endpoint_total_ns`.
    pub endpoint_total_ns: Option<u64>,
    /// The peer's split of that duration, plus its queue wait and slot depth;
    /// all-zero from a peer that sends none.
    pub phases: EndpointPhases,
    /// Lanes fused into the model forward this predict rode in (1 = a lone
    /// predict). `None` when the transport does not group predicts; recorded
    /// as the `group.size` telemetry metric when present.
    pub group_size: Option<u64>,
}

/// What a peer reported about its own handling of one op, as recorded beside the
/// driver's own RPC timing.
pub(crate) struct PeerReport {
    pub(crate) endpoint_total_ns: Option<u64>,
    pub(crate) phases: EndpointPhases,
    pub(crate) group_size: Option<u64>,
}

/// The environment side of a route: reset and step over the wire, plus close.
///
/// The driver holds one clone per group (a lane endpoint is driven with one
/// request per lane in flight), so an implementation is a handle onto the
/// env session: clones share the session and multiplex their requests.
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
pub trait RuntimeModel: Send + Sync {
    /// Predict the ordered action frames for the batched observation (frame 0 is
    /// this step; any further frames replay open-loop before the next call).
    async fn predict(
        &self,
        request: PredictRequest,
    ) -> Result<RuntimeModelPrediction, RuntimeError>;

    /// Predict for several groups at once, one result per request in order.
    /// The default runs the predicts concurrently, which is enough for a
    /// transport that coalesces concurrent predicts itself; an implementation
    /// that can fuse them into one forward pass overrides this and says so
    /// with [`fuses_predicts`](Self::fuses_predicts).
    async fn predict_group(
        &self,
        requests: Vec<PredictRequest>,
    ) -> Vec<Result<RuntimeModelPrediction, RuntimeError>> {
        futures::future::join_all(requests.into_iter().map(|request| self.predict(request))).await
    }

    /// Whether `predict_group` fuses its requests into one forward pass. The
    /// driver then keeps one grouped predict in flight and batches the groups
    /// that become ready behind it into the next. Default `false`: the driver
    /// puts one predict per group in flight, together, and applies each as it
    /// lands, so a lane never waits behind the slowest predict of a batch.
    fn fuses_predicts(&self) -> bool {
        false
    }

    /// Whether the model's route negotiated observation history at resolve
    /// (`RouteNeeds::history`). The driver then carries every env step it
    /// executed from a replayed chunk, without predicting, as a
    /// `PredictRequest.history` row on the next predict, and stamps requests
    /// and rows with the group's step counter. An async prefetch lead keeps
    /// working: the prefetched request carries the rows buffered so far, with
    /// the observation it predicts from as its own rather than a row.
    /// Default `false`: no rows are buffered.
    fn wants_history(&self) -> bool {
        false
    }

    /// Evict the model's per-episode adapter state (frame-stack buffers) for the
    /// ended episodes. Best-effort GC, not a correctness gate: because episode
    /// ids never repeat (UUIDv7), a dropped ResetAdapter only leaks memory and
    /// can never alias a new episode. Default no-op for impls that hold no
    /// per-episode state.
    async fn reset_adapter(&self, _request: ResetAdapterRequest) -> Result<(), RuntimeError> {
        Ok(())
    }

    /// Release the model endpoint within `timeout`. Default no-op.
    async fn release_adapter(
        &self,
        _request: ReleaseAdapterRequest,
        _timeout: Duration,
    ) -> Result<(), String> {
        Ok(())
    }
}

/// Decides which groups waiting for a prediction go out next (as one grouped
/// predict for a model that fuses them, one predict each otherwise). Consulted
/// whenever at least one group is waiting and the model can take more; `busy`
/// is how many groups are still stepping or resetting. An
/// empty plan waits for the next event (a step or reset completing) — the
/// driver forces a full plan when nothing else is in flight, so a scheduler
/// cannot stall the route.
pub trait PredictScheduler: Send {
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
/// in priority order; `None` when none of them is present. Numeric values coerce
/// by truthiness (`1`/`1.0` → true), matching the Python Session's
/// `bool(info[key])` so the two loops report identical success.
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
    /// The model handle, shared into every predict in flight.
    // ponytail: never `None` any more (the handle is shared, not lent); the
    // `Option` stays so the eviction and release paths read unchanged.
    // Collapse to `Arc<M>` when they are next reworked.
    model: Option<Arc<M>>,
    /// Async-inference mode: with this many replay frames (or fewer) left, a
    /// group asks for its next chunk while the current one still executes.
    /// The chunk is conditioned on an observation up to `prefetch_lead` steps
    /// stale — deployment-realistic async semantics, not the benchmark loop.
    /// 0 = predict only when a group has no frame to play.
    prefetch_lead: u32,
    /// The model negotiated observation history (see
    /// [`RuntimeModel::wants_history`]); read once at run start.
    deliver_history: bool,
    scheduler: Box<dyn PredictScheduler>,
    hooks: Arc<dyn RuntimeHooks>,
    relay_policy: Arc<dyn RelayPolicy>,
    /// Each distinct advisory the relay policy raised, for the report.
    advisories: Mutex<Vec<Advisory>>,
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
    Wanted(PredictRequest),
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
    obs_msg: Option<PredictRequest>,
    /// History routes: the observations of the steps this group executed from
    /// its replay buffer since its last predict, oldest first, carried on the
    /// next predict so the model-side windows see every step exactly once.
    history: Vec<ObservationHistoryFrame>,
    /// This group's env-step counter (one per `env.step`), stamped on predict
    /// requests and history rows so the model can hold them to consecutive
    /// steps. Consecutive observations of one episode carry consecutive values.
    steps: i64,
    /// NEXT_STEP autoreset: lanes that completed this step, mapped to the
    /// fresh id minted for their next episode. Set on completion (step t),
    /// consumed on the autoreset roll (step t+1).
    pending_roll: HashMap<u32, String>,
    /// NEXT_STEP autoreset: the `episode_completed` events of the lanes in
    /// `pending_roll`, held until the roll so a hook sees the ended id's last
    /// observation and step before its completion, as the model does.
    pending_completed: Vec<EpisodeCompletedEvent>,
    /// The ids and slots of the episodes a pending reset starts.
    pending_start: Option<(Vec<String>, Vec<u64>)>,
    reset_generation: u64,
    round_started: Instant,
    /// Wall time of the predict(s) that landed since the group's last step,
    /// charged to its next step (see `StepCompletedEvent::predict_ms`).
    pending_predict: Duration,
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
    /// `(group, expected context, request bytes)` per request, in order.
    requests: Vec<(usize, Option<AdapterContext>, u64)>,
    rpc: Duration,
    result: Result<Vec<Result<RuntimeModelPrediction, RuntimeError>>, RuntimeError>,
}

impl<E, M> RuntimeDriver<E, M>
where
    E: RuntimeEnv + Clone + 'static,
    M: RuntimeModel,
{
    pub fn new(spec: RuntimeSessionSpec, env: E, model: M, hooks: Arc<dyn RuntimeHooks>) -> Self {
        Self {
            spec,
            env,
            model: Some(Arc::new(model)),
            prefetch_lead: 0,
            deliver_history: false,
            scheduler: Box::new(EagerScheduler),
            hooks,
            relay_policy: Arc::new(RefusingRelayPolicy),
            advisories: Mutex::new(Vec::new()),
            cancellation_reason: DEFAULT_CANCELLATION_REASON.to_string(),
            // Filled from the validated spec at run time; default until then.
            action_space: Arc::default(),
            observation_space: Arc::default(),
            pending_evictions: Vec::new(),
            held_evictions: Vec::new(),
            trial_options_warned: AtomicBool::new(false),
            vector_replay_warned: false,
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
    fn trial_options(&self, trials: &[u64]) -> Option<MetaMap> {
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

    /// Replace the relay policy (default: [`RefusingRelayPolicy`]).
    pub fn with_relay_policy(mut self, relay_policy: Arc<dyn RelayPolicy>) -> Self {
        self.relay_policy = relay_policy;
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
                pending_roll: HashMap::new(),
                pending_completed: Vec::new(),
                pending_start: None,
                reset_generation: 0,
                round_started: Instant::now(),
                pending_predict: Duration::ZERO,
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
        if group.lane_group {
            // The route-global slot fixes the seed, so an episode's seed depends
            // on its slot alone, never on which lane ran it or when.
            slots
                .iter()
                .map(|&slot| deterministic_reset_seed(base_seed, &self.spec.session_id, slot, 0))
                .collect()
        } else {
            group
                .lanes
                .iter()
                .map(|&lane| {
                    deterministic_reset_seed(
                        base_seed,
                        &self.spec.session_id,
                        group.reset_generation,
                        lane as usize,
                    )
                })
                .collect()
        }
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
        self.deliver_history = self
            .model
            .as_ref()
            .is_some_and(|model| model.wants_history());
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
                let release_request = state.release_adapter_request(reason);
                self.shutdown_terminal_route(&state, reason, release_request)
                    .await;
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
                    advisories: std::mem::take(
                        self.advisories
                            .get_mut()
                            .unwrap_or_else(PoisonError::into_inner),
                    ),
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
            }
        );
        self.relay_contract(state).await?;

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
        if !initial {
            groups[gid].reset_generation += 1;
        }
        let seeds = self.seeds_for(&groups[gid], &slots);
        // The runtime is the sole id authority (R1): mint a fresh UUIDv7 per
        // lane and push them DOWN so the env tags its episodes with our ids; we
        // never read ids back from the env.
        let episode_ids = mint_episode_ids(width);
        for surplus in &episode_ids[slots.len()..] {
            state.mark_surplus(surplus);
        }
        state.note_episode_seeds(&episode_ids, &seeds);
        let trials = self.planned_trial_indices(state, width);
        state.note_episode_trials(&episode_ids, &trials);
        let group = &mut groups[gid];
        group.pending_start = Some((episode_ids.clone(), slots));
        group.replay.clear();
        // The rows belonged to the episodes this reset ends; their windows are
        // evicted, and the new episodes start from their reset observation.
        group.history.clear();
        group.pending_roll.clear();
        let request = ResetRequest {
            seeds,
            options: self.trial_options(&trials),
            timeout_ms: self.spec.limits.env_reset_timeout_ms().max(0) as u64,
            env_indices: if group.whole {
                Vec::new()
            } else {
                group.lanes.clone()
            },
            episode_ids,
        };
        let request_bytes = request.encoded_len() as u64;
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
            let result = match tokio::time::timeout(timeout, env.reset(request)).await {
                Ok(result) => result,
                Err(_) => Err(timeout_error),
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
                action: Some(model_action.clone()),
                raw_action: Some(model_action),
            };
            let (action, relayed_space) = self
                .invoke_transform_action(telemetry, &action_event)
                .await?;
            action_event.action = action;
            if let Some(space) = relayed_space {
                action_event.action_space = space;
            }
            fan_out_event!(self, action_received, action_event.clone());

            let group = &mut groups[gid];
            // Down-push the authoritative per-lane ids. For a NEXT_STEP autoreset
            // roll (a lane in `pending_roll`), substitute the freshly minted id so
            // the env tags its rolled episode with our id; the slot itself rolls to
            // the same id after the step.
            let episode_ids = episode_ids_with_roll(
                state.episode_ids_at(&group.positions),
                &group.lanes,
                &group.pending_roll,
            );
            let request = StepRequest {
                action: action_event.action.map(leaves_value),
                timeout_ms: self.spec.limits.env_step_timeout_ms().max(0) as u64,
                env_indices: if group.whole {
                    Vec::new()
                } else {
                    group.lanes.clone()
                },
                episode_ids,
            };
            let request_bytes = request.encoded_len() as u64;
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
                    Ok(result) => result,
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
        let fuses = self
            .model
            .as_ref()
            .expect("model handle set for the session")
            .fuses_predicts();
        if fuses && !predicts.is_empty() {
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
        if fuses {
            predicts.push(self.predict_for(groups, state, chosen));
        } else {
            for gid in chosen {
                predicts.push(self.predict_for(groups, state, vec![gid]));
            }
        }
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
            metas.push((gid, msg.context.clone(), msg.encoded_len() as u64));
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
        let model = Arc::clone(
            self.model
                .as_ref()
                .expect("model handle set for the session"),
        );
        Box::pin(async move {
            let started = Instant::now();
            let result = match tokio::time::timeout(timeout, model.predict_group(requests)).await {
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
        for ((gid, expected_context, request_bytes), result) in
            outcome.requests.into_iter().zip(results)
        {
            let prediction = result?;
            if prediction.response.context != expected_context {
                let request_id = expected_context
                    .as_ref()
                    .map(|context| context.request_id.clone())
                    .unwrap_or_default();
                return Err(RuntimeError::ModelRouteMismatch {
                    component_id: state.model_component_id().to_string(),
                    request_id,
                });
            }
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
                        group_size: if group_count > 1 {
                            Some(group_count)
                        } else {
                            prediction.group_size
                        },
                    },
                    request_bytes,
                    prediction.response.encoded_len() as u64,
                );
            }
            if prediction.response.actions.is_empty() {
                return Err(RuntimeError::Protocol(format!(
                    "model endpoint {} returned a predict response with no actions",
                    state.model_component_id()
                )));
            }
            // Ordered frames: frame 0 = this step, frames 1.. = open-loop replay.
            let mut frames = VecDeque::with_capacity(prediction.response.actions.len());
            for frame in &prediction.response.actions {
                if let Some(leaves) = value_leaves(Some(frame))? {
                    frames.push_back(leaves);
                }
            }
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
                    reset.response.encoded_len() as u64,
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
                self.invoke_started_episodes(state, &context, started).await;
                groups[gid].round_started = Instant::now();
                let observation = value_leaves(reset.response.observation.as_ref())?;
                self.observe(
                    gid,
                    groups,
                    state,
                    telemetry,
                    observation,
                    reset.response.infos,
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
                    step.response.encoded_len() as u64,
                );
                self.on_step(gid, groups, state, env_ops, telemetry, step.response, rpc)
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
        response: StepResponse,
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

        let step_observation = value_leaves(response.observation.as_ref())?;
        let predict = std::mem::take(&mut groups[gid].pending_predict);
        state.record_step_at(&positions, &response.rewards, rpc, predict);
        let snapshot = state.snapshot_at(&positions);
        // The runtime mints and owns episode ids (R1): a peer-reported completion
        // naming an id this slot has already moved past is a stale echo (the env
        // server's interrupted-episode buffer replaying an episode the runtime
        // already truncated at its cap). Drop it before it is counted, evicted,
        // or used to trigger a reset. Filtered here, ahead of the roll below,
        // so the check reads the slot the env stepped.
        let mut completed_episodes: Vec<EpisodeMetadata> = Vec::new();
        for metadata in &response.completed_episodes {
            if state.episode_id_at(metadata.env_index) == Some(metadata.episode_id.as_str()) {
                completed_episodes.push(metadata.clone());
            } else {
                tracing::debug!(
                    env_index = metadata.env_index,
                    episode_id = %metadata.episode_id,
                    "dropping a stale env-reported episode completion: the lane has moved past \
                     this id"
                );
            }
        }

        // Cap off the FILTERED list: `capped_completions` skips lanes already in
        // the list it is given, so a stale echo there would suppress a genuine cap.
        // Both run ahead of the step event so it can carry the terminal flags;
        // a cap is never configured under NEXT_STEP, so nothing here precedes a
        // roll it should follow.
        let capped = self.capped_completions(
            state,
            &positions,
            &completed_episodes,
            response.infos.as_ref(),
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
            .map(|lane| groups[gid].pending_roll.contains_key(lane))
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

        // Apply any NEXT_STEP autoreset roll the env just performed with the ids
        // we pushed down: roll our own slots to the same ids, each on a fresh
        // route-global slot. The env never mints — the runtime is authoritative.
        if rolled {
            // The step event above was the ended ids' last hook event, so their
            // completions go out now, ahead of the new episodes' start.
            let completions = std::mem::take(&mut groups[gid].pending_completed);
            self.fan_out_completions(completions).await;
            let pending_roll = std::mem::take(&mut groups[gid].pending_roll);
            // The ended ids leave their slots now, so this is when the model
            // drops them (see `queue_evictions` for why not at completion).
            self.queue_evictions(gid, &groups[gid], state, pending_roll.keys().copied());
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
                        .contains_key(lane)
                        .then(|| state.claim_slots(1, true))
                        .flatten()
                        .and_then(|slots| slots.first().copied())
                })
                .collect();
            // A rolled lane the budget no longer covers runs surplus.
            for ((lane, id), slot) in groups[gid].lanes.iter().zip(&roll_ids).zip(&rolling) {
                if pending_roll.contains_key(lane) && slot.is_none() {
                    state.mark_surplus(id);
                }
            }
            let started = state.observe_episode_ids_at(&positions, roll_ids, &rolling);
            self.invoke_started_episodes(state, &context, started).await;
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
            // Under NEXT_STEP, a lane that completed this step (t) autoresets at
            // t+1. Mint its next id now; the next step's down-push and slot roll
            // both consume it. Mirrors the env's `expect_autoreset`.
            if !self.driver_owns_resets() {
                for completed in &completed_episodes {
                    group
                        .pending_roll
                        .entry(completed.env_index)
                        .or_insert_with(mint_episode_id);
                }
            }
        }

        // A group runs until every scored episode completed -- the budget bounds
        // episode starts (slot claims), so `total_episodes` reaching it means
        // nothing scored is live anywhere (surplus lanes the env rolled past the
        // budget are stepped, never counted). The one exception is a lane group
        // whose resets the DRIVER owns: that budget is the slot counter, checked
        // at reset by `claim_slots`. A group left holding only surplus lanes
        // (a lane group rolled past the budget under NEXT_STEP) stops now.
        let budget_spent = self
            .spec
            .max_episodes
            .is_some_and(|limit| state.total_episodes() >= limit as i64);
        if ((!lane_group || !self.driver_owns_resets()) && budget_spent)
            || state.all_surplus_at(&positions)
        {
            // The group never steps again, so the ended lanes' deferred
            // completions and evictions (NEXT_STEP, see `queue_evictions`) go
            // out now: the route-end flush sends the evictions before the
            // model is released.
            let completions = std::mem::take(&mut groups[gid].pending_completed);
            self.fan_out_completions(completions).await;
            let pending_roll = std::mem::take(&mut groups[gid].pending_roll);
            self.queue_evictions(gid, &groups[gid], state, pending_roll.keys().copied());
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
            step_observation,
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
        let mut msg = state.predict_request_at(&positions, observation.clone(), phase);
        let mut event = self.observation_event(
            state,
            context,
            state.snapshot_at(&positions),
            is_reset,
            observation,
            infos,
            groups[gid].width(),
        );
        let (transformed, relayed_space) =
            self.invoke_transform_observation(telemetry, &event).await?;
        event.observation = transformed.clone();
        if let Some(space) = relayed_space {
            event.observation_space = space;
        }
        msg.observation = transformed.map(leaves_value);
        // Emit the transformed observation actually sent to the model, for
        // both step and reset observations, so hooks always see the same
        // payload model.predict receives.
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
                group.history.push(ObservationHistoryFrame {
                    observation: msg.observation.clone(),
                    episode_info: msg.episode_info.clone(),
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
    ) -> Option<PredictRequest> {
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
        let model_close = async {
            match self.model.as_ref() {
                Some(model) => {
                    tokio::time::timeout(timeout, model.release_adapter(request, timeout)).await
                }
                None => Ok(Ok(())),
            }
        };
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

    fn cancelled_error(&self, state: &RouteState, groups: &[Group<E>]) -> RuntimeError {
        let step = groups
            .iter()
            .map(|group| state.snapshot_at(&group.positions).step)
            .max()
            .unwrap_or(0);
        RuntimeError::route_cancelled(state.env_id(), step, self.cancellation_reason.as_str())
    }

    async fn invoke_started_episodes(
        &self,
        state: &RouteState,
        context: &RuntimeEnvContext,
        episodes: Vec<StartedEpisode>,
    ) {
        for episode in episodes {
            if state.is_surplus(&episode.episode_id) {
                continue;
            }
            let record = &episode.record;
            fan_out_event!(
                self,
                episode_started,
                EpisodeStartedEvent {
                    session_id: state.session_id().to_string(),
                    route: context.clone(),
                    episode_id: episode.episode_id.clone(),
                    episode_record_id: record.record_id.clone(),
                    episode_index: record.index,
                    env_index: record.env_index,
                    started_from_auto_reset: record.started_from_auto_reset,
                    seed: state.seed_for_episode(&episode.episode_id),
                    trial_index: state.trial_for_episode(&episode.episode_id),
                }
            );
        }
    }

    /// Runtime-truncated completions for the group's lanes at the step/time cap
    /// this step, excluding lanes the env itself just completed. Built from the
    /// driver's own per-slot accounting (steps, accumulated reward, episode
    /// start time) and this step's `infos`, which carry each lane's last
    /// reported outcome; `validate()` guarantees driver-owned resets (autoreset
    /// `DISABLED`) whenever a cap is configured, so the reset path restarts
    /// these lanes exactly like env-reported completions.
    fn capped_completions(
        &self,
        state: &RouteState,
        positions: &[usize],
        env_completed: &[EpisodeMetadata],
        infos: Option<&rlmesh_proto::spaces::v1::MetaMap>,
    ) -> Vec<EpisodeMetadata> {
        let step_cap = self.spec.max_episode_steps.or_else(|| {
            self.driver_owns_resets()
                .then_some(self.spec.edition_defaults().default_max_episode_steps)
        });
        let time_cap = self.spec.max_episode_seconds;
        if step_cap.is_none() && time_cap.is_none() {
            return Vec::new();
        }
        let env_done: Vec<u32> = env_completed
            .iter()
            .map(|metadata| metadata.env_index)
            .collect();
        let now_ns = crate::state::now_unix_ns();
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
                (steps_capped || time_capped).then(|| EpisodeMetadata {
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
                state.record_episode_summary(crate::spec::EpisodeSummary {
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
        let Some(model) = self.model.as_ref() else {
            return;
        };
        let episode_ids = std::mem::take(&mut self.pending_evictions);
        let episodes = episode_ids.len();
        let request = state.reset_adapter_request(episode_ids);
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
            result = tokio::time::timeout(deadline, model.reset_adapter(request)) => match result {
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

    /// The action the env receives: `transform_action`'s result, then the
    /// relay policy's.
    async fn invoke_transform_action(
        &self,
        telemetry: &Mutex<Aggregator>,
        event: &ActionReceivedEvent,
    ) -> Result<Relayed, RuntimeError> {
        let started = Instant::now();
        let result = self.hooks.transform_action(event.clone()).await;
        lock_agg(telemetry).record(Sample::dur(
            SRC_TRANSFORM_ACTION,
            metrics::RPC_TOTAL,
            started.elapsed(),
        ));
        match result {
            Ok(action) => {
                self.relay(
                    Leg::ModelToEnv,
                    &event.session_id,
                    &event.route,
                    &self.action_space,
                    action,
                )
                .await
            }
            Err(err) => {
                tracing::warn!("runtime hook transform_action failed: {err}");
                Err(RuntimeError::Hook(err))
            }
        }
    }

    /// The observation the model receives: `transform_observation`'s result,
    /// then the relay policy's.
    async fn invoke_transform_observation(
        &self,
        telemetry: &Mutex<Aggregator>,
        event: &ObservationEmittedEvent,
    ) -> Result<Relayed, RuntimeError> {
        let started = Instant::now();
        let result = self.hooks.transform_observation(event.clone()).await;
        lock_agg(telemetry).record(Sample::dur(
            SRC_TRANSFORM_OBS,
            metrics::RPC_TOTAL,
            started.elapsed(),
        ));
        match result {
            Ok(observation) => {
                self.relay(
                    Leg::EnvToModel,
                    &event.session_id,
                    &event.route,
                    &self.observation_space,
                    observation,
                )
                .await
            }
            Err(err) => {
                tracing::warn!("runtime hook transform_observation failed: {err}");
                Err(RuntimeError::Hook(err))
            }
        }
    }

    /// The target leg's ceiling, `None` when that peer runs in-process.
    fn ceiling(&self, leg: Leg) -> Option<&crate::spec::PeerCeiling> {
        match leg {
            Leg::EnvToModel => self.spec.model_ceiling.as_ref(),
            Leg::ModelToEnv => self.spec.env_ceiling.as_ref(),
        }
    }

    /// Checks the env contract against the model's ceiling before any leaf
    /// flows; a converting policy's advisory is recorded, its leaves unused.
    async fn relay_contract(&self, state: &RouteState) -> Result<(), RuntimeError> {
        let Some(ceiling) = self.ceiling(Leg::EnvToModel) else {
            return Ok(());
        };
        let contract_spaces = SpaceSpec {
            spec: Some(space_spec::Spec::Tuple(TupleSpec {
                spaces: vec![
                    SpaceSpec::clone(&self.observation_space),
                    SpaceSpec::clone(&self.action_space),
                ],
            })),
            ..Default::default()
        };
        let payload = PayloadFacts {
            byte_len: self.spec.env_contract.encoded_len(),
            ..PayloadFacts::new(Arc::new(contract_spaces), Vec::new())
        };
        match self
            .relay_policy
            .reconcile(Leg::EnvToModel, ceiling, &payload)
        {
            RelayDecision::Forward => Ok(()),
            RelayDecision::Convert { advisory, .. } => {
                self.raise_advisory(
                    state.session_id(),
                    &state.env_context(),
                    Leg::EnvToModel,
                    advisory,
                )
                .await;
                Ok(())
            }
            RelayDecision::Refuse(reason) => {
                Err(relay_refused(Leg::EnvToModel, "contract", reason))
            }
        }
    }

    /// Hands a payload bound for a served peer to the relay policy.
    async fn relay(
        &self,
        leg: Leg,
        session_id: &str,
        route: &RuntimeEnvContext,
        space: &Arc<SpaceSpec>,
        leaves: Option<Vec<Bytes>>,
    ) -> Result<Relayed, RuntimeError> {
        let Some(ceiling) = self.ceiling(leg) else {
            return Ok((leaves, None));
        };
        let Some(leaves) = leaves else {
            return Ok((None, None));
        };
        let payload = PayloadFacts::new(Arc::clone(space), leaves);
        match self.relay_policy.reconcile(leg, ceiling, &payload) {
            RelayDecision::Forward => Ok((Some(payload.leaves), None)),
            RelayDecision::Convert {
                leaves,
                space,
                advisory,
            } => {
                self.raise_advisory(session_id, route, leg, advisory).await;
                Ok((Some(leaves), space))
            }
            RelayDecision::Refuse(reason) => Err(relay_refused(leg, "payload", reason)),
        }
    }

    /// Records `advisory` once per session and streams it to the hooks.
    async fn raise_advisory(
        &self,
        session_id: &str,
        route: &RuntimeEnvContext,
        leg: Leg,
        advisory: Advisory,
    ) {
        {
            let mut advisories = self
                .advisories
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if advisories.contains(&advisory) {
                return;
            }
            advisories.push(advisory.clone());
        }
        fan_out_event!(
            self,
            relay_advisory,
            RelayAdvisoryEvent {
                session_id: session_id.to_string(),
                route: route.clone(),
                leg,
                advisory,
            }
        );
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
            raw_observation: observation.clone(),
            observation,
            infos,
        }
    }
}

/// The leaves a peer receives, and the space typing them when a relay policy
/// converted their layout.
type Relayed = (Option<Vec<Bytes>>, Option<Arc<SpaceSpec>>);

fn relay_refused(leg: Leg, what: &str, reason: String) -> RuntimeError {
    RuntimeError::Protocol(format!(
        "the runtime cannot relay this {what} to the {}: {reason}",
        leg.target()
    ))
}

/// The per-lane reset seed, derived purely from reproducible inputs: the user's
/// base_seed, the session id, the reset generation (or route-global slot), and
/// the lane index. The container env_id is deliberately NOT mixed in — it is a
/// per-attach random UUIDv7, so including it would make a base_seed
/// non-reproducible across runs.
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
/// stay byte-identical. `ids[i]` is lane `i`'s id (a whole-vector group).
/// `ids` is aligned to `lanes` (the group's own lanes, in order), while
/// `pending_roll` is keyed by the route-global `env_index`, so a roll is placed
/// at the lane's POSITION in the group — a lane group's single id is at 0
/// whatever its lane number, and a whole-vector group's positions already equal
/// its lane numbers.
fn episode_ids_with_roll(
    mut ids: Vec<String>,
    lanes: &[u32],
    pending_roll: &HashMap<u32, String>,
) -> Vec<String> {
    for (env_index, new_id) in pending_roll {
        if let Some(slot) = lanes
            .iter()
            .position(|lane| lane == env_index)
            .and_then(|position| ids.get_mut(position))
        {
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

/// Account the observation-history rows riding a predict request as a
/// `history.rows` sample, if it carries any.
fn record_history_rows(telemetry: &Arc<Mutex<Aggregator>>, msg: &PredictRequest) {
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
