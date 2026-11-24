//! The single-route run loop.
//!
//! One driver drives one route. The route's lanes form groups: one per lane for a lane endpoint
//! (the env advertised `subset_step`), one for the whole vector otherwise. Every group runs its own
//! `reset -> predict -> step` loop, so a slow step or reset in one group never stalls another.
//! Groups waiting for a prediction are planned by the [`PredictScheduler`] into one grouped
//! predict (the handler serves one call at a time); env steps and resets stay concurrent across
//! groups. Every state change fans out to the session's [`RuntimeHooks`].

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

pub(crate) struct PeerReport {
    pub(crate) endpoint_total_ns: Option<u64>,
    pub(crate) phases: EndpointPhases,
    pub(crate) group_size: Option<u64>,
}

/// The environment side of a route: reset and step, plus close.
///
/// The driver holds one clone per group, so an implementation is a handle onto the env session:
/// clones share the session and multiplex their requests.
#[async_trait]
pub trait RuntimeEnv: Send {
    async fn reset(&mut self, request: EnvResetRequest) -> Result<RuntimeEnvReset, RuntimeError>;

    async fn step(&mut self, request: EnvStepRequest) -> Result<RuntimeEnvStep, RuntimeError>;

    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}

struct Prediction {
    frames: Vec<Vec<Bytes>>,
    endpoint_total_ns: Option<u64>,
    phases: EndpointPhases,
    group_size: Option<u64>,
}

/// Decides which groups waiting for a prediction go out next, as one grouped predict.
///
/// Consulted whenever at least one group is waiting and the model can take more; `busy` is how many
/// groups are still stepping or resetting. An empty plan waits for the next event; the driver forces
/// a full plan when nothing else is in flight, so a scheduler cannot stall the route.
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

const DEFAULT_CANCELLATION_REASON: &str = "cancelled by caller";

/// Drives one ready model/env session through its `reset -> predict -> step`
/// loop. Inert until a `run*` method is awaited.
#[must_use = "a RuntimeDriver does nothing until one of its run methods is awaited"]
pub struct RuntimeDriver<E, M> {
    spec: RuntimeSessionSpec,
    env: E,
    model: Arc<tokio::sync::Mutex<M>>,
    contract: Arc<rlmesh_spaces::EnvContract>,
    prefetch_lead: u32,
    deliver_history: bool,
    scheduler: Box<dyn PredictScheduler>,
    hooks: Arc<dyn RuntimeHooks>,
    cancellation_reason: String,
    action_space: Arc<rlmesh_proto::spaces::v1::SpaceSpec>,
    observation_space: Arc<rlmesh_proto::spaces::v1::SpaceSpec>,
    pending_evictions: Vec<String>,
    held_evictions: Vec<(usize, String)>,
    trial_options_warned: AtomicBool,
    vector_replay_warned: bool,
    reset_options: Option<Arc<dyn ModelResetOptions>>,
    instruction_leaf: Option<(usize, rlmesh_spaces::SpaceSpec)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvPhase {
    Ready,
    Stepping,
    Resetting,
    Idle,
}

const HISTORY_BACKLOG_CAP: usize = 1023;

enum PredictState {
    None,
    Wanted(ModelObservation),
    InFlight { stale: bool },
    Ready(VecDeque<Vec<Bytes>>),
}

struct Group<E> {
    lanes: Vec<u32>,
    positions: Vec<usize>,
    lane_group: bool,
    whole: bool,
    phase: EnvPhase,
    predict: PredictState,
    env: Option<E>,
    replay: VecDeque<Vec<Bytes>>,
    obs_msg: Option<ModelObservation>,
    history: Vec<HistoryFrame>,
    steps: i64,
    pending_roll: HashSet<u32>,
    pending_completed: Vec<EpisodeCompletedEvent>,
    pending_start: Option<(Vec<String>, Vec<u64>)>,
    round_started: Instant,
    pending_predict: Duration,
    instruction_starts: Vec<EpisodeStartedEvent>,
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

type PredictFuture<'m> = Pin<Box<dyn Future<Output = PredictOutcome> + Send + 'm>>;

struct PredictOutcome {
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
            action_space: Arc::default(),
            observation_space: Arc::default(),
            pending_evictions: Vec::new(),
            held_evictions: Vec::new(),
            trial_options_warned: AtomicBool::new(false),
            vector_replay_warned: false,
            instruction_leaf: None,
        }
    }

    fn planned_trial_indices(&self, state: &mut RouteState, lanes: usize) -> Vec<u64> {
        if self.driver_owns_resets() {
            state.claim_trial_indices(self.spec.trial_index_base(), lanes)
        } else {
            Vec::new()
        }
    }

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

    /// Enable async inference: a group asks for its next action chunk while `lead` (or fewer) replay frames of the current one remain.
    ///
    /// The chunk sees an observation up to `lead` steps stale, so results are not comparable to the
    /// synchronous loop and runs must be labelled accordingly. `lead == 0` leaves it off.
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

    fn autoreset_mode(&self) -> AutoresetMode {
        AutoresetMode::try_from(self.spec.env_contract.autoreset_mode)
            .unwrap_or(AutoresetMode::Disabled)
    }

    fn driver_owns_resets(&self) -> bool {
        self.spec
            .edition_defaults()
            .driver_owned_reset_modes
            .contains(&self.autoreset_mode())
    }

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
        let Some(&first) = slots.first() else {
            return Vec::new();
        };
        let first = self.spec.trial_index_base().saturating_add(first);
        (0..group.width() as u64)
            .map(|lane| episode_seed(base_seed, &self.spec.seed_key, first.saturating_add(lane)))
            .collect()
    }
}
