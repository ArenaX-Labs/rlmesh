//! Options for runs and sessions, with their defaults and validation rules.

use std::fmt;
use std::time::Duration;

use super::RunHooks;
use crate::{Error, Result};

/// The execution-horizon ceiling.
pub(crate) const MAX_EXECUTION_HORIZON: u32 = 1024;

/// Options for [`run`](crate::run) and [`Model::run`](crate::Model::run).
///
/// Build with `..Default::default()`. Adding an options field is a minor
/// change by project policy.
#[derive(Default)]
pub struct RunOptions<'h> {
    /// The exact number of episodes: one by default, `seeds.len()` when only
    /// seeds are given; the two must agree when both are. `Some(0)` runs
    /// nothing.
    pub episodes: Option<u64>,
    /// One reset seed per episode.
    pub seeds: Option<Vec<i64>>,
    /// Truncate an episode after this many steps (at least 1).
    pub max_episode_steps: Option<u64>,
    /// Truncate an episode after this wall time (must be positive).
    pub max_episode: Option<Duration>,
    /// Observer callbacks, borrowed for the call.
    pub hooks: Option<&'h mut dyn RunHooks>,
    /// Override each text input of a spec’d model for this call.
    pub instruction: Option<String>,
    /// Live viewer; a local `Model::run` refuses a set view.
    pub view: Option<crate::View>,
    /// Shut the caller's env down when the run ends.
    pub close_env: bool,
    /// Override the model's `trust_entrypoints` for this run.
    pub trust_entrypoints: Option<bool>,
    /// Actions of each predicted chunk executed before re-planning
    /// (1..=1024). `None`: the model's own, the one its `load()` chose, else
    /// its spec's `timing.execution`, else 1.
    pub execution_horizon: Option<u32>,
    /// Native-loop async inference: predict the next chunk while this many
    /// replay frames remain (0 = off; at most `u32::MAX`).
    pub prefetch_lead: u64,
    /// The workflow edition this run declares, above
    /// `RLMESH_WORKFLOW_EDITION` and the model's own.
    pub workflow_edition: Option<String>,
    /// The first trial ordinal; episode `i` walks `trial_index_base + i`.
    pub trial_index_base: u64,
    /// Cancels the run.
    pub cancel: Option<crate::ext::CancelToken>,
}

impl fmt::Debug for RunOptions<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunOptions")
            .field("episodes", &self.episodes)
            .field("seeds", &self.seeds)
            .field("max_episode_steps", &self.max_episode_steps)
            .field("max_episode", &self.max_episode)
            .field("hooks", &self.hooks.as_ref().map(|_| "<hooks>"))
            .field("view", &self.view)
            .field("close_env", &self.close_env)
            .field("trust_entrypoints", &self.trust_entrypoints)
            .field("execution_horizon", &self.execution_horizon)
            .field("prefetch_lead", &self.prefetch_lead)
            .field("workflow_edition", &self.workflow_edition)
            .field("trial_index_base", &self.trial_index_base)
            .field("cancel", &self.cancel)
            .finish()
    }
}

/// Options for [`bind`](crate::bind) and [`Model::bind`](crate::Model::bind).
///
/// Build with `..Default::default()`.
#[derive(Default)]
pub struct BindOptions<'h> {
    /// Close a local env (or shut a served one down) with the binding.
    pub close_env: bool,
    /// Actions of each predicted chunk executed before re-planning
    /// (1..=1024). `None`: the model's own, as for
    /// [`RunOptions::execution_horizon`].
    pub execution_horizon: Option<u32>,
    /// The workflow edition this binding declares.
    pub workflow_edition: Option<String>,
    /// Observer callbacks, borrowed for the binding's life: the binding is
    /// their [`RunContext`](super::RunContext), an episode reset over
    /// before it ended counts as truncated, and `on_run_end` fires at close.
    pub hooks: Option<&'h mut dyn RunHooks>,
    /// Live viewer; the native loop refuses a set view (RUN-4).
    pub view: Option<crate::View>,
}

impl fmt::Debug for BindOptions<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BindOptions")
            .field("close_env", &self.close_env)
            .field("execution_horizon", &self.execution_horizon)
            .field("workflow_edition", &self.workflow_edition)
            .field("hooks", &self.hooks.as_ref().map(|_| "<hooks>"))
            .finish()
    }
}

/// Resolve the episode count from the explicit budget and seeds.
pub(crate) fn episode_budget(episodes: Option<u64>, seeds: Option<&[i64]>) -> Result<u64> {
    match (episodes, seeds) {
        (None, Some(seeds)) => Ok(seeds.len() as u64),
        (None, None) => Ok(1),
        (Some(episodes), Some(seeds)) if seeds.len() as u64 != episodes => {
            Err(Error::invalid_argument(format!(
                "episodes={episodes} does not match len(seeds)={}: pass one seed per episode, \
                 or only one of the two",
                seeds.len()
            )))
        }
        (Some(episodes), _) => Ok(episodes),
    }
}

/// Select the action execution horizon and validate it against the model's
/// native chunk. An explicit run value wins over the checkpoint's loaded value,
/// then the spec's timing, then 1. Bindings use this same Rust policy.
#[doc(hidden)]
pub fn resolve_execution_horizon(
    explicit: Option<i128>,
    loaded: Option<i128>,
    timing: Option<i128>,
    native_chunk: Option<u32>,
) -> Result<u32> {
    let horizon = explicit.or(loaded).or(timing).unwrap_or(1);
    if horizon < 1 {
        return Err(Error::invalid_argument(format!(
            "execution_horizon must be >= 1, got {horizon}"
        )));
    }
    if horizon > i128::from(MAX_EXECUTION_HORIZON) {
        return Err(Error::invalid_argument(format!(
            "execution_horizon must be <= {MAX_EXECUTION_HORIZON}, got {horizon}"
        )));
    }
    let horizon = horizon as u32;
    if let Some(chunk) = native_chunk
        && horizon > chunk
    {
        return Err(Error::invalid_argument(format!(
            "execution_horizon={horizon} exceeds the model's declared native_chunk={chunk}: it \
             cannot produce that many actions per predict. Lower the horizon to at most {chunk}."
        )));
    }
    Ok(horizon)
}

/// `execution_horizon` in 1..=1024 and at most a declared native chunk.
pub(crate) fn check_horizon(horizon: u32, native_chunk: Option<u32>) -> Result<()> {
    resolve_execution_horizon(Some(horizon.into()), None, None, native_chunk).map(|_| ())
}

/// The per-episode caps, as the runtime takes them.
pub(crate) struct Caps {
    pub(crate) max_episode_steps: Option<i64>,
    pub(crate) max_episode_seconds: Option<f64>,
}

/// `max_episode_steps` in 1..=i64::MAX and a positive `max_episode`.
pub(crate) fn check_caps(
    max_episode_steps: Option<u64>,
    max_episode: Option<Duration>,
) -> Result<Caps> {
    let max_episode_steps = match max_episode_steps {
        Some(0) => {
            return Err(Error::invalid_argument(
                "max_episode_steps must be >= 1, got 0",
            ));
        }
        Some(steps) => Some(i64::try_from(steps).map_err(|_| {
            Error::invalid_argument(format!(
                "max_episode_steps must be <= {}, got {steps}",
                i64::MAX
            ))
        })?),
        None => None,
    };
    if max_episode == Some(Duration::ZERO) {
        return Err(Error::invalid_argument(
            "max_episode_seconds must be > 0, got 0.0",
        ));
    }
    Ok(Caps {
        max_episode_steps,
        max_episode_seconds: max_episode
            .filter(|cap| *cap != Duration::MAX)
            .map(|cap| cap.as_secs_f64()),
    })
}

/// `prefetch_lead` within the runtime's `u32` bound.
pub(crate) fn check_prefetch(prefetch_lead: u64) -> Result<u32> {
    u32::try_from(prefetch_lead).map_err(|_| {
        Error::invalid_argument(format!(
            "prefetch_lead must be <= {}, got {prefetch_lead}",
            u32::MAX
        ))
    })
}

impl RunOptions<'_> {
    /// Set the text override.
    pub fn instruction(mut self, value: impl Into<String>) -> Self {
        self.instruction = Some(value.into());
        self
    }
}

pub(crate) fn check_address_close(address: &str, close: bool) -> Result<()> {
    if close && crate::address::is_v2(address) {
        return Err(Error::invalid_argument(
            "close_env=True on a session address is gone in rlmesh 0.2: the env belongs to its serving process; an owner shutdown is server.shutdown() there",
        ));
    }
    Ok(())
}
