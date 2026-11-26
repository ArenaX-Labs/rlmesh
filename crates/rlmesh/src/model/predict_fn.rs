//! The language-agnostic predict/resolve holes the served engine calls back
//! into.

use super::types::EpisodeInfo;
use async_trait::async_trait;
use rlmesh_adapters::v1::{CustomTransform, EncodingTransform, ResolvedAdapter, Value};

use crate::binding::spaces::{EnvContract, SpaceSpec, SpaceValue};
use crate::{Result, model::types::ModelObservation};

/// The model's predict callable plus its discovered lifecycle hooks.
///
/// `predict` is the floor: one assembled model input → one raw action, looped per lane for a
/// spec'd route. A spec-less route bypasses the engine through
/// [`predict_spec_less`](PredictFn::predict_spec_less). Methods take `&self`: per-episode
/// state lives in the host model object, and the engine may call from a blocking worker thread.
pub trait PredictFn: Send + Sync {
    /// Spec'd route: one lane's assembled model input → one raw action. The engine has
    /// already frame-stacked / customs'd / enc-shimmed the input, a `Value` tree shaped like
    /// the spec's `InputNode`. `episode` is this lane's identity and reset seed, `None` on the
    /// fused grouped path.
    fn predict(&self, model_input: Value, episode: Option<&EpisodeInfo>) -> Result<Value>;

    /// Single-sample CHUNK corner: one assembled model input → a *chunk* of raw actions
    /// (the leading axis is the chunk axis, unstacked by
    /// [`split_chunk`](rlmesh_adapters::v1::split_chunk)). `None` (the default) falls back to
    /// [`predict`](Self::predict).
    ///
    /// `execution_horizon` is how many actions the runtime executes before re-planning; the
    /// engine takes that prefix of the native chunk and discards the rest, so a fixed-size head
    /// may ignore it. `episode` is as for [`predict`](Self::predict).
    fn predict_chunk(
        &self,
        _model_input: Value,
        _execution_horizon: u32,
        _episode: Option<&EpisodeInfo>,
    ) -> Result<Option<Value>> {
        Ok(None)
    }

    /// Whether this model defines a chunk corner. Without one, a pinned horizon > 1 is
    /// inactive and the runtime re-plans every step.
    fn has_chunk(&self) -> bool {
        false
    }

    /// Batched corner: N assembled lane inputs → N raw actions (one per lane) in a single
    /// call, so the model runs one forward pass for the whole vector. Preferred over the
    /// per-lane loop when [`has_batch`](Self::has_batch) is true, and only called then.
    /// `episodes` is row-aligned with `inputs`.
    fn predict_batch(&self, _inputs: Vec<Value>, _episodes: &[EpisodeInfo]) -> Result<Vec<Value>> {
        Err(crate::Error::model("predict_batch is not implemented"))
    }

    /// Whether this model defines the batched corner ([`predict_batch`](Self::predict_batch)).
    fn has_batch(&self) -> bool {
        false
    }

    /// Batched chunk corner: N assembled lane inputs → N native action *chunks* (leading
    /// axis = chunk) in a single call, when [`has_chunk_batch`](Self::has_chunk_batch) is
    /// true. `execution_horizon` is as for [`predict_chunk`](Self::predict_chunk); `episodes`
    /// is row-aligned with `inputs`.
    fn predict_chunk_batch(
        &self,
        _inputs: Vec<Value>,
        _execution_horizon: u32,
        _episodes: &[EpisodeInfo],
    ) -> Result<Vec<Value>> {
        Err(crate::Error::model(
            "predict_chunk_batch is not implemented",
        ))
    }

    /// Whether this model defines the batched chunk corner ([`predict_chunk_batch`](Self::predict_chunk_batch)).
    fn has_chunk_batch(&self) -> bool {
        false
    }

    /// Spec-less route (no adapter): the whole observation goes straight to the model,
    /// batched, returning one action per lane.
    fn predict_spec_less(&self, observation: ModelObservation) -> Result<Vec<SpaceValue>>;

    /// Spec-less route with a pinned execution horizon: like
    /// [`predict_spec_less`](Self::predict_spec_less), but when the runtime pinned
    /// `execution_horizon > 1` a chunking binding runs its chunk corner on the raw
    /// observation and returns frame 0 plus the replay frames.
    fn predict_spec_less_chunked(
        &self,
        observation: ModelObservation,
        _execution_horizon: u32,
    ) -> Result<super::handler::PredictFrames> {
        Ok(super::handler::PredictFrames {
            actions: self.predict_spec_less(observation)?,
            replay: Vec::new(),
        })
    }

    /// Several spec-less groups of one grouped predict, each with its pinned execution
    /// horizon, returning one outcome per group in order. A binding with a batched corner
    /// overrides this to serve every group's lanes in one corner call, keeping each group's
    /// outcome (and failure) separate. The default serves each group through
    /// [`predict_spec_less_chunked`](Self::predict_spec_less_chunked).
    fn predict_spec_less_grouped(
        &self,
        groups: Vec<(ModelObservation, u32)>,
    ) -> Vec<Result<super::handler::PredictFrames>> {
        groups
            .into_iter()
            .map(|(observation, horizon)| self.predict_spec_less_chunked(observation, horizon))
            .collect()
    }

    /// Whether this model permits the fused forward pass: a grouped predict may concatenate
    /// lanes from *different* routes into ONE batched corner call. Off by default; turn it on
    /// only when the batched corners are lane-independent.
    fn allow_fusion(&self) -> bool {
        false
    }

    /// The model's NATIVE chunk length K: how many per-step actions ONE chunk corner call
    /// returns, when the model declares it. `None` (the default) is elastic: the engine takes
    /// the `min(len, execution_horizon)` prefix. A declared K is enforced: resolve rejects
    /// `execution_horizon > K`, and a chunk of any other length fails the predict.
    fn native_chunk(&self) -> Option<u32> {
        None
    }

    /// Fires when an episode ends (structurally-discovered model hook), driven by the
    /// explicit `ResetAdapter` op — once per id it lists, so a model keyed by `episode_id`
    /// drops exactly the episode that ended. There is no episode-begin hook: per-episode state
    /// is seeded lazily on the first predict, so reset it here.
    fn on_episode_end(&self, _episode_id: &str) -> Result<()> {
        Ok(())
    }

    /// Fires once at shutdown (structurally-discovered model hook, e.g. free
    /// GPU). The engine separately clears all frame buffers.
    fn on_close(&self) -> Result<()> {
        Ok(())
    }

    /// An evict-all `ResetAdapter` (no ids) for `env_id`: drop that env's
    /// per-episode model state silently, firing no episode-end hook.
    fn on_route_reset(&self, _env_id: &str) -> Result<()> {
        Ok(())
    }

    /// Whether the model adds reset options per episode
    /// ([`reset_options`](Self::reset_options)); the runtime asks only then.
    fn has_reset_options(&self) -> bool {
        false
    }

    /// The reset options the model adds for `episode`, which a reset is about
    /// to start as trial `trial_index`; `None` adds none.
    fn reset_options(
        &self,
        _episode: &EpisodeInfo,
        _trial_index: Option<u64>,
    ) -> Result<Option<rlmesh_spaces::MetaMap>> {
        Ok(None)
    }

    /// A `ReleaseAdapter` for `env_id`: drop that env's per-episode model
    /// state silently.
    fn on_route_release(&self, _env_id: &str) -> Result<()> {
        Ok(())
    }
}

/// The resolved per-route state the engine caches at `ResolveAdapter`: the native
/// adapter, the obs/action spaces, and the two host holes.
pub struct RouteConfig {
    pub(crate) adapter: ResolvedAdapter,
    pub(crate) observation_space: SpaceSpec,
    pub(crate) action_space: SpaceSpec,
    pub(crate) customs: Box<dyn CustomTransform + Send + Sync>,
    pub(crate) encodings: Box<dyn EncodingTransform + Send + Sync>,
    pub(crate) execution_horizon: u32,
    pub(crate) delivers_history: bool,
}

impl RouteConfig {
    /// Assemble a route config from its resolved parts. `customs` fills custom-input holes
    /// ([`NoCustoms`](rlmesh_adapters::v1::NoCustoms) for a declarative route); `encodings`
    /// repacks custom rotation encodings ([`NoEncodings`](rlmesh_adapters::v1::NoEncodings)
    /// for none).
    pub fn new(
        adapter: ResolvedAdapter,
        observation_space: SpaceSpec,
        action_space: SpaceSpec,
        customs: Box<dyn CustomTransform + Send + Sync>,
        encodings: Box<dyn EncodingTransform + Send + Sync>,
    ) -> Self {
        Self {
            adapter,
            observation_space,
            action_space,
            customs,
            encodings,
            execution_horizon: 1,
            delivers_history: false,
        }
    }
}

/// Resolves a route's [`RouteConfig`] from its env contract at `ResolveAdapter`.
///
/// Runs off the predict-serialization lock (see
/// [`ModelRouteSetup`](super::handler::ModelRouteSetup)), so it may do blocking host work.
#[async_trait]
pub trait RouteResolver: Send + Sync {
    /// Resolve `route_key`'s config from its `env_contract`, or `None` for a
    /// spec-less route. An error fails route configuration.
    async fn resolve(
        &self,
        route_key: &str,
        env_contract: &EnvContract,
    ) -> Result<Option<RouteConfig>>;
}
