//! The language-agnostic predict/resolve holes the served engine calls back
//! into.
//!
//! The vectorized stateful engine ([`AdaptedModelHandler`](super::engine::AdaptedModelHandler))
//! owns the per-lane loop, the episode-keyed frame buffers, and the native
//! adapter application — all pure Rust. The two genuinely host-language steps
//! cross via these traits, which a binding (PyO3, or any future language)
//! implements; a pure-Rust model implements them with no host runtime at all.

use async_trait::async_trait;
use rlmesh_adapters::v1::{CustomTransform, EncodingTransform, ResolvedAdapter, Value};

use crate::spaces::{EnvContract, SpaceSpec, SpaceValue};
use crate::{Result, model::types::ModelObservation};

/// The model's predict callable plus its discovered lifecycle hooks.
///
/// `predict` is the contract floor: one already-assembled model input → one raw
/// action. The engine loops it per lane (single-sample) for a spec'd route and
/// runs `apply_actions` after. A spec-LESS route (no adapter) bypasses the
/// engine via [`predict_spec_less`](PredictFn::predict_spec_less), which gets the
/// raw observation and preserves the pre-relocation batched path exactly.
///
/// Methods take `&self`: the model's per-episode state lives in the host model
/// object (e.g. a Python policy), not in this Rust handle, so a shared reference
/// suffices and the engine can call back from a blocking worker thread.
pub trait PredictFn: Send + Sync {
    /// Spec'd route: one lane's assembled model input → one raw action. The
    /// engine has already frame-stacked / customs'd / enc-shimmed the input.
    /// The input is a `Value` tree (a `Map`/`List`/leaf payload), matching the
    /// model spec's `InputNode` shape — a bare tensor, a dict, or a tuple.
    fn predict(&self, model_input: Value) -> Result<Value>;

    /// Spec-less route (no adapter): the whole observation goes straight to the
    /// model, batched, returning one action per lane. Preserves the pre-engine
    /// behavior byte-for-byte (the binding reproduces the original path).
    fn predict_spec_less(&self, observation: ModelObservation) -> Result<Vec<SpaceValue>>;

    /// Whether this model permits the future fused forward pass. Default-OFF; an
    /// inert permission door in v1 (the per-lane loop runs regardless) — the seam
    /// the deferred fusion reads.
    fn allow_fusion(&self) -> bool {
        false
    }

    /// Fires on a coarse reset edge (structurally-discovered model hook).
    fn on_reset(&self) -> Result<()> {
        Ok(())
    }

    /// Fires when an episode ends (structurally-discovered model hook). The
    /// engine separately evicts that episode's frame buffers.
    fn on_episode_end(&self) -> Result<()> {
        Ok(())
    }

    /// Fires once at shutdown (structurally-discovered model hook, e.g. free
    /// GPU). The engine separately clears all frame buffers.
    fn on_close(&self) -> Result<()> {
        Ok(())
    }
}

/// The resolved per-route state the engine caches at `ConfigureRoute`: the
/// native adapter, the obs/action spaces, and the two host holes. Built by a
/// [`RouteResolver`] (which has the model spec); held beside an episode-keyed
/// `FrameBuffers` inside the engine.
pub struct RouteConfig {
    pub(crate) adapter: ResolvedAdapter,
    pub(crate) observation_space: SpaceSpec,
    pub(crate) action_space: SpaceSpec,
    pub(crate) customs: Box<dyn CustomTransform + Send>,
    pub(crate) encodings: Box<dyn EncodingTransform + Send>,
}

impl RouteConfig {
    /// Assemble a route config from its resolved parts.
    ///
    /// `customs` fills custom-input holes (e.g. [`NoCustoms`](rlmesh_adapters::v1::NoCustoms)
    /// for a declarative route); `encodings` repacks custom rotation encodings
    /// (e.g. [`NoEncodings`](rlmesh_adapters::v1::NoEncodings) for a route with none).
    pub fn new(
        adapter: ResolvedAdapter,
        observation_space: SpaceSpec,
        action_space: SpaceSpec,
        customs: Box<dyn CustomTransform + Send>,
        encodings: Box<dyn EncodingTransform + Send>,
    ) -> Self {
        Self {
            adapter,
            observation_space,
            action_space,
            customs,
            encodings,
        }
    }
}

/// Resolves a route's [`RouteConfig`] from its env contract at `ConfigureRoute`.
///
/// Returns `None` for a spec-less route (the env sent `NO_ADAPTER`), which the
/// engine then serves through [`PredictFn::predict_spec_less`]. Runs off the
/// predict-serialization lock (see [`ModelRouteSetup`](super::handler::ModelRouteSetup)),
/// so the binding may do blocking host work (resolving a spec) here.
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
