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

    /// Single-sample CHUNK corner: one assembled model input → a *chunk* of raw
    /// actions (the leading axis is the chunk axis, unstacked by
    /// [`split_chunk`](rlmesh_adapters::v1::split_chunk)). `None` (the default)
    /// means the model has no distinct chunk corner, so the engine falls back to
    /// [`predict`](Self::predict). A model that authors a separate chunk policy
    /// (e.g. a Python `predict_chunk`) returns `Some(chunk)`.
    ///
    /// `horizon` is the runtime-chosen replay horizon `h`: the model should return
    /// up to `h` actions (the runtime replays them before re-planning). Emitting
    /// exactly `h` lets an autoregressive head decode only what is used instead of
    /// its full natural chunk; a fixed-size head may return more and the engine
    /// caps to `h`.
    fn predict_chunk(&self, _model_input: Value, _horizon: u32) -> Result<Option<Value>> {
        Ok(None)
    }

    /// Whether this model defines a chunk corner. Queried once at `ConfigureRoute`
    /// so the engine can warn when the runtime pins a horizon > 1 but the model
    /// cannot chunk (chunking is then inactive — the runtime re-plans every step).
    fn has_chunk(&self) -> bool {
        false
    }

    /// Batched corner: N assembled lane inputs → N raw actions (one per lane) in a
    /// single call, so the model runs one forward pass for the whole vector. The
    /// engine prefers this over the per-lane `predict` loop when
    /// [`has_batch`](Self::has_batch) is true. Default unimplemented (only ever
    /// called when the flag is set).
    fn predict_batch(&self, _inputs: Vec<Value>) -> Result<Vec<Value>> {
        Err(crate::Error::model("predict_batch is not implemented"))
    }

    /// Whether this model defines the batched corner ([`predict_batch`](Self::predict_batch)).
    fn has_batch(&self) -> bool {
        false
    }

    /// Batched chunk corner: N assembled lane inputs → N action *chunks* (leading
    /// axis = chunk) in a single call. Preferred for a vectorized chunked route when
    /// [`has_chunk_batch`](Self::has_chunk_batch) is true. `horizon` is the replay horizon `h` (see
    /// [`predict_chunk`](Self::predict_chunk)). Default unimplemented (gated by the
    /// flag).
    fn predict_chunk_batch(&self, _inputs: Vec<Value>, _horizon: u32) -> Result<Vec<Value>> {
        Err(crate::Error::model(
            "predict_chunk_batch is not implemented",
        ))
    }

    /// Whether this model defines the batched chunk corner ([`predict_chunk_batch`](Self::predict_chunk_batch)).
    fn has_chunk_batch(&self) -> bool {
        false
    }

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

    /// Fires when an episode ends (structurally-discovered model hook), driven by
    /// the explicit `ResetAdapter` op. The engine separately evicts that
    /// episode's frame buffers. There is no episode-*begin* hook: per-episode
    /// state is lazy-seeded on first predict, so a stateful model resets its
    /// state here at episode end rather than at a (no-longer-signalled) begin.
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
    /// Runtime-chosen replay horizon, set by the engine from the `ConfigureRoute`
    /// pin (1 = no chunking). Defaulted to 1 by [`new`](RouteConfig::new); the
    /// resolver builds the spec-derived config and the engine stamps the horizon
    /// on top, since it is a runtime decision, not part of the model spec.
    pub(crate) action_horizon: u32,
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
            // Spec-derived default; the engine overwrites it with the route's
            // runtime-pinned action_horizon at ConfigureRoute.
            action_horizon: 1,
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
