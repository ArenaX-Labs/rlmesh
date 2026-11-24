use std::sync::Arc;

use async_trait::async_trait;

use super::types::{EpisodeInfo, ModelObservation};
use crate::{Result, binding::spaces};

/// Resolves per-env adapter state (e.g. an env→model adapter) when an adapter is
/// resolved.
///
/// Obtained once from [`ModelHandler::route_setup`] and shared across every env, it runs
/// without the predict lock, so implementations synchronize their own state. An env's
/// adapter is resolved before that env's first predict and never re-resolved while one of
/// its predicts is in flight.
#[async_trait]
pub trait ModelRouteSetup: Send + Sync {
    /// Resolve and cache the adapter for `env_id` from its `env_contract`, and answer what
    /// the resolved route needs back. An error fails resolution, so the route never predicts.
    /// A later call for the same `env_id` replaces the adapter.
    async fn resolve_adapter(
        &self,
        env_id: &str,
        env_contract: &spaces::EnvContract,
        options: ResolveOptions,
    ) -> Result<RouteNeeds>;

    /// Drop the route-local per-episode adapter state (an engine's episode-keyed frame
    /// windows) for `env_id`'s ended `episode_ids` at `ResetAdapter` — empty means all of
    /// the env's. Runs off the predict lock, before [`ModelHandler::reset_adapter`], and never
    /// overlaps a predict on the same env. Defaults to a no-op.
    async fn reset_adapter(&self, _env_id: &str, _episode_ids: &[String]) -> Result<()> {
        Ok(())
    }

    /// Tear down the adapter cached for `env_id` at `ReleaseAdapter`, so a long-lived
    /// server does not retain per-env state for every session it ever served.
    async fn release_adapter(&self, _env_id: &str) -> Result<()> {
        Ok(())
    }
}

/// The runtime-pinned knobs a route resolves against, carried on
/// `ResolveAdapter`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResolveOptions {
    /// How many actions of each predicted chunk the runtime executes before re-planning (1
    /// = no chunking). The model returns its native chunk and the runtime executes this
    /// prefix of it. At most
    /// [`MAX_EXECUTION_HORIZON`](rlmesh_adapters::v1::MAX_EXECUTION_HORIZON).
    pub execution_horizon: u32,
    /// Whether the runtime will deliver observation-history frames on `Predict` (every env
    /// step it executed from a replayed chunk without predicting). Without it, a stacked
    /// adapter is refused at a horizon above 1.
    pub delivers_history: bool,
    /// The workflow edition the run executes at, which a model served
    /// elsewhere is pinned to (`None`: unpinned).
    pub workflow_edition: Option<rlmesh_proto::Edition>,
}

/// What a resolved route needs from the runtime, answered on
/// `ResolveAdapterResponse`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RouteNeeds {
    /// The model's native chunk length K: how many per-step actions one chunk corner call
    /// returns.
    pub native_chunk: Option<u32>,
    /// Set when the runtime offered observation history
    /// ([`ResolveOptions::delivers_history`]) and this route's adapter keeps a
    /// frame window: the runtime must then deliver every replayed step as a
    /// history row on the next predict, stamped with steps.
    pub history: Option<HistoryNeeds>,
    /// What the route's adapter resolve tolerated or applied (a zero-filled
    /// camera, an allowed projection): carried into the run's result.
    pub advisories: Vec<rlmesh_adapters::v1::Advisory>,
}

/// The frame windows a route keeps, answered on `ResolveAdapterResponse.history`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HistoryNeeds {
    /// Canonical placement keys of the stacked inputs (informational: the
    /// runtime sends whole observations per history row).
    pub keys: Vec<String>,
    /// Whether the runtime may send pruned rows (only the keys' leaves). Always
    /// `false` today.
    pub prunable: bool,
}

/// One predict's per-lane action plus any open-loop chunk replay frames.
#[derive(Debug)]
pub struct PredictFrames {
    /// Frame 0: one action per lane, applied this step.
    pub actions: Vec<spaces::SpaceValue>,
    /// Future-step frames (`replay[step][lane]`), applied without calling the model again;
    /// `replay[j]` is for step `j + 1`. Empty when not chunking.
    pub replay: Vec<Vec<spaces::SpaceValue>>,
}

/// Engine-held per-episode adapter state at a model endpoint, as stamped on
/// predict responses: episodes with live frame-stack windows and the bytes
/// those windows hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeldState {
    /// Episodes with live frame-stack windows across all routes.
    pub episodes: u64,
    /// Bytes those windows hold.
    pub bytes: u64,
}

/// User policy plus episode lifecycle hooks.
///
/// The handler is never invoked concurrently: `predict` and the hooks take `&mut self`,
/// one call at a time.
#[async_trait]
pub trait ModelHandler: Send {
    /// Produce an action for `observation`.
    ///
    /// Read it with [`decoded_lanes`](ModelObservation::decoded_lanes) (or
    /// [`decoded`](ModelObservation::decoded) for a single-env route) and return one action
    /// per lane: the `Vec`'s length is `observation.num_envs`. Return
    /// [`Error::model`](crate::Error::model) or
    /// [`Error::model_recoverable`](crate::Error::model_recoverable) to decline a request.
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<spaces::SpaceValue>>;

    /// Produce an action for `observation` plus any open-loop chunk replay frames
    /// (action chunking). The default calls [`predict`](Self::predict) and emits no replay
    /// frames; [`PredictFrames::actions`] keeps its one-per-lane length.
    async fn predict_chunked(&mut self, observation: ModelObservation) -> Result<PredictFrames> {
        Ok(PredictFrames {
            actions: self.predict(observation).await?,
            replay: Vec::new(),
        })
    }

    /// Produce action frames for a batch of routed observations in one call.
    ///
    /// Each observation belongs to a different route, with its own adapter and horizon. The
    /// default calls [`predict_chunked`](Self::predict_chunked) per group in order; override
    /// it to fuse the groups into one forward. The result aligns 1:1 and in order with
    /// `observations`, each group with its own `Result`, so one group's failure does not
    /// fail the others.
    async fn predict_grouped(
        &mut self,
        observations: Vec<ModelObservation>,
    ) -> Vec<Result<PredictFrames>> {
        let mut results = Vec::with_capacity(observations.len());
        for observation in observations {
            results.push(self.predict_chunked(observation).await);
        }
        results
    }

    /// Whether [`predict_grouped`](Self::predict_grouped) fuses its groups into one
    /// forward, so a served model gains from coalescing requests into one call.
    fn fuses_groups(&self) -> bool {
        false
    }

    /// Engine-held per-episode adapter state (frame-stack windows) at this endpoint, summed
    /// across routes — the state that grows with concurrent episodes and shrinks on
    /// episode-end GC.
    fn held_state(&self) -> Option<HeldState> {
        None
    }

    /// Adapter time (obs assembly + action apply) inside the last predict-family call, in
    /// nanoseconds — the share of the handler's own work that was RLMesh adapter transform
    /// rather than the model's forward.
    fn take_adapter_ns(&mut self) -> u64 {
        0
    }

    /// What a model served elsewhere measured answering the last predict-family call: its
    /// own total and phases (decode, user, encode, queue, in flight), each its request's
    /// share of a batch it fused. Read-and-clear. `None` (the default) for a handler
    /// answering in process, which the caller times itself.
    fn take_endpoint_timing(&mut self) -> Option<(Option<u64>, rlmesh_proto::EndpointPhases)> {
        None
    }

    /// How many lanes rode the forward the last predict-family call was answered in, when
    /// this handler knows it better than its caller: one in front of a batching pool, which
    /// fused this call's groups with other callers' rows. Recorded as the `group.size`
    /// metric; read-and-clear. `None` (the default) records the number of groups in the call.
    fn take_group_size(&mut self) -> Option<u64> {
        None
    }

    /// Per-route setup invoked at `ResolveAdapter`, before any `predict` on the route.
    /// Obtained once when serving begins and run off the predict lock (see
    /// [`ModelRouteSetup`]). `None` (the default) means no per-route setup.
    fn route_setup(&self) -> Option<Arc<dyn ModelRouteSetup>> {
        None
    }

    /// Drop per-episode policy/adapter state for the given episodes (the explicit
    /// `ResetAdapter` op), fired when episodes end. Empty `episode_ids` means all of the
    /// env's episodes; the adapter stays resolved (see
    /// [`ModelRouteSetup::release_adapter`]). Runs under the predict lock, after
    /// [`ModelRouteSetup::reset_adapter`]. Episode ids never repeat, so a missed call only
    /// leaks memory. Defaults to a no-op.
    async fn reset_adapter(&mut self, _env_id: &str, _episode_ids: Vec<String>) -> Result<()> {
        Ok(())
    }

    /// Called once when the worker/session shuts down. Defaults to a no-op.
    async fn on_close(&mut self) -> Result<()> {
        Ok(())
    }

    /// The session a run drives in (a session address), before its route resolves: a model
    /// served elsewhere is invited into it, so its predicts stay there.
    async fn join(&mut self, _session: &str) -> Result<()> {
        Ok(())
    }

    /// The workflow editions a model served elsewhere declared in its session, after
    /// [`join`](Self::join): a run executes at the floor across the env, the model and
    /// itself.
    async fn workflow_editions(&mut self) -> Result<Option<rlmesh_proto::SessionOffer>> {
        Ok(None)
    }

    /// The options this model adds to the env resets that start its episodes, obtained once
    /// when a run begins and called off the predict lock (see [`ModelResetOptions`]).
    fn reset_options(&self) -> Option<Arc<dyn ModelResetOptions>> {
        None
    }
}

/// The options a model adds to the env reset that starts one of its episodes, merged
/// under the runtime's own reserved keys.
#[async_trait]
pub trait ModelResetOptions: Send + Sync {
    /// The options for `episode`, which a reset is about to start as trial
    /// `trial_index`; `None` adds none.
    async fn reset_options(
        &self,
        episode: &EpisodeInfo,
        trial_index: Option<u64>,
    ) -> Result<Option<rlmesh_spaces::MetaMap>>;
}

macro_rules! forward_handler {
    ($($ty:ty),*) => {$(
#[async_trait]
impl<H: ModelHandler + ?Sized> ModelHandler for $ty {
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<spaces::SpaceValue>> {
        (**self).predict(observation).await
    }

    async fn predict_chunked(&mut self, observation: ModelObservation) -> Result<PredictFrames> {
        (**self).predict_chunked(observation).await
    }

    async fn predict_grouped(
        &mut self,
        observations: Vec<ModelObservation>,
    ) -> Vec<Result<PredictFrames>> {
        (**self).predict_grouped(observations).await
    }

    fn fuses_groups(&self) -> bool {
        (**self).fuses_groups()
    }

    fn held_state(&self) -> Option<HeldState> {
        (**self).held_state()
    }

    fn take_adapter_ns(&mut self) -> u64 {
        (**self).take_adapter_ns()
    }

    fn take_endpoint_timing(&mut self) -> Option<(Option<u64>, rlmesh_proto::EndpointPhases)> {
        (**self).take_endpoint_timing()
    }

    fn take_group_size(&mut self) -> Option<u64> {
        (**self).take_group_size()
    }

    fn route_setup(&self) -> Option<Arc<dyn ModelRouteSetup>> {
        (**self).route_setup()
    }

    async fn reset_adapter(&mut self, env_id: &str, episode_ids: Vec<String>) -> Result<()> {
        (**self).reset_adapter(env_id, episode_ids).await
    }

    async fn on_close(&mut self) -> Result<()> {
        (**self).on_close().await
    }

    async fn join(&mut self, session: &str) -> Result<()> {
        (**self).join(session).await
    }

    async fn workflow_editions(&mut self) -> Result<Option<rlmesh_proto::SessionOffer>> {
        (**self).workflow_editions().await
    }

    fn reset_options(&self) -> Option<Arc<dyn ModelResetOptions>> {
        (**self).reset_options()
    }
}
    )*};
}

forward_handler!(Box<H>, &mut H);
