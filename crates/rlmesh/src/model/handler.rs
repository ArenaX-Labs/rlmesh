use std::sync::Arc;

use async_trait::async_trait;

use super::types::ModelObservation;
use crate::{Result, spaces};

/// Resolves per-env adapter state (e.g. an env→model adapter) when an adapter is
/// resolved.
///
/// Obtained once from [`ModelHandler::route_setup`] when serving begins and
/// shared (`Arc`) across every env, so the server can run it at `ResolveAdapter`
/// **without** taking the predict-serialization lock: resolving one env's adapter
/// never blocks on an in-flight `predict` on another. Implementations must
/// therefore synchronize their own state. Resolution happens before any `predict`
/// on the env, and per-env ordering guarantees an adapter is fully resolved
/// before that env's first predict — so an adapter is never re-resolved while its
/// own predict is in flight.
#[async_trait]
pub trait ModelRouteSetup: Send + Sync {
    /// Resolve and cache the adapter for `env_id` from its `env_contract`.
    /// Returning an error fails adapter resolution, so the client never predicts
    /// against an unresolved adapter. Idempotent upsert: a later call updates it.
    ///
    /// `execution_horizon` is how many actions of each predicted chunk the runtime
    /// executes before re-planning, pinned on `ResolveAdapter` (1 = no chunking). The
    /// setup caches it: the model returns its native chunk and the runtime executes a
    /// prefix of it, so an autoregressive head can read the value to decode exactly
    /// that many.
    async fn resolve_adapter(
        &self,
        env_id: &str,
        env_contract: &spaces::EnvContract,
        execution_horizon: u32,
    ) -> Result<()>;

    /// Tear down the adapter cached for `env_id` at `ReleaseAdapter`, so a
    /// long-lived server does not retain per-env state for every session it ever
    /// served. Defaults to a no-op.
    async fn release_adapter(&self, _env_id: &str) -> Result<()> {
        Ok(())
    }
}

/// One predict's per-lane action plus any open-loop chunk replay frames.
///
/// `actions` is frame 0 — one action per lane (`len == num_envs`), applied this
/// step. `replay` is the future-step frames the runtime buffers and applies
/// WITHOUT re-calling the model (action chunking): `replay[j]` is the per-lane
/// actions for future step `j + 1` (each `len == num_envs`). An empty `replay`
/// means the model is not chunking — one action this step, re-plan next step.
pub struct PredictFrames {
    /// Frame 0: one action per lane, applied this step.
    pub actions: Vec<spaces::SpaceValue>,
    /// Future-step frames (`replay[step][lane]`); empty when not chunking.
    pub replay: Vec<Vec<spaces::SpaceValue>>,
}

/// User policy plus episode lifecycle hooks.
///
/// Implement [`predict`](ModelHandler::predict) to map an observation to encoded
/// action bytes. The default hooks let stateful policies track resets, episode
/// ends, and shutdown. Drive the handler with
/// [`ModelWorker::run_local`](crate::ModelWorker::run_local) or host it with
/// [`ModelWorker::serve`](crate::ModelWorker::serve).
#[async_trait]
pub trait ModelHandler: Send {
    /// Produce an action for `observation`.
    ///
    /// Read the observation with
    /// [`decoded_lanes`](ModelObservation::decoded_lanes) (or
    /// [`decoded`](ModelObservation::decoded) for a single-env route) and return
    /// **one typed action per row** — `Vec` length `== observation.num_envs`
    /// (`== route.episode_ids.len()`); a single-env route returns a 1-element
    /// `Vec`. The codec turns the typed values into wire leaves; policy code
    /// never touches bytes.
    ///
    /// Return [`Error::model`](crate::Error::model) or
    /// [`Error::model_recoverable`](crate::Error::model_recoverable) when the
    /// policy declines a request.
    ///
    /// # Concurrency contract (pipelined predict)
    ///
    /// The served model endpoint pipelines Join-stream requests, so responses
    /// may complete out of arrival order. The handler itself is never invoked
    /// concurrently: `predict` and the lifecycle hooks take `&mut self`, and
    /// the server holds a per-handler mutex across each call.
    ///
    /// Per-route lifecycle order is preserved. Calls for different routes may
    /// interleave, still one at a time. The `model.concurrent_predict.v1`
    /// handshake capability advertises this pipelining to clients.
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<spaces::SpaceValue>>;

    /// Produce an action for `observation` plus any open-loop chunk replay frames
    /// (action chunking).
    ///
    /// The default emits no replay frames — behaviorally identical to
    /// [`predict`](ModelHandler::predict) — so a non-chunking handler needs no
    /// change. The stateful engine overrides this to split a chunked policy's
    /// output into per-step frames. The runtime driver calls this and replays the
    /// frames itself; the served endpoint packs them into the ordered
    /// `PredictResponse.actions` list (frame 0 first, replay frames after).
    /// `PredictFrames::actions` keeps the same `== num_envs` length contract as
    /// `predict`.
    async fn predict_chunked(&mut self, observation: ModelObservation) -> Result<PredictFrames> {
        Ok(PredictFrames {
            actions: self.predict(observation).await?,
            replay: Vec::new(),
        })
    }

    /// Produce actions for a batch of routed observations in one call.
    ///
    /// The server calls this for a `GroupedPredictRequest` — a control-plane-
    /// grouped batch where each observation belongs to a *different* configured
    /// route (and so a different env spec/adapter). The default fans out to
    /// [`predict`](ModelHandler::predict) per group, sequentially, which is
    /// behaviorally identical to handling each group as its own predict. A
    /// handler overrides this to fuse the groups into ONE forward pass (e.g. a
    /// single batched GPU inference across env types) — this is the only seam a
    /// fusing model must implement.
    ///
    /// The returned `Vec` aligns 1:1 and in order with `observations`; each
    /// element is that group's own `Result`, so one group's failure is reported
    /// per-group and never sinks the others. An override MUST preserve that
    /// length and order.
    async fn predict_grouped(
        &mut self,
        observations: Vec<ModelObservation>,
    ) -> Vec<Result<Vec<spaces::SpaceValue>>> {
        let mut results = Vec::with_capacity(observations.len());
        for observation in observations {
            results.push(self.predict(observation).await);
        }
        results
    }

    /// Per-route setup invoked at `ConfigureRoute`, before any `predict` on the
    /// route. Returns a cheaply-cloned, independently-synchronized handle (or
    /// `None` for no per-route setup), obtained once when serving begins so the
    /// server runs it **off** the predict-serialization lock — see
    /// [`ModelRouteSetup`].
    ///
    /// A spec'd model returns a setup that resolves and caches its env→model
    /// adapter per route (from the contract's spaces and adapter tags) for
    /// `predict` to apply. Defaults to `None`. Only the served path configures
    /// routes; the in-process [`run_local`](crate::ModelWorker::run_local) path
    /// never calls it.
    fn route_setup(&self) -> Option<Arc<dyn ModelRouteSetup>> {
        None
    }

    /// Drop per-episode policy/adapter state for the given episodes (the explicit
    /// `ResetAdapter` op). The runtime fires this when episodes end, keyed by
    /// `env_id`; with `episode_ids` empty it means "evict ALL of this env's
    /// episode state". The adapter itself stays resolved (see
    /// [`ModelRouteSetup::release_adapter`] for full teardown).
    ///
    /// Defaults to a no-op. Because episode ids never repeat (UUIDv7), a missed
    /// `reset_adapter` only leaks memory — it can never alias a new episode — so
    /// a stateful policy lazy-seeds per-episode state on first `predict` and
    /// evicts it here, with no position-diffing.
    async fn reset_adapter(&mut self, _env_id: &str, _episode_ids: Vec<String>) -> Result<()> {
        Ok(())
    }

    /// Called once when the worker/session shuts down. Defaults to a no-op.
    async fn on_close(&mut self) -> Result<()> {
        Ok(())
    }
}
