use std::sync::Arc;

use async_trait::async_trait;

use super::types::{ModelEpisodeEnd, ModelLaneReset, ModelObservation};
use crate::{Result, spaces};

/// Resolves per-route state (e.g. an env→model adapter) when a route is
/// configured.
///
/// Obtained once from [`ModelHandler::route_setup`] when serving begins and
/// shared (`Arc`) across every route, so the server can run it at
/// `ConfigureRoute` **without** taking the predict-serialization lock:
/// configuring one route never blocks on an in-flight `predict` on another.
/// Implementations must therefore synchronize their own state. Resolution
/// happens before any `predict` on the route, and per-route ordering guarantees
/// a route is fully configured before that route's first predict — so a route is
/// never reconfigured while its own predict is in flight.
#[async_trait]
pub trait ModelRouteSetup: Send + Sync {
    /// Resolve and cache state for `route_key` from its `env_contract`. Returning
    /// an error fails route configuration, so the client never predicts against
    /// unresolved route state.
    async fn configure_route(
        &self,
        route_key: &str,
        env_contract: &spaces::EnvContract,
    ) -> Result<()>;

    /// Tear down state cached for `route_key` at `CloseRoute`, so a long-lived
    /// server does not retain per-route state for every session it ever served.
    /// Defaults to a no-op.
    async fn close_route(&self, _route_key: &str) -> Result<()> {
        Ok(())
    }
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
    /// **one typed action per lane** — `Vec` length `== observation.num_envs`
    /// (`== PredictContext.slots.len()`); a single-env route returns a 1-element
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

    /// Called when an episode begins, before its first `predict`.
    ///
    /// Defaults to a no-op. For a single (non-vectorized) env, use it to reset
    /// per-episode policy state.
    ///
    /// # Vectorized envs: fires on *any* lane's reset, not just whole-vector
    ///
    /// Under a vectorized env this hook fires whenever **any** lane's episode
    /// rolls — at the initial reset and at every NEXT_STEP autoreset boundary —
    /// receiving the whole-batch observation, **not** only when all lanes reset
    /// together. A handler that clears *all* lanes' policy state here will wipe
    /// the still-running lanes each time a single lane rolls, corrupting their
    /// in-flight episodes.
    ///
    /// So a stateful vectorized model must reset per-lane state in
    /// [`on_lane_reset`](ModelHandler::on_lane_reset) (which carries the
    /// `env_index` of the lane that rolled) and treat `on_reset` only as a coarse
    /// "something reset" signal. Restricting `on_reset` to fire only on a true
    /// whole-vector reset is a deliberate semantic change deferred to a future
    /// release; until then, prefer `on_lane_reset` for per-lane affinity.
    async fn on_reset(&mut self, _observation: &ModelObservation) -> Result<()> {
        Ok(())
    }

    /// Records the route key whose lifecycle is about to be processed, before
    /// any per-lane [`on_lane_reset`](ModelHandler::on_lane_reset) for it. The
    /// per-lane event carries only an `env_index`, so a handler that needs the
    /// route (e.g. to resolve a per-route adapter) captures it here. Defaults to
    /// a no-op.
    async fn enter_route(&mut self, _route_key: &str) -> Result<()> {
        Ok(())
    }

    /// Called when a single lane's episode rolls (a per-lane reset edge),
    /// carrying the `env_index`. Fires once per lane whose episode id changed —
    /// at the initial reset and at each NEXT_STEP autoreset boundary. Defaults
    /// to a no-op; a stateful per-lane adapter resets exactly that lane's state.
    /// The route is the one most recently named by
    /// [`enter_route`](ModelHandler::enter_route).
    async fn on_lane_reset(&mut self, _event: ModelLaneReset) -> Result<()> {
        Ok(())
    }

    /// Called when an episode ends. Defaults to a no-op.
    async fn on_episode_end(&mut self, _event: ModelEpisodeEnd) -> Result<()> {
        Ok(())
    }

    /// Called once when the worker/session shuts down. Defaults to a no-op.
    async fn on_close(&mut self) -> Result<()> {
        Ok(())
    }
}
