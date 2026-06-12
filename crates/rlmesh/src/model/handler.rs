use async_trait::async_trait;

use super::types::{ModelEpisodeEnd, ModelObservation};
use crate::{Result, spaces};

/// A user-implemented model: a policy plus its episode-lifecycle hooks.
///
/// Implement [`predict`](ModelHandler::predict) to map an observation to an
/// encoded action; the default-provided hooks
/// ([`on_reset`](ModelHandler::on_reset),
/// [`on_episode_end`](ModelHandler::on_episode_end),
/// [`on_close`](ModelHandler::on_close)) let stateful policies track episode
/// boundaries. Drive a handler with
/// [`ModelWorker::run_local`](crate::ModelWorker::run_local) (in-process
/// against a remote env) or host it with
/// [`ModelWorker::serve`](crate::ModelWorker::serve).
///
/// Every hook returns [`Result`](crate::Result); return
/// [`Error::model`](crate::Error::model) /
/// [`Error::model_recoverable`](crate::Error::model_recoverable) to signal that
/// the model *declined* a request rather than hit an internal fault.
///
/// See [`predict`](ModelHandler::predict) for the concurrency contract.
#[async_trait]
pub trait ModelHandler: Send {
    /// Produce an action for `observation`.
    ///
    /// Returns the encoded action bytes as a
    /// [`BinaryPayload`](crate::spaces::BinaryPayload).
    ///
    /// # Concurrency contract (pipelined predict)
    ///
    /// The served model endpoint **pipelines** Join-stream requests: decode,
    /// encode, and the response pump overlap across requests, so a slow predict
    /// no longer head-of-line-blocks a faster later one and responses may
    /// complete out of arrival order. The handler itself, however, is **never**
    /// invoked concurrently: `predict` and the lifecycle hooks take `&mut self`,
    /// and the server holds a per-handler mutex across each call, so exactly one
    /// hook runs at a time. A handler therefore still does not need internal
    /// synchronization for its own state.
    ///
    /// **Per-route ordering is preserved**: for a given route, `on_reset` /
    /// `predict` / `on_episode_end` are invoked in the order the client sent the
    /// corresponding requests, and a route's `CloseRoute`/`Close` drain runs
    /// after every earlier predict on that route. Calls for *different* routes
    /// may interleave (still one at a time). To increase throughput further, run
    /// multiple connections. The `model.concurrent_predict.v1` handshake
    /// capability advertises this pipelining to clients.
    async fn predict(&mut self, observation: ModelObservation) -> Result<spaces::BinaryPayload>;

    /// Called when a new episode begins, before its first `predict`.
    ///
    /// Defaults to a no-op. Use it to reset per-episode policy state.
    async fn on_reset(&mut self, _observation: &ModelObservation) -> Result<()> {
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
