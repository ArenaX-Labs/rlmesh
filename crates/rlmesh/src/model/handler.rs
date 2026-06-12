use async_trait::async_trait;

use super::types::{ModelEpisodeEnd, ModelObservation};
use crate::{Result, spaces};

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
    /// Returns encoded action bytes as a
    /// [`BinaryPayload`](crate::spaces::BinaryPayload).
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
