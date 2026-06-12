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
/// See [`predict`](ModelHandler::predict) for the single-flight concurrency
/// contract.
#[async_trait]
pub trait ModelHandler: Send {
    /// Produce an action for `observation`.
    ///
    /// Returns the encoded action bytes as a
    /// [`BinaryPayload`](crate::spaces::BinaryPayload).
    ///
    /// # Concurrency contract (single-flight)
    ///
    /// The served model endpoint is **single-flight per connection**:
    /// `predict` (and the other lifecycle hooks) are invoked one at a time, in
    /// stream order, for a given Join stream — the server awaits each request to
    /// completion before reading the next, and the `&mut self` receiver
    /// statically prevents concurrent calls. A handler therefore does not need
    /// internal synchronization for its own state, but must not assume that
    /// multiple predicts can be in flight on one connection. To increase
    /// throughput, run multiple connections (each is independently
    /// single-flight). Pipelined/concurrent predict on a single connection is
    /// not yet supported (see the crate docs).
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
