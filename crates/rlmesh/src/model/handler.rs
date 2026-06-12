use async_trait::async_trait;

use super::types::{ModelEpisodeEnd, ModelObservation};
use crate::{Result, spaces};

#[async_trait]
pub trait ModelHandler: Send {
    /// Produce an action for `observation`.
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

    async fn on_reset(&mut self, _observation: &ModelObservation) -> Result<()> {
        Ok(())
    }

    async fn on_episode_end(&mut self, _event: ModelEpisodeEnd) -> Result<()> {
        Ok(())
    }

    async fn on_close(&mut self) -> Result<()> {
        Ok(())
    }
}
