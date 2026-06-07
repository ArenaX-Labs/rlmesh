use async_trait::async_trait;

use super::types::{ModelEpisodeEnd, ModelObservation};
use crate::{Result, spaces};

#[async_trait]
pub trait ModelHandler: Send {
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
