//! The random baseline (`RANDOM_SAMPLE`) as a model a run or a binding
//! drives.

use std::collections::HashMap;

use async_trait::async_trait;
use rlmesh_spaces::SpaceValue;

use super::handler::ModelHandler;
use super::types::ModelObservation;
use crate::{Error, Result};

/// Samples the env's action space, reproducibly per predict of a seeded
/// episode.
#[derive(Default)]
pub struct RandomSampleHandler {
    /// Each live episode's predicts so far.
    predicts: HashMap<String, u64>,
}

#[async_trait]
impl ModelHandler for RandomSampleHandler {
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<SpaceValue>> {
        let space = observation
            .env_contract
            .as_ref()
            .and_then(|contract| contract.action_space.clone())
            .ok_or_else(|| Error::Internal("the env has no action space".into()))?;
        observation
            .route
            .episodes
            .iter()
            .map(|episode| {
                let index = self.predicts.entry(episode.episode_id.clone()).or_default();
                let seed = match episode.seed {
                    Some(seed) => crate::predict_seed(seed, *index) as u64,
                    None => uuid::Uuid::new_v4().as_u64_pair().0,
                };
                *index += 1;
                rlmesh_spaces::sample_seeded(&space, seed)
                    .map_err(|err| Error::Internal(err.to_string()))
            })
            .collect()
    }

    async fn reset_adapter(&mut self, _env_id: &str, episode_ids: Vec<String>) -> Result<()> {
        if episode_ids.is_empty() {
            self.predicts.clear();
        }
        for id in episode_ids {
            self.predicts.remove(&id);
        }
        Ok(())
    }
}
