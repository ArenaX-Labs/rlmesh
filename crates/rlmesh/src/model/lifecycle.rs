use std::collections::HashMap;

use super::handler::ModelHandler;
use super::types::{ModelEpisodeEnd, ModelObservation};
use crate::Result;

pub(super) async fn finish_lifecycle<H>(
    handler: &mut H,
    active_episodes: &mut HashMap<(String, i32), String>,
) -> Result<()>
where
    H: ModelHandler,
{
    let mut episode_ends = active_episodes
        .drain()
        .map(|((_route_key, env_index), episode_id)| ModelEpisodeEnd {
            episode_id,
            env_index,
        })
        .collect::<Vec<_>>();
    episode_ends.sort_by(|left, right| {
        left.env_index
            .cmp(&right.env_index)
            .then_with(|| left.episode_id.cmp(&right.episode_id))
    });

    for episode_end in episode_ends {
        handler.on_episode_end(episode_end).await?;
    }

    Ok(())
}

pub(super) async fn update_lifecycle<H>(
    handler: &mut H,
    active_episodes: &mut HashMap<(String, i32), String>,
    observation: &ModelObservation,
) -> Result<()>
where
    H: ModelHandler,
{
    let route_key = route_state_key(observation);
    if observation.num_envs > 1 {
        let mut should_emit_reset = observation.reset;
        let episode_ids = observation.route.episode_ids();
        let episode_ids = if episode_ids.is_empty() {
            vec![observation.route.primary_episode_id().to_string()]
        } else {
            episode_ids
        };

        for (env_index, episode_id) in episode_ids.into_iter().enumerate() {
            let env_index = env_index as i32;
            if let Some(previous_episode) =
                active_episodes.insert((route_key.clone(), env_index), episode_id.clone())
                && previous_episode != episode_id
            {
                handler
                    .on_episode_end(ModelEpisodeEnd {
                        episode_id: previous_episode,
                        env_index,
                    })
                    .await?;
                should_emit_reset = true;
            }
        }

        if should_emit_reset {
            handler.on_reset(observation).await?;
        }
    } else {
        let current_episode = observation.route.primary_episode_id().to_string();

        if observation.reset {
            if let Some(previous_episode) = active_episodes.insert(
                (route_key.clone(), observation.route.primary_env_index()),
                current_episode.clone(),
            ) && previous_episode != current_episode
            {
                handler
                    .on_episode_end(ModelEpisodeEnd {
                        episode_id: previous_episode,
                        env_index: observation.route.primary_env_index(),
                    })
                    .await?;
            }

            handler.on_reset(observation).await?;
        } else {
            active_episodes
                .entry((route_key, observation.route.primary_env_index()))
                .or_insert_with(|| current_episode.clone());
        }
    }
    Ok(())
}

fn route_state_key(observation: &ModelObservation) -> String {
    format!(
        "{}:{}",
        observation.route.session_id, observation.route.route_id
    )
}
