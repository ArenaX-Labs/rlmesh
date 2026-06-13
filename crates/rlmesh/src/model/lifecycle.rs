use std::collections::HashMap;

use super::handler::ModelHandler;
use super::types::{ModelEpisodeEnd, ModelLaneReset, ModelObservation};
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

pub(super) async fn finish_route_lifecycle<H>(
    handler: &mut H,
    active_episodes: &mut HashMap<(String, i32), String>,
    route_key: &str,
) -> Result<()>
where
    H: ModelHandler,
{
    let mut episode_ends = active_episodes
        .iter()
        .filter_map(|((active_route_key, env_index), episode_id)| {
            (active_route_key == route_key).then_some(ModelEpisodeEnd {
                episode_id: episode_id.clone(),
                env_index: *env_index,
            })
        })
        .collect::<Vec<_>>();
    episode_ends.sort_by(|left, right| {
        left.env_index
            .cmp(&right.env_index)
            .then_with(|| left.episode_id.cmp(&right.episode_id))
    });

    for episode_end in episode_ends {
        active_episodes.remove(&(route_key.to_string(), episode_end.env_index));
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

        for slot in &observation.route.slots {
            let env_index = slot.env_index;
            let episode_id = slot.episode_id.clone();
            let rolled =
                match active_episodes.insert((route_key.clone(), env_index), episode_id.clone()) {
                    Some(previous_episode) if previous_episode != episode_id => {
                        handler
                            .on_episode_end(ModelEpisodeEnd {
                                episode_id: previous_episode,
                                env_index,
                            })
                            .await?;
                        true
                    }
                    _ => false,
                };
            // Per-lane reset edge: a lane whose episode id rolled, or every lane
            // at the initial whole-vector reset.
            if rolled || observation.reset {
                handler
                    .on_lane_reset(ModelLaneReset {
                        episode_id,
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
        let env_index = observation.route.primary_env_index();

        if observation.reset {
            if let Some(previous_episode) =
                active_episodes.insert((route_key.clone(), env_index), current_episode.clone())
                && previous_episode != current_episode
            {
                handler
                    .on_episode_end(ModelEpisodeEnd {
                        episode_id: previous_episode,
                        env_index,
                    })
                    .await?;
            }

            // Single env is a lane of one: surface the same per-lane reset edge so
            // a stateful single-env adapter resets through the one sink.
            handler
                .on_lane_reset(ModelLaneReset {
                    episode_id: current_episode.clone(),
                    env_index,
                })
                .await?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::{ModelRouteContext, ModelRouteSlot};
    use crate::spaces;

    #[derive(Default)]
    struct RecordingHandler {
        resets: usize,
        episode_ends: Vec<ModelEpisodeEnd>,
        lane_resets: Vec<ModelLaneReset>,
    }

    #[async_trait::async_trait]
    impl ModelHandler for RecordingHandler {
        async fn predict(
            &mut self,
            _observation: ModelObservation,
        ) -> crate::Result<spaces::BinaryPayload> {
            Ok(spaces::BinaryPayload { data: Vec::new() })
        }

        async fn on_reset(&mut self, _observation: &ModelObservation) -> crate::Result<()> {
            self.resets += 1;
            Ok(())
        }

        async fn on_lane_reset(&mut self, event: ModelLaneReset) -> crate::Result<()> {
            self.lane_resets.push(event);
            Ok(())
        }

        async fn on_episode_end(&mut self, event: ModelEpisodeEnd) -> crate::Result<()> {
            self.episode_ends.push(event);
            Ok(())
        }
    }

    fn observation(slots: Vec<(i32, &str)>, reset: bool, num_envs: usize) -> ModelObservation {
        ModelObservation {
            observation: None,
            route: ModelRouteContext {
                session_id: "session".to_string(),
                route_id: "route".to_string(),
                request_id: "request".to_string(),
                slots: slots
                    .into_iter()
                    .map(|(env_index, episode_id)| ModelRouteSlot {
                        env_index,
                        episode_id: episode_id.to_string(),
                        ..Default::default()
                    })
                    .collect(),
            },
            reset,
            num_envs,
            env_contract: None,
        }
    }

    #[tokio::test]
    async fn multi_env_lifecycle_uses_slot_env_index() {
        let mut handler = RecordingHandler::default();
        let mut active_episodes = HashMap::new();

        update_lifecycle(
            &mut handler,
            &mut active_episodes,
            &observation(vec![(2, "episode-a"), (3, "episode-b")], true, 4),
        )
        .await
        .unwrap();
        update_lifecycle(
            &mut handler,
            &mut active_episodes,
            &observation(
                vec![
                    (0, "episode-c"),
                    (1, "episode-d"),
                    (2, "episode-a"),
                    (3, "episode-b"),
                ],
                false,
                4,
            ),
        )
        .await
        .unwrap();

        assert_eq!(handler.resets, 1);
        assert!(handler.episode_ends.is_empty());
        // The initial whole-vector reset fired a per-lane reset for each active
        // lane (2 and 3), carrying their env_index; the second observation rolled
        // no existing lane so it added none.
        let mut reset_lanes: Vec<i32> = handler
            .lane_resets
            .iter()
            .map(|reset| reset.env_index)
            .collect();
        reset_lanes.sort_unstable();
        assert_eq!(reset_lanes, vec![2, 3]);
        assert_eq!(
            active_episodes.get(&("session:route".to_string(), 2)),
            Some(&"episode-a".to_string())
        );
        assert_eq!(
            active_episodes.get(&("session:route".to_string(), 3)),
            Some(&"episode-b".to_string())
        );
    }
}
