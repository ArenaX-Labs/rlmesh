use rlmesh_proto::model::v1::PredictSlot;

use crate::episodes::{EpisodeRecord, EpisodeRecordRegistry};
use crate::hooks::RuntimeRouteContext;
use crate::spec::RuntimeSessionSpec;

use super::{EpisodeState, RouteSnapshot, SlotState, StartedEpisode};

#[derive(Debug)]
pub(crate) struct RouteState {
    session_id: String,
    route_id: String,
    env_component_id: String,
    model_component_id: String,
    slots: Vec<SlotState>,
    request_seq: u64,
    total_steps: i64,
    total_episodes: i64,
    records: EpisodeRecordRegistry,
}

impl RouteState {
    pub(crate) fn new(spec: &RuntimeSessionSpec) -> Self {
        let slots = (0..spec.num_envs.max(1))
            .map(|index| SlotState {
                env_index: index.try_into().unwrap_or(i32::MAX),
                episode: None,
                step: 0,
                reset: true,
            })
            .collect();

        Self {
            session_id: spec.session_id.clone(),
            route_id: spec.route_id.clone(),
            env_component_id: spec.env_component_id.clone(),
            model_component_id: spec.model_component_id.clone(),
            slots,
            request_seq: 0,
            total_steps: 0,
            total_episodes: 0,
            records: EpisodeRecordRegistry::default(),
        }
    }

    pub(crate) fn session_id(&self) -> &str {
        &self.session_id
    }

    pub(crate) fn route_id(&self) -> &str {
        &self.route_id
    }

    pub(crate) fn env_component_id(&self) -> &str {
        &self.env_component_id
    }

    pub(crate) fn model_component_id(&self) -> &str {
        &self.model_component_id
    }

    pub(crate) fn route_context(&self) -> RuntimeRouteContext {
        RuntimeRouteContext {
            route_id: self.route_id.clone(),
            env_component_id: self.env_component_id.clone(),
            model_component_id: self.model_component_id.clone(),
        }
    }

    pub(crate) fn total_steps(&self) -> i64 {
        self.total_steps
    }

    pub(crate) fn total_episodes(&self) -> i64 {
        self.total_episodes
    }

    pub(crate) fn next_request_id(&mut self, phase: &str) -> String {
        self.request_seq += 1;
        // Include route_id: a session can fan out to multiple routes, and
        // request_seq restarts at 0 per RouteState, so omitting it would make
        // sibling routes emit identical request IDs.
        format!(
            "{}:{}:{}:{:06}",
            self.session_id, self.route_id, phase, self.request_seq
        )
    }

    pub(crate) fn slots(&self) -> Vec<PredictSlot> {
        self.slots
            .iter()
            .map(|slot| PredictSlot {
                env_index: slot.env_index,
                episode_id: slot
                    .episode
                    .as_ref()
                    .map(|episode| episode.episode_id.clone())
                    .unwrap_or_default(),
                step: slot.step,
                reset: slot.reset,
            })
            .collect()
    }

    pub(crate) fn snapshot(&self) -> RouteSnapshot {
        let episode_ids = self
            .slots
            .iter()
            .map(|slot| {
                slot.episode
                    .as_ref()
                    .map(|episode| episode.episode_id.clone())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let episode_record_ids = self
            .slots
            .iter()
            .map(|slot| {
                slot.episode
                    .as_ref()
                    .map(|episode| episode.episode_record_id.clone())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let primary = self.slots.first();
        RouteSnapshot {
            episode_id: episode_ids.first().cloned().unwrap_or_default(),
            episode_record_id: episode_record_ids.first().cloned().unwrap_or_default(),
            episode_ids,
            episode_record_ids,
            step: primary.map_or(0, |slot| slot.step),
            env_index: primary.map_or(0, |slot| slot.env_index),
            reset: primary.is_some_and(|slot| slot.reset),
        }
    }

    pub(crate) fn start_episodes(
        &mut self,
        episode_ids: Vec<String>,
        started_from_auto_reset: bool,
    ) -> Vec<StartedEpisode> {
        let (record_ids, started) = self
            .records
            .ensure_for_slots(&episode_ids, started_from_auto_reset);
        self.sync_slots(episode_ids, record_ids, true, started_from_auto_reset);
        started
            .into_iter()
            .map(|(episode_id, record)| StartedEpisode { episode_id, record })
            .collect()
    }

    pub(crate) fn observe_episode_ids(&mut self, episode_ids: Vec<String>) -> Vec<StartedEpisode> {
        let (record_ids, started) = self.records.ensure_for_slots(&episode_ids, true);
        self.sync_slots(episode_ids, record_ids, false, true);
        started
            .into_iter()
            .map(|(episode_id, record)| StartedEpisode { episode_id, record })
            .collect()
    }

    pub(crate) fn record_step(&mut self) {
        self.total_steps += 1;
        for slot in &mut self.slots {
            slot.step += 1;
            slot.reset = false;
        }
    }

    pub(crate) fn complete_episode(&mut self, episode_id: &str) -> Option<EpisodeRecord> {
        self.total_episodes += 1;
        self.records.record_for(episode_id).cloned()
    }

    fn sync_slots(
        &mut self,
        episode_ids: Vec<String>,
        record_ids: Vec<String>,
        reset_steps: bool,
        started_from_auto_reset: bool,
    ) {
        for (index, slot) in self.slots.iter_mut().enumerate() {
            let episode_id = episode_ids.get(index).cloned().unwrap_or_default();
            let episode_record_id = record_ids.get(index).cloned().unwrap_or_default();
            // Did this lane's episode id flip? A NEXT_STEP autoreset rolls the id
            // on a single lane at t+1; only that lane's step counter must reset.
            let previous_id = slot
                .episode
                .as_ref()
                .map(|episode| episode.episode_id.clone())
                .unwrap_or_default();
            let rolled = !episode_id.is_empty() && episode_id != previous_id;
            slot.episode = if episode_id.is_empty() {
                None
            } else {
                let record = self.records.record_for(&episode_id);
                Some(EpisodeState {
                    episode_id,
                    episode_record_id,
                    episode_index: record.map_or(0, |record| record.index),
                    started_from_auto_reset,
                })
            };
            // `reset_steps` force-resets every lane (a whole-vector reset); `rolled`
            // resets only the lane whose id changed (per-lane autoreset).
            if reset_steps || rolled {
                slot.step = 0;
                slot.reset = true;
            }
        }
    }
}
