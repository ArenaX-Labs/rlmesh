//! [`RouteState`]: the per-route bookkeeping the driver advances each step, and
//! the request messages it builds from that state.

use prost::bytes::Bytes;
use rlmesh_proto::model::v1::{
    AdapterContext, PredictRequest, ReleaseAdapterRequest, ResetAdapterRequest,
};
use rlmesh_proto::spaces::v1::SpaceValue;

use crate::episodes::{EpisodeRecord, EpisodeRecordRegistry};
use crate::hooks::RuntimeEnvContext;
use crate::spec::RuntimeSessionSpec;

use super::{EpisodeState, RouteSnapshot, SlotState, StartedEpisode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestPhase {
    ResetObservation,
    StepObservation,
}

impl RequestPhase {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ResetObservation => "reset_observation",
            Self::StepObservation => "step_observation",
        }
    }
}

fn leaves_value(leaves: Vec<Bytes>) -> SpaceValue {
    SpaceValue { leaves }
}

#[derive(Debug)]
pub(crate) struct RouteState {
    session_id: String,
    env_id: String,
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
            env_id: spec.env_id.clone(),
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

    pub(crate) fn env_id(&self) -> &str {
        &self.env_id
    }

    pub(crate) fn env_component_id(&self) -> &str {
        &self.env_component_id
    }

    pub(crate) fn model_component_id(&self) -> &str {
        &self.model_component_id
    }

    pub(crate) fn env_context(&self) -> RuntimeEnvContext {
        RuntimeEnvContext {
            env_id: self.env_id.clone(),
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
        // env_id is globally unique (UUIDv7), so it alone disambiguates request
        // ids across every adapter; request_seq restarts at 0 per RouteState.
        format!("{}:{}:{:06}", self.env_id, phase, self.request_seq)
    }

    /// Ordered per-row episode ids — the self-describing batch. Row `i` belongs
    /// to `episode_ids()[i]`. Empty string for a lane with no active episode.
    pub(crate) fn episode_ids(&self) -> Vec<String> {
        self.slots
            .iter()
            .map(|slot| {
                slot.episode
                    .as_ref()
                    .map(|episode| episode.episode_id.clone())
                    .unwrap_or_default()
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

    pub(crate) fn predict_request(
        &mut self,
        observation: Option<Vec<Bytes>>,
        phase: RequestPhase,
    ) -> PredictRequest {
        let episode_ids = self.episode_ids();
        PredictRequest {
            context: Some(AdapterContext {
                session_id: self.session_id().to_string(),
                env_id: self.env_id().to_string(),
                request_id: self.next_request_id(phase.as_str()),
            }),
            observation: observation.map(leaves_value),
            episode_ids,
        }
    }

    pub(crate) fn reset_adapter_request(
        &mut self,
        episode_ids: Vec<String>,
    ) -> ResetAdapterRequest {
        ResetAdapterRequest {
            context: Some(AdapterContext {
                session_id: self.session_id().to_string(),
                env_id: self.env_id().to_string(),
                request_id: self.next_request_id("reset_adapter"),
            }),
            episode_ids,
        }
    }

    pub(crate) fn release_adapter_request(
        &mut self,
        reason: impl Into<String>,
    ) -> ReleaseAdapterRequest {
        ReleaseAdapterRequest {
            context: Some(AdapterContext {
                session_id: self.session_id().to_string(),
                env_id: self.env_id().to_string(),
                request_id: self.next_request_id("release_adapter"),
            }),
            reason: reason.into(),
        }
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
            let rolled = {
                let previous_id = slot
                    .episode
                    .as_ref()
                    .map(|episode| episode.episode_id.as_str())
                    .unwrap_or("");
                !episode_id.is_empty() && episode_id != previous_id
            };
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
