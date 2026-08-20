//! [`RouteState`]: the per-route bookkeeping the driver advances each step, and
//! the request messages it builds from that state.

use prost::bytes::Bytes;
use rlmesh_proto::model::v1::{
    AdapterContext, EpisodeInfo, PredictRequest, ReleaseAdapterRequest, ResetAdapterRequest,
};
use rlmesh_proto::spaces::v1::SpaceValue;

use std::collections::HashMap;

use crate::episodes::{EpisodeRecord, EpisodeRecordRegistry};
use crate::hooks::RuntimeEnvContext;
use crate::spec::{EpisodeSummary, RuntimeSessionSpec};

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
    episode_summaries: Vec<EpisodeSummary>,
    /// The explicit seed each live episode was reset with, keyed by episode id;
    /// drained into that episode's summary at completion.
    seed_by_episode: HashMap<String, i64>,
    /// How many of the spec's `episode_seeds` have been claimed (episode-start
    /// order).
    seed_cursor: usize,
}

impl RouteState {
    pub(crate) fn new(spec: &RuntimeSessionSpec) -> Self {
        let slots = (0..spec.num_envs.max(1))
            .map(|index| SlotState {
                env_index: index.try_into().unwrap_or(i32::MAX),
                episode: None,
                step: 0,
                reset: true,
                cumulative_reward: 0.0,
                started_at_ns: now_unix_ns(),
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
            episode_summaries: Vec::new(),
            seed_by_episode: HashMap::new(),
            seed_cursor: 0,
        }
    }

    /// Claim the next `lanes` explicit reset seeds from `episode_seeds`
    /// (episode-start order). Once the list cannot cover a whole reset batch the
    /// remaining episodes run unseeded (an all-or-nothing batch: `ResetRequest`
    /// seeds align positionally with the lanes being reset).
    pub(crate) fn claim_episode_seeds(&mut self, episode_seeds: &[i64], lanes: usize) -> Vec<i64> {
        let remaining = episode_seeds.len().saturating_sub(self.seed_cursor);
        if remaining < lanes {
            if remaining > 0 {
                tracing::warn!(
                    remaining,
                    lanes,
                    "episode_seeds cannot cover this reset batch; the remaining \
                     seeds are discarded and further episodes reset unseeded"
                );
            }
            self.seed_cursor = episode_seeds.len();
            return Vec::new();
        }
        let claimed = episode_seeds[self.seed_cursor..self.seed_cursor + lanes].to_vec();
        self.seed_cursor += lanes;
        claimed
    }

    /// Remember which explicit seed each episode in a reset batch received, so
    /// its completion summary can report it. No-op for an unseeded batch.
    pub(crate) fn note_episode_seeds(&mut self, episode_ids: &[String], seeds: &[i64]) {
        for (episode_id, seed) in episode_ids.iter().zip(seeds) {
            self.seed_by_episode.insert(episode_id.clone(), *seed);
        }
    }

    /// The explicit seed `episode_id` was reset with, if any (drained: each
    /// episode completes once).
    pub(crate) fn take_episode_seed(&mut self, episode_id: &str) -> Option<i64> {
        self.seed_by_episode.remove(episode_id)
    }

    /// Record one completed episode's summary (completion order) for the
    /// session report.
    pub(crate) fn record_episode_summary(&mut self, summary: EpisodeSummary) {
        self.episode_summaries.push(summary);
    }

    /// Drain the recorded episode summaries into the returned report.
    pub(crate) fn take_episode_summaries(&mut self) -> Vec<EpisodeSummary> {
        std::mem::take(&mut self.episode_summaries)
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

    pub(crate) fn record_step(&mut self, rewards: &[f64]) {
        self.total_steps += 1;
        for (index, slot) in self.slots.iter_mut().enumerate() {
            slot.step += 1;
            slot.reset = false;
            slot.cumulative_reward += rewards.get(index).copied().unwrap_or(0.0);
        }
    }

    pub(crate) fn complete_episode(&mut self, episode_id: &str) -> Option<EpisodeRecord> {
        self.total_episodes += 1;
        self.records.record_for(episode_id).cloned()
    }

    pub(crate) fn seed_for_episode(&self, episode_id: &str) -> Option<i64> {
        self.seed_by_episode.get(episode_id).copied()
    }

    pub(crate) fn predict_request(
        &mut self,
        observation: Option<Vec<Bytes>>,
        phase: RequestPhase,
    ) -> PredictRequest {
        let episode_info = self
            .episode_ids()
            .into_iter()
            .map(|episode_id| {
                let seed = self.seed_for_episode(&episode_id);
                EpisodeInfo { episode_id, seed }
            })
            .collect();
        PredictRequest {
            context: Some(AdapterContext {
                session_id: self.session_id().to_string(),
                env_id: self.env_id().to_string(),
                request_id: self.next_request_id(phase.as_str()),
            }),
            observation: observation.map(leaves_value),
            episode_info,
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

    pub(crate) fn slots(&self) -> &[SlotState] {
        &self.slots
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
                slot.cumulative_reward = 0.0;
                slot.started_at_ns = now_unix_ns();
            }
        }
    }
}

/// Current wall-clock time as unix nanoseconds (saturating; the epoch is
/// always in the past on a sane clock). The single clock both the per-slot
/// episode-start stamp and the driver's cap check read, so elapsed times are
/// always computed against the same convention.
pub(crate) fn now_unix_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| i64::try_from(elapsed.as_nanos()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}
