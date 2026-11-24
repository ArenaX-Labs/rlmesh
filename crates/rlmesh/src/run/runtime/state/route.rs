//! [`RouteState`]: the per-route bookkeeping the driver advances each step, and
//! the model observations it builds from that state.
//!
//! Slots are per lane (`env_index`). The driver works on groups of lanes (one
//! per lane for a lane endpoint, one for the whole vector otherwise), so every
//! accessor takes the group's slot `positions` and touches only those lanes.
//! Episode indices and seeds come from the route-global slot counter
//! ([`claim_slots`](RouteState::claim_slots)), so they depend on start order
//! alone, never on which group ran the episode.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bytes::Bytes;

use crate::model::types::{EpisodeInfo, ModelObservation, ModelRouteContext};

use crate::run::runtime::episodes::{EpisodeRecord, EpisodeRecordRegistry};
use crate::run::runtime::hooks::RuntimeEnvContext;
use crate::run::runtime::spec::{EpisodeSummary, RuntimeSessionSpec};

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
    seed_by_episode: HashMap<String, i64>,
    next_slot: u64,
    max_episodes: Option<u64>,
    trial_by_episode: HashMap<String, u64>,
    trial_cursor: u64,
    surplus: HashSet<String>,
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
                predict_ns: 0,
                step_ns: 0,
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
            next_slot: 0,
            max_episodes: spec.max_episodes,
            trial_by_episode: HashMap::new(),
            trial_cursor: 0,
            surplus: HashSet::new(),
        }
    }

    pub(crate) fn claim_trial_indices(&mut self, base: u64, lanes: usize) -> Vec<u64> {
        let start = base.saturating_add(self.trial_cursor);
        self.trial_cursor += lanes as u64;
        (0..lanes as u64).map(|offset| start + offset).collect()
    }

    pub(crate) fn note_episode_trials(&mut self, episode_ids: &[String], trials: &[u64]) {
        for (episode_id, trial) in episode_ids.iter().zip(trials) {
            self.trial_by_episode.insert(episode_id.clone(), *trial);
        }
    }

    pub(crate) fn trial_for_episode(&self, episode_id: &str) -> Option<u64> {
        self.trial_by_episode.get(episode_id).copied()
    }

    pub(crate) fn slot_position(&self, env_index: u32) -> Option<usize> {
        let env_index = i32::try_from(env_index).ok()?;
        self.slots
            .iter()
            .position(|slot| slot.env_index == env_index)
    }

    pub(crate) fn claim_slots(&mut self, count: usize, bounded: bool) -> Option<Vec<u64>> {
        let first = self.next_slot;
        let last = first + count as u64;
        if bounded && self.max_episodes.is_some_and(|max| last > max) {
            return None;
        }
        self.next_slot = last;
        Some((first..last).collect())
    }

    pub(crate) fn claim_slots_upto(&mut self, count: usize) -> Vec<u64> {
        let first = self.next_slot;
        let remaining = self
            .max_episodes
            .map_or(count as u64, |max| max.saturating_sub(first));
        let last = first + (count as u64).min(remaining);
        self.next_slot = last;
        (first..last).collect()
    }

    pub(crate) fn mark_surplus(&mut self, episode_id: &str) {
        self.surplus.insert(episode_id.to_string());
    }

    pub(crate) fn is_surplus(&self, episode_id: &str) -> bool {
        self.surplus.contains(episode_id)
    }

    pub(crate) fn all_surplus_at(&self, positions: &[usize]) -> bool {
        positions.iter().all(|&position| {
            self.slots
                .get(position)
                .and_then(|slot| slot.episode.as_ref())
                .is_some_and(|episode| self.surplus.contains(&episode.episode_id))
        })
    }

    pub(crate) fn slot_timings_ms(&self, env_index: u32) -> (Option<f64>, Option<f64>) {
        let Some(slot) = self
            .slot_position(env_index)
            .and_then(|p| self.slots.get(p))
        else {
            return (None, None);
        };
        if slot.step <= 0 {
            return (None, None);
        }
        let per_step = |ns: u64| Some(ns as f64 / 1e6 / slot.step as f64);
        (per_step(slot.predict_ns), per_step(slot.step_ns))
    }

    pub(crate) fn note_episode_seeds(&mut self, episode_ids: &[String], seeds: &[i64]) {
        for (episode_id, seed) in episode_ids.iter().zip(seeds) {
            self.seed_by_episode.insert(episode_id.clone(), *seed);
        }
    }

    pub(crate) fn record_episode_summary(&mut self, summary: EpisodeSummary) {
        self.episode_summaries.push(summary);
    }

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
            lane: None,
        }
    }

    pub(crate) fn group_context(&self, positions: &[usize], lane_group: bool) -> RuntimeEnvContext {
        RuntimeEnvContext {
            lane: if lane_group {
                positions
                    .first()
                    .and_then(|&position| self.slots.get(position))
                    .and_then(|slot| u32::try_from(slot.env_index).ok())
            } else {
                None
            },
            ..self.env_context()
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
        format!("{}:{}:{:06}", self.env_id, phase, self.request_seq)
    }

    pub(crate) fn slots_at(&self, positions: &[usize]) -> Vec<&SlotState> {
        positions
            .iter()
            .filter_map(|&position| self.slots.get(position))
            .collect()
    }

    pub(crate) fn episode_ids_at(&self, positions: &[usize]) -> Vec<String> {
        self.slots_at(positions)
            .into_iter()
            .map(|slot| {
                slot.episode
                    .as_ref()
                    .map(|episode| episode.episode_id.clone())
                    .unwrap_or_default()
            })
            .collect()
    }

    pub(crate) fn snapshot_at(&self, positions: &[usize]) -> RouteSnapshot {
        let slots = self.slots_at(positions);
        let episode_ids = slots
            .iter()
            .map(|slot| {
                slot.episode
                    .as_ref()
                    .map(|episode| episode.episode_id.clone())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let episode_record_ids = slots
            .iter()
            .map(|slot| {
                slot.episode
                    .as_ref()
                    .map(|episode| episode.episode_record_id.clone())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let primary = slots.first().copied();
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

    pub(crate) fn start_episodes_at(
        &mut self,
        positions: &[usize],
        episode_ids: Vec<String>,
        started_from_auto_reset: bool,
        slots: &[u64],
    ) -> Vec<StartedEpisode> {
        let indices: Vec<Option<i64>> = positions
            .iter()
            .enumerate()
            .map(|(i, _)| slots.get(i).map(|slot| *slot as i64 + 1))
            .collect();
        let (record_ids, started) = self.records.ensure_for_slots(
            positions,
            &episode_ids,
            started_from_auto_reset,
            &indices,
        );
        self.sync_slots(
            positions,
            episode_ids,
            record_ids,
            true,
            started_from_auto_reset,
        );
        started
            .into_iter()
            .map(|(episode_id, record)| StartedEpisode { episode_id, record })
            .collect()
    }

    pub(crate) fn observe_episode_ids_at(
        &mut self,
        positions: &[usize],
        episode_ids: Vec<String>,
        slots: &[Option<u64>],
    ) -> Vec<StartedEpisode> {
        let indices: Vec<Option<i64>> = positions
            .iter()
            .enumerate()
            .map(|(i, _)| slots.get(i).copied().flatten().map(|slot| slot as i64 + 1))
            .collect();
        let (record_ids, started) =
            self.records
                .ensure_for_slots(positions, &episode_ids, true, &indices);
        self.sync_slots(positions, episode_ids, record_ids, false, true);
        started
            .into_iter()
            .map(|(episode_id, record)| StartedEpisode { episode_id, record })
            .collect()
    }

    pub(crate) fn record_step_at(
        &mut self,
        positions: &[usize],
        rewards: &[f64],
        step: std::time::Duration,
        predict: std::time::Duration,
    ) {
        self.total_steps += 1;
        for (i, &position) in positions.iter().enumerate() {
            if let Some(slot) = self.slots.get_mut(position) {
                slot.step += 1;
                slot.reset = false;
                slot.cumulative_reward += rewards.get(i).copied().unwrap_or(0.0);
                slot.step_ns += step.as_nanos() as u64;
                slot.predict_ns += predict.as_nanos() as u64;
            }
        }
    }

    pub(crate) fn complete_episode(&mut self, episode_id: &str) -> Option<EpisodeRecord> {
        self.total_episodes += 1;
        self.records.record_for(episode_id).cloned()
    }

    pub(crate) fn seed_for_episode(&self, episode_id: &str) -> Option<i64> {
        self.seed_by_episode.get(episode_id).copied()
    }

    pub(crate) fn end_episode_at(&mut self, env_index: u32) -> Option<String> {
        let position = self.slot_position(env_index)?;
        let episode = self.slots.get_mut(position)?.episode.as_mut()?;
        if episode.ended {
            return None;
        }
        episode.ended = true;
        Some(episode.episode_id.clone())
    }

    pub(crate) fn end_live_episodes(&mut self) -> Vec<String> {
        self.slots
            .iter_mut()
            .filter_map(|slot| slot.episode.as_mut())
            .filter(|episode| episode.predicted && !episode.ended)
            .map(|episode| {
                episode.ended = true;
                episode.episode_id.clone()
            })
            .collect()
    }

    pub(crate) fn mark_predicted(&mut self, positions: &[usize]) {
        for &position in positions {
            if let Some(episode) = self
                .slots
                .get_mut(position)
                .and_then(|slot| slot.episode.as_mut())
            {
                episode.predicted = true;
            }
        }
    }

    pub(crate) fn predict_request_at(
        &mut self,
        positions: &[usize],
        observation: Option<Vec<Bytes>>,
        phase: RequestPhase,
        contract: &Arc<rlmesh_spaces::EnvContract>,
    ) -> ModelObservation {
        let env_id = self.env_id().to_string();
        let episodes: Vec<EpisodeInfo> = self
            .episode_ids_at(positions)
            .into_iter()
            .map(|episode_id| EpisodeInfo {
                seed: self.seed_for_episode(&episode_id),
                episode_id,
                env_id: env_id.clone(),
            })
            .collect();
        ModelObservation {
            observation,
            num_envs: episodes.len().max(1),
            route: ModelRouteContext {
                session_id: self.session_id().to_string(),
                env_id,
                request_id: self.next_request_id(phase.as_str()),
                episodes,
            },
            env_contract: Some(Arc::clone(contract)),
            history: Vec::new(),
            step: None,
        }
    }

    fn sync_slots(
        &mut self,
        positions: &[usize],
        episode_ids: Vec<String>,
        record_ids: Vec<String>,
        reset_steps: bool,
        started_from_auto_reset: bool,
    ) {
        for (index, &position) in positions.iter().enumerate() {
            let Some(slot) = self.slots.get_mut(position) else {
                continue;
            };
            let episode_id = episode_ids.get(index).cloned().unwrap_or_default();
            let episode_record_id = record_ids.get(index).cloned().unwrap_or_default();
            let previous_id = slot
                .episode
                .as_ref()
                .map(|episode| episode.episode_id.clone())
                .unwrap_or_default();
            let rolled = !episode_id.is_empty() && episode_id != previous_id;
            // The outgoing episode's seed is dropped only when its lane rolls (never at completion
            // emit), so the completion iteration's final predict still reports it.
            if rolled && !previous_id.is_empty() {
                self.seed_by_episode.remove(&previous_id);
                self.trial_by_episode.remove(&previous_id);
                self.surplus.remove(&previous_id);
            }
            let (predicted, ended) = match slot.episode.as_ref() {
                Some(previous) if previous.episode_id == episode_id => {
                    (previous.predicted, previous.ended)
                }
                _ => (false, false),
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
                    predicted,
                    ended,
                })
            };
            if reset_steps || rolled {
                slot.step = 0;
                slot.reset = true;
                slot.cumulative_reward = 0.0;
                slot.started_at_ns = now_unix_ns();
                slot.predict_ns = 0;
                slot.step_ns = 0;
            }
        }
    }
}

pub(crate) fn now_unix_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_nanos()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}
