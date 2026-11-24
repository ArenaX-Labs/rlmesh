//! Episode lifecycle: fanning out starts, building completions (env-ended and
//! capped), minting episode ids, and evicting ended episodes' model state.

use std::collections::HashSet;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rlmesh_proto::env::v1::EpisodeMetadata;
use tokio_util::sync::CancellationToken;

use super::model_calls::evict;
use super::telemetry::{SRC_RESET_ADAPTER, lock_agg};
use super::{Group, PredictState, RuntimeDriver, RuntimeEnv, RuntimeEnvStep};
use crate::model::handler::ModelHandler;
use crate::run::runtime::hooks::{EpisodeCompletedEvent, EpisodeStartedEvent, RuntimeEnvContext};
use crate::run::runtime::state::{RouteState, StartedEpisode};
use crate::run::runtime::telemetry::{Aggregator, Sample, metrics};

fn success_from_final_info(
    keys: &[&str],
    final_info: Option<&rlmesh_proto::spaces::v1::MetaMap>,
) -> Option<bool> {
    use rlmesh_proto::spaces::v1::meta_value::Kind;
    let entries = &final_info?.entries;
    keys.iter()
        .find_map(|key| match entries.get(*key)?.kind.as_ref()? {
            Kind::Bool(value) => Some(*value),
            Kind::Integer(value) => Some(*value != 0),
            Kind::Number(value) => Some(*value != 0.0),
            _ => None,
        })
}
impl<E, M> RuntimeDriver<E, M>
where
    E: RuntimeEnv + Clone + 'static,
    M: ModelHandler,
{
    pub(super) async fn invoke_started_episodes(
        &self,
        state: &RouteState,
        context: &RuntimeEnvContext,
        episodes: Vec<StartedEpisode>,
    ) -> Vec<EpisodeStartedEvent> {
        let mut instruction_starts = Vec::new();
        for episode in episodes {
            if state.is_surplus(&episode.episode_id) {
                continue;
            }
            let record = &episode.record;
            let event = EpisodeStartedEvent {
                session_id: state.session_id().to_string(),
                route: context.clone(),
                episode_id: episode.episode_id.clone(),
                episode_record_id: record.record_id.clone(),
                episode_index: record.index,
                env_index: record.env_index,
                started_from_auto_reset: record.started_from_auto_reset,
                seed: state.seed_for_episode(&episode.episode_id),
                trial_index: state.trial_for_episode(&episode.episode_id),
            };
            if self.instruction_leaf.is_some() {
                instruction_starts.push(event.clone());
            }
            fan_out_event!(self, episode_started, event);
        }
        instruction_starts
    }

    pub(super) fn env_completions(
        &self,
        state: &RouteState,
        group: &Group<E>,
        response: &RuntimeEnvStep,
    ) -> Vec<EpisodeMetadata> {
        let now_ns = crate::run::runtime::state::now_unix_ns();
        let width = group.lanes.len();
        state
            .slots_at(&group.positions)
            .into_iter()
            .zip(&group.lanes)
            .enumerate()
            .filter_map(|(lane, (slot, env_index))| {
                let terminated = response.terminated.get(lane).copied().unwrap_or(false);
                let truncated = response.truncated.get(lane).copied().unwrap_or(false);
                if !(terminated || truncated) || group.pending_roll.contains(env_index) {
                    return None;
                }
                let episode = slot.episode.as_ref()?;
                Some(EpisodeMetadata {
                    episode_id: episode.episode_id.clone(),
                    seed: None,
                    env_index: *env_index,
                    step_count: slot.step,
                    cumulative_reward: slot.cumulative_reward,
                    terminated,
                    truncated,
                    start_timestamp_ns: slot.started_at_ns,
                    end_timestamp_ns: now_ns,
                    final_info: rlmesh_proto::lane_final_info(response.infos.as_ref(), lane, width),
                })
            })
            .collect()
    }

    pub(super) fn capped_completions(
        &self,
        state: &RouteState,
        positions: &[usize],
        env_completed: &[EpisodeMetadata],
        infos: Option<&rlmesh_proto::spaces::v1::MetaMap>,
        ended: bool,
    ) -> Vec<EpisodeMetadata> {
        let step_cap = self.spec.max_episode_steps.or_else(|| {
            self.driver_owns_resets()
                .then_some(self.spec.edition_defaults().default_max_episode_steps)
        });
        let time_cap = self.spec.max_episode_seconds;
        if step_cap.is_none() && time_cap.is_none() && !ended {
            return Vec::new();
        }
        let env_done: Vec<u32> = env_completed
            .iter()
            .map(|metadata| metadata.env_index)
            .collect();
        let now_ns = crate::run::runtime::state::now_unix_ns();
        state
            .slots_at(positions)
            .into_iter()
            .enumerate()
            .filter_map(|(lane, slot)| {
                let episode = slot.episode.as_ref()?;
                let env_index = u32::try_from(slot.env_index).ok()?;
                if env_done.contains(&env_index) {
                    return None;
                }
                let steps_capped = step_cap.is_some_and(|cap| slot.step >= cap);
                let elapsed_seconds = (now_ns - slot.started_at_ns).max(0) as f64 / 1e9;
                let time_capped = time_cap.is_some_and(|cap| elapsed_seconds >= cap);
                (ended || steps_capped || time_capped).then(|| EpisodeMetadata {
                    episode_id: episode.episode_id.clone(),
                    seed: None,
                    env_index,
                    step_count: slot.step,
                    cumulative_reward: slot.cumulative_reward,
                    terminated: false,
                    truncated: true,
                    start_timestamp_ns: slot.started_at_ns,
                    end_timestamp_ns: now_ns,
                    final_info: rlmesh_proto::lane_final_info(infos, lane, positions.len()),
                })
            })
            .collect()
    }

    pub(super) fn complete_episodes(
        &self,
        state: &mut RouteState,
        context: &RuntimeEnvContext,
        episodes: &[EpisodeMetadata],
    ) -> Vec<EpisodeCompletedEvent> {
        let mut events = Vec::with_capacity(episodes.len());
        for completed in episodes {
            if state.is_surplus(&completed.episode_id) {
                continue;
            }
            let (predict_ms, step_ms) = state.slot_timings_ms(completed.env_index);
            let record = state.complete_episode(&completed.episode_id);
            let episode_record_id = record
                .as_ref()
                .map(|record| record.record_id.clone())
                .unwrap_or_default();
            let env_index = i32::try_from(completed.env_index).unwrap_or(i32::MAX);
            let seed = state.seed_for_episode(&completed.episode_id);
            let trial_index = state.trial_for_episode(&completed.episode_id);
            if self.spec.max_episodes.is_some() {
                state.record_episode_summary(crate::run::runtime::spec::EpisodeSummary {
                    episode_index: record.as_ref().map_or(0, |record| record.index),
                    env_index,
                    seed,
                    trial_index,
                    step_count: completed.step_count,
                    cumulative_reward: completed.cumulative_reward,
                    terminated: completed.terminated,
                    truncated: completed.truncated,
                    duration_ms: (completed.end_timestamp_ns - completed.start_timestamp_ns).max(0)
                        / 1_000_000,
                    success: success_from_final_info(
                        self.spec.edition_defaults().success_info_keys,
                        completed.final_info.as_ref(),
                    ),
                    predict_ms,
                    step_ms,
                });
            }
            events.push(EpisodeCompletedEvent {
                session_id: state.session_id().to_string(),
                route: context.clone(),
                episode_id: completed.episode_id.clone(),
                episode_record_id,
                episode_index: record.as_ref().map_or(0, |record| record.index),
                env_index,
                step_count: completed.step_count,
                cumulative_reward: completed.cumulative_reward,
                terminated: completed.terminated,
                truncated: completed.truncated,
                duration_ms: (completed.end_timestamp_ns - completed.start_timestamp_ns).max(0)
                    / 1_000_000,
                success: success_from_final_info(
                    self.spec.edition_defaults().success_info_keys,
                    completed.final_info.as_ref(),
                ),
                final_info: completed.final_info.clone(),
                seed,
                trial_index,
                predict_ms,
                step_ms,
            });
        }
        events
    }

    pub(super) async fn fan_out_completions(&self, events: Vec<EpisodeCompletedEvent>) {
        for event in events {
            fan_out_event!(self, episode_completed, event);
        }
    }

    pub(super) fn queue_evictions(
        &mut self,
        gid: usize,
        group: &Group<E>,
        state: &mut RouteState,
        env_indices: impl Iterator<Item = u32>,
    ) {
        let ids = env_indices.filter_map(|env_index| state.end_episode_at(env_index));
        if matches!(group.predict, PredictState::InFlight { .. }) {
            self.held_evictions.extend(ids.map(|id| (gid, id)));
        } else {
            self.pending_evictions.extend(ids);
        }
    }

    pub(super) async fn flush_evictions(
        &mut self,
        state: &mut RouteState,
        telemetry: &Mutex<Aggregator>,
        cancellation: Option<&CancellationToken>,
        deadline: Duration,
    ) {
        if self.pending_evictions.is_empty() {
            return;
        }
        let episode_ids = std::mem::take(&mut self.pending_evictions);
        let episodes = episode_ids.len();
        let evicted = evict(&self.model, state.env_id().to_string(), episode_ids);
        let cancelled = async {
            match cancellation {
                Some(cancellation) => cancellation.cancelled().await,
                None => std::future::pending().await,
            }
        };
        let started = Instant::now();
        tokio::select! {
            biased;
            () = cancelled => {
                tracing::warn!(episodes, "model reset_adapter (evict) abandoned: route cancelled");
            }
            result = tokio::time::timeout(deadline, evicted) => match result {
                Ok(result) => {
                    lock_agg(telemetry).record(Sample::dur(
                        SRC_RESET_ADAPTER,
                        metrics::RPC_TOTAL,
                        started.elapsed(),
                    ));
                    if let Err(err) = result {
                        tracing::warn!("model reset_adapter (evict) failed: {err}");
                    }
                }
                Err(_) => tracing::warn!(
                    episodes,
                    timeout_ms = deadline.as_millis(),
                    "model reset_adapter (evict) timed out; abandoning"
                ),
            }
        }
    }
}

fn mint_episode_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

pub(super) fn mint_episode_ids(count: usize) -> Vec<String> {
    (0..count).map(|_| mint_episode_id()).collect()
}

pub(super) fn episode_ids_with_roll(
    mut ids: Vec<String>,
    lanes: &[u32],
    pending_roll: &HashSet<u32>,
) -> Vec<String> {
    for (id, lane) in ids.iter_mut().zip(lanes) {
        if pending_roll.contains(lane) {
            *id = mint_episode_id();
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::success_from_final_info;
    use rlmesh_proto::spaces::v1::{MetaMap, MetaValue, meta_value::Kind};

    const KEYS: &[&str] = &["is_success", "success", "task_success"];

    fn info(entries: &[(&str, Option<Kind>)]) -> MetaMap {
        MetaMap {
            entries: entries
                .iter()
                .map(|(key, kind)| (key.to_string(), MetaValue { kind: kind.clone() }))
                .collect(),
        }
    }

    #[test]
    fn success_skips_keys_whose_kind_carries_no_outcome() {
        let skipped = info(&[
            ("is_success", Some(Kind::Text("False".into()))),
            ("success", None),
            ("task_success", Some(Kind::Integer(1))),
        ]);
        assert_eq!(success_from_final_info(KEYS, Some(&skipped)), Some(true));

        let undecided = info(&[("is_success", Some(Kind::Text("True".into())))]);
        assert_eq!(success_from_final_info(KEYS, Some(&undecided)), None);
        assert_eq!(success_from_final_info(KEYS, None), None);
    }

    #[test]
    fn success_takes_the_first_deciding_key_by_priority() {
        let both = info(&[
            ("is_success", Some(Kind::Number(0.0))),
            ("success", Some(Kind::Bool(true))),
        ]);
        assert_eq!(success_from_final_info(KEYS, Some(&both)), Some(false));
    }
}
