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

/// The env-reported task outcome from an episode's final-step info, read under
/// the `keys` the session's edition governs
/// ([`EditionDefaults::success_info_keys`](rlmesh_proto::EditionDefaults::success_info_keys)),
/// in priority order: the first key holding a bool, integer, or number decides
/// (numbers by truthiness, `1`/`1.0` → true), and keys of any other kind are
/// skipped; `None` when no key decides. The Python Session's `_episode_success`
/// applies the same rule so the two loops report identical success.
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
    /// Fan out the scored episodes' starts; on an instruction route, hand
    /// them back for `transform_instruction` at their first observation.
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

    /// The completions of the lanes the env ended this step: each built
    /// from the lane's episode as this driver accounted it (steps, reward,
    /// start time) and the lane's share of this step's info.
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

    /// Runtime-truncated completions for the group's lanes at the step/time cap
    /// this step, excluding lanes the env itself just completed. Built from the
    /// driver's own per-slot accounting (steps, accumulated reward, episode
    /// start time) and this step's `infos`, which carry each lane's last
    /// reported outcome; `validate()` guarantees driver-owned resets (autoreset
    /// `DISABLED`) whenever a cap is configured, so the reset path restarts
    /// these lanes exactly like env-reported completions. `ended` caps every
    /// live lane (an observer ended the episode).
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

    /// Complete each episode: registry + summary (bounded runs only), and
    /// build its `episode_completed` hook event for the caller to fan out once
    /// the ended id's last event is out (see `Group::pending_completed`).
    /// Summaries are recorded only when `max_episodes` is set — the report is
    /// drained once at route end, so an unbounded (`max_episodes: None`)
    /// session would otherwise accumulate one entry per episode for its whole
    /// lifetime with no reader.
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
            // Proto env_index is uint32; events are i32/i64.
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

    /// `episode_completed` is the last hook event under its episode id.
    pub(super) async fn fan_out_completions(&self, events: Vec<EpisodeCompletedEvent>) {
        for event in events {
            fan_out_event!(self, episode_completed, event);
        }
    }

    /// Queue the ended episodes' model-side state for eviction. The id to
    /// evict is resolved POSITIONALLY from the runtime's own slot by
    /// `env_index` — the env's `completed_episodes[].episode_id` echo is never
    /// trusted as the authority — so this must run while the slot still holds
    /// the ended id. Under driver-owned resets that is the completion step.
    /// Under NEXT_STEP the lockstep group predicts once more on the terminal
    /// observation (the autoreset step's action, which the env discards for
    /// that lane) and that predict has to land on the ended id — evicting first
    /// would make the model re-seed the ended episode, and re-tagging it with
    /// the new id would leak the old episode's last frame into the new one —
    /// so the eviction waits for the roll at t+1, just before the slot moves
    /// on. Either way the model sees its `on_episode_end` after the last
    /// predict under that id and never a predict after it. Sent by
    /// [`flush_evictions`](Self::flush_evictions) on the next loop turn — or,
    /// while the group still has a predict in flight (a prefetch conditioned
    /// on the ended id), held until that predict lands. Each id is ended on
    /// the slot once, so the teardown sweep does not end it again.
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

    /// Best-effort GC (R2): a failure is logged and the route keeps moving — a
    /// missed evict only leaks model memory, never corrupts state (ids never
    /// repeat). Bounded by `deadline` and by the route's cancellation, so a
    /// model that keeps its stream open but never answers cannot hold the
    /// route (or its teardown) on an eviction; the abandoned ids are dropped,
    /// not retried — the model ends whatever it still holds at release.
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
        // The eviction's wall time is a telemetry row of its own: an
        // answered RPC (ok or failed) records `rpc.total` under
        // `model.reset_adapter` next to `model.predict`, so a model that
        // serializes its evictions behind its forwards shows up in the
        // windows instead of only in the inter-episode gap.
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

/// Mint one authoritative episode id. UUIDv7 is time-ordered (sortable by
/// creation) and never repeats, so a missed ResetAdapter can only leak memory —
/// never alias a fresh episode.
fn mint_episode_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

/// Mint `count` fresh episode ids (one per lane being started).
pub(super) fn mint_episode_ids(count: usize) -> Vec<String> {
    (0..count).map(|_| mint_episode_id()).collect()
}

/// Current per-lane ids with the NEXT_STEP autoreset rolls substituted in:
/// lanes not rolling keep their current id, a rolling lane takes a freshly
/// minted one. `ids` is aligned to `lanes` (the group's own lanes, in order),
/// while `pending_roll` holds route-global `env_index`es, so a roll is placed
/// at the lane's POSITION in the group.
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
        // A string (even "False"), a null, or a list never decides; the next
        // key in priority order does.
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
