//! The env side of the loop: putting resets and steps in flight and applying
//! what they return.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rlmesh_proto::env::v1::EpisodeMetadata;
use tokio::task::JoinSet;

use super::lifecycle::{episode_ids_with_roll, mint_episode_ids};
use super::model_calls::{merge_options, model_reset_options};
use super::observe::transform_failed;
use super::telemetry::{
    SRC_RESET, SRC_ROUND, SRC_STEP, SRC_TRANSFORM_ACTION, leaf_bytes, lock_agg, record_op,
    record_transform,
};
use super::{
    EnvOutcome, EnvPhase, EnvResetRequest, EnvStepRequest, Group, PeerReport, PredictState,
    RuntimeDriver, RuntimeEnv, RuntimeEnvStep, RuntimeError,
};
use crate::model::handler::ModelHandler;
use crate::model::types::EpisodeInfo;
use crate::run::runtime::hooks::{
    ActionReceivedEvent, LogEvent, LogLevel, RunDirective, StepCompletedEvent,
};
use crate::run::runtime::state::{RequestPhase, RouteState};
use crate::run::runtime::telemetry::{Aggregator, Sample, metrics};

impl<E, M> RuntimeDriver<E, M>
where
    E: RuntimeEnv + Clone + 'static,
    M: ModelHandler,
{
    pub(super) fn begin_reset(
        &mut self,
        gid: usize,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        env_ops: &mut JoinSet<EnvOutcome<E>>,
        initial: bool,
    ) {
        let width = groups[gid].width();
        let slots = if groups[gid].lane_group {
            state.claim_slots(width, true).unwrap_or_default()
        } else {
            state.claim_slots_upto(width)
        };
        if slots.is_empty() {
            groups[gid].phase = EnvPhase::Idle;
            groups[gid].predict = PredictState::None;
            return;
        }
        let seeds = self.seeds_for(&groups[gid], &slots);
        let episode_ids = mint_episode_ids(width);
        for surplus in &episode_ids[slots.len()..] {
            state.mark_surplus(surplus);
        }
        state.note_episode_seeds(&episode_ids, &seeds);
        let starts: Vec<EpisodeInfo> = match self.reset_options {
            Some(_) => episode_ids
                .iter()
                .enumerate()
                .map(|(lane, episode_id)| EpisodeInfo {
                    episode_id: episode_id.clone(),
                    seed: seeds.get(lane).copied(),
                    env_id: state.env_id().to_string(),
                })
                .collect(),
            None => Vec::new(),
        };
        let trials = self.planned_trial_indices(state, width);
        state.note_episode_trials(&episode_ids, &trials);
        let group = &mut groups[gid];
        group.pending_start = Some((episode_ids.clone(), slots));
        group.replay.clear();
        group.history.clear();
        group.pending_roll.clear();
        let reset_options = self.reset_options.clone();
        let mut request = EnvResetRequest {
            seeds,
            options: self.trial_options(&trials),
            env_indices: if group.whole {
                Vec::new()
            } else {
                group.lanes.clone()
            },
        };
        let request_bytes = (request.seeds.len() * 8) as u64;
        let timeout = self.spec.limits.env_reset_timeout;
        let timeout_error = RuntimeError::operation_timeout(
            state.env_id(),
            state.env_component_id(),
            "env.reset",
            0,
            timeout,
        );
        let mut env = group
            .env
            .take()
            .expect("group env handle present while ready");
        group.phase = EnvPhase::Resetting;
        env_ops.spawn(async move {
            let started = Instant::now();
            let result = match model_reset_options(reset_options, &starts, &trials).await {
                Ok(options) => {
                    request.options = merge_options(options, request.options.take());
                    match tokio::time::timeout(timeout, env.reset(request)).await {
                        Ok(result) => result,
                        Err(_) => Err(timeout_error),
                    }
                }
                Err(err) => Err(err),
            };
            EnvOutcome::Reset {
                group: gid,
                env,
                initial,
                request_bytes,
                rpc: started.elapsed(),
                result,
            }
        });
    }

    #[allow(clippy::needless_range_loop)]
    pub(super) async fn dispatch_steps(
        &mut self,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        env_ops: &mut JoinSet<EnvOutcome<E>>,
        telemetry: &Arc<Mutex<Aggregator>>,
    ) -> Result<(), RuntimeError> {
        for gid in 0..groups.len() {
            if groups[gid].phase != EnvPhase::Ready {
                continue;
            }
            if groups[gid].replay.is_empty() {
                match std::mem::replace(&mut groups[gid].predict, PredictState::None) {
                    PredictState::Ready(frames) => groups[gid].replay = frames,
                    other => {
                        groups[gid].predict = other;
                        continue;
                    }
                }
            }
            let Some(model_action) = groups[gid].replay.pop_front() else {
                continue;
            };
            if self.prefetch_lead > 0
                && groups[gid].replay.len() <= self.prefetch_lead as usize
                && matches!(groups[gid].predict, PredictState::None)
                && let Some(msg) = self.latest_request(&mut groups[gid], telemetry)
            {
                groups[gid].predict = PredictState::Wanted(msg);
            }

            let group = &mut groups[gid];
            let snapshot = state.snapshot_at(&group.positions);
            let context = state.group_context(&group.positions, group.lane_group);
            let action_step = snapshot.step + 1;
            let mut action_event = ActionReceivedEvent {
                session_id: state.session_id().to_string(),
                route: context,
                episode_id: snapshot.episode_id.clone(),
                episode_record_id: snapshot.episode_record_id.clone(),
                episode_ids: snapshot.episode_ids.clone(),
                episode_record_ids: snapshot.episode_record_ids.clone(),
                step: action_step,
                env_index: snapshot.env_index,
                action_space: Arc::clone(&self.action_space),
                action: Some(model_action),
                raw_action: None,
            };
            let started = Instant::now();
            let transformed = self
                .hooks
                .transform_action(&action_event)
                .await
                .map_err(|err| transform_failed("transform_action", err))?;
            if let Some(action) = transformed {
                record_transform(telemetry, SRC_TRANSFORM_ACTION, started);
                action_event.raw_action = action_event.action.replace(action);
            }
            let action = action_event.action.clone().unwrap_or_default();
            fan_out_event!(self, action_received, action_event);

            let group = &mut groups[gid];
            let request_bytes = leaf_bytes(&action);
            let request = EnvStepRequest {
                action,
                env_indices: if group.whole {
                    Vec::new()
                } else {
                    group.lanes.clone()
                },
            };
            let timeout = self.spec.limits.env_step_timeout;
            let timeout_error = RuntimeError::operation_timeout(
                state.env_id(),
                state.env_component_id(),
                "env.step",
                action_step,
                timeout,
            );
            let mut env = group
                .env
                .take()
                .expect("group env handle present while ready");
            group.phase = EnvPhase::Stepping;
            env_ops.spawn(async move {
                let started = Instant::now();
                let result = match tokio::time::timeout(timeout, env.step(request)).await {
                    Ok(result) => result.map_err(|err| err.at_step(action_step)),
                    Err(_) => Err(timeout_error),
                };
                EnvOutcome::Step {
                    group: gid,
                    env,
                    request_bytes,
                    rpc: started.elapsed(),
                    result,
                }
            });
        }
        Ok(())
    }

    pub(super) async fn on_env_outcome(
        &mut self,
        outcome: EnvOutcome<E>,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        env_ops: &mut JoinSet<EnvOutcome<E>>,
        telemetry: &Arc<Mutex<Aggregator>>,
    ) -> Result<(), RuntimeError> {
        match outcome {
            EnvOutcome::Reset {
                group: gid,
                env,
                initial,
                request_bytes,
                rpc,
                result,
            } => {
                groups[gid].env = Some(env);
                let reset = result?;
                record_op(
                    telemetry,
                    SRC_RESET,
                    rpc,
                    PeerReport {
                        endpoint_total_ns: reset.endpoint_total_ns,
                        phases: reset.phases,
                        group_size: None,
                    },
                    request_bytes,
                    leaf_bytes(&reset.observation),
                );
                let group = &mut groups[gid];
                let (episode_ids, slots) = group
                    .pending_start
                    .take()
                    .expect("a reset in flight has its episodes staged");
                let context = state.group_context(&group.positions, group.lane_group);
                if initial {
                    fan_out_event!(
                        self,
                        log,
                        LogEvent {
                            session_id: state.session_id().to_string(),
                            route: context.clone(),
                            level: LogLevel::Info,
                            message: format!(
                                "env reset complete in {:.0}ms ({} episode(s) ready)",
                                rpc.as_secs_f64() * 1000.0,
                                episode_ids.len()
                            ),
                            source: Some("runtime".to_string()),
                        }
                    );
                }
                let started =
                    state.start_episodes_at(&groups[gid].positions, episode_ids, !initial, &slots);
                let started = self.invoke_started_episodes(state, &context, started).await;
                groups[gid].instruction_starts.extend(started);
                groups[gid].round_started = Instant::now();
                self.observe(
                    gid,
                    groups,
                    state,
                    telemetry,
                    Some(reset.observation),
                    reset.infos,
                    RequestPhase::ResetObservation,
                    true,
                )
                .await
            }
            EnvOutcome::Step {
                group: gid,
                env,
                request_bytes,
                rpc,
                result,
            } => {
                groups[gid].env = Some(env);
                let step = result?;
                record_op(
                    telemetry,
                    SRC_STEP,
                    rpc,
                    PeerReport {
                        endpoint_total_ns: step.endpoint_total_ns,
                        phases: step.phases,
                        group_size: None,
                    },
                    request_bytes,
                    leaf_bytes(&step.observation),
                );
                self.on_step(gid, groups, state, env_ops, telemetry, step, rpc)
                    .await
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn on_step(
        &mut self,
        gid: usize,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        env_ops: &mut JoinSet<EnvOutcome<E>>,
        telemetry: &Arc<Mutex<Aggregator>>,
        response: RuntimeEnvStep,
        rpc: Duration,
    ) -> Result<(), RuntimeError> {
        let positions = groups[gid].positions.clone();
        let lane_group = groups[gid].lane_group;
        let context = state.group_context(&positions, lane_group);
        lock_agg(telemetry).record(Sample::dur(
            SRC_ROUND,
            metrics::RPC_TOTAL,
            groups[gid].round_started.elapsed(),
        ));
        groups[gid].round_started = Instant::now();
        groups[gid].phase = EnvPhase::Ready;
        groups[gid].steps += 1;

        let predict = std::mem::take(&mut groups[gid].pending_predict);
        state.record_step_at(&positions, &response.rewards, rpc, predict);
        let snapshot = state.snapshot_at(&positions);
        let mut completed_episodes = self.env_completions(state, &groups[gid], &response);

        let directive = self.hooks.directive();
        let capped = self.capped_completions(
            state,
            &positions,
            &completed_episodes,
            response.infos.as_ref(),
            directive != RunDirective::Continue && self.driver_owns_resets(),
        );
        completed_episodes.extend(capped);
        let lane_ids = state.episode_ids_at(&positions);
        let lane_flags = |ended: fn(&EpisodeMetadata) -> bool| -> Vec<bool> {
            lane_ids
                .iter()
                .map(|id| {
                    completed_episodes
                        .iter()
                        .any(|m| m.episode_id == *id && ended(m))
                })
                .collect()
        };
        let terminated = lane_flags(|m| m.terminated);
        let truncated = lane_flags(|m| m.truncated);
        let autoreset_roll: Vec<bool> = groups[gid]
            .lanes
            .iter()
            .map(|lane| groups[gid].pending_roll.contains(lane))
            .collect();
        let rolled = !groups[gid].pending_roll.is_empty();
        fan_out_event!(
            self,
            step_completed,
            StepCompletedEvent {
                session_id: state.session_id().to_string(),
                route: context.clone(),
                episode_id: snapshot.episode_id.clone(),
                episode_record_id: snapshot.episode_record_id.clone(),
                step: snapshot.step,
                env_index: snapshot.env_index,
                rewards: response.rewards.clone(),
                infos: if rolled { None } else { response.infos.clone() },
                terminated,
                truncated,
                autoreset_roll,
                predict_ms: predict.as_secs_f64() * 1e3,
                step_ms: rpc.as_secs_f64() * 1e3,
                observation_space: Arc::clone(&self.observation_space),
                observation: (!response.observation.is_empty())
                    .then(|| response.observation.clone()),
            }
        );

        if rolled {
            let completions = std::mem::take(&mut groups[gid].pending_completed);
            self.fan_out_completions(completions).await;
            let pending_roll = std::mem::take(&mut groups[gid].pending_roll);
            self.queue_evictions(gid, &groups[gid], state, pending_roll.iter().copied());
            let roll_ids = episode_ids_with_roll(
                state.episode_ids_at(&positions),
                &groups[gid].lanes,
                &pending_roll,
            );
            let rolling: Vec<Option<u64>> = groups[gid]
                .lanes
                .iter()
                .map(|lane| {
                    pending_roll
                        .contains(lane)
                        .then(|| state.claim_slots(1, true))
                        .flatten()
                        .and_then(|slots| slots.first().copied())
                })
                .collect();
            for ((lane, id), slot) in groups[gid].lanes.iter().zip(&roll_ids).zip(&rolling) {
                if pending_roll.contains(lane) && slot.is_none() {
                    state.mark_surplus(id);
                }
            }
            let started = state.observe_episode_ids_at(&positions, roll_ids, &rolling);
            let started = self.invoke_started_episodes(state, &context, started).await;
            groups[gid].instruction_starts.extend(started);
        }

        let completions = self.complete_episodes(state, &context, &completed_episodes);
        if self.driver_owns_resets() {
            self.fan_out_completions(completions).await;
            self.queue_evictions(
                gid,
                &groups[gid],
                state,
                completed_episodes.iter().map(|c| c.env_index),
            );
        } else {
            groups[gid].pending_completed.extend(completions);
        }

        if !completed_episodes.is_empty() {
            let group = &mut groups[gid];
            if !group.replay.is_empty() && group.width() > 1 && !self.vector_replay_warned {
                self.vector_replay_warned = true;
                tracing::warn!(
                    num_envs = group.width(),
                    discarded_frames = group.replay.len(),
                    "a lane's episode ended mid-chunk on a lockstep vector group: chunk replay \
                     is whole-batch, so every lane's buffered frames were discarded and the \
                     group re-plans. Serve lanes, or use an execution horizon of 1.",
                );
            }
            group.replay.clear();
            group.predict = match std::mem::replace(&mut group.predict, PredictState::None) {
                PredictState::InFlight { .. } => PredictState::InFlight { stale: true },
                _ => PredictState::None,
            };
            if !self.driver_owns_resets() {
                group
                    .pending_roll
                    .extend(completed_episodes.iter().map(|c| c.env_index));
            }
        }

        if !self.driver_owns_resets() {
            let bound = self.spec.edition_defaults().default_max_episode_steps;
            let endless = state.slots_at(&positions).into_iter().find(|slot| {
                slot.step >= bound
                    && slot.episode.as_ref().is_some_and(|episode| {
                        !state.is_surplus(&episode.episode_id)
                            && !completed_episodes
                                .iter()
                                .any(|ended| ended.episode_id == episode.episode_id)
                    })
            });
            if let Some(slot) = endless {
                return Err(RuntimeError::EnvRpc {
                    operation: "step",
                    step: slot.step,
                    message: format!(
                        "lane {}'s episode ran {bound} steps without the env ending it; under \
                         NEXT_STEP autoreset the env owns its episodes, so the runtime cannot \
                         truncate one: give the env a step limit (e.g. Gymnasium's TimeLimit)",
                        slot.env_index
                    ),
                    recoverable: false,
                    source: None,
                });
            }
        }

        let budget_spent = self
            .spec
            .max_episodes
            .is_some_and(|limit| state.total_episodes() >= limit as i64);
        if directive == RunDirective::Stop
            || ((!lane_group || !self.driver_owns_resets()) && budget_spent)
            || state.all_surplus_at(&positions)
        {
            let completions = std::mem::take(&mut groups[gid].pending_completed);
            self.fan_out_completions(completions).await;
            let pending_roll = std::mem::take(&mut groups[gid].pending_roll);
            self.queue_evictions(gid, &groups[gid], state, pending_roll.iter().copied());
            groups[gid].phase = EnvPhase::Idle;
            groups[gid].predict = PredictState::None;
            return Ok(());
        }

        if self.driver_owns_resets() {
            let done: HashSet<u32> = completed_episodes
                .iter()
                .map(|metadata| metadata.env_index)
                .collect();
            if !done.is_empty() {
                self.begin_reset(gid, groups, state, env_ops, false);
                return Ok(());
            }
        }

        self.observe(
            gid,
            groups,
            state,
            telemetry,
            Some(response.observation),
            if rolled { response.infos } else { None },
            RequestPhase::StepObservation,
            false,
        )
        .await
    }
}
