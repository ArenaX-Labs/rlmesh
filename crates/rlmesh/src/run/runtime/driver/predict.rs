//! The model side of the loop: planning which waiting groups predict next,
//! the predict in flight, and applying the frames it returns.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::stream::FuturesUnordered;

use super::model_calls::predict_grouped;
use super::telemetry::{SRC_PREDICT, leaf_bytes, observation_bytes, record_op};
use super::{
    EnvPhase, Group, PeerReport, PredictFuture, PredictOutcome, PredictState, RuntimeDriver,
    RuntimeEnv, RuntimeError,
};
use crate::model::handler::ModelHandler;
use crate::run::runtime::state::RouteState;
use crate::run::runtime::telemetry::Aggregator;

impl<E, M> RuntimeDriver<E, M>
where
    E: RuntimeEnv + Clone + 'static,
    M: ModelHandler,
{
    pub(super) fn dispatch_predict<'m>(
        &mut self,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        predicts: &mut FuturesUnordered<PredictFuture<'m>>,
        force: bool,
    ) where
        M: 'm,
    {
        if !predicts.is_empty() {
            return;
        }
        let waiting: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter(|(_, group)| matches!(group.predict, PredictState::Wanted(_)))
            .map(|(gid, _)| gid)
            .collect();
        if waiting.is_empty() {
            return;
        }
        let busy = groups.iter().filter(|group| group.busy()).count();
        let mut chosen = self.scheduler.plan(&waiting, busy);
        chosen.retain(|gid| waiting.contains(gid));
        if chosen.is_empty() {
            if !force {
                return;
            }
            chosen = waiting;
        }
        predicts.push(self.predict_for(groups, state, chosen));
    }

    fn predict_for<'m>(
        &mut self,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        chosen: Vec<usize>,
    ) -> PredictFuture<'m>
    where
        M: 'm,
    {
        let mut requests = Vec::with_capacity(chosen.len());
        let mut metas = Vec::with_capacity(chosen.len());
        let mut step = 0;
        for gid in chosen {
            let PredictState::Wanted(msg) = std::mem::replace(
                &mut groups[gid].predict,
                PredictState::InFlight { stale: false },
            ) else {
                unreachable!("only waiting groups are planned")
            };
            step = step.max(state.snapshot_at(&groups[gid].positions).step);
            state.mark_predicted(&groups[gid].positions);
            metas.push((gid, observation_bytes(&msg)));
            requests.push(msg);
        }
        let timeout = self.spec.limits.model_predict_timeout;
        let timeout_error = RuntimeError::operation_timeout(
            state.env_id(),
            state.model_component_id(),
            "model.predict",
            step,
            timeout,
        );
        let model = Arc::clone(&self.model);
        let contract = Arc::clone(&self.contract);
        Box::pin(async move {
            let started = Instant::now();
            let predicted = predict_grouped(&model, &contract, requests);
            let result = match tokio::time::timeout(timeout, predicted).await {
                Ok(results) => Ok(results),
                Err(_) => Err(timeout_error),
            };
            PredictOutcome {
                requests: metas,
                rpc: started.elapsed(),
                result,
            }
        })
    }

    pub(super) fn on_predict_outcome(
        &mut self,
        outcome: PredictOutcome,
        groups: &mut [Group<E>],
        state: &RouteState,
        telemetry: &Arc<Mutex<Aggregator>>,
    ) -> Result<(), RuntimeError> {
        let results = outcome.result?;
        if results.len() != outcome.requests.len() {
            return Err(RuntimeError::Protocol(format!(
                "model endpoint {} answered {} of {} grouped predicts",
                state.model_component_id(),
                results.len(),
                outcome.requests.len()
            )));
        }
        let group_count = outcome.requests.len() as u64;
        let rpc = outcome.rpc;
        let mut recorded = false;
        for ((gid, request_bytes), result) in outcome.requests.into_iter().zip(results) {
            let prediction = result?;
            if !recorded {
                recorded = true;
                record_op(
                    telemetry,
                    SRC_PREDICT,
                    outcome.rpc,
                    PeerReport {
                        endpoint_total_ns: prediction.endpoint_total_ns,
                        phases: prediction.phases,
                        group_size: prediction
                            .group_size
                            .or((group_count > 1).then_some(group_count)),
                    },
                    request_bytes,
                    prediction
                        .frames
                        .iter()
                        .map(|frame| leaf_bytes(frame))
                        .sum(),
                );
            }
            if prediction.frames.is_empty() {
                return Err(RuntimeError::Protocol(format!(
                    "model endpoint {} returned a predict response with no actions",
                    state.model_component_id()
                )));
            }
            let frames = VecDeque::from(prediction.frames);
            let group = &mut groups[gid];
            group.pending_predict += rpc;
            group.predict = match std::mem::replace(&mut group.predict, PredictState::None) {
                PredictState::InFlight { stale: true } => {
                    if group.phase == EnvPhase::Ready && group.replay.is_empty() {
                        self.latest_request(group, telemetry)
                            .map_or(PredictState::None, PredictState::Wanted)
                    } else {
                        PredictState::None
                    }
                }
                PredictState::InFlight { stale: false } => {
                    if group.replay.is_empty() {
                        group.replay = frames;
                        PredictState::None
                    } else {
                        PredictState::Ready(frames)
                    }
                }
                other => other,
            };
            self.pending_evictions.extend(
                self.held_evictions
                    .extract_if(.., |(held, _)| *held == gid)
                    .map(|(_, id)| id),
            );
        }
        Ok(())
    }
}
