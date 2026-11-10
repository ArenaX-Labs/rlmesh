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
    /// Put the groups waiting for a prediction in flight, per the scheduler:
    /// a model that fuses predicts takes the plan as one grouped predict and
    /// the next forms behind it; any other takes one predict per group, in
    /// flight together, each landing on its own.
    pub(super) fn dispatch_predict<'m>(
        &mut self,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        predicts: &mut FuturesUnordered<PredictFuture<'m>>,
        force: bool,
    ) where
        M: 'm,
    {
        // A model fuses its waiting groups into one `predict_grouped`: one
        // grouped predict is in flight at a time, and the groups that become
        // ready behind it form the next.
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

    /// One predict call carrying the `chosen` groups' requests, sharing the
    /// model handle for its duration.
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

    /// Apply a finished predict: every group in it gets its frames, or has
    /// them discarded if its episode ended in the meantime, and the evictions
    /// held behind it go out.
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
            // One grouped call is one predict op: record its RPC once, with the
            // peer's report from the first reply and the fused width.
            if !recorded {
                recorded = true;
                record_op(
                    telemetry,
                    SRC_PREDICT,
                    outcome.rpc,
                    PeerReport {
                        endpoint_total_ns: prediction.endpoint_total_ns,
                        phases: prediction.phases,
                        group_size: (group_count > 1).then_some(group_count),
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
            // Ordered frames: frame 0 = this step, frames 1.. = open-loop replay.
            let frames = VecDeque::from(prediction.frames);
            let group = &mut groups[gid];
            group.pending_predict += rpc;
            group.predict = match std::mem::replace(&mut group.predict, PredictState::None) {
                // Conditioned on an observation from before an episode boundary:
                // the chunk must not leak into the new episode. If the next
                // observation already landed while this was in flight, nothing
                // else will re-plan from it: re-arm here or the group stalls.
                //
                // History routes deliver that observation exactly once either
                // way. It landed with this predict in flight, so `observe`
                // buffered it as a row, and `latest_request` promotes it off the
                // backlog to the re-armed request's own. Under driver-owned
                // resets it is the reset observation: `begin_reset` cleared the
                // ended episode's rows before it, so it is the backlog's only
                // row and the request carries none. Under NEXT_STEP it is the
                // terminal observation, predicted on under the ended id before
                // its eviction at t+1; the ended episode's earlier rows are
                // still buffered (completion does not clear them) and ride that
                // request, where they belong. If instead the stale result lands
                // before the observation (a reset still in flight), nothing is
                // re-armed and `observe` re-plans from it as its own.
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
