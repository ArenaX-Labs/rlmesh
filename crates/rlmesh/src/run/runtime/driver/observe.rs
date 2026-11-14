//! Observations: building each group's predict request, the observation and
//! instruction transforms, and observation-history rows.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use bytes::Bytes;

use super::telemetry::{SRC_TRANSFORM_OBS, record_history_rows, record_transform};
use super::{
    EnvPhase, Group, HISTORY_BACKLOG_CAP, PredictState, RuntimeDriver, RuntimeEnv, RuntimeError,
};
use crate::model::handler::ModelHandler;
use crate::model::types::{HistoryFrame, ModelObservation};
use crate::run::runtime::hooks::{HookError, ObservationEmittedEvent, RuntimeEnvContext};
use crate::run::runtime::state::{RequestPhase, RouteSnapshot, RouteState};
use crate::run::runtime::telemetry::Aggregator;

impl<E, M> RuntimeDriver<E, M>
where
    E: RuntimeEnv + Clone + 'static,
    M: ModelHandler,
{
    /// Hand a group its next observation: build the predict request, run the
    /// observation transform, emit it, and ask for a prediction unless replay
    /// frames still cover the next step.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn observe(
        &mut self,
        gid: usize,
        groups: &mut [Group<E>],
        state: &mut RouteState,
        telemetry: &Arc<Mutex<Aggregator>>,
        observation: Option<Vec<Bytes>>,
        infos: Option<rlmesh_proto::spaces::v1::MetaMap>,
        phase: RequestPhase,
        is_reset: bool,
    ) -> Result<(), RuntimeError> {
        let positions = groups[gid].positions.clone();
        let context = state.group_context(&positions, groups[gid].lane_group);
        let mut event = self.observation_event(
            state,
            context,
            state.snapshot_at(&positions),
            is_reset,
            observation,
            infos,
            groups[gid].width(),
        );
        let started = Instant::now();
        self.transform_instruction(&mut groups[gid], &mut event)
            .await?;
        let transformed = self
            .hooks
            .transform_observation(&event)
            .await
            .map_err(|err| transform_failed("transform_observation", err))?;
        if let Some(observation) = transformed {
            let raw = event.observation.replace(observation);
            event
                .raw_observation
                .get_or_insert_with(|| raw.unwrap_or_default());
        }
        if event.raw_observation.is_some() {
            record_transform(telemetry, SRC_TRANSFORM_OBS, started);
        }
        let mut msg =
            state.predict_request_at(&positions, event.observation.clone(), phase, &self.contract);
        // Hooks see the observation the model receives, for both step and
        // reset observations, and the env's when a transform replaced it.
        fan_out_event!(self, observation_emitted, event);

        let group = &mut groups[gid];
        // The observation re-plans now only if nothing covers the next step: no
        // replay frame left, and no chunk in flight or waiting to be played
        // (with a prefetch lead the replay can be spent while the prefetched
        // chunk is still on its way, so the replay alone does not decide).
        let replans = group.replay.is_empty() && matches!(group.predict, PredictState::None);
        if self.deliver_history {
            // Exactly-once: an observation is either the predict's own (it
            // re-plans now, carrying the backlog as its rows) or a history row
            // (a buffered or arriving frame will act on it). Never both, never
            // neither. A row stays the group's latest observation: should an
            // async prefetch or a stale-result re-arm re-plan from it later,
            // `latest_request` promotes it off the backlog to that request's
            // own, so it still leaves the group once.
            msg.step = Some(group.steps);
            if replans {
                msg.history = std::mem::take(&mut group.history);
                record_history_rows(telemetry, &msg);
            } else {
                if group.history.len() >= HISTORY_BACKLOG_CAP {
                    return Err(RuntimeError::Protocol(format!(
                        "route {} buffered {} observation-history rows without a predict: the \
                         model endpoint returned a chunk longer than any execution horizon \
                         the runtime supports",
                        state.env_id(),
                        group.history.len()
                    )));
                }
                group.history.push(HistoryFrame {
                    observation: msg.observation.clone(),
                    episodes: msg.route.episodes.clone(),
                    step: group.steps,
                });
            }
        }
        group.obs_msg = Some(msg.clone());
        group.phase = EnvPhase::Ready;
        if replans {
            group.predict = PredictState::Wanted(msg);
        }
        Ok(())
    }

    /// The request a re-plan from the group's latest observation sends: the
    /// prefetch arm in `dispatch_steps` and the stale-result re-arm in
    /// `on_predict_outcome` both predict from `obs_msg` rather than from a
    /// fresh observation. On a history route that observation was buffered as
    /// a row when it landed (nothing re-planned then), so it is popped off the
    /// backlog as the request's own and the rows before it ride along: every
    /// step still leaves the group exactly once. If it is not the backlog's
    /// trailing row it already left as an earlier request's own (a lead of at
    /// least the chunk length re-arms before the next observation lands);
    /// re-sending it would deliver that step twice, so wait for the next
    /// observation instead. Routes without history re-send it as before.
    pub(super) fn latest_request(
        &self,
        group: &mut Group<E>,
        telemetry: &Arc<Mutex<Aggregator>>,
    ) -> Option<ModelObservation> {
        let mut msg = group.obs_msg.clone()?;
        if self.deliver_history {
            if group.history.last().map(|row| row.step) != msg.step {
                return None;
            }
            group.history.pop();
            msg.history = std::mem::take(&mut group.history);
            record_history_rows(telemetry, &msg);
        }
        Some(msg)
    }

    #[allow(clippy::too_many_arguments)]
    fn observation_event(
        &self,
        state: &RouteState,
        route: RuntimeEnvContext,
        snapshot: RouteSnapshot,
        is_reset: bool,
        observation: Option<Vec<Bytes>>,
        infos: Option<rlmesh_proto::spaces::v1::MetaMap>,
        width: usize,
    ) -> ObservationEmittedEvent {
        ObservationEmittedEvent {
            session_id: state.session_id().to_string(),
            route,
            episode_id: snapshot.episode_id,
            episode_record_id: snapshot.episode_record_id,
            episode_ids: snapshot.episode_ids,
            episode_record_ids: snapshot.episode_record_ids,
            step: snapshot.step,
            env_index: snapshot.env_index,
            is_reset,
            num_envs: width as u32,
            observation_space: Arc::clone(&self.observation_space),
            observation,
            raw_observation: None,
            infos,
        }
    }

    /// Instruction routes: ask `transform_instruction` for the episodes that
    /// started since the group's last observation, then put each lane's
    /// replacement into its instruction leaf while that episode runs.
    async fn transform_instruction(
        &self,
        group: &mut Group<E>,
        event: &mut ObservationEmittedEvent,
    ) -> Result<(), RuntimeError> {
        let Some((leaf, space)) = &self.instruction_leaf else {
            return Ok(());
        };
        let Some(leaves) = event.observation.as_ref() else {
            return Ok(());
        };
        let width = group.width();
        let decode = |leaves: &[Bytes]| {
            let slab = leaves.get(*leaf..=*leaf).ok_or_else(|| {
                RuntimeError::Protocol(format!("observation has no instruction leaf {leaf}"))
            })?;
            rlmesh_spaces::leaves::decode_leaf_slab(slab, space, width)
                .map_err(|err| RuntimeError::Protocol(format!("instruction leaf: {err}")))
        };
        let mut texts = None;
        for episode in std::mem::take(&mut group.instruction_starts) {
            let Some(lane) = group
                .lanes
                .iter()
                .position(|&lane| i64::from(lane) == i64::from(episode.env_index))
            else {
                continue;
            };
            if texts.is_none() {
                texts = Some(decode(leaves)?);
            }
            let current = match texts.as_ref().and_then(|texts| texts.get(lane)) {
                Some(rlmesh_spaces::SpaceValue::Text(text)) => text.as_str(),
                _ => "",
            };
            group.instructions[lane] = self
                .hooks
                .transform_instruction(&episode, current)
                .await
                .map_err(|err| transform_failed("transform_instruction", err))?
                .map(|text| (episode.episode_id, text));
        }
        let live = |lane: usize| {
            group.instructions[lane]
                .as_ref()
                .filter(|(id, _)| event.episode_ids.get(lane) == Some(id))
                .map(|(_, text)| text.clone())
        };
        if !(0..width).any(|lane| live(lane).is_some()) {
            return Ok(());
        }
        let mut texts = match texts {
            Some(texts) => texts,
            None => decode(leaves)?,
        };
        for (lane, value) in texts.iter_mut().enumerate() {
            if let Some(text) = live(lane) {
                *value = rlmesh_spaces::SpaceValue::Text(text);
            }
        }
        let slab = rlmesh_spaces::leaves::encode_leaf_slab(&texts, space)
            .map_err(|err| RuntimeError::Protocol(format!("instruction leaf: {err}")))?
            .pop()
            .unwrap_or_default();
        let mut rewritten = leaves.clone();
        rewritten[*leaf] = slab;
        event.raw_observation = event.observation.replace(rewritten);
        Ok(())
    }
}

/// A fatal transform's failure, naming the hook.
pub(super) fn transform_failed(hook: &str, err: HookError) -> RuntimeError {
    tracing::warn!("runtime hook {hook} failed: {err}");
    RuntimeError::Hook(HookError::Message(format!("{hook}: {err}")))
}

/// The canonical leaf index and space of the Text observation leaf the env
/// tags `text/instruction`, walking its tag tree beside the space.
pub(super) fn instruction_leaf(
    contract: &rlmesh_spaces::EnvContract,
) -> Option<(usize, rlmesh_spaces::SpaceSpec)> {
    use crate::adapters::{EnvTags, INSTRUCTION, ObsLeaf, ObsNode};
    use rlmesh_spaces::SpaceKind;
    fn leaves(space: &rlmesh_spaces::SpaceSpec) -> usize {
        rlmesh_spaces::leaf_specs(space).len()
    }
    fn walk<'a>(
        node: Option<&ObsNode>,
        space: &'a rlmesh_spaces::SpaceSpec,
        offset: &mut usize,
    ) -> Option<&'a rlmesh_spaces::SpaceSpec> {
        match (node, space.spec.as_ref()) {
            (Some(ObsNode::Dict(nodes)), Some(SpaceKind::Dict(dict))) => {
                dict.keys.iter().zip(&dict.spaces).find_map(|(key, child)| {
                    let node = nodes.iter().find(|(k, _)| k == key).map(|(_, n)| n);
                    walk(node, child, offset)
                })
            }
            (Some(ObsNode::Tuple(nodes)), Some(SpaceKind::Tuple(tuple))) => tuple
                .spaces
                .iter()
                .enumerate()
                .find_map(|(i, child)| walk(nodes.get(i), child, offset)),
            (Some(ObsNode::Leaf(ObsLeaf::TextTag(tag))), Some(SpaceKind::Text(_)))
                if tag.role == INSTRUCTION =>
            {
                Some(space)
            }
            _ => {
                *offset += leaves(space);
                None
            }
        }
    }
    let facade = crate::specs::EnvContract::from_wire(contract).ok()?;
    let tags = EnvTags::from_metadata(facade.metadata()?).ok()??;
    let space = contract.observation_space.as_ref()?;
    let mut offset = 0;
    let found = walk(Some(tags.observation()), space, &mut offset)?;
    Some((offset, found.clone()))
}
