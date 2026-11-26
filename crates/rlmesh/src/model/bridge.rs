//! The facade [`Model`](super::Model) on the engine's seams: a
//! [`PredictFn`] that queues each corner and hook call on the model's
//! thread, a [`RouteResolver`] that resolves the model's spec per route, and
//! the route-kind dispatch of a spec-less route.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use rlmesh_adapters::v1::{ENV_METADATA_KEY, Value as AdapterValue};

use super::engine::{AdaptedModelHandler, ShortChunkWarned, short_chunk_text};
use super::facade::{Row, Shared};
use super::handler::PredictFrames;
use super::policy::{Corners, SpecArg};
use super::predict_fn::{PredictFn, RouteConfig, RouteResolver};
use super::types::{EpisodeInfo, ModelObservation};
use crate::binding::spaces::{EnvContract, SpaceValue};
use crate::ext::TensorExt;
use crate::{DType, Error, Metadata, Result, Tensor, Value};

pub(crate) fn engine_handler(shared: Arc<Shared>) -> AdaptedModelHandler {
    let trust = shared.decl.trust_entrypoints;
    engine_handler_trusting(shared, trust)
}

pub(crate) fn engine_handler_trusting(
    shared: Arc<Shared>,
    trust_entrypoints: bool,
) -> AdaptedModelHandler {
    engine_handler_configured(shared, trust_entrypoints, None, None)
}

pub(crate) fn engine_handler_configured(
    shared: Arc<Shared>,
    trust_entrypoints: bool,
    instruction: Option<String>,
    context: Option<Arc<crate::run::hooks::LibraryContext>>,
) -> AdaptedModelHandler {
    let envs = RouteEnvs::default();
    let resolver = Arc::new(SpecResolver {
        spec: shared.decl.spec.clone(),
        trust_entrypoints,
        context,
        envs: envs.clone(),
    });
    AdaptedModelHandler::new(
        Arc::new(ModelPredict {
            shared,
            warned: ShortChunkWarned::default(),
            instruction,
            envs,
        }),
        Some(resolver),
    )
}

pub(crate) fn to_adapter(value: &Value) -> Result<AdapterValue> {
    Ok(match value {
        Value::Tensor(tensor) => AdapterValue::Tensor(tensor.core().clone()),
        Value::Text(text) => AdapterValue::Text(text.clone()),
        Value::Bytes(bytes) => AdapterValue::Bytes(bytes.clone()),
        Value::Int(number) => AdapterValue::Int(*number),
        Value::Float(number) => AdapterValue::Number(*number),
        Value::Bool(flag) => AdapterValue::Bool(*flag),
        Value::Tuple(items) => {
            AdapterValue::Tuple(items.iter().map(to_adapter).collect::<Result<_>>()?)
        }
        Value::List(items) => {
            AdapterValue::List(items.iter().map(to_adapter).collect::<Result<_>>()?)
        }
        Value::Dict(map) => AdapterValue::Map(
            map.iter()
                .map(|(key, item)| Ok((key.to_string(), to_adapter(item)?)))
                .collect::<Result<BTreeMap<_, _>>>()?,
        ),
        Value::None => AdapterValue::None,
        #[allow(unreachable_patterns)]
        _ => return Err(Error::Type("unsupported model value".to_string())),
    })
}

pub(crate) fn from_adapter(value: AdapterValue) -> Result<Value> {
    Ok(match value {
        AdapterValue::None => Value::None,
        AdapterValue::Bool(b) => Value::Bool(b),
        AdapterValue::Int(i) => Value::Int(i),
        AdapterValue::Tuple(items) => {
            Value::Tuple(items.into_iter().map(from_adapter).collect::<Result<_>>()?)
        }
        AdapterValue::Tensor(tensor) => Value::Tensor(crate::binding::tensor_from_core(tensor)?),
        AdapterValue::Text(text) => Value::Text(text),
        AdapterValue::Bytes(bytes) => Value::Bytes(bytes),
        AdapterValue::Number(number) => Value::Float(number),
        AdapterValue::List(items) => {
            Value::List(items.into_iter().map(from_adapter).collect::<Result<_>>()?)
        }
        AdapterValue::Map(map) => {
            let mut out = Metadata::new();
            for (key, item) in map {
                out.insert(key, from_adapter(item)?);
            }
            Value::Dict(out)
        }
    })
}

pub(crate) fn split_frames(chunk: Value) -> Result<Vec<Value>> {
    match chunk {
        Value::List(items) => Ok(items),
        Value::Tensor(tensor) if tensor.ndim() == 0 => Ok(vec![Value::Tensor(tensor)]),
        Value::Tensor(tensor) => Ok(tensor.unstack()?.into_iter().map(Value::Tensor).collect()),
        Value::Dict(map) => {
            let mut fields: Vec<(String, Vec<Value>)> = Vec::with_capacity(map.len());
            let mut horizon: Option<usize> = None;
            for (key, value) in map.iter() {
                let frames = split_frames(value.clone())?;
                match horizon {
                    Some(h) if h != frames.len() => {
                        return Err(Error::invalid_argument(format!(
                            "action chunk fields disagree on horizon: {h} vs {}",
                            frames.len()
                        )));
                    }
                    _ => horizon = Some(frames.len()),
                }
                fields.push((key.to_string(), frames));
            }
            Ok((0..horizon.unwrap_or(0))
                .map(|index| {
                    let mut frame = Metadata::new();
                    for (key, frames) in &fields {
                        frame.insert(key.clone(), frames[index].clone());
                    }
                    Value::Dict(frame)
                })
                .collect())
        }
        other => Ok(vec![other]),
    }
}

fn chunk_to_adapter(chunk: Value) -> Result<AdapterValue> {
    Ok(AdapterValue::List(
        split_frames(chunk)?
            .iter()
            .map(to_adapter)
            .collect::<Result<_>>()?,
    ))
}

fn scalar_tensor<T: Copy>(
    values: &[T],
    dtype: DType,
    bytes: impl Fn(T) -> Vec<u8>,
) -> Result<Value> {
    let data: Vec<u8> = values.iter().flat_map(|value| bytes(*value)).collect();
    Ok(Value::Tensor(Tensor::from_vec(
        data,
        &[values.len()],
        dtype,
    )?))
}

pub(crate) fn stack_values(lanes: Vec<Value>) -> Result<Value> {
    let Some(first) = lanes.first() else {
        return Ok(Value::List(Vec::new()));
    };
    match first {
        Value::Tensor(_) if lanes.iter().all(|lane| matches!(lane, Value::Tensor(_))) => {
            let tensors: Vec<Tensor> = lanes
                .into_iter()
                .map(|lane| match lane {
                    Value::Tensor(tensor) => tensor,
                    _ => unreachable!("checked above"),
                })
                .collect();
            Ok(Value::Tensor(Tensor::stack(&tensors)?))
        }
        Value::Int(_) if lanes.iter().all(|lane| matches!(lane, Value::Int(_))) => {
            let values: Vec<i64> = lanes.iter().filter_map(|lane| lane.as_int().ok()).collect();
            scalar_tensor(&values, DType::Int64, |value| value.to_le_bytes().to_vec())
        }
        Value::Float(_) if lanes.iter().all(|lane| matches!(lane, Value::Float(_))) => {
            let values: Vec<f64> = lanes
                .iter()
                .filter_map(|lane| lane.as_float().ok())
                .collect();
            scalar_tensor(&values, DType::Float64, |value| {
                value.to_le_bytes().to_vec()
            })
        }
        Value::Bool(_) if lanes.iter().all(|lane| matches!(lane, Value::Bool(_))) => {
            let values: Vec<bool> = lanes
                .iter()
                .filter_map(|lane| lane.as_bool().ok())
                .collect();
            scalar_tensor(&values, DType::Bool, |value| vec![u8::from(value)])
        }
        Value::Dict(first) => {
            let keys: Vec<String> = first.keys().map(str::to_string).collect();
            let mut out = Metadata::new();
            for key in keys {
                let column = lanes
                    .iter()
                    .map(|lane| {
                        lane.as_dict()
                            .ok()
                            .and_then(|map| map.get(&key).cloned())
                            .ok_or_else(|| {
                                Error::invalid_argument(format!(
                                    "lanes disagree on dict key '{key}'"
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                out.insert(key, stack_values(column)?);
            }
            Ok(Value::Dict(out))
        }
        Value::Tuple(first) => {
            let width = first.len();
            let mut out = Vec::with_capacity(width);
            for index in 0..width {
                let column = lanes
                    .iter()
                    .map(|lane| {
                        lane.as_tuple()
                            .ok()
                            .and_then(|items| items.get(index).cloned())
                            .ok_or_else(|| {
                                Error::invalid_argument("lanes disagree on tuple length")
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                out.push(stack_values(column)?);
            }
            Ok(Value::Tuple(out))
        }
        _ => Ok(Value::List(lanes)),
    }
}

pub(crate) fn unstack_value(value: Value, n: usize) -> Result<Vec<Value>> {
    let lanes = match value {
        Value::Tensor(tensor) if tensor.ndim() >= 1 => {
            tensor.unstack()?.into_iter().map(Value::Tensor).collect()
        }
        Value::List(items) => items,
        Value::Dict(map) => {
            let mut columns: Vec<(String, Vec<Value>)> = Vec::with_capacity(map.len());
            for (key, item) in map.iter() {
                columns.push((key.to_string(), unstack_value(item.clone(), n)?));
            }
            (0..n)
                .map(|index| {
                    let mut lane = Metadata::new();
                    for (key, column) in &columns {
                        lane.insert(key.clone(), column[index].clone());
                    }
                    Value::Dict(lane)
                })
                .collect()
        }
        Value::Tuple(items) => {
            let columns = items
                .into_iter()
                .map(|item| unstack_value(item, n))
                .collect::<Result<Vec<_>>>()?;
            (0..n)
                .map(|index| {
                    Value::Tuple(columns.iter().map(|column| column[index].clone()).collect())
                })
                .collect()
        }
        other => {
            return Err(Error::Type(format!(
                "predict received a batch of {n} observations from a vector env and must return \
                 {n} actions with a leading batch axis, got a {:?}",
                other.kind()
            )));
        }
    };
    if lanes.len() != n {
        return Err(Error::Type(format!(
            "predict received a batch of {n} observations from a vector env and must return {n} \
             actions with a leading batch axis, got {}; define predict_batch for a batched \
             forward, or serve the env as lanes for one observation per predict",
            lanes.len()
        )));
    }
    Ok(lanes)
}

struct ModelPredict {
    instruction: Option<String>,
    envs: RouteEnvs,
    shared: Arc<Shared>,
    warned: ShortChunkWarned,
}

#[derive(Clone, Default)]
struct RouteEnvs(Arc<std::sync::Mutex<std::collections::HashMap<String, Arc<crate::EnvContract>>>>);

impl RouteEnvs {
    fn map(
        &self,
    ) -> std::sync::MutexGuard<'_, std::collections::HashMap<String, Arc<crate::EnvContract>>> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn get(&self, env_id: &str) -> Option<Arc<crate::EnvContract>> {
        self.map().get(env_id).cloned()
    }

    fn remember(&self, env_id: &str, contract: Option<&EnvContract>) {
        let Some(contract) = contract else {
            return;
        };
        if env_id.is_empty() || self.map().contains_key(env_id) {
            return;
        }
        if let Ok(contract) = crate::EnvContract::from_wire(contract) {
            self.map().insert(env_id.to_string(), Arc::new(contract));
        }
    }

    fn row(&self, episode: Option<&EpisodeInfo>) -> Row {
        let mut row = Row::of(episode);
        row.env = self.get(&row.env_id);
        row
    }

    fn rows(&self, episodes: &[EpisodeInfo]) -> Vec<Row> {
        episodes
            .iter()
            .map(|episode| self.row(Some(episode)))
            .collect()
    }
}

impl ModelPredict {
    fn corners(&self) -> Corners {
        self.shared.decl.corners
    }

    fn encode_action(
        value: &Value,
        space: &crate::binding::spaces::SpaceSpec,
    ) -> Result<SpaceValue> {
        crate::binding::value_to_space(space, value)
    }
}

struct SpecLessLanes {
    lanes: Vec<Value>,
    rows: Vec<Row>,
    action_space: crate::binding::spaces::SpaceSpec,
    num_envs: usize,
    route: String,
}

fn spec_less_warn_key(row: &Row, route: &str) -> String {
    if row.episode_id.is_empty() {
        format!("route:{route}")
    } else {
        row.episode_id.clone()
    }
}

impl ModelPredict {
    fn stage_spec_less(&self, observation: ModelObservation) -> Result<SpecLessLanes> {
        let contract = observation.env_contract.clone();
        let observation_space = contract
            .as_ref()
            .and_then(|contract| contract.observation_space.clone())
            .ok_or_else(|| {
                Error::Runtime("model worker requires observation space metadata".to_string())
            })?;
        let action_space = contract
            .as_ref()
            .and_then(|contract| contract.action_space.clone())
            .ok_or_else(|| {
                Error::Runtime("model worker requires action space metadata".to_string())
            })?;
        let lanes: Vec<Value> = if observation.observation.is_some() {
            observation
                .decoded_lanes()?
                .into_iter()
                .map(|lane| crate::binding::value_from_space(&observation_space, lane))
                .collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        let route = observation.route.env_id;
        self.envs.remember(&route, contract.as_deref());
        let rows = observation
            .route
            .episodes
            .iter()
            .map(|episode| {
                let mut row = Row::of(Some(episode));
                if row.env_id.is_empty() {
                    row.env_id = route.clone();
                }
                row.env = self.envs.get(&row.env_id);
                row
            })
            .collect();
        Ok(SpecLessLanes {
            lanes,
            rows,
            action_space,
            num_envs: observation.num_envs,
            route,
        })
    }

    /// The fused bucket a spec-less group joins at `horizon`: `1` for the batched corner,
    /// the horizon for the batched chunk corner, `None` when it has no batched corner.
    fn spec_less_bucket(&self, horizon: u32) -> Option<u32> {
        let horizon = horizon.max(1);
        if horizon > 1 && self.shared.decl.has_chunk() {
            self.corners()
                .contains(Corners::PREDICT_CHUNK_BATCH)
                .then_some(horizon)
        } else {
            self.corners().contains(Corners::PREDICT_BATCH).then_some(1)
        }
    }

    /// Serve several spec-less groups' lanes in ONE batched corner call: `predict_batch`
    /// without a horizon, `predict_chunk_batch` with one. Each group keeps its own
    /// outcome; a corner failure fails every group in the call.
    fn predict_spec_less_fused(
        &self,
        groups: Vec<SpecLessLanes>,
        horizon: Option<u32>,
    ) -> Vec<Result<PredictFrames>> {
        let mut inputs = Vec::new();
        let mut rows = Vec::new();
        let mut split = Vec::with_capacity(groups.len());
        for group in groups {
            if group.rows.len() != group.lanes.len() {
                split.push(Err(Error::model(format!(
                    "predict carries {} episode rows for {} lanes; episode identity is \
                     row-aligned with the observation",
                    group.rows.len(),
                    group.lanes.len()
                ))));
                continue;
            }
            let keys: Vec<String> = group
                .rows
                .iter()
                .map(|row| spec_less_warn_key(row, &group.route))
                .collect();
            inputs.extend(group.lanes);
            rows.extend(group.rows);
            split.push(Ok((keys, group.action_space)));
        }
        let total = inputs.len();
        let (corner, called) = match horizon {
            Some(h) => (
                "predict_chunk_batch",
                self.shared
                    .call(move |core| core.predict_chunk_batch(inputs, h, &rows)),
            ),
            None => (
                "predict_batch",
                self.shared
                    .call(move |core| core.predict_batch(inputs, &rows)),
            ),
        };
        let called = called.and_then(|items| {
            if items.len() == total {
                Ok(items)
            } else {
                Err(Error::model(format!(
                    "{corner} returned {} results for {total} lanes",
                    items.len()
                )))
            }
        });
        let mut items = match called {
            Ok(items) => items.into_iter(),
            Err(error) => {
                return split
                    .into_iter()
                    .map(|group| group.and(Err(error.clone())))
                    .collect();
            }
        };
        split
            .into_iter()
            .map(|group| {
                let (keys, action_space) = group?;
                let lane_items: Vec<Value> = items.by_ref().take(keys.len()).collect();
                match horizon {
                    None => Ok(PredictFrames {
                        actions: lane_items
                            .iter()
                            .map(|action| Self::encode_action(action, &action_space))
                            .collect::<Result<_>>()?,
                        replay: Vec::new(),
                    }),
                    Some(h) => PredictFrames::from_lane_frames(
                        lane_items
                            .into_iter()
                            .zip(&keys)
                            .map(|(chunk, key)| self.chunk_lane(chunk, h, key, &action_space))
                            .collect::<Result<Vec<_>>>()?,
                    ),
                }
            })
            .collect()
    }

    /// One lane's native chunk as its executed frames: checked against a declared
    /// `native_chunk`, warned once per episode when short, cut to `horizon`.
    fn chunk_lane(
        &self,
        chunk: Value,
        horizon: u32,
        warn_key: &str,
        action_space: &crate::binding::spaces::SpaceSpec,
    ) -> Result<Vec<SpaceValue>> {
        let mut frames = split_frames(chunk)?;
        match self.shared.decl.native_chunk {
            Some(native) if frames.len() != native as usize => {
                return Err(Error::invalid_argument(format!(
                    "model declares native_chunk={native} but its chunk corner returned {} \
                     frames at execution_horizon={horizon}: return the WHOLE native chunk and \
                     let the runtime execute its prefix (do not slice to the horizon), or drop \
                     the declaration",
                    frames.len()
                )));
            }
            None if frames.len() < horizon as usize && self.warned.first(warn_key) => {
                crate::warnings::warn(
                    crate::WarningCategory::RuntimeWarning,
                    short_chunk_text(frames.len(), horizon),
                )?;
            }
            _ => {}
        }
        frames.truncate(horizon as usize);
        if frames.is_empty() {
            return Err(Error::invalid_argument(
                "a chunked model (execute_horizon>1) returned an empty action chunk",
            ));
        }
        frames
            .iter()
            .map(|frame| Self::encode_action(frame, action_space))
            .collect()
    }
}

impl ModelPredict {
    fn input(&self, value: AdapterValue) -> Result<Value> {
        let value = from_adapter(value)?;
        Ok(match &self.shared.decl.spec {
            Some(SpecArg::ModelSpec(spec)) => crate::adapters::inject_instruction(
                crate::adapters::restore_shape(value, spec.input()),
                spec.input(),
                self.instruction.as_deref(),
            ),
            _ => value,
        })
    }
}

impl PredictFn for ModelPredict {
    fn predict(
        &self,
        model_input: AdapterValue,
        episode: Option<&EpisodeInfo>,
    ) -> Result<AdapterValue> {
        let input = self.input(model_input)?;
        let row = self.envs.row(episode);
        let action = self.shared.call(move |core| core.predict(input, &row))?;
        to_adapter(&action)
    }

    fn predict_chunk(
        &self,
        model_input: AdapterValue,
        execution_horizon: u32,
        episode: Option<&EpisodeInfo>,
    ) -> Result<Option<AdapterValue>> {
        if !self.shared.decl.has_chunk() {
            return Ok(None);
        }
        let input = self.input(model_input)?;
        let row = self.envs.row(episode);
        let chunk = self
            .shared
            .call(move |core| core.predict_chunk(input, execution_horizon, &row))?;
        chunk_to_adapter(chunk).map(Some)
    }

    fn has_chunk(&self) -> bool {
        self.shared.decl.has_chunk()
    }

    fn predict_batch(
        &self,
        inputs: Vec<AdapterValue>,
        episodes: &[EpisodeInfo],
    ) -> Result<Vec<AdapterValue>> {
        let inputs = inputs
            .into_iter()
            .map(|v| self.input(v))
            .collect::<Result<Vec<_>>>()?;
        let rows = self.envs.rows(episodes);
        let actions = self
            .shared
            .call(move |core| core.predict_batch(inputs, &rows))?;
        actions.iter().map(to_adapter).collect()
    }

    fn has_batch(&self) -> bool {
        self.corners().contains(Corners::PREDICT_BATCH)
    }

    fn predict_chunk_batch(
        &self,
        inputs: Vec<AdapterValue>,
        execution_horizon: u32,
        episodes: &[EpisodeInfo],
    ) -> Result<Vec<AdapterValue>> {
        let inputs = inputs
            .into_iter()
            .map(|v| self.input(v))
            .collect::<Result<Vec<_>>>()?;
        let rows = self.envs.rows(episodes);
        let chunks = self
            .shared
            .call(move |core| core.predict_chunk_batch(inputs, execution_horizon, &rows))?;
        chunks.into_iter().map(chunk_to_adapter).collect()
    }

    fn has_chunk_batch(&self) -> bool {
        self.corners().contains(Corners::PREDICT_CHUNK_BATCH)
    }

    fn allow_fusion(&self) -> bool {
        self.shared.decl.allow_fusion && (self.has_batch() || self.has_chunk_batch())
    }

    fn native_chunk(&self) -> Option<u32> {
        self.shared.decl.native_chunk
    }

    fn predict_spec_less(&self, observation: ModelObservation) -> Result<Vec<SpaceValue>> {
        Ok(self.predict_spec_less_chunked(observation, 1)?.actions)
    }

    fn predict_spec_less_chunked(
        &self,
        observation: ModelObservation,
        execution_horizon: u32,
    ) -> Result<PredictFrames> {
        let staged = self.stage_spec_less(observation)?;
        let horizon = execution_horizon.max(1);
        if staged.num_envs <= 1 || staged.lanes.len() <= 1 {
            let SpecLessLanes {
                lanes,
                rows,
                action_space,
                route,
                ..
            } = staged;
            let input = lanes.into_iter().next().unwrap_or(Value::None);
            let row = rows.into_iter().next().unwrap_or_default();
            if horizon > 1 && self.shared.decl.has_chunk() {
                let warn_key = spec_less_warn_key(&row, &route);
                let chunk = self
                    .shared
                    .call(move |core| core.predict_chunk(input, horizon, &row))?;
                let frames = self.chunk_lane(chunk, horizon, &warn_key, &action_space)?;
                return PredictFrames::from_lane_frames(vec![frames]);
            }
            let action = self.shared.call(move |core| core.predict(input, &row))?;
            return Ok(PredictFrames {
                actions: vec![Self::encode_action(&action, &action_space)?],
                replay: Vec::new(),
            });
        }
        let bucket = self
            .spec_less_bucket(horizon)
            .or_else(|| self.corners().contains(Corners::PREDICT_BATCH).then_some(1));
        if let Some(key) = bucket {
            let horizon = (key > 1).then_some(key);
            return self
                .predict_spec_less_fused(vec![staged], horizon)
                .pop()
                .expect("one outcome per group");
        }

        let SpecLessLanes {
            lanes,
            action_space,
            num_envs,
            route,
            ..
        } = staged;
        let batched = stack_values(lanes)?;
        let row = Row {
            env: self.envs.get(&route),
            ..Row::default()
        };
        let action = self.shared.call(move |core| core.predict(batched, &row))?;
        let actions = unstack_value(action, num_envs)?;
        Ok(PredictFrames {
            actions: actions
                .iter()
                .map(|action| Self::encode_action(action, &action_space))
                .collect::<Result<_>>()?,
            replay: Vec::new(),
        })
    }

    fn predict_spec_less_grouped(
        &self,
        groups: Vec<(ModelObservation, u32)>,
    ) -> Vec<Result<PredictFrames>> {
        let mut outcomes: Vec<Option<Result<PredictFrames>>> =
            (0..groups.len()).map(|_| None).collect();
        let mut buckets: BTreeMap<u32, Vec<(usize, ModelObservation, u32)>> = BTreeMap::new();
        let mut single: Vec<(usize, ModelObservation, u32)> = Vec::new();
        for (index, (observation, horizon)) in groups.into_iter().enumerate() {
            match self.spec_less_bucket(horizon) {
                Some(key) if observation.observation.is_some() => {
                    buckets
                        .entry(key)
                        .or_default()
                        .push((index, observation, horizon));
                }
                _ => single.push((index, observation, horizon)),
            }
        }
        for (key, members) in buckets {
            if members.len() == 1 {
                single.extend(members);
                continue;
            }
            let mut indices = Vec::with_capacity(members.len());
            let mut staged = Vec::with_capacity(members.len());
            for (index, observation, _) in members {
                match self.stage_spec_less(observation) {
                    Ok(lanes) => {
                        indices.push(index);
                        staged.push(lanes);
                    }
                    Err(error) => outcomes[index] = Some(Err(error)),
                }
            }
            if staged.is_empty() {
                continue;
            }
            let horizon = (key > 1).then_some(key);
            for (index, outcome) in indices
                .into_iter()
                .zip(self.predict_spec_less_fused(staged, horizon))
            {
                outcomes[index] = Some(outcome);
            }
        }
        for (index, observation, horizon) in single {
            outcomes[index] = Some(self.predict_spec_less_chunked(observation, horizon));
        }
        outcomes
            .into_iter()
            .map(|outcome| {
                outcome.unwrap_or_else(|| {
                    Err(Error::Internal(
                        "spec-less grouped predict left a group unserved".to_string(),
                    ))
                })
            })
            .collect()
    }

    fn on_episode_end(&self, episode_id: &str) -> Result<()> {
        self.warned.remove(episode_id);
        let episode_id = episode_id.to_string();
        self.shared.call(move |core| core.episode_end(&episode_id))
    }

    fn on_route_reset(&self, env_id: &str) -> Result<()> {
        let env_id = env_id.to_string();
        self.shared.call(move |core| {
            core.drop_env(&env_id);
            Ok(())
        })
    }

    fn on_route_release(&self, env_id: &str) -> Result<()> {
        self.envs.map().remove(env_id);
        self.on_route_reset(env_id)
    }

    fn has_reset_options(&self) -> bool {
        self.shared.decl.reset_options
    }

    fn reset_options(
        &self,
        episode: &EpisodeInfo,
        trial_index: Option<u64>,
    ) -> Result<Option<rlmesh_spaces::MetaMap>> {
        if !self.shared.decl.reset_options {
            return Ok(None);
        }
        let mut start = Metadata::new();
        start.insert("episode_id", Value::Text(episode.episode_id.clone()));
        start.insert("seed", episode.seed.map_or(Value::None, Value::Int));
        start.insert(
            "trial_index",
            trial_index.map_or(Value::None, |trial| Value::Int(trial as i64)),
        );
        let options = self.shared.call(move |core| core.reset_options(&start))?;
        options
            .map(|options| crate::value::meta_codec::metadata_to_wire(&options))
            .transpose()
    }
}

struct SpecResolver {
    context: Option<Arc<crate::run::hooks::LibraryContext>>,
    spec: Option<SpecArg>,
    trust_entrypoints: bool,
    envs: RouteEnvs,
}

#[async_trait]
impl RouteResolver for SpecResolver {
    async fn resolve(
        &self,
        route_key: &str,
        env_contract: &EnvContract,
    ) -> Result<Option<RouteConfig>> {
        self.envs.map().remove(route_key);
        self.envs.remember(route_key, Some(env_contract));
        if matches!(self.spec, Some(SpecArg::NoAdapter)) {
            return Ok(None);
        }
        let metadata = env_contract
            .metadata
            .clone()
            .map(crate::binding::metadata_from_wire)
            .unwrap_or_default();
        let tagged = metadata.contains_key(ENV_METADATA_KEY);
        let spec = match &self.spec {
            None if tagged => {
                return Err(Error::adapter_resolution(
                    "the env publishes adapter tags but this model has spec=None; pass \
                     spec=<ModelSpec> to adapt, or spec=NO_ADAPTER if the model adapts its own \
                     observations",
                ));
            }
            None => return Ok(None),
            Some(SpecArg::NoAdapter) => return Ok(None),
            Some(SpecArg::ModelSpec(spec)) => spec,
        };
        let observation_space = env_contract
            .observation_space
            .clone()
            .ok_or_else(|| Error::invalid_argument("env contract missing observation_space"))?;
        let action_space = env_contract
            .action_space
            .clone()
            .ok_or_else(|| Error::invalid_argument("env contract missing action_space"))?;
        let contract = crate::EnvContract::from_wire(env_contract)?;
        let tags = crate::adapters::contract_tags(&contract)?;
        let parts = crate::adapters::resolve_parts(
            &tags,
            contract.observation_space(),
            contract.action_space(),
            spec,
            crate::adapters::ResolveOptions {
                trust_entrypoints: self.trust_entrypoints,
                ..Default::default()
            },
        )?;
        if let Some(context) = &self.context {
            context.set_advisories(
                rlmesh_adapters::v1::advisories(&parts.plan)
                    .into_iter()
                    .map(crate::adapters::Advisory::from_core)
                    .collect(),
            );
        }
        Ok(Some(RouteConfig::new(
            parts.plan,
            observation_space,
            action_space,
            Box::new(parts.customs),
            Box::new(parts.encodings),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(values: &[f32], shape: &[usize]) -> Value {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Value::Tensor(Tensor::from_vec(data, shape, DType::Float32).unwrap())
    }

    #[test]
    fn split_frames_keeps_a_tuple_whole_and_splits_lists_tensors_and_dicts() {
        let tuple = Value::Tuple(vec![Value::Int(1), Value::Int(2)]);
        assert_eq!(split_frames(tuple.clone()).unwrap(), vec![tuple]);
        assert_eq!(
            split_frames(Value::List(vec![Value::Int(1), Value::Int(2)]))
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            split_frames(tensor(&[1.0, 2.0, 3.0], &[3])).unwrap().len(),
            3
        );
        assert_eq!(split_frames(tensor(&[1.0], &[])).unwrap().len(), 1);
        let mut dict = Metadata::new();
        dict.insert("a", tensor(&[1.0, 2.0], &[2]));
        dict.insert("b", Value::List(vec![Value::Int(0), Value::Int(1)]));
        let frames = split_frames(Value::Dict(dict)).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[1].as_dict().unwrap().get("b"), Some(&Value::Int(1)));
        assert_eq!(split_frames(Value::Int(4)).unwrap(), vec![Value::Int(4)]);
    }

    #[test]
    fn stack_and_unstack_round_trip() {
        let lanes = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
        let stacked = stack_values(lanes).unwrap();
        assert_eq!(stacked.as_tensor().unwrap().shape(), &[3]);
        let back = unstack_value(stacked, 3).unwrap();
        assert_eq!(back.len(), 3);
        let error = unstack_value(Value::Int(0), 3).unwrap_err();
        assert_eq!(error.class(), crate::ErrorClass::TypeError);
    }

    #[test]
    fn values_cross_the_adapter_seam() {
        let value = Value::Tuple(vec![Value::Int(2), Value::Text("x".into())]);
        let adapter = to_adapter(&value).unwrap();
        assert_eq!(from_adapter(adapter).unwrap(), value);
        assert_eq!(
            from_adapter(to_adapter(&Value::None).unwrap()).unwrap(),
            Value::None
        );
    }
}
