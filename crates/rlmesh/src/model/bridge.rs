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

/// The engine handler for one model: its predict hole and its route
/// resolver (always present, so every route resolves, spec'd or not).
pub(crate) fn engine_handler(shared: Arc<Shared>) -> AdaptedModelHandler {
    let trust = shared.decl.trust_entrypoints;
    engine_handler_trusting(shared, trust)
}

/// [`engine_handler`] with a run's or session's `trust_entrypoints`.
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
    let resolver = Arc::new(SpecResolver {
        spec: shared.decl.spec.clone(),
        trust_entrypoints,
        context,
    });
    AdaptedModelHandler::new(
        Arc::new(ModelPredict {
            shared,
            warned: ShortChunkWarned::default(),
            instruction,
        }),
        Some(resolver),
    )
}

/// A facade value as the engine's adapter value.
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

/// An engine adapter value as a facade value.
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

/// Split a chunk into frames by the native `split_chunk` rule: a `List` into
/// its items, a tensor along its leading axis (a 0-d one is one frame), a
/// `Dict` per field; a `Tuple` and any other value are one frame.
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

/// A chunk for the engine: pre-split by the native rule, so the engine's
/// own split (of a list) yields exactly these frames.
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

/// Stack per-lane values on a new leading axis N: tensors stack, numbers and
/// flags become a 1-d tensor, dicts and tuples stack per field, anything
/// else becomes a list of the lanes.
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

/// Split a batched value back into `n` lanes along its leading axis (the
/// inverse of [`stack_values`]).
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

/// The model's predict hole: every corner and hook call queued on the
/// model's thread.
struct ModelPredict {
    instruction: Option<String>,
    shared: Arc<Shared>,
    /// The spec-less route's short-chunk once-set (the spec'd routes use the
    /// engine's).
    warned: ShortChunkWarned,
}

fn rows(episodes: &[EpisodeInfo]) -> Vec<Row> {
    episodes
        .iter()
        .map(|episode| Row::of(Some(episode)))
        .collect()
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
        let row = Row::of(episode);
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
        let row = Row::of(episode);
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
        let rows = rows(episodes);
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
        let rows = rows(episodes);
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
        let num_envs = observation.num_envs;
        let lanes: Vec<Value> = if observation.observation.is_some() {
            observation
                .decoded_lanes()?
                .into_iter()
                .map(|lane| crate::binding::value_from_space(&observation_space, lane))
                .collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        let rows: Vec<Row> = observation
            .route
            .episodes
            .iter()
            .map(|episode| {
                let mut row = Row::of(Some(episode));
                if row.env_id.is_empty() {
                    row.env_id = observation.route.env_id.clone();
                }
                row
            })
            .collect();
        let horizon = execution_horizon.max(1);
        let encode = |value: &Value| Self::encode_action(value, &action_space);

        // N = 1: predict, or predict_chunk at a horizon above 1, with the
        // lane's own context.
        if num_envs <= 1 || lanes.len() <= 1 {
            let input = lanes.into_iter().next().unwrap_or(Value::None);
            let row = rows.into_iter().next().unwrap_or_default();
            if horizon > 1 && self.shared.decl.has_chunk() {
                let warn_key = if row.episode_id.is_empty() {
                    format!("route:{}", observation.route.env_id)
                } else {
                    row.episode_id.clone()
                };
                let chunk = self
                    .shared
                    .call(move |core| core.predict_chunk(input, horizon, &row))?;
                let mut frames = split_frames(chunk)?;
                match self.shared.decl.native_chunk {
                    Some(native) if frames.len() != native as usize => {
                        return Err(Error::invalid_argument(format!(
                            "model declares native_chunk={native} but its chunk corner returned \
                             {} frames at execution_horizon={horizon}: return the WHOLE native \
                             chunk and let the runtime execute its prefix (do not slice to the \
                             horizon), or drop the declaration",
                            frames.len()
                        )));
                    }
                    None if frames.len() < horizon as usize && self.warned.first(&warn_key) => {
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
                let mut encoded = frames.iter().map(encode).collect::<Result<Vec<_>>>()?;
                let first = encoded.remove(0);
                return Ok(PredictFrames {
                    actions: vec![first],
                    replay: encoded.into_iter().map(|frame| vec![frame]).collect(),
                });
            }
            let action = self.shared.call(move |core| core.predict(input, &row))?;
            return Ok(PredictFrames {
                actions: vec![encode(&action)?],
                replay: Vec::new(),
            });
        }

        // N > 1 with predict_batch: the per-lane values, per-lane contexts.
        if self.corners().contains(Corners::PREDICT_BATCH) {
            let actions = self
                .shared
                .call(move |core| core.predict_batch(lanes, &rows))?;
            if actions.len() != num_envs {
                return Err(Error::model(format!(
                    "predict_batch returned {} actions for {num_envs} lanes",
                    actions.len()
                )));
            }
            return Ok(PredictFrames {
                actions: actions.iter().map(encode).collect::<Result<_>>()?,
                replay: Vec::new(),
            });
        }

        // N > 1 otherwise: one batched value, an anonymous context.
        let batched = stack_values(lanes)?;
        let action = self
            .shared
            .call(move |core| core.predict(batched, &Row::default()))?;
        let actions = unstack_value(action, num_envs)?;
        Ok(PredictFrames {
            actions: actions.iter().map(encode).collect::<Result<_>>()?,
            replay: Vec::new(),
        })
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

/// Resolves the model's spec per route: none for `NO_ADAPTER` or a spec-less
/// model on an untagged env, an adapter for a spec, and a refusal for a
/// tagged env with no spec.
struct SpecResolver {
    context: Option<Arc<crate::run::hooks::LibraryContext>>,
    spec: Option<SpecArg>,
    trust_entrypoints: bool,
}

#[async_trait]
impl RouteResolver for SpecResolver {
    async fn resolve(
        &self,
        _route_key: &str,
        env_contract: &EnvContract,
    ) -> Result<Option<RouteConfig>> {
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
