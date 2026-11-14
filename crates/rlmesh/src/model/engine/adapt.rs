//! The adapter work around a model call: assembling each lane's model input
//! (and ingesting history rows), and applying the returned action frames.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rlmesh_adapters::v1::{
    FrameBuffers, ObsPlan, Value, apply_actions, assemble_obs, observe_obs, space_value_to_obs_map,
};

use super::route::{RouteEntry, admit_fresh_lanes};
use crate::binding::spaces::{SpaceKind, SpaceValue};
use crate::model::handler::PredictFrames;
use crate::model::predict_fn::RouteConfig;
use crate::model::types::{EpisodeInfo, ModelObservation};
use crate::{Error, Result};

/// The top-level obs keys to materialize for `config`: a declarative-only route
/// needs just the referenced keys (lazy); a route with custom holes needs the
/// full observation so the custom callback sees everything.
fn obs_keys(config: &RouteConfig) -> BTreeSet<String> {
    let referenced = config.adapter.referenced_obs_keys();
    let has_customs = config
        .adapter
        .obs_plans
        .iter()
        .any(|plan| matches!(plan, ObsPlan::Custom(_)));
    if !has_customs {
        return referenced;
    }
    // Customs see the full per-lane obs: include every top-level key.
    match config.observation_space.spec.as_ref() {
        Some(SpaceKind::Dict(dict)) => dict.keys.iter().cloned().collect(),
        _ => [".".to_owned()].into_iter().collect(),
    }
}

/// One value's dtype/shape signature for error context: `float32[8]`,
/// `{image: uint8[3, 256, 256], state: float32[8]}`.
fn value_summary(value: &Value) -> String {
    match value {
        Value::Tensor(tensor) => format!("{}{:?}", tensor.dtype().name(), tensor.shape()),
        Value::Text(text) => format!("text(len={})", text.len()),
        Value::Bytes(bytes) => format!("bytes(len={})", bytes.len()),
        Value::None => "None".into(),
        Value::Int(_) => "int".into(),
        Value::Bool(_) => "bool".into(),
        Value::Tuple(v) => format!("tuple(len={})", v.len()),
        Value::Number(_) => "number".to_owned(),
        Value::List(items) => match items.first() {
            Some(first) => format!("list(len={}, first={})", items.len(), value_summary(first)),
            None => "list(len=0)".to_owned(),
        },
        Value::Map(map) => {
            let entries: Vec<String> = map
                .iter()
                .map(|(key, value)| format!("{key}: {}", value_summary(value)))
                .collect();
            format!("{{{}}}", entries.join(", "))
        }
    }
}

/// The signature of the assembled model inputs handed to predict. Lanes share
/// one shape (a homogeneous vectorized fleet), so summarize lane 0 plus the
/// lane count.
pub(super) fn inputs_summary(inputs: &[Value]) -> String {
    match inputs.first() {
        Some(first) => format!(
            "adapter-assembled model input (per lane): {}; lanes: {}",
            value_summary(first),
            inputs.len()
        ),
        None => "adapter-assembled model input: (no lanes)".to_owned(),
    }
}

/// Append the assembled-input signature to a predict failure, so a shape/dtype
/// error raised inside the model states what the adapter actually handed it
/// instead of surfacing as an opaque framework traceback.
pub(super) fn annotate_predict_error(err: Error, summary: &str) -> Error {
    match err {
        Error::Model(mut model) => {
            model.message = format!("{}\n{summary}", model.message);
            Error::Model(model)
        }
        Error::Internal(message) => Error::Internal(format!("{message}\n{summary}")),
        other => other,
    }
}

/// Validate the observation and assemble each lane's model input. Runs under
/// the route's entry lock (frame-stacking mutates the per-episode buffers
/// here, once per call).
///
/// Every lane must carry a non-empty episode_id: the engine keys all
/// per-episode state (frame windows) by it. The session layer already
/// enforces `num_envs == slots.len()` with non-empty ids, but the engine must
/// not silently fall back to a shared "" buffer (which would cross-contaminate
/// lanes) if any other producer violates that.
pub(super) fn assemble_route_inputs(
    entry: &mut RouteEntry,
    observation: &ModelObservation,
    adapter_ns: &AtomicU64,
) -> Result<Vec<Value>> {
    admit_fresh_lanes(entry, &observation.route.episodes)?;
    let started = Instant::now();
    let result = assemble_route_inputs_inner(entry, observation);
    adapter_ns.fetch_add(rlmesh_proto::elapsed_ns(started), Ordering::Relaxed);
    entry.publish_held();
    result
}

fn assemble_route_inputs_inner(
    entry: &mut RouteEntry,
    observation: &ModelObservation,
) -> Result<Vec<Value>> {
    let episodes = &observation.route.episodes;
    let num_envs = observation.num_envs;

    // The wire contract requires every predict request to carry a decodable
    // observation; validate the structure up front (cheap) so a malformed request
    // errors here. Every call re-plans (the runtime owns replay), so the obs is
    // always decoded below.
    observation.ensure_decodable()?;

    let RouteEntry {
        config, buffers, ..
    } = entry;
    let referenced = obs_keys(config);
    let customs: &dyn rlmesh_adapters::v1::CustomTransform = config.customs.as_ref();
    let encodings: &dyn rlmesh_adapters::v1::EncodingTransform = config.encodings.as_ref();

    // A route that negotiated history advances its windows through the
    // replayed steps first, then holds this request's own row to the next
    // step. A route that did not must not be handed rows: the producer thinks
    // it is feeding a window this route does not keep.
    let stamped = if config.delivers_history {
        ingest_history(config, buffers, observation, &referenced)?;
        Some(observation.step.ok_or_else(|| {
            Error::model(
                "predict request carries no step but this route negotiated observation history; \
                 stamp every request and history row with the producer's env-step counter",
            )
        })?)
    } else if observation.history.is_empty() {
        None
    } else {
        return Err(Error::model(
            "predict request carries observation history rows but this route did not negotiate \
             history (ResolveAdapterResponse.history was unset): send rows only to a route that \
             asked for them",
        ));
    };
    // A route that reads its previous action needs every step numbered even
    // when the runtime stamps none (no history offered, so execution_horizon
    // is 1 and every predict is one step): the engine counts the episode's
    // predicts itself. Any other unstamped route holds no per-step state.
    let counted = stamped.is_none() && config.adapter.reads_previous_action();

    let decoded = observation.decoded_lanes()?;

    let mut inputs: Vec<Value> = Vec::with_capacity(num_envs);
    for (index, lane) in decoded.iter().enumerate() {
        let episode_id = episodes
            .get(index)
            .map(|episode| episode.episode_id.as_str())
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                Error::model(format!(
                    "predict request lane {index} has no episode_id (num_envs={num_envs}, \
                     episodes={}); every lane must carry a non-empty episode_id",
                    episodes.len()
                ))
            })?;
        let step = stamped.unwrap_or_else(|| buffers.next_step(episode_id));
        if stamped.is_some() || counted {
            buffers
                .advance_step(episode_id, step)
                .map_err(Error::from_apply)?;
        }
        let raw = space_value_to_obs_map(lane, &config.observation_space, &referenced)
            .map_err(Error::from_apply)?;
        inputs.push(
            assemble_obs(
                &config.adapter,
                &raw,
                episode_id,
                step,
                buffers,
                customs,
                encodings,
            )
            .map_err(Error::from_apply)?,
        );
    }
    Ok(inputs)
}

/// Advance the route's frame windows through the replayed steps the runtime
/// delivered as history rows, oldest first, holding each row to the step after
/// the one its episode last saw. Nothing is assembled: a row only ticks the
/// windows, exactly as [`observe_obs`] does on the local path.
fn ingest_history(
    config: &RouteConfig,
    buffers: &mut FrameBuffers,
    observation: &ModelObservation,
    referenced: &BTreeSet<String>,
) -> Result<()> {
    for frame in &observation.history {
        let lanes = observation.decoded_history_lanes(frame)?;
        if lanes.len() != frame.episodes.len() {
            return Err(Error::model(format!(
                "history row at step {} decoded to {} lanes for {} episode rows",
                frame.step,
                lanes.len(),
                frame.episodes.len()
            )));
        }
        for (lane, episode) in lanes.iter().zip(&frame.episodes) {
            if episode.episode_id.is_empty() {
                return Err(Error::model(format!(
                    "history row at step {} has a lane with no episode_id",
                    frame.step
                )));
            }
            buffers
                .advance_step(&episode.episode_id, frame.step)
                .map_err(Error::from_apply)?;
            let raw = space_value_to_obs_map(lane, &config.observation_space, referenced)
                .map_err(Error::from_apply)?;
            observe_obs(&config.adapter, &raw, &episode.episode_id, buffers)
                .map_err(Error::from_apply)?;
        }
    }
    Ok(())
}
/// Apply the per-step adapter transform to each frame, peel frame 0 (this
/// step's action) from the future frames (which the runtime replays), and
/// transpose the per-lane future frames into per-step batched frames
/// (`replay[step][lane]`), to the shortest lane (uniform for a homogeneous
/// fleet; a short lane caps the batch — receding horizon).
///
/// Each lane's frame `k` is applied under the step it executes at, `k` past
/// the step the lane assembled at, so a route reading its previous action
/// sees, at the re-plan, the frame the runtime replayed last. That record
/// lives in the route's buffers, so the entry lock is taken again here (after
/// the model call, never across it).
pub(super) fn finish_route_frames(
    entry: &Arc<Mutex<RouteEntry>>,
    lane_raw_steps: Vec<Vec<Value>>,
    episodes: &[EpisodeInfo],
    num_envs: usize,
    adapter_ns: &AtomicU64,
) -> Result<PredictFrames> {
    let started = Instant::now();
    let result = {
        let mut guard = entry.lock().expect("route entry poisoned");
        let result = finish_route_frames_inner(&mut guard, lane_raw_steps, episodes, num_envs);
        guard.publish_held();
        result
    };
    adapter_ns.fetch_add(rlmesh_proto::elapsed_ns(started), Ordering::Relaxed);
    result
}

fn finish_route_frames_inner(
    entry: &mut RouteEntry,
    lane_raw_steps: Vec<Vec<Value>>,
    episodes: &[EpisodeInfo],
    num_envs: usize,
) -> Result<PredictFrames> {
    let RouteEntry {
        config, buffers, ..
    } = entry;
    let encodings: &dyn rlmesh_adapters::v1::EncodingTransform = config.encodings.as_ref();
    let mut frame0 = Vec::with_capacity(num_envs);
    let mut lane_replays: Vec<Vec<SpaceValue>> = Vec::with_capacity(num_envs);
    for (lane, raw_steps) in lane_raw_steps.into_iter().enumerate() {
        // Frame `k` executes at `step + k`, where `step` is where the frames
        // the runtime still holds queued end: the episode's last counted step
        // when nothing is queued, or past the last recorded action under an
        // async prefetch lead. A route that counts none holds no per-step
        // state, so the value goes unread.
        let episode_id = episodes
            .get(lane)
            .map_or("", |episode| episode.episode_id.as_str());
        let step = buffers.chunk_start(episode_id);
        let mut applied = raw_steps
            .into_iter()
            .enumerate()
            .map(|(frame, raw_action)| {
                apply_actions(
                    &config.adapter,
                    raw_action,
                    &config.action_space,
                    encodings,
                    buffers,
                    episode_id,
                    step + frame as i64,
                )
            })
            .collect::<std::result::Result<Vec<SpaceValue>, _>>()
            .map_err(Error::from_apply)?
            .into_iter();
        let first = applied.next().ok_or_else(|| {
            Error::invalid_argument("a chunked model returned an empty action chunk")
        })?;
        frame0.push(first);
        lane_replays.push(applied.collect());
    }

    let replay_len = lane_replays.iter().map(Vec::len).min().unwrap_or(0);
    let mut replay = Vec::with_capacity(replay_len);
    for step in 0..replay_len {
        let mut per_lane = Vec::with_capacity(num_envs);
        for lane in &lane_replays {
            per_lane.push(lane[step].clone());
        }
        replay.push(per_lane);
    }

    Ok(PredictFrames {
        actions: frame0,
        replay,
    })
}

#[cfg(test)]
mod input_context_tests {
    use std::collections::BTreeMap;

    use rlmesh_spaces::{DType, Tensor};

    use super::*;

    fn sample_input() -> Value {
        Value::Map(BTreeMap::from([
            (
                "image".to_owned(),
                Value::Tensor(Tensor::from_vec(vec![0; 48], vec![3, 4, 4], DType::Uint8).unwrap()),
            ),
            (
                "state".to_owned(),
                Value::Tensor(Tensor::from_vec(vec![0; 28], vec![7], DType::Float32).unwrap()),
            ),
        ]))
    }

    #[test]
    fn inputs_summary_names_keys_dtypes_shapes_and_lanes() {
        let summary = inputs_summary(&[sample_input(), sample_input()]);
        assert_eq!(
            summary,
            "adapter-assembled model input (per lane): \
             {image: uint8[3, 4, 4], state: float32[7]}; lanes: 2"
        );
    }

    #[test]
    fn model_error_gains_the_input_signature() {
        let annotated = annotate_predict_error(
            Error::model("RuntimeError: size mismatch"),
            "adapter-assembled model input (per lane): {state: float32[7]}; lanes: 1",
        );
        match annotated {
            Error::Model(model) => assert_eq!(
                model.message,
                "RuntimeError: size mismatch\nadapter-assembled model input (per lane): \
                 {state: float32[7]}; lanes: 1"
            ),
            other => panic!("expected Error::Model, got {other:?}"),
        }
    }

    #[test]
    fn transport_errors_pass_through_unannotated() {
        let annotated = annotate_predict_error(Error::Connection("reset".to_owned()), "ctx");
        assert_eq!(annotated, Error::Connection("reset".to_owned()));
    }
}
