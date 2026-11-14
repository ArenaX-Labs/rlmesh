//! Choosing and calling the model's predict corner, and capping the chunk it
//! returns to the execution horizon.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use rlmesh_adapters::v1::{Value, split_chunk};

use super::adapt::{annotate_predict_error, inputs_summary};
use crate::model::predict_fn::PredictFn;
use crate::model::types::EpisodeInfo;
use crate::{Error, Result};

/// The episodes (or, for an anonymous lane, the routes) already warned that
/// their model returned a chunk shorter than the horizon; warns once per
/// episode.
#[derive(Debug, Default)]
pub(crate) struct ShortChunkWarned(Mutex<HashSet<String>>);

impl ShortChunkWarned {
    /// Whether `key` has not been warned yet (and mark it).
    pub(crate) fn first(&self, key: &str) -> bool {
        self.0
            .lock()
            .expect("short-chunk set poisoned")
            .insert(key.to_string())
    }

    /// Forget `key` (its episode ended).
    pub(crate) fn remove(&self, key: &str) {
        self.0.lock().expect("short-chunk set poisoned").remove(key);
    }
}

/// The once-key of a lane's short-chunk warning: its episode, or its route
/// for an anonymous lane.
fn warn_key(episode: Option<&EpisodeInfo>) -> String {
    match episode {
        Some(episode) if !episode.episode_id.is_empty() => episode.episode_id.clone(),
        Some(episode) => format!("route:{}", episode.env_id),
        None => "route:".to_string(),
    }
}
/// Dispatch to the most-specific available corner, yielding each lane's raw
/// chunk frames (a single-element Vec when not chunking). Prefer a *batched*
/// corner (one forward for the whole vector) over the per-lane loop, and a
/// *chunk* corner when the runtime pinned a horizon > 1. A horizon > 1 with no
/// chunk corner was already warned at configure and falls through to a
/// single-action corner (the runtime then re-plans every step). split_chunk
/// caps each chunk to the horizon (a receding-horizon model may over-produce).
///
/// This is the ONE corner-precedence table: the direct predict path wraps it
/// with the input-signature annotation ([`dispatch_route_corners`]) and the
/// fused grouped path calls it on a cross-route batch whose corner
/// [`bucket_fuses`] proved identical, so grouping can never change which model
/// function runs.
///
/// `episodes` is row-aligned with `inputs` on EVERY corner: the per-lane ones
/// take row `i`'s identity, the batched ones take the whole row-aligned slice
/// (a fused bucket concatenates lanes from independent episodes, so identity is
/// per row, never per call). A length disagreement is a malformed request, so
/// it fails here rather than silently mis-attributing a lane.
pub(super) fn dispatch_corners(
    predict: &Arc<dyn PredictFn>,
    inputs: Vec<Value>,
    episodes: &[EpisodeInfo],
    horizon: u32,
    num_envs: usize,
    short_chunk_warned: &ShortChunkWarned,
) -> Result<Vec<Vec<Value>>> {
    let native_chunk = predict.native_chunk();
    if episodes.len() != num_envs {
        return Err(Error::model(format!(
            "predict carries {} episode rows for {num_envs} lanes; episode identity \
             is row-aligned with the observation",
            episodes.len()
        )));
    }
    if horizon > 1 && predict.has_chunk_batch() {
        let chunks = predict.predict_chunk_batch(inputs, horizon, episodes)?;
        if chunks.len() != num_envs {
            return Err(Error::model(format!(
                "predict_chunk_batch returned {} chunks for {num_envs} lanes",
                chunks.len()
            )));
        }
        chunks
            .into_iter()
            .enumerate()
            .map(|(index, chunk)| {
                take_prefix(
                    split_chunk(chunk).map_err(Error::from_apply)?,
                    horizon,
                    native_chunk,
                    short_chunk_warned,
                    &warn_key(episodes.get(index)),
                )
            })
            .collect::<Result<Vec<_>>>()
    } else if horizon > 1 && predict.has_chunk() {
        inputs
            .into_iter()
            .enumerate()
            .map(|(index, input)| -> Result<Vec<Value>> {
                let chunk = predict
                    .predict_chunk(input, horizon, episodes.get(index))?
                    .ok_or_else(|| {
                        Error::model(
                            "model reports a chunk corner (has_chunk) but predict_chunk \
                             returned None",
                        )
                    })?;
                take_prefix(
                    split_chunk(chunk).map_err(Error::from_apply)?,
                    horizon,
                    native_chunk,
                    short_chunk_warned,
                    &warn_key(episodes.get(index)),
                )
            })
            .collect::<Result<Vec<_>>>()
    } else if predict.has_batch() {
        let actions = predict.predict_batch(inputs, episodes)?;
        if actions.len() != num_envs {
            return Err(Error::model(format!(
                "predict_batch returned {} actions for {num_envs} lanes",
                actions.len()
            )));
        }
        Ok(actions.into_iter().map(|action| vec![action]).collect())
    } else {
        inputs
            .into_iter()
            .enumerate()
            .map(|(index, input)| -> Result<Vec<Value>> {
                Ok(vec![predict.predict(input, episodes.get(index))?])
            })
            .collect::<Result<Vec<_>>>()
    }
}

/// [`dispatch_corners`] with a predict failure annotated with the
/// assembled-input signature.
pub(super) fn dispatch_route_corners(
    predict: &Arc<dyn PredictFn>,
    inputs: Vec<Value>,
    episodes: &[EpisodeInfo],
    horizon: u32,
    num_envs: usize,
    short_chunk_warned: &ShortChunkWarned,
) -> Result<Vec<Vec<Value>>> {
    let input_summary = inputs_summary(&inputs);
    dispatch_corners(
        predict,
        inputs,
        episodes,
        horizon,
        num_envs,
        short_chunk_warned,
    )
    .map_err(|err| annotate_predict_error(err, &input_summary))
}

/// Cap one lane's split chunk to the runtime's execution prefix — the ONE place
/// the engine turns a native chunk into executed frames.
///
/// A model that DECLARED its native chunk K is held to it exactly: anything but
/// K frames means the model is slicing its own chunk to the horizon (the glue
/// this contract removes), which silently short-replays instead of failing, so
/// it is a model error. An UNDECLARED model is elastic — the prefix is
/// `min(len, horizon)` — but a chunk shorter than the pinned horizon still means
/// the runtime re-plans sooner than the run was configured for, so warn once per
/// endpoint (the flag is endpoint-wide because the fused grouped path spans
/// routes).
pub(super) fn take_prefix(
    mut frames: Vec<Value>,
    horizon: u32,
    native_chunk: Option<u32>,
    short_chunk_warned: &ShortChunkWarned,
    warn_key: &str,
) -> Result<Vec<Value>> {
    match native_chunk {
        Some(native) if frames.len() != native as usize => {
            return Err(Error::invalid_argument(format!(
                "model declares native_chunk={native} but its chunk corner returned {} \
                 frames at execution_horizon={horizon}: return the WHOLE native chunk and \
                 let the runtime execute its prefix (do not slice to the horizon), or drop \
                 the declaration",
                frames.len(),
            )));
        }
        Some(_) => {}
        None => {
            if frames.len() < horizon as usize && short_chunk_warned.first(warn_key) {
                crate::warnings::warn(
                    crate::WarningCategory::RuntimeWarning,
                    short_chunk_text(frames.len(), horizon),
                )?;
            }
        }
    }
    if frames.is_empty() && horizon > 1 {
        return Err(Error::invalid_argument(
            "a chunked model (execute_horizon>1) returned an empty action chunk",
        ));
    }
    frames.truncate(horizon as usize);
    Ok(frames)
}

/// Warning text for an undeclared model's short chunk.
pub(crate) fn short_chunk_text(frames: usize, horizon: u32) -> String {
    format!(
        "the model returned {frames} actions for execution_horizon={horizon}; replaying what \
         there is and re-planning early. Declare Model.native_chunk so the horizon can be \
         validated up front."
    )
}

/// Whether a bucket of grouped lanes at `horizon` may fuse into one batched
/// corner call: true only when that batched corner is the SAME corner
/// [`dispatch_corners`] picks for a direct predict (`predict_chunk_batch` at
/// horizon > 1, else the per-lane chunk corner, else `predict_batch`, else the
/// per-lane predict loop). Fusion can only substitute a batched corner, so a
/// bucket whose direct choice is a per-lane corner serves per group instead —
/// grouping is purely an optimization and never changes which model function
/// runs or its chunk semantics.
pub(super) fn bucket_fuses(predict: &dyn PredictFn, horizon: u32) -> bool {
    if horizon > 1 {
        predict.has_chunk_batch() || (!predict.has_chunk() && predict.has_batch())
    } else {
        predict.has_batch()
    }
}
