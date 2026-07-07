//! The vectorized stateful adapter engine: a [`ModelHandler`] that owns the
//! per-lane drive, episode-keyed frame buffers, and native adapter application
//! in pure Rust, calling back into a [`PredictFn`] only for the model's predict
//! and into the custom/encoding holes only where a route declares them.
//!
//! This replaces a binding's hand-rolled predict loop: a PyO3 (or any future
//! language) binding constructs `AdaptedModelHandler::new(predict, resolver)`
//! and serves it; a pure-Rust model does the same with no host runtime.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rlmesh_adapters::v1::{
    FrameBuffers, ObsPlan, Value, apply_actions, assemble_obs, space_value_to_obs_map, split_chunk,
};

use super::handler::{ModelHandler, ModelRouteSetup, PredictFrames};
use super::predict_fn::{PredictFn, RouteConfig, RouteResolver};
use super::types::ModelObservation;
use crate::spaces::{EnvContract, SpaceKind, SpaceValue};
use crate::{Error, Result};

/// One configured route's resolved config plus its live per-episode frame-stack
/// windows.
///
/// Action-chunk replay no longer lives here: the engine emits the whole chunk
/// (frame 0 + future frames) as [`PredictFrames`] and the runtime driver owns the
/// per-step replay buffer. Only the frame-stack windows remain engine state.
struct RouteEntry {
    config: RouteConfig,
    buffers: FrameBuffers,
}

/// `route_key -> route state`. The outer lock is held only to look up/insert a
/// route; the per-route [`Mutex`] is what predict holds across its (blocking)
/// per-lane loop, so configuring one route never blocks predict on another.
type Routes = Arc<Mutex<HashMap<String, Arc<Mutex<RouteEntry>>>>>;

/// `env_id -> execution horizon` for SPEC-LESS routes (no [`RouteEntry`]): the
/// horizon is pinned at `ResolveAdapter` like a spec'd route's, but there is no
/// config to stamp it on, so it lives here for the spec-less predict branch to
/// read (chunk corner support without an adapter).
type SpecLessHorizons = Arc<Mutex<HashMap<String, u32>>>;

/// A served [`ModelHandler`] that drives the vectorized stateful adapter engine.
pub struct AdaptedModelHandler {
    predict: Arc<dyn PredictFn>,
    resolver: Option<Arc<dyn RouteResolver>>,
    routes: Routes,
    spec_less_horizons: SpecLessHorizons,
}

impl AdaptedModelHandler {
    /// Build a handler from the model's predict hole and an optional route
    /// resolver. `resolver = None` means a spec-less model: every env serves
    /// through [`PredictFn::predict_spec_less`].
    pub fn new(predict: Arc<dyn PredictFn>, resolver: Option<Arc<dyn RouteResolver>>) -> Self {
        Self {
            predict,
            resolver,
            routes: Arc::new(Mutex::new(HashMap::new())),
            spec_less_horizons: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Look up the per-env entry (cloning the `Arc`), if the env is spec'd.
    fn entry(&self, env_id: &str) -> Option<Arc<Mutex<RouteEntry>>> {
        self.routes
            .lock()
            .expect("routes map poisoned")
            .get(env_id)
            .cloned()
    }

    /// The horizon pinned for a spec-less route (1 = no chunking / never pinned).
    fn spec_less_horizon(&self, env_id: &str) -> u32 {
        self.spec_less_horizons
            .lock()
            .expect("spec-less horizons poisoned")
            .get(env_id)
            .copied()
            .unwrap_or(1)
    }
}

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
fn inputs_summary(inputs: &[Value]) -> String {
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
fn annotate_predict_error(err: Error, summary: &str) -> Error {
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
/// per-episode state (frame windows) by it. The grpc wire layer already
/// enforces `num_envs == slots.len()` with non-empty ids, but the engine must
/// not silently fall back to a shared "" buffer (which would cross-contaminate
/// lanes) if any other producer violates that.
fn assemble_route_inputs(
    entry: &mut RouteEntry,
    observation: &ModelObservation,
) -> Result<Vec<Value>> {
    let episode_ids = &observation.route.episode_ids;
    let num_envs = observation.num_envs;

    // The wire contract requires every predict request to carry a decodable
    // observation; validate the structure up front (cheap) so a malformed request
    // errors here. Every call re-plans (the runtime owns replay), so the obs is
    // always decoded below.
    observation.ensure_decodable()?;

    let RouteEntry { config, buffers } = entry;
    let referenced = obs_keys(config);
    let customs: &dyn rlmesh_adapters::v1::CustomTransform = config.customs.as_ref();
    let encodings: &dyn rlmesh_adapters::v1::EncodingTransform = config.encodings.as_ref();

    let decoded = observation.decoded_lanes()?;

    let mut inputs: Vec<Value> = Vec::with_capacity(num_envs);
    for (index, lane) in decoded.iter().enumerate() {
        let episode_id = episode_ids
            .get(index)
            .map(String::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                Error::model(format!(
                    "predict request lane {index} has no episode_id (num_envs={num_envs}, \
                     episode_ids={}); every lane must carry a non-empty episode_id",
                    episode_ids.len()
                ))
            })?;
        let raw = space_value_to_obs_map(lane, &config.observation_space, &referenced)?;
        inputs.push(assemble_obs(
            &config.adapter,
            &raw,
            episode_id,
            buffers,
            customs,
            encodings,
        )?);
    }
    Ok(inputs)
}

/// Dispatch to the most-specific available corner, yielding each lane's raw
/// chunk frames (a single-element Vec when not chunking). Prefer a *batched*
/// corner (one forward for the whole vector) over the per-lane loop, and a
/// *chunk* corner when the runtime pinned a horizon > 1. A horizon > 1 with no
/// chunk corner was already warned at configure and falls through to a
/// single-action corner (the runtime then re-plans every step). split_chunk
/// caps each chunk to the horizon (a receding-horizon model may over-produce).
/// A predict failure is annotated with the assembled-input signature.
fn dispatch_route_corners(
    predict: &Arc<dyn PredictFn>,
    inputs: Vec<Value>,
    horizon: u32,
    num_envs: usize,
) -> Result<Vec<Vec<Value>>> {
    let input_summary = inputs_summary(&inputs);
    let dispatch = || -> Result<Vec<Vec<Value>>> {
        if horizon > 1 && predict.has_chunk_batch() {
            let chunks = predict.predict_chunk_batch(inputs, horizon)?;
            if chunks.len() != num_envs {
                return Err(Error::model(format!(
                    "predict_chunk_batch returned {} chunks for {num_envs} lanes",
                    chunks.len()
                )));
            }
            chunks
                .into_iter()
                .map(|chunk| -> Result<Vec<Value>> {
                    Ok(split_chunk(chunk)?
                        .into_iter()
                        .take(horizon as usize)
                        .collect())
                })
                .collect::<Result<Vec<_>>>()
        } else if horizon > 1 && predict.has_chunk() {
            inputs
                .into_iter()
                .map(|input| -> Result<Vec<Value>> {
                    let chunk = predict.predict_chunk(input, horizon)?.ok_or_else(|| {
                        Error::model(
                            "model reports a chunk corner (has_chunk) but predict_chunk \
                             returned None",
                        )
                    })?;
                    Ok(split_chunk(chunk)?
                        .into_iter()
                        .take(horizon as usize)
                        .collect())
                })
                .collect::<Result<Vec<_>>>()
        } else if predict.has_batch() {
            let actions = predict.predict_batch(inputs)?;
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
                .map(|input| -> Result<Vec<Value>> { Ok(vec![predict.predict(input)?]) })
                .collect::<Result<Vec<_>>>()
        }
    };
    dispatch().map_err(|err| annotate_predict_error(err, &input_summary))
}

/// Apply the per-step adapter transform to each frame, peel frame 0 (this
/// step's action) from the future frames (which the runtime replays), and
/// transpose the per-lane future frames into per-step batched frames
/// (`replay[step][lane]`), to the shortest lane (uniform for a homogeneous
/// fleet; a short lane caps the batch — receding horizon).
fn finish_route_frames(
    config: &RouteConfig,
    lane_raw_steps: Vec<Vec<Value>>,
    num_envs: usize,
) -> Result<PredictFrames> {
    let encodings: &dyn rlmesh_adapters::v1::EncodingTransform = config.encodings.as_ref();
    let mut frame0 = Vec::with_capacity(num_envs);
    let mut lane_replays: Vec<Vec<SpaceValue>> = Vec::with_capacity(num_envs);
    for raw_steps in lane_raw_steps {
        let mut applied = raw_steps
            .into_iter()
            .map(|raw_action| {
                apply_actions(&config.adapter, raw_action, &config.action_space, encodings)
            })
            .collect::<std::result::Result<Vec<SpaceValue>, _>>()?
            .into_iter();
        let first = applied
            .next()
            .ok_or_else(|| Error::model("a chunked model returned an empty action chunk"))?;
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

/// The spec'd per-lane loop (CPU + the model's predict callback), run on a
/// blocking worker thread. Holds the per-route entry lock across the lanes so
/// the frame buffers mutate in place.
///
/// Emits each lane's action chunk as [`PredictFrames`]: frame 0 per lane plus the
/// future-step frames the runtime driver replays. With execution horizon 1 every
/// lane yields a single frame and `replay` is empty — the unchanged single-action
/// path. The engine no longer replays internally; it returns the executed prefix of
/// the model's native chunk and re-plans on every call (the runtime decides when).
fn predict_route(
    entry: &Arc<Mutex<RouteEntry>>,
    predict: &Arc<dyn PredictFn>,
    observation: ModelObservation,
) -> Result<PredictFrames> {
    let num_envs = observation.num_envs;
    let mut guard = entry.lock().expect("route entry poisoned");
    let inputs = assemble_route_inputs(&mut guard, &observation)?;
    // Runtime-chosen execution horizon (pinned on ResolveAdapter), not the model spec.
    let horizon = guard.config.execution_horizon;
    let lane_raw_steps = dispatch_route_corners(predict, inputs, horizon, num_envs)?;
    finish_route_frames(&guard.config, lane_raw_steps, num_envs)
}

/// One fused corner call for a same-horizon bucket of `total` lanes, yielding
/// each lane's ordered raw frames. Horizon > 1 prefers the batched chunk corner
/// and falls back to the plain batched corner (single-step, as warned at
/// configure); horizon 1 prefers the plain batched corner and falls back to a
/// 1-frame chunk batch when only the chunk corner exists. The caller's fusion
/// gate guarantees at least one batched corner is present. A predict failure is
/// annotated with the assembled-input signature.
fn fused_bucket_frames(
    predict: &Arc<dyn PredictFn>,
    flat: Vec<Value>,
    horizon: u32,
    total: usize,
) -> Result<Vec<Vec<Value>>> {
    let input_summary = inputs_summary(&flat);
    let dispatch = || -> Result<Vec<Vec<Value>>> {
        if horizon > 1 && predict.has_chunk_batch() {
            let chunks = predict.predict_chunk_batch(flat, horizon)?;
            if chunks.len() != total {
                return Err(Error::model(format!(
                    "predict_chunk_batch returned {} chunks for {total} grouped lanes",
                    chunks.len()
                )));
            }
            chunks
                .into_iter()
                .map(|chunk| -> Result<Vec<Value>> {
                    Ok(split_chunk(chunk)?
                        .into_iter()
                        .take(horizon as usize)
                        .collect())
                })
                .collect::<Result<Vec<_>>>()
        } else if predict.has_batch() {
            let actions = predict.predict_batch(flat)?;
            if actions.len() != total {
                return Err(Error::model(format!(
                    "predict_batch returned {} actions for {total} grouped lanes",
                    actions.len()
                )));
            }
            Ok(actions.into_iter().map(|action| vec![action]).collect())
        } else {
            let chunks = predict.predict_chunk_batch(flat, 1)?;
            if chunks.len() != total {
                return Err(Error::model(format!(
                    "predict_chunk_batch returned {} chunks for {total} grouped lanes",
                    chunks.len()
                )));
            }
            chunks
                .into_iter()
                .map(|chunk| -> Result<Vec<Value>> {
                    Ok(split_chunk(chunk)?.into_iter().take(1).collect())
                })
                .collect::<Result<Vec<_>>>()
        }
    };
    dispatch().map_err(|err| annotate_predict_error(err, &input_summary))
}

/// The fused grouped predict (the batched forward across routes), run on a
/// blocking worker thread. Each group prepares under its own route entry lock
/// (frame-stack buffers mutate there); prepared groups are bucketed by their
/// pinned execution horizon (a fused corner call takes one horizon — in
/// practice one runner pins one value, so this is a single bucket), dispatched
/// through ONE batched corner call per bucket with lanes concatenated in group
/// order, split back by lane count, and finished per group with replay frames
/// intact. Lanes from different routes are independent by the batched-corner
/// contract, so a cross-route batch is semantically identical to a vectorized
/// route's lanes. Per-group errors stay per-group; a fused corner failure
/// reports to every group in its bucket.
fn predict_grouped_fused(
    entries: Vec<Option<Arc<Mutex<RouteEntry>>>>,
    spec_less_horizons: Vec<u32>,
    observations: Vec<ModelObservation>,
    predict: &Arc<dyn PredictFn>,
) -> Vec<Result<PredictFrames>> {
    enum Slot<'a> {
        Done(Result<PredictFrames>),
        Ready {
            guard: std::sync::MutexGuard<'a, RouteEntry>,
            inputs: Option<Vec<Value>>,
            num_envs: usize,
            frames: Option<Result<Vec<Vec<Value>>>>,
        },
    }

    let mut slots: Vec<Slot> = Vec::with_capacity(observations.len());
    for ((entry, horizon), observation) in entries.iter().zip(spec_less_horizons).zip(observations)
    {
        match entry {
            // A spec-less route has no adapter or chunk semantics of its own;
            // serve it through the preserved raw path (chunked through the
            // model's chunk corner when a horizon was pinned), matching
            // `predict_chunked`.
            None => slots.push(Slot::Done(
                predict.predict_spec_less_chunked(observation, horizon),
            )),
            Some(entry) => {
                let mut guard = entry.lock().expect("route entry poisoned");
                let num_envs = observation.num_envs;
                match assemble_route_inputs(&mut guard, &observation) {
                    Ok(inputs) => slots.push(Slot::Ready {
                        guard,
                        inputs: Some(inputs),
                        num_envs,
                        frames: None,
                    }),
                    Err(error) => slots.push(Slot::Done(Err(error))),
                }
            }
        }
    }

    let mut buckets: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (index, slot) in slots.iter().enumerate() {
        if let Slot::Ready { guard, .. } = slot {
            buckets
                .entry(guard.config.execution_horizon.max(1))
                .or_default()
                .push(index);
        }
    }

    for (horizon, indices) in buckets {
        let mut flat: Vec<Value> = Vec::new();
        let mut lane_counts: Vec<usize> = Vec::with_capacity(indices.len());
        for &index in &indices {
            let Slot::Ready { inputs, .. } = &mut slots[index] else {
                unreachable!("bucketed slot is Ready");
            };
            let inputs = inputs.take().expect("bucketed group inputs taken once");
            lane_counts.push(inputs.len());
            flat.extend(inputs);
        }
        let total: usize = lane_counts.iter().sum();
        match fused_bucket_frames(predict, flat, horizon, total) {
            Ok(mut all_frames) => {
                for (&index, count) in indices.iter().zip(lane_counts) {
                    let rest = all_frames.split_off(count);
                    let group_frames = std::mem::replace(&mut all_frames, rest);
                    let Slot::Ready { frames, .. } = &mut slots[index] else {
                        unreachable!("bucketed slot is Ready");
                    };
                    *frames = Some(Ok(group_frames));
                }
            }
            Err(error) => {
                let message = error.to_string();
                for &index in &indices {
                    let Slot::Ready { frames, .. } = &mut slots[index] else {
                        unreachable!("bucketed slot is Ready");
                    };
                    *frames = Some(Err(Error::model(message.clone())));
                }
            }
        }
    }

    slots
        .into_iter()
        .map(|slot| match slot {
            Slot::Done(result) => result,
            Slot::Ready {
                guard,
                num_envs,
                frames,
                ..
            } => match frames {
                Some(Ok(lane_raw_steps)) => {
                    finish_route_frames(&guard.config, lane_raw_steps, num_envs)
                }
                Some(Err(error)) => Err(error),
                None => Err(Error::Internal(
                    "grouped predict left a prepared group undispatched".to_string(),
                )),
            },
        })
        .collect()
}

/// Seeds for the two distinct probe observations (A must differ from B, or an
/// intervening B cannot perturb accumulating state).
const PROBE_SEED_A: u64 = 0x5EED_000A;
const PROBE_SEED_B: u64 = 0x5EED_000B;
/// Replay drift must exceed the model's own back-to-back floor by this factor
/// (plus an absolute floor) to count as state leakage rather than nondeterminism.
const PROBE_TOLERANCE: f64 = 8.0;
const PROBE_ATOL: f64 = 1e-6;

/// Detect a model that mutates internal state across `predict` calls — which
/// would corrupt a shared-object vectorized loop — by replaying a fixed input
/// around an intervening different input and comparing the output drift to the
/// model's own back-to-back nondeterminism floor (so GPU/dropout noise is not
/// mistaken for state). Runs once at configure when `num_envs > 1`; returns a
/// model error (failing route configuration) on detection. Calls `predict`, so
/// it must run on a blocking worker thread.
fn probe_model_internal_state(predict: &Arc<dyn PredictFn>, config: &RouteConfig) -> Result<()> {
    let space = &config.observation_space;
    let referenced = obs_keys(config);
    // Sample the env obs space, then run it through the engine so `predict` sees
    // the production-shaped (frame-stacked / adapted) input, not a raw env obs.
    let assemble = |seed: u64, episode: &str| -> Result<rlmesh_adapters::v1::Value> {
        let sampled = rlmesh_spaces::sample_seeded(space, seed)
            .map_err(|err| Error::Internal(format!("probe sample failed: {err}")))?;
        let mut scratch = FrameBuffers::new();
        let raw = space_value_to_obs_map(&sampled, space, &referenced)?;
        Ok(assemble_obs(
            &config.adapter,
            &raw,
            episode,
            &mut scratch,
            config.customs.as_ref(),
            config.encodings.as_ref(),
        )?)
    };
    let a = assemble(PROBE_SEED_A, "probe-a")?;
    let b = assemble(PROBE_SEED_B, "probe-b")?;
    // The model's own back-to-back nondeterminism floor (GPU/dropout noise).
    let floor = rlmesh_adapters::v1::value_max_abs_diff(
        &predict.predict(a.clone())?,
        &predict.predict(a.clone())?,
    )
    .unwrap_or(f64::INFINITY);
    // Replay A around an intervening, distinct B: drift beyond the floor is state.
    let before = predict.predict(a.clone())?;
    let _ = predict.predict(b)?;
    let after = predict.predict(a)?;
    let delta = rlmesh_adapters::v1::value_max_abs_diff(&before, &after).unwrap_or(0.0);
    if delta > (floor * PROBE_TOLERANCE).max(PROBE_ATOL) {
        return Err(Error::model(
            "this model carries internal state across predict() calls, so it cannot be \
             served against a vectorized route (num_envs>1): one shared model instance \
             across lanes would interleave their state. Serve it against num_envs=1, or \
             make predict() pure (move per-step state into the adapter).",
        ));
    }
    Ok(())
}

#[async_trait]
impl ModelHandler for AdaptedModelHandler {
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<SpaceValue>> {
        Ok(self.predict_chunked(observation).await?.actions)
    }

    async fn predict_chunked(&mut self, observation: ModelObservation) -> Result<PredictFrames> {
        let entry = self.entry(&observation.route.env_id);
        let spec_less_horizon = if entry.is_none() {
            self.spec_less_horizon(&observation.route.env_id)
        } else {
            1
        };
        let predict = Arc::clone(&self.predict);
        // Decode + frame-stack + the model's predict are CPU/host work; run them
        // off the async worker so concurrent (pipelined) requests on other routes
        // are not stalled. A spec'd route runs the per-lane engine loop (emitting
        // chunk frames); a spec-less route takes the preserved batched raw path
        // (chunked through the model's chunk corner when a horizon was pinned).
        tokio::task::spawn_blocking(move || match entry {
            Some(entry) => predict_route(&entry, &predict, observation),
            None => predict.predict_spec_less_chunked(observation, spec_less_horizon),
        })
        .await
        .map_err(|err| Error::Internal(format!("predict task panicked: {err}")))?
    }

    async fn predict_grouped(
        &mut self,
        observations: Vec<ModelObservation>,
    ) -> Vec<Result<PredictFrames>> {
        // Fusion needs a batched corner and the model's permission
        // (`allow_fusion`); without both, keep the chunk-preserving sequential
        // default — correct per group, just unfused.
        let fusable = self.predict.allow_fusion()
            && (self.predict.has_chunk_batch() || self.predict.has_batch());
        if observations.len() <= 1 || !fusable {
            let mut results = Vec::with_capacity(observations.len());
            for observation in observations {
                results.push(self.predict_chunked(observation).await);
            }
            return results;
        }
        let entries: Vec<Option<Arc<Mutex<RouteEntry>>>> = observations
            .iter()
            .map(|observation| self.entry(&observation.route.env_id))
            .collect();
        let spec_less_horizons: Vec<u32> = observations
            .iter()
            .zip(&entries)
            .map(|(observation, entry)| match entry {
                Some(_) => 1,
                None => self.spec_less_horizon(&observation.route.env_id),
            })
            .collect();
        let predict = Arc::clone(&self.predict);
        let group_count = observations.len();
        tokio::task::spawn_blocking(move || {
            predict_grouped_fused(entries, spec_less_horizons, observations, &predict)
        })
        .await
        .unwrap_or_else(|err| {
            let message = format!("grouped predict task panicked: {err}");
            (0..group_count)
                .map(|_| Err(Error::Internal(message.clone())))
                .collect()
        })
    }

    fn route_setup(&self) -> Option<Arc<dyn ModelRouteSetup>> {
        let resolver = self.resolver.clone()?;
        Some(Arc::new(AdaptedRouteSetup {
            resolver,
            routes: Arc::clone(&self.routes),
            predict: Arc::clone(&self.predict),
            spec_less_horizons: Arc::clone(&self.spec_less_horizons),
        }))
    }

    async fn reset_adapter(&mut self, env_id: &str, episode_ids: Vec<String>) -> Result<()> {
        // Explicit GC (R2): evict the ended episodes' frame buffers on this env's
        // adapter. Buffers are lazy-seeded on each episode's first predict (via
        // `assemble_obs`), so there is no seed step and no position-diffing. Empty
        // `episode_ids` evicts ALL of this env's episode state.
        if let Some(entry) = self.entry(env_id) {
            let mut guard = entry.lock().expect("route entry poisoned");
            if episode_ids.is_empty() {
                guard.buffers.clear();
            } else {
                for episode_id in &episode_ids {
                    guard.buffers.evict(episode_id);
                }
            }
        }
        // Surface the episode-end edge to the model's own hook (one call per
        // ended episode), e.g. to reset a single-env model's recurrent state. An
        // empty `episode_ids` is an evict-ALL/teardown, not an episode end, so it
        // fires nothing here — model shutdown is `on_close`'s job.
        let count = episode_ids.len();
        if count == 0 {
            return Ok(());
        }
        let predict = Arc::clone(&self.predict);
        tokio::task::spawn_blocking(move || {
            for _ in 0..count {
                predict.on_episode_end()?;
            }
            Ok(())
        })
        .await
        .map_err(|err| Error::Internal(format!("on_episode_end task panicked: {err}")))?
    }

    async fn on_close(&mut self) -> Result<()> {
        // Drop every route's per-episode state as the authoritative shutdown sweep.
        for entry in self.routes.lock().expect("routes map poisoned").values() {
            let mut guard = entry.lock().expect("route entry poisoned");
            guard.buffers.clear();
        }
        let predict = Arc::clone(&self.predict);
        tokio::task::spawn_blocking(move || predict.on_close())
            .await
            .map_err(|err| Error::Internal(format!("on_close task panicked: {err}")))?
    }
}

/// The [`ModelRouteSetup`] the engine returns: resolves a route's config off the
/// predict lock and caches it for predict to read. A `None` resolution is a
/// spec-less route, left absent so predict takes the spec-less branch.
struct AdaptedRouteSetup {
    resolver: Arc<dyn RouteResolver>,
    routes: Routes,
    predict: Arc<dyn PredictFn>,
    spec_less_horizons: SpecLessHorizons,
}

#[async_trait]
impl ModelRouteSetup for AdaptedRouteSetup {
    async fn resolve_adapter(
        &self,
        env_id: &str,
        env_contract: &EnvContract,
        execution_horizon: u32,
    ) -> Result<()> {
        let Some(mut config) = self.resolver.resolve(env_id, env_contract).await? else {
            let horizon = execution_horizon.max(1);
            if horizon > 1 && !self.predict.has_chunk() {
                tracing::warn!(
                    env_id = %env_id,
                    execution_horizon = horizon,
                    "runtime pinned execution_horizon > 1 but the model defines no chunk \
                     corner (predict_chunk); chunking is inactive — the model re-plans \
                     every step",
                );
            }
            let mut horizons = self
                .spec_less_horizons
                .lock()
                .expect("spec-less horizons poisoned");
            if horizon > 1 {
                horizons.insert(env_id.to_string(), horizon);
            } else {
                horizons.remove(env_id);
            }
            return Ok(());
        };
        // Surface the adapter's advisories once at configure. These are the
        // tolerant reader's only operator signal that something degraded: a
        // dropped unknown-kind modality (an old core ignoring data a newer env
        // declares under a kind it cannot read), a zero-filled absent camera, an
        // aspect crop/letterbox. The route runs regardless, so without this the
        // signal stays buried on the adapter handle and the degradation is silent
        // in practice.
        for note in config.adapter.advisories() {
            tracing::warn!(
                env_id = %env_id,
                severity = note.severity.as_str(),
                "adapter advisory: {note}"
            );
        }
        // Stamp the runtime-chosen execution horizon onto the resolved config (1 = no
        // chunking). Warn once here when the runtime asks for chunking but the model
        // has no chunk corner: the route still runs, re-planning every step.
        config.execution_horizon = execution_horizon.max(1);
        if config.execution_horizon > 1 && !self.predict.has_chunk() {
            tracing::warn!(
                env_id = %env_id,
                execution_horizon = config.execution_horizon,
                "runtime pinned execution_horizon > 1 but the model defines no chunk corner \
                 (predict_chunk); chunking is inactive — the model re-plans every step",
            );
        }
        // Frame-stacking + chunking would stack only decision-point frames (the
        // engine assembles observations once per chunk, not every step), not the
        // consecutive history a stacked policy expects. Reject the combination
        // rather than feed temporally-aliased frames. This was a resolve-time check,
        // relocated here now that the horizon is a runtime decision.
        if config.execution_horizon > 1
            && let Some((key, depth)) = config.adapter.stacks().into_iter().next()
        {
            return Err(Error::model(format!(
                "frame-stacking (input '{key}' stack={depth}) cannot be combined with action \
                 chunking (execution_horizon={}): the engine assembles observations once per chunk, \
                 so the frame window would hold only decision-point frames. Use stack=1 or \
                 execution_horizon=1.",
                config.execution_horizon,
            )));
        }
        // A vectorized route runs ONE shared model object across lanes, so a model
        // that mutates internal state across predict() calls would interleave them.
        // Probe once at configure and fail num_envs>1 for such a model. The
        // adapter's own frame-stack state is engine-managed and lane-correct, so it
        // is NOT what this gates.
        let config = if env_contract.num_envs > 1 {
            let predict = Arc::clone(&self.predict);
            tokio::task::spawn_blocking(move || -> Result<RouteConfig> {
                probe_model_internal_state(&predict, &config)?;
                Ok(config)
            })
            .await
            .map_err(|err| Error::Internal(format!("probe task panicked: {err}")))??
        } else {
            config
        };
        let entry = Arc::new(Mutex::new(RouteEntry {
            config,
            buffers: FrameBuffers::new(),
        }));
        self.routes
            .lock()
            .expect("routes map poisoned")
            .insert(env_id.to_string(), entry);
        Ok(())
    }

    async fn release_adapter(&self, env_id: &str) -> Result<()> {
        self.routes
            .lock()
            .expect("routes map poisoned")
            .remove(env_id);
        self.spec_less_horizons
            .lock()
            .expect("spec-less horizons poisoned")
            .remove(env_id);
        Ok(())
    }
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

#[cfg(test)]
mod fused_predict_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::model::types::ModelRouteContext;

    /// Counts corner calls and emits `native_chunk`-frame `Value::List` chunks
    /// from the batched chunk corner (or single Numbers from the plain batched
    /// corner), so the fused dispatch's corner selection, lane math, and chunk
    /// capping are observable without an adapter resolver. `short` drops one
    /// output to exercise the lane-count guard.
    struct CountingPredict {
        batch: bool,
        chunk_batch: bool,
        native_chunk: usize,
        short: bool,
        batch_calls: AtomicUsize,
        chunk_batch_calls: AtomicUsize,
    }

    impl CountingPredict {
        fn new(batch: bool, chunk_batch: bool, native_chunk: usize, short: bool) -> Arc<Self> {
            Arc::new(Self {
                batch,
                chunk_batch,
                native_chunk,
                short,
                batch_calls: AtomicUsize::new(0),
                chunk_batch_calls: AtomicUsize::new(0),
            })
        }
    }

    impl PredictFn for CountingPredict {
        fn predict(&self, _model_input: Value) -> Result<Value> {
            Ok(Value::Number(0.0))
        }

        fn predict_spec_less(&self, observation: ModelObservation) -> Result<Vec<SpaceValue>> {
            Ok((0..observation.num_envs)
                .map(|_| SpaceValue::Discrete(0))
                .collect())
        }

        fn has_batch(&self) -> bool {
            self.batch
        }

        fn predict_batch(&self, inputs: Vec<Value>) -> Result<Vec<Value>> {
            self.batch_calls.fetch_add(1, Ordering::SeqCst);
            let keep = inputs.len() - usize::from(self.short);
            Ok((0..keep).map(|_| Value::Number(1.0)).collect())
        }

        fn has_chunk_batch(&self) -> bool {
            self.chunk_batch
        }

        fn predict_chunk_batch(
            &self,
            inputs: Vec<Value>,
            _execution_horizon: u32,
        ) -> Result<Vec<Value>> {
            self.chunk_batch_calls.fetch_add(1, Ordering::SeqCst);
            let keep = inputs.len() - usize::from(self.short);
            Ok((0..keep)
                .map(|_| {
                    Value::List(
                        (0..self.native_chunk)
                            .map(|frame| Value::Number(frame as f64))
                            .collect(),
                    )
                })
                .collect())
        }

        fn allow_fusion(&self) -> bool {
            true
        }
    }

    fn lanes(count: usize) -> Vec<Value> {
        (0..count).map(|lane| Value::Number(lane as f64)).collect()
    }

    #[test]
    fn fused_bucket_prefers_chunk_batch_and_caps_to_horizon() {
        let counting = CountingPredict::new(true, true, 10, false);
        let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

        let frames = fused_bucket_frames(&predict, lanes(5), 4, 5).expect("fused frames");

        assert_eq!(counting.chunk_batch_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 0);
        assert_eq!(frames.len(), 5);
        assert!(
            frames.iter().all(|lane| lane.len() == 4),
            "each lane keeps the horizon prefix of its native chunk"
        );
    }

    #[test]
    fn fused_bucket_horizon_one_uses_plain_batch() {
        let counting = CountingPredict::new(true, true, 10, false);
        let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

        let frames = fused_bucket_frames(&predict, lanes(3), 1, 3).expect("fused frames");

        assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counting.chunk_batch_calls.load(Ordering::SeqCst), 0);
        assert!(frames.iter().all(|lane| lane.len() == 1));
    }

    #[test]
    fn fused_bucket_horizon_one_falls_back_to_chunk_batch() {
        let counting = CountingPredict::new(false, true, 10, false);
        let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

        let frames = fused_bucket_frames(&predict, lanes(2), 1, 2).expect("fused frames");

        assert_eq!(counting.chunk_batch_calls.load(Ordering::SeqCst), 1);
        assert!(
            frames.iter().all(|lane| lane.len() == 1),
            "a 1-frame prefix of the native chunk stands in for predict_batch"
        );
    }

    #[test]
    fn fused_bucket_reports_lane_count_mismatch() {
        let counting = CountingPredict::new(true, true, 10, true);
        let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

        let error = fused_bucket_frames(&predict, lanes(4), 4, 4).expect_err("short output fails");

        assert!(
            error.to_string().contains("grouped lanes"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn grouped_predict_serves_spec_less_groups_per_group() {
        let counting = CountingPredict::new(true, true, 10, false);
        let mut handler =
            AdaptedModelHandler::new(Arc::clone(&counting) as Arc<dyn PredictFn>, None);
        let observations = (0..3)
            .map(|index| ModelObservation {
                observation: None,
                route: ModelRouteContext {
                    env_id: format!("env-{index}"),
                    episode_ids: vec![format!("ep-{index}")],
                    ..Default::default()
                },
                num_envs: 1,
                env_contract: None,
            })
            .collect();

        let results = handler.predict_grouped(observations).await;

        assert_eq!(results.len(), 3, "one result per group, in order");
        for result in results {
            let frames = result.expect("spec-less group serves");
            assert_eq!(frames.actions.len(), 1);
            assert!(frames.replay.is_empty());
        }
        // Spec-less routes bypass the engine corners entirely.
        assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 0);
        assert_eq!(counting.chunk_batch_calls.load(Ordering::SeqCst), 0);
    }
}
