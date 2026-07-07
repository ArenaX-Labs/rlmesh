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
    config: Arc<RouteConfig>,
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
///
/// This is the ONE corner-precedence table: the direct predict path wraps it
/// with the input-signature annotation ([`dispatch_route_corners`]) and the
/// fused grouped path calls it on a cross-route batch whose corner
/// [`bucket_fuses`] proved identical, so grouping can never change which model
/// function runs.
fn dispatch_corners(
    predict: &Arc<dyn PredictFn>,
    inputs: Vec<Value>,
    horizon: u32,
    num_envs: usize,
) -> Result<Vec<Vec<Value>>> {
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
}

/// [`dispatch_corners`] with a predict failure annotated with the
/// assembled-input signature.
fn dispatch_route_corners(
    predict: &Arc<dyn PredictFn>,
    inputs: Vec<Value>,
    horizon: u32,
    num_envs: usize,
) -> Result<Vec<Vec<Value>>> {
    let input_summary = inputs_summary(&inputs);
    dispatch_corners(predict, inputs, horizon, num_envs)
        .map_err(|err| annotate_predict_error(err, &input_summary))
}

/// Whether a bucket of grouped lanes at `horizon` may fuse into one batched
/// corner call: true only when that batched corner is the SAME corner
/// [`dispatch_corners`] picks for a direct predict (`predict_chunk_batch` at
/// horizon > 1, else the per-lane chunk corner, else `predict_batch`, else the
/// per-lane predict loop). Fusion can only substitute a batched corner, so a
/// bucket whose direct choice is a per-lane corner serves per group instead —
/// grouping is purely an optimization and never changes which model function
/// runs or its chunk semantics.
fn bucket_fuses(predict: &dyn PredictFn, horizon: u32) -> bool {
    if horizon > 1 {
        predict.has_chunk_batch() || (!predict.has_chunk() && predict.has_batch())
    } else {
        predict.has_batch()
    }
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
/// blocking worker thread. Holds the per-route entry lock only across input
/// assembly (the frame buffers mutate in place there); the horizon it
/// dispatches with is the runtime-chosen execution horizon pinned on
/// `ResolveAdapter`, not the model spec.
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
    let (inputs, config) = {
        let mut guard = entry.lock().expect("route entry poisoned");
        let inputs = assemble_route_inputs(&mut guard, &observation)?;
        (inputs, Arc::clone(&guard.config))
    };
    let lane_raw_steps =
        dispatch_route_corners(predict, inputs, config.execution_horizon, num_envs)?;
    finish_route_frames(&config, lane_raw_steps, num_envs)
}

/// One grouped predict group's serving lane, classified once (on the async
/// side) from the routes map: a spec-less route carries the horizon pinned for
/// it (there is no [`RouteEntry`] to stamp it on); a spec'd route carries its
/// entry.
enum GroupLane {
    SpecLess { horizon: u32 },
    Routed { entry: Arc<Mutex<RouteEntry>> },
}

/// The fused grouped predict (the batched forward across routes), run on a
/// blocking worker thread.
///
/// A spec-less group has no adapter or chunk semantics of its own and serves
/// inline through the preserved raw path, matching `predict_chunked`. Each
/// spec'd group assembles its inputs under a short-lived per-route entry lock
/// (the frame-stack buffers mutate there); the lock drops before any model
/// call, so a route repeated within one request re-locks sequentially instead
/// of self-deadlocking, and a panic can poison at most the route being
/// assembled. Prepared groups are bucketed by their pinned execution horizon
/// (in practice one runner pins one value, so this is a single bucket). A
/// bucket fuses into ONE batched corner call — lanes concatenated in group
/// order, split back by lane count, finished per group with replay frames
/// intact — only when [`bucket_fuses`] proves that corner is the one a direct
/// predict would pick; otherwise the bucket serves per group through the same
/// dispatch as a direct predict. Lanes from different routes are independent
/// by the batched-corner contract, so a fused cross-route batch is
/// semantically identical to a vectorized route's lanes. Per-group errors stay
/// per-group; a fused corner failure is cloned to every group in its bucket
/// (variant and recoverability preserved) and annotated with that group's own
/// assembled-input signature.
fn predict_grouped_fused(
    lanes: Vec<GroupLane>,
    observations: Vec<ModelObservation>,
    predict: &Arc<dyn PredictFn>,
) -> Vec<Result<PredictFrames>> {
    struct Prepared {
        index: usize,
        inputs: Vec<Value>,
        config: Arc<RouteConfig>,
        num_envs: usize,
    }

    let mut results: Vec<Option<Result<PredictFrames>>> = Vec::with_capacity(observations.len());
    let mut prepared: Vec<Prepared> = Vec::new();
    for (index, (lane, observation)) in lanes.into_iter().zip(observations).enumerate() {
        match lane {
            GroupLane::SpecLess { horizon } => results.push(Some(
                predict.predict_spec_less_chunked(observation, horizon),
            )),
            GroupLane::Routed { entry } => {
                let num_envs = observation.num_envs;
                let assembled = {
                    let mut guard = entry.lock().expect("route entry poisoned");
                    assemble_route_inputs(&mut guard, &observation)
                        .map(|inputs| (inputs, Arc::clone(&guard.config)))
                };
                match assembled {
                    Ok((inputs, config)) => {
                        results.push(None);
                        prepared.push(Prepared {
                            index,
                            inputs,
                            config,
                            num_envs,
                        });
                    }
                    Err(error) => results.push(Some(Err(error))),
                }
            }
        }
    }

    let mut buckets: BTreeMap<u32, Vec<Prepared>> = BTreeMap::new();
    for group in prepared {
        buckets
            .entry(group.config.execution_horizon.max(1))
            .or_default()
            .push(group);
    }

    for (horizon, groups) in buckets {
        if bucket_fuses(predict.as_ref(), horizon) {
            struct FusedGroup {
                index: usize,
                lane_count: usize,
                summary: String,
                config: Arc<RouteConfig>,
                num_envs: usize,
            }
            let mut flat: Vec<Value> = Vec::new();
            let mut fused: Vec<FusedGroup> = Vec::with_capacity(groups.len());
            for group in groups {
                fused.push(FusedGroup {
                    index: group.index,
                    lane_count: group.inputs.len(),
                    summary: inputs_summary(&group.inputs),
                    config: group.config,
                    num_envs: group.num_envs,
                });
                flat.extend(group.inputs);
            }
            let total = flat.len();
            match dispatch_corners(predict, flat, horizon, total) {
                Ok(all_frames) => {
                    let mut frames = all_frames.into_iter();
                    for group in fused {
                        let group_frames: Vec<Vec<Value>> =
                            frames.by_ref().take(group.lane_count).collect();
                        results[group.index] = Some(finish_route_frames(
                            &group.config,
                            group_frames,
                            group.num_envs,
                        ));
                    }
                }
                Err(error) => {
                    for group in fused {
                        results[group.index] =
                            Some(Err(annotate_predict_error(error.clone(), &group.summary)));
                    }
                }
            }
        } else {
            for group in groups {
                results[group.index] = Some(
                    dispatch_route_corners(predict, group.inputs, horizon, group.num_envs)
                        .and_then(|raw| finish_route_frames(&group.config, raw, group.num_envs)),
                );
            }
        }
    }

    results
        .into_iter()
        .map(|result| {
            result.unwrap_or_else(|| {
                Err(Error::Internal(
                    "grouped predict left a prepared group unserved".to_string(),
                ))
            })
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

    /// Fusion needs a batched corner and the model's permission
    /// (`allow_fusion`); without both, keep the chunk-preserving sequential
    /// default — correct per group, just unfused.
    async fn predict_grouped(
        &mut self,
        observations: Vec<ModelObservation>,
    ) -> Vec<Result<PredictFrames>> {
        let fusable = self.predict.allow_fusion()
            && (self.predict.has_chunk_batch() || self.predict.has_batch());
        if observations.len() <= 1 || !fusable {
            let mut results = Vec::with_capacity(observations.len());
            for observation in observations {
                results.push(self.predict_chunked(observation).await);
            }
            return results;
        }
        let lanes: Vec<GroupLane> = observations
            .iter()
            .map(|observation| match self.entry(&observation.route.env_id) {
                Some(entry) => GroupLane::Routed { entry },
                None => GroupLane::SpecLess {
                    horizon: self.spec_less_horizon(&observation.route.env_id),
                },
            })
            .collect();
        let predict = Arc::clone(&self.predict);
        let group_count = observations.len();
        tokio::task::spawn_blocking(move || predict_grouped_fused(lanes, observations, &predict))
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
            config: Arc::new(config),
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
    /// from the chunk corners (or single Numbers from the plain corners), so
    /// corner selection, lane math, and chunk capping are observable without an
    /// adapter resolver. `short` drops one batched output to exercise the
    /// lane-count guard.
    struct CountingPredict {
        batch: bool,
        chunk: bool,
        chunk_batch: bool,
        native_chunk: usize,
        short: bool,
        predict_calls: AtomicUsize,
        chunk_calls: AtomicUsize,
        batch_calls: AtomicUsize,
        chunk_batch_calls: AtomicUsize,
    }

    impl CountingPredict {
        fn new(
            batch: bool,
            chunk: bool,
            chunk_batch: bool,
            native_chunk: usize,
            short: bool,
        ) -> Arc<Self> {
            Arc::new(Self {
                batch,
                chunk,
                chunk_batch,
                native_chunk,
                short,
                predict_calls: AtomicUsize::new(0),
                chunk_calls: AtomicUsize::new(0),
                batch_calls: AtomicUsize::new(0),
                chunk_batch_calls: AtomicUsize::new(0),
            })
        }

        fn native_chunk_value(&self) -> Value {
            Value::List(
                (0..self.native_chunk)
                    .map(|frame| Value::Number(frame as f64))
                    .collect(),
            )
        }
    }

    impl PredictFn for CountingPredict {
        fn predict(&self, _model_input: Value) -> Result<Value> {
            self.predict_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Value::Number(0.0))
        }

        fn predict_spec_less(&self, observation: ModelObservation) -> Result<Vec<SpaceValue>> {
            Ok((0..observation.num_envs)
                .map(|_| SpaceValue::Discrete(0))
                .collect())
        }

        fn has_chunk(&self) -> bool {
            self.chunk
        }

        fn predict_chunk(&self, _model_input: Value, _horizon: u32) -> Result<Option<Value>> {
            self.chunk_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Some(self.native_chunk_value()))
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
            Ok((0..keep).map(|_| self.native_chunk_value()).collect())
        }

        fn allow_fusion(&self) -> bool {
            true
        }
    }

    fn lanes(count: usize) -> Vec<Value> {
        (0..count).map(|lane| Value::Number(lane as f64)).collect()
    }

    #[test]
    fn dispatch_prefers_chunk_batch_and_caps_to_horizon() {
        let counting = CountingPredict::new(true, false, true, 10, false);
        let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

        let frames = dispatch_route_corners(&predict, lanes(5), 4, 5).expect("frames");

        assert_eq!(counting.chunk_batch_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 0);
        assert_eq!(frames.len(), 5);
        assert!(
            frames.iter().all(|lane| lane.len() == 4),
            "each lane keeps the horizon prefix of its native chunk"
        );
    }

    #[test]
    fn dispatch_horizon_one_uses_plain_batch() {
        let counting = CountingPredict::new(true, false, true, 10, false);
        let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

        let frames = dispatch_route_corners(&predict, lanes(3), 1, 3).expect("frames");

        assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counting.chunk_batch_calls.load(Ordering::SeqCst), 0);
        assert!(frames.iter().all(|lane| lane.len() == 1));
    }

    /// The chunk-preserving corner precedence a declined fusion falls back to:
    /// a chunk + batch model (no batched chunk corner) at horizon > 1 keeps its
    /// per-lane chunk corner and full replay frames.
    #[test]
    fn dispatch_horizon_gt_one_prefers_per_lane_chunk_over_batch() {
        let counting = CountingPredict::new(true, true, false, 10, false);
        let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

        let frames = dispatch_route_corners(&predict, lanes(3), 4, 3).expect("frames");

        assert_eq!(counting.chunk_calls.load(Ordering::SeqCst), 3);
        assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 0);
        assert!(
            frames.iter().all(|lane| lane.len() == 4),
            "per-lane chunks keep the horizon prefix"
        );
    }

    #[test]
    fn dispatch_reports_lane_count_mismatch() {
        let counting = CountingPredict::new(true, false, true, 10, true);
        let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

        let error = dispatch_route_corners(&predict, lanes(4), 4, 4).expect_err("short fails");

        assert!(
            error.to_string().contains("lanes"),
            "unexpected error: {error}"
        );
    }

    /// The behavior-parity fusion gate: a bucket fuses ONLY when the batched
    /// corner it would run is the corner a direct predict picks, so grouping
    /// never changes which model function runs.
    #[test]
    fn bucket_fuses_only_when_the_direct_corner_is_batched() {
        let chunk_batch = CountingPredict::new(true, true, true, 10, false);
        assert!(bucket_fuses(chunk_batch.as_ref(), 4));
        assert!(bucket_fuses(chunk_batch.as_ref(), 1));

        let chunk_and_batch = CountingPredict::new(true, true, false, 10, false);
        assert!(
            !bucket_fuses(chunk_and_batch.as_ref(), 4),
            "direct picks the per-lane chunk corner; fusing would drop chunking"
        );
        assert!(bucket_fuses(chunk_and_batch.as_ref(), 1));

        let batch_only = CountingPredict::new(true, false, false, 10, false);
        assert!(
            bucket_fuses(batch_only.as_ref(), 4),
            "direct already degrades to single-step batch (warned at configure)"
        );

        let chunk_batch_only = CountingPredict::new(false, false, true, 10, false);
        assert!(
            !bucket_fuses(chunk_batch_only.as_ref(), 1),
            "direct picks the per-lane predict loop at horizon 1"
        );
        assert!(bucket_fuses(chunk_batch_only.as_ref(), 4));

        let per_lane_only = CountingPredict::new(false, false, false, 10, false);
        assert!(!bucket_fuses(per_lane_only.as_ref(), 1));
        assert!(!bucket_fuses(per_lane_only.as_ref(), 4));
    }

    /// A fused corner failure is cloned per group (not rebuilt from its
    /// message), so the model's recoverable flag survives to the wire and the
    /// annotation carries the group's own input signature.
    #[test]
    fn fused_error_broadcast_preserves_recoverability() {
        let error = Error::model_recoverable("transient OOM, retry");

        let annotated = annotate_predict_error(error.clone(), "sig: float32[8]; lanes: 2");

        assert!(annotated.is_recoverable(), "recoverable flag must survive");
        match annotated {
            Error::Model(model) => assert_eq!(
                model.message,
                "transient OOM, retry\nsig: float32[8]; lanes: 2"
            ),
            other => panic!("expected Error::Model, got {other:?}"),
        }
    }

    /// Spec-less routes bypass the engine corners entirely: each group serves
    /// through the preserved raw path, one result per group, in order.
    #[tokio::test]
    async fn grouped_predict_serves_spec_less_groups_per_group() {
        let counting = CountingPredict::new(true, false, true, 10, false);
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
        assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 0);
        assert_eq!(counting.chunk_batch_calls.load(Ordering::SeqCst), 0);
    }
}

#[cfg(test)]
mod fused_route_tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use rlmesh_adapters::v1::{EnvTags, ModelSpec, NoCustoms, NoEncodings, SpaceView, resolve};

    use super::*;
    use crate::model::types::ModelRouteContext;
    use crate::spaces::{self, DType, Tensor};

    const ENV_TAGS: &str = r#"{
        "observation": {"state": {"type": "state", "role": "proprio/gripper"}},
        "action": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;
    const MODEL_SPEC: &str = r#"{
        "input": {"state": {"type": "state", "container": "list", "dtype": "float32",
                            "components": [{"role": "proprio/gripper", "dim": 1}]}},
        "output": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;

    /// Resolves every route through the one ENV_TAGS x MODEL_SPEC pairing — the
    /// minimal real spec'd route (one gripper state in, one gripper action out),
    /// so grouped predicts exercise assemble -> fuse -> split -> finish against
    /// genuine adapter plans instead of hand-built frames.
    struct TagResolver;

    #[async_trait]
    impl RouteResolver for TagResolver {
        async fn resolve(
            &self,
            _route_key: &str,
            env_contract: &EnvContract,
        ) -> Result<Option<RouteConfig>> {
            let tags: EnvTags = serde_json::from_str(ENV_TAGS).expect("env tags parse");
            let spec: ModelSpec = serde_json::from_str(MODEL_SPEC).expect("model spec parse");
            let obs = env_contract
                .observation_space
                .clone()
                .expect("contract obs space");
            let action = env_contract
                .action_space
                .clone()
                .expect("contract action space");
            let adapter = resolve(
                &tags,
                &SpaceView::from(&obs),
                &SpaceView::from(&action),
                &spec,
                true,
            )
            .map_err(|err| Error::model(err.message))?;
            Ok(Some(RouteConfig::new(
                adapter,
                obs,
                action,
                Box::new(NoCustoms),
                Box::new(NoEncodings),
            )))
        }
    }

    /// Echoes each lane's state value back as its action (lane-identifying, so a
    /// split-back misalignment returns the wrong route's actions and fails the
    /// equality asserts). The chunk corner emits `CHUNK_FRAMES` frames of
    /// `state + 0.125 * frame`; `fail_recoverable` makes the batched corner
    /// return a recoverable model error instead.
    struct EchoModel {
        chunk: bool,
        fail_recoverable: bool,
        predict_calls: AtomicUsize,
        chunk_calls: AtomicUsize,
        batch_calls: AtomicUsize,
    }

    const CHUNK_FRAMES: usize = 6;

    impl EchoModel {
        fn new(chunk: bool, fail_recoverable: bool) -> Arc<Self> {
            Arc::new(Self {
                chunk,
                fail_recoverable,
                predict_calls: AtomicUsize::new(0),
                chunk_calls: AtomicUsize::new(0),
                batch_calls: AtomicUsize::new(0),
            })
        }
    }

    /// The first number reachable in the assembled input tree (the echoed state).
    fn state_number(input: &Value) -> f64 {
        match input {
            Value::Number(n) => *n,
            Value::List(items) => items.first().map(state_number).unwrap_or(0.0),
            Value::Map(map) => map.values().next().map(state_number).unwrap_or(0.0),
            _ => 0.0,
        }
    }

    fn action_value(state: f64) -> Value {
        Value::List(vec![Value::Number(state)])
    }

    impl PredictFn for EchoModel {
        fn predict(&self, model_input: Value) -> Result<Value> {
            self.predict_calls.fetch_add(1, Ordering::SeqCst);
            Ok(action_value(state_number(&model_input)))
        }

        fn predict_spec_less(&self, _observation: ModelObservation) -> Result<Vec<SpaceValue>> {
            Err(Error::model("EchoModel serves spec'd routes only"))
        }

        fn has_chunk(&self) -> bool {
            self.chunk
        }

        fn predict_chunk(&self, model_input: Value, _horizon: u32) -> Result<Option<Value>> {
            self.chunk_calls.fetch_add(1, Ordering::SeqCst);
            let state = state_number(&model_input);
            Ok(Some(Value::List(
                (0..CHUNK_FRAMES)
                    .map(|frame| action_value(state + 0.125 * frame as f64))
                    .collect(),
            )))
        }

        fn has_batch(&self) -> bool {
            true
        }

        fn predict_batch(&self, inputs: Vec<Value>) -> Result<Vec<Value>> {
            self.batch_calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_recoverable {
                return Err(Error::model_recoverable("transient forward failure"));
            }
            Ok(inputs
                .iter()
                .map(|input| action_value(state_number(input)))
                .collect())
        }

        fn allow_fusion(&self) -> bool {
            true
        }
    }

    fn obs_space() -> spaces::SpaceSpec {
        spaces::spaces::DictSpaceBuilder::new()
            .insert(
                "state",
                spaces::spaces::BoxSpaceBuilder::scalar(0.0, 10.0, vec![1])
                    .dtype(DType::Float32)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap()
    }

    fn action_space() -> spaces::SpaceSpec {
        spaces::spaces::BoxSpaceBuilder::scalar(0.0, 10.0, vec![1])
            .dtype(DType::Float32)
            .build()
            .unwrap()
    }

    fn contract(env_id: &str, num_envs: u32) -> spaces::EnvContract {
        spaces::EnvContract {
            id: env_id.to_string(),
            observation_space: Some(obs_space()),
            action_space: Some(action_space()),
            metadata: None,
            render_mode: String::new(),
            num_envs,
            autoreset_mode: Default::default(),
        }
    }

    fn box_f32(value: f32) -> SpaceValue {
        SpaceValue::Box(
            Tensor::from_vec(value.to_le_bytes().to_vec(), vec![1], DType::Float32).unwrap(),
        )
    }

    /// A grouped-member observation for `env_id`: one lane per value, each lane's
    /// state = the value, episode ids derived from the values (unique per lane).
    fn grouped_obs(
        env_id: &str,
        values: &[f32],
        env_contract: &Arc<spaces::EnvContract>,
    ) -> ModelObservation {
        let lanes: Vec<SpaceValue> = values
            .iter()
            .map(|value| {
                SpaceValue::Dict(BTreeMap::from([(
                    "state".to_string(),
                    box_f32(*value).clone(),
                )]))
            })
            .collect();
        let wire = rlmesh_grpc::wire::encode_batched_partial_values(&lanes, &obs_space()).unwrap();
        ModelObservation {
            observation: Some(wire.leaves),
            route: ModelRouteContext {
                env_id: env_id.to_string(),
                episode_ids: values
                    .iter()
                    .map(|value| format!("{env_id}-ep-{value}"))
                    .collect(),
                ..Default::default()
            },
            num_envs: values.len(),
            env_contract: Some(Arc::clone(env_contract)),
        }
    }

    /// Build a handler with real resolved routes (one per `(env_id, lanes)`),
    /// pinning `horizon` on each at resolve.
    async fn spec_handler(
        predict: Arc<dyn PredictFn>,
        envs: &[(&str, u32)],
        horizon: u32,
    ) -> (AdaptedModelHandler, Vec<Arc<spaces::EnvContract>>) {
        let handler = AdaptedModelHandler::new(predict, Some(Arc::new(TagResolver)));
        let setup = handler.route_setup().expect("resolver-backed route setup");
        let mut contracts = Vec::with_capacity(envs.len());
        for (env_id, lanes) in envs {
            let env_contract = contract(env_id, *lanes);
            setup
                .resolve_adapter(env_id, &env_contract, horizon)
                .await
                .expect("route resolves");
            contracts.push(Arc::new(env_contract));
        }
        (handler, contracts)
    }

    #[tokio::test]
    async fn fused_grouped_predict_runs_one_forward_and_splits_actions_per_group() {
        let echo = EchoModel::new(false, false);
        let (mut handler, contracts) = spec_handler(
            Arc::clone(&echo) as Arc<dyn PredictFn>,
            &[("env-a", 2), ("env-b", 3)],
            1,
        )
        .await;
        let batch_baseline = echo.batch_calls.load(Ordering::SeqCst);

        let results = handler
            .predict_grouped(vec![
                grouped_obs("env-a", &[1.0, 2.0], &contracts[0]),
                grouped_obs("env-b", &[3.0, 4.0, 5.0], &contracts[1]),
            ])
            .await;

        assert_eq!(
            echo.batch_calls.load(Ordering::SeqCst) - batch_baseline,
            1,
            "the whole group rides ONE fused forward"
        );
        assert_eq!(results.len(), 2);
        let a = results[0].as_ref().expect("env-a serves");
        assert_eq!(
            a.actions,
            vec![box_f32(1.0), box_f32(2.0)],
            "env-a gets its own lanes back"
        );
        assert!(a.replay.is_empty());
        let b = results[1].as_ref().expect("env-b serves");
        assert_eq!(
            b.actions,
            vec![box_f32(3.0), box_f32(4.0), box_f32(5.0)],
            "env-b gets its own lanes back"
        );
        assert!(b.replay.is_empty());
    }

    #[tokio::test]
    async fn grouped_chunk_and_batch_model_keeps_chunking_when_grouped() {
        let echo = EchoModel::new(true, false);
        let (mut handler, contracts) = spec_handler(
            Arc::clone(&echo) as Arc<dyn PredictFn>,
            &[("env-a", 2), ("env-b", 3)],
            4,
        )
        .await;
        let batch_baseline = echo.batch_calls.load(Ordering::SeqCst);
        let chunk_baseline = echo.chunk_calls.load(Ordering::SeqCst);

        let results = handler
            .predict_grouped(vec![
                grouped_obs("env-a", &[1.0, 2.0], &contracts[0]),
                grouped_obs("env-b", &[3.0, 4.0, 5.0], &contracts[1]),
            ])
            .await;

        assert_eq!(
            echo.batch_calls.load(Ordering::SeqCst) - batch_baseline,
            0,
            "the parity gate declines fusion: batching would drop chunking"
        );
        assert_eq!(
            echo.chunk_calls.load(Ordering::SeqCst) - chunk_baseline,
            5,
            "every lane runs the per-lane chunk corner, exactly as ungrouped"
        );
        for (result, lane_states) in results.iter().zip([vec![1.0f32, 2.0], vec![3.0, 4.0, 5.0]]) {
            let frames = result.as_ref().expect("group serves chunked");
            let frame0: Vec<SpaceValue> = lane_states.iter().map(|s| box_f32(*s)).collect();
            assert_eq!(frames.actions, frame0);
            assert_eq!(frames.replay.len(), 3, "horizon 4 = frame 0 + 3 replays");
            for (step, row) in frames.replay.iter().enumerate() {
                let expected: Vec<SpaceValue> = lane_states
                    .iter()
                    .map(|s| box_f32(s + 0.125 * (step as f32 + 1.0)))
                    .collect();
                assert_eq!(row, &expected, "replay step {step} keeps lane order");
            }
        }
    }

    /// Regression for the grouped-fusion deadlock: the fused path once held every
    /// route's entry guard across the whole call, so a request repeating one
    /// env_id re-locked the same mutex on one thread and hung forever. Locks are
    /// now short-lived; a duplicate route must simply serve twice.
    #[tokio::test]
    async fn grouped_predict_with_duplicate_env_completes() {
        let echo = EchoModel::new(false, false);
        let (mut handler, contracts) =
            spec_handler(Arc::clone(&echo) as Arc<dyn PredictFn>, &[("env-a", 2)], 1).await;

        let results = tokio::time::timeout(
            Duration::from_secs(10),
            handler.predict_grouped(vec![
                grouped_obs("env-a", &[1.0, 2.0], &contracts[0]),
                grouped_obs("env-a", &[3.0, 4.0], &contracts[0]),
            ]),
        )
        .await
        .expect("a grouped predict repeating an env must not deadlock");

        assert_eq!(results.len(), 2);
        let first = results[0].as_ref().expect("first duplicate serves");
        assert_eq!(first.actions, vec![box_f32(1.0), box_f32(2.0)]);
        let second = results[1].as_ref().expect("second duplicate serves");
        assert_eq!(second.actions, vec![box_f32(3.0), box_f32(4.0)]);
    }

    #[tokio::test]
    async fn fused_failure_reports_recoverable_error_with_each_groups_own_signature() {
        let echo = EchoModel::new(false, true);
        let (mut handler, contracts) = spec_handler(
            Arc::clone(&echo) as Arc<dyn PredictFn>,
            &[("env-a", 2), ("env-b", 3)],
            1,
        )
        .await;

        let results = handler
            .predict_grouped(vec![
                grouped_obs("env-a", &[1.0, 2.0], &contracts[0]),
                grouped_obs("env-b", &[3.0, 4.0, 5.0], &contracts[1]),
            ])
            .await;

        for (result, lanes) in results.iter().zip([2usize, 3]) {
            let error = result
                .as_ref()
                .expect_err("fused failure reaches the group");
            assert!(
                error.is_recoverable(),
                "the model's recoverable flag survives the fused broadcast: {error}"
            );
            assert!(
                error.to_string().contains(&format!("lanes: {lanes}")),
                "each group is annotated with its OWN input signature, got: {error}"
            );
        }
    }
}
