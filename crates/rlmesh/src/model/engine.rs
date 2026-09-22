//! The vectorized stateful adapter engine: a [`ModelHandler`] that owns the
//! per-lane drive, episode-keyed frame buffers, and native adapter application
//! in pure Rust, calling back into a [`PredictFn`] only for the model's predict
//! and into the custom/encoding holes only where a route declares them.
//!
//! This replaces a binding's hand-rolled predict loop: a PyO3 (or any future
//! language) binding constructs `AdaptedModelHandler::new(predict, resolver)`
//! and serves it; a pure-Rust model does the same with no host runtime.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use rayon::prelude::*;
use rlmesh_adapters::v1::{
    FrameBuffers, HistoryWindow, MAX_EXECUTION_HORIZON, ObsPlan, Value, apply_actions,
    assemble_obs, observe_obs, space_value_to_obs_map, split_chunk,
};

use super::handler::{
    HeldState, HistoryNeeds, ModelHandler, ModelRouteSetup, PredictFrames, ResolveOptions,
    RouteNeeds,
};
use super::predict_fn::{PredictFn, RouteConfig, RouteResolver};
use super::types::{EpisodeInfo, ModelObservation};
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
    /// The endpoint-wide held-state total this route publishes into (see
    /// [`HeldCells`]).
    total: Arc<HeldCells>,
    /// What this route last published into `total`, so a publish (or the drop
    /// of a released route) applies only the difference.
    published: HeldTotals,
    /// Bytes one lane's frame windows hold once full (`Σ span × frame_bytes`);
    /// `0` for a route with no stacked input.
    lane_bytes: u64,
    /// The endpoint-wide frame-window budget this route's lanes are admitted
    /// against (see [`FRAME_HISTORY_LIMIT_ENV`]).
    history_limit: u64,
}

impl RouteEntry {
    /// Publish this route's held frame-stack state into the endpoint total, as
    /// the delta against its previous publish.
    fn publish_held(&mut self) {
        let episodes = self.buffers.episodes() as u64;
        let now = HeldTotals {
            episodes,
            bytes: self.buffers.state_bytes(),
            projected: episodes.saturating_mul(self.lane_bytes),
        };
        // One wrapping add per cell: a shrink adds the two's complement of the
        // drop, so a concurrent read never sees the total dip below the other
        // routes' share.
        self.total.episodes.fetch_add(
            now.episodes.wrapping_sub(self.published.episodes),
            Ordering::Relaxed,
        );
        self.total.bytes.fetch_add(
            now.bytes.wrapping_sub(self.published.bytes),
            Ordering::Relaxed,
        );
        self.total.projected.fetch_add(
            now.projected.wrapping_sub(self.published.projected),
            Ordering::Relaxed,
        );
        self.published = now;
    }
}

impl Drop for RouteEntry {
    /// A released (or re-resolved) route takes its share back out of the total.
    fn drop(&mut self) {
        self.total
            .episodes
            .fetch_sub(self.published.episodes, Ordering::Relaxed);
        self.total
            .bytes
            .fetch_sub(self.published.bytes, Ordering::Relaxed);
        self.total
            .projected
            .fetch_sub(self.published.projected, Ordering::Relaxed);
    }
}

/// One route's last publish into the endpoint total: the [`HeldState`] pair
/// plus the projected full-window bytes the admission guard charges.
#[derive(Clone, Copy, Default)]
struct HeldTotals {
    episodes: u64,
    bytes: u64,
    projected: u64,
}

/// The endpoint-wide held frame-stack total, kept by per-route deltas published
/// from under each entry lock, so `held_state` reads it in O(1) without the
/// routes map or any entry lock that may be held across a model forward.
#[derive(Default)]
struct HeldCells {
    episodes: AtomicU64,
    bytes: AtomicU64,
    /// What every route's held episodes' windows hold once full
    /// (`Σ episodes × lane_bytes`): the admission guard reads this rather than
    /// `bytes`, so a window still filling is charged what it will reach, not
    /// what it has so far.
    projected: AtomicU64,
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
    /// The held-state total every route publishes into (see [`HeldCells`]).
    held: Arc<HeldCells>,
    /// Adapter time (obs assembly + action apply) accumulated by the current
    /// predict-family call, drained per request via `take_adapter_ns`. Shared
    /// (`Arc`) because the work runs on `spawn_blocking` threads.
    adapter_ns: Arc<AtomicU64>,
    /// Whether the short-chunk warning has already fired at this endpoint (see
    /// [`take_prefix`]). Endpoint-wide, not per route: the fused grouped path
    /// dispatches lanes concatenated ACROSS routes, so a per-route flag could
    /// not name the offender anyway — and one line per process is the point.
    short_chunk_warned: Arc<AtomicBool>,
    /// The endpoint-wide frame-window budget (see [`FRAME_HISTORY_LIMIT_ENV`]),
    /// read once here so every route is admitted against the same number.
    history_limit: u64,
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
            held: Arc::new(HeldCells::default()),
            adapter_ns: Arc::new(AtomicU64::new(0)),
            short_chunk_warned: Arc::new(AtomicBool::new(false)),
            history_limit: frame_history_limit(),
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
            buffers.advance_step(episode_id, step)?;
        }
        let raw = space_value_to_obs_map(lane, &config.observation_space, &referenced)?;
        inputs.push(assemble_obs(
            &config.adapter,
            &raw,
            episode_id,
            step,
            buffers,
            customs,
            encodings,
        )?);
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
            buffers.advance_step(&episode.episode_id, frame.step)?;
            let raw = space_value_to_obs_map(lane, &config.observation_space, referenced)?;
            observe_obs(&config.adapter, &raw, &episode.episode_id, buffers)?;
        }
    }
    Ok(())
}

/// Env var naming the most bytes this endpoint's frame windows may project to
/// hold across every route's live lanes (`Σ lanes × Σ span × frame_bytes`);
/// default 2 GiB. Checked twice: per route at resolve, for the lanes the
/// contract names, and per fresh lane at predict, against the endpoint total.
const FRAME_HISTORY_LIMIT_ENV: &str = "RLMESH_FRAME_HISTORY_LIMIT_BYTES";
const FRAME_HISTORY_LIMIT_DEFAULT: u64 = 2 << 30;

fn frame_history_limit() -> u64 {
    std::env::var(FRAME_HISTORY_LIMIT_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(FRAME_HISTORY_LIMIT_DEFAULT)
}

/// Bytes one lane's windows hold once full.
fn lane_window_bytes(windows: &[HistoryWindow]) -> u64 {
    windows
        .iter()
        .map(|window| u64::from(window.span).saturating_mul(window.frame_bytes))
        .sum()
}

/// Refuse at configure a stacked route whose windows would outgrow the budget
/// on their own, instead of finding out from `held.bytes` once the lanes are
/// live. Per-route validation only: the served resolve names no lanes (this
/// checks one), so the endpoint total is enforced where lanes are real, in
/// [`admit_fresh_lanes`].
fn check_history_budget(num_envs: u32, windows: &[HistoryWindow], limit: u64) -> Result<()> {
    let per_lane = lane_window_bytes(windows);
    let projected = per_lane.saturating_mul(u64::from(num_envs));
    if projected > limit {
        return Err(Error::model(format!(
            "frame windows would hold {projected} bytes across {num_envs} lane(s) ({per_lane} per \
             lane), above the {limit}-byte budget: shrink stack/size, or raise \
             {FRAME_HISTORY_LIMIT_ENV}"
        )));
    }
    Ok(())
}

/// Refuse a predict that would open frame windows for episodes this route does
/// not hold yet when the endpoint's projected windows (every route's held
/// episodes at their full size, plus these) would outgrow the budget. Runs
/// under the route's entry lock before anything is seeded, so a refused
/// request leaves no state behind; a route with no stacked input is never
/// charged. Reads the endpoint total the routes publish into, so it takes no
/// other lock.
///
/// ponytail: the other routes' share is a snapshot, so two routes admitting
/// fresh lanes at the same instant can each pass by up to one request's
/// windows; a strict reservation would need a cross-route lock.
fn admit_fresh_lanes(entry: &RouteEntry, episodes: &[EpisodeInfo]) -> Result<()> {
    if entry.lane_bytes == 0 {
        return Ok(());
    }
    let fresh = episodes
        .iter()
        .filter(|episode| {
            !episode.episode_id.is_empty() && !entry.buffers.holds(&episode.episode_id)
        })
        .count() as u64;
    if fresh == 0 {
        return Ok(());
    }
    let endpoint = entry.total.projected.load(Ordering::Relaxed);
    let others = endpoint.saturating_sub(entry.published.projected);
    let route = (entry.buffers.episodes() as u64)
        .saturating_add(fresh)
        .saturating_mul(entry.lane_bytes);
    let projected = others.saturating_add(route);
    if projected > entry.history_limit {
        return Err(Error::model(format!(
            "frame windows would hold {projected} bytes across this endpoint's live lanes \
             ({fresh} new on this route at {} per lane, {others} projected by other routes), \
             above the {}-byte budget: run fewer lanes or routes per model endpoint, shrink \
             stack/size, or raise {FRAME_HISTORY_LIMIT_ENV}",
            entry.lane_bytes, entry.history_limit
        )));
    }
    Ok(())
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
fn dispatch_corners(
    predict: &Arc<dyn PredictFn>,
    inputs: Vec<Value>,
    episodes: &[EpisodeInfo],
    horizon: u32,
    num_envs: usize,
    short_chunk_warned: &AtomicBool,
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
            .map(|chunk| {
                take_prefix(
                    split_chunk(chunk)?,
                    horizon,
                    native_chunk,
                    short_chunk_warned,
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
                    split_chunk(chunk)?,
                    horizon,
                    native_chunk,
                    short_chunk_warned,
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
fn dispatch_route_corners(
    predict: &Arc<dyn PredictFn>,
    inputs: Vec<Value>,
    episodes: &[EpisodeInfo],
    horizon: u32,
    num_envs: usize,
    short_chunk_warned: &AtomicBool,
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
fn take_prefix(
    mut frames: Vec<Value>,
    horizon: u32,
    native_chunk: Option<u32>,
    short_chunk_warned: &AtomicBool,
) -> Result<Vec<Value>> {
    match native_chunk {
        Some(native) if frames.len() != native as usize => {
            return Err(Error::model(format!(
                "model declares native_chunk={native} but its chunk corner returned {} \
                 frames at execution_horizon={horizon}: return the WHOLE native chunk and \
                 let the runtime execute its prefix (do not slice to the horizon), or drop \
                 the declaration",
                frames.len(),
            )));
        }
        Some(_) => {}
        None => {
            if frames.len() < horizon as usize && !short_chunk_warned.swap(true, Ordering::Relaxed)
            {
                tracing::warn!(
                    execution_horizon = horizon,
                    chunk_len = frames.len(),
                    "model returned a chunk shorter than the pinned execution horizon; the \
                     runtime replays what there is and re-plans early. Declare native_chunk \
                     (Model.native_chunk) so the horizon can be validated at resolve",
                );
            }
        }
    }
    frames.truncate(horizon as usize);
    Ok(frames)
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
///
/// Each lane's frame `k` is applied under the step it executes at, `k` past
/// the step the lane assembled at, so a route reading its previous action
/// sees, at the re-plan, the frame the runtime replayed last. That record
/// lives in the route's buffers, so the entry lock is taken again here (after
/// the model call, never across it).
fn finish_route_frames(
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
/// blocking worker thread. Holds the per-route entry lock across input
/// assembly (the frame buffers mutate in place there) and again, after the
/// model call, across the per-frame apply (the action window records there),
/// never across the call itself; the horizon it dispatches with is the
/// runtime-chosen execution horizon pinned on `ResolveAdapter`, not the model
/// spec.
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
    adapter_ns: &AtomicU64,
    short_chunk_warned: &AtomicBool,
) -> Result<PredictFrames> {
    let num_envs = observation.num_envs;
    let (inputs, config) = {
        let mut guard = entry.lock().expect("route entry poisoned");
        let inputs = assemble_route_inputs(&mut guard, &observation, adapter_ns)?;
        (inputs, Arc::clone(&guard.config))
    };
    let lane_raw_steps = dispatch_route_corners(
        predict,
        inputs,
        &observation.route.episodes,
        config.execution_horizon,
        num_envs,
        short_chunk_warned,
    )?;
    finish_route_frames(
        entry,
        lane_raw_steps,
        &observation.route.episodes,
        num_envs,
        adapter_ns,
    )
}

/// One grouped predict group's serving lane, classified once (on the async
/// side) from the routes map: a spec-less route carries the horizon pinned for
/// it (there is no [`RouteEntry`] to stamp it on); a spec'd route carries its
/// entry.
enum GroupLane {
    SpecLess { horizon: u32 },
    Routed { entry: Arc<Mutex<RouteEntry>> },
}

/// Whether a grouped request's spec'd groups may assemble and finish on the
/// rayon pool: every routed group names a distinct route (so no two threads
/// contend one entry lock in undefined order) and no route's plan runs a
/// Python custom transform (which would take the GIL from the pool).
fn assembles_in_parallel(lanes: &[GroupLane]) -> bool {
    let mut seen: HashSet<*const Mutex<RouteEntry>> = HashSet::new();
    let mut routed = 0usize;
    for lane in lanes {
        if let GroupLane::Routed { entry } = lane {
            routed += 1;
            if !seen.insert(Arc::as_ptr(entry)) {
                return false;
            }
            let guard = entry.lock().expect("route entry poisoned");
            if guard
                .config
                .adapter
                .obs_plans
                .iter()
                .any(|plan| matches!(plan, ObsPlan::Custom(_)))
            {
                return false;
            }
        }
    }
    routed > 1
}

/// Runs a parallel adapter section and charges `adapter_ns` its wall time.
/// The per-lane kernels still stamp their own spans, into `scratch`, which is
/// dropped: on the rayon pool those spans overlap, and summing them reported
/// thread time (about 2.5x the wall at 32 lanes) as the endpoint's `adapter`
/// phase, which is meant to be the time the request spent in adapter work.
fn charge_wall<T>(adapter_ns: &AtomicU64, section: impl FnOnce(&AtomicU64) -> T) -> T {
    let scratch = AtomicU64::new(0);
    let started = Instant::now();
    let out = section(&scratch);
    adapter_ns.fetch_add(rlmesh_proto::elapsed_ns(started), Ordering::Relaxed);
    out
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
    adapter_ns: &AtomicU64,
    short_chunk_warned: &AtomicBool,
) -> Vec<Result<PredictFrames>> {
    struct Prepared {
        index: usize,
        inputs: Vec<Value>,
        episodes: Vec<EpisodeInfo>,
        entry: Arc<Mutex<RouteEntry>>,
        config: Arc<RouteConfig>,
        num_envs: usize,
    }

    // A spec'd group's assembly is host work under its own route entry lock,
    // independent of every other group's, so the groups assemble in parallel
    // (indexed, so results keep request order) — unless two groups name the
    // same route (their buffer mutations would then interleave in no defined
    // order; the runtime never groups one route twice) or a plan runs Python
    // (a custom transform takes the GIL, which serialises the pool anyway).
    // A spec-less group calls the model itself and stays on this thread.
    enum Staged {
        SpecLess {
            observation: ModelObservation,
            horizon: u32,
        },
        Routed {
            assembled: Result<(Vec<Value>, Arc<RouteConfig>)>,
            episodes: Vec<EpisodeInfo>,
            entry: Arc<Mutex<RouteEntry>>,
            num_envs: usize,
        },
    }
    let parallel = assembles_in_parallel(&lanes);
    let stage =
        |(lane, observation): (GroupLane, ModelObservation), adapter_ns: &AtomicU64| match lane {
            GroupLane::SpecLess { horizon } => Staged::SpecLess {
                observation,
                horizon,
            },
            GroupLane::Routed { entry } => {
                let num_envs = observation.num_envs;
                let assembled = {
                    let mut guard = entry.lock().expect("route entry poisoned");
                    assemble_route_inputs(&mut guard, &observation, adapter_ns)
                        .map(|inputs| (inputs, Arc::clone(&guard.config)))
                };
                Staged::Routed {
                    assembled,
                    episodes: observation.route.episodes,
                    entry,
                    num_envs,
                }
            }
        };
    let staged: Vec<Staged> = if parallel {
        charge_wall(adapter_ns, |ns| {
            lanes
                .into_par_iter()
                .zip(observations.into_par_iter())
                .map(|pair| stage(pair, ns))
                .collect()
        })
    } else {
        lanes
            .into_iter()
            .zip(observations)
            .map(|pair| stage(pair, adapter_ns))
            .collect()
    };

    let mut results: Vec<Option<Result<PredictFrames>>> = Vec::with_capacity(staged.len());
    let mut prepared: Vec<Prepared> = Vec::new();
    for (index, staged) in staged.into_iter().enumerate() {
        match staged {
            Staged::SpecLess {
                observation,
                horizon,
            } => results.push(Some(
                predict.predict_spec_less_chunked(observation, horizon),
            )),
            Staged::Routed {
                assembled: Ok((inputs, config)),
                episodes,
                entry,
                num_envs,
            } => {
                results.push(None);
                prepared.push(Prepared {
                    index,
                    inputs,
                    episodes,
                    entry,
                    config,
                    num_envs,
                });
            }
            Staged::Routed {
                assembled: Err(error),
                ..
            } => results.push(Some(Err(error))),
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
                entry: Arc<Mutex<RouteEntry>>,
                episodes: Vec<EpisodeInfo>,
                num_envs: usize,
            }
            let mut flat: Vec<Value> = Vec::new();
            let mut flat_episodes: Vec<EpisodeInfo> = Vec::new();
            let mut fused: Vec<FusedGroup> = Vec::with_capacity(groups.len());
            for group in groups {
                fused.push(FusedGroup {
                    index: group.index,
                    lane_count: group.inputs.len(),
                    summary: inputs_summary(&group.inputs),
                    entry: group.entry,
                    episodes: group.episodes.clone(),
                    num_envs: group.num_envs,
                });
                flat.extend(group.inputs);
                flat_episodes.extend(group.episodes);
            }
            let total = flat.len();
            // Each group's episodes were assembled alongside its own inputs, so a
            // disagreement here is an engine bug, not a bad request.
            let fused_result = if flat_episodes.len() == total {
                dispatch_corners(
                    predict,
                    flat,
                    &flat_episodes,
                    horizon,
                    total,
                    short_chunk_warned,
                )
            } else {
                Err(Error::Internal(format!(
                    "fused predict concatenated {} episode rows for {total} lanes",
                    flat_episodes.len()
                )))
            };
            match fused_result {
                Ok(all_frames) => {
                    // Split the fused chunk per group in order, then apply each
                    // group's action frames under its own entry lock — in
                    // parallel under the same conditions as assembly.
                    let mut frames = all_frames.into_iter();
                    let split: Vec<(FusedGroup, Vec<Vec<Value>>)> = fused
                        .into_iter()
                        .map(|group| {
                            let group_frames = frames.by_ref().take(group.lane_count).collect();
                            (group, group_frames)
                        })
                        .collect();
                    let finish = |(group, group_frames): (FusedGroup, Vec<Vec<Value>>),
                                  adapter_ns: &AtomicU64| {
                        let result = finish_route_frames(
                            &group.entry,
                            group_frames,
                            &group.episodes,
                            group.num_envs,
                            adapter_ns,
                        );
                        (group.index, result)
                    };
                    let finished: Vec<(usize, Result<PredictFrames>)> = if parallel {
                        charge_wall(adapter_ns, |ns| {
                            split.into_par_iter().map(|pair| finish(pair, ns)).collect()
                        })
                    } else {
                        split
                            .into_iter()
                            .map(|pair| finish(pair, adapter_ns))
                            .collect()
                    };
                    for (index, result) in finished {
                        results[index] = Some(result);
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
            // Each non-fused group is exactly one route, so its row-aligned
            // episode identity travels with it — the same dispatch a direct
            // predict of that route would run.
            for group in groups {
                results[group.index] = Some(
                    dispatch_route_corners(
                        predict,
                        group.inputs,
                        &group.episodes,
                        horizon,
                        group.num_envs,
                        short_chunk_warned,
                    )
                    .and_then(|raw| {
                        finish_route_frames(
                            &group.entry,
                            raw,
                            &group.episodes,
                            group.num_envs,
                            adapter_ns,
                        )
                    }),
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
        let adapter_ns = Arc::clone(&self.adapter_ns);
        // Decode + frame-stack + the model's predict are CPU/host work; run them
        // off the async worker so concurrent (pipelined) requests on other routes
        // are not stalled. A spec'd route runs the per-lane engine loop (emitting
        // chunk frames); a spec-less route takes the preserved batched raw path
        // (chunked through the model's chunk corner when a horizon was pinned).
        let short_chunk_warned = Arc::clone(&self.short_chunk_warned);
        tokio::task::spawn_blocking(move || match entry {
            Some(entry) => predict_route(
                &entry,
                &predict,
                observation,
                &adapter_ns,
                &short_chunk_warned,
            ),
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
        let adapter_ns = Arc::clone(&self.adapter_ns);
        let group_count = observations.len();
        let short_chunk_warned = Arc::clone(&self.short_chunk_warned);
        tokio::task::spawn_blocking(move || {
            predict_grouped_fused(
                lanes,
                observations,
                &predict,
                &adapter_ns,
                &short_chunk_warned,
            )
        })
        .await
        .unwrap_or_else(|err| {
            let message = format!("grouped predict task panicked: {err}");
            (0..group_count)
                .map(|_| Err(Error::Internal(message.clone())))
                .collect()
        })
    }

    fn take_adapter_ns(&mut self) -> u64 {
        self.adapter_ns.swap(0, Ordering::Relaxed)
    }

    fn held_state(&self) -> Option<HeldState> {
        Some(HeldState {
            episodes: self.held.episodes.load(Ordering::Relaxed),
            bytes: self.held.bytes.load(Ordering::Relaxed),
        })
    }

    fn route_setup(&self) -> Option<Arc<dyn ModelRouteSetup>> {
        let resolver = self.resolver.clone()?;
        Some(Arc::new(AdaptedRouteSetup {
            resolver,
            routes: Arc::clone(&self.routes),
            predict: Arc::clone(&self.predict),
            spec_less_horizons: Arc::clone(&self.spec_less_horizons),
            held: Arc::clone(&self.held),
            history_limit: self.history_limit,
        }))
    }

    async fn reset_adapter(&mut self, _env_id: &str, episode_ids: Vec<String>) -> Result<()> {
        // The frame buffers were already evicted off the predict lock by
        // `AdaptedRouteSetup::reset_adapter`. What is left is the model's own
        // hook, which shares the model's state with its forwards and so stays
        // serialized behind them here.
        //
        // Surface the episode-end edge to the model's own hook (one call per
        // ended episode), e.g. to reset a single-env model's recurrent state. An
        // empty `episode_ids` is an evict-ALL/teardown, not an episode end, so it
        // fires nothing here — model shutdown is `on_close`'s job.
        if episode_ids.is_empty() {
            return Ok(());
        }
        let predict = Arc::clone(&self.predict);
        tokio::task::spawn_blocking(move || {
            for episode_id in &episode_ids {
                predict.on_episode_end(episode_id)?;
            }
            Ok(())
        })
        .await
        .map_err(|err| Error::Internal(format!("on_episode_end task panicked: {err}")))?
    }

    async fn on_close(&mut self) -> Result<()> {
        // Drop every route's per-episode state as the authoritative shutdown sweep.
        // Snapshot the entries and release the map lock first: an entry lock
        // may be held across a blocking forward, and the map lock must stay
        // free for the route lookups (reset/release) racing this close.
        let entries: Vec<Arc<Mutex<RouteEntry>>> = self
            .routes
            .lock()
            .expect("routes map poisoned")
            .values()
            .cloned()
            .collect();
        for entry in entries {
            let mut guard = entry.lock().expect("route entry poisoned");
            guard.buffers.clear();
            guard.publish_held();
        }
        let predict = Arc::clone(&self.predict);
        tokio::task::spawn_blocking(move || predict.on_close())
            .await
            .map_err(|err| Error::Internal(format!("on_close task panicked: {err}")))?
    }
}

/// The resolve-time horizon doors, shared by the spec'd and spec-less branches.
///
/// Three configuration errors, each of which would otherwise only show up as a
/// wrong-length action chunk on every predict of the run:
///
/// 1. a horizon past [`MAX_EXECUTION_HORIZON`] — always a mis-set knob;
/// 2. a declared native chunk on a model with no chunk corner — the declaration
///    could never be honored, and nothing would ever check it;
/// 3. `execution_horizon > native_chunk` — the runtime would replay frames the
///    model does not produce.
///
/// There is deliberately NO lane check here: `ResolveAdapterRequest` carries no
/// `num_envs` (the served path decodes it as 0), so the engine cannot see lanes.
/// The lane guards live where the count is real — `run_local`, `RemoteModel`,
/// and the managed runner's route connect.
fn check_horizon_doors(
    execution_horizon: u32,
    native_chunk: Option<u32>,
    predict: &dyn PredictFn,
) -> Result<()> {
    if execution_horizon > MAX_EXECUTION_HORIZON {
        return Err(Error::model(format!(
            "runtime pinned execution_horizon={execution_horizon}, over the \
             {MAX_EXECUTION_HORIZON} bound"
        )));
    }
    let Some(native) = native_chunk else {
        return Ok(());
    };
    if !predict.has_chunk() && !predict.has_chunk_batch() {
        return Err(Error::model(format!(
            "model declares native_chunk={native} but defines no chunk corner \
             (predict_chunk / predict_chunk_batch); drop the declaration or add the corner"
        )));
    }
    if execution_horizon > native {
        return Err(Error::model(format!(
            "runtime pinned execution_horizon={execution_horizon} but the model declares \
             native_chunk={native}: it cannot produce that many actions per predict. Lower \
             the horizon to at most {native}"
        )));
    }
    Ok(())
}

/// The [`ModelRouteSetup`] the engine returns: resolves a route's config off the
/// predict lock and caches it for predict to read. A `None` resolution is a
/// spec-less route, left absent so predict takes the spec-less branch.
struct AdaptedRouteSetup {
    resolver: Arc<dyn RouteResolver>,
    routes: Routes,
    predict: Arc<dyn PredictFn>,
    spec_less_horizons: SpecLessHorizons,
    held: Arc<HeldCells>,
    history_limit: u64,
}

#[async_trait]
impl ModelRouteSetup for AdaptedRouteSetup {
    async fn resolve_adapter(
        &self,
        env_id: &str,
        env_contract: &EnvContract,
        options: ResolveOptions,
    ) -> Result<RouteNeeds> {
        let execution_horizon = options.execution_horizon;
        let native_chunk = self.predict.native_chunk();
        let mut needs = RouteNeeds {
            native_chunk,
            history: None,
        };
        // The horizon doors, before any resolution work: a pin this side of the
        // bound, a declaration the model can actually honor, and a horizon the
        // declared chunk covers. All three are configuration errors that would
        // otherwise surface as a mis-shaped action on every predict.
        check_horizon_doors(execution_horizon, native_chunk, self.predict.as_ref())?;
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
            return Ok(needs);
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
        // A stacked adapter keeps a frame window that has to see EVERY env step,
        // and one reading its previous action needs every executed action
        // numbered. At execution_horizon 1 every step is a predict, so both
        // are fed by construction. Above 1 the runtime must deliver the replayed
        // steps as history rows: a runtime that offered to gets the route's
        // needs back (and the engine then holds it to consecutive steps); one
        // that did not would leave the window holding decision points only, so
        // refuse.
        let history_keys = config.adapter.history_keys();
        let lane_bytes = lane_window_bytes(&config.adapter.history_windows());
        if !history_keys.is_empty() {
            // Per-route: the lanes the contract names (one on the served path,
            // which carries none). The endpoint total is admitted per fresh
            // lane at predict, where the lane count is real.
            check_history_budget(
                env_contract.num_envs.max(1),
                &config.adapter.history_windows(),
                self.history_limit,
            )?;
            if options.delivers_history {
                config.delivers_history = true;
                needs.history = Some(HistoryNeeds {
                    keys: history_keys,
                    prunable: false,
                });
            } else if config.execution_horizon > 1 {
                let what = match config.adapter.stacks().into_iter().next() {
                    Some((key, depth)) => format!("frame-stacking (input '{key}' stack={depth})"),
                    None => format!("a previous-action part (input '{}')", history_keys[0]),
                };
                return Err(Error::model(format!(
                    "{what} needs every env step, but the runtime did not offer observation \
                     history (delivers_history) at execution_horizon={}: the window would hold \
                     only decision-point frames. Use a runtime that delivers history, stack=1 \
                     (or no Previous part), or execution_horizon=1.",
                    config.execution_horizon,
                )));
            }
        }
        let entry = Arc::new(Mutex::new(RouteEntry {
            config: Arc::new(config),
            buffers: FrameBuffers::new(),
            total: Arc::clone(&self.held),
            published: HeldTotals::default(),
            lane_bytes,
            history_limit: self.history_limit,
        }));
        // Re-resolving a live route drops the old entry, which takes its
        // published share back out of the total.
        self.routes
            .lock()
            .expect("routes map poisoned")
            .insert(env_id.to_string(), entry);
        Ok(needs)
    }

    async fn reset_adapter(&self, env_id: &str, episode_ids: &[String]) -> Result<()> {
        // Explicit GC (R2): evict the ended episodes' frame buffers on this env's
        // adapter. Buffers are lazy-seeded on each episode's first predict (via
        // `assemble_obs`), so there is no seed step and no position-diffing. Empty
        // `episode_ids` evicts ALL of this env's episode state. Route-local, so
        // it runs off the predict lock: the entry lock is the only serialization
        // it needs, and per-env ordering keeps it clear of this env's predicts.
        let entry = self
            .routes
            .lock()
            .expect("routes map poisoned")
            .get(env_id)
            .cloned();
        if let Some(entry) = entry {
            let mut guard = entry.lock().expect("route entry poisoned");
            if episode_ids.is_empty() {
                guard.buffers.clear();
            } else {
                for episode_id in episode_ids {
                    guard.buffers.evict(episode_id);
                }
            }
            guard.publish_held();
        }
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
        /// The row-aligned episode ids each batched-corner call was handed.
        batched_ids: Mutex<Vec<Vec<String>>>,
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
                batched_ids: Mutex::new(Vec::new()),
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
        fn predict(&self, _model_input: Value, _episode: Option<&EpisodeInfo>) -> Result<Value> {
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

        fn predict_chunk(
            &self,
            _model_input: Value,
            _horizon: u32,
            _episode: Option<&EpisodeInfo>,
        ) -> Result<Option<Value>> {
            self.chunk_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Some(self.native_chunk_value()))
        }

        fn has_batch(&self) -> bool {
            self.batch
        }

        fn predict_batch(
            &self,
            inputs: Vec<Value>,
            episodes: &[EpisodeInfo],
        ) -> Result<Vec<Value>> {
            self.batch_calls.fetch_add(1, Ordering::SeqCst);
            self.batched_ids
                .lock()
                .expect("batched ids poisoned")
                .push(episodes.iter().map(|e| e.episode_id.clone()).collect());
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
            episodes: &[EpisodeInfo],
        ) -> Result<Vec<Value>> {
            self.chunk_batch_calls.fetch_add(1, Ordering::SeqCst);
            self.batched_ids
                .lock()
                .expect("batched ids poisoned")
                .push(episodes.iter().map(|e| e.episode_id.clone()).collect());
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

    /// Row-aligned episode identity for `count` lanes: `ep0..ep{count-1}`.
    fn episodes(count: usize) -> Vec<EpisodeInfo> {
        (0..count)
            .map(|lane| EpisodeInfo {
                episode_id: format!("ep{lane}"),
                seed: Some(lane as i64),
            })
            .collect()
    }

    #[test]
    fn dispatch_prefers_chunk_batch_and_caps_to_horizon() {
        let counting = CountingPredict::new(true, false, true, 10, false);
        let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

        let frames = dispatch_route_corners(
            &predict,
            lanes(5),
            &episodes(5),
            4,
            5,
            &AtomicBool::new(false),
        )
        .expect("frames");

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

        let frames = dispatch_route_corners(
            &predict,
            lanes(3),
            &episodes(3),
            1,
            3,
            &AtomicBool::new(false),
        )
        .expect("frames");

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

        let frames = dispatch_route_corners(
            &predict,
            lanes(3),
            &episodes(3),
            4,
            3,
            &AtomicBool::new(false),
        )
        .expect("frames");

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

        let error = dispatch_route_corners(
            &predict,
            lanes(4),
            &episodes(4),
            4,
            4,
            &AtomicBool::new(false),
        )
        .expect_err("short fails");

        assert!(
            error.to_string().contains("lanes"),
            "unexpected error: {error}"
        );
    }

    /// The batched corners see the SAME row-aligned episode identity the per-lane
    /// ones do: a fused batch is N independent episodes, so identity is per row.
    #[test]
    fn batched_corners_receive_row_aligned_episodes() {
        let counting = CountingPredict::new(true, false, true, 10, false);
        let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

        dispatch_route_corners(
            &predict,
            lanes(3),
            &episodes(3),
            4,
            3,
            &AtomicBool::new(false),
        )
        .expect("chunk batch");
        dispatch_route_corners(
            &predict,
            lanes(2),
            &episodes(2),
            1,
            2,
            &AtomicBool::new(false),
        )
        .expect("batch");

        assert_eq!(
            *counting.batched_ids.lock().expect("batched ids poisoned"),
            vec![
                vec!["ep0".to_string(), "ep1".to_string(), "ep2".to_string()],
                vec!["ep0".to_string(), "ep1".to_string()],
            ]
        );
    }

    /// A request whose episode rows do not match its lanes is malformed: fail
    /// rather than mis-attribute a lane's state to another episode.
    #[test]
    fn dispatch_rejects_episode_rows_that_do_not_match_the_lanes() {
        let counting = CountingPredict::new(true, false, false, 10, false);
        let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

        let error = dispatch_route_corners(
            &predict,
            lanes(3),
            &episodes(2),
            1,
            3,
            &AtomicBool::new(false),
        )
        .expect_err("row-count mismatch fails");

        assert!(
            error.to_string().contains("episode rows"),
            "unexpected error: {error}"
        );
        assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 0);
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
                history: Vec::new(),
                step: None,
                observation: None,
                route: ModelRouteContext {
                    env_id: format!("env-{index}"),
                    episodes: vec![EpisodeInfo {
                        episode_id: format!("ep-{index}"),
                        seed: None,
                    }],
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
        has_batch: bool,
        predict_calls: AtomicUsize,
        chunk_calls: AtomicUsize,
        batch_calls: AtomicUsize,
        episodes_seen: std::sync::Mutex<Vec<(String, Option<i64>)>>,
    }

    const CHUNK_FRAMES: usize = 6;

    impl EchoModel {
        fn new(chunk: bool, fail_recoverable: bool) -> Arc<Self> {
            Arc::new(Self {
                chunk,
                fail_recoverable,
                has_batch: true,
                predict_calls: AtomicUsize::new(0),
                chunk_calls: AtomicUsize::new(0),
                batch_calls: AtomicUsize::new(0),
                episodes_seen: std::sync::Mutex::new(Vec::new()),
            })
        }

        /// A single-lane model (no batched corner, so the engine always runs
        /// the per-lane `predict()` loop), for asserting real episode identity
        /// reaches it.
        fn new_single_lane() -> Arc<Self> {
            Arc::new(Self {
                chunk: false,
                fail_recoverable: false,
                has_batch: false,
                predict_calls: AtomicUsize::new(0),
                chunk_calls: AtomicUsize::new(0),
                batch_calls: AtomicUsize::new(0),
                episodes_seen: std::sync::Mutex::new(Vec::new()),
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
        fn predict(&self, model_input: Value, episode: Option<&EpisodeInfo>) -> Result<Value> {
            self.predict_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(episode) = episode {
                self.episodes_seen
                    .lock()
                    .expect("episodes_seen poisoned")
                    .push((episode.episode_id.clone(), episode.seed));
            }
            Ok(action_value(state_number(&model_input)))
        }

        fn predict_spec_less(&self, _observation: ModelObservation) -> Result<Vec<SpaceValue>> {
            Err(Error::model("EchoModel serves spec'd routes only"))
        }

        fn has_chunk(&self) -> bool {
            self.chunk
        }

        fn predict_chunk(
            &self,
            model_input: Value,
            _horizon: u32,
            _episode: Option<&EpisodeInfo>,
        ) -> Result<Option<Value>> {
            self.chunk_calls.fetch_add(1, Ordering::SeqCst);
            let state = state_number(&model_input);
            Ok(Some(Value::List(
                (0..CHUNK_FRAMES)
                    .map(|frame| action_value(state + 0.125 * frame as f64))
                    .collect(),
            )))
        }

        fn has_batch(&self) -> bool {
            self.has_batch
        }

        fn predict_batch(
            &self,
            inputs: Vec<Value>,
            episodes: &[EpisodeInfo],
        ) -> Result<Vec<Value>> {
            self.batch_calls.fetch_add(1, Ordering::SeqCst);
            let mut seen = self.episodes_seen.lock().expect("episodes_seen poisoned");
            seen.extend(
                episodes
                    .iter()
                    .map(|episode| (episode.episode_id.clone(), episode.seed)),
            );
            drop(seen);
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
            history: Vec::new(),
            step: None,
            observation: Some(wire.leaves),
            route: ModelRouteContext {
                env_id: env_id.to_string(),
                episodes: values
                    .iter()
                    .map(|value| EpisodeInfo {
                        episode_id: format!("{env_id}-ep-{value}"),
                        seed: None,
                    })
                    .collect(),
                ..Default::default()
            },
            num_envs: values.len(),
            env_contract: Some(Arc::clone(env_contract)),
        }
    }

    /// A single-lane observation for `env_id` carrying real episode id/seed
    /// metadata, unlike `grouped_obs` (which leaves `episode_seeds` default).
    fn single_obs_with_episode(
        env_id: &str,
        value: f32,
        episode_id: &str,
        seed: Option<i64>,
        env_contract: &Arc<spaces::EnvContract>,
    ) -> ModelObservation {
        let lane = SpaceValue::Dict(BTreeMap::from([("state".to_string(), box_f32(value))]));
        let wire = rlmesh_grpc::wire::encode_batched_partial_values(&[lane], &obs_space()).unwrap();
        ModelObservation {
            history: Vec::new(),
            step: None,
            observation: Some(wire.leaves),
            route: ModelRouteContext {
                env_id: env_id.to_string(),
                episodes: vec![EpisodeInfo {
                    episode_id: episode_id.to_string(),
                    seed,
                }],
                ..Default::default()
            },
            num_envs: 1,
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
                .resolve_adapter(
                    env_id,
                    &env_contract,
                    ResolveOptions {
                        execution_horizon: horizon,
                        delivers_history: false,
                    },
                )
                .await
                .expect("route resolves");
            contracts.push(Arc::new(env_contract));
        }
        (handler, contracts)
    }

    #[tokio::test]
    async fn direct_predict_carries_real_episode_id_and_seed() {
        let echo = EchoModel::new_single_lane();
        let (mut handler, contracts) =
            spec_handler(Arc::clone(&echo) as Arc<dyn PredictFn>, &[("env-a", 1)], 1).await;

        handler
            .predict(single_obs_with_episode(
                "env-a",
                1.0,
                "ep-linear",
                Some(42),
                &contracts[0],
            ))
            .await
            .expect("predict succeeds");

        assert_eq!(
            echo.episodes_seen.lock().expect("poisoned").as_slice(),
            [("ep-linear".to_string(), Some(42))],
        );
    }

    #[tokio::test]
    async fn grouped_predict_concatenates_row_aligned_episode_identity() {
        // Grouped predict serves multiple routes' episodes in ONE batched
        // forward. There is no single episode identity for the call, but every
        // ROW has one: the fused batch concatenates each group's inputs and its
        // episodes in the same order, so the batched corner can key per-episode
        // state by row.
        let echo = EchoModel::new(false, false);
        let (mut handler, contracts) = spec_handler(
            Arc::clone(&echo) as Arc<dyn PredictFn>,
            &[("env-a", 1), ("env-b", 1)],
            1,
        )
        .await;

        let results = handler
            .predict_grouped(vec![
                grouped_obs("env-a", &[1.0], &contracts[0]),
                grouped_obs("env-b", &[2.0], &contracts[1]),
            ])
            .await;

        assert!(results.iter().all(Result::is_ok));
        assert_eq!(echo.batch_calls.load(Ordering::SeqCst), 1, "fusion ran");
        assert_eq!(
            echo.predict_calls.load(Ordering::SeqCst),
            0,
            "the per-lane predict() corner never runs when fused"
        );
        assert_eq!(
            echo.episodes_seen.lock().expect("poisoned").as_slice(),
            [
                ("env-a-ep-1".to_string(), None),
                ("env-b-ep-2".to_string(), None),
            ],
            "rows arrive in group order, aligned with the concatenated inputs"
        );
    }

    #[tokio::test]
    async fn grouped_non_fused_predict_carries_each_routes_episode_identity() {
        let echo = EchoModel::new_single_lane();
        let (mut handler, contracts) = spec_handler(
            Arc::clone(&echo) as Arc<dyn PredictFn>,
            &[("env-a", 1), ("env-b", 1)],
            1,
        )
        .await;

        let results = handler
            .predict_grouped(vec![
                single_obs_with_episode("env-a", 1.0, "ep-a", Some(7), &contracts[0]),
                single_obs_with_episode("env-b", 2.0, "ep-b", None, &contracts[1]),
            ])
            .await;

        assert!(results.iter().all(Result::is_ok));
        assert_eq!(echo.batch_calls.load(Ordering::SeqCst), 0, "must not fuse");
        assert_eq!(
            echo.episodes_seen.lock().expect("poisoned").as_slice(),
            [("ep-a".to_string(), Some(7)), ("ep-b".to_string(), None)],
        );
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
    /// A wide fused batch — one route per lane, the fleet shape — assembles and
    /// finishes its groups on the rayon pool; every result must still land at
    /// its own group's index with its own lane's state, across the chunked and
    /// un-chunked corners.
    #[tokio::test]
    async fn wide_grouped_predict_keeps_every_group_in_request_order() {
        for chunk in [false, true] {
            let echo = EchoModel::new(chunk, false);
            let names: Vec<String> = (0..48).map(|i| format!("env-{i}")).collect();
            let specs: Vec<(&str, u32)> = names.iter().map(|n| (n.as_str(), 1)).collect();
            let horizon = if chunk { 3 } else { 1 };
            let (mut handler, contracts) =
                spec_handler(Arc::clone(&echo) as Arc<dyn PredictFn>, &specs, horizon).await;
            for round in 0..5 {
                let offset = 100.0 * round as f32;
                let observations = (0..48)
                    .map(|i| grouped_obs(&names[i], &[offset + i as f32], &contracts[i]))
                    .collect();
                let results = handler.predict_grouped(observations).await;
                assert_eq!(results.len(), 48);
                for (i, result) in results.iter().enumerate() {
                    let frames = result.as_ref().expect("every group serves");
                    let state = offset + i as f32;
                    assert_eq!(frames.actions, vec![box_f32(state)], "group {i}");
                    if chunk {
                        assert_eq!(frames.replay.len(), 2, "group {i} replays horizon-1 frames");
                        for (k, frame) in frames.replay.iter().enumerate() {
                            assert_eq!(
                                frame,
                                &vec![box_f32((f64::from(state) + 0.125 * (k + 1) as f64) as f32)]
                            );
                        }
                    }
                }
            }
            assert!(
                echo.batch_calls.load(Ordering::SeqCst) + echo.chunk_calls.load(Ordering::SeqCst)
                    > 0
            );
        }
    }

    /// The camera route (the SmolVLA-on-LIBERO shape): one 256x256x3 uint8 frame
    /// tagged upside-down, consumed as normalised float32 CHW, plus a gripper state.
    const IMAGE_ENV_TAGS: &str = r#"{
        "observation": {"cam": {"type": "image", "role": "image/primary", "upside_down": true},
                        "state": {"type": "state", "role": "proprio/gripper"}},
        "action": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;
    const IMAGE_MODEL_SPEC: &str = r#"{
        "input": {"cam": {"type": "image", "role": "image/primary", "height": 256, "width": 256, "fit": "pad",
                          "layout": "chw", "dtype": "float32", "normalize": true},
                  "state": {"type": "state", "container": "list", "dtype": "float32",
                            "components": [{"role": "proprio/gripper", "dim": 1}]}},
        "output": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;

    struct ImageTagResolver;

    #[async_trait]
    impl RouteResolver for ImageTagResolver {
        async fn resolve(
            &self,
            _route_key: &str,
            env_contract: &EnvContract,
        ) -> Result<Option<RouteConfig>> {
            let tags: EnvTags = serde_json::from_str(IMAGE_ENV_TAGS).expect("image env tags parse");
            let spec: ModelSpec =
                serde_json::from_str(IMAGE_MODEL_SPEC).expect("image model spec parse");
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

    fn image_obs_space() -> spaces::SpaceSpec {
        spaces::spaces::DictSpaceBuilder::new()
            .insert(
                "cam",
                spaces::spaces::BoxSpaceBuilder::scalar(0.0, 255.0, vec![256, 256, 3])
                    .dtype(DType::Uint8)
                    .build()
                    .unwrap(),
            )
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

    fn image_contract(env_id: &str) -> spaces::EnvContract {
        spaces::EnvContract {
            id: env_id.to_string(),
            observation_space: Some(image_obs_space()),
            action_space: Some(action_space()),
            metadata: None,
            render_mode: String::new(),
            num_envs: 1,
            autoreset_mode: Default::default(),
        }
    }

    fn grouped_image_obs(env_id: &str, seed: u32, value: f32) -> ModelObservation {
        let mut state = seed | 1;
        let pixels: Vec<u8> = (0..256 * 256 * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state & 0xff) as u8
            })
            .collect();
        let lane = SpaceValue::Dict(BTreeMap::from([
            (
                "cam".to_string(),
                SpaceValue::Box(Tensor::from_vec(pixels, vec![256, 256, 3], DType::Uint8).unwrap()),
            ),
            ("state".to_string(), box_f32(value)),
        ]));
        let wire =
            rlmesh_grpc::wire::encode_batched_partial_values(&[lane], &image_obs_space()).unwrap();
        ModelObservation {
            history: Vec::new(),
            step: None,
            observation: Some(wire.leaves),
            route: ModelRouteContext {
                session_id: "session-1".to_string(),
                request_id: format!("req-{env_id}-{seed}"),
                env_id: env_id.to_string(),
                episodes: vec![EpisodeInfo {
                    episode_id: format!("ep-{env_id}"),
                    seed: None,
                }],
            },
            num_envs: 1,
            env_contract: Some(Arc::new(image_contract(env_id))),
        }
    }

    /// `cargo test -p rlmesh --release camera_grouped_predict_timing -- --ignored --nocapture`
    /// (and again with `RAYON_NUM_THREADS=1`): wall time of one fused grouped predict
    /// over 32 camera routes and the adapter span it reports, so the lane fan-out's
    /// gain is a number rather than a projection.
    #[tokio::test]
    #[ignore = "timing, not correctness: run with --release --nocapture"]
    #[allow(clippy::print_stdout, reason = "the point of the test is the report")]
    async fn camera_grouped_predict_timing() {
        let echo = EchoModel::new(true, false);
        let handler_predict = Arc::clone(&echo) as Arc<dyn PredictFn>;
        let mut handler =
            AdaptedModelHandler::new(handler_predict, Some(Arc::new(ImageTagResolver)));
        let setup = handler.route_setup().expect("resolver-backed route setup");
        let names: Vec<String> = (0..32).map(|i| format!("cam-{i}")).collect();
        for name in &names {
            setup
                .resolve_adapter(
                    name,
                    &image_contract(name),
                    ResolveOptions {
                        execution_horizon: 10,
                        delivers_history: false,
                    },
                )
                .await
                .expect("camera route resolves");
        }
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0);
        let pool = rayon::current_num_threads();
        for round in 0..6 {
            let observations: Vec<ModelObservation> = (0..32)
                .map(|i| grouped_image_obs(&names[i], 1000 * round + i as u32 + 1, i as f32))
                .collect();
            let started = std::time::Instant::now();
            let results = handler.predict_grouped(observations).await;
            let wall = started.elapsed();
            let adapter = handler.take_adapter_ns();
            assert!(
                results.iter().all(Result::is_ok),
                "every camera group serves"
            );
            println!(
                "round {round}: 32 camera groups in {:.1} ms wall, adapter span {:.1} ms (available_parallelism {threads}, rayon pool {pool})",
                wall.as_secs_f64() * 1e3,
                adapter as f64 / 1e6
            );
        }
    }

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

    /// A chunk model that emits `emit` frames per chunk call and optionally
    /// DECLARES a native chunk length, so the resolve doors and `take_prefix`
    /// can be driven against every combination of (declared, emitted, horizon).
    struct DeclaredChunkModel {
        emit: usize,
        declared: Option<u32>,
        has_chunk: bool,
    }

    impl DeclaredChunkModel {
        fn new(emit: usize, declared: Option<u32>) -> Arc<Self> {
            Arc::new(Self {
                emit,
                declared,
                has_chunk: true,
            })
        }

        /// Declares a chunk length but defines no chunk corner at all.
        fn cornerless(declared: u32) -> Arc<Self> {
            Arc::new(Self {
                emit: 0,
                declared: Some(declared),
                has_chunk: false,
            })
        }
    }

    impl PredictFn for DeclaredChunkModel {
        fn predict(&self, model_input: Value, _episode: Option<&EpisodeInfo>) -> Result<Value> {
            Ok(action_value(state_number(&model_input)))
        }

        fn predict_spec_less(&self, _observation: ModelObservation) -> Result<Vec<SpaceValue>> {
            Err(Error::model("DeclaredChunkModel serves spec'd routes only"))
        }

        fn has_chunk(&self) -> bool {
            self.has_chunk
        }

        fn predict_chunk(
            &self,
            model_input: Value,
            _horizon: u32,
            _episode: Option<&EpisodeInfo>,
        ) -> Result<Option<Value>> {
            let state = state_number(&model_input);
            Ok(Some(Value::List(
                (0..self.emit)
                    .map(|frame| action_value(state + 0.125 * frame as f64))
                    .collect(),
            )))
        }

        fn native_chunk(&self) -> Option<u32> {
            self.declared
        }
    }

    /// A resolver that declares every route spec-LESS, so the doors can be driven
    /// down the branch that never builds a `RouteConfig`.
    struct SpecLessResolver;

    #[async_trait]
    impl RouteResolver for SpecLessResolver {
        async fn resolve(
            &self,
            _route_key: &str,
            _env_contract: &EnvContract,
        ) -> Result<Option<RouteConfig>> {
            Ok(None)
        }
    }

    /// Resolve one route at `horizon` and hand back the setup's answer.
    async fn resolve_once(
        predict: Arc<dyn PredictFn>,
        resolver: Arc<dyn RouteResolver>,
        lanes: u32,
        horizon: u32,
    ) -> Result<RouteNeeds> {
        let handler = AdaptedModelHandler::new(predict, Some(resolver));
        let setup = handler.route_setup().expect("resolver-backed route setup");
        let env_contract = contract("env-h", lanes);
        setup
            .resolve_adapter(
                "env-h",
                &env_contract,
                ResolveOptions {
                    execution_horizon: horizon,
                    delivers_history: false,
                },
            )
            .await
    }

    /// Total per-step frames one predict emitted (frame 0 plus the replay tail).
    async fn frame_count(handler: &mut AdaptedModelHandler, obs: ModelObservation) -> usize {
        let frames = handler
            .predict_chunked(obs)
            .await
            .expect("predict succeeds");
        1 + frames.replay.len()
    }

    // 1. A horizon past the bound is a mis-set knob, refused before any work.
    #[tokio::test]
    async fn resolve_refuses_a_horizon_over_the_bound() {
        let error = resolve_once(
            DeclaredChunkModel::new(4, None),
            Arc::new(TagResolver),
            1,
            MAX_EXECUTION_HORIZON + 1,
        )
        .await
        .expect_err("over the bound");
        assert!(
            error
                .to_string()
                .contains(&MAX_EXECUTION_HORIZON.to_string()),
            "the bound should name itself: {error}"
        );
    }

    // 2. The bound itself is legal (it is a ceiling, not an exclusive limit).
    #[tokio::test]
    async fn resolve_accepts_the_horizon_bound_exactly() {
        resolve_once(
            DeclaredChunkModel::new(4, None),
            Arc::new(TagResolver),
            1,
            MAX_EXECUTION_HORIZON,
        )
        .await
        .expect("the bound itself resolves");
    }

    // 3. The same door guards the spec-LESS branch, which never builds a config.
    #[tokio::test]
    async fn the_horizon_doors_guard_the_spec_less_branch_too() {
        resolve_once(
            DeclaredChunkModel::new(4, Some(6)),
            Arc::new(SpecLessResolver),
            1,
            8,
        )
        .await
        .expect_err("a spec-less route is bound by the same doors");
    }

    // 4. Declaring K without a chunk corner could never be honored or checked.
    #[tokio::test]
    async fn resolve_refuses_a_declared_chunk_with_no_chunk_corner() {
        let error = resolve_once(
            DeclaredChunkModel::cornerless(6),
            Arc::new(TagResolver),
            1,
            1,
        )
        .await
        .expect_err("declared K, no corner");
        assert!(
            error.to_string().contains("no chunk corner"),
            "unexpected error: {error}"
        );
    }

    // 5. h > K asks for actions the model cannot produce.
    #[tokio::test]
    async fn resolve_refuses_a_horizon_over_the_declared_chunk() {
        let error = resolve_once(
            DeclaredChunkModel::new(6, Some(6)),
            Arc::new(TagResolver),
            1,
            8,
        )
        .await
        .expect_err("h > K");
        assert!(
            error.to_string().contains("native_chunk=6"),
            "unexpected error: {error}"
        );
    }

    // 6. A resolved route answers the declaration back to the runtime.
    #[tokio::test]
    async fn resolve_answers_the_declared_native_chunk() {
        let needs = resolve_once(
            DeclaredChunkModel::new(6, Some(6)),
            Arc::new(TagResolver),
            1,
            6,
        )
        .await
        .expect("h == K resolves");
        assert_eq!(needs.native_chunk, Some(6));
    }

    // 7. An undeclared model answers nothing: the elastic contract.
    #[tokio::test]
    async fn resolve_answers_none_when_undeclared() {
        let needs = resolve_once(
            DeclaredChunkModel::new(6, None),
            Arc::new(TagResolver),
            1,
            4,
        )
        .await
        .expect("undeclared resolves");
        assert_eq!(needs.native_chunk, None);
    }

    // 8. The served path sees no lanes: `ResolveAdapterRequest` carries no
    //    num_envs, so the engine must NOT try to guard lanes here. A vector
    //    contract with a chunking horizon resolves; the refusal lives in the
    //    layers that actually know the lane count.
    #[tokio::test]
    async fn served_resolve_sees_no_lanes() {
        resolve_once(
            DeclaredChunkModel::new(6, Some(6)),
            Arc::new(TagResolver),
            4,
            4,
        )
        .await
        .expect("the engine carries no lane check");
    }

    // 9. Declared K, whole chunk returned, h < K: the runtime takes the prefix.
    #[tokio::test]
    async fn a_declared_model_replays_the_horizon_prefix_of_its_whole_chunk() {
        let model = DeclaredChunkModel::new(6, Some(6));
        let (mut handler, contracts) =
            spec_handler(model as Arc<dyn PredictFn>, &[("env-a", 1)], 4).await;
        let obs = single_obs_with_episode("env-a", 1.0, "ep", None, &contracts[0]);
        assert_eq!(frame_count(&mut handler, obs).await, 4);
    }

    // 10. Declared K at h == K: the whole chunk is executed.
    #[tokio::test]
    async fn a_declared_model_at_the_full_horizon_replays_every_frame() {
        let model = DeclaredChunkModel::new(6, Some(6));
        let (mut handler, contracts) =
            spec_handler(model as Arc<dyn PredictFn>, &[("env-a", 1)], 6).await;
        let obs = single_obs_with_episode("env-a", 1.0, "ep", None, &contracts[0]);
        assert_eq!(frame_count(&mut handler, obs).await, 6);
    }

    // 11. Declared K but the model still slices to the horizon (the glue this
    //     contract removes): the predict fails instead of short-replaying.
    #[tokio::test]
    async fn a_declared_model_that_slices_its_own_chunk_fails_the_predict() {
        let model = DeclaredChunkModel::new(4, Some(6));
        let (mut handler, contracts) =
            spec_handler(model as Arc<dyn PredictFn>, &[("env-a", 1)], 4).await;
        let obs = single_obs_with_episode("env-a", 1.0, "ep", None, &contracts[0]);
        let error = handler
            .predict_chunked(obs)
            .await
            .expect_err("a declared K is exact");
        let message = error.to_string();
        assert!(
            message.contains("native_chunk=6") && message.contains("4"),
            "the error should name both lengths: {message}"
        );
    }

    // 12. Undeclared and short: elastic, so the run continues on what there is.
    #[tokio::test]
    async fn an_undeclared_short_chunk_replays_what_there_is() {
        let model = DeclaredChunkModel::new(3, None);
        let (mut handler, contracts) =
            spec_handler(model as Arc<dyn PredictFn>, &[("env-a", 1)], 6).await;
        let obs = single_obs_with_episode("env-a", 1.0, "ep", None, &contracts[0]);
        assert_eq!(frame_count(&mut handler, obs).await, 3);
    }

    // 13. That short-chunk warning is latched once per endpoint, not per predict.
    #[test]
    fn the_short_chunk_warning_latches_once_per_endpoint() {
        let warned = AtomicBool::new(false);
        let frames = || vec![Value::Number(0.0), Value::Number(1.0)];
        assert_eq!(
            take_prefix(frames(), 6, None, &warned)
                .expect("elastic")
                .len(),
            2
        );
        assert!(
            warned.load(Ordering::Relaxed),
            "the first short chunk warns"
        );
        assert_eq!(
            take_prefix(frames(), 6, None, &warned)
                .expect("elastic")
                .len(),
            2,
            "a latched flag must not change what is replayed"
        );
    }

    // ---- observation history (a stacked route fed replayed steps as rows) ----

    const STACK_ENV_TAGS: &str = r#"{
        "observation": {"cam": {"type": "image", "role": "image/primary", "layout": "hwc"}},
        "action": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;
    const STACK_MODEL_SPEC: &str = r#"{
        "input": {"type": "image", "role": "image/primary", "layout": "hwc", "dtype": "uint8",
                  "stack": 3},
        "output": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;

    /// Resolves every route as one 1x1 RGB camera stacked 3 deep, so the
    /// engine's frame windows are real and every predict's input is the stack.
    struct StackResolver;

    #[async_trait]
    impl RouteResolver for StackResolver {
        async fn resolve(
            &self,
            _route_key: &str,
            env_contract: &EnvContract,
        ) -> Result<Option<RouteConfig>> {
            let tags: EnvTags = serde_json::from_str(STACK_ENV_TAGS).expect("env tags parse");
            let spec: ModelSpec = serde_json::from_str(STACK_MODEL_SPEC).expect("spec parse");
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

    fn cam_obs_space() -> spaces::SpaceSpec {
        spaces::spaces::DictSpaceBuilder::new()
            .insert(
                "cam",
                spaces::spaces::BoxSpaceBuilder::scalar(0.0, 255.0, vec![1, 1, 3])
                    .dtype(DType::Uint8)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap()
    }

    fn cam_contract(env_id: &str) -> Arc<spaces::EnvContract> {
        Arc::new(spaces::EnvContract {
            id: env_id.to_string(),
            observation_space: Some(cam_obs_space()),
            action_space: Some(action_space()),
            metadata: None,
            render_mode: String::new(),
            num_envs: 1,
            autoreset_mode: Default::default(),
        })
    }

    /// The wire leaves of a one-lane camera frame whose every byte is `tag`.
    fn cam_leaves(tag: u8) -> Vec<rlmesh_grpc::wire::Bytes> {
        let lane = SpaceValue::Dict(BTreeMap::from([(
            "cam".to_string(),
            SpaceValue::Box(Tensor::from_vec(vec![tag; 3], vec![1, 1, 3], DType::Uint8).unwrap()),
        )]));
        rlmesh_grpc::wire::encode_batched_partial_values(&[lane], &cam_obs_space())
            .unwrap()
            .leaves
    }

    fn episode(id: &str) -> Vec<EpisodeInfo> {
        vec![EpisodeInfo {
            episode_id: id.to_string(),
            seed: None,
        }]
    }

    /// A predict at `step` carrying the frame `tag`, preceded by `rows` of
    /// `(step, tag)` replayed steps.
    fn cam_predict(
        env_id: &str,
        episode_id: &str,
        step: Option<i64>,
        tag: u8,
        rows: &[(i64, u8)],
    ) -> ModelObservation {
        ModelObservation {
            observation: Some(cam_leaves(tag)),
            route: ModelRouteContext {
                env_id: env_id.to_string(),
                episodes: episode(episode_id),
                ..Default::default()
            },
            num_envs: 1,
            env_contract: Some(cam_contract(env_id)),
            history: rows
                .iter()
                .map(|&(step, tag)| crate::model::types::HistoryFrame {
                    observation: Some(cam_leaves(tag)),
                    episodes: episode(episode_id),
                    step,
                })
                .collect(),
            step,
        }
    }

    /// Records the stacked frame bytes each predict was handed.
    struct WindowRecorder {
        stacks: Mutex<Vec<Vec<u8>>>,
    }

    impl PredictFn for WindowRecorder {
        fn predict(&self, model_input: Value, _episode: Option<&EpisodeInfo>) -> Result<Value> {
            fn first_tensor(value: &Value) -> Option<Vec<u8>> {
                match value {
                    Value::Tensor(tensor) => Some(tensor.to_contiguous_bytes().into_owned()),
                    Value::Map(map) => map.values().find_map(first_tensor),
                    Value::List(items) => items.iter().find_map(first_tensor),
                    _ => None,
                }
            }
            self.stacks
                .lock()
                .expect("stacks poisoned")
                .push(first_tensor(&model_input).expect("a stacked tensor input"));
            Ok(Value::Number(1.0))
        }

        fn predict_spec_less(&self, observation: ModelObservation) -> Result<Vec<SpaceValue>> {
            Ok((0..observation.num_envs)
                .map(|_| SpaceValue::Discrete(0))
                .collect())
        }
    }

    /// A stacked handler resolved for `env_id`, with or without the runtime
    /// offering history, at `horizon`.
    async fn stacked_handler(
        env_id: &str,
        delivers_history: bool,
        horizon: u32,
    ) -> Result<(AdaptedModelHandler, Arc<WindowRecorder>, RouteNeeds)> {
        let recorder = Arc::new(WindowRecorder {
            stacks: Mutex::new(Vec::new()),
        });
        let handler = AdaptedModelHandler::new(
            Arc::clone(&recorder) as Arc<dyn PredictFn>,
            Some(Arc::new(StackResolver) as Arc<dyn RouteResolver>),
        );
        let needs = handler
            .route_setup()
            .expect("resolver-backed route setup")
            .resolve_adapter(
                env_id,
                &cam_contract(env_id),
                ResolveOptions {
                    execution_horizon: horizon,
                    delivers_history,
                },
            )
            .await?;
        Ok((handler, recorder, needs))
    }

    #[tokio::test]
    async fn a_stacked_route_needs_history_only_when_the_runtime_offers_it() {
        // Offered: the route answers its window keys and the horizon is fine.
        let (_, _, needs) = stacked_handler("env-offer", true, 4).await.unwrap();
        let history = needs.history.expect("a stacked route asks for history");
        assert_eq!(history.keys.len(), 1, "one stacked input: {history:?}");
        assert!(!history.prunable);

        // Not offered: above horizon 1 the window would hold decision points
        // only, so the route is refused; at horizon 1 every step predicts.
        let refused = stacked_handler("env-refuse", false, 4)
            .await
            .err()
            .expect("stack + chunk without history is refused")
            .to_string();
        assert!(
            refused.contains("did not offer observation history"),
            "{refused}"
        );
        let (_, _, needs) = stacked_handler("env-h1", false, 1).await.unwrap();
        assert!(needs.history.is_none());

        // An unstacked route never asks, offered or not.
        let handler = AdaptedModelHandler::new(
            EchoModel::new(false, false) as Arc<dyn PredictFn>,
            Some(Arc::new(TagResolver) as Arc<dyn RouteResolver>),
        );
        let needs = handler
            .route_setup()
            .unwrap()
            .resolve_adapter(
                "env-flat",
                &contract("env-flat", 1),
                ResolveOptions {
                    execution_horizon: 4,
                    delivers_history: true,
                },
            )
            .await
            .unwrap();
        assert!(needs.history.is_none());
    }

    #[tokio::test]
    async fn history_rows_advance_the_window_exactly_like_sequential_predicts() {
        // The invariant the wire exists for: a predict carrying the replayed
        // steps as rows hands the model the same stack a predict-every-step
        // route would have at that step.
        let (mut every_step, sequential, _) = stacked_handler("env-seq", true, 1).await.unwrap();
        for step in 0..8u8 {
            every_step
                .predict(cam_predict(
                    "env-seq",
                    "ep",
                    Some(i64::from(step)),
                    step,
                    &[],
                ))
                .await
                .expect("sequential predict");
        }
        let expected = sequential.stacks.lock().unwrap().clone();
        assert_eq!(expected.len(), 8);

        let (mut chunked, recorder, _) = stacked_handler("env-hist", true, 4).await.unwrap();
        // Predict at 0; steps 1..3 replay; predict at 4 with rows 1..3; steps
        // 5 and 6 replay; predict at 7 with rows 5 and 6.
        for (step, rows) in [
            (0i64, vec![]),
            (4, vec![(1, 1u8), (2, 2), (3, 3)]),
            (7, vec![(5, 5), (6, 6)]),
        ] {
            chunked
                .predict(cam_predict("env-hist", "ep", Some(step), step as u8, &rows))
                .await
                .expect("chunked predict");
        }
        let got = recorder.stacks.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![
                expected[0].clone(),
                expected[4].clone(),
                expected[7].clone()
            ]
        );
    }

    #[tokio::test]
    async fn held_state_is_an_endpoint_total_kept_by_route_deltas() {
        // Two stacked routes, each holding one episode's window.
        let (mut handler, _, _) = stacked_handler("env-a", true, 1).await.unwrap();
        let setup = handler.route_setup().expect("resolver-backed route setup");
        setup
            .resolve_adapter(
                "env-b",
                &cam_contract("env-b"),
                ResolveOptions {
                    execution_horizon: 1,
                    delivers_history: true,
                },
            )
            .await
            .unwrap();
        handler
            .predict(cam_predict("env-a", "ep-a", Some(0), 1, &[]))
            .await
            .unwrap();
        handler
            .predict(cam_predict("env-b", "ep-b", Some(0), 2, &[]))
            .await
            .unwrap();
        let both = handler
            .held_state()
            .expect("the engine accounts held state");
        assert_eq!(both.episodes, 2);
        assert!(both.bytes > 0, "{both:?}");

        // The total answers without the roster: read it from another thread
        // while this one holds the routes map lock.
        let handler = Arc::new(handler);
        let roster = handler.routes.lock().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let reader = Arc::clone(&handler);
        std::thread::spawn(move || tx.send(reader.held_state()));
        let read = rx
            .recv_timeout(std::time::Duration::from_millis(50))
            .expect("held_state must not take the routes map lock");
        drop(roster);
        assert_eq!(read, Some(both));

        // Evicting one route's episode, then releasing the other route, each
        // take exactly their share back out of the total.
        setup
            .reset_adapter("env-a", &["ep-a".to_string()])
            .await
            .unwrap();
        let one = handler.held_state().unwrap();
        assert_eq!(one.episodes, 1);
        assert!(
            one.bytes > 0 && one.bytes < both.bytes,
            "{one:?} vs {both:?}"
        );
        setup.release_adapter("env-b").await.unwrap();
        assert_eq!(handler.held_state().unwrap(), HeldState::default());
    }

    // ---- previous action (a route reading back its own executed frames) ----

    const PREVIOUS_ENV_TAGS: &str = r#"{
        "observation": {"g": {"type": "state", "role": "proprio/gripper"}},
        "action": {"components": [{"role": "action/joint_pos", "dim": 2}]}
    }"#;
    const PREVIOUS_MODEL_SPEC: &str = r#"{
        "input": {"type": "state", "components": ["proprio/gripper",
                  {"role": "action/joint_pos", "source": "action"}]},
        "output": {"components": [{"role": "action/joint_pos", "dim": 2}]}
    }"#;
    const PLAIN_MODEL_SPEC: &str = r#"{
        "input": {"type": "state", "components": ["proprio/gripper"]},
        "output": {"components": [{"role": "action/joint_pos", "dim": 2}]}
    }"#;

    /// Resolves every route as a scalar gripper reading plus, with
    /// [`PREVIOUS_MODEL_SPEC`], the model's own previous 2-d joint command.
    struct PreviousResolver(&'static str);

    #[async_trait]
    impl RouteResolver for PreviousResolver {
        async fn resolve(
            &self,
            _route_key: &str,
            env_contract: &EnvContract,
        ) -> Result<Option<RouteConfig>> {
            let tags: EnvTags = serde_json::from_str(PREVIOUS_ENV_TAGS).expect("env tags parse");
            let spec: ModelSpec = serde_json::from_str(self.0).expect("spec parse");
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

    fn g_obs_space() -> spaces::SpaceSpec {
        spaces::spaces::DictSpaceBuilder::new()
            .insert(
                "g",
                spaces::spaces::BoxSpaceBuilder::scalar(0.0, 100.0, vec![1])
                    .dtype(DType::Float32)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap()
    }

    fn g_contract(env_id: &str) -> Arc<spaces::EnvContract> {
        Arc::new(spaces::EnvContract {
            id: env_id.to_string(),
            observation_space: Some(g_obs_space()),
            action_space: Some(
                spaces::spaces::BoxSpaceBuilder::scalar(-1000.0, 1000.0, vec![2])
                    .dtype(DType::Float32)
                    .build()
                    .unwrap(),
            ),
            metadata: None,
            render_mode: String::new(),
            num_envs: 1,
            autoreset_mode: Default::default(),
        })
    }

    fn g_leaves(g: f32) -> Vec<rlmesh_grpc::wire::Bytes> {
        let lane = SpaceValue::Dict(BTreeMap::from([(
            "g".to_string(),
            SpaceValue::Box(
                Tensor::from_vec(g.to_le_bytes().to_vec(), vec![1], DType::Float32).unwrap(),
            ),
        )]));
        rlmesh_grpc::wire::encode_batched_partial_values(&[lane], &g_obs_space())
            .unwrap()
            .leaves
    }

    /// A predict at `step` reading `g`, preceded by `rows` of `(step, g)`
    /// replayed steps.
    fn g_predict(
        env_id: &str,
        episode_id: &str,
        step: Option<i64>,
        g: f32,
        rows: &[(i64, f32)],
    ) -> ModelObservation {
        ModelObservation {
            observation: Some(g_leaves(g)),
            route: ModelRouteContext {
                env_id: env_id.to_string(),
                episodes: episode(episode_id),
                ..Default::default()
            },
            num_envs: 1,
            env_contract: Some(g_contract(env_id)),
            history: rows
                .iter()
                .map(|&(step, g)| crate::model::types::HistoryFrame {
                    observation: Some(g_leaves(g)),
                    episodes: episode(episode_id),
                    step,
                })
                .collect(),
            step,
        }
    }

    /// Records the state each predict was handed and answers a chunk whose
    /// frame `k` of call `c` is `[100c + 10 + k, 100c + 20 + k]`, so every
    /// executed frame is telling.
    struct PreviousRecorder {
        inputs: Mutex<Vec<Vec<f32>>>,
        calls: AtomicUsize,
    }

    impl PreviousRecorder {
        fn record(&self, model_input: &Value) -> usize {
            let Value::Tensor(state) = model_input else {
                panic!("expected a bare state tensor, got {model_input:?}");
            };
            let values: Vec<f32> = state
                .to_contiguous_bytes()
                .chunks(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            self.inputs.lock().expect("inputs poisoned").push(values);
            self.calls.fetch_add(1, Ordering::SeqCst)
        }

        fn frame(call: usize, k: usize) -> Value {
            let (a, b) = ((100 * call + 10 + k) as f32, (100 * call + 20 + k) as f32);
            let mut bytes = a.to_le_bytes().to_vec();
            bytes.extend(b.to_le_bytes());
            Value::Tensor(Tensor::from_vec(bytes, vec![2], DType::Float32).unwrap())
        }
    }

    impl PredictFn for PreviousRecorder {
        fn predict(&self, model_input: Value, _episode: Option<&EpisodeInfo>) -> Result<Value> {
            let call = self.record(&model_input);
            Ok(Self::frame(call, 0))
        }

        fn has_chunk(&self) -> bool {
            true
        }

        fn predict_chunk(
            &self,
            model_input: Value,
            _horizon: u32,
            _episode: Option<&EpisodeInfo>,
        ) -> Result<Option<Value>> {
            let call = self.record(&model_input);
            Ok(Some(Value::List(
                (0..4).map(|k| Self::frame(call, k)).collect(),
            )))
        }

        fn predict_spec_less(&self, observation: ModelObservation) -> Result<Vec<SpaceValue>> {
            Ok((0..observation.num_envs)
                .map(|_| SpaceValue::Discrete(0))
                .collect())
        }
    }

    async fn previous_handler(
        env_id: &str,
        spec: &'static str,
        delivers_history: bool,
        horizon: u32,
    ) -> Result<(AdaptedModelHandler, Arc<PreviousRecorder>, RouteNeeds)> {
        let recorder = Arc::new(PreviousRecorder {
            inputs: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
        });
        let handler = AdaptedModelHandler::new(
            Arc::clone(&recorder) as Arc<dyn PredictFn>,
            Some(Arc::new(PreviousResolver(spec)) as Arc<dyn RouteResolver>),
        );
        let needs = handler
            .route_setup()
            .expect("resolver-backed route setup")
            .resolve_adapter(
                env_id,
                &g_contract(env_id),
                ResolveOptions {
                    execution_horizon: horizon,
                    delivers_history,
                },
            )
            .await?;
        Ok((handler, recorder, needs))
    }

    #[tokio::test]
    async fn a_previous_action_part_reads_the_last_replayed_frame_at_the_replan() {
        // execution_horizon 4: the predict at step 0 emits frames 0..3, the
        // runtime replays 1..3 and re-plans at 4 carrying their rows. The
        // policy then reads frame 3 (the action executed at step 3), not
        // frame 0 (the one it saw itself emit), and the fill before any.
        let (mut handler, recorder, needs) =
            previous_handler("env-prev", PREVIOUS_MODEL_SPEC, true, 4)
                .await
                .unwrap();
        let history = needs
            .history
            .expect("a previous-action route asks for history");
        assert_eq!(history.keys, vec!["<root>".to_string()]);
        let frames = handler
            .predict_chunked(g_predict("env-prev", "ep", Some(0), 1.0, &[]))
            .await
            .expect("first predict");
        assert_eq!(frames.replay.len(), 3);
        handler
            .predict_chunked(g_predict(
                "env-prev",
                "ep",
                Some(4),
                5.0,
                &[(1, 2.0), (2, 3.0), (3, 4.0)],
            ))
            .await
            .expect("re-plan");
        assert_eq!(
            recorder.inputs.lock().unwrap().clone(),
            vec![vec![1.0, 0.0, 0.0], vec![5.0, 13.0, 23.0]]
        );
    }

    #[tokio::test]
    async fn a_previous_action_part_under_a_prefetch_lead_reads_the_frame_that_executed() {
        // execution_horizon 4 with a prefetch lead of 1: the re-plan is
        // requested at step 3, one frame before the first chunk runs out, so
        // its own observation is step 3 but its frames execute from step 4.
        // The predict at step 7 must read call 1's frame 2 (executed at 6),
        // not frame 3, which the request's own step would have recorded there.
        let (mut handler, recorder, _) = previous_handler("env-lead", PREVIOUS_MODEL_SPEC, true, 4)
            .await
            .unwrap();
        handler
            .predict_chunked(g_predict("env-lead", "ep", Some(0), 1.0, &[]))
            .await
            .expect("first predict");
        handler
            .predict_chunked(g_predict(
                "env-lead",
                "ep",
                Some(3),
                4.0,
                &[(1, 2.0), (2, 3.0)],
            ))
            .await
            .expect("prefetched re-plan");
        handler
            .predict_chunked(g_predict(
                "env-lead",
                "ep",
                Some(7),
                8.0,
                &[(4, 5.0), (5, 6.0), (6, 7.0)],
            ))
            .await
            .expect("second prefetched re-plan");
        assert_eq!(
            recorder.inputs.lock().unwrap().clone(),
            vec![
                vec![1.0, 0.0, 0.0],
                vec![4.0, 12.0, 22.0],
                vec![8.0, 112.0, 122.0]
            ]
        );
    }

    #[tokio::test]
    async fn a_previous_action_route_counts_its_own_steps_when_none_are_stamped() {
        // No history offered at execution_horizon 1: every predict is one env
        // step, so the engine numbers them itself and each predict reads the
        // frame the one before it emitted.
        let (mut handler, recorder, needs) =
            previous_handler("env-count", PREVIOUS_MODEL_SPEC, false, 1)
                .await
                .unwrap();
        assert!(needs.history.is_none());
        for g in [1.0f32, 2.0, 3.0] {
            handler
                .predict_chunked(g_predict("env-count", "ep", None, g, &[]))
                .await
                .expect("predict");
        }
        assert_eq!(
            recorder.inputs.lock().unwrap().clone(),
            vec![
                vec![1.0, 0.0, 0.0],
                vec![2.0, 10.0, 20.0],
                vec![3.0, 110.0, 120.0]
            ]
        );
        // Above 1 without the offer the route is refused, like a stacked one.
        let refused = previous_handler("env-refuse", PREVIOUS_MODEL_SPEC, false, 4)
            .await
            .err()
            .expect("a chunked previous-action route without history is refused")
            .to_string();
        assert!(
            refused.contains("a previous-action part (input '<root>') needs every env step"),
            "{refused}"
        );
        // A route without the part negotiates nothing, offered or not, and
        // its predicts are the plain ones.
        let (mut plain, recorder, needs) = previous_handler("env-plain", PLAIN_MODEL_SPEC, true, 4)
            .await
            .unwrap();
        assert!(needs.history.is_none());
        plain
            .predict_chunked(g_predict("env-plain", "ep", None, 1.0, &[]))
            .await
            .expect("plain predict");
        assert_eq!(recorder.inputs.lock().unwrap().clone(), vec![vec![1.0]]);
    }

    #[tokio::test]
    async fn a_negotiated_route_rejects_a_step_gap_repeat_or_omission() {
        let (mut handler, _, _) = stacked_handler("env-steps", true, 4).await.unwrap();
        handler
            .predict(cam_predict("env-steps", "ep", Some(0), 0, &[]))
            .await
            .expect("first predict");

        let gap = handler
            .predict(cam_predict("env-steps", "ep", Some(2), 2, &[]))
            .await
            .unwrap_err()
            .to_string();
        assert!(gap.contains("not consecutive"), "{gap}");

        let repeat = handler
            .predict(cam_predict("env-steps", "ep", Some(1), 1, &[(0, 0)]))
            .await
            .unwrap_err()
            .to_string();
        assert!(repeat.contains("step 0 arrived after step 0"), "{repeat}");

        let omitted = handler
            .predict(cam_predict("env-steps", "ep", None, 1, &[]))
            .await
            .unwrap_err()
            .to_string();
        assert!(omitted.contains("carries no step"), "{omitted}");

        // A fresh episode starts its own sequence at any step.
        handler
            .predict(cam_predict("env-steps", "ep-2", Some(40), 0, &[]))
            .await
            .expect("new episode, any step");
    }

    #[tokio::test]
    async fn rows_sent_to_a_route_that_did_not_negotiate_history_are_refused() {
        let (mut handler, _, needs) = stacked_handler("env-none", false, 1).await.unwrap();
        assert!(needs.history.is_none());
        let refused = handler
            .predict(cam_predict("env-none", "ep", Some(1), 1, &[(0, 0)]))
            .await
            .unwrap_err()
            .to_string();
        assert!(refused.contains("did not negotiate history"), "{refused}");
    }

    #[test]
    fn the_frame_history_budget_refuses_a_window_that_outgrows_it() {
        let window = |frame_bytes: u64| HistoryWindow {
            key: "cam".to_string(),
            span: 4,
            frame_bytes,
        };
        let limit = FRAME_HISTORY_LIMIT_DEFAULT;
        check_history_budget(8, &[window(224 * 224 * 3)], limit).expect("a few MiB per lane");
        let refused = check_history_budget(8, &[window(1 << 30)], limit)
            .unwrap_err()
            .to_string();
        assert!(refused.contains(FRAME_HISTORY_LIMIT_ENV), "{refused}");
    }

    /// A predict for `env_id` at `step` carrying one frame per lane, each lane
    /// its own episode.
    fn cam_predict_lanes(env_id: &str, episode_ids: &[&str], step: i64) -> ModelObservation {
        let lanes: Vec<SpaceValue> = episode_ids
            .iter()
            .map(|_| {
                SpaceValue::Dict(BTreeMap::from([(
                    "cam".to_string(),
                    SpaceValue::Box(
                        Tensor::from_vec(vec![step as u8; 3], vec![1, 1, 3], DType::Uint8).unwrap(),
                    ),
                )]))
            })
            .collect();
        ModelObservation {
            observation: Some(
                rlmesh_grpc::wire::encode_batched_partial_values(&lanes, &cam_obs_space())
                    .unwrap()
                    .leaves,
            ),
            route: ModelRouteContext {
                env_id: env_id.to_string(),
                episodes: episode_ids
                    .iter()
                    .map(|id| EpisodeInfo {
                        episode_id: id.to_string(),
                        seed: None,
                    })
                    .collect(),
                ..Default::default()
            },
            num_envs: episode_ids.len(),
            env_contract: Some(cam_contract(env_id)),
            history: Vec::new(),
            step: Some(step),
        }
    }

    #[tokio::test]
    async fn fresh_lanes_are_admitted_against_the_endpoint_window_budget() {
        // The 1x1x3 uint8 camera stacked 3 deep: 9 bytes per lane once full.
        // Budget two lanes' worth for the whole endpoint.
        const LANE_BYTES: u64 = 9;
        let recorder = Arc::new(WindowRecorder {
            stacks: Mutex::new(Vec::new()),
        });
        let mut handler = AdaptedModelHandler::new(
            recorder as Arc<dyn PredictFn>,
            Some(Arc::new(StackResolver) as Arc<dyn RouteResolver>),
        );
        handler.history_limit = 2 * LANE_BYTES;
        let setup = handler.route_setup().expect("resolver-backed");
        for env_id in ["env-a", "env-b"] {
            // Each route fits on its own: the served resolve sees one lane.
            setup
                .resolve_adapter(
                    env_id,
                    &cam_contract(env_id),
                    ResolveOptions {
                        execution_horizon: 1,
                        delivers_history: true,
                    },
                )
                .await
                .expect("one lane fits");
        }
        assert_eq!(
            handler.entry("env-a").unwrap().lock().unwrap().lane_bytes,
            LANE_BYTES
        );

        // Many lanes at once: the multiplier is the real lane count, not the
        // resolve-time placeholder, and a refusal seeds nothing.
        let refused = handler
            .predict(cam_predict_lanes("env-a", &["a1", "a2", "a3"], 0))
            .await
            .unwrap_err()
            .to_string();
        assert!(refused.contains(FRAME_HISTORY_LIMIT_ENV), "{refused}");
        assert_eq!(handler.held_state(), Some(HeldState::default()));

        // Two lanes fill the budget (their windows are charged at full size
        // from the first frame), and revisiting held lanes is never charged.
        for step in 0..2 {
            handler
                .predict(cam_predict_lanes("env-a", &["a1", "a2"], step))
                .await
                .expect("two lanes fit");
        }
        // A second route that fits on its own is refused against the total.
        let refused = handler
            .predict(cam_predict_lanes("env-b", &["b1"], 0))
            .await
            .unwrap_err()
            .to_string();
        assert!(refused.contains("other routes"), "{refused}");

        // An ended episode gives its window back (the route setup evicts the
        // window; the handler's reset is only the model's episode-end hook).
        setup
            .reset_adapter("env-a", &["a2".to_string()])
            .await
            .unwrap();
        handler
            .predict(cam_predict_lanes("env-b", &["b1"], 0))
            .await
            .expect("room again after an episode ends");
    }
}
