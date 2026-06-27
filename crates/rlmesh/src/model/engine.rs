//! The vectorized stateful adapter engine: a [`ModelHandler`] that owns the
//! per-lane drive, episode-keyed frame buffers, and native adapter application
//! in pure Rust, calling back into a [`PredictFn`] only for the model's predict
//! and into the custom/encoding holes only where a route declares them.
//!
//! This replaces a binding's hand-rolled predict loop: a PyO3 (or any future
//! language) binding constructs `AdaptedModelHandler::new(predict, resolver)`
//! and serves it; a pure-Rust model does the same with no host runtime.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rlmesh_adapters::v1::{
    FrameBuffers, ObsPlan, Value, apply_actions, assemble_obs, space_value_to_obs_map, split_chunk,
};

use super::handler::{ModelHandler, ModelRouteSetup, PredictFrames};
use super::predict_fn::{PredictFn, RouteConfig, RouteResolver};
use super::types::{ModelEpisodeEnd, ModelLaneReset, ModelObservation, ModelRouteContext};
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

/// A served [`ModelHandler`] that drives the vectorized stateful adapter engine.
pub struct AdaptedModelHandler {
    predict: Arc<dyn PredictFn>,
    resolver: Option<Arc<dyn RouteResolver>>,
    routes: Routes,
    /// The route the server is currently processing, set by `enter_route` before
    /// its per-lane resets fire (the lane events carry no route key).
    current_route: Option<String>,
}

impl AdaptedModelHandler {
    /// Build a handler from the model's predict hole and an optional route
    /// resolver. `resolver = None` means a spec-less model: every route serves
    /// through [`PredictFn::predict_spec_less`].
    pub fn new(predict: Arc<dyn PredictFn>, resolver: Option<Arc<dyn RouteResolver>>) -> Self {
        Self {
            predict,
            resolver,
            routes: Arc::new(Mutex::new(HashMap::new())),
            current_route: None,
        }
    }

    /// Look up the per-route entry (cloning the `Arc`), if the route is spec'd.
    fn entry(&self, route_key: &str) -> Option<Arc<Mutex<RouteEntry>>> {
        self.routes
            .lock()
            .expect("routes map poisoned")
            .get(route_key)
            .cloned()
    }
}

fn route_key(route: &ModelRouteContext) -> String {
    format!("{}:{}", route.session_id, route.route_id)
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

/// The spec'd per-lane loop (CPU + the model's predict callback), run on a
/// blocking worker thread. Holds the per-route entry lock across the lanes so
/// the frame buffers mutate in place.
///
/// Emits each lane's action chunk as [`PredictFrames`]: frame 0 per lane plus the
/// future-step frames the runtime driver replays. With horizon 1 every lane
/// yields a single frame and `replay` is empty — the unchanged single-action
/// path. The engine no longer replays internally; it returns the whole chunk and
/// re-plans on every call (the runtime decides when to call it).
fn predict_route(
    entry: &Arc<Mutex<RouteEntry>>,
    predict: &Arc<dyn PredictFn>,
    observation: ModelObservation,
) -> Result<PredictFrames> {
    let episode_ids = observation.episode_ids();
    let num_envs = observation.num_envs;

    // The wire contract requires every predict request to carry a decodable
    // observation; validate the structure up front (cheap) so a malformed request
    // errors here. Every call re-plans (the runtime owns replay), so the obs is
    // always decoded below.
    observation.ensure_decodable()?;

    let mut guard = entry.lock().expect("route entry poisoned");
    let RouteEntry { config, buffers } = &mut *guard;
    let referenced = obs_keys(config);
    // Runtime-chosen replay horizon (pinned on ConfigureRoute), not the model spec.
    let horizon = config.action_horizon;
    let customs: &dyn rlmesh_adapters::v1::CustomTransform = config.customs.as_ref();
    let encodings: &dyn rlmesh_adapters::v1::EncodingTransform = config.encodings.as_ref();
    let has_chunk = predict.has_chunk();

    let decoded = observation.decoded_lanes()?;

    // Assemble each lane's model input (frame-stacking mutates the per-episode
    // buffers here, once per call). Every lane must carry a non-empty episode_id:
    // the engine keys all per-episode state (frame windows) by it. The grpc wire
    // layer already enforces `num_envs == slots.len()` with non-empty ids, but the
    // engine must not silently fall back to a shared "" buffer (which would
    // cross-contaminate lanes) if any other producer violates that.
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

    // Dispatch to the most-specific available corner, yielding each lane's raw
    // chunk frames (a single-element Vec when not chunking). Prefer a *batched*
    // corner (one forward for the whole vector) over the per-lane loop, and a
    // *chunk* corner when the runtime pinned a horizon > 1. A horizon > 1 with no
    // chunk corner was already warned at configure and falls through to a
    // single-action corner (the runtime then re-plans every step). split_chunk caps
    // each chunk to the horizon (a receding-horizon model may over-produce).
    let lane_raw_steps: Vec<Vec<Value>> = if horizon > 1 && predict.has_chunk_batch() {
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
            .collect::<Result<Vec<_>>>()?
    } else if horizon > 1 && has_chunk {
        inputs
            .into_iter()
            .map(|input| -> Result<Vec<Value>> {
                let chunk = predict.predict_chunk(input, horizon)?.ok_or_else(|| {
                    Error::model(
                        "model reports a chunk corner (has_chunk) but predict_chunk returned None",
                    )
                })?;
                Ok(split_chunk(chunk)?
                    .into_iter()
                    .take(horizon as usize)
                    .collect())
            })
            .collect::<Result<Vec<_>>>()?
    } else if predict.has_batch() {
        let actions = predict.predict_batch(inputs)?;
        if actions.len() != num_envs {
            return Err(Error::model(format!(
                "predict_batch returned {} actions for {num_envs} lanes",
                actions.len()
            )));
        }
        actions.into_iter().map(|action| vec![action]).collect()
    } else {
        inputs
            .into_iter()
            .map(|input| -> Result<Vec<Value>> { Ok(vec![predict.predict(input)?]) })
            .collect::<Result<Vec<_>>>()?
    };

    // Apply the per-step adapter transform to each frame, then peel frame 0 (this
    // step's action) from the future frames (which the runtime replays).
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

    // Transpose the per-lane future frames into per-step batched frames
    // (`replay[step][lane]`), to the shortest lane (uniform for a homogeneous
    // fleet; a short lane caps the batch — receding horizon).
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
        let entry = self.entry(&route_key(&observation.route));
        let predict = Arc::clone(&self.predict);
        // Decode + frame-stack + the model's predict are CPU/host work; run them
        // off the async worker so concurrent (pipelined) requests on other routes
        // are not stalled. A spec'd route runs the per-lane engine loop (emitting
        // chunk frames); a spec-less route takes the preserved batched raw path
        // (one action per lane, no chunk).
        tokio::task::spawn_blocking(move || match entry {
            Some(entry) => predict_route(&entry, &predict, observation),
            None => Ok(PredictFrames {
                actions: predict.predict_spec_less(observation)?,
                replay: Vec::new(),
            }),
        })
        .await
        .map_err(|err| Error::Internal(format!("predict task panicked: {err}")))?
    }

    fn route_setup(&self) -> Option<Arc<dyn ModelRouteSetup>> {
        let resolver = self.resolver.clone()?;
        Some(Arc::new(AdaptedRouteSetup {
            resolver,
            routes: Arc::clone(&self.routes),
            predict: Arc::clone(&self.predict),
        }))
    }

    async fn enter_route(&mut self, route_key: &str) -> Result<()> {
        self.current_route = Some(route_key.to_string());
        Ok(())
    }

    async fn on_lane_reset(&mut self, event: ModelLaneReset) -> Result<()> {
        // Seed the new episode's (empty) buffer on the current route. Edge-driven:
        // the row's first predict first-frame-pads. Asserts the episode is not
        // already mid-stack — a missed END surfaces loudly here.
        if let Some(route_key) = self.current_route.as_deref()
            && let Some(entry) = self.entry(route_key)
        {
            let seeded = entry
                .lock()
                .expect("route entry poisoned")
                .buffers
                .seed(&event.episode_id);
            debug_assert!(
                seeded,
                "episode {} already mid-stack at reset (missed on_episode_end)",
                event.episode_id
            );
        }
        Ok(())
    }

    async fn on_reset(&mut self, _observation: &ModelObservation) -> Result<()> {
        let predict = Arc::clone(&self.predict);
        tokio::task::spawn_blocking(move || predict.on_reset())
            .await
            .map_err(|err| Error::Internal(format!("on_reset task panicked: {err}")))?
    }

    async fn on_episode_end(&mut self, event: ModelEpisodeEnd) -> Result<()> {
        // Evict the ended episode's buffers (edge-driven). During close sweeps no
        // route is entered, so this may target the wrong/absent route — harmless,
        // since close_route / on_close are the authoritative reclaimers.
        if let Some(route_key) = self.current_route.as_deref()
            && let Some(entry) = self.entry(route_key)
        {
            let mut guard = entry.lock().expect("route entry poisoned");
            guard.buffers.evict(&event.episode_id);
        }
        let predict = Arc::clone(&self.predict);
        tokio::task::spawn_blocking(move || predict.on_episode_end())
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
}

#[async_trait]
impl ModelRouteSetup for AdaptedRouteSetup {
    async fn configure_route(
        &self,
        route_key: &str,
        env_contract: &EnvContract,
        action_horizon: u32,
    ) -> Result<()> {
        let Some(mut config) = self.resolver.resolve(route_key, env_contract).await? else {
            return Ok(());
        };
        // Stamp the runtime-chosen replay horizon onto the resolved config (1 = no
        // chunking). Warn once here when the runtime asks for chunking but the model
        // has no chunk corner: the route still runs, re-planning every step.
        config.action_horizon = action_horizon.max(1);
        if config.action_horizon > 1 && !self.predict.has_chunk() {
            tracing::warn!(
                route = %route_key,
                action_horizon = config.action_horizon,
                "runtime pinned action_horizon > 1 but the model defines no chunk corner \
                 (predict_chunk); chunking is inactive — the model re-plans every step",
            );
        }
        // Frame-stacking + chunking would stack only decision-point frames (the
        // engine assembles observations once per chunk, not every step), not the
        // consecutive history a stacked policy expects. Reject the combination
        // rather than feed temporally-aliased frames. This was a resolve-time check,
        // relocated here now that the horizon is a runtime decision.
        if config.action_horizon > 1
            && let Some((key, depth)) = config.adapter.stacks().into_iter().next()
        {
            return Err(Error::model(format!(
                "frame-stacking (input '{key}' stack={depth}) cannot be combined with action \
                 chunking (action_horizon={}): the engine assembles observations once per chunk, \
                 so the frame window would hold only decision-point frames. Use stack=1 or \
                 action_horizon=1.",
                config.action_horizon,
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
            .insert(route_key.to_string(), entry);
        Ok(())
    }

    async fn close_route(&self, route_key: &str) -> Result<()> {
        self.routes
            .lock()
            .expect("routes map poisoned")
            .remove(route_key);
        Ok(())
    }
}
