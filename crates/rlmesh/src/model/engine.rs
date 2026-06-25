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
    ChunkBuffers, FrameBuffers, ObsPlan, Value, apply_actions, assemble_obs,
    space_value_to_obs_map, split_chunk,
};

use super::handler::{ModelHandler, ModelRouteSetup};
use super::predict_fn::{PredictFn, RouteConfig, RouteResolver};
use super::types::{ModelEpisodeEnd, ModelLaneReset, ModelObservation, ModelRouteContext};
use crate::spaces::{EnvContract, SpaceKind, SpaceValue};
use crate::{Error, Result};

/// One configured route's resolved config plus its live per-episode state: the
/// frame-stack windows and the action-chunk replay queues.
struct RouteEntry {
    config: RouteConfig,
    buffers: FrameBuffers,
    chunks: ChunkBuffers,
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
fn predict_route(
    entry: &Arc<Mutex<RouteEntry>>,
    predict: &Arc<dyn PredictFn>,
    observation: ModelObservation,
) -> Result<Vec<SpaceValue>> {
    let episode_ids = observation.episode_ids();
    let num_envs = observation.num_envs;

    // The wire contract requires every predict request to carry an observation,
    // even on a pure chunk-replay step that will skip decoding it. Validate
    // presence up front (cheap — no decode) so a malformed request still errors
    // here rather than silently replaying buffered actions; the image-sized decode
    // itself stays lazy below (skipped entirely when every lane is mid-replay).
    if observation.observation.is_none() {
        return Err(Error::model(
            "observation absent; a predict request must carry an observation",
        ));
    }

    let mut guard = entry.lock().expect("route entry poisoned");
    let RouteEntry {
        config,
        buffers,
        chunks,
    } = &mut *guard;
    let referenced = obs_keys(config);
    let horizon = config.adapter.action_plan.execute_horizon;
    let customs: &dyn rlmesh_adapters::v1::CustomTransform = config.customs.as_ref();
    let encodings: &dyn rlmesh_adapters::v1::EncodingTransform = config.encodings.as_ref();

    // Lanes are decoded lazily: a step where every lane is mid-chunk-replay needs
    // no observation at all, so the (image-sized) batch decode is skipped entirely.
    // The wire decode is batched, so the first lane that must re-plan materializes
    // all lanes; a fully-replaying step pays nothing. (Decode now sits under the
    // route lock, which already spans predict — the dominant per-lane cost.)
    let mut decoded: Option<Vec<SpaceValue>> = None;

    let mut actions = Vec::with_capacity(num_envs);
    for index in 0..num_envs {
        let episode_id = episode_ids
            .get(index)
            .map(String::as_str)
            .unwrap_or_default();
        // Action-chunk replay: when this lane's episode still has queued actions
        // from an earlier predicted chunk, emit the next one and skip predict (and
        // therefore the obs decode + assembly) entirely. Only re-plan — decode the
        // obs, assemble, call predict, refill the queue — when the chunk drains.
        // With horizon == 1 the queue is never used, so this is the unchanged path.
        let raw_action = match (horizon > 1)
            .then(|| chunks.next_action(episode_id))
            .flatten()
        {
            Some(queued) => queued,
            None => {
                if decoded.is_none() {
                    decoded = Some(observation.decoded_lanes()?);
                }
                let lane = &decoded.as_ref().expect("decoded above")[index];
                let raw = space_value_to_obs_map(lane, &config.observation_space, &referenced)?;
                let input = assemble_obs(
                    &config.adapter,
                    &raw,
                    episode_id,
                    buffers,
                    customs,
                    encodings,
                )?;
                let predicted = predict.predict(input)?;
                if horizon > 1 {
                    refill_and_take_first(chunks, episode_id, predicted, horizon)?
                } else {
                    predicted
                }
            }
        };
        let env_action = apply_actions(
            &config.adapter,
            &raw_action,
            &config.action_space,
            encodings,
        )?;
        actions.push(env_action);
    }
    Ok(actions)
}

/// Split a freshly predicted chunk into per-step actions, queue all but the first
/// (capped at `horizon` — a receding-horizon model may emit a longer chunk than
/// it re-plans), and return the first action to emit now.
fn refill_and_take_first(
    chunks: &mut ChunkBuffers,
    episode_id: &str,
    predicted: Value,
    horizon: u32,
) -> Result<Value> {
    let mut steps = split_chunk(predicted)?.into_iter().take(horizon as usize);
    let first = steps.next().ok_or_else(|| {
        Error::model("a chunked model (execute_horizon>1) returned an empty action chunk")
    })?;
    chunks.refill(episode_id, steps);
    Ok(first)
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
    let assemble =
        |seed: u64, episode: &str| -> Result<BTreeMap<String, rlmesh_adapters::v1::Value>> {
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
        let entry = self.entry(&route_key(&observation.route));
        let predict = Arc::clone(&self.predict);
        // Decode + frame-stack + the model's predict are CPU/host work; run them
        // off the async worker so concurrent (pipelined) requests on other routes
        // are not stalled. A spec'd route runs the per-lane engine loop; a
        // spec-less route takes the preserved batched raw path.
        tokio::task::spawn_blocking(move || match entry {
            Some(entry) => predict_route(&entry, &predict, observation),
            None => predict.predict_spec_less(observation),
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
            guard.chunks.evict(&event.episode_id);
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
            guard.chunks.clear();
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
    async fn configure_route(&self, route_key: &str, env_contract: &EnvContract) -> Result<()> {
        let Some(config) = self.resolver.resolve(route_key, env_contract).await? else {
            return Ok(());
        };
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
            chunks: ChunkBuffers::new(),
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
