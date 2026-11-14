//! The vectorized stateful adapter engine: a [`ModelHandler`] that owns the
//! per-lane drive, episode-keyed frame buffers, and native adapter application
//! in pure Rust, calling back into a [`PredictFn`] only for the model's predict
//! and into the custom/encoding holes only where a route declares them.
//!
//! This replaces a binding's hand-rolled predict loop: a PyO3 (or any future
//! language) binding constructs `AdaptedModelHandler::new(predict, resolver)`
//! and serves it; a pure-Rust model does the same with no host runtime.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::handler::{HeldState, ModelHandler, ModelResetOptions, ModelRouteSetup, PredictFrames};
use super::predict_fn::{PredictFn, RouteResolver};
use super::types::{EpisodeInfo, ModelObservation};
use crate::Result;
use crate::binding::spaces::SpaceValue;

mod adapt;
mod corners;
mod predict;
mod route;
mod setup;

#[cfg(test)]
mod fused_predict_tests;
#[cfg(test)]
mod fused_route_tests;

pub(crate) use corners::{ShortChunkWarned, short_chunk_text};
use predict::{GroupLane, predict_grouped_fused, predict_route};
use route::{HeldCells, RouteEntry, Routes, SpecLessHorizons, frame_history_limit};
use setup::AdaptedRouteSetup;

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
    /// The episodes the short-chunk warning already fired for (see
    /// [`take_prefix`]): once per episode.
    short_chunk_warned: Arc<ShortChunkWarned>,
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
            short_chunk_warned: Arc::new(ShortChunkWarned::default()),
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

/// A model's reset options, asked of its predict hole on a blocking thread.
struct PredictResetOptions(Arc<dyn PredictFn>);

#[async_trait]
impl ModelResetOptions for PredictResetOptions {
    async fn reset_options(
        &self,
        episode: &EpisodeInfo,
        trial_index: Option<u64>,
    ) -> Result<Option<rlmesh_spaces::MetaMap>> {
        let predict = Arc::clone(&self.0);
        let episode = episode.clone();
        crate::warnings::blocking(move || predict.reset_options(&episode, trial_index)).await?
    }
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
        crate::warnings::blocking(move || match entry {
            Some(entry) => predict_route(
                &entry,
                &predict,
                observation,
                &adapter_ns,
                &short_chunk_warned,
            ),
            None => predict.predict_spec_less_chunked(observation, spec_less_horizon),
        })
        .await?
    }

    /// Fusion needs a batched corner and the model's permission
    /// (`allow_fusion`); without both, keep the chunk-preserving sequential
    /// default — correct per group, just unfused.
    async fn predict_grouped(
        &mut self,
        observations: Vec<ModelObservation>,
    ) -> Vec<Result<PredictFrames>> {
        if observations.len() <= 1 || !self.fuses_groups() {
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
        crate::warnings::blocking(move || {
            predict_grouped_fused(
                lanes,
                observations,
                &predict,
                &adapter_ns,
                &short_chunk_warned,
            )
        })
        .await
        .unwrap_or_else(|err| (0..group_count).map(|_| Err(err.clone())).collect())
    }

    fn fuses_groups(&self) -> bool {
        self.predict.allow_fusion() && (self.predict.has_chunk_batch() || self.predict.has_batch())
    }

    fn take_adapter_ns(&mut self) -> u64 {
        self.adapter_ns.swap(0, Ordering::Relaxed)
    }

    fn reset_options(&self) -> Option<Arc<dyn ModelResetOptions>> {
        self.predict
            .has_reset_options()
            .then(|| Arc::new(PredictResetOptions(Arc::clone(&self.predict))) as _)
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

    async fn reset_adapter(&mut self, env_id: &str, episode_ids: Vec<String>) -> Result<()> {
        // The frame buffers were already evicted off the predict lock by
        // `AdaptedRouteSetup::reset_adapter`. What is left is the model's own
        // hook, which shares the model's state with its forwards and so stays
        // serialized behind them here.
        //
        // Surface the episode-end edge to the model's own hook (one call per
        // ended episode), e.g. to reset a single-env model's recurrent state. An
        // empty `episode_ids` is an evict-ALL/teardown, not an episode end, so it
        // fires nothing here — model shutdown is `on_close`'s job — but it
        // drops the env's per-episode model state silently (MDL-8).
        for episode_id in &episode_ids {
            self.short_chunk_warned.remove(episode_id);
        }
        let predict = Arc::clone(&self.predict);
        if episode_ids.is_empty() {
            let env_id = env_id.to_string();
            return crate::warnings::blocking(move || predict.on_route_reset(&env_id)).await?;
        }
        crate::warnings::blocking(move || {
            for episode_id in &episode_ids {
                predict.on_episode_end(episode_id)?;
            }
            Ok(())
        })
        .await?
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
        crate::warnings::blocking(move || predict.on_close()).await?
    }
}
