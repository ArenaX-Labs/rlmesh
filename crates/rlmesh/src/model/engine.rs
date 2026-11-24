//! The vectorized stateful adapter engine: a [`ModelHandler`] that owns the
//! per-lane drive, episode-keyed frame buffers, and native adapter application
//! in pure Rust, calling back into a [`PredictFn`] only for the model's predict
//! and into the custom/encoding holes only where a route declares them.

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
    held: Arc<HeldCells>,
    adapter_ns: Arc<AtomicU64>,
    short_chunk_warned: Arc<ShortChunkWarned>,
    history_limit: u64,
}

impl AdaptedModelHandler {
    /// Build a handler from the model's predict hole and an optional route resolver.
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

    fn entry(&self, env_id: &str) -> Option<Arc<Mutex<RouteEntry>>> {
        self.routes
            .lock()
            .expect("routes map poisoned")
            .get(env_id)
            .cloned()
    }

    fn spec_less_horizon(&self, env_id: &str) -> u32 {
        self.spec_less_horizons
            .lock()
            .expect("spec-less horizons poisoned")
            .get(env_id)
            .copied()
            .unwrap_or(1)
    }
}

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
