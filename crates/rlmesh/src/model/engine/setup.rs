//! Route setup: resolving, resetting and releasing a route's adapter, behind
//! the resolve-time horizon doors.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rlmesh_adapters::v1::{FrameBuffers, MAX_EXECUTION_HORIZON};

use super::route::{
    HeldCells, HeldTotals, RouteEntry, Routes, SpecLessHorizons, check_history_budget,
    lane_window_bytes,
};
use crate::binding::spaces::EnvContract;
use crate::model::handler::{HistoryNeeds, ModelRouteSetup, ResolveOptions, RouteNeeds};
use crate::model::predict_fn::{PredictFn, RouteResolver};
use crate::{Error, Result};

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
/// and the runtime driver's route connect.
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
pub(super) struct AdaptedRouteSetup {
    pub(super) resolver: Arc<dyn RouteResolver>,
    pub(super) routes: Routes,
    pub(super) predict: Arc<dyn PredictFn>,
    pub(super) spec_less_horizons: SpecLessHorizons,
    pub(super) held: Arc<HeldCells>,
    pub(super) history_limit: u64,
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
            ..RouteNeeds::default()
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
        let advisories = rlmesh_adapters::v1::advisories(&config.adapter);
        for note in &advisories {
            tracing::warn!(
                env_id = %env_id,
                severity = note.severity.as_str(),
                "adapter advisory: {note}"
            );
        }
        // And hand them to the run's driver, which carries them into its result.
        needs.advisories = advisories;
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
        // The model's own per-episode state for this env goes too (MDL-8);
        // the model serializes it behind its in-flight corner calls.
        let predict = Arc::clone(&self.predict);
        let env_id = env_id.to_string();
        crate::warnings::blocking(move || predict.on_route_release(&env_id)).await?
    }
}
