//! Per-route engine state: each route's resolved config and frame windows,
//! the endpoint-wide held-state total, and the frame-history byte budget.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rlmesh_adapters::v1::{FrameBuffers, HistoryWindow};

use crate::model::predict_fn::RouteConfig;
use crate::model::types::EpisodeInfo;
use crate::{Error, Result};

/// One configured route's resolved config plus its live per-episode frame-stack
/// windows.
///
/// Action-chunk replay no longer lives here: the engine emits the whole chunk
/// (frame 0 + future frames) as [`PredictFrames`] and the runtime driver owns the
/// per-step replay buffer. Only the frame-stack windows remain engine state.
pub(super) struct RouteEntry {
    pub(super) config: Arc<RouteConfig>,
    pub(super) buffers: FrameBuffers,
    /// The endpoint-wide held-state total this route publishes into (see
    /// [`HeldCells`]).
    pub(super) total: Arc<HeldCells>,
    /// What this route last published into `total`, so a publish (or the drop
    /// of a released route) applies only the difference.
    pub(super) published: HeldTotals,
    /// Bytes one lane's frame windows hold once full (`Σ span × frame_bytes`);
    /// `0` for a route with no stacked input.
    pub(super) lane_bytes: u64,
    /// The endpoint-wide frame-window budget this route's lanes are admitted
    /// against (see [`FRAME_HISTORY_LIMIT_ENV`]).
    pub(super) history_limit: u64,
}

impl RouteEntry {
    /// Publish this route's held frame-stack state into the endpoint total, as
    /// the delta against its previous publish.
    pub(super) fn publish_held(&mut self) {
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
pub(super) struct HeldTotals {
    episodes: u64,
    bytes: u64,
    projected: u64,
}

/// The endpoint-wide held frame-stack total, kept by per-route deltas published
/// from under each entry lock, so `held_state` reads it in O(1) without the
/// routes map or any entry lock that may be held across a model forward.
#[derive(Default)]
pub(super) struct HeldCells {
    pub(super) episodes: AtomicU64,
    pub(super) bytes: AtomicU64,
    /// What every route's held episodes' windows hold once full
    /// (`Σ episodes × lane_bytes`): the admission guard reads this rather than
    /// `bytes`, so a window still filling is charged what it will reach, not
    /// what it has so far.
    projected: AtomicU64,
}

/// `route_key -> route state`. The outer lock is held only to look up/insert a
/// route; the per-route [`Mutex`] is what predict holds across its (blocking)
/// per-lane loop, so configuring one route never blocks predict on another.
pub(super) type Routes = Arc<Mutex<HashMap<String, Arc<Mutex<RouteEntry>>>>>;

/// `env_id -> execution horizon` for SPEC-LESS routes (no [`RouteEntry`]): the
/// horizon is pinned at `ResolveAdapter` like a spec'd route's, but there is no
/// config to stamp it on, so it lives here for the spec-less predict branch to
/// read (chunk corner support without an adapter).
pub(super) type SpecLessHorizons = Arc<Mutex<HashMap<String, u32>>>;
/// Env var naming the most bytes this endpoint's frame windows may project to
/// hold across every route's live lanes (`Σ lanes × Σ span × frame_bytes`);
/// default 2 GiB. Checked twice: per route at resolve, for the lanes the
/// contract names, and per fresh lane at predict, against the endpoint total.
pub(super) const FRAME_HISTORY_LIMIT_ENV: &str = "RLMESH_FRAME_HISTORY_LIMIT_BYTES";
pub(super) const FRAME_HISTORY_LIMIT_DEFAULT: u64 = 2 << 30;

pub(super) fn frame_history_limit() -> u64 {
    std::env::var(FRAME_HISTORY_LIMIT_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(FRAME_HISTORY_LIMIT_DEFAULT)
}

/// Bytes one lane's windows hold once full.
pub(super) fn lane_window_bytes(windows: &[HistoryWindow]) -> u64 {
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
pub(super) fn check_history_budget(
    num_envs: u32,
    windows: &[HistoryWindow],
    limit: u64,
) -> Result<()> {
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
pub(super) fn admit_fresh_lanes(entry: &RouteEntry, episodes: &[EpisodeInfo]) -> Result<()> {
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
