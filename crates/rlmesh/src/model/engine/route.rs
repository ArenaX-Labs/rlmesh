//! Per-route engine state: each route's resolved config and frame windows,
//! the endpoint-wide held-state total, and the frame-history byte budget.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rlmesh_adapters::v1::{FrameBuffers, HistoryWindow};

use crate::model::predict_fn::RouteConfig;
use crate::model::types::EpisodeInfo;
use crate::{Error, Result};

pub(super) struct RouteEntry {
    pub(super) config: Arc<RouteConfig>,
    pub(super) buffers: FrameBuffers,
    pub(super) total: Arc<HeldCells>,
    pub(super) published: HeldTotals,
    pub(super) lane_bytes: u64,
    pub(super) history_limit: u64,
}

impl RouteEntry {
    pub(super) fn publish_held(&mut self) {
        let episodes = self.buffers.episodes() as u64;
        let now = HeldTotals {
            episodes,
            bytes: self.buffers.state_bytes(),
            projected: episodes.saturating_mul(self.lane_bytes),
        };
        // One wrapping add per cell: a shrink adds the two's complement of the drop, so a concurrent
        // read never sees the total dip below the other routes' share.
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

#[derive(Clone, Copy, Default)]
pub(super) struct HeldTotals {
    episodes: u64,
    bytes: u64,
    projected: u64,
}

#[derive(Default)]
pub(super) struct HeldCells {
    pub(super) episodes: AtomicU64,
    pub(super) bytes: AtomicU64,
    projected: AtomicU64,
}

pub(super) type Routes = Arc<Mutex<HashMap<String, Arc<Mutex<RouteEntry>>>>>;

pub(super) type SpecLessHorizons = Arc<Mutex<HashMap<String, u32>>>;
pub(super) const FRAME_HISTORY_LIMIT_ENV: &str = "RLMESH_FRAME_HISTORY_LIMIT_BYTES";
pub(super) const FRAME_HISTORY_LIMIT_DEFAULT: u64 = 2 << 30;

pub(super) fn frame_history_limit() -> u64 {
    std::env::var(FRAME_HISTORY_LIMIT_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(FRAME_HISTORY_LIMIT_DEFAULT)
}

pub(super) fn lane_window_bytes(windows: &[HistoryWindow]) -> u64 {
    windows
        .iter()
        .map(|window| u64::from(window.span).saturating_mul(window.frame_bytes))
        .sum()
}

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
