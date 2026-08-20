//! Plain state records shared across the route-state module: per-slot state, a
//! point-in-time snapshot, and the started-episode handoff.

use crate::episodes::EpisodeRecord;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EpisodeState {
    pub(crate) episode_id: String,
    pub(crate) episode_record_id: String,
    pub(crate) episode_index: i64,
    pub(crate) started_from_auto_reset: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct StartedEpisode {
    pub(crate) episode_id: String,
    pub(crate) record: EpisodeRecord,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SlotState {
    pub(crate) env_index: i32,
    pub(crate) episode: Option<EpisodeState>,
    pub(crate) step: i64,
    pub(crate) reset: bool,
    /// Reward accumulated over the slot's current episode; feeds the
    /// runtime-synthesized completion when a step/time cap truncates the lane.
    pub(crate) cumulative_reward: f64,
    /// Unix-ns timestamp of the slot's current episode start (reset), for the
    /// `max_episode_seconds` cap and synthesized completion timestamps.
    pub(crate) started_at_ns: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RouteSnapshot {
    pub(crate) episode_id: String,
    pub(crate) episode_record_id: String,
    pub(crate) episode_ids: Vec<String>,
    pub(crate) episode_record_ids: Vec<String>,
    pub(crate) step: i64,
    pub(crate) env_index: i32,
    pub(crate) reset: bool,
}
