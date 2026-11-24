//! Plain state records shared across the route-state module: per-slot state, a
//! point-in-time snapshot, and the started-episode handoff.

use crate::run::runtime::episodes::EpisodeRecord;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EpisodeState {
    pub(crate) episode_id: String,
    pub(crate) episode_record_id: String,
    pub(crate) episode_index: i64,
    pub(crate) started_from_auto_reset: bool,
    pub(crate) predicted: bool,
    pub(crate) ended: bool,
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
    pub(crate) cumulative_reward: f64,
    pub(crate) started_at_ns: i64,
    pub(crate) predict_ns: u64,
    pub(crate) step_ns: u64,
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
