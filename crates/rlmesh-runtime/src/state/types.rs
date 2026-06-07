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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SlotState {
    pub(crate) env_index: i32,
    pub(crate) episode: Option<EpisodeState>,
    pub(crate) step: i64,
    pub(crate) reset: bool,
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
