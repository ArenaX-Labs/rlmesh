use crate::spaces;

pub use spaces::{CloseRequest, RenderRequest, RenderResult};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct EpisodeMetadata {
    pub episode_id: String,
    pub seed: i64,
    pub env_index: i32,
    pub step_count: i64,
    pub cumulative_reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub start_timestamp_ns: i64,
    pub end_timestamp_ns: i64,
    pub duration_ms: i64,
    pub final_info: Option<spaces::MetaMap>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CloseResult {
    pub final_episodes: Vec<EpisodeMetadata>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResetRequest {
    pub seeds: Vec<i64>,
    pub options: Option<spaces::MetaMap>,
    pub timeout_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StepRequest {
    pub actions: Vec<spaces::SpaceValue>,
    pub timeout_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResetResult {
    pub observations: Vec<spaces::SpaceValue>,
    pub info: Option<spaces::MetaMap>,
    pub episode_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StepResult {
    pub observations: Vec<spaces::SpaceValue>,
    pub rewards: Vec<f64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub info: Option<spaces::MetaMap>,
    pub completed_episodes: Vec<EpisodeMetadata>,
    pub episode_ids: Vec<String>,
}
