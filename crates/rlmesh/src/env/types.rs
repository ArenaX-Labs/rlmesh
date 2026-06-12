use crate::spaces;

pub use spaces::{CloseRequest, RenderRequest, RenderResult};

/// Summary of one finished episode, reported on step completion and on close.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EpisodeMetadata {
    /// Unique identifier for the episode.
    pub episode_id: String,
    /// Seed the episode was reset with.
    pub seed: i64,
    /// Index of the sub-environment that produced the episode.
    pub env_index: i32,
    /// Number of steps the episode ran.
    pub step_count: i64,
    /// Sum of rewards over the episode.
    pub cumulative_reward: f64,
    /// Whether the episode ended in a terminal state.
    pub terminated: bool,
    /// Whether the episode was truncated (e.g. by a time limit).
    pub truncated: bool,
    /// Episode start time, nanoseconds since the Unix epoch.
    pub start_timestamp_ns: i64,
    /// Episode end time, nanoseconds since the Unix epoch.
    pub end_timestamp_ns: i64,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: i64,
    /// Final `info` map emitted by the environment, if any.
    pub final_info: Option<spaces::MetaMap>,
}

/// The result of closing an environment or env session.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CloseResult {
    /// Metadata for episodes that were still in flight at close time.
    pub final_episodes: Vec<EpisodeMetadata>,
}

/// A vectorized reset request: one seed per sub-environment.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResetRequest {
    /// One seed per sub-environment (empty leaves seeding to the env).
    pub seeds: Vec<i64>,
    /// Optional reset options forwarded to the environment.
    pub options: Option<spaces::MetaMap>,
    /// Per-call deadline in milliseconds; `0` means no deadline.
    pub timeout_ms: i64,
}

/// A vectorized step request: one action per sub-environment.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StepRequest {
    /// One action per sub-environment, in env-index order.
    pub actions: Vec<spaces::SpaceValue>,
    /// Per-call deadline in milliseconds; `0` means no deadline.
    pub timeout_ms: i64,
}

/// The result of a vectorized [`ResetRequest`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResetResult {
    /// One initial observation per sub-environment.
    pub observations: Vec<spaces::SpaceValue>,
    /// Optional batched `info` map from the environment.
    pub info: Option<spaces::MetaMap>,
    /// The episode id assigned to each sub-environment.
    pub episode_ids: Vec<String>,
}

/// The result of a vectorized [`StepRequest`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StepResult {
    /// One observation per sub-environment.
    pub observations: Vec<spaces::SpaceValue>,
    /// One reward per sub-environment.
    pub rewards: Vec<f64>,
    /// Per-sub-environment terminal flags.
    pub terminated: Vec<bool>,
    /// Per-sub-environment truncation flags.
    pub truncated: Vec<bool>,
    /// Optional batched `info` map from the environment.
    pub info: Option<spaces::MetaMap>,
    /// Metadata for episodes that finished on this step (auto-reset).
    pub completed_episodes: Vec<EpisodeMetadata>,
    /// The current episode id for each sub-environment.
    pub episode_ids: Vec<String>,
}
