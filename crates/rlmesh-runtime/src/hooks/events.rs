use std::sync::Arc;

use prost::bytes::Bytes;
use rlmesh_proto::spaces::v1::MetaMap;
use rlmesh_proto::spaces::v1::SpaceSpec;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeRouteContext {
    pub route_id: String,
    pub env_component_id: String,
    pub model_component_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvConnectedEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
    pub env_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConnectedEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionStartedEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
    pub env_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionEndedEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
    pub reason: String,
    pub total_steps: i64,
    pub total_episodes: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionFailedEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
    pub level: LogLevel,
    pub message: String,
    pub source: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeStartedEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
    pub episode_id: String,
    pub episode_record_id: String,
    pub episode_index: i64,
    pub env_index: i32,
    pub started_from_auto_reset: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeCompletedEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
    pub episode_id: String,
    pub episode_record_id: String,
    pub episode_index: i64,
    pub env_index: i32,
    pub step_count: i64,
    pub cumulative_reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub duration_ms: i64,
    pub final_info: Option<MetaMap>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionReceivedEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
    pub episode_id: String,
    pub episode_record_id: String,
    pub episode_ids: Vec<String>,
    pub episode_record_ids: Vec<String>,
    pub step: i64,
    pub env_index: i32,
    /// Shared so the per-step, per-hook event fan-out clones an `Arc` pointer
    /// rather than deep-copying the action space spec on every step.
    pub action_space: Arc<SpaceSpec>,
    /// Opaque per-leaf wire bytes; the relay is content-blind (§13).
    pub action: Option<Vec<Bytes>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepCompletedEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
    pub episode_id: String,
    pub episode_record_id: String,
    pub step: i64,
    pub env_index: i32,
    pub rewards: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationEmittedEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
    pub episode_id: String,
    pub episode_record_id: String,
    pub episode_ids: Vec<String>,
    pub episode_record_ids: Vec<String>,
    pub step: i64,
    pub env_index: i32,
    pub is_reset: bool,
    pub num_envs: u32,
    /// Shared so the per-step, per-hook event fan-out clones an `Arc` pointer
    /// rather than deep-copying the observation space spec on every step.
    pub observation_space: Arc<SpaceSpec>,
    /// Opaque per-leaf wire bytes; the relay is content-blind (§13).
    pub observation: Option<Vec<Bytes>>,
}
