use std::sync::Arc;

use prost_types::Struct;
use rlmesh_proto::common::v1::MessageBytes;
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

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
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
    pub final_info: Option<Struct>,
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
    pub action: Option<MessageBytes>,
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
    pub observation: Option<MessageBytes>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimingSummary {
    pub operation: String,
    pub component_id: String,
    pub name: String,
    pub sample_count: u64,
    pub avg_ms: Option<f64>,
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
}

/// Kind of a non-duration metric reported via [`MetricSummary`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// `OperationMetric::ByteCount` — a byte gauge/counter sample.
    ByteCount,
    /// `OperationMetric::Number` — a generic numeric gauge sample.
    Number,
}

/// Aggregated non-duration operation metric (byte counts and generic numbers
/// carried by `OperationTelemetry`). Duration metrics are reported via
/// [`TimingSummary`]; these were previously dropped at the accumulator.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricSummary {
    pub operation: String,
    pub component_id: String,
    pub name: String,
    pub kind: MetricKind,
    pub sample_count: u64,
    pub avg: Option<f64>,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryWindowEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
    pub window_seconds: u32,
    pub sample_count: u64,
    pub steps_per_second: Option<f64>,
    pub request_bytes_per_second: Option<f64>,
    pub response_bytes_per_second: Option<f64>,
    pub timings: Vec<TimingSummary>,
    pub metrics: Vec<MetricSummary>,
    pub env_latency_ms_avg: Option<f64>,
    pub env_latency_ms_p50: Option<f64>,
    pub env_latency_ms_p95: Option<f64>,
    pub env_latency_ms_p99: Option<f64>,
    pub model_latency_ms_avg: Option<f64>,
    pub model_latency_ms_p50: Option<f64>,
    pub model_latency_ms_p95: Option<f64>,
    pub model_latency_ms_p99: Option<f64>,
    pub round_trip_ms_avg: Option<f64>,
    pub round_trip_ms_p50: Option<f64>,
    pub round_trip_ms_p95: Option<f64>,
    pub round_trip_ms_p99: Option<f64>,
    pub reconnects: u64,
    pub drops: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TelemetrySummaryEvent {
    pub session_id: String,
    pub route: RuntimeRouteContext,
    pub total_seconds: u32,
    pub sample_count: u64,
    pub steps_per_second: Option<f64>,
    pub request_bytes_per_second: Option<f64>,
    pub response_bytes_per_second: Option<f64>,
    pub timings: Vec<TimingSummary>,
    pub metrics: Vec<MetricSummary>,
    pub env_latency_ms_avg: Option<f64>,
    pub env_latency_ms_p50: Option<f64>,
    pub env_latency_ms_p95: Option<f64>,
    pub env_latency_ms_p99: Option<f64>,
    pub model_latency_ms_avg: Option<f64>,
    pub model_latency_ms_p50: Option<f64>,
    pub model_latency_ms_p95: Option<f64>,
    pub model_latency_ms_p99: Option<f64>,
    pub round_trip_ms_avg: Option<f64>,
    pub round_trip_ms_p50: Option<f64>,
    pub round_trip_ms_p95: Option<f64>,
    pub round_trip_ms_p99: Option<f64>,
    pub reconnects: u64,
    pub drops: u64,
}
