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

/// Per-episode timing depth, computed natively by the runtime and snapshotted
/// at episode completion. Surfaced ONLY on
/// [`EpisodeCompletedEvent::final_episode_telemetry`] — a push-side hook
/// attachment. It is intentionally NOT part of [`crate::spec::RuntimeReport`]:
/// window/session telemetry has both push (hooks) and pull (`RuntimeReport`)
/// surfaces, but per-episode telemetry is a hook attachment only.
///
/// Averages + totals only — no percentiles (that needs a per-episode sample
/// buffer, not worth the hot-path memory) and no min/max (no consumer yet; add
/// additively when one exists). `steps_per_second` is intentionally not
/// precomputed: a consumer derives it from `step_count` and the
/// env-self-reported [`EpisodeCompletedEvent::duration_ms`].
///
/// `endpoint_op_ms_avg` is the mean endpoint-local op duration BLENDED across
/// the per-step `model.predict` and `env.step` ops (it EXCLUDES `env.reset`, a
/// distinct once-per-episode op). It is a coarse per-op pod-side latency, not a
/// per-step total; split into model/env endpoint averages additively if a
/// consumer ever needs the per-side breakdown.
///
/// VALIDITY: only emitted for single-lane routes (`num_envs == 1`) when exactly
/// one episode completes in a step sweep. For `num_envs > 1` (or a
/// multi-completion sweep) the per-step accumulator folds interleaved lanes, so
/// a rollup would be a per-route slice, not true per-episode attribution; the
/// driver leaves it `None`. True per-lane attribution defers to the vector
/// engine.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeTelemetryRollup {
    pub episode_record_id: String,
    pub env_index: i32,
    pub step_count: u64,
    pub model_latency_ms_avg: Option<f64>,
    pub env_latency_ms_avg: Option<f64>,
    pub round_trip_ms_avg: Option<f64>,
    pub endpoint_op_ms_avg: Option<f64>,
    pub request_bytes_total: u64,
    pub response_bytes_total: u64,
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
    /// Native per-episode timing depth captured at completion. `None` for
    /// vectorized routes (`num_envs > 1`) or any sweep that completes more than
    /// one episode; see [`EpisodeTelemetryRollup`].
    pub final_episode_telemetry: Option<EpisodeTelemetryRollup>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetricKind {
    /// A byte gauge/counter sample (wire `Unit::BYTES`).
    ByteCount,
    /// A generic numeric gauge sample (wire `Unit::COUNT`).
    Number,
}

/// Aggregated non-duration operation metric (byte counts and generic numbers).
/// Surfaced on the wire as `MetricSummary`; duration metrics are reported via
/// [`TimingSummary`].
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
