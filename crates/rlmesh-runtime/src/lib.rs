//! RLMesh single-session runtime driver.
//!
//! Shared `reset -> predict -> step` semantics for one ready model/environment
//! session. Scheduling, endpoint pools, route fan-out, and cluster lifecycle
//! live elsewhere.

mod driver;
pub mod hooks;
pub mod spec;

mod episodes;
mod state;
mod timing;

pub use driver::{
    RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep, RuntimeError, RuntimeModel,
    RuntimeModelPrediction,
};
pub use hooks::{
    ActionReceivedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, HookError, LogEvent, LogLevel,
    MetricKind, MetricSummary, NoopRuntimeHooks, ObservationEmittedEvent, RuntimeHooks,
    RuntimeRouteContext, SessionEndedEvent, SessionFailedEvent, SessionStartedEvent,
    StepCompletedEvent, TelemetrySummaryEvent, TelemetryWindowEvent, TimingSummary,
};
pub use spec::{RuntimeLimits, RuntimeReport, RuntimeSessionSpec};

/// Protocol types used by the runtime public API.
///
/// Runtime traits and hooks name generated `rlmesh_proto::*::v1` types directly.
/// Re-exporting the crate gives implementors the version-correct path. Protocol
/// regeneration or a major `prost`/`tonic` bump is therefore a runtime breaking
/// change.
pub use rlmesh_proto;
