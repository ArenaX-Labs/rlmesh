//! RLMesh single-session runtime driver.
//!
//! This crate owns the shared `reset -> predict -> step` execution semantics
//! for one ready environment/model session. It deliberately does not own
//! managed route fan-out, endpoint pooling, workload expansion, scheduling, or
//! cluster lifecycle.

mod driver;
pub mod hooks;
pub mod spec;

mod episodes;
mod route;
mod state;
mod timing;

pub use driver::{
    RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep, RuntimeError, RuntimeModel,
    RuntimeModelPrediction,
};
pub use hooks::{
    ActionReceivedEvent, EnvConnectedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, HookError,
    LogEvent, LogLevel, ModelConnectedEvent, NoopRuntimeHooks, ObservationEmittedEvent,
    RuntimeHookChain, RuntimeHooks, RuntimeRouteContext, SessionEndedEvent, SessionFailedEvent,
    SessionStartedEvent, StepCompletedEvent, TelemetrySummaryEvent, TelemetryWindowEvent,
    TimingSummary,
};
pub use spec::{RuntimeLimits, RuntimeReport, RuntimeSessionSpec};
