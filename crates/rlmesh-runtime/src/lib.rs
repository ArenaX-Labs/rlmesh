//! RLMesh single-session runtime driver.
//!
//! Shared `reset -> predict -> step` semantics for one ready model/environment
//! session. Scheduling, endpoint pools, route fan-out, and cluster lifecycle
//! live elsewhere.

mod driver;
pub mod hooks;
pub mod spec;
pub mod telemetry;

mod episodes;
mod state;

pub use driver::{
    EagerScheduler, PredictScheduler, RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep,
    RuntimeError, RuntimeModel, RuntimeModelPrediction,
};
pub use hooks::{
    ActionReceivedEvent, EnvConnectedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, HookError,
    LogEvent, LogLevel, ModelConnectedEvent, NoopRuntimeHooks, ObservationEmittedEvent,
    RuntimeEnvContext, RuntimeHooks, SessionEndedEvent, SessionFailedEvent, SessionStartedEvent,
    StepCompletedEvent, TelemetrySnapshotEvent,
};
pub use rlmesh_proto::EndpointPhases;
pub use spec::{
    ENV_RESET_OPTIONS_KEY, RuntimeLimits, RuntimeReport, RuntimeSessionSpec, TRIAL_INDEX_OPTION,
    declares_reset_option, reset_options_for,
};

/// Protocol types used by the runtime public API.
///
/// Runtime traits and hooks name generated `rlmesh_proto::*::v1` types directly.
/// Re-exporting the crate gives implementors the version-correct path. Protocol
/// regeneration or a major `prost`/`tonic` bump is therefore a runtime breaking
/// change.
pub use rlmesh_proto;
