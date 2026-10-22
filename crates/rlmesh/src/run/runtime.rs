//! The single-session runtime driver: shared `reset -> predict -> step`
//! semantics for one ready model/environment session. Public to bindings
//! through `binding::runtime`.

// Moved from the `rlmesh-runtime` crate, which did not document every
// public item.
#![allow(missing_docs)]

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
    Leg, LogEvent, LogLevel, ModelConnectedEvent, NoopRuntimeHooks, ObservationEmittedEvent,
    PayloadFacts, RefusingRelayPolicy, RelayAdvisoryEvent, RelayDecision, RelayPolicy,
    RuntimeEnvContext, RuntimeHooks, SessionEndedEvent, SessionFailedEvent, SessionStartedEvent,
    StepCompletedEvent, TelemetrySnapshotEvent,
};
pub use rlmesh_proto::EndpointPhases;
pub use rlmesh_spaces::{Advisory, AdvisorySeverity, DType};
pub use spec::{
    ENV_RESET_OPTIONS_KEY, PeerCeiling, RuntimeLimits, RuntimeReport, RuntimeSessionSpec,
    TRIAL_INDEX_OPTION, declares_reset_option, reset_options_for,
};
