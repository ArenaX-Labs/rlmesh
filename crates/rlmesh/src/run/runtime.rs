//! The single-session runtime driver: shared `reset -> predict -> step`
//! semantics for one ready model/environment session. Public to bindings
//! through `binding::runtime`.

// Moved from the `rlmesh-runtime` crate, which did not document every
// public item.
#![allow(missing_docs)]

mod driver;
pub mod hooks;
mod seeding;
pub mod spec;
pub mod telemetry;

mod episodes;
mod state;

pub use driver::{
    EagerScheduler, EnvResetRequest, EnvStepRequest, PredictScheduler, RuntimeDriver, RuntimeEnv,
    RuntimeEnvReset, RuntimeEnvStep, RuntimeError,
};
pub use hooks::{
    ActionReceivedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, HookError, LogEvent, LogLevel,
    NoopRuntimeHooks, ObservationEmittedEvent, RunDirective, RuntimeEnvContext, RuntimeHooks,
    SessionEndedEvent, SessionFailedEvent, SessionStartedEvent, StepCompletedEvent,
    TelemetrySnapshotEvent,
};
pub use rlmesh_proto::EndpointPhases;
pub use seeding::episode_seed;
pub use spec::{
    ENV_RENDER_PARAMS_KEY, ENV_REQUIRES_EPISODE_CAP_KEY, ENV_RESET_OPTIONS_KEY, RuntimeLimits,
    RuntimeReport, RuntimeSessionSpec, TRIAL_INDEX_OPTION, declares_reset_option,
    requires_episode_cap, reset_options_for,
};
