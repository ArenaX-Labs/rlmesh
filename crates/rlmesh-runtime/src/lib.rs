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
    LogEvent, LogLevel, MetricKind, MetricSummary, ModelConnectedEvent, NoopRuntimeHooks,
    ObservationEmittedEvent, RuntimeHookChain, RuntimeHooks, RuntimeRouteContext,
    SessionEndedEvent, SessionFailedEvent, SessionStartedEvent, StepCompletedEvent,
    TelemetrySummaryEvent, TelemetryWindowEvent, TimingSummary,
};
pub use spec::{RuntimeLimits, RuntimeReport, RuntimeSessionSpec};

/// Re-export of the protocol crate whose generated types appear in this
/// crate's public API.
///
/// The `RuntimeEnv`/`RuntimeModel` traits, the hook events, and
/// [`RuntimeSessionSpec`] are currently expressed directly in
/// `rlmesh_proto::*::v1` generated types (e.g. [`rlmesh_proto::env::v1::ResetRequest`],
/// [`rlmesh_proto::common::v1::MessageBytes`], [`rlmesh_proto::spaces::v1::SpaceSpec`]).
/// Downstream implementors must be able to name those types without taking an
/// independent dependency on `rlmesh-proto` (which would have to be kept at an
/// exactly matching version), so they are re-exported here as the sanctioned
/// path.
///
/// Coupling note: because the public surface is proto-generated, a protocol
/// regeneration or a major `prost`/`tonic` bump is a breaking change for this
/// crate and all `RuntimeHooks`/`RuntimeEnv`/`RuntimeModel` implementors. A
/// future crate-owned domain boundary (mirroring `rlmesh-spaces`) could
/// decouple them; until then, depend on this re-export rather than on
/// `rlmesh-proto` directly.
pub use rlmesh_proto;
