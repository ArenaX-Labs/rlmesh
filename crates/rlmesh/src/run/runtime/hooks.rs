//! Runtime hook events and the observer trait the driver fans them out to.

mod events;
mod relay;
mod traits;

pub use events::{
    ActionReceivedEvent, EnvConnectedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, LogEvent,
    LogLevel, ModelConnectedEvent, ObservationEmittedEvent, RelayAdvisoryEvent, RuntimeEnvContext,
    SessionEndedEvent, SessionFailedEvent, SessionStartedEvent, StepCompletedEvent,
    TelemetrySnapshotEvent,
};
pub use relay::{Leg, PayloadFacts, RefusingRelayPolicy, RelayDecision, RelayPolicy};
pub use traits::{HookError, NoopRuntimeHooks, RuntimeHooks};
