//! The [`RuntimeHooks`] observer trait and its no-op default.

use async_trait::async_trait;

use super::{
    ActionReceivedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, LogEvent,
    ObservationEmittedEvent, SessionEndedEvent, SessionFailedEvent, SessionStartedEvent,
    StepCompletedEvent, TelemetrySnapshotEvent,
};

/// Error returned by a [`RuntimeHooks`] callback.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HookError {
    #[error("{0}")]
    Message(String),
}

/// What an observer asks of the loop at a group's next step: a live viewer's
/// skip (end the episode, truncated) and quit (that, and start no more).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RunDirective {
    #[default]
    Continue,
    EndEpisode,
    Stop,
}

/// A [`RuntimeHooks`] that ignores every event — the default when a session
/// installs no observer.
#[derive(Debug, Default)]
pub struct NoopRuntimeHooks;

#[async_trait]
impl RuntimeHooks for NoopRuntimeHooks {}

/// Callbacks the driver fans out around the `reset -> predict -> step` loop.
///
/// Every hook is best-effort: the driver logs a failure and keeps the route
/// moving. One shared instance serves every concurrent route, so each event
/// carries its route/session identity inline.
#[async_trait]
pub trait RuntimeHooks: Send + Sync {
    async fn session_started(&self, _event: SessionStartedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn episode_started(&self, _event: EpisodeStartedEvent) -> Result<(), HookError> {
        Ok(())
    }

    /// The last event under its episode id: no observation or step of that
    /// episode follows it. Under `NEXT_STEP` autoreset the ended episode is
    /// observed and stepped once more (the autoreset step) before this fires.
    async fn episode_completed(&self, _event: EpisodeCompletedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn action_received(&self, _event: ActionReceivedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn step_completed(&self, _event: StepCompletedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn observation_emitted(&self, _event: ObservationEmittedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn session_ended(&self, _event: SessionEndedEvent) -> Result<(), HookError> {
        Ok(())
    }

    /// Live telemetry push, best-effort. A background ticker streams a `Window`
    /// snapshot (the live tier, cleared each `RuntimeLimits::telemetry_window`)
    /// while the session runs; one final cumulative `Session` snapshot is
    /// delivered at session end (the durable tier, also returned on
    /// `RuntimeReport.telemetry`). Branch on `event.snapshot.horizon` for window
    /// vs session. Dispatched from a separate task, so this may run concurrently
    /// with the other hooks — do not assume serialized delivery.
    async fn on_telemetry(&self, _event: TelemetrySnapshotEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn session_failed(&self, _event: SessionFailedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn log(&self, _event: LogEvent) -> Result<(), HookError> {
        Ok(())
    }

    /// Asked before each step's completions: an `EndEpisode` is taken once,
    /// a `Stop` holds. Only a loop that owns resets can end an episode early.
    fn directive(&self) -> RunDirective {
        RunDirective::Continue
    }
}
