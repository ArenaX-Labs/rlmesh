//! The [`RuntimeHooks`] observer trait and its no-op default.

use async_trait::async_trait;
use prost::bytes::Bytes;

use super::{
    ActionReceivedEvent, EnvConnectedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, LogEvent,
    ModelConnectedEvent, ObservationEmittedEvent, SessionEndedEvent, SessionFailedEvent,
    SessionStartedEvent, StepCompletedEvent, TelemetrySnapshotEvent,
};

/// Error returned by a [`RuntimeHooks`] callback.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HookError {
    #[error("{0}")]
    Message(String),
}

/// A [`RuntimeHooks`] that ignores every event — the default when a session
/// installs no observer.
#[derive(Debug, Default)]
pub struct NoopRuntimeHooks;

#[async_trait]
impl RuntimeHooks for NoopRuntimeHooks {}

/// Callbacks the driver fans out around the `reset -> predict -> step` loop.
///
/// Lifecycle, progress, and log hooks are best-effort: the driver logs a failure
/// and keeps the route moving. The two transform hooks
/// ([`Self::transform_action`], [`Self::transform_observation`]) are fatal — a
/// failed transform leaves the next wire payload undefined, so the route fails
/// and shuts down. One shared instance serves every concurrent route, so each
/// event carries its route/session identity inline.
#[async_trait]
pub trait RuntimeHooks: Send + Sync {
    async fn env_connected(&self, _event: EnvConnectedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn model_connected(&self, _event: ModelConnectedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn session_started(&self, _event: SessionStartedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn episode_started(&self, _event: EpisodeStartedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn episode_completed(&self, _event: EpisodeCompletedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn action_received(&self, _event: ActionReceivedEvent) -> Result<(), HookError> {
        Ok(())
    }

    /// Fatal hook: a failed transform leaves the next action payload undefined,
    /// so the route fails rather than send something undefined to the env.
    async fn transform_action(
        &self,
        event: ActionReceivedEvent,
    ) -> Result<Option<Vec<Bytes>>, HookError> {
        Ok(event.action)
    }

    async fn step_completed(&self, _event: StepCompletedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn observation_emitted(&self, _event: ObservationEmittedEvent) -> Result<(), HookError> {
        Ok(())
    }

    /// Fatal hook; see [`Self::transform_action`].
    async fn transform_observation(
        &self,
        event: ObservationEmittedEvent,
    ) -> Result<Option<Vec<Bytes>>, HookError> {
        Ok(event.observation)
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
}
