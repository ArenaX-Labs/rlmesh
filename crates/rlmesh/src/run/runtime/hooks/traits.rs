//! The [`RuntimeHooks`] observer trait and its no-op default.

use async_trait::async_trait;
use prost::bytes::Bytes;

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
/// Lifecycle, progress and log hooks are best-effort: the driver logs a failure
/// and keeps the route moving. The transform hooks
/// ([`Self::transform_observation`], [`Self::transform_action`],
/// [`Self::transform_instruction`]) are fatal: a failed transform fails and
/// shuts down the route. One shared instance serves every concurrent route, so
/// each event carries its route/session identity.
///
/// A transform returns `None` to keep the payload, or leaf slabs of the same
/// space, lanes and leaf layout to send in its place. Each runs once per
/// payload before the observers see it, so `observation_emitted` and
/// `action_received` carry the transformed payload, and the original in
/// `raw_observation` / `raw_action` when a transform replaced it.
#[async_trait]
pub trait RuntimeHooks: Send + Sync {
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

    async fn transform_observation(
        &self,
        _event: &ObservationEmittedEvent,
    ) -> Result<Option<Vec<Bytes>>, HookError> {
        Ok(None)
    }

    async fn transform_action(
        &self,
        _event: &ActionReceivedEvent,
    ) -> Result<Option<Vec<Bytes>>, HookError> {
        Ok(None)
    }

    async fn transform_instruction(
        &self,
        _episode: &EpisodeStartedEvent,
        _instruction: &str,
    ) -> Result<Option<String>, HookError> {
        Ok(None)
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

    async fn on_telemetry(&self, _event: TelemetrySnapshotEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn session_failed(&self, _event: SessionFailedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn log(&self, _event: LogEvent) -> Result<(), HookError> {
        Ok(())
    }

    fn directive(&self) -> RunDirective {
        RunDirective::Continue
    }
}
