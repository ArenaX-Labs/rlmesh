use async_trait::async_trait;
use prost::bytes::Bytes;

use super::{
    ActionReceivedEvent, EnvConnectedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, LogEvent,
    ModelConnectedEvent, ObservationEmittedEvent, SessionEndedEvent, SessionFailedEvent,
    SessionStartedEvent, StepCompletedEvent, TelemetrySummaryEvent, TelemetryWindowEvent,
};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HookError {
    #[error("{0}")]
    Message(String),
}

#[derive(Debug, Default)]
pub struct NoopRuntimeHooks;

#[async_trait]
impl RuntimeHooks for NoopRuntimeHooks {}

#[async_trait]
pub trait RuntimeHooks: Send + Sync {
    // Lifecycle/progress/telemetry/log hooks are best-effort. The runtime logs
    // failures from these hooks and keeps the route moving.
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

    // Transform hooks are fatal. A failed transform means the runtime cannot
    // safely define the next wire payload, so the route fails and shuts down.
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

    // See transform_action: transform failures are fatal by design.
    async fn transform_observation(
        &self,
        event: ObservationEmittedEvent,
    ) -> Result<Option<Vec<Bytes>>, HookError> {
        Ok(event.observation)
    }

    async fn telemetry_window(&self, _event: TelemetryWindowEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn telemetry_summary(&self, _event: TelemetrySummaryEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn session_ended(&self, _event: SessionEndedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn session_failed(&self, _event: SessionFailedEvent) -> Result<(), HookError> {
        Ok(())
    }

    async fn log(&self, _event: LogEvent) -> Result<(), HookError> {
        Ok(())
    }
}
