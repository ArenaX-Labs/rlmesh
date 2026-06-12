use async_trait::async_trait;
use rlmesh_proto::common::v1::MessageBytes;

use super::{
    ActionReceivedEvent, EnvConnectedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, HookError,
    LogEvent, ModelConnectedEvent, ObservationEmittedEvent, RuntimeHooks, SessionEndedEvent,
    SessionFailedEvent, SessionStartedEvent, StepCompletedEvent, TelemetrySummaryEvent,
    TelemetryWindowEvent,
};

/// Ordered runtime hook composition.
///
/// Lifecycle, progress, telemetry, and log events are sent to every hook.
/// Transform hooks are applied in order, with each hook receiving the payload
/// returned by the previous hook.
#[derive(Default)]
pub struct RuntimeHookChain {
    hooks: Vec<std::sync::Arc<dyn RuntimeHooks>>,
}

impl RuntimeHookChain {
    /// Creates a chain from hooks in invocation order.
    pub fn new(hooks: Vec<std::sync::Arc<dyn RuntimeHooks>>) -> Self {
        Self { hooks }
    }

    /// Creates an empty chain with the same behavior as `NoopRuntimeHooks`.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Returns the number of hooks in the chain.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// Returns true when the chain contains no hooks.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }
}

/// Broadcasts `$event` to every hook via `$method`, running all hooks even when
/// some fail and returning the first error (lifecycle/progress/telemetry/log
/// semantics).
macro_rules! fan_out_event {
    ($self:ident, $method:ident, $event:ident) => {{
        let mut first_error = None;
        for hook in &$self.hooks {
            if let Err(error) = hook.$method($event.clone()).await {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }};
}

/// Folds `$event`'s `$payload` field through every hook via `$method`, feeding
/// each hook's output into the next and short-circuiting on the first error
/// (transform semantics).
macro_rules! fold_transform {
    ($self:ident, $method:ident, $event:ident, $payload:ident) => {{
        let mut event = $event;
        for hook in &$self.hooks {
            event.$payload = hook.$method(event.clone()).await?;
        }
        Ok(event.$payload)
    }};
}

#[async_trait]
impl RuntimeHooks for RuntimeHookChain {
    async fn env_connected(&self, event: EnvConnectedEvent) -> Result<(), HookError> {
        fan_out_event!(self, env_connected, event)
    }

    async fn model_connected(&self, event: ModelConnectedEvent) -> Result<(), HookError> {
        fan_out_event!(self, model_connected, event)
    }

    async fn session_started(&self, event: SessionStartedEvent) -> Result<(), HookError> {
        fan_out_event!(self, session_started, event)
    }

    async fn episode_started(&self, event: EpisodeStartedEvent) -> Result<(), HookError> {
        fan_out_event!(self, episode_started, event)
    }

    async fn episode_completed(&self, event: EpisodeCompletedEvent) -> Result<(), HookError> {
        fan_out_event!(self, episode_completed, event)
    }

    async fn action_received(&self, event: ActionReceivedEvent) -> Result<(), HookError> {
        fan_out_event!(self, action_received, event)
    }

    async fn transform_action(
        &self,
        event: ActionReceivedEvent,
    ) -> Result<Option<MessageBytes>, HookError> {
        fold_transform!(self, transform_action, event, action)
    }

    async fn step_completed(&self, event: StepCompletedEvent) -> Result<(), HookError> {
        fan_out_event!(self, step_completed, event)
    }

    async fn observation_emitted(&self, event: ObservationEmittedEvent) -> Result<(), HookError> {
        fan_out_event!(self, observation_emitted, event)
    }

    async fn transform_observation(
        &self,
        event: ObservationEmittedEvent,
    ) -> Result<Option<MessageBytes>, HookError> {
        fold_transform!(self, transform_observation, event, observation)
    }

    async fn telemetry_window(&self, event: TelemetryWindowEvent) -> Result<(), HookError> {
        fan_out_event!(self, telemetry_window, event)
    }

    async fn telemetry_summary(&self, event: TelemetrySummaryEvent) -> Result<(), HookError> {
        fan_out_event!(self, telemetry_summary, event)
    }

    async fn session_ended(&self, event: SessionEndedEvent) -> Result<(), HookError> {
        fan_out_event!(self, session_ended, event)
    }

    async fn session_failed(&self, event: SessionFailedEvent) -> Result<(), HookError> {
        fan_out_event!(self, session_failed, event)
    }

    async fn log(&self, event: LogEvent) -> Result<(), HookError> {
        fan_out_event!(self, log, event)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use rlmesh_proto::common::v1::MessageBytes;
    use rlmesh_proto::spaces::v1::SpaceSpec;

    use super::*;
    use crate::hooks::{LogLevel, RuntimeRouteContext};

    struct RecordingHook {
        name: &'static str,
        calls: Arc<Mutex<Vec<String>>>,
        log_error: Option<&'static str>,
        action_suffix: Option<u8>,
        transform_error: Option<&'static str>,
    }

    impl RecordingHook {
        fn new(name: &'static str, calls: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                name,
                calls,
                log_error: None,
                action_suffix: None,
                transform_error: None,
            }
        }

        fn with_log_error(mut self, error: &'static str) -> Self {
            self.log_error = Some(error);
            self
        }

        fn with_action_suffix(mut self, suffix: u8) -> Self {
            self.action_suffix = Some(suffix);
            self
        }

        fn with_transform_error(mut self, error: &'static str) -> Self {
            self.transform_error = Some(error);
            self
        }

        fn record(&self, call: impl Into<String>) {
            self.calls
                .lock()
                .expect("calls mutex poisoned")
                .push(call.into());
        }
    }

    #[async_trait]
    impl RuntimeHooks for RecordingHook {
        async fn log(&self, event: LogEvent) -> Result<(), HookError> {
            self.record(format!("{}:log:{}", self.name, event.message));
            if let Some(error) = self.log_error {
                return Err(HookError::Message(error.to_string()));
            }
            Ok(())
        }

        async fn transform_action(
            &self,
            event: ActionReceivedEvent,
        ) -> Result<Option<MessageBytes>, HookError> {
            let data = event
                .action
                .as_ref()
                .map(|action| action.data.clone())
                .unwrap_or_default();
            self.record(format!("{}:action:{data:?}", self.name));
            if let Some(error) = self.transform_error {
                return Err(HookError::Message(error.to_string()));
            }
            Ok(event.action.map(|mut action| {
                if let Some(suffix) = self.action_suffix {
                    action.data.push(suffix);
                }
                action
            }))
        }
    }

    fn hook(hook: RecordingHook) -> Arc<dyn RuntimeHooks> {
        Arc::new(hook)
    }

    fn recorded(calls: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
        calls.lock().expect("calls mutex poisoned").clone()
    }

    fn log_event() -> LogEvent {
        LogEvent {
            session_id: "session".to_string(),
            route: RuntimeRouteContext::default(),
            level: LogLevel::Info,
            message: "hello".to_string(),
            source: None,
        }
    }

    fn action_event(data: Vec<u8>) -> ActionReceivedEvent {
        ActionReceivedEvent {
            session_id: "session".to_string(),
            route: RuntimeRouteContext::default(),
            episode_id: "episode".to_string(),
            episode_record_id: "episode-artifact".to_string(),
            episode_ids: vec!["episode".to_string()],
            episode_record_ids: vec!["episode-artifact".to_string()],
            step: 1,
            env_index: 0,
            action_space: SpaceSpec::default(),
            action: Some(MessageBytes { data }),
        }
    }

    #[tokio::test]
    async fn event_hooks_call_every_hook_and_return_first_error() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let chain = RuntimeHookChain::new(vec![
            hook(RecordingHook::new("first", calls.clone()).with_log_error("first failed")),
            hook(RecordingHook::new("second", calls.clone()).with_log_error("second failed")),
            hook(RecordingHook::new("third", calls.clone())),
        ]);

        let error = chain.log(log_event()).await.unwrap_err();

        assert_eq!(error.to_string(), "first failed");
        assert_eq!(
            recorded(&calls),
            vec!["first:log:hello", "second:log:hello", "third:log:hello"]
        );
    }

    #[tokio::test]
    async fn transform_hooks_run_in_order() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let chain = RuntimeHookChain::new(vec![
            hook(RecordingHook::new("first", calls.clone()).with_action_suffix(1)),
            hook(RecordingHook::new("second", calls.clone()).with_action_suffix(2)),
        ]);

        let action = chain
            .transform_action(action_event(vec![0]))
            .await
            .unwrap()
            .unwrap();

        assert_eq!(action.data, vec![0, 1, 2]);
        assert_eq!(
            recorded(&calls),
            vec!["first:action:[0]", "second:action:[0, 1]"]
        );
    }

    #[tokio::test]
    async fn transform_hooks_stop_on_first_error() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let chain = RuntimeHookChain::new(vec![
            hook(RecordingHook::new("first", calls.clone()).with_action_suffix(1)),
            hook(RecordingHook::new("second", calls.clone()).with_transform_error("bad action")),
            hook(RecordingHook::new("third", calls.clone()).with_action_suffix(3)),
        ]);

        let error = chain
            .transform_action(action_event(vec![0]))
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "bad action");
        assert_eq!(
            recorded(&calls),
            vec!["first:action:[0]", "second:action:[0, 1]"]
        );
    }

    #[tokio::test]
    async fn empty_chain_is_a_noop() {
        let chain = RuntimeHookChain::empty();

        chain.log(log_event()).await.unwrap();
        let action = chain
            .transform_action(action_event(vec![7]))
            .await
            .unwrap()
            .unwrap();

        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert_eq!(action.data, vec![7]);
    }
}
