use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use prost::Message;
use rlmesh_proto::common::v1::MessageBytes;
use rlmesh_proto::core::v1::OperationTelemetry;
use rlmesh_proto::env::v1::{
    EpisodeMetadata, ResetRequest, ResetResponse, StepRequest, StepResponse,
};
use rlmesh_proto::model::v1::{CloseRouteRequest, PredictRequest, PredictResponse};
use rlmesh_proto::spaces::v1::SpaceValue;
use tokio_util::sync::CancellationToken;

use crate::episodes::EpisodeRecord;
use crate::hooks::{
    ActionReceivedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, HookError, LogEvent, LogLevel,
    NoopRuntimeHooks, ObservationEmittedEvent, RuntimeHooks, SessionEndedEvent, SessionFailedEvent,
    SessionStartedEvent, StepCompletedEvent, TelemetrySummaryEvent, TelemetryWindowEvent,
};
use crate::route::requests::RequestPhase;
use crate::spec::{RuntimeReport, RuntimeSessionSpec};
use crate::state::{RouteSnapshot, RouteState, StartedEpisode};
use crate::timing::{RuntimeTiming, StepTimingSample};

/// Type-erased structured error preserved as the `#[source]` of RPC failures.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RuntimeError {
    #[error("invalid runtime session spec: {0}")]
    InvalidSpec(String),

    #[error(
        "{operation} timed out on route {route_id} component {component_id} at runtime step {step} after {timeout:?}"
    )]
    OperationTimeout {
        route_id: String,
        component_id: String,
        operation: &'static str,
        step: i64,
        timeout: Duration,
    },

    #[error("route {route_id} cancelled at runtime step {step}: {reason}")]
    RouteCancelled {
        route_id: String,
        step: i64,
        reason: String,
    },

    #[error(
        "environment {operation} failed at runtime step {step}: {message}. If the source is 'transport error: connection closed', the environment server exited, crashed, or received SIGTERM before replying; inspect the environment container logs immediately before the runtime error timestamp"
    )]
    EnvRpc {
        operation: &'static str,
        step: i64,
        message: String,
        /// Whether the underlying transport error is retryable. Captured at
        /// construction by the adapter, which owns the structured error.
        recoverable: bool,
        /// The structured underlying error, preserved so callers can downcast
        /// or inspect the chain. `rlmesh-runtime` does not depend on
        /// `rlmesh-grpc`, so the concrete type is erased here.
        #[source]
        source: Option<BoxError>,
    },

    #[error("model endpoint {component_id} request failed: {message}")]
    ModelRpc {
        component_id: String,
        message: String,
        /// Whether the underlying error is retryable. Captured at construction.
        recoverable: bool,
        #[source]
        source: Option<BoxError>,
    },

    #[error(
        "model endpoint {component_id} returned mismatched route identity for request {request_id}"
    )]
    ModelRouteMismatch {
        component_id: String,
        request_id: String,
    },

    #[error("protocol error: {0}")]
    Protocol(String),

    #[error("runtime hook failed: {0}")]
    Hook(HookError),
}

impl RuntimeError {
    pub fn operation_timeout(
        route_id: impl Into<String>,
        component_id: impl Into<String>,
        operation: &'static str,
        step: i64,
        timeout: Duration,
    ) -> Self {
        Self::OperationTimeout {
            route_id: route_id.into(),
            component_id: component_id.into(),
            operation,
            step,
            timeout,
        }
    }

    pub fn route_cancelled(
        route_id: impl Into<String>,
        step: i64,
        reason: impl Into<String>,
    ) -> Self {
        Self::RouteCancelled {
            route_id: route_id.into(),
            step,
            reason: reason.into(),
        }
    }

    /// Constructs an [`EnvRpc`](Self::EnvRpc) error, capturing the underlying
    /// error's recoverability and preserving it as a structured `#[source]`.
    pub fn env_rpc<E>(operation: &'static str, step: i64, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::env_rpc_with_recoverability(operation, step, false, source)
    }

    /// Constructs an [`EnvRpc`](Self::EnvRpc) error with an explicit
    /// recoverability flag (e.g. from `GrpcError::is_recoverable`), preserving
    /// the structured source.
    pub fn env_rpc_with_recoverability<E>(
        operation: &'static str,
        step: i64,
        recoverable: bool,
        source: E,
    ) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::EnvRpc {
            operation,
            step,
            message: source.to_string(),
            recoverable,
            source: Some(Box::new(source)),
        }
    }

    /// Constructs a [`ModelRpc`](Self::ModelRpc) error, preserving the
    /// structured source. Recoverability defaults to `false`.
    pub fn model_rpc<E>(component_id: impl Into<String>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::model_rpc_with_recoverability(component_id, false, source)
    }

    /// Constructs a [`ModelRpc`](Self::ModelRpc) error with an explicit
    /// recoverability flag, preserving the structured source.
    pub fn model_rpc_with_recoverability<E>(
        component_id: impl Into<String>,
        recoverable: bool,
        source: E,
    ) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::ModelRpc {
            component_id: component_id.into(),
            message: source.to_string(),
            recoverable,
            source: Some(Box::new(source)),
        }
    }

    /// Whether this error is recoverable (retryable).
    ///
    /// For RPC failures this reflects the recoverability captured from the
    /// underlying transport error at construction. All other variants are
    /// treated as non-recoverable.
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::EnvRpc { recoverable, .. } | Self::ModelRpc { recoverable, .. } => *recoverable,
            _ => false,
        }
    }
}

pub struct RuntimeEnvReset {
    pub response: ResetResponse,
    pub telemetry: Option<OperationTelemetry>,
}

pub struct RuntimeEnvStep {
    pub response: StepResponse,
    pub telemetry: Option<OperationTelemetry>,
}

pub struct RuntimeModelPrediction {
    pub response: PredictResponse,
    pub telemetry: Option<OperationTelemetry>,
}

#[async_trait]
pub trait RuntimeEnv: Send {
    async fn reset(&mut self, request: ResetRequest) -> Result<RuntimeEnvReset, RuntimeError>;

    async fn step(&mut self, request: StepRequest) -> Result<RuntimeEnvStep, RuntimeError>;

    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}

#[async_trait]
pub trait RuntimeModel: Send {
    async fn predict(
        &mut self,
        request: PredictRequest,
    ) -> Result<RuntimeModelPrediction, RuntimeError>;

    async fn close_route(
        &mut self,
        _request: CloseRouteRequest,
        _timeout: Duration,
    ) -> Result<(), String> {
        Ok(())
    }
}

/// Default reason attributed to a cancellation when the caller does not supply
/// one via [`RuntimeDriver::run_with_cancellation_reason`].
const DEFAULT_CANCELLATION_REASON: &str = "cancelled by caller";

#[must_use = "a RuntimeDriver does nothing until one of its run methods is awaited"]
pub struct RuntimeDriver<E, M> {
    spec: RuntimeSessionSpec,
    env: E,
    model: M,
    hooks: Arc<dyn RuntimeHooks>,
    cancellation_reason: String,
}

impl<E, M> RuntimeDriver<E, M>
where
    E: RuntimeEnv,
    M: RuntimeModel,
{
    pub fn new(spec: RuntimeSessionSpec, env: E, model: M, hooks: Arc<dyn RuntimeHooks>) -> Self {
        Self {
            spec,
            env,
            model,
            hooks,
            cancellation_reason: DEFAULT_CANCELLATION_REASON.to_string(),
        }
    }

    pub fn without_hooks(spec: RuntimeSessionSpec, env: E, model: M) -> Self {
        Self::new(spec, env, model, Arc::new(NoopRuntimeHooks))
    }

    fn reset_seeds(&self, reset_generation: u64) -> Vec<i64> {
        self.spec
            .base_seed
            .map(|base_seed| {
                (0..self.spec.num_envs)
                    .map(|env_index| {
                        deterministic_reset_seed(
                            base_seed,
                            &self.spec.session_id,
                            &self.spec.route_id,
                            reset_generation,
                            env_index,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub async fn run(self) -> Result<RuntimeReport, RuntimeError> {
        self.run_with_cancellation(CancellationToken::new()).await
    }

    pub async fn run_with_cancellation(
        self,
        cancellation: CancellationToken,
    ) -> Result<RuntimeReport, RuntimeError> {
        self.run_with_cancellation_reason(cancellation, DEFAULT_CANCELLATION_REASON)
            .await
    }

    /// Runs the session, attributing any cancellation of `cancellation` to
    /// `reason`.
    ///
    /// The reason is carried into [`RuntimeError::RouteCancelled`], the
    /// `session_failed` hook event, and the `CloseRoute` reason, so callers
    /// (e.g. an owner that cancels for Ctrl+C, a deadline, or a sibling-route
    /// failure) can supply an accurate cause instead of a hardcoded one.
    pub async fn run_with_cancellation_reason(
        mut self,
        cancellation: CancellationToken,
        reason: impl Into<String>,
    ) -> Result<RuntimeReport, RuntimeError> {
        self.cancellation_reason = reason.into();
        self.spec.validate().map_err(RuntimeError::InvalidSpec)?;
        let mut state = RouteState::new(&self.spec);
        let result = self.run_loop(&mut state, &cancellation).await;
        if let Err(error) = &result {
            self.shutdown_after_failure(&mut state, error).await;
        }
        result
    }

    async fn run_loop(
        &mut self,
        state: &mut RouteState,
        cancellation: &CancellationToken,
    ) -> Result<RuntimeReport, RuntimeError> {
        let mut timings = RuntimeTiming::default();
        self.invoke_session_started(state, &self.spec.env_id).await;

        let reset_started = Instant::now();
        let mut reset_generation = 0_u64;
        let reset_timeout = self.spec.limits.env_reset_timeout;
        let reset_timeout_ms = self.spec.limits.env_reset_timeout_ms();
        let reset_seeds = self.reset_seeds(reset_generation);
        let reset_ok = await_runtime_operation(
            cancellation,
            reset_timeout,
            RuntimeError::operation_timeout(
                state.route_id(),
                state.env_component_id(),
                "env.reset",
                0,
                reset_timeout,
            ),
            self.cancelled_error(state, 0),
            self.env.reset(ResetRequest {
                seeds: reset_seeds,
                options: None,
                timeout_ms: reset_timeout_ms,
            }),
        )
        .await?;
        let reset_latency = reset_started.elapsed();
        timings.reset.record(reset_latency);
        timings
            .window
            .record_operation_telemetry(state.env_component_id(), reset_ok.telemetry.as_ref());
        self.invoke_log(
            state,
            LogLevel::Info,
            format!(
                "env reset complete in {:.0}ms ({} episode(s) ready)",
                reset_latency.as_secs_f64() * 1000.0,
                reset_ok
                    .response
                    .episode_ids
                    .iter()
                    .filter(|value| !value.is_empty())
                    .count()
            ),
        )
        .await;

        let reset_observation = value_bytes(reset_ok.response.observation.as_ref())?;
        let started_episodes = state.start_episodes(reset_ok.response.episode_ids, false);
        self.invoke_started_episodes(state, started_episodes).await;

        let mut reset_msg =
            state.predict_request(reset_observation.clone(), RequestPhase::ResetObservation);
        let mut reset_event =
            self.observation_event(state, state.snapshot(), true, reset_observation.clone());
        let transformed_reset_observation = self
            .invoke_transform_observation(reset_event.clone())
            .await?;
        reset_event.observation = transformed_reset_observation.clone();
        reset_msg.observation = transformed_reset_observation.map(bytes_value);
        self.invoke_observation_emitted(reset_event).await;

        let mut pending_observation_msg = reset_msg;

        loop {
            if cancellation.is_cancelled() {
                return Err(self.cancelled_error(state, state.snapshot().step));
            }

            let predict_snapshot = state.snapshot();
            let model_wait_started = Instant::now();
            let predict_timeout = self.spec.limits.model_predict_timeout;
            let expected_context = pending_observation_msg.context.clone();
            let action_msg = await_runtime_operation(
                cancellation,
                predict_timeout,
                RuntimeError::operation_timeout(
                    state.route_id(),
                    state.model_component_id(),
                    "model.predict",
                    predict_snapshot.step,
                    predict_timeout,
                ),
                self.cancelled_error(state, predict_snapshot.step),
                self.model.predict(pending_observation_msg),
            )
            .await?;
            if action_msg.response.context != expected_context {
                let request_id = expected_context
                    .as_ref()
                    .map(|context| context.request_id.clone())
                    .unwrap_or_default();
                return Err(RuntimeError::ModelRouteMismatch {
                    component_id: state.model_component_id().to_string(),
                    request_id,
                });
            }
            let model_action = value_bytes(action_msg.response.action.as_ref())?;
            let model_wait_latency = model_wait_started.elapsed();
            timings.model_wait.record(model_wait_latency);
            timings.window.record_operation_telemetry(
                state.model_component_id(),
                action_msg.telemetry.as_ref(),
            );

            let action_step = predict_snapshot.step + 1;
            let mut action_event = ActionReceivedEvent {
                session_id: state.session_id().to_string(),
                route: state.route_context(),
                episode_id: predict_snapshot.episode_id.clone(),
                episode_record_id: predict_snapshot.episode_record_id.clone(),
                episode_ids: predict_snapshot.episode_ids.clone(),
                episode_record_ids: predict_snapshot.episode_record_ids.clone(),
                step: action_step,
                env_index: predict_snapshot.env_index,
                action_space: self.spec.action_space_validated().clone(),
                action: model_action,
            };
            action_event.action = self.invoke_transform_action(action_event.clone()).await?;
            let request_bytes = action_event
                .action
                .as_ref()
                .map(|action| action.data.len())
                .unwrap_or_default();
            self.invoke_action_received(action_event.clone()).await;

            let env_step_started = Instant::now();
            let step_timeout = self.spec.limits.env_step_timeout;
            let step_timeout_ms = self.spec.limits.env_step_timeout_ms();
            let step_ok = await_runtime_operation(
                cancellation,
                step_timeout,
                RuntimeError::operation_timeout(
                    state.route_id(),
                    state.env_component_id(),
                    "env.step",
                    action_step,
                    step_timeout,
                ),
                self.cancelled_error(state, action_step),
                self.env.step(StepRequest {
                    action: action_event.action.map(bytes_value),
                    timeout_ms: step_timeout_ms,
                }),
            )
            .await?;
            let env_step_latency = env_step_started.elapsed();
            let step_observation = value_bytes(step_ok.response.observation.as_ref())?;
            timings.env_step.record(env_step_latency);
            timings
                .window
                .record_operation_telemetry(state.env_component_id(), step_ok.telemetry.as_ref());
            let response_bytes = step_observation
                .as_ref()
                .map(|obs| obs.data.len())
                .unwrap_or_default()
                + step_ok
                    .response
                    .infos
                    .as_ref()
                    .map(Message::encoded_len)
                    .unwrap_or_default();
            timings.window.record_step(StepTimingSample {
                model_wait: model_wait_latency,
                env_step: env_step_latency,
                request_bytes,
                response_bytes,
                env_component_id: state.env_component_id(),
                model_component_id: state.model_component_id(),
            });

            state.record_step();
            let step_snapshot = state.snapshot();
            self.invoke_step_completed(StepCompletedEvent {
                session_id: state.session_id().to_string(),
                route: state.route_context(),
                episode_id: step_snapshot.episode_id.clone(),
                episode_record_id: step_snapshot.episode_record_id.clone(),
                step: step_snapshot.step,
                env_index: step_snapshot.env_index,
                rewards: step_ok.response.rewards.clone(),
            })
            .await;

            if let Some(event) = timings.maybe_emit_window(
                state.session_id(),
                state.route_context(),
                self.spec.limits.telemetry_window,
            ) {
                self.invoke_telemetry_window(event).await;
            }

            if !step_ok.response.episode_ids.is_empty() {
                let started_episodes = state.observe_episode_ids(step_ok.response.episode_ids);
                self.invoke_started_episodes(state, started_episodes).await;
            }
            // The observation_emitted hook fires once per observation actually
            // sent to the model, post-transform, below (or at the initial
            // reset). Emitting the raw step observation here would expose
            // pre-transform bytes and, when the episode completes, an
            // observation the model never sees.

            self.emit_completed_episodes(state, &step_ok.response.completed_episodes)
                .await;

            if self
                .spec
                .max_episodes
                .is_some_and(|limit| state.total_episodes() >= limit as i64)
            {
                if let Some(event) = timings.flush_window(state.session_id(), state.route_context())
                {
                    self.invoke_telemetry_window(event).await;
                }
                if let Some(event) =
                    timings.telemetry_summary(state.session_id(), state.route_context())
                {
                    self.invoke_telemetry_summary(event).await;
                }
                let close_request = state.close_route_request("completed requested episodes");
                self.shutdown_terminal_route(state, "completed requested episodes", close_request)
                    .await;
                self.invoke_session_ended(
                    state,
                    "completed requested episodes",
                    state.total_steps(),
                    state.total_episodes(),
                )
                .await;
                timings.log_summary(state.total_steps(), state.total_episodes());
                return Ok(RuntimeReport {
                    session_id: state.session_id().to_string(),
                    route_id: self.spec.route_id.clone(),
                    total_steps: state.total_steps(),
                    total_episodes: state.total_episodes(),
                });
            }

            let need_reset = !step_ok.response.completed_episodes.is_empty();
            let (next_obs, phase, is_reset_msg) = if need_reset {
                let reset_started = Instant::now();
                reset_generation += 1;
                let step = state.snapshot().step;
                let reset_timeout = self.spec.limits.env_reset_timeout;
                let reset_timeout_ms = self.spec.limits.env_reset_timeout_ms();
                let reset_seeds = self.reset_seeds(reset_generation);
                let reset_ok = await_runtime_operation(
                    cancellation,
                    reset_timeout,
                    RuntimeError::operation_timeout(
                        state.route_id(),
                        state.env_component_id(),
                        "env.reset",
                        step,
                        reset_timeout,
                    ),
                    self.cancelled_error(state, step),
                    self.env.reset(ResetRequest {
                        seeds: reset_seeds,
                        options: None,
                        timeout_ms: reset_timeout_ms,
                    }),
                )
                .await?;
                timings.reset.record(reset_started.elapsed());
                timings.window.record_operation_telemetry(
                    state.env_component_id(),
                    reset_ok.telemetry.as_ref(),
                );
                let next_obs = value_bytes(reset_ok.response.observation.as_ref())?;
                let started_episodes = state.start_episodes(reset_ok.response.episode_ids, true);
                self.invoke_started_episodes(state, started_episodes).await;
                (next_obs, RequestPhase::ResetObservation, true)
            } else {
                (
                    step_observation.clone(),
                    RequestPhase::StepObservation,
                    false,
                )
            };

            let mut obs_msg = state.predict_request(next_obs.clone(), phase);
            let mut outgoing_observation_event =
                self.observation_event(state, state.snapshot(), is_reset_msg, next_obs);
            let transformed_observation = self
                .invoke_transform_observation(outgoing_observation_event.clone())
                .await?;
            outgoing_observation_event.observation = transformed_observation.clone();
            obs_msg.observation = transformed_observation.map(bytes_value);
            // Emit the transformed observation actually sent to the model, for
            // both step and reset observations, so hooks always see the same
            // payload model.predict receives.
            self.invoke_observation_emitted(outgoing_observation_event)
                .await;

            pending_observation_msg = obs_msg;
        }
    }

    async fn shutdown_after_failure(&mut self, state: &mut RouteState, error: &RuntimeError) {
        let reason = error.to_string();
        let request = state.close_route_request(reason.clone());
        self.shutdown_terminal_route(state, &reason, request).await;

        if let Err(err) = self
            .hooks
            .session_failed(SessionFailedEvent {
                session_id: state.session_id().to_string(),
                route: state.route_context(),
                reason,
            })
            .await
        {
            tracing::warn!("runtime hook session_failed failed: {err}");
        }
    }

    async fn shutdown_terminal_route(
        &mut self,
        state: &RouteState,
        reason: &str,
        request: CloseRouteRequest,
    ) {
        let timeout = self.spec.limits.service_close_timeout;
        // The timeout is also forwarded to the impls, but the driver enforces
        // it independently: a close impl that blocks (e.g. an RPC on a hung
        // connection) without honoring the deadline must not be able to hang
        // run()/run_with_cancellation() forever during shutdown.
        let model_close = tokio::time::timeout(timeout, self.model.close_route(request, timeout));
        if self.spec.close_env_on_end {
            let env_close = tokio::time::timeout(timeout, self.env.close(timeout));
            let (env_result, model_result) = tokio::join!(env_close, model_close);
            match env_result {
                Ok(Err(err)) => {
                    tracing::warn!(error = %err, "environment close failed during route shutdown");
                }
                Err(_) => {
                    tracing::warn!(
                        timeout_ms = timeout.as_millis(),
                        "environment close timed out during route shutdown; abandoning close"
                    );
                }
                Ok(Ok(())) => {}
            }
            log_model_close_result(model_result, reason, timeout);
            return;
        }

        tracing::debug!(
            route_id = %state.route_id(),
            reason,
            "skipping environment close for route; endpoint remains owned by the run"
        );
        log_model_close_result(model_close.await, reason, timeout);
    }

    fn cancelled_error(&self, state: &RouteState, step: i64) -> RuntimeError {
        RuntimeError::route_cancelled(state.route_id(), step, self.cancellation_reason.as_str())
    }

    async fn invoke_session_started(&self, state: &RouteState, env_id: &str) {
        if let Err(err) = self
            .hooks
            .session_started(SessionStartedEvent {
                session_id: state.session_id().to_string(),
                route: state.route_context(),
                env_id: env_id.to_string(),
            })
            .await
        {
            tracing::warn!("runtime hook session_started failed: {err}");
        }
    }

    async fn invoke_started_episodes(&self, state: &RouteState, episodes: Vec<StartedEpisode>) {
        for episode in episodes {
            self.invoke_episode_started(state, &episode.episode_id, &episode.record)
                .await;
        }
    }

    async fn invoke_episode_started(
        &self,
        state: &RouteState,
        episode_id: &str,
        record: &EpisodeRecord,
    ) {
        if let Err(err) = self
            .hooks
            .episode_started(EpisodeStartedEvent {
                session_id: state.session_id().to_string(),
                route: state.route_context(),
                episode_id: episode_id.to_string(),
                episode_record_id: record.record_id.clone(),
                episode_index: record.index,
                env_index: record.env_index,
                started_from_auto_reset: record.started_from_auto_reset,
            })
            .await
        {
            tracing::warn!("runtime hook episode_started failed: {err}");
        }
    }

    async fn emit_completed_episodes(&self, state: &mut RouteState, episodes: &[EpisodeMetadata]) {
        for completed in episodes {
            let record = state.complete_episode(&completed.episode_id);
            self.invoke_episode_completed(EpisodeCompletedEvent {
                session_id: state.session_id().to_string(),
                route: state.route_context(),
                episode_id: completed.episode_id.clone(),
                episode_record_id: record
                    .as_ref()
                    .map(|record| record.record_id.clone())
                    .unwrap_or_default(),
                episode_index: record.as_ref().map_or(0, |record| record.index),
                env_index: completed.env_index,
                step_count: completed.step_count,
                cumulative_reward: completed.cumulative_reward,
                terminated: completed.terminated,
                truncated: completed.truncated,
                duration_ms: completed.duration_ms,
                final_info: completed.final_info.clone(),
            })
            .await;
        }
    }

    async fn invoke_episode_completed(&self, event: EpisodeCompletedEvent) {
        if let Err(err) = self.hooks.episode_completed(event).await {
            tracing::warn!("runtime hook episode_completed failed: {err}");
        }
    }

    async fn invoke_action_received(&self, event: ActionReceivedEvent) {
        if let Err(err) = self.hooks.action_received(event).await {
            tracing::warn!("runtime hook action_received failed: {err}");
        }
    }

    async fn invoke_transform_action(
        &self,
        event: ActionReceivedEvent,
    ) -> Result<Option<MessageBytes>, RuntimeError> {
        match self.hooks.transform_action(event).await {
            Ok(action) => Ok(action),
            Err(err) => {
                tracing::warn!("runtime hook transform_action failed: {err}");
                Err(RuntimeError::Hook(err))
            }
        }
    }

    async fn invoke_step_completed(&self, event: StepCompletedEvent) {
        if let Err(err) = self.hooks.step_completed(event).await {
            tracing::warn!("runtime hook step_completed failed: {err}");
        }
    }

    async fn invoke_observation_emitted(&self, event: ObservationEmittedEvent) {
        if let Err(err) = self.hooks.observation_emitted(event).await {
            tracing::warn!("runtime hook observation_emitted failed: {err}");
        }
    }

    async fn invoke_transform_observation(
        &self,
        event: ObservationEmittedEvent,
    ) -> Result<Option<MessageBytes>, RuntimeError> {
        match self.hooks.transform_observation(event).await {
            Ok(observation) => Ok(observation),
            Err(err) => {
                tracing::warn!("runtime hook transform_observation failed: {err}");
                Err(RuntimeError::Hook(err))
            }
        }
    }

    async fn invoke_telemetry_window(&self, event: TelemetryWindowEvent) {
        if let Err(err) = self.hooks.telemetry_window(event).await {
            tracing::warn!("runtime hook telemetry_window failed: {err}");
        }
    }

    async fn invoke_telemetry_summary(&self, event: TelemetrySummaryEvent) {
        if let Err(err) = self.hooks.telemetry_summary(event).await {
            tracing::warn!("runtime hook telemetry_summary failed: {err}");
        }
    }

    async fn invoke_log(&self, state: &RouteState, level: LogLevel, message: impl Into<String>) {
        if let Err(err) = self
            .hooks
            .log(LogEvent {
                session_id: state.session_id().to_string(),
                route: state.route_context(),
                level,
                message: message.into(),
                source: Some("runtime".to_string()),
            })
            .await
        {
            tracing::warn!("runtime hook log failed: {err}");
        }
    }

    async fn invoke_session_ended(
        &self,
        state: &RouteState,
        reason: &str,
        total_steps: i64,
        total_episodes: i64,
    ) {
        if let Err(err) = self
            .hooks
            .session_ended(SessionEndedEvent {
                session_id: state.session_id().to_string(),
                route: state.route_context(),
                reason: reason.to_string(),
                total_steps,
                total_episodes,
            })
            .await
        {
            tracing::warn!("runtime hook session_ended failed: {err}");
        }
    }

    fn observation_event(
        &self,
        state: &RouteState,
        snapshot: RouteSnapshot,
        is_reset: bool,
        observation: Option<MessageBytes>,
    ) -> ObservationEmittedEvent {
        ObservationEmittedEvent {
            session_id: state.session_id().to_string(),
            route: state.route_context(),
            episode_id: snapshot.episode_id,
            episode_record_id: snapshot.episode_record_id,
            episode_ids: snapshot.episode_ids,
            episode_record_ids: snapshot.episode_record_ids,
            step: snapshot.step,
            env_index: snapshot.env_index,
            is_reset,
            num_envs: self.spec.num_envs as u32,
            observation_space: self.spec.observation_space_validated().clone(),
            observation,
        }
    }
}

fn deterministic_reset_seed(
    base_seed: i64,
    session_id: &str,
    route_id: &str,
    reset_generation: u64,
    env_index: usize,
) -> i64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    fn update(mut hash: u64, bytes: &[u8]) -> u64 {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    let mut hash = FNV_OFFSET;
    hash = update(hash, &base_seed.to_le_bytes());
    hash = update(hash, &[0xff]);
    hash = update(hash, session_id.as_bytes());
    hash = update(hash, &[0xfe]);
    hash = update(hash, route_id.as_bytes());
    hash = update(hash, &[0xfd]);
    hash = update(hash, &reset_generation.to_le_bytes());
    hash = update(hash, &[0xfc]);
    hash = update(hash, &(env_index as u64).to_le_bytes());
    (hash & i64::MAX as u64) as i64
}

async fn await_runtime_operation<T, F>(
    cancellation: &CancellationToken,
    timeout: Duration,
    timeout_error: RuntimeError,
    cancelled_error: RuntimeError,
    operation: F,
) -> Result<T, RuntimeError>
where
    F: Future<Output = Result<T, RuntimeError>>,
{
    tokio::select! {
        _ = cancellation.cancelled() => Err(cancelled_error),
        result = tokio::time::timeout(timeout, operation) => match result {
            Ok(result) => result,
            Err(_) => Err(timeout_error),
        },
    }
}

fn log_model_close_result(
    result: Result<Result<(), String>, tokio::time::error::Elapsed>,
    reason: &str,
    timeout: Duration,
) {
    match result {
        Ok(Err(err)) => {
            tracing::warn!(
                error = %err,
                reason,
                "model route close failed during route shutdown; relying on owner shutdown"
            );
        }
        Err(_) => {
            tracing::warn!(
                timeout_ms = timeout.as_millis(),
                reason,
                "model route close timed out during route shutdown; relying on owner shutdown"
            );
        }
        Ok(Ok(())) => {}
    }
}

fn bytes_value(value: MessageBytes) -> SpaceValue {
    SpaceValue { bytes: Some(value) }
}

fn value_bytes(payload: Option<&SpaceValue>) -> Result<Option<MessageBytes>, RuntimeError> {
    let Some(payload) = payload else {
        return Ok(None);
    };
    Ok(payload.bytes.clone())
}
