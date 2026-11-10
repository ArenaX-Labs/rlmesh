//! Running a route: the `run*` entry points, the route's event loop, and
//! its shutdown paths.

use std::sync::{Arc, Mutex};

use futures::stream::{FuturesUnordered, StreamExt};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use super::model_calls::release;
use super::observe::instruction_leaf;
use super::telemetry::{TelemetryTicker, lock_agg};
use super::{
    DEFAULT_CANCELLATION_REASON, EnvOutcome, EnvPhase, Group, PredictFuture, RuntimeDriver,
    RuntimeEnv, RuntimeError,
};
use crate::model::handler::ModelHandler;
use crate::run::runtime::hooks::{
    SessionEndedEvent, SessionFailedEvent, SessionStartedEvent, TelemetrySnapshotEvent,
};
use crate::run::runtime::spec::RuntimeReport;
use crate::run::runtime::state::RouteState;
use crate::run::runtime::telemetry::{Aggregator, Horizon};

impl<E, M> RuntimeDriver<E, M>
where
    E: RuntimeEnv + Clone + 'static,
    M: ModelHandler,
{
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
    /// `session_failed` hook event, and the `ReleaseAdapter` reason, so callers
    /// (e.g. an owner that cancels for Ctrl+C, a deadline, or a sibling-route
    /// failure) can supply an accurate cause instead of a hardcoded one.
    pub async fn run_with_cancellation_reason(
        mut self,
        cancellation: CancellationToken,
        reason: impl Into<String>,
    ) -> Result<RuntimeReport, RuntimeError> {
        self.cancellation_reason = reason.into();
        self.spec.validate().map_err(RuntimeError::InvalidSpec)?;
        // validate() confirmed both spaces are present; cache them as shared
        // Arcs so per-step hook events clone a pointer, not the whole spec.
        self.action_space = Arc::new(self.spec.action_space_validated().clone());
        self.observation_space = Arc::new(self.spec.observation_space_validated().clone());
        self.instruction_leaf = instruction_leaf(&self.contract);
        let mut state = RouteState::new(&self.spec);
        // Telemetry lives here, not in run_loop, so the final Session snapshot is
        // delivered on EVERY exit (including aborts). The background ticker only
        // ever pushes Window snapshots (the live tier); the cumulative Session
        // total is pushed once below and returned on the report (the durable
        // tier), so a late ticker tick cannot race or supersede it.
        let telemetry = Arc::new(Mutex::new(Aggregator::default()));
        // A zero window disables live streaming (it would otherwise be a 1ms hot
        // loop); the final session push below still fires.
        let ticker = (!self.spec.limits.telemetry_window.is_zero()).then(|| {
            TelemetryTicker::spawn(
                Arc::clone(&telemetry),
                Arc::clone(&self.hooks),
                self.spec.limits.telemetry_window,
                state.session_id().to_string(),
                state.env_context(),
            )
        });
        let mut env_ops: JoinSet<EnvOutcome<E>> = JoinSet::new();
        let mut predicts: FuturesUnordered<PredictFuture<'_>> = FuturesUnordered::new();
        let result = self
            .run_loop(
                &mut state,
                &cancellation,
                &telemetry,
                &mut env_ops,
                &mut predicts,
            )
            .await;
        // Whatever ended the loop, let the predicts still in flight land
        // before the route releases the model; a hung predict is abandoned
        // after the close timeout.
        env_ops.abort_all();
        let mut abandoned = false;
        if !predicts.is_empty() {
            let drain = async { while predicts.next().await.is_some() {} };
            if tokio::time::timeout(self.spec.limits.service_close_timeout, drain)
                .await
                .is_err()
            {
                abandoned = true;
                tracing::warn!("predict still in flight at route end; abandoned");
            }
        }
        drop(predicts);
        // A route ending mid-episode (a failure, a cancellation, a sibling
        // lane cut by the episode budget) still has episodes the model has
        // predicted on and never heard the end of; its per-episode state (a
        // Python model's episode store) ends through the same hook a completed
        // episode's does, so end them here, once, before the release. Together
        // with the evictions held behind those predicts and any queued since
        // the last loop turn, they go out on the shutdown budget, cancelled or
        // not — but never to an endpoint that just missed the close deadline:
        // it is provably hung, so the flush would only burn its own deadline,
        // and the deadline-bounded release below frees that state anyway.
        if abandoned {
            self.pending_evictions.clear();
            self.held_evictions.clear();
        } else {
            self.pending_evictions
                .extend(self.held_evictions.drain(..).map(|(_, id)| id));
            self.pending_evictions.extend(state.end_live_episodes());
            self.flush_evictions(
                &mut state,
                &telemetry,
                None,
                self.spec.limits.service_close_timeout,
            )
            .await;
        }
        // Stop the ticker (it only emits Window snapshots, so it cannot contend
        // this Session push), then deliver the durable session total exactly once
        // on every exit path.
        drop(ticker);
        let final_snapshot = lock_agg(&telemetry).snapshot(Horizon::Session);
        fan_out_event!(
            self,
            on_telemetry,
            TelemetrySnapshotEvent {
                session_id: state.session_id().to_string(),
                route: state.env_context(),
                snapshot: final_snapshot.clone(),
            }
        );
        match result {
            Ok(reason) => {
                self.shutdown_terminal_route(&state, reason).await;
                fan_out_event!(
                    self,
                    session_ended,
                    SessionEndedEvent {
                        session_id: state.session_id().to_string(),
                        route: state.env_context(),
                        reason: reason.to_string(),
                        total_steps: state.total_steps(),
                        total_episodes: state.total_episodes(),
                    }
                );
                Ok(RuntimeReport {
                    session_id: state.session_id().to_string(),
                    env_id: self.spec.env_id.clone(),
                    total_steps: state.total_steps(),
                    total_episodes: state.total_episodes(),
                    episodes: state.take_episode_summaries(),
                    telemetry: final_snapshot,
                    advisories: Vec::new(),
                })
            }
            Err(error) => {
                self.shutdown_after_failure(&mut state, &error).await;
                Err(error)
            }
        }
    }

    /// Session/route-level span: lets an OpenTelemetry subscriber attach
    /// to `rlmesh.route`; inert under the default
    /// subscriber. Created once per session, not per step; `skip_all` records only
    /// the cheap ids. Any future per-step span MUST be trace-level + target-gated.
    #[tracing::instrument(
        name = "rlmesh.route",
        level = "info",
        skip_all,
        fields(
            session_id = %state.session_id(),
            env_id = %self.spec.env_id,
            num_envs = self.spec.num_envs,
            lanes = self.spec.subset_step,
        ),
    )]
    async fn run_loop<'m>(
        &mut self,
        state: &mut RouteState,
        cancellation: &CancellationToken,
        telemetry: &Arc<Mutex<Aggregator>>,
        env_ops: &mut JoinSet<EnvOutcome<E>>,
        predicts: &mut FuturesUnordered<PredictFuture<'m>>,
    ) -> Result<&'static str, RuntimeError>
    where
        M: 'm,
    {
        fan_out_event!(
            self,
            session_started,
            SessionStartedEvent {
                session_id: state.session_id().to_string(),
                route: state.env_context(),
                env_id: self.spec.env_id.clone(),
                env_contract: Arc::new(self.spec.env_contract.clone()),
            }
        );

        let mut groups = self.groups();
        let group_count = groups.len();
        for gid in 0..group_count {
            self.begin_reset(gid, &mut groups, state, env_ops, true);
        }

        loop {
            if cancellation.is_cancelled() {
                return Err(self.cancelled_error(state, &groups));
            }
            if groups.iter().all(|group| group.phase == EnvPhase::Idle)
                && env_ops.is_empty()
                && predicts.is_empty()
            {
                return Ok("completed requested episodes");
            }
            // A predict future parked in `predicts` is not polled while this
            // await runs, and a local model's eviction shares its handler lock
            // with predict: flushing here would wait out the predict timeout.
            if predicts.is_empty() {
                self.flush_evictions(
                    state,
                    telemetry,
                    Some(cancellation),
                    self.spec.limits.model_predict_timeout,
                )
                .await;
            }
            self.dispatch_steps(&mut groups, state, env_ops, telemetry)
                .await?;
            // With nothing else in flight, a waiting group must be predicted now
            // or the route would sit forever; the scheduler's plan is advisory
            // only while something else can wake the loop.
            let force = env_ops.is_empty() && predicts.is_empty();
            self.dispatch_predict(&mut groups, state, predicts, force);
            if env_ops.is_empty() && predicts.is_empty() {
                // Nothing in flight and no group could be advanced: a live group
                // with neither a frame nor an observation, which the transitions
                // above never produce.
                return Err(RuntimeError::Protocol(format!(
                    "route {} stalled with no operation in flight",
                    state.env_id()
                )));
            }

            tokio::select! {
                _ = cancellation.cancelled() => {
                    return Err(self.cancelled_error(state, &groups));
                }
                Some(joined) = env_ops.join_next() => {
                    let outcome = joined.map_err(|error| RuntimeError::Protocol(format!(
                        "env operation task failed: {error}"
                    )))?;
                    self.on_env_outcome(outcome, &mut groups, state, env_ops, telemetry)
                        .await?;
                }
                Some(outcome) = predicts.next(), if !predicts.is_empty() => {
                    self.on_predict_outcome(outcome, &mut groups, state, telemetry)?;
                }
            }
        }
    }

    async fn shutdown_after_failure(&mut self, state: &mut RouteState, error: &RuntimeError) {
        let reason = error.to_string();
        self.shutdown_terminal_route(state, &reason).await;

        if let Err(err) = self
            .hooks
            .session_failed(SessionFailedEvent {
                session_id: state.session_id().to_string(),
                route: state.env_context(),
                reason,
            })
            .await
        {
            tracing::warn!("runtime hook session_failed failed: {err}");
        }
    }

    /// Releases the model's route and, when the session owns it, closes the
    /// env, each within the close timeout: the driver enforces it, so a
    /// close that blocks (an RPC on a hung connection) cannot hang the run's
    /// shutdown.
    async fn shutdown_terminal_route(&mut self, state: &RouteState, reason: &str) {
        let timeout = self.spec.limits.service_close_timeout;
        let released = release(&self.model, state.env_id().to_string());
        match tokio::time::timeout(timeout, released).await {
            Ok(Err(err)) => tracing::warn!(
                error = %err,
                reason,
                "model route release failed during route shutdown"
            ),
            Err(_) => tracing::warn!(
                timeout_ms = timeout.as_millis(),
                reason,
                "model route release timed out during route shutdown; abandoning it"
            ),
            Ok(Ok(())) => {}
        }
        if !self.spec.close_env_on_end {
            tracing::debug!(
                env_id = %state.env_id(),
                reason,
                "skipping environment close; the endpoint remains owned by the run"
            );
            return;
        }
        match tokio::time::timeout(timeout, self.env.close(timeout)).await {
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
    }

    fn cancelled_error(&self, state: &RouteState, groups: &[Group<E>]) -> RuntimeError {
        let step = groups
            .iter()
            .map(|group| state.snapshot_at(&group.positions).step)
            .max()
            .unwrap_or(0);
        RuntimeError::route_cancelled(state.env_id(), step, self.cancellation_reason.as_str())
    }
}
