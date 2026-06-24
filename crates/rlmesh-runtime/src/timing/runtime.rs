use std::time::Duration;

use crate::hooks::{RuntimeRouteContext, TelemetrySummaryEvent, TelemetryWindowEvent};

use super::{PhaseTiming, TelemetryWindowAccumulator};

#[derive(Debug, Clone, Default)]
pub(crate) struct RuntimeTiming {
    pub(crate) reset: PhaseTiming,
    pub(crate) model_wait: PhaseTiming,
    pub(crate) env_step: PhaseTiming,
    pub(crate) window: TelemetryWindowAccumulator,
}

pub(crate) struct StepTimingSample<'a> {
    pub(crate) model_wait: Duration,
    pub(crate) env_step: Duration,
    pub(crate) request_bytes: usize,
    pub(crate) response_bytes: usize,
    pub(crate) env_component_id: &'a str,
    pub(crate) model_component_id: &'a str,
}

impl RuntimeTiming {
    pub(crate) fn maybe_emit_window(
        &mut self,
        session_id: &str,
        route: RuntimeRouteContext,
        minimum_window: Duration,
    ) -> Option<TelemetryWindowEvent> {
        self.window.maybe_emit(session_id, route, minimum_window)
    }

    pub(crate) fn flush_window(
        &mut self,
        session_id: &str,
        route: RuntimeRouteContext,
    ) -> Option<TelemetryWindowEvent> {
        self.window.flush(session_id, route)
    }

    pub(crate) fn episode_rollup(&mut self) -> crate::hooks::EpisodeTelemetryRollup {
        self.window.episode_rollup()
    }

    pub(crate) fn telemetry_summary(
        &self,
        session_id: &str,
        route: RuntimeRouteContext,
    ) -> Option<TelemetrySummaryEvent> {
        self.window.summary(session_id, route)
    }

    pub(crate) fn log_summary(&self, total_steps: i64, total_episodes: i64) {
        tracing::info!(
            total_steps,
            total_episodes,
            reset_count = self.reset.count,
            reset_avg_ms = self.reset.avg_ms(),
            reset_min_ms = self.reset.min_ms(),
            reset_max_ms = self.reset.max_ms(),
            reset_total_ms = self.reset.total_ms(),
            model_wait_count = self.model_wait.count,
            model_wait_avg_ms = self.model_wait.avg_ms(),
            model_wait_min_ms = self.model_wait.min_ms(),
            model_wait_max_ms = self.model_wait.max_ms(),
            model_wait_total_ms = self.model_wait.total_ms(),
            env_step_count = self.env_step.count,
            env_step_avg_ms = self.env_step.avg_ms(),
            env_step_min_ms = self.env_step.min_ms(),
            env_step_max_ms = self.env_step.max_ms(),
            env_step_total_ms = self.env_step.total_ms(),
            "runtime timing summary"
        );
    }
}
