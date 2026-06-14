use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use rlmesh_proto::core::v1::{OperationTelemetry, operation_metric};

use crate::hooks::{
    MetricKind, MetricSummary, RuntimeRouteContext, TelemetrySummaryEvent, TelemetryWindowEvent,
    TimingSummary,
};

use super::StepTimingSample;
use super::reservoir::{DurationReservoir, ValueReservoir};
use super::stats::{average_f64, average_ms, percentile_f64_samples, percentile_ms};

#[derive(Debug, Clone)]
pub(crate) struct TelemetryWindowAccumulator {
    pub(super) started_at: Instant,
    summary_started_at: Instant,
    step_count: u64,
    total_step_count: u64,
    request_bytes: u64,
    total_request_bytes: u64,
    response_bytes: u64,
    total_response_bytes: u64,
    timing_samples: BTreeMap<TimingKey, Vec<Duration>>,
    // `total_*` series accumulate over the whole (potentially unbounded)
    // session, so they are bounded reservoirs rather than raw Vecs to keep
    // memory constant while preserving representative summary statistics.
    total_timing_samples: BTreeMap<TimingKey, DurationReservoir>,
    // Non-duration metrics (byte counts / generic numbers) from
    // OperationTelemetry. Window samples are cleared each flush; total samples
    // persist for the session summary under the same reservoir bound.
    metric_samples: BTreeMap<MetricKey, ValueReservoir>,
    total_metric_samples: BTreeMap<MetricKey, ValueReservoir>,
    model_wait_samples: Vec<Duration>,
    total_model_wait_samples: DurationReservoir,
    env_step_samples: Vec<Duration>,
    total_env_step_samples: DurationReservoir,
    round_trip_samples: Vec<Duration>,
    total_round_trip_samples: DurationReservoir,
}

impl Default for TelemetryWindowAccumulator {
    fn default() -> Self {
        Self {
            started_at: Instant::now(),
            summary_started_at: Instant::now(),
            step_count: 0,
            total_step_count: 0,
            request_bytes: 0,
            total_request_bytes: 0,
            response_bytes: 0,
            total_response_bytes: 0,
            timing_samples: BTreeMap::new(),
            total_timing_samples: BTreeMap::new(),
            metric_samples: BTreeMap::new(),
            total_metric_samples: BTreeMap::new(),
            model_wait_samples: Vec::new(),
            total_model_wait_samples: DurationReservoir::default(),
            env_step_samples: Vec::new(),
            total_env_step_samples: DurationReservoir::default(),
            round_trip_samples: Vec::new(),
            total_round_trip_samples: DurationReservoir::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TimingKey {
    operation: String,
    component_id: String,
    name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct MetricKey {
    operation: String,
    component_id: String,
    name: String,
    kind: MetricKindKey,
}

/// Ordered, hashable mirror of [`MetricKind`] for use as a map key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum MetricKindKey {
    ByteCount,
    Number,
}

impl From<MetricKindKey> for MetricKind {
    fn from(kind: MetricKindKey) -> Self {
        match kind {
            MetricKindKey::ByteCount => MetricKind::ByteCount,
            MetricKindKey::Number => MetricKind::Number,
        }
    }
}

impl TelemetryWindowAccumulator {
    pub(crate) fn record_step(&mut self, sample: StepTimingSample<'_>) {
        self.step_count += 1;
        self.total_step_count += 1;
        self.request_bytes += sample.request_bytes as u64;
        self.total_request_bytes += sample.request_bytes as u64;
        self.response_bytes += sample.response_bytes as u64;
        self.total_response_bytes += sample.response_bytes as u64;
        self.model_wait_samples.push(sample.model_wait);
        self.total_model_wait_samples.push(sample.model_wait);
        self.env_step_samples.push(sample.env_step);
        self.total_env_step_samples.push(sample.env_step);
        self.round_trip_samples
            .push(sample.model_wait + sample.env_step);
        self.total_round_trip_samples
            .push(sample.model_wait + sample.env_step);
        // (env_step / model_wait totals pushed above.)
        self.record_timing(
            "model.predict",
            sample.model_component_id,
            "rpc.total",
            sample.model_wait,
        );
        self.record_timing(
            "env.step",
            sample.env_component_id,
            "rpc.total",
            sample.env_step,
        );
    }

    pub(crate) fn record_timing(
        &mut self,
        operation: impl Into<String>,
        component_id: impl Into<String>,
        name: impl Into<String>,
        duration: Duration,
    ) {
        let key = TimingKey {
            operation: operation.into(),
            component_id: component_id.into(),
            name: name.into(),
        };
        self.timing_samples
            .entry(key.clone())
            .or_default()
            .push(duration);
        self.total_timing_samples
            .entry(key)
            .or_default()
            .push(duration);
    }

    pub(crate) fn record_operation_telemetry(
        &mut self,
        fallback_component_id: &str,
        telemetry: Option<&OperationTelemetry>,
    ) {
        let Some(telemetry) = telemetry else {
            return;
        };
        let operation = if telemetry.operation.is_empty() {
            "unknown"
        } else {
            telemetry.operation.as_str()
        };
        let component_id = if telemetry.component_id.is_empty() {
            fallback_component_id
        } else {
            telemetry.component_id.as_str()
        };
        for metric in &telemetry.metrics {
            let name = if metric.name.is_empty() {
                "unspecified"
            } else {
                metric.name.as_str()
            };
            match metric.value {
                Some(operation_metric::Value::DurationNs(duration_ns)) => {
                    self.record_timing(
                        operation,
                        component_id,
                        name,
                        Duration::from_nanos(duration_ns),
                    );
                }
                Some(operation_metric::Value::ByteCount(bytes)) => {
                    self.record_metric(
                        operation,
                        component_id,
                        name,
                        MetricKindKey::ByteCount,
                        bytes as f64,
                    );
                }
                Some(operation_metric::Value::Number(number)) => {
                    self.record_metric(
                        operation,
                        component_id,
                        name,
                        MetricKindKey::Number,
                        number,
                    );
                }
                None => {}
            }
        }
    }

    fn record_metric(
        &mut self,
        operation: &str,
        component_id: &str,
        name: &str,
        kind: MetricKindKey,
        value: f64,
    ) {
        let key = MetricKey {
            operation: operation.to_string(),
            component_id: component_id.to_string(),
            name: name.to_string(),
            kind,
        };
        self.metric_samples
            .entry(key.clone())
            .or_default()
            .push(value);
        self.total_metric_samples
            .entry(key)
            .or_default()
            .push(value);
    }

    pub(crate) fn maybe_emit(
        &mut self,
        session_id: &str,
        route: RuntimeRouteContext,
        minimum_window: Duration,
    ) -> Option<TelemetryWindowEvent> {
        if self.started_at.elapsed() < minimum_window {
            return None;
        }
        self.flush(session_id, route)
    }

    pub(crate) fn flush(
        &mut self,
        session_id: &str,
        route: RuntimeRouteContext,
    ) -> Option<TelemetryWindowEvent> {
        let elapsed = self.started_at.elapsed();
        if elapsed.is_zero() || self.step_count == 0 {
            self.reset();
            return None;
        }

        let event = TelemetryWindowEvent {
            session_id: session_id.to_string(),
            route,
            window_seconds: elapsed.as_secs().max(1) as u32,
            sample_count: self.step_count,
            steps_per_second: Some(self.step_count as f64 / elapsed.as_secs_f64()),
            request_bytes_per_second: Some(self.request_bytes as f64 / elapsed.as_secs_f64()),
            response_bytes_per_second: Some(self.response_bytes as f64 / elapsed.as_secs_f64()),
            timings: timing_summaries(
                self.timing_samples
                    .iter()
                    .map(|(key, values)| (key, values.as_slice(), values.len() as u64)),
            ),
            metrics: metric_summaries(&self.metric_samples),
            env_latency_ms_avg: average_ms(&self.env_step_samples),
            env_latency_ms_p50: percentile_ms(&self.env_step_samples, 0.50),
            env_latency_ms_p95: percentile_ms(&self.env_step_samples, 0.95),
            env_latency_ms_p99: percentile_ms(&self.env_step_samples, 0.99),
            model_latency_ms_avg: average_ms(&self.model_wait_samples),
            model_latency_ms_p50: percentile_ms(&self.model_wait_samples, 0.50),
            model_latency_ms_p95: percentile_ms(&self.model_wait_samples, 0.95),
            model_latency_ms_p99: percentile_ms(&self.model_wait_samples, 0.99),
            round_trip_ms_avg: average_ms(&self.round_trip_samples),
            round_trip_ms_p50: percentile_ms(&self.round_trip_samples, 0.50),
            round_trip_ms_p95: percentile_ms(&self.round_trip_samples, 0.95),
            round_trip_ms_p99: percentile_ms(&self.round_trip_samples, 0.99),
            reconnects: 0,
            drops: 0,
        };
        self.reset();
        Some(event)
    }

    pub(crate) fn summary(
        &self,
        session_id: &str,
        route: RuntimeRouteContext,
    ) -> Option<TelemetrySummaryEvent> {
        let elapsed = self.summary_started_at.elapsed();
        if elapsed.is_zero() || self.total_step_count == 0 {
            return None;
        }

        Some(TelemetrySummaryEvent {
            session_id: session_id.to_string(),
            route,
            total_seconds: elapsed.as_secs().max(1) as u32,
            sample_count: self.total_step_count,
            steps_per_second: Some(self.total_step_count as f64 / elapsed.as_secs_f64()),
            request_bytes_per_second: Some(self.total_request_bytes as f64 / elapsed.as_secs_f64()),
            response_bytes_per_second: Some(
                self.total_response_bytes as f64 / elapsed.as_secs_f64(),
            ),
            timings: timing_summaries(self.total_timing_samples.iter().map(|(key, reservoir)| {
                // Report the true number of observed samples, not the bounded
                // reservoir size, so summary counts stay accurate.
                (key, reservoir.samples(), reservoir.seen())
            })),
            metrics: metric_summaries(&self.total_metric_samples),
            env_latency_ms_avg: average_ms(self.total_env_step_samples.samples()),
            env_latency_ms_p50: percentile_ms(self.total_env_step_samples.samples(), 0.50),
            env_latency_ms_p95: percentile_ms(self.total_env_step_samples.samples(), 0.95),
            env_latency_ms_p99: percentile_ms(self.total_env_step_samples.samples(), 0.99),
            model_latency_ms_avg: average_ms(self.total_model_wait_samples.samples()),
            model_latency_ms_p50: percentile_ms(self.total_model_wait_samples.samples(), 0.50),
            model_latency_ms_p95: percentile_ms(self.total_model_wait_samples.samples(), 0.95),
            model_latency_ms_p99: percentile_ms(self.total_model_wait_samples.samples(), 0.99),
            round_trip_ms_avg: average_ms(self.total_round_trip_samples.samples()),
            round_trip_ms_p50: percentile_ms(self.total_round_trip_samples.samples(), 0.50),
            round_trip_ms_p95: percentile_ms(self.total_round_trip_samples.samples(), 0.95),
            round_trip_ms_p99: percentile_ms(self.total_round_trip_samples.samples(), 0.99),
            reconnects: 0,
            drops: 0,
        })
    }

    /// Largest retained session-lifetime sample buffer, for memory-bound tests.
    #[cfg(test)]
    pub(super) fn max_total_sample_buffer(&self) -> usize {
        self.total_model_wait_samples
            .samples()
            .len()
            .max(self.total_env_step_samples.samples().len())
            .max(self.total_round_trip_samples.samples().len())
            .max(
                self.total_timing_samples
                    .values()
                    .map(|reservoir| reservoir.samples().len())
                    .max()
                    .unwrap_or(0),
            )
    }

    fn reset(&mut self) {
        self.started_at = Instant::now();
        self.step_count = 0;
        self.request_bytes = 0;
        self.response_bytes = 0;
        self.timing_samples.clear();
        self.metric_samples.clear();
        self.model_wait_samples.clear();
        self.env_step_samples.clear();
        self.round_trip_samples.clear();
    }
}

/// Build one [`TimingSummary`] per `(key, durations, sample_count)` triple.
///
/// Window samples report `sample_count` as the raw buffer length; session
/// totals report the reservoir's `seen()` count instead, so the caller supplies
/// the count alongside the (possibly bounded) duration slice.
fn timing_summaries<'a>(
    summaries: impl Iterator<Item = (&'a TimingKey, &'a [Duration], u64)>,
) -> Vec<TimingSummary> {
    summaries
        .map(|(key, durations, sample_count)| TimingSummary {
            operation: key.operation.clone(),
            component_id: key.component_id.clone(),
            name: key.name.clone(),
            sample_count,
            avg_ms: average_ms(durations),
            p50_ms: percentile_ms(durations, 0.50),
            p95_ms: percentile_ms(durations, 0.95),
            p99_ms: percentile_ms(durations, 0.99),
        })
        .collect()
}

fn metric_summaries(samples: &BTreeMap<MetricKey, ValueReservoir>) -> Vec<MetricSummary> {
    samples
        .iter()
        .map(|(key, reservoir)| {
            let values = reservoir.samples();
            MetricSummary {
                operation: key.operation.clone(),
                component_id: key.component_id.clone(),
                name: key.name.clone(),
                kind: key.kind.into(),
                // Report the true number of observed samples, not the bounded
                // reservoir size.
                sample_count: reservoir.seen(),
                avg: average_f64(values),
                p50: percentile_f64_samples(values, 0.50),
                p95: percentile_f64_samples(values, 0.95),
                p99: percentile_f64_samples(values, 0.99),
            }
        })
        .collect()
}
