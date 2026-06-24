//! Windowed / aggregate telemetry accumulation.
//!
//! # Canonical telemetry key space (cardinality invariant)
//!
//! Per-route telemetry series are keyed by `(operation, component_id, name)`.
//! To keep per-route memory constant, the `(operation, name)` pairs the runtime
//! emits are a closed, enumerated set (see `CANONICAL_TIMING_KEYS` /
//! `CANONICAL_METRIC_KEYS`):
//!   - `model.predict` × {`rpc.total`, `endpoint.total`}
//!   - `env.step`      × {`rpc.total`, `endpoint.total`, `batch.size`}
//!   - `env.reset`     × {`endpoint.total`}
//!
//! BANNED as key components (high-cardinality / per-event): `episode_id`,
//! `request_id`, `step`, `seed`, free-form peer labels. `component_id` is bounded
//! by the connected-component count and is fine. `record_timing` / `record_metric`
//! `debug_assert!` the pair is canonical.
//!
//! NOTE: this guard covers ONLY the fixed compile-time key set above. Future
//! dynamic user tags (`rlmesh.profile.span`) require a SEPARATE release-mode
//! per-route cap + overflow bucket; this debug-only guard does NOT satisfy that
//! requirement.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::hooks::{
    EpisodeTelemetryRollup, MetricKind, MetricSummary, RuntimeRouteContext, TelemetrySummaryEvent,
    TelemetryWindowEvent, TimingSummary,
};

use super::StepTimingSample;
use super::reservoir::{DurationReservoir, ValueReservoir};
use super::stats::{average_f64, average_ms, percentile_f64_samples, percentile_ms};

/// Canonical string handle for the endpoint-local op-duration metric. Mirrors
/// `MetricKey::METRIC_KEY_ENDPOINT_TOTAL`; the dual-write `key_name` the future
/// windowed channel emits alongside the enum.
const ENDPOINT_TOTAL_KEY_NAME: &str = "endpoint.total";

/// Canonical string handle for the per-step lane count (num_envs / slot count).
/// Mirrors `MetricKey::METRIC_KEY_BATCH_SIZE`.
const BATCH_SIZE_KEY_NAME: &str = "batch.size";

/// Canonical string handle for the host-observed per-op round-trip duration.
const RPC_TOTAL_KEY_NAME: &str = "rpc.total";

/// Canonical `(operation, name)` pairs for duration series. See the module doc.
const CANONICAL_TIMING_KEYS: &[(&str, &str)] = &[
    ("model.predict", RPC_TOTAL_KEY_NAME),
    ("model.predict", ENDPOINT_TOTAL_KEY_NAME),
    ("env.step", RPC_TOTAL_KEY_NAME),
    ("env.step", ENDPOINT_TOTAL_KEY_NAME),
    ("env.reset", ENDPOINT_TOTAL_KEY_NAME),
];

/// Canonical `(operation, name)` pairs for non-duration metric series.
const CANONICAL_METRIC_KEYS: &[(&str, &str)] = &[("env.step", BATCH_SIZE_KEY_NAME)];

/// Debug-only allowlist guard locking the duration key space. Never panics in
/// release. Covers ONLY today's fixed key set — dynamic user tags need a
/// separate release-mode cap + overflow bucket (see module doc).
fn assert_canonical_timing_key(operation: &str, name: &str) {
    debug_assert!(
        CANONICAL_TIMING_KEYS
            .iter()
            .any(|(op, n)| *op == operation && *n == name),
        "non-canonical timing key ({operation:?}, {name:?}); see window.rs canonical key space"
    );
}

/// Forward guard for the metric key space (today's only caller passes the
/// canonical pair). Never panics in release.
fn assert_canonical_metric_key(operation: &str, name: &str) {
    debug_assert!(
        CANONICAL_METRIC_KEYS
            .iter()
            .any(|(op, n)| *op == operation && *n == name),
        "non-canonical metric key ({operation:?}, {name:?}); see window.rs canonical key space"
    );
}

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
    // Non-duration metrics (byte counts / generic numbers). Window samples are
    // cleared each flush; total samples persist for the session summary under
    // the same reservoir bound.
    metric_samples: BTreeMap<MetricKey, ValueReservoir>,
    total_metric_samples: BTreeMap<MetricKey, ValueReservoir>,
    model_wait_samples: Vec<Duration>,
    total_model_wait_samples: DurationReservoir,
    env_step_samples: Vec<Duration>,
    total_env_step_samples: DurationReservoir,
    round_trip_samples: Vec<Duration>,
    total_round_trip_samples: DurationReservoir,
    // O(1) running per-episode scalars, folded in `record_step` /
    // `record_endpoint_total` and snapshotted by `episode_rollup` at episode
    // completion. Reset (not cleared) per episode — distinct from the window
    // Vecs (cleared per flush) and the session reservoirs; it deliberately
    // survives window flushes so an episode spanning multiple windows is whole.
    episode: EpisodeAccum,
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
            episode: EpisodeAccum::default(),
        }
    }
}

/// O(1) running per-episode aggregates. Folded in `record_step` (per-step model
/// wait, env step, bytes) and `record_endpoint_total` (per-step endpoint-local
/// op duration for `model.predict` / `env.step` only), reset by `rollup` at
/// episode completion. No per-step allocation; no min/max retention (no
/// consumer). NOT cleared by the window `reset()`.
#[derive(Debug, Clone, Default)]
struct EpisodeAccum {
    step_count: u64,
    model_wait_sum: Duration,
    env_step_sum: Duration,
    round_trip_sum: Duration,
    endpoint_op_sum: Duration,
    endpoint_op_count: u64,
    request_bytes_total: u64,
    response_bytes_total: u64,
}

impl EpisodeAccum {
    fn record_step(&mut self, model_wait: Duration, env_step: Duration, req: u64, resp: u64) {
        self.step_count += 1;
        self.model_wait_sum += model_wait;
        self.env_step_sum += env_step;
        self.round_trip_sum += model_wait + env_step;
        self.request_bytes_total += req;
        self.response_bytes_total += resp;
    }

    /// Fold one per-step endpoint-local op duration. Only the per-step ops
    /// (`model.predict`, `env.step`) are folded; `env.reset` is excluded so the
    /// average is a coherent per-op value, not a cross-op blend with a
    /// once-per-episode reset.
    fn record_endpoint_op(&mut self, duration: Duration) {
        self.endpoint_op_sum += duration;
        self.endpoint_op_count += 1;
    }

    /// Snapshot the accumulated scalars into a rollup and reset for the next
    /// episode. `episode_record_id` / `env_index` are filled by the driver from
    /// the completed episode's record.
    fn rollup(&mut self) -> EpisodeTelemetryRollup {
        let rollup = EpisodeTelemetryRollup {
            episode_record_id: String::new(),
            env_index: 0,
            step_count: self.step_count,
            model_latency_ms_avg: episode_avg_ms(self.model_wait_sum, self.step_count),
            env_latency_ms_avg: episode_avg_ms(self.env_step_sum, self.step_count),
            round_trip_ms_avg: episode_avg_ms(self.round_trip_sum, self.step_count),
            endpoint_op_ms_avg: episode_avg_ms(self.endpoint_op_sum, self.endpoint_op_count),
            request_bytes_total: self.request_bytes_total,
            response_bytes_total: self.response_bytes_total,
        };
        *self = Self::default();
        rollup
    }
}

/// Mean of a duration sum over a count, in milliseconds; `None` for an empty
/// accumulator (distinct from a real `0.0`).
fn episode_avg_ms(sum: Duration, count: u64) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some(sum.as_secs_f64() * 1000.0 / count as f64)
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
    kind: MetricKind,
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
        // Per-episode running scalars (O(1), allocation-free); reset at episode
        // completion via `episode_rollup`, NOT at window flush.
        self.episode.record_step(
            sample.model_wait,
            sample.env_step,
            sample.request_bytes as u64,
            sample.response_bytes as u64,
        );
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

    /// Snapshot the per-episode running scalars into an [`EpisodeTelemetryRollup`]
    /// and reset the accumulator for the next episode. The driver fills in
    /// `episode_record_id` / `env_index` from the completed episode's record.
    /// NOT cleared by `reset()` (window flush) — the accumulator spans the whole
    /// episode and survives window boundaries.
    pub(crate) fn episode_rollup(&mut self) -> EpisodeTelemetryRollup {
        self.episode.rollup()
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
        // Cardinality guard (debug/test only; never a release panic). See module doc.
        assert_canonical_timing_key(&key.operation, &key.name);
        self.timing_samples
            .entry(key.clone())
            .or_default()
            .push(duration);
        self.total_timing_samples
            .entry(key)
            .or_default()
            .push(duration);
    }

    /// Record the per-step endpoint-local op duration carried by the new hot
    /// scalar `JoinResponse.endpoint_total_ns`. The nested per-step telemetry
    /// message (with its always-empty `component_id` and dead `labels`) is gone:
    /// the runtime attributes by connection, so the authoritative `component_id`
    /// comes from route state, and the metric maps
    /// onto `MetricKey::EndpointTotal` (string handle "endpoint.total", the
    /// dual-write `key_name` for the future windowed channel).
    pub(crate) fn record_endpoint_total(
        &mut self,
        operation: &str,
        component_id: &str,
        endpoint_total_ns: Option<u64>,
    ) {
        let Some(duration_ns) = endpoint_total_ns else {
            return;
        };
        let duration = Duration::from_nanos(duration_ns);
        // Fold per-step ops only into the per-episode endpoint average; `env.reset`
        // is a distinct once-per-episode op and would blend the per-op average.
        if operation == "model.predict" || operation == "env.step" {
            self.episode.record_endpoint_op(duration);
        }
        self.record_timing(operation, component_id, ENDPOINT_TOTAL_KEY_NAME, duration);
    }

    /// Record the per-step lane count (num_envs / slot count) as a non-duration
    /// metric. Already computable per step, so it is promoted now (the design's
    /// `batch_size` row); `queue_depth` is deferred until a pipelined driver
    /// exists (today's driver loop is strictly serial). Attributed to the env
    /// component, which owns the lane count.
    pub(crate) fn record_batch_size(&mut self, component_id: &str, num_envs: u32) {
        self.record_metric(
            "env.step",
            component_id,
            BATCH_SIZE_KEY_NAME,
            MetricKind::Number,
            f64::from(num_envs),
        );
    }

    fn record_metric(
        &mut self,
        operation: &str,
        component_id: &str,
        name: &str,
        kind: MetricKind,
        value: f64,
    ) {
        // Cardinality guard (debug/test only); see module doc.
        assert_canonical_metric_key(operation, name);
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
                kind: key.kind,
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
