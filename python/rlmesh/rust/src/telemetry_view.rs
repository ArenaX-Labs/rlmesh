//! Read-only Python views over the runtime's final session telemetry summary.
//!
//! The native model worker (`PyModel.run_local` / `run_local_for_episodes`)
//! returns the session's telemetry to the user as a [`PyTelemetrySummary`]. The
//! carried shape is the wire
//! [`TelemetryWindow`](rlmesh_proto::core::v1::TelemetryWindow) the runtime's
//! native summary serializes to (via `rlmesh::telemetry_summary_to_proto`), so
//! the Python user reads the same canonical telemetry the protocol carries:
//! throughput/byte rates plus the per-operation and model_wait/env_step/round_trip
//! timing rows with avg/p50/p95/p99.

use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rlmesh_proto::core::v1::{MetricSummary, TelemetryWindow, TimingSummary, Unit};

/// One aggregated duration row (e.g. the model_wait/env_step/round_trip split,
/// or a per-operation RPC/endpoint timing). Times are in milliseconds; the
/// percentiles are `None` when too few samples were observed to compute them.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(
    module = "rlmesh._rlmesh",
    name = "TelemetryTiming",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyTelemetryTiming {
    inner: TimingSummary,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyTelemetryTiming {
    /// Logical operation the row aggregates (e.g. `"model.predict"`, `"step"`).
    #[getter]
    fn operation(&self) -> String {
        self.inner.operation.clone()
    }

    /// Component the timing is attributed to (env or model component id).
    #[getter]
    fn component_id(&self) -> String {
        self.inner.component_id.clone()
    }

    /// Canonical metric handle (e.g. `"rpc.total"`, `"model.wait"`,
    /// `"round.trip"`). Stable across builds even when `key` is unrecognized.
    #[getter]
    fn key_name(&self) -> String {
        self.inner.key_name.clone()
    }

    #[getter]
    fn sample_count(&self) -> u64 {
        self.inner.sample_count
    }

    #[getter]
    fn avg_ms(&self) -> Option<f64> {
        self.inner.avg_ms
    }

    #[getter]
    fn p50_ms(&self) -> Option<f64> {
        self.inner.p50_ms
    }

    #[getter]
    fn p95_ms(&self) -> Option<f64> {
        self.inner.p95_ms
    }

    #[getter]
    fn p99_ms(&self) -> Option<f64> {
        self.inner.p99_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "TelemetryTiming(key_name={:?}, operation={:?}, sample_count={}, avg_ms={:?})",
            self.inner.key_name, self.inner.operation, self.inner.sample_count, self.inner.avg_ms
        )
    }
}

/// One aggregated non-duration metric row (byte counts / generic numbers, e.g.
/// `"batch.size"`). `unit` is the canonical unit name (`"UNIT_COUNT"`,
/// `"UNIT_BYTES"`, ...).
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(
    module = "rlmesh._rlmesh",
    name = "TelemetryMetric",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyTelemetryMetric {
    inner: MetricSummary,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyTelemetryMetric {
    #[getter]
    fn operation(&self) -> String {
        self.inner.operation.clone()
    }

    #[getter]
    fn component_id(&self) -> String {
        self.inner.component_id.clone()
    }

    #[getter]
    fn key_name(&self) -> String {
        self.inner.key_name.clone()
    }

    #[getter]
    fn unit(&self) -> String {
        Unit::try_from(self.inner.unit)
            .unwrap_or(Unit::Unspecified)
            .as_str_name()
            .to_string()
    }

    #[getter]
    fn sample_count(&self) -> u64 {
        self.inner.sample_count
    }

    #[getter]
    fn avg(&self) -> Option<f64> {
        self.inner.avg
    }

    #[getter]
    fn p50(&self) -> Option<f64> {
        self.inner.p50
    }

    #[getter]
    fn p95(&self) -> Option<f64> {
        self.inner.p95
    }

    #[getter]
    fn p99(&self) -> Option<f64> {
        self.inner.p99
    }

    fn __repr__(&self) -> String {
        format!(
            "TelemetryMetric(key_name={:?}, unit={}, sample_count={}, avg={:?})",
            self.inner.key_name,
            self.unit(),
            self.inner.sample_count,
            self.inner.avg
        )
    }
}

/// Read-only view over a session's final telemetry summary.
///
/// Returned by `Model.run_local` / `Model.run_local_for_episodes` (and the
/// native `PyModel` worker), or `None` when no telemetry window elapsed (a
/// zero-step or sub-window run). Always a session total
/// (`is_session_total == True`).
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(
    module = "rlmesh._rlmesh",
    name = "TelemetrySummary",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyTelemetrySummary {
    inner: TelemetryWindow,
}

impl PyTelemetrySummary {
    /// Wrap the wire telemetry window the runtime summary serialized to.
    pub(crate) fn new(window: TelemetryWindow) -> Self {
        Self { inner: window }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyTelemetrySummary {
    #[getter]
    fn session_id(&self) -> String {
        self.inner.session_id.clone()
    }

    #[getter]
    fn route_id(&self) -> String {
        self.inner.route_id.clone()
    }

    #[getter]
    fn env_component_id(&self) -> String {
        self.inner.env_component_id.clone()
    }

    #[getter]
    fn model_component_id(&self) -> String {
        self.inner.model_component_id.clone()
    }

    /// Seconds the summarized session spanned.
    #[getter]
    fn window_seconds(&self) -> u32 {
        self.inner.window_seconds
    }

    /// Total number of steps summarized.
    #[getter]
    fn sample_count(&self) -> u64 {
        self.inner.sample_count
    }

    #[getter]
    fn steps_per_second(&self) -> Option<f64> {
        self.inner.steps_per_second
    }

    #[getter]
    fn request_bytes_per_second(&self) -> Option<f64> {
        self.inner.request_bytes_per_second
    }

    #[getter]
    fn response_bytes_per_second(&self) -> Option<f64> {
        self.inner.response_bytes_per_second
    }

    /// Always `True`: this view only ever wraps a session total.
    #[getter]
    fn is_session_total(&self) -> bool {
        self.inner.is_session_total
    }

    /// Per-operation and phase-split duration rows, including the
    /// model_wait / env_step / round_trip split.
    #[getter]
    fn timings(&self) -> Vec<PyTelemetryTiming> {
        self.inner
            .timings
            .iter()
            .cloned()
            .map(|inner| PyTelemetryTiming { inner })
            .collect()
    }

    /// Non-duration metric rows (byte counts, generic gauges such as batch size).
    #[getter]
    fn metrics(&self) -> Vec<PyTelemetryMetric> {
        self.inner
            .metrics
            .iter()
            .cloned()
            .map(|inner| PyTelemetryMetric { inner })
            .collect()
    }

    /// Convenience accessor for one timing row by its canonical `key_name`
    /// (e.g. `"model.wait"`, `"env.step.phase"`, `"round.trip"`).
    fn timing(&self, key_name: &str) -> Option<PyTelemetryTiming> {
        self.inner
            .timings
            .iter()
            .find(|row| row.key_name == key_name)
            .cloned()
            .map(|inner| PyTelemetryTiming { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "TelemetrySummary(session_id={:?}, sample_count={}, window_seconds={}, steps_per_second={:?})",
            self.inner.session_id,
            self.inner.sample_count,
            self.inner.window_seconds,
            self.inner.steps_per_second
        )
    }
}
