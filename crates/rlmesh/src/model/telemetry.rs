//! Host -> proto serialization for the runtime's native session telemetry.
//!
//! The runtime computes a final session-total summary natively (the
//! transport-agnostic [`TelemetrySummaryEvent`]) and hands it back on
//! [`RuntimeReport::telemetry_summary`](rlmesh_runtime::RuntimeReport). This is
//! the canonical mapping of that host summary onto the wire
//! [`TelemetryWindow`](rlmesh_proto::core::v1::TelemetryWindow) shape
//! (`is_session_total = true`), used to surface telemetry to callers (e.g. the
//! Python worker run result).

use rlmesh_proto::core::v1::{
    MetricKey, MetricSummary as ProtoMetricSummary, TelemetryWindow,
    TimingSummary as ProtoTimingSummary, Unit,
};
use rlmesh_runtime::{
    MetricKind, MetricSummary as HostMetricSummary, TelemetrySummaryEvent,
    TimingSummary as HostTimingSummary,
};

/// Canonical string handle the runtime stamps on the model-wait phase timing
/// (mirrors [`MetricKey::ModelWait`]). The runtime carries this split on the
/// summary's dedicated `model_latency_*` fields rather than a named timing row,
/// so it has no host `name`; this is the key_name the wire row gets.
const MODEL_WAIT_KEY_NAME: &str = "model.wait";
/// Canonical string handle for the env-step phase timing
/// ([`MetricKey::EnvStep`]); see [`MODEL_WAIT_KEY_NAME`].
const ENV_STEP_KEY_NAME: &str = "env.step.phase";
/// Canonical string handle for the per-step round-trip timing
/// ([`MetricKey::RoundTrip`]); see [`MODEL_WAIT_KEY_NAME`].
const ROUND_TRIP_KEY_NAME: &str = "round.trip";

/// Serialize the runtime's native session-total [`TelemetrySummaryEvent`] onto
/// the wire [`TelemetryWindow`] (`is_session_total = true`).
///
/// This is the canonical host->proto telemetry mapping:
///
/// - The host's per-operation `timings` rows map 1:1 onto proto
///   [`TimingSummary`](ProtoTimingSummary) rows, with the host `name` carried as
///   `key_name` and resolved to a [`MetricKey`] where it is a recognized handle
///   (else [`MetricKey::Unspecified`], the open-enum opaque case).
/// - The host's `metrics` rows map onto proto
///   [`MetricSummary`](ProtoMetricSummary) rows; the host [`MetricKind`] picks
///   the [`Unit`] (`ByteCount` -> bytes, `Number` -> count).
/// - The summary's dedicated model-wait / env-step / round-trip phase split
///   (carried on `*_latency_*` / `round_trip_*` fields, not as named timing
///   rows) is appended as up to three extra timing rows keyed
///   [`MetricKey::ModelWait`] / [`MetricKey::EnvStep`] / [`MetricKey::RoundTrip`],
///   so the wire `timings` includes the documented model_wait/env_step/round_trip
///   rows. A phase with no timing at all (average and every percentile absent) is
///   omitted rather than emitted as an empty placeholder row.
/// - `Option<f64>` percentile fields preserve their present/absent distinction.
pub fn telemetry_summary_to_proto(summary: &TelemetrySummaryEvent) -> TelemetryWindow {
    let mut timings: Vec<ProtoTimingSummary> =
        summary.timings.iter().map(host_timing_to_proto).collect();
    // Append the phase split the runtime keeps on dedicated summary fields
    // rather than as named per-op timing rows. The proto `timings` list is
    // documented to include these model_wait/env_step/round_trip rows.
    timings.extend(phase_timing_row(
        "model.predict",
        &summary.route.model_component_id,
        MetricKey::ModelWait,
        MODEL_WAIT_KEY_NAME,
        summary.sample_count,
        summary.model_latency_ms_avg,
        summary.model_latency_ms_p50,
        summary.model_latency_ms_p95,
        summary.model_latency_ms_p99,
    ));
    timings.extend(phase_timing_row(
        "env.step",
        &summary.route.env_component_id,
        MetricKey::EnvStep,
        ENV_STEP_KEY_NAME,
        summary.sample_count,
        summary.env_latency_ms_avg,
        summary.env_latency_ms_p50,
        summary.env_latency_ms_p95,
        summary.env_latency_ms_p99,
    ));
    timings.extend(phase_timing_row(
        "step",
        &summary.route.env_component_id,
        MetricKey::RoundTrip,
        ROUND_TRIP_KEY_NAME,
        summary.sample_count,
        summary.round_trip_ms_avg,
        summary.round_trip_ms_p50,
        summary.round_trip_ms_p95,
        summary.round_trip_ms_p99,
    ));

    TelemetryWindow {
        session_id: summary.session_id.clone(),
        route_id: summary.route.route_id.clone(),
        env_component_id: summary.route.env_component_id.clone(),
        model_component_id: summary.route.model_component_id.clone(),
        window_seconds: summary.total_seconds,
        sample_count: summary.sample_count,
        steps_per_second: summary.steps_per_second,
        request_bytes_per_second: summary.request_bytes_per_second,
        response_bytes_per_second: summary.response_bytes_per_second,
        timings,
        metrics: summary.metrics.iter().map(host_metric_to_proto).collect(),
        is_session_total: true,
    }
}

fn host_timing_to_proto(row: &HostTimingSummary) -> ProtoTimingSummary {
    ProtoTimingSummary {
        operation: row.operation.clone(),
        component_id: row.component_id.clone(),
        key: metric_key_for_name(&row.name) as i32,
        key_name: row.name.clone(),
        sample_count: row.sample_count,
        avg_ms: row.avg_ms,
        p50_ms: row.p50_ms,
        p95_ms: row.p95_ms,
        p99_ms: row.p99_ms,
    }
}

/// Build a phase-split timing row, or `None` when the phase carries no timing at
/// all (average and every percentile absent). Skipping the empty row keeps the
/// wire `timings` list free of placeholder rows that would carry only a key_name
/// and a sample count but no measurement.
#[allow(clippy::too_many_arguments)]
fn phase_timing_row(
    operation: &str,
    component_id: &str,
    key: MetricKey,
    key_name: &str,
    sample_count: u64,
    avg_ms: Option<f64>,
    p50_ms: Option<f64>,
    p95_ms: Option<f64>,
    p99_ms: Option<f64>,
) -> Option<ProtoTimingSummary> {
    if avg_ms.is_none() && p50_ms.is_none() && p95_ms.is_none() && p99_ms.is_none() {
        return None;
    }
    Some(ProtoTimingSummary {
        operation: operation.to_string(),
        component_id: component_id.to_string(),
        key: key as i32,
        key_name: key_name.to_string(),
        sample_count,
        avg_ms,
        p50_ms,
        p95_ms,
        p99_ms,
    })
}

fn host_metric_to_proto(row: &HostMetricSummary) -> ProtoMetricSummary {
    ProtoMetricSummary {
        operation: row.operation.clone(),
        component_id: row.component_id.clone(),
        key: metric_key_for_name(&row.name) as i32,
        key_name: row.name.clone(),
        unit: unit_for_kind(row.kind) as i32,
        sample_count: row.sample_count,
        avg: row.avg,
        p50: row.p50,
        p95: row.p95,
        p99: row.p99,
    }
}

/// Resolve a host metric/timing `name` to its stable [`MetricKey`]. An
/// unrecognized name resolves to [`MetricKey::Unspecified`] (proto open-enum
/// opaque case); the canonical string still rides on `key_name`.
fn metric_key_for_name(name: &str) -> MetricKey {
    match name {
        "endpoint.total" => MetricKey::EndpointTotal,
        "rpc.total" => MetricKey::RpcTotal,
        "batch.size" => MetricKey::BatchSize,
        MODEL_WAIT_KEY_NAME => MetricKey::ModelWait,
        ENV_STEP_KEY_NAME => MetricKey::EnvStep,
        ROUND_TRIP_KEY_NAME => MetricKey::RoundTrip,
        _ => MetricKey::Unspecified,
    }
}

fn unit_for_kind(kind: MetricKind) -> Unit {
    match kind {
        MetricKind::ByteCount => Unit::Bytes,
        MetricKind::Number => Unit::Count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_runtime::RuntimeRouteContext;

    fn route() -> RuntimeRouteContext {
        RuntimeRouteContext {
            route_id: "route-1".to_string(),
            env_component_id: "env-1".to_string(),
            model_component_id: "model-1".to_string(),
        }
    }

    fn sample_summary() -> TelemetrySummaryEvent {
        TelemetrySummaryEvent {
            session_id: "sess-1".to_string(),
            route: route(),
            total_seconds: 12,
            sample_count: 100,
            steps_per_second: Some(8.5),
            request_bytes_per_second: Some(1024.0),
            response_bytes_per_second: Some(2048.0),
            timings: vec![
                HostTimingSummary {
                    operation: "model.predict".to_string(),
                    component_id: "model-1".to_string(),
                    name: "rpc.total".to_string(),
                    sample_count: 100,
                    avg_ms: Some(3.0),
                    p50_ms: Some(2.5),
                    p95_ms: Some(6.0),
                    p99_ms: Some(9.0),
                },
                HostTimingSummary {
                    operation: "env.reset".to_string(),
                    component_id: "env-1".to_string(),
                    name: "endpoint.total".to_string(),
                    sample_count: 4,
                    avg_ms: Some(1.0),
                    p50_ms: None,
                    p95_ms: None,
                    p99_ms: None,
                },
            ],
            metrics: vec![HostMetricSummary {
                operation: "env.step".to_string(),
                component_id: "env-1".to_string(),
                name: "batch.size".to_string(),
                kind: MetricKind::Number,
                sample_count: 100,
                avg: Some(4.0),
                p50: Some(4.0),
                p95: Some(4.0),
                p99: Some(4.0),
            }],
            env_latency_ms_avg: Some(1.2),
            env_latency_ms_p50: Some(1.0),
            env_latency_ms_p95: Some(2.0),
            env_latency_ms_p99: Some(3.0),
            model_latency_ms_avg: Some(2.8),
            model_latency_ms_p50: Some(2.5),
            model_latency_ms_p95: Some(5.5),
            model_latency_ms_p99: Some(8.5),
            round_trip_ms_avg: Some(4.0),
            round_trip_ms_p50: Some(3.5),
            round_trip_ms_p95: Some(7.5),
            round_trip_ms_p99: Some(11.5),
            reconnects: 0,
            drops: 0,
        }
    }

    #[test]
    fn maps_top_level_fields_and_marks_session_total() {
        let window = telemetry_summary_to_proto(&sample_summary());
        assert_eq!(window.session_id, "sess-1");
        assert_eq!(window.route_id, "route-1");
        assert_eq!(window.env_component_id, "env-1");
        assert_eq!(window.model_component_id, "model-1");
        assert_eq!(window.window_seconds, 12);
        assert_eq!(window.sample_count, 100);
        assert_eq!(window.steps_per_second, Some(8.5));
        assert_eq!(window.request_bytes_per_second, Some(1024.0));
        assert_eq!(window.response_bytes_per_second, Some(2048.0));
        assert!(window.is_session_total);
    }

    #[test]
    fn maps_host_timing_rows_with_key_and_key_name() {
        let window = telemetry_summary_to_proto(&sample_summary());
        let rpc = window
            .timings
            .iter()
            .find(|t| t.key_name == "rpc.total")
            .expect("rpc.total row present");
        assert_eq!(rpc.key, MetricKey::RpcTotal as i32);
        assert_eq!(rpc.operation, "model.predict");
        assert_eq!(rpc.component_id, "model-1");
        assert_eq!(rpc.sample_count, 100);
        assert_eq!(rpc.avg_ms, Some(3.0));
        assert_eq!(rpc.p99_ms, Some(9.0));

        let endpoint = window
            .timings
            .iter()
            .find(|t| t.key_name == "endpoint.total")
            .expect("endpoint.total row present");
        assert_eq!(endpoint.key, MetricKey::EndpointTotal as i32);
        // present/absent percentile distinction is preserved.
        assert_eq!(endpoint.p50_ms, None);
    }

    #[test]
    fn appends_phase_split_rows() {
        let window = telemetry_summary_to_proto(&sample_summary());

        let model_wait = window
            .timings
            .iter()
            .find(|t| t.key == MetricKey::ModelWait as i32)
            .expect("model_wait row present");
        assert_eq!(model_wait.key_name, "model.wait");
        assert_eq!(model_wait.avg_ms, Some(2.8));
        assert_eq!(model_wait.p99_ms, Some(8.5));

        let env_step = window
            .timings
            .iter()
            .find(|t| t.key == MetricKey::EnvStep as i32)
            .expect("env_step row present");
        assert_eq!(env_step.key_name, "env.step.phase");
        assert_eq!(env_step.avg_ms, Some(1.2));

        let round_trip = window
            .timings
            .iter()
            .find(|t| t.key == MetricKey::RoundTrip as i32)
            .expect("round_trip row present");
        assert_eq!(round_trip.key_name, "round.trip");
        assert_eq!(round_trip.avg_ms, Some(4.0));
        assert_eq!(round_trip.p95_ms, Some(7.5));
    }

    #[test]
    fn empty_phase_split_rows_are_omitted() {
        // A summary with no phase timing at all (all *_latency_* / round_trip_*
        // absent) must not emit placeholder ModelWait/EnvStep/RoundTrip rows.
        let mut summary = sample_summary();
        summary.model_latency_ms_avg = None;
        summary.model_latency_ms_p50 = None;
        summary.model_latency_ms_p95 = None;
        summary.model_latency_ms_p99 = None;
        summary.env_latency_ms_avg = None;
        summary.env_latency_ms_p50 = None;
        summary.env_latency_ms_p95 = None;
        summary.env_latency_ms_p99 = None;
        summary.round_trip_ms_avg = None;
        summary.round_trip_ms_p50 = None;
        summary.round_trip_ms_p95 = None;
        summary.round_trip_ms_p99 = None;

        let window = telemetry_summary_to_proto(&summary);
        for key in [
            MetricKey::ModelWait,
            MetricKey::EnvStep,
            MetricKey::RoundTrip,
        ] {
            assert!(
                !window.timings.iter().any(|t| t.key == key as i32),
                "phase {key:?} with no timing must be omitted"
            );
        }
        // The host-supplied rows (rpc.total, endpoint.total) are still present.
        assert!(window.timings.iter().any(|t| t.key_name == "rpc.total"));
    }

    #[test]
    fn maps_metric_rows_with_unit_from_kind() {
        let window = telemetry_summary_to_proto(&sample_summary());
        assert_eq!(window.metrics.len(), 1);
        let batch = &window.metrics[0];
        assert_eq!(batch.key, MetricKey::BatchSize as i32);
        assert_eq!(batch.key_name, "batch.size");
        assert_eq!(batch.unit, Unit::Count as i32);
        assert_eq!(batch.sample_count, 100);
        assert_eq!(batch.avg, Some(4.0));
    }

    #[test]
    fn unrecognized_name_falls_back_to_unspecified_with_key_name() {
        let mut summary = sample_summary();
        summary.timings.push(HostTimingSummary {
            operation: "custom.op".to_string(),
            component_id: "env-1".to_string(),
            name: "experimental.metric".to_string(),
            sample_count: 1,
            avg_ms: Some(0.5),
            p50_ms: None,
            p95_ms: None,
            p99_ms: None,
        });
        let window = telemetry_summary_to_proto(&summary);
        let custom = window
            .timings
            .iter()
            .find(|t| t.key_name == "experimental.metric")
            .expect("custom row present");
        assert_eq!(custom.key, MetricKey::Unspecified as i32);
    }

    #[test]
    fn byte_count_kind_maps_to_bytes_unit() {
        let mut summary = sample_summary();
        summary.metrics.push(HostMetricSummary {
            operation: "model.predict".to_string(),
            component_id: "model-1".to_string(),
            name: "request.bytes".to_string(),
            kind: MetricKind::ByteCount,
            sample_count: 100,
            avg: Some(512.0),
            p50: Some(500.0),
            p95: Some(600.0),
            p99: Some(700.0),
        });
        let window = telemetry_summary_to_proto(&summary);
        let bytes = window
            .metrics
            .iter()
            .find(|m| m.key_name == "request.bytes")
            .expect("bytes row present");
        assert_eq!(bytes.unit, Unit::Bytes as i32);
        assert_eq!(bytes.key, MetricKey::Unspecified as i32);
    }
}
