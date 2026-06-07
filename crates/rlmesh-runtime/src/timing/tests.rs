use std::time::{Duration, Instant};

use rlmesh_proto::core::v1::{OperationMetric, OperationTelemetry, operation_metric};

use super::stats::percentile_ms;
use super::{PhaseTiming, StepTimingSample, TelemetryWindowAccumulator};
use crate::hooks::RuntimeRouteContext;

#[test]
fn phase_timing_tracks_basic_stats() {
    let mut timing = PhaseTiming::default();
    timing.record(Duration::from_millis(10));
    timing.record(Duration::from_millis(30));
    timing.record(Duration::from_millis(20));

    assert_eq!(timing.count, 3);
    assert_eq!(timing.total, Duration::from_millis(60));
    assert_eq!(timing.min, Some(Duration::from_millis(10)));
    assert_eq!(timing.max, Duration::from_millis(30));
    assert!((timing.avg_ms() - 20.0).abs() < f64::EPSILON);
}

#[test]
fn percentile_uses_sorted_window_samples() {
    let samples = [
        Duration::from_millis(10),
        Duration::from_millis(30),
        Duration::from_millis(20),
        Duration::from_millis(40),
    ];
    assert_eq!(percentile_ms(&samples, 0.95), Some(40.0));
}

#[test]
fn telemetry_window_includes_generic_runtime_and_endpoint_timings() {
    let mut accumulator = TelemetryWindowAccumulator::default();
    accumulator.started_at = Instant::now() - Duration::from_secs(2);
    accumulator.record_operation_telemetry(
        "model-a",
        Some(&OperationTelemetry {
            operation: "model.predict".to_string(),
            component_id: String::new(),
            metrics: vec![OperationMetric {
                name: "endpoint.total".to_string(),
                labels: Default::default(),
                value: Some(operation_metric::Value::DurationNs(2_000_000)),
            }],
        }),
    );
    accumulator.record_step(StepTimingSample {
        model_wait: Duration::from_millis(10),
        env_step: Duration::from_millis(20),
        request_bytes: 3,
        response_bytes: 4,
        env_component_id: "env-a",
        model_component_id: "model-a",
    });

    let event = accumulator
        .flush("session-a", RuntimeRouteContext::default())
        .unwrap();

    assert!(event.timings.iter().any(|timing| {
        timing.operation == "model.predict"
            && timing.component_id == "model-a"
            && timing.name == "endpoint.total"
            && timing.avg_ms == Some(2.0)
    }));
    assert!(event.timings.iter().any(|timing| {
        timing.operation == "model.predict"
            && timing.component_id == "model-a"
            && timing.name == "rpc.total"
            && timing.avg_ms == Some(10.0)
    }));
    assert!(event.timings.iter().any(|timing| {
        timing.operation == "env.step"
            && timing.component_id == "env-a"
            && timing.name == "rpc.total"
            && timing.avg_ms == Some(20.0)
    }));
}

#[test]
fn telemetry_summary_keeps_samples_after_window_flush() {
    let mut accumulator = TelemetryWindowAccumulator::default();
    accumulator.started_at = Instant::now() - Duration::from_secs(2);
    accumulator.record_step(StepTimingSample {
        model_wait: Duration::from_millis(10),
        env_step: Duration::from_millis(20),
        request_bytes: 3,
        response_bytes: 4,
        env_component_id: "env-a",
        model_component_id: "model-a",
    });

    let window = accumulator
        .flush("session-a", RuntimeRouteContext::default())
        .unwrap();
    assert_eq!(window.sample_count, 1);

    accumulator.started_at = Instant::now() - Duration::from_secs(2);
    accumulator.record_step(StepTimingSample {
        model_wait: Duration::from_millis(30),
        env_step: Duration::from_millis(40),
        request_bytes: 5,
        response_bytes: 6,
        env_component_id: "env-a",
        model_component_id: "model-a",
    });

    let summary = accumulator
        .summary("session-a", RuntimeRouteContext::default())
        .unwrap();
    assert_eq!(summary.sample_count, 2);
    assert_eq!(summary.env_latency_ms_avg, Some(30.0));
    assert_eq!(summary.model_latency_ms_avg, Some(20.0));
    assert_eq!(summary.round_trip_ms_avg, Some(50.0));
}
