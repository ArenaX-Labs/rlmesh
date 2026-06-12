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
fn session_lifetime_samples_stay_bounded() {
    let mut accumulator = TelemetryWindowAccumulator::default();
    // Far more steps than any reasonable reservoir capacity; total_* buffers
    // would grow to STEPS in length before the reservoir bound was added.
    const STEPS: u64 = 100_000;
    for step in 0..STEPS {
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
            model_wait: Duration::from_millis(step % 50),
            env_step: Duration::from_millis(step % 30),
            request_bytes: 1,
            response_bytes: 1,
            env_component_id: "env-a",
            model_component_id: "model-a",
        });
    }

    // Memory must stay bounded well below the number of steps.
    assert!(
        accumulator.max_total_sample_buffer() <= 8192,
        "session-lifetime sample buffer grew unbounded: {}",
        accumulator.max_total_sample_buffer()
    );

    // The summary must still report the true total sample count.
    let summary = accumulator
        .summary("session-a", RuntimeRouteContext::default())
        .unwrap();
    assert_eq!(summary.sample_count, STEPS);
    let endpoint = summary
        .timings
        .iter()
        .find(|timing| timing.name == "endpoint.total")
        .expect("endpoint.total timing present");
    assert_eq!(endpoint.sample_count, STEPS);
}

#[test]
fn records_byte_count_and_number_metrics() {
    use crate::hooks::MetricKind;

    let mut accumulator = TelemetryWindowAccumulator::default();
    accumulator.started_at = Instant::now() - Duration::from_secs(2);
    accumulator.record_operation_telemetry(
        "env-a",
        Some(&OperationTelemetry {
            operation: "env.step".to_string(),
            component_id: String::new(),
            metrics: vec![
                OperationMetric {
                    name: "payload.bytes".to_string(),
                    labels: Default::default(),
                    value: Some(operation_metric::Value::ByteCount(2048)),
                },
                OperationMetric {
                    name: "batch.size".to_string(),
                    labels: Default::default(),
                    value: Some(operation_metric::Value::Number(8.0)),
                },
            ],
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

    let bytes = event
        .metrics
        .iter()
        .find(|metric| metric.name == "payload.bytes")
        .expect("byte_count metric recorded");
    assert_eq!(bytes.kind, MetricKind::ByteCount);
    assert_eq!(bytes.sample_count, 1);
    assert_eq!(bytes.avg, Some(2048.0));

    let number = event
        .metrics
        .iter()
        .find(|metric| metric.name == "batch.size")
        .expect("number metric recorded");
    assert_eq!(number.kind, MetricKind::Number);
    assert_eq!(number.avg, Some(8.0));
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
