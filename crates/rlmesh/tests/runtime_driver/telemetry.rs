//! Telemetry: the wall-clock ticker, peer-reported phases and gauges, and
//! step timings under chunking.

use super::*;

// Real-time (not start_paused, which needs tokio's test-util feature): the
// 100ms stall vs 10ms cadence is a wide enough margin to be robust.
#[tokio::test]
async fn telemetry_ticks_on_wall_clock_during_a_stalled_step() {
    let env = TestEnv {
        terminal_after: 1,
        ..Default::default()
    };
    let model = TestModel {
        predict_delay: Some(Duration::from_millis(100)),
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());

    let mut spec = one_episode_spec();
    spec.limits.telemetry_window = Duration::from_millis(10);

    let report = new_driver(spec, env, model, hooks.clone())
        .run()
        .await
        .unwrap();

    // The lone predict stalls 100ms; with a 10ms wall-clock cadence the
    // background ticker emits a (non-empty) Window snapshot *during* the stall —
    // the old step-gated path could not, being parked inside the await. The
    // ticker streams only Window deltas; the cumulative Session total is pushed
    // once by the epilogue, never per tick.
    assert!(
        hooks.telemetry_windows.load(Ordering::SeqCst) >= 1,
        "expected a wall-clock window snapshot during the stalled predict",
    );
    assert_eq!(
        hooks.telemetry_sessions.load(Ordering::SeqCst),
        1,
        "exactly one final Session push (epilogue) — the ticker emits no sessions",
    );
    assert!(hooks.telemetry_session_rows.load(Ordering::SeqCst) >= 1);
    // The final pull still carries the per-op session total.
    assert!(
        report
            .telemetry
            .rows
            .iter()
            .any(|row| row.source.op == "model.predict" && row.metric.name == "rpc.total"),
    );
}

#[tokio::test]
async fn peer_reported_phases_land_in_the_session_snapshot() {
    let env = TestEnv {
        endpoint_total_ns: Some(9_000_000),
        phases: EndpointPhases {
            decode_ns: 1_000_000,
            user_ns: 6_000_000,
            encode_ns: 2_000_000,
            ..EndpointPhases::default()
        },
        ..TestEnv::default()
    };
    // A model served elsewhere, which fused this predict with another
    // route's and charged it its share of the forward.
    let model = TestModel {
        adapter_ns: 2_000_000,
        held: Some((5, 6_000_000)),
        predict_delay: Some(Duration::from_millis(20)),
        endpoint: Some((
            Some(4_000_000),
            EndpointPhases {
                user_ns: 3_000_000,
                queue_ns: 1_000_000,
                in_flight: 2,
                ..EndpointPhases::default()
            },
        )),
        ..TestModel::default()
    };

    let report = new_driver(
        one_episode_spec(),
        env,
        model,
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    let avg = |op: &str, metric: &str| {
        report
            .telemetry
            .rows
            .iter()
            .find(|row| row.source.op == op && row.metric.name == metric)
            .unwrap_or_else(|| panic!("{op} {metric} recorded"))
            .avg
    };

    // Durations are reported in ms, so the split reads back as the peer stamped it.
    assert_eq!(avg("env.step", "endpoint.decode"), 1.0);
    assert_eq!(avg("env.step", "endpoint.user"), 6.0);
    assert_eq!(avg("env.step", "endpoint.encode"), 2.0);
    // The model's own timing, not the call's wall time, which stays the RPC.
    assert_eq!(avg("model.predict", "endpoint.total"), 4.0);
    assert_eq!(avg("model.predict", "endpoint.user"), 3.0);
    assert_eq!(avg("model.predict", "endpoint.queue"), 1.0);
    assert_eq!(avg("model.predict", "predict.in_flight"), 2.0);
    assert!(avg("model.predict", "rpc.total") >= 20.0);
    // The adapter's share of the handler's own work: forward = user - adapter.
    assert_eq!(avg("model.predict", "predict.adapter"), 2.0);
    // Held state gauges: an episode count and raw bytes, not durations.
    assert_eq!(avg("model.predict", "held.episodes"), 5.0);
    assert_eq!(avg("model.predict", "held.bytes"), 6_000_000.0);

    // A model that reports its phases but no total gets none made up here.
    let model = TestModel {
        endpoint: Some((
            None,
            EndpointPhases {
                user_ns: 3_000_000,
                ..EndpointPhases::default()
            },
        )),
        ..TestModel::default()
    };
    let report = new_driver(
        one_episode_spec(),
        TestEnv::default(),
        model,
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();
    let recorded = |metric: &str| {
        report
            .telemetry
            .rows
            .iter()
            .any(|row| row.source.op == "model.predict" && row.metric.name == metric)
    };
    assert!(recorded("endpoint.user"));
    assert!(!recorded("endpoint.total"));
}

#[tokio::test]
async fn a_straggler_lane_records_its_skew_over_the_median_lane() {
    // A vector env timing its own lanes stamps one dispersion scalar per step,
    // so a single slow lane among healthy ones is visible without a series per
    // lane. The env is otherwise silent: skew does not depend on a phase split.
    let env = TestEnv {
        phases: EndpointPhases {
            lane_skew_ns: Some(8_000_000),
            ..EndpointPhases::default()
        },
        ..TestEnv::default()
    };

    let report = new_driver(
        one_episode_spec(),
        env,
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    let skew = |op: &str| {
        report
            .telemetry
            .rows
            .iter()
            .find(|row| row.metric.name == "lane.skew" && row.source.op == op)
    };
    assert_eq!(skew("env.step").expect("env.step lane.skew").avg, 8.0);
    // Only the env reports lanes.
    assert!(skew("model.predict").is_none());
}

#[tokio::test]
async fn a_measured_zero_gauge_is_a_sample_not_an_absence() {
    // An even vector has zero skew and an engine that just evicted holds zero
    // episodes: both are real observations the percentiles need, unlike a peer
    // that never measures (which records nothing, see the test below).
    let env = TestEnv {
        phases: EndpointPhases {
            lane_skew_ns: Some(0),
            ..EndpointPhases::default()
        },
        ..TestEnv::default()
    };
    let model = TestModel {
        held: Some((0, 0)),
        ..TestModel::default()
    };
    let report = new_driver(
        one_episode_spec(),
        env,
        model,
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    let row = |op: &str, metric: &str| {
        report
            .telemetry
            .rows
            .iter()
            .find(|row| row.source.op == op && row.metric.name == metric)
            .unwrap_or_else(|| panic!("{op} {metric} recorded"))
    };
    assert_eq!(row("env.step", "lane.skew").avg, 0.0);
    assert!(row("env.step", "lane.skew").count > 0);
    assert_eq!(row("model.predict", "held.episodes").avg, 0.0);
    assert_eq!(row("model.predict", "held.bytes").avg, 0.0);
}

#[tokio::test]
async fn a_peer_that_reports_no_phases_records_only_the_total() {
    // An env that stamps `endpoint_total_ns` and nothing else: the phase fields
    // arrive absent, decode to zero, and record no env rows at all.
    let env = TestEnv {
        endpoint_total_ns: Some(9_000_000),
        ..TestEnv::default()
    };
    let report = new_driver(
        one_episode_spec(),
        env,
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    let recorded = |metric: &str| {
        report
            .telemetry
            .rows
            .iter()
            .any(|row| row.source.op == "env.step" && row.metric.name == metric)
    };
    assert!(recorded("endpoint.total"));
    for metric in [
        "endpoint.decode",
        "endpoint.user",
        "endpoint.encode",
        "endpoint.queue",
        "predict.in_flight",
        "predict.adapter",
        "held.episodes",
        "held.bytes",
        "lane.skew",
    ] {
        assert!(!recorded(metric), "{metric} must stay unrecorded");
    }
}

#[tokio::test]
async fn step_timings_charge_a_chunk_predict_to_the_step_that_consumed_it() {
    // A chunk of five frames: the predict lands on the step that first consumed
    // it, the four replayed steps carry no predict time, and every step carries
    // its own env round trip. The episode mean spreads the predict over its steps.
    let model = TestModel {
        replay_frames: 4,
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let report = new_driver(
        vector_spec(1, 1),
        VectorTestEnv::new(vec![5]),
        model,
        hooks.clone(),
    )
    .run()
    .await
    .unwrap();

    let timings = hooks.step_timings.lock().unwrap().clone();
    assert_eq!(timings.len(), 5, "{timings:?}");
    assert!(
        timings[0].0 > 0.0,
        "the first step consumed the chunk: {timings:?}"
    );
    assert!(
        timings[1..].iter().all(|(predict, _)| *predict == 0.0),
        "replayed steps carry no predict: {timings:?}"
    );
    assert!(timings.iter().all(|(_, step)| *step > 0.0), "{timings:?}");
    let episode = &report.episodes[0];
    let predict_total: f64 = timings.iter().map(|(p, _)| p).sum();
    let step_total: f64 = timings.iter().map(|(_, s)| s).sum();
    assert!((episode.predict_ms.unwrap() - predict_total / 5.0).abs() < 1e-9);
    assert!((episode.step_ms.unwrap() - step_total / 5.0).abs() < 1e-9);
}
