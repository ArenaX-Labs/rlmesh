//! The cockpit's model on a sans-IO UR10e session: every panel's data,
//! derived from what an observer receives, at injected times.

mod common;

use common::{MS, World, pid};
use rlmesh_cockpit::model::Verb;
use rlmesh_cockpit::timeline::Tone;
use rlmesh_cockpit::values::Value;
use rlmesh_cockpit::{Input, View};
use rlmesh_session::wire::{Command, EdgeMode, LeaseToken, Mode, ReasonCode};

fn close(a: Option<f64>, b: f64, tol: f64) -> bool {
    a.is_some_and(|a| (a - b).abs() <= tol)
}

fn stream<'a>(view: &'a View, label: &str) -> &'a rlmesh_cockpit::view::StreamRow {
    view.streams
        .iter()
        .find(|r| r.label == label)
        .unwrap_or_else(|| panic!("no stream {label}"))
}

#[test]
fn live_channels_show_the_path_tree_values_ages_and_losses() {
    let mut w = World::ur10e();
    w.run(1_000);
    let arm = w.channel("arm");
    w.lose(arm);
    w.run(10);
    let view = w.model.view(w.now);
    let labels: Vec<(usize, &str, Option<&str>)> = view
        .tree
        .iter()
        .map(|r| (r.depth, r.label.as_str(), r.kind.as_deref()))
        .collect();
    assert_eq!(
        labels,
        [
            (0, "ur10e", None),
            (1, "arm", Some("joints")),
            (2, "_applied", Some("applied")),
            (1, "wrist", None),
            (2, "cam", Some("camera")),
            (2, "ft", Some("wrench")),
        ]
    );
    let arm_row = &view.tree[1];
    let Some(Value::Numbers(values)) = &arm_row.value else {
        panic!("joint values, got {:?}", arm_row.value);
    };
    assert_eq!(values.len(), 12, "pos and vel, 6 joints each");
    assert!(values[0].0.starts_with("pos"), "{:?}", values[0]);
    assert_eq!(arm_row.lost, 1);
    assert!(
        arm_row.age_ms.is_some_and(|a| a < 5.0),
        "{:?}",
        arm_row.age_ms
    );
    assert!(!arm_row.stale);
    assert_eq!(arm_row.plot.len(), 3, "three series plotted");
    assert!(arm_row.plot[0].len() > 40, "a 20 ms plot over a second");
    let cam = &view.tree[4];
    let Some(Value::Image(thumb)) = &cam.value else {
        panic!("a thumbnail, got {:?}", cam.value);
    };
    assert_eq!((thumb.width, thumb.height), (64, 48));

    // Silence ages the value and goes stale past its bound; the joints'
    // contract bound is derived (§8.7).
    let quiet = w.now + 900 * MS;
    let view = w.model.view(quiet);
    let q = stream(&view, "ur10e/arm");
    let (bound, from) = q.bound.expect("a bound");
    assert_eq!(from, "derived");
    assert!(q.stale, "{:?} past {bound}", q.age_ms);
}

#[test]
fn declared_and_measured_rates_stay_separate_and_name_their_clocks() {
    let mut w = World::ur10e();
    w.run(3_000);
    let view = w.model.view(w.now);

    let arm = stream(&view, "ur10e/arm");
    assert_eq!(arm.declared_hz, Some(500.0));
    assert!(close(arm.received_hz, 500.0, 5.0), "{:?}", arm.received_hz);
    assert!(
        close(arm.published_hz, 500.0, 5.0),
        "{:?}",
        arm.published_hz
    );
    assert!(close(arm.capture_hz, 500.0, 5.0), "{:?}", arm.capture_hz);
    assert_eq!(
        arm.admitted,
        [("policy".to_string(), "latest@50".to_string())]
    );
    assert_eq!(arm.admitted_hz, Some(50.0));
    assert_eq!(arm.plan, "latest@50");
    let (who, rate) = &arm.heard[0];
    assert_eq!(who, "policy");
    assert!(
        close(*rate, 500.0, 5.0),
        "the policy's seq advance: {rate:?}"
    );
    assert!(arm.drift.is_none());

    // The camera: declared 30, produced 30, published 20 (the publisher
    // coalesced every third frame), received 20.
    let cam = stream(&view, "ur10e/wrist/cam");
    assert_eq!(cam.declared_hz, Some(30.0));
    assert!(close(cam.produced_hz, 30.0, 1.0), "{:?}", cam.produced_hz);
    assert!(close(cam.published_hz, 20.0, 1.0), "{:?}", cam.published_hz);
    assert!(close(cam.received_hz, 20.0, 1.0), "{:?}", cam.received_hz);
    assert!(cam.drift.is_none(), "produced matches declared");

    // The policy: 50 starts and completions per second, chunks at 50 Hz
    // with a 2 ms age at receipt, and coverage left on its lease.
    let policy = view
        .controllers
        .iter()
        .find(|c| c.id == "policy")
        .expect("policy");
    assert!(!policy.manual);
    assert_eq!(policy.replan_hz, Some(50.0));
    assert!(close(policy.starts_hz, 50.0, 1.0), "{:?}", policy.starts_hz);
    assert!(close(policy.completions_hz, 50.0, 1.0));
    assert!(close(policy.chunks_hz, 50.0, 2.0), "{:?}", policy.chunks_hz);
    let (age, max) = policy.chunk_age_ms.expect("chunk ages");
    assert!(
        (age - 2.0).abs() < 1e-6 && (max - 2.0).abs() < 1e-6,
        "{age} {max}"
    );
    // 20 actions on the binding's 2 ms grid = 40 ms from the newest t0.
    let coverage = policy.coverage_ms.expect("coverage");
    assert!(coverage > 15.0 && coverage <= 40.0, "{coverage}");
    assert_eq!(policy.leases, ["ur10e/arm"]);
    let operator = view
        .controllers
        .iter()
        .find(|c| c.id == "operator")
        .expect("operator");
    assert!(operator.manual, "no run loop reports");

    // The edge: 500 Hz on its own ticks, no misses, driving.
    let edge = &view.edges[0];
    assert_eq!(edge.label, "ur10e/arm");
    assert!(close(edge.tick_hz, 500.0, 5.0), "{:?}", edge.tick_hz);
    assert_eq!((edge.misses, edge.queue_depth), (0, Some(18)));
    assert_eq!(edge.mode.map(|(m, _)| m), Some(EdgeMode::Driving));
    assert!(
        edge.holder
            .as_deref()
            .is_some_and(|h| h.starts_with("policy (model)"))
    );
    assert!(
        edge.last_chunk
            .as_deref()
            .is_some_and(|c| c.starts_with("policy t"))
    );
}

#[test]
fn a_slow_policy_and_a_stalled_edge_show_as_drift_skips_and_deadline_misses() {
    let mut w = World::ur10e();
    w.run(1_000);
    // The edge thread stalls for 20 ms: one late tick, counted once.
    w.stall_until = w.now + 20 * MS;
    w.run(100);
    let view = w.model.view(w.now);
    let edge = &view.edges[0];
    assert_eq!((edge.misses, edge.recent_misses), (1, 1));
    assert!(
        close(edge.longest_gap_ms, 20.0, 2.0),
        "{:?}",
        edge.longest_gap_ms
    );

    // A seq lost on the way is not a late tick.
    let applied = w
        .auth
        .part(w.channel("arm"))
        .and_then(|p| p.applied_channel)
        .expect("applied");
    w.lose(applied);
    w.run(10);
    assert_eq!(w.model.view(w.now).edges[0].misses, 1);

    // Busy skips and an inference pending are the controller's own counts.
    w.run.busy_skips = 40;
    w.run.pending_ns = 2_000 * MS as u64;
    w.run.drifting = true;
    w.run(500);
    let view = w.model.view(w.now);
    let policy = view
        .controllers
        .iter()
        .find(|c| c.id == "policy")
        .expect("policy");
    assert_eq!(policy.busy_skips, 40);
    assert!(policy.drifting);
    assert!(close(policy.pending_ms, 2_000.0, 1e-6));

    // Hold, then safe, as the edge reports it: in the lanes and the events.
    w.edge_mode = EdgeMode::Hold;
    w.run(50);
    w.edge_mode = EdgeMode::Safe;
    w.run(50);
    let view = w.model.view(w.now);
    let edge_events: Vec<&str> = view
        .timeline
        .iter()
        .filter(|r| r.kind == "edge")
        .map(|r| r.text.as_str())
        .collect();
    assert_eq!(
        edge_events,
        [
            "edge ur10e/arm: hold -> safe",
            "edge ur10e/arm: driving -> hold",
            "edge ur10e/arm: driving",
        ]
    );
    let lane = view
        .lanes
        .iter()
        .find(|l| l.label == "edge ur10e/arm")
        .expect("lane");
    let states: Vec<&str> = lane.segments.iter().map(|s| s.2.as_str()).collect();
    assert_eq!(states, ["driving", "hold", "safe"]);
    assert_eq!(lane.segments[2].3, Tone::Bad);
    assert_eq!(view.edges[0].mode.map(|(m, _)| m), Some(EdgeMode::Safe));
}

#[test]
fn the_timeline_follows_leases_epochs_preemption_and_estop() {
    let mut w = World::ur10e();
    w.run(300);
    // The operator jumps in: its lease preempts the policy's.
    w.grant("operator");
    w.run(300);
    // The policy's late chunk under its old token is fenced, not coverage.
    let stale = LeaseToken(1);
    w.chunk("policy", stale);
    w.verb(
        "cockpit",
        Command::Estop {
            reason: "test".into(),
            code: ReasonCode::EstopRemote,
            origin: None,
        },
    )
    .expect("estop");
    w.publish_snapshot();
    let view = w.model.view(w.now);
    let events: Vec<(&str, &str)> = view
        .timeline
        .iter()
        .rev()
        .filter(|r| r.kind != "edge")
        .map(|r| (r.kind, r.text.as_str()))
        .collect();
    let has = |kind: &str, text: &str| events.iter().any(|(k, t)| *k == kind && t.contains(text));
    assert!(has("mode", "attached: DISARMED in epoch 1"), "{events:#?}");
    assert!(has("binding", "policy bound"));
    assert!(has("mode", "epoch 1 -> 2"));
    assert!(has("mode", "DISARMED -> ARMED"));
    assert!(has("lease", "ur10e/arm granted to policy (model"));
    assert!(has("mode", "ARMED -> ACTIVE"));
    assert!(has(
        "lease",
        "ur10e/arm preempted: operator (operator) took it from policy (model)"
    ));
    assert!(has("mode", "ACTIVE -> ESTOP"));
    assert!(has("estop", "e-stop latched"));
    assert!(has("lease", "revoked (ESTOP)"));
    assert!(view.edges[0].fenced >= 1);

    let lane = view
        .lanes
        .iter()
        .find(|l| l.label == "lease ur10e/arm")
        .expect("lane");
    let holders: Vec<&str> = lane.segments.iter().map(|s| s.2.as_str()).collect();
    assert_eq!(holders.len(), 4, "{holders:?}");
    assert!(
        holders[0] == "free"
            && holders[1].starts_with("policy")
            && holders[2].starts_with("operator")
            && holders[3] == "free"
    );
    let mode = view
        .lanes
        .iter()
        .find(|l| l.label == "session mode")
        .expect("mode lane");
    assert_eq!(mode.segments.last().map(|s| s.2.as_str()), Some("ESTOP e2"));

    // The controls follow the mode: a latched e-stop is the robot's to clear.
    assert!(!view.header.can_arm && !view.header.can_disarm);
    assert!(
        view.header
            .estop_note
            .as_deref()
            .is_some_and(|n| n.contains("clears it locally"))
    );
    w.verb("ur10e", Command::Clear).expect("local clear");
    w.publish_snapshot();
    let header = w.model.view(w.now).header;
    assert!(header.can_disarm && !header.can_arm);
    w.verb("cockpit", Command::Disarm).expect("disarm");
    w.publish_snapshot();
    let header = w.model.view(w.now).header;
    assert_eq!(header.mode, Some(Mode::Disarmed));
    assert!(header.can_arm && !header.can_disarm);
}

#[test]
fn the_binding_table_follows_the_spec_layout() {
    let mut w = World::ur10e();
    w.run(10);
    let view = w.model.view(w.now);
    let operator = view
        .bindings
        .iter()
        .find(|b| b.subtitle.starts_with("operator"))
        .expect("operator");
    assert_eq!(
        operator.title,
        "identity binding by operator: commands arm, fingerprint 0000000000000be7"
    );
    let b = view
        .bindings
        .iter()
        .find(|b| b.subtitle.starts_with("policy"))
        .expect("policy");
    assert!(
        b.title
            .starts_with("binding reach on ur10e: 2 inputs, 1 actions, fingerprint "),
        "{}",
        b.title
    );
    assert!(
        b.subtitle.contains("replan 50 Hz") && b.subtitle.contains("holds its leases"),
        "{}",
        b.subtitle
    );
    let entries: Vec<[&str; 4]> = b
        .entries
        .iter()
        .map(|r| [r[0].as_str(), r[1].as_str(), r[2].as_str(), r[3].as_str()])
        .collect();
    assert_eq!(
        entries,
        [
            ["q", "in", "arm", "bound"],
            ["cam", "in", "wrist/cam", "bound"],
            ["arm", "out", "arm", "bound"],
        ]
    );
    assert!(
        b.entries[1][4].contains("pad") || b.entries[1][4].contains("resize"),
        "{:?}",
        b.entries[1]
    );
    let q = b.inputs.iter().find(|r| r[0] == "q").expect("q");
    assert_eq!(&q[1..4], ["latest@50", "500 Hz", "50 Hz"]);
    assert!(q[4].ends_with("(derived)"), "{}", q[4]);
    let arm = &b.actions[0];
    assert_eq!(&arm[..3], ["arm", "500 Hz", "20 × 2 ms = 40 ms"]);
    assert_eq!(&arm[4..], ["500 Hz", "1000 ms"]);
}

#[test]
fn verbs_answers_and_host_loss_reach_the_header_and_the_timeline() {
    let mut w = World::ur10e();
    let now = w.now;
    w.model.ingest(
        Input::Outcome {
            verb: Verb::Arm,
            result: Err("InvalidTransition: already ACTIVE".into()),
        },
        now,
    );
    w.model.ingest(Input::HostPresent(false), now);
    w.model.ingest(Input::Overrun(12), now);
    let view = w.model.view(now + 3_000 * MS);
    let (verb, result, ago) = view.header.last_outcome.clone().expect("outcome");
    assert_eq!(verb, Verb::Arm);
    assert!(result.is_err() && (ago - 3.0).abs() < 1e-9);
    assert!(!view.header.host_present);
    assert_eq!(view.header.overrun, 12);
    assert!(view.timeline[0].text.contains("session host lost"));
    assert!(view.timeline[1].text.contains("cockpit sent arm: refused"));
    assert_eq!(view.participants.len(), 4);
    assert!(
        view.participants
            .iter()
            .any(|p| p.id == "cockpit (this cockpit)")
    );
    let _ = pid;
}
