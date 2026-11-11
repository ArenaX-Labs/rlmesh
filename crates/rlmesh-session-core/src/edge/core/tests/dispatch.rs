//! Dispatch: action grids, clock conversion, a dry queue, the applied stream,
//! limits and virtual ticks.

use super::*;

#[test]
fn a_chunk_without_dt_takes_the_bindings_action_grid() {
    let mut f = Fixture::new();
    let cmd = f.ch("cmd");
    Fixture::run(
        &mut f.auth,
        "policy",
        Command::DeclareBinding {
            fingerprint: Fingerprint::from(7),
            revision: 2,
            table_json: "{}".into(),
            channels: vec![cmd],
            replan_hz: 0.0,
            contracts: Vec::new(),
            grids: vec![crate::wire::ActionGrid {
                channel: cmd,
                dt: Duration::from_millis(10),
            }],
        },
    );
    let lease = f.grant("policy", 0);
    f.publish(0);
    let mut chunk = f.chunk("policy", &lease, 1, 0, 5);
    chunk.dt_ns = 0;
    f.accept(chunk, 0).expect("accept");
    let first = f.tick(9 * MS).action.expect("due");
    assert_eq!((first.index, first.dt_ns), (0, 10 * MS));
    assert_eq!(f.tick(10 * MS).action.expect("due").index, 1);

    // Without a grid for the part, dt = 0 is the control period (1 ms).
    Fixture::run(
        &mut f.auth,
        "policy",
        Command::DeclareBinding {
            fingerprint: Fingerprint::from(8),
            revision: 3,
            table_json: "{}".into(),
            channels: vec![cmd],
            replan_hz: 0.0,
            contracts: Vec::new(),
            grids: Vec::new(),
        },
    );
    let lease = f.grant("policy", 11 * MS);
    f.publish(11 * MS);
    let mut chunk = f.chunk("policy", &lease, 1, 11 * MS, 5);
    chunk.dt_ns = 0;
    f.accept(chunk, 11 * MS).expect("accept");
    assert_eq!(f.tick(12 * MS).action.expect("due").dt_ns, MS);
}

#[test]
fn clock_conversion_never_extends_a_deadline_and_transit_is_checked() {
    let mut f = Fixture::new();
    // The edge's clock reads 1 s behind session time (offset +1 s),
    // known within 2 ms; edge time 0 is session time 1 s.
    let session_now = 1_000 * MS;
    let lease = f.grant("policy", session_now);
    f.mapping = Some((1_000 * MS, 2 * MS as u64));
    f.publish(session_now);
    let mut chunk = f.chunk("policy", &lease, 1, session_now, 100);
    chunk.valid_until_ns = session_now + 30 * MS;
    chunk.publish_ns = 0;
    f.accept(chunk, 0).expect("accept");
    // valid_until converts to 30 − 2 = 28 ms local.
    assert_eq!(f.tick(27 * MS).mode, EdgeMode::Driving);
    let tick = f.tick(29 * MS);
    assert_eq!(
        tick.mode,
        EdgeMode::Hold,
        "void after the conservative deadline"
    );

    // A chunk that spent longer than max_action_age (200 ms) in transit,
    // with the error bounds added, is refused.
    let mut stale = f.chunk("policy", &lease, 2, session_now, 10);
    stale.publish_ns = -197 * MS;
    assert_eq!(
        f.accept(stale, 0),
        Err(Rejection::TooOld),
        "197 + 2 + 2 > 200"
    );
    let mut fresh = f.chunk("policy", &lease, 3, session_now, 10);
    fresh.publish_ns = -195 * MS;
    f.accept(fresh, 0).expect("195 + 4 <= 200");

    // The lease (granted at session 1 s with a 100 ms grace) expires at
    // 100 ms local, less the error bound.
    assert_eq!(
        f.accept(f.chunk("policy", &lease, 4, session_now, 10), 99 * MS),
        Err(Rejection::LeaseExpired)
    );

    // An uncertain or missing mapping refuses chunks (5 ms by default).
    let mut f = Fixture::new();
    let lease = f.grant("policy", 0);
    f.mapping = Some((0, 6 * MS as u64));
    f.publish(0);
    assert_eq!(
        f.accept(f.chunk("policy", &lease, 1, 0, 10), 0),
        Err(Rejection::ClockUncertain)
    );
    f.mapping = None;
    f.publish(0);
    assert_eq!(
        f.accept(f.chunk("policy", &lease, 2, 0, 10), 0),
        Err(Rejection::NoClockMapping)
    );
    assert_eq!(
        f.tick(0).mode,
        EdgeMode::Idle,
        "untimed and nothing applied"
    );
}

#[test]
fn a_holder_lost_before_the_host_noticed_parks_its_part_unreported() {
    let mut f = Fixture::new();
    let lease = f.grant("policy", 0);
    f.publish(0);
    f.accept(f.chunk("policy", &lease, 1, 0, 5), 0)
        .expect("accept");
    let ticks: Vec<EdgeTick> = (0..20).map(|t| f.tick(t * MS)).collect();
    assert_eq!(ticks.last().map(|t| t.mode), Some(EdgeMode::Safe));
    f.auth.participant_lost(&pid("policy"), 25 * MS);
    f.publish(25 * MS);
    let tick = f.tick(26 * MS);
    assert_eq!(tick.mode, EdgeMode::Safe);
    assert_eq!(tick.safe_behavior.as_deref(), Some("brake"));
    assert!(ticks.iter().chain([&tick]).all(|t| t.report.is_none()));
    assert_eq!(f.edge.stats().safe_reports, 0);
    // The next holder resumes it.
    let next = f.grant("operator", 30 * MS);
    f.publish(30 * MS);
    f.accept(f.chunk("operator", &next, 1, 30 * MS, 5), 30 * MS)
        .expect("accept");
    assert_eq!(f.tick(31 * MS).mode, EdgeMode::Driving);
}

#[test]
fn a_holder_the_host_lost_first_parks_its_part_as_a_release() {
    let mut f = Fixture::new();
    let lease = f.grant("policy", 0);
    f.publish(0);
    f.accept(f.chunk("policy", &lease, 1, 0, 5), 0)
        .expect("accept");
    assert_eq!(f.tick(0).mode, EdgeMode::Driving);
    f.auth.participant_lost(&pid("policy"), MS);
    f.publish(MS);
    let ticks: Vec<EdgeTick> = (1..20).map(|t| f.tick(t * MS)).collect();
    let last = ticks.last().expect("ticks");
    assert_eq!(last.mode, EdgeMode::Safe);
    assert_eq!(last.safe_behavior.as_deref(), Some("brake"));
    assert!(ticks.iter().all(|t| t.report.is_none()));
    assert_eq!(f.auth.mode(), Mode::Armed);
}

#[test]
fn a_dry_queue_holds_then_goes_safe_and_estop_is_immediate() {
    let mut f = Fixture::new();
    let lease = f.grant("policy", 0);
    f.publish(0);
    f.accept(f.chunk("policy", &lease, 1, 0, 5), 0)
        .expect("accept");
    let ticks: Vec<EdgeTick> = (0..20).map(|t| f.tick(t * MS)).collect();
    let count = |m| ticks.iter().filter(|t| t.mode == m).count();
    assert_eq!(count(EdgeMode::Driving), 5);
    assert_eq!(count(EdgeMode::Hold), 10, "hold_max is 10 ms");
    assert_eq!(count(EdgeMode::Safe), 5);
    assert!(ticks.iter().all(|t| t.report.is_none()), "not judged yet");
    // A renewal from after the fall: its holder is alive, so it is reported.
    let epoch = f.auth.epoch();
    f.auth.heartbeat(&pid("policy"), epoch, 10 * MS);
    f.publish(16 * MS);
    assert!(f.tick(20 * MS).report.is_none(), "from before the fall");
    f.auth.heartbeat(&pid("policy"), epoch, 21 * MS);
    f.publish(21 * MS);
    assert_eq!(f.tick(22 * MS).report, Some((epoch, SafeCause::QueueDry)));
    assert_eq!(f.edge.stats().safe_reports, 1);
    assert!(f.tick(23 * MS).report.is_none(), "reported once");

    let mut f = Fixture::new();
    let lease = f.grant("policy", 0);
    f.publish(0);
    f.accept(f.chunk("policy", &lease, 1, 0, 50), 0)
        .expect("accept");
    assert_eq!(f.tick(0).mode, EdgeMode::Driving);
    Fixture::run(
        &mut f.auth,
        "operator",
        Command::Estop {
            reason: "test".into(),
            code: ReasonCode::EstopRemote,
            origin: None,
            at_ns: 0,
        },
    );
    f.publish(MS);
    let tick = f.tick(MS);
    assert_eq!(tick.mode, EdgeMode::Estop);
    assert_eq!(tick.safe_behavior.as_deref(), Some("brake"));
    assert_eq!(tick.applied[0].applied.reason, ReasonCode::EstopRemote);
    assert!(tick.report.is_none(), "the host already knows");
    assert_eq!(
        f.accept(f.chunk("policy", &lease, 2, MS, 50), MS),
        Err(Rejection::Estop)
    );

    // A local latch needs no host: refused chunks, purged queues, Estop.
    let mut f = Fixture::new();
    let lease = f.grant("policy", 0);
    f.publish(0);
    f.accept(f.chunk("policy", &lease, 1, 0, 50), 0)
        .expect("accept");
    assert_eq!(f.tick(0).mode, EdgeMode::Driving);
    f.edge.set_local_estop(Some(ReasonCode::EstopLocal));
    assert_eq!(
        f.accept(f.chunk("policy", &lease, 2, MS, 50), MS),
        Err(Rejection::Estop)
    );
    let tick = f.tick(MS);
    assert_eq!(
        (tick.mode, tick.applied[0].applied.reason),
        (EdgeMode::Estop, ReasonCode::EstopLocal)
    );
    assert_eq!(f.edge.queue_depth(), 0);
    // Cleared, the host still shows the lease, but nothing queued
    // survived the latch.
    f.edge.set_local_estop(None);
    assert_eq!(f.tick(2 * MS).mode, EdgeMode::Idle);
}

#[test]
fn the_applied_stream_shows_what_was_dispatched_each_tick() {
    let mut f = Fixture::new();
    let lease = f.grant("policy", 0);
    f.publish(0);
    let applied_id = f
        .auth
        .part(f.ch("cmd"))
        .and_then(|p| p.applied_channel)
        .expect("applied");
    let tick = f.tick(0);
    assert_eq!(tick.applied.len(), 1);
    assert_eq!(tick.applied[0].stream, applied_id);
    assert_eq!(tick.applied[0].applied, Applied::absent(EdgeMode::Idle));
    f.accept(f.chunk_for("cmd", "policy", &lease, 7, MS, 2, 0.5), MS)
        .expect("accept");
    let tick = f.tick(2 * MS);
    let sample = &tick.applied[0];
    assert_eq!(
        sample.applied,
        Applied {
            mode: EdgeMode::Driving,
            present: true,
            lease_token: lease.token,
            chunk_seq: 7,
            action_index: 1,
            reason: ReasonCode::Unspecified,
        }
    );
    assert_eq!(&sample.value[..4], &0.5_f32.to_le_bytes());
    // Hold (position): the held value, mode HOLD, present.
    let tick = f.tick(4 * MS);
    assert_eq!(tick.applied[0].applied.mode, EdgeMode::Hold);
    assert!(tick.applied[0].applied.present);
    // Safe: nothing, and why.
    let tick = f.tick(20 * MS);
    assert_eq!(
        tick.applied[0].applied,
        Applied {
            reason: ReasonCode::QueueDry,
            ..Applied::absent(EdgeMode::Safe)
        }
    );
    assert!(tick.applied[0].value.is_empty());
}

#[test]
fn limits_and_the_binary_snap_apply_before_queueing() {
    let mut f = Fixture::with(EdgeConfig::default(), true, &["grip"], None);
    let lease = f.grant_with("policy", "grip", &[], 0);
    f.publish(0);
    f.accept(f.chunk_for("grip", "policy", &lease, 1, 0, 2, 0.8), 0)
        .expect("accept");
    let tick = f.tick(0);
    assert_eq!(
        &tick.action.expect("due").value[..4],
        &1.0_f32.to_le_bytes()
    );
    assert_eq!(&tick.applied[0].value[..4], &1.0_f32.to_le_bytes());
    assert_eq!(f.edge.stats().clamped, 2);
}

#[test]
fn a_virtual_session_runs_the_edge_on_session_ticks() {
    let mut f = Fixture::with(
        EdgeConfig::default(),
        true,
        &["cmd"],
        Some(VirtualConfig::default()),
    );
    let lease = f.grant("policy", 0);
    f.mapping = None;
    f.publish(0);
    // No mapping is needed and the lease is timed by the host: a chunk
    // received at tick 1000 (1 s) with an old publish stamp is fine.
    let mut chunk = f.chunk("policy", &lease, 1, 1_000 * MS, 5);
    chunk.publish_ns = -10_000 * MS;
    f.accept(chunk, 1_000 * MS).expect("accept");
    assert_eq!(f.tick(1_002 * MS).mode, EdgeMode::Driving);
}

/// A latched part commanded once at 0, its lease then renewed for 2 s with
/// nothing else arriving.
fn latched_for_two_seconds() -> (Fixture, Lease) {
    let mut f = Fixture::with(EdgeConfig::default(), true, &["latched"], None);
    let lease = f.grant_with("policy", "latched", &[], 0);
    f.publish(0);
    f.accept(f.chunk_for("latched", "policy", &lease, 1, 0, 1, 0.5), 0)
        .expect("accept");
    assert_eq!(f.tick(0).mode, EdgeMode::Driving);
    for t in (50..=2_000).step_by(50) {
        let now = t * MS;
        let renewed = f.grant_with("policy", "latched", &[], now);
        assert_eq!(renewed.token, lease.token);
        f.publish(now);
        let tick = f.tick(now);
        assert_eq!(tick.mode, EdgeMode::Driving, "at {t} ms");
        assert!(tick.action.is_some() && tick.safe_behavior.is_none());
        assert!(tick.report.is_none(), "at {t} ms");
    }
    (f, lease)
}

#[test]
fn a_latched_part_is_still_driving_after_two_seconds_of_silence() {
    let (f, _) = latched_for_two_seconds();
    let stats = f.edge.stats();
    assert_eq!(stats.accepted, 1, "one chunk, two seconds ago");
    assert_eq!(stats.safe_reports, 0, "no QUEUE_DRY");
    assert_eq!(stats.mode(EdgeMode::Hold).ticks, 0);
}

#[test]
fn a_latched_part_goes_safe_when_its_lease_is_released() {
    let (mut f, lease) = latched_for_two_seconds();
    let channel = f.ch("latched");
    Fixture::run(
        &mut f.auth,
        "policy",
        Command::ReleaseLease {
            channels: vec![channel],
            token: lease.token,
        },
    );
    f.publish(2_001 * MS);
    assert_eq!(f.tick(2_001 * MS).mode, EdgeMode::Hold);
    let tick = f.tick(2_011 * MS);
    assert_eq!(tick.mode, EdgeMode::Safe);
    assert_eq!(tick.safe_behavior.as_deref(), Some("brake"));
    assert!(tick.report.is_none(), "a release is the host's own");
}

#[test]
fn a_latched_part_goes_safe_when_its_lease_lapses() {
    let (mut f, _) = latched_for_two_seconds();
    let tick = f.tick(2_000 * MS + GRACE + MS);
    assert_eq!(tick.mode, EdgeMode::Safe, "safe at once, no hold");
    assert_eq!(tick.safe_behavior.as_deref(), Some("brake"));
    assert_eq!(
        tick.report.map(|(_, cause)| cause),
        Some(SafeCause::LeaseLapsed)
    );
}
