//! Fencing: preempted and rebound tokens, sequence numbers and lapsed leases.

use super::*;

#[test]
fn a_preempted_controllers_queued_and_inflight_chunks_are_fenced() {
    let mut f = Fixture::with(EdgeConfig::default(), false, &["cmd"], None);
    let model = f.grant("policy", 0);
    f.publish(0);
    f.accept(f.chunk("policy", &model, 1, 0, 50), 0)
        .expect("accept");
    assert_eq!(f.tick(MS).mode, EdgeMode::Driving);

    let human = f.grant("operator", 2 * MS);
    f.publish(2 * MS);
    // The model's in-flight chunk is fenced; its queued actions go at the
    // next tick, and the edge waits (braking) until the operator
    // delivers.
    assert_eq!(
        f.accept(f.chunk("policy", &model, 2, 2 * MS, 50), 2 * MS),
        Err(Rejection::StaleToken)
    );
    let tick = f.tick(3 * MS);
    assert_eq!(tick.mode, EdgeMode::Hold);
    assert!(tick.action.is_none() && tick.safe_behavior.is_some());
    // A chunk claiming the operator's token but sent by someone else, or
    // under another binding.
    assert_eq!(
        f.accept(f.chunk("policy", &human, 1, 4 * MS, 50), 4 * MS),
        Err(Rejection::WrongSource)
    );
    let mut other = f.chunk("operator", &human, 1, 4 * MS, 50);
    other.binding = Fingerprint::from(1);
    assert_eq!(f.accept(other, 4 * MS), Err(Rejection::WrongBinding));
    f.accept(f.chunk("operator", &human, 1, 4 * MS, 50), 4 * MS)
        .expect("accept");
    let tick = f.tick(4 * MS);
    assert_eq!(tick.mode, EdgeMode::Driving);
    assert_eq!(tick.action.expect("due").chunk.source, pid("operator"));
    let stats = f.edge.stats();
    assert_eq!(stats.accepted, 2);
    assert_eq!(stats.rejected(Rejection::StaleToken), 1);
    assert_eq!(stats.rejected(Rejection::WrongSource), 1);
    assert_eq!(stats.rejected(Rejection::WrongBinding), 1);
    assert_eq!(stats.queue.fenced, 49, "{stats:?}");
}

#[test]
fn a_rebinding_fences_the_old_tokens_chunks() {
    let mut f = Fixture::new();
    let lease = f.grant("policy", 0);
    f.publish(0);
    f.accept(f.chunk("policy", &lease, 1, 0, 20), 0)
        .expect("accept");
    assert_eq!(f.tick(0).mode, EdgeMode::Driving);
    let channels = vec![f.ch("cmd")];
    Fixture::run(
        &mut f.auth,
        "policy",
        Command::DeclareBinding {
            fingerprint: Fingerprint::from(10),
            revision: 2,
            table_json: "{}".into(),
            channels,
            replan_hz: 0.0,
            contracts: Vec::new(),
            grids: Vec::new(),
        },
    );
    f.publish(MS);
    let rebound = f.auth.lease(f.ch("cmd")).expect("lease").clone();
    assert!(rebound.token > lease.token);
    assert_eq!(
        f.accept(f.chunk("policy", &lease, 2, MS, 20), MS),
        Err(Rejection::StaleToken),
        "an in-flight chunk of the old binding"
    );
    // The queue of the old token is purged on the next tick, and the new
    // token's chunks drive.
    let tick = f.tick(MS);
    assert_eq!(tick.mode, EdgeMode::Hold, "nothing of the new token yet");
    f.accept(f.chunk("policy", &rebound, 1, 2 * MS, 20), 2 * MS)
        .expect("accept");
    assert_eq!(
        f.tick(2 * MS).action.map(|a| a.chunk.lease_token),
        Some(rebound.token)
    );
    assert!(f.edge.stats().queue.fenced > 0);
}

#[test]
fn seq_is_per_token_and_void_or_invalid_chunks_are_rejected() {
    let mut f = Fixture::new();
    let model = f.grant("policy", 0);
    f.publish(0);
    let mut late = f.chunk("policy", &model, 1, 0, 3);
    late.valid_until_ns = 5 * MS;
    assert_eq!(f.accept(late, 6 * MS), Err(Rejection::Expired));
    let nan = f.chunk_for("cmd", "policy", &model, 1, 0, 3, f32::NAN);
    assert_eq!(f.accept(nan, 0), Err(Rejection::InvalidValue));
    let mut short = f.chunk("policy", &model, 1, 0, 3);
    short.actions = short.actions.slice(..20);
    assert_eq!(f.accept(short, 0), Err(Rejection::Malformed));
    let mut empty = f.chunk("policy", &model, 1, 0, 3);
    empty.count = 0;
    empty.actions = Bytes::new();
    assert_eq!(f.accept(empty, 0), Err(Rejection::Empty));
    for (seq, ok) in [(2, true), (2, false), (1, false)] {
        assert_eq!(
            f.accept(f.chunk("policy", &model, seq, 0, 3), 0).is_ok(),
            ok
        );
    }
    // A new holder's token starts its own seq.
    let human = f.grant("operator", MS);
    f.publish(MS);
    f.accept(f.chunk("operator", &human, 1, MS, 3), MS)
        .expect("accept");
    let stats = f.edge.stats();
    assert_eq!(stats.accepted, 2);
    assert_eq!(stats.rejected(Rejection::StaleSeq), 2);
    // A closing scene's view fences every lease, before the host ends
    // them.
    if let Some(view) = f.snapshot.as_mut() {
        view.scene = Some(Box::new(crate::wire::SceneInfo {
            run: "r-7k2m".into(),
            name: "rig".into(),
            fingerprint: crate::wire::Fingerprint::from(7),
            closing: Some(crate::wire::Reason::new(ReasonCode::SceneEnded)),
            rejoin_until_ns: None,
        }));
    }
    assert_eq!(
        f.accept(f.chunk("operator", &human, 2, MS, 3), MS),
        Err(Rejection::NoLease)
    );
}

#[test]
fn the_lease_lapses_by_the_edge_clock_alone_and_the_edge_reports_safe() {
    let mut f = Fixture::new();
    let lease = f.grant("policy", 0);
    // One snapshot, then the host goes quiet: no renewal ever arrives.
    f.publish(0);
    f.accept(f.chunk("policy", &lease, 1, 0, 140), 0)
        .expect("accept");
    assert_eq!(f.tick(MS).mode, EdgeMode::Driving);
    let tick = f.tick(GRACE + MS);
    assert_eq!(tick.mode, EdgeMode::Safe, "safe at once, no hold");
    assert_eq!(tick.safe_behavior.as_deref(), Some("brake"));
    assert!(tick.report.is_none(), "the host may be losing the holder");
    let mut late = f.chunk("policy", &lease, 2, GRACE + 2 * MS, 10);
    late.publish_ns = GRACE + 2 * MS;
    assert_eq!(f.accept(late, GRACE + 2 * MS), Err(Rejection::LeaseExpired));
    // The host comes back with the holder alive: the fall was the edge's.
    let epoch = f.auth.epoch();
    f.auth.heartbeat(&pid("policy"), epoch, 2 * GRACE);
    f.publish(2 * GRACE);
    assert_eq!(
        f.tick(2 * GRACE + MS).report,
        Some((epoch, SafeCause::LeaseLapsed))
    );
}

#[test]
fn each_member_counts_its_chunks_losses_lateness_and_longest_gap() {
    let mut f = Fixture::new();
    let model = f.grant("policy", 0);
    f.publish(0);
    let cmd = f.ch("cmd");
    // Seq 1 on its slot, 2 lost, 3 three ms late, 4 refused (a copy of 3),
    // then 6 with 5 lost, 45 ms late and 40 ms after 3.
    for (seq, t0, at, ok) in [
        (1, 0, 0, true),
        (3, 2 * MS, 5 * MS, true),
        (3, 5 * MS, 5 * MS, false),
        (6, 0, 45 * MS, true),
    ] {
        assert_eq!(
            f.accept(f.chunk("policy", &model, seq, t0, 3), at).is_ok(),
            ok
        );
    }
    // A chunk that never opened reached the edge all the same.
    f.edge.count_unread(cmd, Rejection::Unauthenticated);
    let stats = f.edge.stats();
    let timing = *stats.timing(cmd).expect("cmd's timing");
    assert_eq!((timing.received, timing.accepted, timing.lost), (5, 3, 3));
    assert_eq!(timing.late_ms, [1, 0, 1, 0, 0, 0, 1, 0]);
    assert_eq!(timing.late_max_us, 45_000);
    assert_eq!(timing.gap_max_us, 40_000);
    assert_eq!(f.edge.timing().count(), 1);
    // A new token is a new run: no loss, no gap across it.
    let human = f.grant("operator", 100 * MS);
    f.publish(100 * MS);
    f.accept(f.chunk("operator", &human, 9, 100 * MS, 3), 170 * MS)
        .expect("accept");
    let timing = *f.edge.stats().timing(cmd).expect("timing");
    assert_eq!((timing.lost, timing.gap_max_us), (3, 40_000));
    assert_eq!(timing.late_ms[7], 1, "70 ms late");
}

#[test]
fn lateness_buckets_are_powers_of_two_milliseconds() {
    let bucket = crate::wire::ChunkTiming::bucket;
    assert_eq!(
        [
            -5,
            0,
            999_999,
            MS,
            2 * MS - 1,
            2 * MS,
            63 * MS,
            64 * MS,
            i64::MAX
        ]
        .map(bucket),
        [0, 0, 0, 1, 1, 2, 6, 7, 7]
    );
}
