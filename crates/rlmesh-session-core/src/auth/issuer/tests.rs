//! The key schedule: rotations, read permission, missed acks and grant caps.

use super::*;
use crate::auth::keyring::Keyring;
use crate::authority::DeviceAccess;
use crate::crypto::sealed::Header;
use crate::crypto::test_rng::TestRng;
use crate::wire::Sample;

fn pid(id: &str) -> ParticipantId {
    ParticipantId::new(id).expect("id")
}

/// Delivers what is due and acknowledges it, except at `deaf`.
fn exchange(
    issuer: &mut KeyIssuer,
    rings: &mut BTreeMap<ParticipantId, Keyring>,
    deaf: &[&str],
    now: i64,
) {
    if let Some(delivery) = issuer.host_delivery() {
        let ack = KeysAck {
            installed: delivery.keys.iter().map(|k| k.key_id).collect(),
            sending: Vec::new(),
            retired: delivery.retire_through.clone(),
        };
        issuer.host_ack(&ack, now);
    }
    for (who, delivery) in issuer.deliveries(now) {
        let Some(ring) = rings.get_mut(&who) else {
            continue;
        };
        let ack = ring.apply(&delivery, now);
        if !deaf.contains(&who.as_str()) {
            let generation = issuer.members[&who].generation;
            issuer.ack(&who, generation, &ack, now);
        }
    }
}

fn arm_stream() -> Vec<StreamSlot> {
    vec![StreamSlot {
        robot: pid("arm"),
        path: "j".into(),
        channels: vec![ChannelId(1), ChannelId(2)],
        security: Security::Mac,
    }]
}

#[test]
fn a_robots_new_generation_rotates_its_streams_and_retires_the_old_version() {
    let mut rng = TestRng::new(5);
    let mut issuer = KeyIssuer::default();
    let mut rings = BTreeMap::new();
    for (who, generation) in [("arm", 1), ("policy", 2)] {
        issuer
            .attach(
                &mut rng,
                Joining {
                    participant: &pid(who),
                    generation,
                    device: "local",
                    robot: false,
                },
                &Access::default(),
                0,
            )
            .expect("attach");
        rings.insert(pid(who), Keyring::new(Some(pid(who)), generation));
    }
    issuer
        .sync_streams(&mut rng, &arm_stream(), &Access::default(), 0)
        .expect("sync");
    for t in 0..4 {
        exchange(&mut issuer, &mut rings, &[], t);
    }
    assert!(issuer.pending().is_empty(), "version 1 is in place");
    assert!(rings[&pid("arm")].sends_stream(ChannelId(1)));
    let v1: Vec<u32> = rings[&pid("policy")]
        .scopes()
        .map(|(s, _)| s.key_id)
        .collect();

    // The robot restarts: generation 3 takes over its slot at version 2.
    issuer
        .attach(
            &mut rng,
            Joining {
                participant: &pid("arm"),
                generation: 3,
                device: "local",
                robot: false,
            },
            &Access::default(),
            10,
        )
        .expect("attach");
    rings.insert(pid("arm"), Keyring::new(Some(pid("arm")), 3));
    exchange(&mut issuer, &mut rings, &[], 11);
    exchange(&mut issuer, &mut rings, &[], 12);
    exchange(&mut issuer, &mut rings, &[], 13);
    assert!(issuer.pending().is_empty());
    let arm = &rings[&pid("arm")];
    let sending: Vec<(u32, bool)> = arm.scopes().map(|(s, tx)| (s.version, tx)).collect();
    assert_eq!(sending, vec![(2, true)]);
    let policy: Vec<u32> = rings[&pid("policy")]
        .scopes()
        .map(|(s, _)| s.version)
        .collect();
    assert_eq!(policy, vec![2], "retire_through dropped version 1 ({v1:?})");
    assert!(
        v1.iter().all(|id| !issuer.keys.contains_key(id)),
        "the host forgot version 1's key"
    );
    // Recovery hands the new generation its sending key only.
    let recovered: Vec<IssuedKey> = issuer
        .get_keys(&pid("arm"), 3, &Access::default())
        .into_iter()
        .filter(|k| k.purpose == KeyPurpose::Stream)
        .collect();
    assert_eq!(recovered.len(), 1);
    assert!(recovered[0].send && recovered[0].version == 2);
    assert!(
        issuer
            .get_keys(&pid("arm"), 1, &Access::default())
            .is_empty(),
        "an ended generation"
    );
}

#[test]
fn a_device_losing_read_permission_rotates_the_stream_past_its_readers() {
    let mut rng = TestRng::new(8);
    let mut issuer = KeyIssuer::default();
    let mut access = Access::default();
    access.set("laptop", DeviceAccess::new(Some(PriorityClass::Model)));
    let mut rings = BTreeMap::new();
    for (who, generation, device) in [
        ("arm", 1, "local"),
        ("eye", 2, "laptop"),
        ("policy", 3, "local"),
    ] {
        issuer
            .attach(
                &mut rng,
                Joining {
                    participant: &pid(who),
                    generation,
                    device,
                    robot: false,
                },
                &access,
                0,
            )
            .expect("attach");
        rings.insert(pid(who), Keyring::new(Some(pid(who)), generation));
    }
    issuer
        .sync_streams(&mut rng, &arm_stream(), &access, 0)
        .expect("sync");
    for t in 0..4 {
        exchange(&mut issuer, &mut rings, &[], t);
    }
    let sample = Sample {
        channel: ChannelId(1),
        epoch: crate::wire::Epoch(1),
        seq: 1,
        capture_ns: 0,
        publish_ns: 0,
        tick: 0,
        payload: bytes::Bytes::from_static(b"q"),
        applied: None,
    };
    let sealed = |rings: &mut BTreeMap<ParticipantId, Keyring>, seq| {
        let sample = Sample {
            seq,
            ..sample.clone()
        };
        let ring = rings.get_mut(&pid("arm")).expect("arm");
        ring.seal_sample("d/arm/j", &sample, false).expect("seal")
    };
    let read = |rings: &mut BTreeMap<ParticipantId, Keyring>, who: &str, msg: Vec<u8>| {
        let ring = rings.get_mut(&pid(who)).expect(who);
        ring.open_sample("d/arm/j", msg, &pid("arm"), false).is_ok()
    };
    let first = sealed(&mut rings, 1);
    assert!(read(&mut rings, "eye", first.clone()) && read(&mut rings, "policy", first));

    // The laptop's entry narrows to another robot: the slot rotates once,
    // past it, and the robot's next samples are the policy's alone.
    let mut narrowed = DeviceAccess::new(Some(PriorityClass::Model));
    narrowed.parts = vec!["ur10e/**".into()];
    access.set("laptop", narrowed);
    issuer.reauthorize(&mut rng, &access, 10).expect("rotate");
    issuer.reauthorize(&mut rng, &access, 10).expect("once");
    for t in 11..15 {
        exchange(&mut issuer, &mut rings, &[], t);
    }
    assert!(issuer.pending().is_empty());
    let next = sealed(&mut rings, 2);
    assert_eq!(
        Header::parse(&next).expect("header").key_id,
        issuer.slots.values().next().expect("slot").versions[&2]
    );
    assert!(!read(&mut rings, "eye", next.clone()), "the old reader");
    assert!(read(&mut rings, "policy", next));
    assert!(
        issuer
            .get_keys(&pid("eye"), 2, &access)
            .iter()
            .all(|k| k.purpose == KeyPurpose::Estop)
    );
}

#[test]
fn a_missed_ack_leaves_the_retire_pending_and_never_detaches() {
    let mut rng = TestRng::new(6);
    let mut issuer = KeyIssuer::default();
    let mut rings = BTreeMap::new();
    for (who, generation) in [("arm", 1), ("policy", 2), ("board", 3)] {
        issuer
            .attach(
                &mut rng,
                Joining {
                    participant: &pid(who),
                    generation,
                    device: "local",
                    robot: false,
                },
                &Access::default(),
                0,
            )
            .expect("attach");
        rings.insert(pid(who), Keyring::new(Some(pid(who)), generation));
    }
    issuer
        .sync_streams(&mut rng, &arm_stream(), &Access::default(), 0)
        .expect("sync");
    for t in 0..4 {
        exchange(&mut issuer, &mut rings, &[], t);
    }
    issuer
        .attach(
            &mut rng,
            Joining {
                participant: &pid("arm"),
                generation: 4,
                device: "local",
                robot: false,
            },
            &Access::default(),
            10,
        )
        .expect("attach");
    rings.insert(pid("arm"), Keyring::new(Some(pid("arm")), 4));
    // The board never acknowledges: install waits 500 ms, then the
    // sender switches; the retire stays open, naming the board.
    exchange(&mut issuer, &mut rings, &["board"], 11);
    assert_eq!(issuer.pending()[0].stage, "install");
    assert_eq!(issuer.pending()[0].waiting, vec![pid("board")]);
    let later = 11 + INSTALL_WAIT_NS;
    for t in 0..3 {
        exchange(&mut issuer, &mut rings, &["board"], later + t * RETRY_NS);
    }
    let pending = issuer.pending();
    assert_eq!(pending.len(), 1);
    assert_eq!((pending[0].stage, pending[0].version), ("retire", 2));
    assert_eq!(pending[0].waiting, vec![pid("board")]);
    assert!(rings[&pid("arm")].sends_stream(ChannelId(1)));
    assert!(issuer.members.contains_key(&pid("board")), "still attached");
    // Once it acknowledges (a retry reaches it), the rotation completes.
    exchange(&mut issuer, &mut rings, &[], later + 10 * RETRY_NS);
    assert!(issuer.pending().is_empty());
}

#[test]
fn grants_are_capped_per_device_and_robot_and_a_higher_class_evicts_the_newest_lowest() {
    let mut rng = TestRng::new(7);
    let mut issuer = KeyIssuer::default();
    issuer
        .attach(
            &mut rng,
            Joining {
                participant: &pid("arm"),
                generation: 1,
                device: "local",
                robot: false,
            },
            &Access::default(),
            0,
        )
        .expect("attach");
    let request = |holder: &str, device: &str, class, token| GrantRequest {
        holder: pid(holder),
        holder_generation: 1,
        device: device.into(),
        class,
        token: LeaseToken(token),
        parts: vec![GrantPart {
            robot: pid("arm"),
            channel: ChannelId(1),
            priority: ChannelPriority::Realtime,
            security: Security::Mac,
        }],
        reads_all: vec![pid("arm")],
    };
    let model = PriorityClass::Model;
    let a = issuer.prepare_grant(&mut rng, request("a", "d1", model, 1), 0);
    let b = issuer.prepare_grant(&mut rng, request("b", "d1", model, 2), 0);
    let busy = issuer.prepare_grant(&mut rng, request("c", "d1", model, 3), 0);
    assert_eq!(busy.expect_err("third").status, CtlStatus::LeaseBusy);
    let c = issuer.prepare_grant(&mut rng, request("c", "d2", model, 3), 0);
    let d = issuer.prepare_grant(&mut rng, request("d", "d2", model, 4), 0);
    let (a, b, c, d) = (a.expect("a"), b.expect("b"), c.expect("c"), d.expect("d"));
    let full = issuer.prepare_grant(&mut rng, request("e", "d3", model, 5), 0);
    assert_eq!(full.expect_err("robot full").status, CtlStatus::LeaseBusy);
    let operator =
        issuer.prepare_grant(&mut rng, request("o", "d3", PriorityClass::Operator, 6), 0);
    assert_eq!(
        operator.expect("evicts").evicted,
        vec![d.id],
        "the newest model grant"
    );
    assert!(issuer.grant(d.id).is_none());
    // Nothing installed: they all time out together.
    let decided = issuer.ready_grants(GRANT_WAIT_NS);
    assert_eq!(decided.len(), 4);
    assert!(decided.iter().all(|(_, r)| r.is_err()));
    let _ = (a, b, c);
}

fn join(issuer: &mut KeyIssuer, rng: &mut TestRng, who: &str, generation: u32, robot: bool) {
    issuer
        .attach(
            rng,
            Joining {
                participant: &pid(who),
                generation,
                device: "local",
                robot,
            },
            &Access::default(),
            0,
        )
        .expect("attach");
}

#[test]
fn attaching_the_generation_in_effect_again_changes_nothing() {
    let mut rng = TestRng::new(9);
    let mut issuer = KeyIssuer::default();
    join(&mut issuer, &mut rng, "arm", 1, true);
    let first = issuer.estop_key(&pid("arm"), 1).expect("key").key_id;
    join(&mut issuer, &mut rng, "arm", 1, true);
    assert_eq!(issuer.estop_key(&pid("arm"), 1).expect("key").key_id, first);
}

#[test]
fn a_generation_that_ends_takes_back_its_undelivered_estop_key() {
    let mut rng = TestRng::new(10);
    let mut issuer = KeyIssuer::default();
    join(&mut issuer, &mut rng, "arm", 1, true);
    join(&mut issuer, &mut rng, "policy", 2, false);
    let estop = issuer.estop_key(&pid("policy"), 2).expect("key").key_id;
    assert!(issuer.outbox[&pid("arm")].installing(estop), "on its way");
    issuer.detach(&pid("policy"));
    for (who, delivery) in issuer.deliveries(0) {
        assert!(
            delivery.keys.iter().all(|k| k.key_id != estop),
            "still going to {who}"
        );
    }
}

#[test]
fn a_detach_fails_the_pending_grants_that_needed_it_and_no_others() {
    let mut rng = TestRng::new(11);
    let mut issuer = KeyIssuer::default();
    join(&mut issuer, &mut rng, "arm", 1, true);
    join(&mut issuer, &mut rng, "gripper", 2, true);
    let mut prepare = |holder: &str, robot: &str, token| {
        let request = GrantRequest {
            holder: pid(holder),
            holder_generation: 1,
            device: holder.into(),
            class: PriorityClass::Model,
            token: LeaseToken(token),
            parts: vec![GrantPart {
                robot: pid(robot),
                channel: ChannelId(1),
                priority: ChannelPriority::Realtime,
                security: Security::Mac,
            }],
            reads_all: Vec::new(),
        };
        issuer
            .prepare_grant(&mut rng, request, 0)
            .expect("prepared")
            .id
    };
    let by_a = prepare("a", "arm", 1);
    let on_gripper = prepare("b", "gripper", 2);
    let on_arm = prepare("c", "arm", 3);
    // A robot's cap counts its own pending grants, not the gripper's.
    let more = [prepare("d", "arm", 4), prepare("e", "arm", 5)];
    assert_eq!(issuer.detach(&pid("stranger")), Vec::<GrantId>::new());
    assert_eq!(issuer.detach(&pid("a")), vec![by_a], "as holder");
    assert_eq!(
        issuer.detach(&pid("arm")),
        vec![on_arm, more[0], more[1]],
        "as robot"
    );
    assert!(issuer.grant(on_gripper).is_some());
}

/// The lease keys of token `token` in `keys`, as `(key id, send)`.
fn lease_keys(keys: &[IssuedKey], token: u64) -> Vec<(u32, bool)> {
    let mut out: Vec<(u32, bool)> = keys
        .iter()
        .filter(|k| k.token == LeaseToken(token))
        .map(|k| (k.key_id, k.send))
        .collect();
    out.sort_unstable();
    out
}

#[test]
fn a_committed_grant_hands_its_holder_the_keys_until_its_token_ends() {
    let mut rng = TestRng::new(13);
    let mut issuer = KeyIssuer::default();
    let mut rings = BTreeMap::new();
    // One generation for all, so only identity tells the keys apart.
    for (who, robot) in [("arm", true), ("policy", false), ("viewer", false)] {
        join(&mut issuer, &mut rng, who, 1, robot);
        rings.insert(pid(who), Keyring::new(Some(pid(who)), 1));
    }
    let request = |holder: &str, token, reads_all: Vec<ParticipantId>| GrantRequest {
        holder: pid(holder),
        holder_generation: 1,
        device: holder.into(),
        class: PriorityClass::Model,
        token: LeaseToken(token),
        parts: [
            (1, ChannelPriority::Realtime),
            (3, ChannelPriority::Control),
        ]
        .map(|(channel, priority)| GrantPart {
            robot: pid("arm"),
            channel: ChannelId(channel),
            priority,
            security: Security::Mac,
        })
        .to_vec(),
        reads_all,
    };
    let access = Access::default();
    let id = issuer
        .prepare_grant(&mut rng, request("policy", 7, vec![pid("arm")]), 0)
        .expect("prepared")
        .id;
    assert!(
        issuer.ready_grants(0).is_empty(),
        "the robot has not installed"
    );
    assert!(
        lease_keys(&issuer.get_keys(&pid("policy"), 1, &access), 7).is_empty(),
        "nothing for the holder before the commit"
    );
    for t in 0..4 {
        exchange(&mut issuer, &mut rings, &[], t);
    }
    assert!(
        issuer.host_delivery().is_none() && issuer.deliveries(4 + RETRY_NS).is_empty(),
        "everything acknowledged: nothing to send, even after the retry interval"
    );
    let ready: Vec<(GrantId, bool)> = issuer
        .ready_grants(4)
        .into_iter()
        .map(|(g, r)| (g, r.is_ok()))
        .collect();
    assert_eq!(ready, [(id, true)]);
    let committed = issuer.commit_grant(id).expect("pending");
    assert!(issuer.commit_grant(id).is_none(), "committed once");
    // A k_cmd per transport priority, which the holder sends under, and
    // the robot's k_obs, which it receives.
    let mut held: Vec<(KeyPurpose, bool, Security)> = committed
        .keys
        .iter()
        .map(|k| (k.purpose, k.send, k.security))
        .collect();
    held.sort_by_key(|(p, ..)| *p);
    assert_eq!(
        held,
        [
            (KeyPurpose::Command, true, Security::Mac),
            (KeyPurpose::Command, true, Security::Mac),
            (KeyPurpose::LockstepReply, false, Security::Mac),
        ]
    );
    let mut parts: Vec<ChannelId> = committed.key_ids.iter().map(|(c, _)| *c).collect();
    parts.sort_unstable();
    assert_eq!(parts, [ChannelId(1), ChannelId(3)]);
    assert_ne!(committed.key_ids[0].1, committed.key_ids[1].1);
    // Recovery: the holder and the robot get the grant's keys, each
    // sending under its own; a bystander of the same generation gets none.
    let holder = lease_keys(&committed.keys, 7);
    let flipped: Vec<(u32, bool)> = holder.iter().map(|(id, send)| (*id, !send)).collect();
    assert_eq!(
        lease_keys(&issuer.get_keys(&pid("policy"), 1, &access), 7),
        holder
    );
    assert_eq!(
        lease_keys(&issuer.get_keys(&pid("arm"), 1, &access), 7),
        flipped
    );
    assert!(lease_keys(&issuer.get_keys(&pid("viewer"), 1, &access), 7).is_empty());
    // The robot also holds every other member's k_estop, to receive.
    let estops: Vec<(Option<ParticipantId>, bool)> = issuer
        .get_keys(&pid("arm"), 1, &access)
        .into_iter()
        .filter(|k| k.purpose == KeyPurpose::Estop)
        .map(|k| (k.sender, k.send))
        .collect();
    assert_eq!(
        estops,
        [
            (Some(pid("arm")), true),
            (Some(pid("policy")), false),
            (Some(pid("viewer")), false)
        ]
    );
    // Its token ends: the committed keys go; a pending grant's stay.
    let pending = issuer
        .prepare_grant(&mut rng, request("viewer", 8, Vec::new()), 5)
        .expect("prepared");
    issuer.end_tokens(|_| false);
    let robot = issuer.get_keys(&pid("arm"), 1, &access);
    assert!(lease_keys(&robot, 7).is_empty());
    assert_eq!(
        lease_keys(&robot, 8).len(),
        2,
        "grant {} pending",
        pending.id
    );
}

#[test]
fn syncing_the_same_streams_again_sends_nothing() {
    let mut rng = TestRng::new(14);
    let mut issuer = KeyIssuer::default();
    let mut rings = BTreeMap::new();
    for (who, generation) in [("arm", 1), ("policy", 2)] {
        join(&mut issuer, &mut rng, who, generation, false);
        rings.insert(pid(who), Keyring::new(Some(pid(who)), generation));
    }
    issuer
        .sync_streams(&mut rng, &arm_stream(), &Access::default(), 0)
        .expect("sync");
    for t in 0..4 {
        exchange(&mut issuer, &mut rings, &[], t);
    }
    issuer
        .sync_streams(&mut rng, &arm_stream(), &Access::default(), 4)
        .expect("again");
    assert!(issuer.host_delivery().is_none() && issuer.deliveries(5).is_empty());
}

#[test]
fn an_issuers_debug_output_never_shows_a_keys_bytes() {
    let mut rng = TestRng::new(12);
    let mut issuer = KeyIssuer::default();
    join(&mut issuer, &mut rng, "arm", 1, true);
    let key = issuer.estop_key(&pid("arm"), 1).expect("key");
    crate::crypto::assert_redacted(&issuer, &key.key.0);
}
