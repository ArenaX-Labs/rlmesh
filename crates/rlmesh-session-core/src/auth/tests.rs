//! Handshakes, links, the host's admission and the request window, end
//! to end.

use super::*;
use crate::authority::Access;
use crate::crypto::sealed::Kind;
use crate::crypto::test_rng::TestRng;
use crate::prelude::*;
use crate::wire::codec;
use crate::wire::heartbeat::{Heartbeat, HeartbeatEcho};
use crate::wire::{Command, CtlError, CtlReply, CtlStatus, DeviceView, Epoch, ParticipantId};

fn pid(id: &str) -> ParticipantId {
    ParticipantId::new(id).expect("id")
}

fn host() -> (HostIdentity, HostAuth) {
    let identity = HostIdentity::generate(&mut TestRng::new(1)).expect("identity");
    let mut auth = HostAuth::new(&identity, "local", "demo", 7);
    let laptop = Credential::local(&[9; 32], [0; 32]);
    auth.enroll(*laptop.device_public(), "laptop-josh");
    (identity, auth)
}

fn handshake(
    identity: &HostIdentity,
    auth: &mut HostAuth,
    credential: &Credential,
    name: &str,
) -> Result<(Attached, Candidate), CtlError> {
    let mut rng = TestRng::new(3);
    let (initiator, request) = credential
        .start(&mut rng, "local", "demo", 7, &pid(name))
        .expect("start");
    let (msg2, candidate) = auth.handshake(&mut rng, identity, &request, &Access::default())?;
    Ok((finish(initiator, &msg2).expect("finish"), candidate))
}

#[test]
fn a_link_seals_verbs_and_heartbeats_and_takes_only_the_hosts_replies_and_broadcasts() {
    let (identity, mut auth) = host();
    let (attached, mut candidate) =
        handshake(&identity, &mut auth, &identity.local_credential(), "robot").expect("handshake");
    assert_eq!(attached.accept.device, LOCAL_DEVICE);
    assert_eq!(attached.accept.host_instance, 7);
    assert_eq!(attached.accept.generation, candidate.generation);
    let mut link = Link::new(pid("robot"), attached);

    // A verb: the host opens it; its sealed reply answers that id only.
    let (id, mut msg) = link.request("ctl", None, &Command::Arm).expect("seal");
    let opened = candidate.up.open("ctl", &mut msg).expect("open");
    let request = codec::decode_request(opened.body).expect("request");
    assert_eq!((request.request_id, request.command), (1, Command::Arm));
    let snapshot = crate::authority::Authority::new(
        crate::wire::SessionId::named("demo"),
        crate::authority::AuthorityConfig::default(),
    )
    .snapshot(Vec::new(), 0);
    let reply = |request_id| CtlReply {
        result: Ok(()),
        state: snapshot.clone(),
        leases: Vec::new(),
        request_id,
        keys: Vec::new(),
        assignment: None,
        run: None,
    };
    let mut sealed = candidate
        .down
        .seal(Kind::CtlReply, "ctl", &codec::encode_reply(&reply(id)))
        .expect("seal");
    assert_eq!(
        link.reply("ctl", &mut sealed.clone(), id, 0).expect("ours"),
        ReplyOutcome::Answer(Box::new(reply(id)))
    );
    assert!(
        link.reply("ctl", &mut sealed, id, 0).is_err(),
        "a replay is dropped"
    );
    // A retry keeps its id under a fresh seq.
    let mut again = link.resend("ctl", id, None, &Command::Arm).expect("seal");
    let retried = codec::decode_request(candidate.up.open("ctl", &mut again).expect("open").body);
    assert_eq!(retried.expect("request").request_id, id);
    // An unanswered id holds the window: 63 more ids, then no new one
    // until it is answered.
    assert!(!link.unanswered(id), "answered");
    let (lost, _) = link.request("ctl", None, &Command::Arm).expect("seal");
    for _ in 0..63 {
        let (next, _) = link.request("ctl", None, &Command::Arm).expect("seal");
        let mut sealed = candidate
            .down
            .seal(Kind::CtlReply, "ctl", &codec::encode_reply(&reply(next)))
            .expect("seal");
        link.reply("ctl", &mut sealed, next, 0).expect("answered");
    }
    assert!(matches!(
        link.request("ctl", None, &Command::Arm),
        Err(LinkError::Outstanding(id)) if id == lost
    ));
    let mut late = candidate
        .down
        .seal(Kind::CtlReply, "ctl", &codec::encode_reply(&reply(lost)))
        .expect("seal");
    link.reply("ctl", &mut late, lost, 0)
        .expect("the retry's answer");
    assert!(link.request("ctl", None, &Command::Arm).is_ok());
    // Anyone can answer `ctl`: a plain refusal is only a hint, a plain
    // answer nothing.
    let refusal = CtlError::new(CtlStatus::Unauthenticated, "superseded");
    let mut plain = codec::encode_reply(&CtlReply {
        result: Err(refusal.clone()),
        ..reply(id)
    });
    assert_eq!(
        link.reply("ctl", &mut plain, id, 0).expect("a hint"),
        ReplyOutcome::Hint(refusal)
    );
    assert!(
        link.reply("ctl", &mut codec::encode_reply(&reply(id)), id, 0)
            .is_err()
    );

    // Heartbeats up, echoes down.
    let heartbeat = Heartbeat {
        participant: pid("robot"),
        epoch: Epoch(1),
        mode: crate::wire::Mode::Disarmed,
        seq: 1,
        t1_ns: 5,
        previous: None,
        last_seq_seen: Vec::new(),
        last_chunk_accepted: 0,
        last_chunk_applied: 0,
        queue_depth: 0,
        health: crate::wire::heartbeat::Health::Ok,
        rejected: Vec::new(),
        clock_refused: Vec::new(),
    };
    let mut hb = link.heartbeat("hb", &heartbeat).expect("seal");
    assert_eq!(
        candidate.up.open("hb", &mut hb).expect("open").header.kind,
        Kind::Heartbeat
    );
    let echo = HeartbeatEcho {
        seq: 1,
        t1_ns: 5,
        t2_ns: 6,
        t3_ns: 7,
        epoch: Epoch(1),
    };
    let mut sealed = candidate
        .down
        .seal(Kind::HeartbeatEcho, "echo", &codec::encode_echo(&echo))
        .expect("seal");
    assert_eq!(link.echo("echo", &mut sealed).expect("echo"), echo);

    // Broadcasts: signed by the host's instance, newer than the last.
    let mut signer = identity.signer(7);
    let body = codec::encode_state(&snapshot);
    let first = signer
        .sign(crate::crypto::SignedKind::AuthorityState, "state", &body)
        .expect("sign");
    let second = signer
        .sign(crate::crypto::SignedKind::AuthorityState, "state", &body)
        .expect("sign");
    let mut verifier = link.host_verifier();
    assert!(link.state("state", &second).is_ok());
    assert!(link.state("state", &first).is_err(), "stale");
    assert!(
        open_state(&mut verifier, "state", &first).is_ok(),
        "its own floor"
    );
    assert!(link.state("state", &body).is_err(), "unsigned");
    let stranger =
        identity
            .signer(8)
            .sign(crate::crypto::SignedKind::AuthorityState, "state", &body);
    assert!(
        link.state("state", &stranger.expect("sign")).is_err(),
        "another instance"
    );

    // The robot's device view: the same checks, on its own floor, and
    // only a view of the robot its key names.
    let key = "s/demo/host/view/robot";
    let view = DeviceView::of(&snapshot, &pid("robot"));
    let body = codec::encode_view(&view);
    let older = signer
        .sign(crate::crypto::SignedKind::DeviceView, key, &body)
        .expect("sign");
    let newer = signer
        .sign(crate::crypto::SignedKind::DeviceView, key, &body)
        .expect("sign");
    assert_eq!(link.open_view(key, &newer).expect("verified"), view);
    assert!(link.open_view(key, &older).is_err(), "stale");
    assert!(link.open_view(key, &newer).is_err(), "replayed");
    let forged = HostIdentity::generate(&mut TestRng::new(3))
        .expect("identity")
        .signer(7)
        .sign(crate::crypto::SignedKind::DeviceView, key, &body)
        .expect("sign");
    assert!(link.open_view(key, &forged).is_err(), "another host's key");
    let state = signer
        .sign(crate::crypto::SignedKind::AuthorityState, key, &body)
        .expect("sign");
    assert!(link.open_view(key, &state).is_err(), "not a view");
    let other = "s/demo/host/view/arm";
    let misnamed = signer
        .sign(crate::crypto::SignedKind::DeviceView, other, &body)
        .expect("sign");
    assert!(link.open_view(other, &misnamed).is_err(), "another robot's");
}

#[test]
fn unknown_keys_stale_instances_and_taken_names_are_refused() {
    let (identity, mut auth) = host();
    // A key the host never enrolled.
    let stranger = Credential::local(&[4; 32], *identity.static_public());
    let err = handshake(&identity, &mut auth, &stranger, "robot").expect_err("unknown");
    assert_eq!(err.status, CtlStatus::Unauthenticated);
    // Another host's static key (a stale address).
    let other = HostIdentity::generate(&mut TestRng::new(2)).expect("other");
    let stale = Credential::local(identity.local_secret(), *other.static_public());
    let err = handshake(&identity, &mut auth, &stale, "robot").expect_err("stale");
    assert_eq!(err.status, CtlStatus::Unauthenticated);
    // A name binds to the first device it took effect for.
    let (_, local) =
        handshake(&identity, &mut auth, &identity.local_credential(), "eye-3").expect("local");
    auth.take_effect(&local.participant, &local.device, local.generation);
    let laptop = Credential::local(&[9; 32], *identity.static_public());
    let err = handshake(&identity, &mut auth, &laptop, "eye-3").expect_err("taken");
    assert_eq!(err.status, CtlStatus::NotAuthorized);
    assert!(err.message.contains("belongs to device local"), "{err}");
    handshake(&identity, &mut auth, &laptop, "cockpit").expect("another name is free");
}

#[test]
fn a_pin_with_a_board_refuses_another_board_at_declare() {
    let (_, mut auth) = host();
    let skull = pid("skull");
    let board = [0xb0, 0xb1, 0, 0xb3, 0xb4, 0, 0, 0xb7];
    auth.pin(
        &skull,
        Some(Pin {
            device: "rig-pi".into(),
            board_id: Some(board),
        }),
    );
    assert!(auth.check_board(&skull, "rig-pi", Some(&board)).is_ok());
    for other in [Some(&[1; 8]), None] {
        let err = auth
            .check_board(&skull, "rig-pi", other)
            .expect_err("refused");
        assert_eq!(err.status, CtlStatus::NotAuthorized);
    }
    let err = auth
        .check_board(&skull, "rig-pi", Some(&[1; 8]))
        .expect_err("refused");
    assert_eq!(
        err.message,
        "board on rig-pi is 0101010101010101, expected skull's b0b100b3b40000b7"
    );
    // A pin without a board, and an unpinned name, check none.
    auth.pin(
        &skull,
        Some(Pin {
            device: "rig-pi".into(),
            board_id: None,
        }),
    );
    assert!(auth.check_board(&skull, "rig-pi", Some(&[1; 8])).is_ok());
    assert!(auth.check_board(&pid("eye-l"), "eye-l", None).is_ok());
}

#[test]
fn only_a_newer_generation_takes_effect() {
    let (identity, mut auth) = host();
    let local = identity.local_credential();
    let (_, older) = handshake(&identity, &mut auth, &local, "policy").expect("first");
    let (_, newer) = handshake(&identity, &mut auth, &local, "policy").expect("second");
    assert!(newer.generation > older.generation);
    assert!(
        newer.up_key_id > older.up_key_id + 1,
        "key ids are never reused"
    );
    let take = |auth: &HostAuth, c: &Candidate| {
        auth.may_take_effect(&c.participant, &c.device, c.generation)
    };
    take(&auth, &newer).expect("newer");
    auth.take_effect(&newer.participant, &newer.device, newer.generation);
    let err = take(&auth, &older).expect_err("superseded");
    assert_eq!(err.status, CtlStatus::Unauthenticated);
    assert_eq!(auth.current(&pid("policy")), Some(newer.generation));
}

#[test]
fn the_request_window_answers_retries_from_its_cache_and_expires_old_ids() {
    let mut window = RequestWindow::<&str>::default();
    assert_eq!(window.admit(0), Admit::Expired, "ids start at 1");
    assert_eq!(window.admit(1), Admit::Execute);
    window.complete(1, "one");
    assert_eq!(
        window.admit(1),
        Admit::Done(&"one"),
        "a retry is answered again"
    );
    assert_eq!(window.admit(64), Admit::Execute);
    assert_eq!(window.admit(1), Admit::Done(&"one"), "64 ids fit");
    // An id above the window moves the floor: 1 falls out.
    assert_eq!(window.admit(65), Admit::Execute);
    assert_eq!(window.admit(1), Admit::Expired);
    window.complete(65, "sixty-five");
    assert_eq!(window.admit(200), Admit::Execute);
    assert_eq!(window.admit(65), Admit::Expired);
    assert_eq!(window.admit(137), Admit::Execute);
}
