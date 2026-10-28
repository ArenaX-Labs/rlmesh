//! Runs under Node through `wasm-bindgen-test-runner` (no browser):
//! `mise run test:js:wasm`. Fixtures are real v2 messages, encoded by
//! `rlmesh_session_core::wire` and normalised by `rlmesh_session_core::spec`.
#![cfg(target_arch = "wasm32")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use bytes::Bytes;
use js_sys::{Array, Float64Array, Reflect, Uint8Array};
use rlmesh_session_core::spec::normalise;
use rlmesh_session_core::wire::codec;
use rlmesh_session_core::wire::version::SPEC_VERSION;
use rlmesh_session_core::wire::{
    Applied, AuthoritySnapshot, BindingRecord, ChannelId, ClockMapping, ClockMode, Command,
    CtlError, CtlReply, CtlStatus, EdgeMode, Epoch, Fingerprint, HeartbeatEcho, LastStop, Lease,
    LeaseToken, Mode, PartSpec, ParticipantId, PriorityClass, Reason, ReasonCode, Sample,
    SessionId,
};
use rlmesh_wasm::frame;
use rlmesh_wasm::*;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

const ROBOT: &str = r#"{
    "arm": {"kind": "joints", "count": 7, "fields": ["pos", "vel"], "hz": 1000,
            "command": "position", "hold_max_ms": 100, "max_age_ms": 50},
    "gripper": {"kind": "gripper", "hz": 100, "command": "position", "safe": "hold"},
    "scene/cam": {"kind": "camera", "width": 640, "height": 480, "hz": 20,
                  "encoding": "jpeg", "quality": 85}
}"#;
const SESSION: &str = "0192f3a4-5b6c-7d8e-9f00-112233445566";
const MS: i64 = 1_000_000;
/// Local and session clocks around 2023-11-14 (ms since the epoch).
const T0: f64 = 1_700_000_000_000.0;

fn get(o: &JsValue, k: &str) -> JsValue {
    Reflect::get(o, &JsValue::from_str(k)).unwrap()
}

fn id(s: &str) -> ParticipantId {
    ParticipantId::new(s).unwrap()
}

fn parts() -> Vec<PartSpec> {
    let mut parts = normalise(&id("arm7"), SPEC_VERSION, ROBOT).unwrap();
    for (i, p) in parts.iter_mut().enumerate() {
        p.channel = ChannelId(10 * (i as u32 + 1));
        p.applied_channel = p
            .command
            .as_ref()
            .map(|_| ChannelId(10 * (i as u32 + 1) + 1));
    }
    parts
}

fn mapping(who: &str, offset_ns: i64) -> ClockMapping {
    ClockMapping {
        participant: id(who),
        offset_ns,
        error_ns: 100_000,
        rtt_ns: 200_000,
        sample_ns: 0,
        drift_bound_ppm: 100.0,
    }
}

fn snapshot(revision: u64, leases: Vec<Lease>) -> AuthoritySnapshot {
    AuthoritySnapshot {
        session: SESSION.parse::<SessionId>().unwrap(),
        epoch: Epoch(1),
        revision,
        mode: Mode::Active,
        estop_latched: false,
        roster: Vec::new(),
        leases,
        clocks: vec![mapping("arm7", 0), mapping("web-1", 0)],
        host_ns: (T0 as i64) * MS + revision as i64,
        parts: parts(),
        protocol_minor: 0,
        supports: Vec::new(),
        clock_mode: ClockMode::Wall,
        tick_ns: 0,
        bindings: vec![BindingRecord {
            participant: id("web-1"),
            fingerprint: Fingerprint::from(0xabcd),
            revision: 1,
            table_json: "{}".into(),
            channels: vec![ChannelId(10), ChannelId(20)],
            replan_hz: 0.0,
            contracts: Vec::new(),
            grids: Vec::new(),
        }],
        contract: BTreeMap::new(),
        demands: Vec::new(),
        mode_reason: Reason::default(),
        ended: Vec::new(),
        last_stop: LastStop::default(),
    }
}

fn lease(channel: u32, holder: &str, token: u64, expires_ns: i64) -> Lease {
    Lease {
        channel: ChannelId(channel),
        holder: id(holder),
        priority: PriorityClass::Operator,
        token: LeaseToken(token),
        ttl_ns: 250 * MS as u64,
        granted_ns: 0,
        expires_ns,
        binding: Fingerprint::from(0xabcd),
        key_id: 0,
    }
}

fn sample_frame(sub: u32, sample: &Sample) -> Vec<u8> {
    let mut out = vec![frame::SAMPLE];
    out.extend_from_slice(&sub.to_le_bytes());
    out.push(0);
    for s in codec::encode_sample(sample).segments {
        out.extend_from_slice(&s);
    }
    out
}

fn joints(seq: u64, capture_ms: f64, value: f32) -> Sample {
    // pos [7] f32 at 0, vel [7] f32 at 32 (fields are 8-aligned), stride 64.
    let mut payload = vec![0_u8; 64];
    for i in 0..7 {
        payload[4 * i..4 * i + 4].copy_from_slice(&(value + i as f32).to_le_bytes());
        payload[32 + 4 * i..36 + 4 * i].copy_from_slice(&(value + 7.0 + i as f32).to_le_bytes());
    }
    Sample {
        channel: ChannelId(10),
        epoch: Epoch(1),
        seq,
        capture_ns: (capture_ms * 1e6) as i64,
        publish_ns: (capture_ms * 1e6) as i64 + 50_000,
        tick: 0,
        payload: Bytes::from(payload),
        applied: None,
    }
}

#[wasm_bindgen_test]
fn web_time_works_where_std_time_does_not() {
    let now = web_time_now_ms();
    let date = js_sys::Date::now();
    assert!((now - date).abs() < 1000.0, "{now} vs {date}");
}

#[wasm_bindgen_test]
#[should_panic]
fn std_instant_now_panics_on_wasm32_unknown_unknown() {
    std_instant_now_panics();
}

#[wasm_bindgen_test]
fn samples_decode_with_gaps_stats_and_copies_into_caller_buffers() {
    let mut s = Streams::new();
    assert!(s.add_stream("arm7/arm", 2.0).is_err(), "no snapshot yet");
    s.set_state(&codec::encode_state(&snapshot(1, Vec::new())))
        .unwrap();
    s.set_clock("web-1", 0.0, 0.1);
    let decl = s.add_stream("arm7/arm", 2.0).unwrap();
    assert_eq!(get(&decl, "kind").as_string().as_deref(), Some("f64"));
    assert_eq!(get(&decl, "class").as_string().as_deref(), Some("state"));
    assert_eq!(get(&decl, "index").as_f64(), Some(0.0));
    for n in 0..10_u64 {
        let seq = if n >= 5 { n + 3 } else { n + 1 }; // two lost
        let t = T0 + n as f64;
        assert_eq!(
            s.push(sample_frame(0, &joints(seq, t, 0.5)), t + 0.25)
                .unwrap(),
            1
        );
    }
    assert_eq!(s.seq(), 12.0);
    assert!((s.latency_ms() - 0.2).abs() < 1e-3);
    assert!((s.age_ms() - 0.25).abs() < 1e-3);
    let values = s.data_f64().unwrap();
    assert_eq!(values.length(), 14);
    assert_eq!(values.get_index(3), 3.5);
    let into = Float64Array::new_with_length(32);
    assert_eq!(s.data_f64_into(&into).unwrap(), 14);
    assert_eq!(into.get_index(13), 13.5);
    let st = s.stats(0);
    assert_eq!(get(&st, "received").as_f64(), Some(10.0));
    assert_eq!(get(&st, "lost").as_f64(), Some(2.0));
    assert_eq!(get(&st, "late_pct").as_f64(), Some(0.0));
    // A repeated seq is dropped, not delivered.
    assert_eq!(
        s.push(sample_frame(0, &joints(12, T0, 0.0)), T0).unwrap(),
        0
    );
    assert_eq!(get(&s.stats(0), "out_of_order").as_f64(), Some(1.0));
    // Freshness against max_age (50 ms).
    assert!(!s.is_stale(0, T0 + 9.0 + 40.0));
    assert!(s.is_stale(0, T0 + 9.0 + 60.0));
}

#[wasm_bindgen_test]
fn garbage_and_misfits_are_errors_or_counts_never_panics() {
    let mut s = Streams::new();
    assert!(s.set_state(&[0xff, 0xff, 0xff]).is_err());
    s.set_state(&codec::encode_state(&snapshot(1, Vec::new())))
        .unwrap();
    s.add_stream("arm7/arm", 2.0).unwrap();
    assert!(s.push(vec![0xff, 0xff, 0xff], T0).is_err());
    assert!(
        s.push(vec![frame::SAMPLE, 0, 0, 0, 0, 0, 0xff, 0xff], T0)
            .is_err()
    );
    assert!(s.push(vec![], T0).is_err());
    // An unknown subscription is ignored.
    assert_eq!(s.push(sample_frame(9, &joints(1, T0, 0.0)), T0).unwrap(), 0);
    // A payload that does not fit the layout is counted, not delivered.
    let mut short = joints(1, T0, 0.0);
    short.payload = Bytes::from_static(&[0; 8]);
    assert_eq!(s.push(sample_frame(0, &short), T0).unwrap(), 0);
    assert_eq!(get(&s.stats(0), "malformed").as_f64(), Some(1.0));
    assert!(s.data_f64().is_err());
    // The control side refuses garbage the same way.
    let mut c = ControlSession::new();
    assert!(c.on_frame(vec![], T0).is_err());
    assert!(c.on_frame(vec![frame::STATE, 0xff, 0xff], T0).is_err());
    assert!(c.on_frame(vec![0x7f, 1, 2], T0).is_err());
    assert!(c.set_participant("not an id").is_err());
}

#[wasm_bindgen_test]
fn frames_copy_into_pooled_buffers_and_applied_samples_carry_the_edge_mode() {
    let mut s = Streams::new();
    s.set_state(&codec::encode_state(&snapshot(1, Vec::new())))
        .unwrap();
    let decl = s.add_stream("arm7/scene/cam", 2.0).unwrap();
    assert_eq!(get(&decl, "class").as_string().as_deref(), Some("bulk"));
    assert_eq!(get(&decl, "kind").as_string().as_deref(), Some("jpeg"));
    let jpeg: Vec<u8> = (0..50_000_u32).map(|i| (i % 251) as u8).collect();
    let cam = Sample {
        channel: ChannelId(30),
        epoch: Epoch(1),
        seq: 1,
        capture_ns: 0,
        publish_ns: 0,
        tick: 0,
        payload: Bytes::from(jpeg),
        applied: None,
    };
    assert_eq!(s.push(sample_frame(0, &cam), T0).unwrap(), 1);
    assert_eq!((s.width(), s.height()), (640, 480));
    let small = Uint8Array::new_with_length(1000);
    assert!(s.data_into(&small).is_err());
    let pool = Uint8Array::new_with_length(64 * 1024);
    assert_eq!(s.data_into(&pool).unwrap(), 50_000);
    assert_eq!(pool.get_index(49_999), (49_999 % 251) as u8);

    s.add_stream("arm7/arm/_applied", 2.0).unwrap();
    let mut applied = joints(1, T0, 1.0);
    applied.channel = ChannelId(11);
    applied.payload = applied.payload.slice(..32);
    applied.applied = Some(Applied {
        mode: EdgeMode::Hold,
        present: true,
        lease_token: LeaseToken(5),
        chunk_seq: 3,
        action_index: 0,
        reason: ReasonCode::Unspecified,
    });
    assert_eq!(s.push(sample_frame(1, &applied), T0).unwrap(), 1);
    assert_eq!(s.applied_mode(), "hold");
    assert_eq!(s.applied_token(), 5.0);
    assert_eq!(s.data_f64().unwrap().length(), 7);
}

/// Attaches `c` as `web-1` with a snapshot and one completed clock exchange
/// (session = local + 2 ms, 0.1 ms each way).
fn attached() -> ControlSession {
    let mut c = ControlSession::new();
    assert!(c.heartbeat(T0).is_none(), "no heartbeat before the hello");
    c.set_participant("web-1").unwrap();
    let ev = c
        .on_frame(
            frame::tagged(frame::STATE, &codec::encode_state(&snapshot(1, Vec::new()))),
            T0,
        )
        .unwrap();
    assert_eq!(get(&ev, "mode").as_string().as_deref(), Some("active"));
    let hb = c.heartbeat(T0).unwrap();
    assert_eq!(hb[0], frame::HEARTBEAT);
    let hb = codec::decode_heartbeat(&hb[1..]).unwrap();
    assert_eq!(hb.participant, id("web-1"));
    let t1 = hb.t1_ns;
    let echo = HeartbeatEcho {
        seq: hb.seq,
        t1_ns: t1,
        t2_ns: t1 + 2 * MS + 100_000,
        t3_ns: t1 + 2 * MS + 110_000,
        epoch: Epoch(1),
    };
    let ev = c
        .on_frame(
            frame::tagged(frame::ECHO, &codec::encode_echo(&echo)),
            T0 + 0.21,
        )
        .unwrap();
    assert_eq!(get(&ev, "ours").as_bool(), Some(true));
    // A foreign echo (the gateway's own heartbeat) is ignored.
    let foreign = HeartbeatEcho { t1_ns: 7, ..echo };
    let ev = c
        .on_frame(
            frame::tagged(frame::ECHO, &codec::encode_echo(&foreign)),
            T0 + 0.3,
        )
        .unwrap();
    assert_eq!(get(&ev, "ours").as_bool(), Some(false));
    let clock = c.clock(T0 + 0.3);
    let offset = get(&clock, "offset_ms").as_f64().unwrap();
    assert!((offset - 2.0).abs() < 1e-3, "offset {offset}");
    // The next heartbeat carries the completed exchange.
    let next = codec::decode_heartbeat(&c.heartbeat(T0 + 50.0).unwrap()[1..]).unwrap();
    assert_eq!(next.previous.map(|p| p.seq), Some(hb.seq));
    c
}

/// Grants `path` to `web-1`; the reply's snapshot also shows `held`.
fn grant(
    c: &mut ControlSession,
    path: &str,
    channel: u32,
    token: u64,
    expires_ns: i64,
    revision: u64,
    held: &[Lease],
) {
    let req = c.request_lease(path, f64::NAN, T0 + 1.0).unwrap();
    assert_eq!(req[0], frame::REQUEST);
    let rid = frame::read_id(&req).unwrap();
    let request = codec::decode_request(&req[5..]).unwrap();
    assert_eq!(request.from, id("web-1"));
    assert!(matches!(
        request.command,
        Command::GrantLease { ref channels, binding, .. } if channels == &[ChannelId(channel)] && binding == Fingerprint::from(0xabcd)
    ));
    let granted = lease(channel, "web-1", token, expires_ns);
    let reply = CtlReply {
        request_id: 0,
        result: Ok(()),
        state: snapshot(
            revision,
            held.iter().cloned().chain([granted.clone()]).collect(),
        ),
        leases: vec![granted],
        keys: Vec::new(),
    };
    let ev = c
        .on_frame(
            frame::with_id(frame::REPLY, rid, &codec::encode_reply(&reply)),
            T0 + 1.5,
        )
        .unwrap();
    assert_eq!(
        get(&ev, "event").as_string().as_deref(),
        Some("grant"),
        "{ev:?}"
    );
    assert_eq!(get(&ev, "token").as_f64(), Some(token as f64));
    assert_eq!(
        get(&ev, "priority").as_string().as_deref(),
        Some("operator")
    );
}

#[wasm_bindgen_test]
fn clock_lease_chunks_and_fencing() {
    let mut c = attached();
    assert!(
        c.command("arm7/arm", &[0.1; 7], 10.0, T0 + 1.0).is_err(),
        "no lease"
    );
    assert!(
        c.request_lease("arm7/scene/cam", f64::NAN, T0).is_err(),
        "not commandable"
    );
    // Host expiry 1 s after T0 (session time is local + 2 ms).
    let expires = (T0 as i64) * MS + 1_000 * MS;
    grant(&mut c, "arm7/arm", 10, 42, expires, 2, &[]);
    assert_eq!(c.lease_state("arm7/arm", T0 + 2.0), "held");
    let chunk = c.command("arm7/arm", &[0.1; 7], 10.0, T0 + 2.0).unwrap();
    assert_eq!(chunk[0], frame::CHUNK);
    let chunk = codec::decode_chunk(Bytes::from(chunk).slice(1..)).unwrap();
    assert_eq!(
        (chunk.lease_token, chunk.count, chunk.source.as_str()),
        (LeaseToken(42), 2, "web-1")
    );
    assert_eq!(chunk.binding, Fingerprint::from(0xabcd));
    assert_eq!(
        chunk.actions.len(),
        64,
        "two 32-byte actions (7 f32, 8-aligned)"
    );
    assert_eq!(chunk.dt_ns, 10 * MS as u64);
    assert!((chunk.t0_ns - ((T0 + 2.0) as i64 * MS + 2 * MS)).abs() < 1_000);
    assert!(
        c.command("arm7/arm", &[0.1; 3], 10.0, T0 + 2.0).is_err(),
        "wrong value count"
    );
    // The deadline is the host expiry less offset and error: before 1 s.
    let remaining = c.lease_remaining_ms("arm7/arm", T0 + 2.0);
    assert!(remaining > 900.0 && remaining < 998.0, "{remaining}");
    // Past it: refused locally, and a later snapshot does not renew it.
    assert!(
        c.command("arm7/arm", &[0.1; 7], 10.0, T0 + 1_000.0)
            .is_err()
    );
    assert_eq!(c.lease_state("arm7/arm", T0 + 1_000.0), "expired");
    let late = snapshot(3, vec![lease(10, "web-1", 42, expires + 9_000 * MS)]);
    c.on_frame(
        frame::tagged(frame::STATE, &codec::encode_state(&late)),
        T0 + 1_001.0,
    )
    .unwrap();
    assert_eq!(c.lease_state("arm7/arm", T0 + 1_001.0), "expired");
    // The debug path crafts a stale-token chunk and reports the fence.
    let out: Array = c
        .command_with_token("arm7/arm", &[0.0; 7], 41.0, T0 + 1_001.0)
        .unwrap();
    let verdict = out.get(1).as_string().unwrap();
    assert!(
        verdict.contains("t41") && verdict.contains("fenced"),
        "{verdict}"
    );
    // Release names the token.
    let release = c.release("arm7/arm").unwrap().unwrap();
    let request = codec::decode_request(&release[5..]).unwrap();
    assert_eq!(
        request.command,
        Command::ReleaseLease {
            channels: vec![ChannelId(10)],
            token: LeaseToken(42)
        }
    );
    assert_eq!(c.lease_state("arm7/arm", T0 + 1_002.0), "idle");
}

#[wasm_bindgen_test]
fn leases_on_two_parts_are_independent_and_preemption_and_forget_end_them() {
    let mut c = attached();
    let expires = (T0 as i64) * MS + 10_000 * MS;
    grant(&mut c, "arm7/arm", 10, 7, expires, 2, &[]);
    grant(
        &mut c,
        "arm7/gripper",
        20,
        8,
        expires,
        3,
        &[lease(10, "web-1", 7, expires)],
    );
    assert_eq!(c.lease_state("arm7/arm", T0 + 2.0), "held");
    assert_eq!(c.lease_paths().len(), 2);
    let g = c.command("arm7/gripper", &[0.5], 20.0, T0 + 2.0).unwrap();
    let g = codec::decode_chunk(Bytes::from(g).slice(1..)).unwrap();
    assert_eq!((g.channel, g.lease_token), (ChannelId(20), LeaseToken(8)));
    // A safety supervisor takes the arm; the gripper stays ours.
    let mut supervisor = lease(10, "supervisor", 9, expires);
    supervisor.priority = PriorityClass::Safety;
    let next = snapshot(4, vec![supervisor, lease(20, "web-1", 8, expires)]);
    let ev = c
        .on_frame(
            frame::tagged(frame::STATE, &codec::encode_state(&next)),
            T0 + 3.0,
        )
        .unwrap();
    let changes: Array = get(&ev, "changes").into();
    let revoked = (0..changes.length())
        .map(|i| changes.get(i))
        .find(|ch| get(ch, "change").as_string().as_deref() == Some("revoked"))
        .expect("a revocation");
    assert_eq!(
        get(&revoked, "path").as_string().as_deref(),
        Some("arm7/arm")
    );
    assert!(
        get(&revoked, "reason")
            .as_string()
            .unwrap()
            .contains("preempted by supervisor")
    );
    assert_eq!(c.lease_state("arm7/arm", T0 + 3.0), "revoked");
    assert_eq!(c.lease_state("arm7/gripper", T0 + 3.0), "held");
    let status = c.status("arm7/arm").unwrap();
    assert_eq!(
        get(&status, "holder").as_string().as_deref(),
        Some("supervisor")
    );
    assert_eq!(
        get(&status, "priority").as_string().as_deref(),
        Some("safety")
    );
    // A dropped connection: forget, and the token is never used again.
    assert_eq!(c.forget("arm7/gripper"), 8.0);
    assert!(c.command("arm7/gripper", &[0.0], 20.0, T0 + 4.0).is_err());
    assert_eq!(c.lease_state("arm7/gripper", T0 + 4.0), "idle");
}

#[wasm_bindgen_test]
fn refusals_and_verbs_answer_with_the_host_state() {
    let mut c = attached();
    let req = c.request_lease("arm7/arm", 500.0, T0 + 1.0).unwrap();
    let rid = frame::read_id(&req).unwrap();
    assert!(matches!(
        codec::decode_request(&req[5..]).unwrap().command,
        Command::GrantLease {
            ttl_ns: Some(500_000_000),
            ..
        }
    ));
    let reply = CtlReply {
        request_id: 0,
        result: Err(CtlError::new(CtlStatus::LeaseBusy, "held by safety")),
        state: snapshot(2, Vec::new()),
        leases: Vec::new(),
        keys: Vec::new(),
    };
    let ev = c
        .on_frame(
            frame::with_id(frame::REPLY, rid, &codec::encode_reply(&reply)),
            T0 + 2.0,
        )
        .unwrap();
    assert_eq!(get(&ev, "event").as_string().as_deref(), Some("refused"));
    assert!(
        get(&ev, "reason")
            .as_string()
            .unwrap()
            .contains("held by safety")
    );
    assert_eq!(c.lease_state("arm7/arm", T0 + 2.0), "refused");
    let arm = c.verb("arm", "").unwrap();
    assert_eq!(
        codec::decode_request(&arm[5..]).unwrap().command,
        Command::Arm
    );
    assert!(c.verb("launch", "").is_err());
    let decls = c.declarations();
    let controls: Array = get(&decls, "controls").into();
    assert_eq!(controls.length(), 2);
    assert_eq!(get(&controls.get(0), "bound").as_bool(), Some(true));
    let streams: Array = get(&decls, "streams").into();
    assert_eq!(
        streams.length(),
        5,
        "arm, its applied stream, gripper, its applied stream, cam"
    );
}
