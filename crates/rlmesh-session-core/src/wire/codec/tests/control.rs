//! Control messages, device views and bindings.

use super::*;
use crate::wire::codec::convert::{clock_mode_from_pb, mode_from_pb};
use crate::wire::codec::ctl::status_from_pb;
use crate::wire::codec::part::channel_priority_from_pb;
use crate::wire::codec::state::{state_from_pb, state_to_pb};
use crate::wire::codec::stream::edge_mode_from_pb;

#[test]
fn a_device_view_decodes_on_its_own_path() {
    let mut g = Goldens::new("device_view");
    let state = snapshot();
    let view = DeviceView {
        robot: pid("ur10e"),
        epoch: state.epoch,
        revision: state.revision,
        mode: state.mode,
        mode_reason: ReasonCode::FirstLease,
        last_stop: state.last_stop,
        host_ns: state.host_ns,
        clock_mode: ClockMode::Wall,
        tick_ns: 0,
        generation: 3,
        parts: state.parts.clone(),
        leases: state.leases.clone(),
        clocks: state.clocks.clone(),
        grids: vec![ViewGrid {
            binding: Fingerprint::from(77),
            channel: ChannelId(1),
            dt: Duration::from_millis(10),
        }],
        demands: state.demands.clone(),
        members: vec![Member {
            id: pid("op"),
            generation: 3,
        }],
        closing: Some(Reason::new(ReasonCode::SceneEnded)),
    };
    let bytes = encode_view(&view);
    g.add("view", &bytes);
    assert_eq!(decode_view(&bytes).expect("decode"), view);
    // The view is its own message, not a state a board could misread.
    assert!(decode_state(&bytes).is_err());

    // A ctl reply to a robot that takes its view carries it in field 9
    // and no state; the view path reads any reply, skipping `state`.
    let reply = ViewReply {
        result: Err(CtlError::new(CtlStatus::LeaseBusy, "held")),
        view: Some(view),
        leases: state.leases.clone(),
        request_id: 13,
        keys: Vec::new(),
        assignment: None,
        run: None,
    };
    let bytes = encode_view_reply(&reply);
    g.add("view_reply", &bytes);
    assert_eq!(decode_view_reply(&bytes).expect("decode"), reply);
    assert!(decode_reply(&bytes).is_err(), "no state in it");
    let full = encode_reply(&CtlReply {
        result: Ok(()),
        state: state.clone(),
        leases: Vec::new(),
        request_id: 14,
        keys: Vec::new(),
        assignment: None,
        run: None,
    });
    let read = decode_view_reply(&full).expect("decode");
    assert_eq!((read.request_id, read.view), (14, None));
    g.check();
}

#[test]
fn binding_records_round_trip() {
    let mut g = Goldens::new("binding");
    let state = snapshot();
    let binding = &state.bindings[0];
    let decoded = decode_binding(g.add("binding", &encode_binding(binding))).expect("decodes");
    assert_eq!(&decoded, binding);
    g.check();
}

#[test]
fn control_messages_round_trip() {
    let mut g = Goldens::new("control");
    let requests = [
        Command::Attach {
            name: "Operator".into(),
            roles: vec![Role::Controller, Role::Observer, Role::Embodiment],
            priority: PriorityClass::Operator,
            protocol_minor: 3,
            supports: vec!["frame.fixed".into()],
            requires: vec!["merge.blend".into()],
            workflow_edition: "2026.06".into(),
            workflow_editions: vec!["2026.06".into()],
            estop: None,
        },
        Command::DeclareParts {
            spec_version: 1,
            robot_json: r#"{"arm": {"kind": "x/arm"}}"#.into(),
            seal: Vec::new(),
            board_id: Some([0xb0, 0xb1, 0, 0xb3, 0xb4, 0, 0, 0xb7]),
        },
        Command::DeclareBinding {
            fingerprint: Fingerprint::from(9),
            revision: 4,
            table_json: "{}".into(),
            channels: vec![ChannelId(1), ChannelId(5)],
            replan_hz: 20.0,
            contracts: vec![StreamContract {
                channel: ChannelId(2),
                mode: ContractMode::Latest,
                delivery_hz: 20.0,
                history: 1,
                max_age: None,
                max_age_explicit: false,
            }],
            grids: vec![ActionGrid {
                channel: ChannelId(5),
                dt: Duration::from_nanos(66_666_667),
            }],
        },
        Command::DemandAck {
            channel: ChannelId(3),
            revision: 2,
        },
        Command::Arm,
        Command::Disarm,
        Command::Detach,
        Command::Clear,
        Command::GrantLease {
            channels: vec![ChannelId(4), ChannelId(6)],
            release: vec![ChannelId(5)],
            binding: Fingerprint::from(9),
            ttl_ns: Some(5),
        },
        Command::GrantLease {
            channels: vec![ChannelId(4)],
            release: Vec::new(),
            binding: Fingerprint::from(9),
            ttl_ns: None,
        },
        Command::ReleaseLease {
            channels: vec![ChannelId(4)],
            token: LeaseToken(2),
        },
        Command::Estop {
            reason: "red button".into(),
            code: ReasonCode::EstopLocal,
            origin: None,
            at_ns: 1_234,
        },
        Command::EnterSafe {
            reason: "queue dry".into(),
            code: ReasonCode::DeviceWatchdog,
        },
        Command::Ready {
            tick: 7,
            watermarks: vec![Watermark {
                channel: ChannelId(2),
                seq: 8,
            }],
        },
        Command::Decided {
            tick: 7,
            chunks: vec![ChunkRef {
                channel: ChannelId(1),
                lease_token: LeaseToken(3),
                seq: 9,
            }],
        },
        Command::LobbyWait(LobbyWait {
            run: "r-7k2m".into(),
            last: Some(CtlError::new(CtlStatus::NotAuthorized, "board on rig-pi")),
        }),
        Command::RunScene {
            scene_toml: "[scene]\nformat = 1\n".into(),
            manual_arm: true,
        },
        Command::SceneStatus {
            run: "r-7k2m".into(),
            after: 4,
        },
        Command::StopScene {
            run: "r-7k2m".into(),
        },
        Command::ArmScene {
            run: "r-7k2m".into(),
        },
        Command::HostStarted(HostStarted {
            run: "r-7k2m".into(),
            result: Ok(([5; 32], vec!["10.0.0.2:7447".into()])),
        }),
        Command::HostStarted(HostStarted {
            run: "r-7k2m".into(),
            result: Err(CtlError::new(CtlStatus::NotPresent, "no GPU")),
        }),
        Command::Keep {
            label: "eye-2 froze".into(),
        },
    ];
    for command in requests {
        let request = Request {
            from: pid("op"),
            epoch: Some(Epoch(2)),
            request_id: 11,
            command,
        };
        assert_eq!(
            decode_request(g.add("request", &encode_request(&request))).expect("decode"),
            request
        );
    }

    let state = snapshot();
    assert_eq!(
        decode_state(g.add("state", &encode_state(&state))).expect("decode"),
        state
    );
    for result in [
        Ok(()),
        Err(CtlError::new(CtlStatus::GroupBusy, "busy")),
        Err(CtlError::new(CtlStatus::NotBound, "not in the binding")),
        Err(CtlError::new(CtlStatus::UnknownPath, "no such part")),
        Err(CtlError::new(
            CtlStatus::NotReady,
            "wrist/cam has not installed its plan",
        )),
        Err(CtlError::new(CtlStatus::Unauthenticated, "unknown key")),
        Err(CtlError::new(CtlStatus::NotAuthorized, "bound to eye-3")),
        Err(CtlError::new(CtlStatus::Expired, "below the window")),
        Err(CtlError::new(CtlStatus::RobotBusy, "arm is in run r-7k2m")),
        Err(CtlError::new(
            CtlStatus::NotPresent,
            "eye-l is not in the lobby",
        )),
    ] {
        let reply = CtlReply {
            result,
            leases: state.leases.clone(),
            state: state.clone(),
            request_id: 11,
            keys: Vec::new(),
            assignment: None,
            run: None,
        };
        assert_eq!(
            decode_reply(g.add("reply", &encode_reply(&reply))).expect("decode"),
            reply
        );
    }
    let session = SessionId::named("wire-goldens");
    let assignments = [
        Assignment::Join(Join {
            run: "r-7k2m".into(),
            session,
            robot: pid("eye-l"),
            addresses: vec!["10.0.0.2:7447".into(), "rig-pi.local:7447".into()],
            host_key: [6; 32],
            host_id: "rlm-3f9a2c".into(),
            tenant: "local".into(),
        }),
        Assignment::Host(HostOrder {
            run: "r-7k2m".into(),
            session,
            scene_toml: "[scene]".into(),
            devices: vec![DeviceRecord {
                id: "laptop-josh".into(),
                public_key: [8; 32],
                max_priority: Some(PriorityClass::Model),
                parts: vec!["**".into()],
                verbs: vec!["arm".into()],
            }],
            pins: vec![
                RobotPin {
                    robot: pid("skull"),
                    device: "local".into(),
                    board_id: Some([1, 2, 3, 4, 5, 6, 7, 8]),
                },
                RobotPin {
                    robot: pid("eye-l"),
                    device: "eye-l".into(),
                    board_id: None,
                },
            ],
        }),
        Assignment::Stop(StopOrder {
            run: "r-7k2m".into(),
            reason: Reason {
                code: ReasonCode::SceneFailed,
                detail: "home daemon lost".into(),
                ..Reason::default()
            },
        }),
    ];
    let run = RunState {
        run: "r-7k2m".into(),
        version: 7,
        stage: RunStage::Rejoining,
        members: vec![MemberState {
            name: pid("eye-l"),
            stage: MemberStage::Lost,
            error: Some(CtlError::new(CtlStatus::Incompatible, "a changed map")),
        }],
        error: None,
        session: Some(session),
        addresses: vec!["10.0.0.3:7447".into()],
        host_key: Some([9; 32]),
        scene_fingerprint: Some(Fingerprint::from(0x5ce5)),
    };
    let replies = assignments.into_iter().map(|a| (Some(a), None)).chain([
        (None, Some(run)),
        (
            None,
            Some(RunState {
                run: "r-7k2m".into(),
                version: 1,
                stage: RunStage::Checking,
                members: Vec::new(),
                error: None,
                session: None,
                addresses: Vec::new(),
                host_key: None,
                scene_fingerprint: None,
            }),
        ),
    ]);
    for (assignment, run) in replies {
        let reply = CtlReply {
            result: Ok(()),
            state: state.clone(),
            leases: Vec::new(),
            request_id: 12,
            keys: Vec::new(),
            assignment,
            run,
        };
        assert_eq!(
            decode_reply(g.add("reply", &encode_reply(&reply))).expect("decode"),
            reply
        );
    }

    let hello = HandshakeRequest {
        participant: pid("op"),
        noise: vec![1, 2, 3],
    };
    assert_eq!(
        decode_handshake_request(g.add("handshake_request", &encode_handshake_request(&hello)))
            .expect("decode"),
        hello
    );
    for reply in [
        Ok(vec![4, 5]),
        Err(CtlError::new(CtlStatus::Unauthenticated, "unknown key")),
    ] {
        assert_eq!(
            decode_handshake_reply(g.add("handshake_reply", &encode_handshake_reply(&reply)))
                .expect("decode"),
            reply
        );
    }
    let accept = HandshakeAccept {
        generation: 2,
        device: "local".into(),
        max_priority: Some(PriorityClass::Safety),
        host_sign_key: [7; 32],
        host_instance: 0xfeed,
        up_key_id: 3,
        down_key_id: 4,
    };
    assert_eq!(
        decode_handshake_accept(g.add("handshake_accept", &encode_handshake_accept(&accept)))
            .expect("decode"),
        accept
    );

    let hb = Heartbeat {
        participant: pid("op"),
        epoch: Epoch(3),
        mode: Mode::Armed,
        seq: 8,
        t1_ns: -5,
        previous: Some(TimingSample {
            seq: 7,
            t1_ns: 1,
            t2_ns: 2,
            t3_ns: 3,
            t4_ns: 4,
        }),
        last_seq_seen: vec![(ChannelId(2), 77)],
        last_chunk_accepted: 3,
        last_chunk_applied: 2,
        queue_depth: 40,
        health: Health::Degraded,
        rejected: vec![(Rejection::StaleToken, 3), (Rejection::TooOld, 1)],
    };
    assert_eq!(
        decode_heartbeat(g.add("heartbeat", &encode_heartbeat(&hb))).expect("decode"),
        hb
    );
    let echo = HeartbeatEcho {
        seq: 8,
        t1_ns: -5,
        t2_ns: 10,
        t3_ns: 11,
        epoch: Epoch(3),
    };
    assert_eq!(
        decode_echo(g.add("echo", &encode_echo(&echo))).expect("decode"),
        echo
    );
    for run in [
        None,
        Some(RunCounters {
            starts: 120,
            completions: 118,
            busy_skips: 4,
            waiting_for: 2,
            stale_input: 1,
            window_gap: 1,
            decode: 0,
            no_lease: 9,
            chunks_sent: 118,
            pending_ns: 2_000_000_000,
            latency_p50_ns: 8_000_000,
            latency_p99_ns: 31_000_000,
            drifting: true,
        }),
    ] {
        let report = ParticipantReport {
            participant: pid("policy"),
            epoch: Epoch(3),
            sent_ns: -9,
            streams: vec![PublisherCounters {
                channel: ChannelId(4),
                published: 600,
                coalesced: 300,
            }],
            run,
        };
        assert_eq!(
            decode_report(g.add("report", &encode_report(&report))).expect("decode"),
            report
        );
    }
    let report = ParticipantReport {
        participant: pid("policy"),
        epoch: Epoch(3),
        sent_ns: -9,
        streams: vec![PublisherCounters {
            channel: ChannelId(4),
            published: 600,
            coalesced: 300,
        }],
        run: None,
    };
    for health in [
        HostHealth::default(),
        HostHealth {
            epoch: Epoch(3),
            host_ns: 1_000,
            links: vec![
                LinkHealth {
                    participant: Some(pid("eye-1")),
                    heartbeats: 40,
                    rtt_ns: 2_000_000,
                    jitter_ns: 300_000,
                    heartbeat_gap_max_ns: 120_000_000,
                    last_heartbeat_age_ns: 50_000_000,
                    offset: Some((-7, 900)),
                    last_chunk_accepted: 12,
                    last_chunk_applied: 11,
                    queue_depth: 3,
                    health: Health::Degraded,
                    rejected: vec![(Rejection::Unauthenticated, 2), (Rejection::NoLease, 5)],
                    report: None,
                    last_seq_seen: vec![(ChannelId(2), 77)],
                },
                LinkHealth {
                    participant: Some(pid("policy")),
                    report: Some(report.clone()),
                    ..LinkHealth::default()
                },
            ],
        },
    ] {
        assert_eq!(
            decode_health(g.add("health", &encode_health(&health))).expect("decode"),
            health
        );
    }
    assert!(
        decode_report(b"\x0a\x01!").is_err(),
        "an invalid participant id"
    );
    assert!(decode_state(b"\xff\xff\xff").is_err());

    for tick in [
        VirtualTick::Observe(TickState {
            epoch: Epoch(2),
            tick: 40,
            watermarks: vec![Watermark {
                channel: ChannelId(1),
                seq: 41,
            }],
            due: vec![pid("policy")],
        }),
        VirtualTick::Step(Step {
            epoch: Epoch(2),
            tick: 40,
            chunks: vec![ChunkRef {
                channel: ChannelId(1),
                lease_token: LeaseToken(3),
                seq: 2,
            }],
        }),
    ] {
        assert_eq!(
            decode_tick(g.add("tick", &encode_tick(&tick))).expect("decode"),
            tick
        );
    }
    let query = HistoryQuery {
        epoch: Epoch(2),
        after_seq: 10,
        until_seq: Some(20),
        limit: None,
        request_id: 0,
    };
    assert_eq!(
        decode_history_query(g.add("history_query", &encode_history_query(&query)))
            .expect("decode"),
        query
    );
    g.check();
}

#[test]
fn a_bad_fingerprint_or_part_is_malformed() {
    let mut state = state_to_pb(&snapshot());
    state.leases[0].binding = Bytes::from_static(&[1, 2, 3]);
    assert!(state_from_pb(state).is_err());
    let mut state = state_to_pb(&snapshot());
    state.parts[0].path = "Wrist/Cam".into();
    assert!(state_from_pb(state).is_err(), "an invalid path");
    let mut state = state_to_pb(&snapshot());
    state.parts[1].fields[0].dtype = 13;
    assert!(
        state_from_pb(state).is_err(),
        "bfloat16 is not a v0.2 dtype"
    );
    let mut state = state_to_pb(&snapshot());
    state.parts[0].quality = 0;
    assert!(state_from_pb(state).is_err());
}

#[test]
fn unknown_enum_values_are_refused_but_unknown_statuses_are_refusals() {
    assert!(mode_from_pb(0).is_err());
    assert!(mode_from_pb(99).is_err());
    assert!(edge_mode_from_pb(0).is_err());
    assert!(channel_priority_from_pb(0).is_err());
    assert_eq!(clock_mode_from_pb(0).ok(), Some(ClockMode::Wall));
    assert_eq!(
        status_from_pb(pb::CtlStatus::LeaseBusy as i32, "busy".into()).map_err(|e| e.status),
        Err(CtlStatus::LeaseBusy)
    );
    assert_eq!(
        status_from_pb(999, String::new()).map_err(|e| e.status),
        Err(CtlStatus::InvalidRequest)
    );
    assert!(status_from_pb(pb::CtlStatus::Ok as i32, String::new()).is_ok());
}
