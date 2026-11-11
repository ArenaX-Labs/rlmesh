//! Control messages, device views and bindings.

use super::*;
use crate::wire::codec::convert::{clock_mode_from_pb, mode_from_pb};
use crate::wire::codec::ctl::status_from_pb;
use crate::wire::codec::part::channel_priority_from_pb;
use crate::wire::codec::state::{state_from_pb, state_to_pb};
use crate::wire::codec::stream::edge_mode_from_pb;
use crate::wire::error::{Unread, WireError};

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
        datagram: None,
    };
    let bytes = encode_view(&view);
    g.add("view", &bytes);
    assert_eq!(decode_view(&bytes).expect("decode"), view);
    // A robot on the datagram path: its routes ride the view as field 18.
    let routed = DeviceView {
        datagram: Some(DatagramRoutes {
            probe: 9,
            routes: vec![
                DatagramRoute {
                    route: 10,
                    channel: ChannelId(1),
                    up: false,
                },
                DatagramRoute {
                    route: 0xffff,
                    channel: ChannelId(2),
                    up: true,
                },
            ],
        }),
        ..view.clone()
    };
    let routed_bytes = encode_view(&routed);
    g.add("view_datagram", &routed_bytes);
    assert_eq!(decode_view(&routed_bytes).expect("decode"), routed);
    assert_eq!(
        routed_bytes[..bytes.len()],
        bytes[..],
        "the routes only follow the view"
    );
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
    // A binding that declares how many actions its chunks carry.
    let mut chunked = binding.clone();
    chunked.grids[0].chunk = 4;
    let decoded =
        decode_binding(g.add("binding_chunk", &encode_binding(&chunked))).expect("decodes");
    assert_eq!(decoded, chunked);
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
                chunk: 0,
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
        Command::Rearm {
            robot: pid("eye-1"),
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
        datagram: None,
    };
    assert_eq!(
        decode_handshake_accept(g.add("handshake_accept", &encode_handshake_accept(&accept)))
            .expect("decode"),
        accept
    );
    let offering = HandshakeAccept {
        datagram: Some(DatagramOffer {
            port: 7448,
            version: crate::datagram::VERSION,
        }),
        ..accept.clone()
    };
    assert_eq!(
        decode_handshake_accept(g.add(
            "handshake_accept_datagram",
            &encode_handshake_accept(&offering)
        ))
        .expect("decode"),
        offering
    );
    // An offer this side can't read is no offer.
    let mut unreadable =
        pb::HandshakeAccept::decode(&encode_handshake_accept(&offering)[..]).expect("decode");
    unreadable.datagram = Some(pb::DatagramOffer {
        port: 70_000,
        version: 1,
    });
    assert_eq!(
        decode_handshake_accept(&unreadable.encode_to_vec())
            .expect("decode")
            .datagram,
        None
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
        clock_refused: vec![ClockRefusal {
            channel: ChannelId(2),
            participant: pid("ur10e"),
            error_ns: 19_500_000,
            bound_ns: 5_000_000,
        }],
        chunk_timing: Vec::new(),
        chunks_accepted: 0,
        datagram: None,
    };
    assert_eq!(
        decode_heartbeat(g.add("heartbeat", &encode_heartbeat(&hb))).expect("decode"),
        hb
    );
    let timed = Heartbeat {
        chunk_timing: vec![timing()],
        chunks_accepted: 912,
        datagram: Some(datagram_stats()),
        ..hb.clone()
    };
    assert_eq!(
        decode_heartbeat(g.add("heartbeat_timing", &encode_heartbeat(&timed))).expect("decode"),
        timed
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
                    clock_refused: vec![ClockRefusal {
                        channel: ChannelId(2),
                        participant: pid("op"),
                        error_ns: 6_000_000,
                        bound_ns: 5_000_000,
                    }],
                    report: None,
                    last_seq_seen: vec![(ChannelId(2), 77)],
                    ..LinkHealth::default()
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
    let routed = HostHealth {
        epoch: Epoch(3),
        host_ns: 1_000,
        links: vec![LinkHealth {
            participant: Some(pid("eye-1")),
            heartbeats: 40,
            chunk_timing: vec![timing()],
            chunks_accepted: 912,
            data_path: DataPath::Datagram,
            datagram_board: Some(datagram_stats()),
            datagram_host: Some(DatagramStats {
                active: true,
                sent: 950,
                received: 3_000,
                ..DatagramStats::default()
            }),
            ..LinkHealth::default()
        }],
    };
    assert_eq!(
        decode_health(g.add("health_datagram", &encode_health(&routed))).expect("decode"),
        routed
    );
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

/// The view as prost and the conversions decoded it before
/// [`decode_view_into`]: what that must agree with.
fn prost_view(buf: &[u8]) -> Result<DeviceView, WireError> {
    use crate::wire::codec::convert::{collect, fingerprint_from_pb};
    use crate::wire::codec::part::part_from_pb;
    use crate::wire::codec::state::{
        clock_from_pb, demand_from_pb, last_stop_from_pb, lease_from_pb, reason_code_from_pb,
        reason_from_pb,
    };
    let v = pb::DeviceView::decode(buf)?;
    Ok(DeviceView {
        robot: crate::wire::ParticipantId::new(v.robot)?,
        epoch: Epoch(v.epoch),
        revision: v.revision,
        mode: mode_from_pb(v.mode)?,
        mode_reason: reason_code_from_pb(v.mode_reason),
        last_stop: last_stop_from_pb(v.last_stop),
        host_ns: v.host_ns,
        clock_mode: clock_mode_from_pb(v.clock_mode)?,
        tick_ns: v.tick_ns,
        generation: v.generation,
        parts: collect(v.parts, part_from_pb)?,
        leases: collect(v.leases, lease_from_pb)?,
        clocks: collect(v.clocks, clock_from_pb)?,
        grids: collect(v.grids, |g| {
            Ok::<_, WireError>(ViewGrid {
                binding: fingerprint_from_pb("view grid", &g.binding)?,
                channel: ChannelId(g.channel),
                dt: Duration::from_nanos(g.dt_ns),
            })
        })?,
        demands: collect(v.demands, demand_from_pb)?,
        members: collect(v.members, |m| {
            Ok::<_, WireError>(Member {
                id: crate::wire::ParticipantId::new(m.id)?,
                generation: m.generation,
            })
        })?,
        closing: v.closing.map(reason_from_pb),
        datagram: v.datagram.map(|d| DatagramRoutes {
            probe: u16::try_from(d.probe).expect("a route id"),
            routes: d
                .routes
                .into_iter()
                .map(|r| DatagramRoute {
                    route: u16::try_from(r.route).expect("a route id"),
                    channel: ChannelId(r.channel),
                    up: r.up,
                })
                .collect(),
        }),
    })
}

/// eye-1's view in the demo's session (two boards, a camera, a tracker, a
/// desk, a driver and a viewer), as the host rendered it with its part.
const DEMO_VIEW: &str = "0a056579652d311002181e20032803320208013887d4d99941400150015ab8010a046579657312066a6f696e7473180120022a0f0a03706f73100b1a010222037261643100000000000049403a0565766572794a0370616e4a0474696c7452037261776a5f0a08706f736974696f6e120f0a03706f73100b1a010222037261641a290a03706f731210333333333333e3bf9a9999999999d9bf1a10333333333333e33f9a9999999999d93f2204686f6c642880c2d72f308084af5f380158016880e1eb1778028201056579652d3199010000000000004940622b08011206647269766572180220022880e59a7730cbbbd1d20538a79b838a4242082e9bfc4af9a6ddc048196a250a0664726976657210e6c2c79f041884a30d210000000000407f4028a4c3133094d1ffe2406a230a056579652d3110af9fb90718c6ea0c210000000000407f40289bf71230aab29fe54072110a082e9bfc4af9a6ddc010011880dac4097a1108011001180339000000000000244048018201070a0363616d10038201080a046465736b100582010a0a0664726976657210068201090a056579652d3110018201090a056579652d32100282010b0a07747261636b6572100482010a0a067669657765721007";

/// [`DEMO_VIEW`]'s bytes.
fn demo_view() -> Vec<u8> {
    (0..DEMO_VIEW.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&DEMO_VIEW[i..i + 2], 16).expect("hex"))
        .collect()
}

/// A view as a board in a full session holds it.
fn full_view() -> DeviceView {
    let state = snapshot();
    DeviceView {
        robot: pid("ur10e"),
        epoch: state.epoch,
        revision: state.revision,
        mode: Mode::Active,
        mode_reason: ReasonCode::FirstLease,
        last_stop: LastStop {
            epoch: Epoch(2),
            revision: 4,
        },
        host_ns: -7,
        clock_mode: ClockMode::Virtual,
        tick_ns: 10,
        generation: 3,
        parts: vec![arm()],
        leases: state.leases.clone(),
        clocks: state.clocks.clone(),
        grids: vec![ViewGrid {
            binding: Fingerprint::from(77),
            channel: ChannelId(1),
            dt: Duration::from_millis(10),
        }],
        demands: vec![
            Demand {
                channel: ChannelId(1),
                revision: 2,
                plan: Plan::Latest { hz: 12.5 },
                retain: Duration::from_millis(3),
                acked: 1,
            },
            Demand {
                channel: ChannelId(2),
                revision: 1,
                plan: Plan::Every,
                retain: Duration::ZERO,
                acked: 1,
            },
        ],
        members: ["op", "policy", "cam", "ur10e"]
            .iter()
            .zip(1..)
            .map(|(id, generation)| Member {
                id: pid(id),
                generation,
            })
            .collect(),
        closing: Some(Reason {
            code: ReasonCode::SceneEnded,
            by: "op".into(),
            detail: "the run ended".into(),
            at_ns: 5,
            revision: 9,
            verified: true,
        }),
        datagram: Some(DatagramRoutes {
            probe: 3,
            routes: vec![
                DatagramRoute {
                    route: 4,
                    channel: ChannelId(1),
                    up: false,
                },
                DatagramRoute {
                    route: 5,
                    channel: ChannelId(1),
                    up: true,
                },
            ],
        }),
    }
}

#[test]
fn a_view_decoded_in_place_is_the_one_prost_decodes_whatever_the_bytes() {
    let view = full_view();
    let bytes = encode_view(&view);
    let mut held = DeviceView::empty(pid("x"));
    decode_view_into(&bytes, &mut held, true).expect("decodes");
    assert_eq!(held, view);
    assert_eq!(prost_view(&bytes).expect("decodes"), view);
    // Without its parts, the rest is the same.
    decode_view_into(&bytes, &mut held, false).expect("decodes");
    assert!(held.parts.is_empty());
    assert_eq!(
        DeviceView {
            parts: vec![arm()],
            ..held.clone()
        },
        view
    );
    // Whatever a byte becomes, a view decoded in place is never one prost
    // reads otherwise.
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut taken = 0;
    for _ in 0..20_000 {
        let mut bytes = bytes.clone();
        for _ in 0..1 + next() % 3 {
            let at = usize::try_from(next() % bytes.len() as u64).expect("at");
            match next() % 3 {
                0 => bytes[at] = u8::try_from(next() & 0xff).expect("byte"),
                1 => bytes.truncate(at),
                _ => bytes.insert(at, u8::try_from(next() & 0xff).expect("byte")),
            }
            if bytes.is_empty() {
                break;
            }
        }
        if decode_view_into(&bytes, &mut held, true).is_ok() {
            taken += 1;
            let reference = prost_view(&bytes).expect("prost reads what was taken in place");
            assert_eq!(held, reference);
        }
    }
    assert!(taken > 100, "{taken} mutated views decoded");
}

#[test]
fn a_hostile_or_oversized_view_is_refused_unread() {
    let refused = |bytes: &[u8]| {
        let mut held = DeviceView::empty(pid("x"));
        decode_view_into(bytes, &mut held, true)
            .err()
            .and_then(|e| e.why_unread())
    };
    let robot = |mut tail: Vec<u8>| {
        let mut bytes = vec![0x0a, 0x06];
        bytes.extend(b"ur10e1");
        bytes.append(&mut tail);
        bytes
    };
    // Two thousand empty members: 4 KB on the wire, refused before one is
    // made.
    assert_eq!(
        refused(&robot([0x82, 0x01, 0x00].repeat(2000))),
        Some(Unread::TooLarge("members"))
    );
    // Thirty-three parts.
    assert_eq!(
        refused(&robot([0x5a, 0x00].repeat(33))),
        Some(Unread::TooLarge("parts"))
    );
    // A length prefix of four gigabytes.
    assert_eq!(
        refused(&robot(vec![0x82, 0x01, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x0a])),
        Some(Unread::Malformed("length past the end"))
    );
    // An id longer than any, and one that isn't one.
    let mut long = vec![0x82, 0x01, 0x84, 0x01, 0x0a, 0x81, 0x01];
    long.extend([b'a'; 129]);
    assert_eq!(
        refused(&robot(long)),
        Some(Unread::Malformed("participant id"))
    );
    assert_eq!(
        refused(&robot(vec![0x82, 0x01, 0x03, 0x0a, 0x01, b'/'])),
        Some(Unread::Malformed("participant id"))
    );
    // A closing reason past what a board keeps.
    let mut view = full_view();
    view.closing.as_mut().expect("closing").detail = "x".repeat(MAX_VIEW_TEXT + 1);
    assert_eq!(
        refused(&encode_view(&view)),
        Some(Unread::TooLarge("reason"))
    );
    // No robot, an unknown mode: malformed.
    assert_eq!(refused(&[0x10, 0x01]), Some(Unread::Malformed("robot")));
    assert_eq!(
        refused(&robot(vec![0x20, 0x63])),
        Some(Unread::Malformed("mode"))
    );
}

#[test]
fn a_view_decoded_into_the_one_held_keeps_its_buffers() {
    let view = full_view();
    let mut next = view.clone();
    next.host_ns += 1;
    next.leases[0].expires_ns += 10;
    next.clocks[0].error_ns += 1;
    let (one, two) = (encode_view(&view), encode_view(&next));
    let mut held = DeviceView::empty(pid("x"));
    decode_view_into(&one, &mut held, false).expect("decodes");
    let buffers = |v: &DeviceView| {
        (
            v.robot.as_str().as_ptr(),
            v.leases.as_ptr(),
            v.leases[0].holder.as_str().as_ptr(),
            v.clocks.as_ptr(),
            v.members.as_ptr(),
            v.members[0].id.as_str().as_ptr(),
            v.closing.as_ref().map(|c| c.detail.as_ptr()),
        )
    };
    let before = buffers(&held);
    decode_view_into(&two, &mut held, false).expect("decodes");
    assert_eq!(held.leases[0].expires_ns, next.leases[0].expires_ns);
    assert_eq!(buffers(&held), before, "every buffer reused");
    // A view refused part way leaves the next one decodable all the same.
    assert!(decode_view_into(&two[..two.len() - 3], &mut held, false).is_err());
    decode_view_into(&one, &mut held, false).expect("decodes");
    assert_eq!(
        DeviceView {
            parts: vec![arm()],
            ..held
        },
        view
    );
}

/// eye-1's view in the demo's session decodes in place to what prost made
/// of it, its part skipped or not, and lean.
#[test]
fn the_demo_view_decodes_in_place_as_prost_did() {
    let full = demo_view();
    let view = prost_view(&full).expect("decodes");
    assert_eq!(view.parts.len(), 1);
    let lean = encode_view(&DeviceView {
        parts: Vec::new(),
        ..view.clone()
    });
    let mut held = DeviceView::empty(pid("x"));
    decode_view_into(&full, &mut held, true).expect("decodes");
    assert_eq!(held, view);
    for bytes in [&full, &lean] {
        decode_view_into(bytes, &mut held, false).expect("decodes");
        assert_eq!(
            held,
            DeviceView {
                parts: Vec::new(),
                ..view.clone()
            }
        );
    }
}

#[test]
fn every_message_a_board_decodes_is_refused_unread_when_it_holds_too_much() {
    let why = |r: Result<(), WireError>| r.err().and_then(|e| e.why_unread());
    // Two thousand empty keys or leases: refused before a vector is made.
    let keys = [0x32, 0x00].repeat(2000);
    let mut view = DeviceView::empty(pid("x"));
    assert_eq!(
        why(decode_view_reply_into(&keys, &mut view, false).map(drop)),
        Some(Unread::Malformed("too short"))
    );
    let leases = [0x22, 0x00].repeat(2000);
    assert_eq!(
        why(decode_view_reply_into(&leases, &mut view, false).map(drop)),
        Some(Unread::Malformed("too short"))
    );
    assert_eq!(
        why(decode_key_delivery(&[0x0a, 0x00].repeat(2000)).map(drop)),
        Some(Unread::Malformed("too short"))
    );
    // More keys than any delivery holds, each a valid size.
    let mut many = Vec::new();
    for _ in 0..1025 {
        many.extend([0x0a, 34, 0x42, 32]);
        many.extend([7; 32]);
    }
    assert_eq!(
        why(decode_key_delivery(&many).map(drop)),
        Some(Unread::TooLarge("key delivery"))
    );
    assert_eq!(
        why(decode_key_delivery(
            &[0x12, 0x81, 0x02]
                .iter()
                .copied()
                .chain([1; 257])
                .collect::<Vec<_>>()
        )
        .map(drop)),
        Some(Unread::TooLarge("key delivery"))
    );
    // Two views in one reply, a view past its caps.
    let two = [0x4a, 0x00, 0x4a, 0x00].to_vec();
    assert_eq!(
        why(decode_view_reply_into(&two, &mut view, false).map(drop)),
        Some(Unread::Malformed("two views"))
    );
    // A chunk computed from two thousand empty inputs.
    assert_eq!(
        why(decode_chunk(Bytes::from([0x32, 0x00].repeat(2000))).map(drop)),
        Some(Unread::TooLarge("action chunk"))
    );
    // A stranger's unsealed answer: only its status and message are read,
    // and anything else in it is never decoded.
    let mut stranger = vec![0x08, 17, 0x12, 0x03, b'n', b'o', b'!'];
    stranger.extend([0x4a, 0x00].repeat(1000));
    let hint = decode_unauthenticated(&stranger).expect("a hint");
    assert_eq!(hint.message, "no!");
    let refusal = encode_view_reply(&ViewReply {
        result: Err(CtlError::new(CtlStatus::Unauthenticated, "x".repeat(300))),
        view: None,
        leases: Vec::new(),
        request_id: 0,
        keys: Vec::new(),
        assignment: None,
        run: None,
    });
    let hint = decode_unauthenticated(&refusal).expect("a hint");
    assert_eq!(
        (hint.status, hint.message.len()),
        (CtlStatus::Unauthenticated, 256)
    );
    assert!(decode_unauthenticated(&[0xff; 64]).is_none());
}

fn timing() -> ChunkTiming {
    ChunkTiming {
        channel: ChannelId(2),
        received: 950,
        accepted: 912,
        lost: 30,
        late_ms: [700, 150, 40, 10, 5, 4, 2, 1],
        late_max_us: 98_000,
        gap_max_us: 412_000,
    }
}

fn datagram_stats() -> DatagramStats {
    DatagramStats {
        active: true,
        sent: 3_000,
        received: 920,
        lost: 30,
        reordered: 2,
        unopened: 1,
        fallbacks: 0,
    }
}
