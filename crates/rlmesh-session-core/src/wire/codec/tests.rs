//! The codec's round trips against the byte goldens, by message family,
//! and the goldens and snapshot they share.

mod control;
mod lockstep;
mod model;
mod stream;

use super::*;
use crate::edge::Rejection;
use crate::wire::action::{ActionChunk, ChunkRef, InputRange};
use crate::wire::authority::{
    AuthoritySnapshot, BindingRecord, ClockMapping, ClockMode, Command, CtlError, CtlReply,
    CtlStatus, HandshakeAccept, HandshakeRequest, HeldRobot, LastStop, Lease, LeaseEnd, Mode,
    ParticipantInfo, PriorityClass, Reason, ReasonCode, Request, Role, Watermark,
};
use crate::wire::contract::{ActionGrid, ContractMode, Demand, Plan, StreamContract};
use crate::wire::datagram::{DatagramOffer, DatagramRoute, DatagramRoutes};
use crate::wire::health::{DataPath, HostHealth, LinkHealth};
use crate::wire::heartbeat::{
    ChunkTiming, ClockRefusal, DatagramStats, Health, Heartbeat, HeartbeatEcho, TimingSample,
};
use crate::wire::ids::{ChannelId, Epoch, Fingerprint, LeaseToken};
use crate::wire::lockstep::{
    LockstepOp, LockstepReply, LockstepRequest, LockstepReset, LockstepResponse, LockstepResult,
    LockstepStep, PartValues,
};
use crate::wire::model::{
    ModelEpisode, ModelFault, ModelGroup, ModelHistoryRow, ModelReply, ModelRequest, ModelResolve,
    ModelResolved,
};
use crate::wire::part::{MAX_BATCH, MAX_CHUNK_ACTIONS};
use crate::wire::report::{ParticipantReport, PublisherCounters, RunCounters};
use crate::wire::sample::{
    Applied, EdgeMode, HistoryQuery, Sample, SampleBatch, SampleMeta, SeqRange,
};
use crate::wire::scene::{
    Assignment, DeviceRecord, HostOrder, HostStarted, Join, LobbyWait, MemberStage, MemberState,
    Place, Pose, RobotPin, RunStage, RunState, SceneInfo, StopOrder,
};
use crate::wire::tick::{Step, TickState, VirtualTick};
use crate::wire::timing::EndpointTiming;
use crate::wire::view::{DeviceView, Member, ViewGrid, ViewReply};
use bytes::Bytes;
use core::time::Duration;
use prost::Message as _;

use crate::wire::ids::SessionId;

use crate::wire::part::tests::{arm, camera, pid};

use crate::wire::version::PROTOCOL_MINOR;

/// Byte-level goldens of the v2 wire: every message a round-trip test
/// encodes, in order, as hex in `tests/vectors/wire/{test}.txt`. A
/// change to one is a wire change: regenerate with
/// `RLMESH_REGENERATE_WIRE_VECTORS=1 cargo test -p rlmesh-session-core
/// --lib wire::codec` only when the change is deliberate.
struct Goldens {
    test: &'static str,
    lines: Vec<String>,
    seen: alloc::collections::BTreeMap<&'static str, usize>,
}

impl Goldens {
    fn new(test: &'static str) -> Self {
        Self {
            test,
            lines: Vec::new(),
            seen: alloc::collections::BTreeMap::new(),
        }
    }

    fn add<'a>(&mut self, message: &'static str, bytes: &'a [u8]) -> &'a [u8] {
        let n = self.seen.entry(message).or_default();
        *n += 1;
        let hex: String = bytes.iter().map(|b| alloc::format!("{b:02x}")).collect();
        self.lines.push(alloc::format!("{message}#{n} {hex}"));
        bytes
    }

    fn check(self) {
        extern crate std;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/vectors/wire")
            .join(alloc::format!("{}.txt", self.test));
        let mut text = self.lines.join("\n");
        text.push('\n');
        if std::env::var_os("RLMESH_REGENERATE_WIRE_VECTORS").is_some() {
            std::fs::create_dir_all(path.parent().expect("dir")).expect("mkdir");
            std::fs::write(&path, &text).expect("write");
        }
        let want = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            want == text,
            "{}: the v2 wire bytes changed (a wire change); regenerate only when it is deliberate",
            path.display()
        );
    }
}

fn snapshot() -> AuthoritySnapshot {
    AuthoritySnapshot {
        session: SessionId::named("wire-goldens"),
        epoch: Epoch(3),
        revision: 9,
        mode: Mode::Active,
        estop_latched: false,
        roster: vec![ParticipantInfo {
            id: pid("op"),
            name: "Operator".into(),
            roles: vec![Role::Controller, Role::Observer],
            priority: PriorityClass::Operator,
            attached_ns: 12,
            protocol_minor: PROTOCOL_MINOR,
            supports: vec!["frame.fixed".into()],
            requires: vec!["merge.blend".into()],
            workflow_edition: "2026.06".into(),
            workflow_editions: vec!["2026.06".into()],
            present: false,
            left: Some(Reason {
                code: ReasonCode::RobotLost,
                by: "op".into(),
                at_ns: 10,
                revision: 8,
                ..Reason::default()
            }),
            device: "local".into(),
            generation: 3,
        }],
        leases: vec![Lease {
            channel: ChannelId(1),
            holder: pid("op"),
            priority: PriorityClass::Operator,
            token: LeaseToken(4),
            ttl_ns: 5,
            granted_ns: 6,
            expires_ns: 11,
            binding: Fingerprint::from(77),
            key_id: 0,
        }],
        clocks: vec![ClockMapping {
            participant: pid("op"),
            offset_ns: -42,
            error_ns: 7,
            rtt_ns: 12,
            sample_ns: 3,
            drift_bound_ppm: 100.0,
        }],
        host_ns: 99,
        parts: vec![camera(), arm()],
        protocol_minor: PROTOCOL_MINOR,
        supports: Vec::new(),
        clock_mode: ClockMode::Virtual,
        tick_ns: 1_000_000,
        bindings: vec![BindingRecord {
            participant: pid("op"),
            fingerprint: Fingerprint::from(77),
            revision: 2,
            table_json: "{}".into(),
            channels: vec![ChannelId(1)],
            replan_hz: 20.0,
            contracts: vec![StreamContract {
                channel: ChannelId(3),
                mode: ContractMode::History,
                delivery_hz: 20.0,
                history: 50,
                max_age: Some(Duration::from_millis(150)),
                max_age_explicit: false,
            }],
            grids: vec![ActionGrid {
                channel: ChannelId(1),
                dt: Duration::from_millis(10),
                chunk: 0,
            }],
        }],
        contract: [(pid("ur10e"), Fingerprint::from(0xabcd))].into(),
        demands: vec![
            Demand {
                channel: ChannelId(3),
                revision: 2,
                plan: Plan::Every,
                retain: Duration::from_secs(2),
                acked: 1,
            },
            Demand {
                channel: ChannelId(4),
                revision: 1,
                plan: Plan::Latest { hz: 15.0 },
                retain: Duration::ZERO,
                acked: 0,
            },
            Demand {
                channel: ChannelId(5),
                revision: 1,
                plan: Plan::Every,
                retain: Duration::ZERO,
                acked: 1,
            },
        ],
        mode_reason: Reason {
            code: ReasonCode::FirstLease,
            by: "op".into(),
            detail: String::new(),
            at_ns: 6,
            revision: 8,
            verified: false,
        },
        ended: vec![LeaseEnd {
            lease: Lease {
                channel: ChannelId(1),
                holder: pid("model"),
                priority: PriorityClass::Model,
                token: LeaseToken(3),
                ttl_ns: 5,
                granted_ns: 2,
                expires_ns: 7,
                binding: Fingerprint::from(76),
                key_id: 0,
            },
            reason: Reason {
                code: ReasonCode::Preempted,
                by: "op".into(),
                at_ns: 6,
                revision: 8,
                ..Reason::default()
            },
            by: None,
            to: None,
        }],
        last_stop: LastStop {
            epoch: Epoch(2),
            revision: 5,
        },
        places: vec![Place {
            robot: pid("ur10e"),
            at: "arm".into(),
            from: String::new(),
            frame: "base".into(),
            pose: Pose {
                xyz: [0.4, -0.2, 0.0],
                quat_xyzw: [0.0, 0.0, 0.6, 0.8],
            },
        }],
        held: vec![HeldRobot {
            robot: pid("ur10e"),
            reason: Reason {
                code: ReasonCode::DeviceWatchdog,
                by: "ur10e".into(),
                detail: "board link".into(),
                at_ns: 7,
                revision: 8,
                verified: true,
            },
        }],
        scene: Some(Box::new(SceneInfo {
            run: "r-7k2m".into(),
            name: "eyeball-rig".into(),
            fingerprint: Fingerprint::from(0x5ce5),
            closing: Some(Reason::new(ReasonCode::SceneEnded)),
            rejoin_until_ns: Some(60),
        })),
    }
}

#[test]
fn descriptors_name_every_recorded_envelope() {
    let names = String::from_utf8_lossy(FILE_DESCRIPTOR_SET);
    assert!(names.contains("rlmesh/session/v2/session.proto"));
    for full in [
        schema::SAMPLE,
        schema::ACTION_CHUNK,
        schema::AUTHORITY_STATE,
        schema::BINDING_RECORD,
    ] {
        let message = full.rsplit('.').next().expect("a message name");
        assert!(names.contains(message), "{message} is not described");
    }
}
