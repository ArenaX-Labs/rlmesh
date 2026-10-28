//! A UR10e-shaped session for the cockpit's tests, sans-IO: the host's real
//! authority state machine, a real binding from the binder, and synthetic
//! traffic on one injected clock (the cockpit's, equal to session time).

#![allow(dead_code)]

use bytes::Bytes;
use rlmesh_cockpit::{Input, Model, ModelConfig};
use rlmesh_session::spec::{Policy, RunOptions};
use rlmesh_session::wire::version::{PROTOCOL_MINOR, SPEC_VERSION};
use rlmesh_session::wire::{
    ActionChunk, Applied, ChannelId, ClockMapping, Command, DType, EdgeMode, Epoch, Health,
    Heartbeat, LeaseToken, ParticipantId, ParticipantReport, PriorityClass, PublisherCounters,
    ReasonCode, Request, Role, RunCounters, Sample, SessionId, StreamSpec,
};
use rlmesh_session::{Authority, AuthorityConfig};

pub const MS: i64 = 1_000_000;

pub const ROBOT: &str = r#"{
    "arm": {"kind": "joints", "names": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
            "fields": ["pos", "vel"], "hz": 500, "command": "position", "safe": "stop",
            "hold_max_ms": 100, "max_action_age_ms": 1000},
    "wrist/ft": {"kind": "wrench", "frame": "tool0", "hz": 500},
    "wrist/cam": {"kind": "camera", "width": 64, "height": 48, "hz": 30, "frame": "wrist_cam_optical"}
}"#;

pub const POLICY: &str = r#"{
    "hz": 50,
    "inputs": {
        "q": {"kind": "joints", "at": "arm", "fields": ["pos"],
              "names": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]},
        "cam": {"kind": "camera", "at": "wrist/cam", "size": [32, 32]}
    },
    "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position",
                        "names": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
                        "hz": 500, "horizon": 20}},
    "allow": ["resize", "pad"]
}"#;

pub fn pid(s: &str) -> ParticipantId {
    ParticipantId::new(s).expect("id")
}

/// The session, the cockpit's model, and the clock.
pub struct World {
    pub auth: Authority,
    pub model: Model,
    pub now: i64,
    seqs: std::collections::BTreeMap<ChannelId, u64>,
    chunk_seq: u64,
    /// The policy's run counters so far.
    pub run: RunCounters,
    /// Coalesced camera frames so far (the publisher's count).
    pub coalesced: u64,
    /// Skip edge ticks while this is in the future (a CPU stall).
    pub stall_until: i64,
    /// The edge mode `_applied` reports.
    pub edge_mode: EdgeMode,
    /// What a recorder would have written (snapshots, samples, chunks), by
    /// receipt time, in ingest order.
    pub recorded: Vec<(i64, Input)>,
}

impl World {
    /// robot, policy (model), operator and the cockpit attached; the
    /// policy bound and holding the arm's lease; the session ACTIVE.
    pub fn ur10e() -> Self {
        let mut w = Self {
            auth: Authority::new(SessionId::new(), AuthorityConfig::default()),
            model: Model::new(pid("cockpit"), ModelConfig::default()),
            now: 1_000 * MS,
            seqs: Default::default(),
            chunk_seq: 0,
            run: RunCounters::default(),
            coalesced: 0,
            stall_until: 0,
            edge_mode: EdgeMode::Driving,
            recorded: Vec::new(),
        };
        for (id, role, priority) in [
            ("ur10e", Role::Embodiment, PriorityClass::Scripted),
            ("policy", Role::Controller, PriorityClass::Model),
            ("operator", Role::Controller, PriorityClass::Operator),
            ("cockpit", Role::Observer, PriorityClass::Operator),
        ] {
            w.verb(
                id,
                Command::Attach {
                    name: id.into(),
                    roles: vec![role],
                    priority,
                    protocol_minor: PROTOCOL_MINOR,
                    supports: vec![],
                    requires: vec![],
                    workflow_edition: String::new(),
                    workflow_editions: vec![],
                },
            )
            .expect("attach");
        }
        w.publish_snapshot();
        w.verb(
            "ur10e",
            Command::DeclareParts {
                spec_version: SPEC_VERSION,
                robot_json: ROBOT.into(),
            },
        )
        .expect("declare");
        let parts: Vec<_> = w.auth.parts().cloned().collect();
        let binding = rlmesh_session::spec::bind(
            &Policy::parse(POLICY).expect("policy"),
            &parts,
            &RunOptions::named("reach"),
        )
        .expect("bound");
        w.verb(
            "policy",
            Command::DeclareBinding {
                fingerprint: binding.fingerprint,
                revision: 1,
                table_json: binding.table_text(),
                channels: binding.channels(),
                replan_hz: binding.replan_hz,
                contracts: binding.contracts.clone(),
                grids: binding.grids(),
            },
        )
        .expect("bind");
        let demands: Vec<_> = w
            .auth
            .snapshot(vec![], w.now)
            .demands
            .iter()
            .map(|d| (d.channel, d.revision))
            .collect();
        for (channel, revision) in demands {
            w.verb("ur10e", Command::DemandAck { channel, revision })
                .expect("ack");
        }
        // The operator commands the arm by hand: an identity binding.
        let arm = w.channel("arm");
        w.verb(
            "operator",
            Command::DeclareBinding {
                fingerprint: rlmesh_session::wire::Fingerprint::from(0x0be7_u64),
                revision: 1,
                table_json: "{}".into(),
                channels: vec![arm],
                replan_hz: 0.0,
                contracts: vec![],
                grids: vec![],
            },
        )
        .expect("operator binding");
        w.publish_snapshot();
        w.verb("cockpit", Command::Arm).expect("arm");
        w.publish_snapshot();
        w.grant("policy");
        w
    }

    pub fn verb(
        &mut self,
        from: &str,
        command: Command,
    ) -> Result<(), rlmesh_session::wire::CtlError> {
        let epoch = Some(self.auth.epoch());
        self.auth
            .apply(
                Request {
                    from: pid(from),
                    epoch,
                    request_id: 0,
                    command,
                },
                self.now,
            )
            .map(drop)
    }

    pub fn grant(&mut self, who: &str) {
        let arm = self.channel("arm");
        let binding = self
            .auth
            .binding(&pid(who))
            .map(|b| b.fingerprint)
            .unwrap_or_default();
        self.verb(
            who,
            Command::GrantLease {
                channels: vec![arm],
                release: vec![],
                binding,
                ttl_ns: None,
            },
        )
        .expect("grant");
        self.publish_snapshot();
    }

    pub fn channel(&self, path: &str) -> ChannelId {
        self.auth
            .parts()
            .find(|p| p.path.as_str() == path)
            .map(|p| p.channel)
            .expect(path)
    }

    fn stream(&self, channel: ChannelId) -> StreamSpec {
        self.model
            .snapshot()
            .and_then(|s| s.stream(channel))
            .expect("stream")
    }

    /// The host publishes its snapshot; the cockpit takes it in.
    pub fn publish_snapshot(&mut self) {
        let clocks = ["ur10e", "policy", "operator", "cockpit"]
            .iter()
            .map(|id| ClockMapping {
                participant: pid(id),
                offset_ns: 0,
                error_ns: 20_000,
                rtt_ns: 60_000,
                sample_ns: self.now,
                drift_bound_ppm: 100.0,
            })
            .collect();
        let snapshot = self.auth.snapshot(clocks, self.now);
        self.record(Input::Authority(snapshot), self.now);
    }

    fn payload(spec: &StreamSpec, v: f32) -> Bytes {
        let fields: Vec<Vec<u8>> = spec
            .fields
            .iter()
            .map(|f| {
                let n = f.elements().unwrap_or(0) as usize;
                match f.dtype {
                    DType::U8 => (0..n).map(|i| (i as u8).wrapping_add(v as u8)).collect(),
                    _ => (0..n)
                        .flat_map(|i| (v + i as f32 * 0.1).to_le_bytes())
                        .collect(),
                }
            })
            .collect();
        let refs: Vec<&[u8]> = fields.iter().map(Vec::as_slice).collect();
        Bytes::from(spec.layout.pack(&refs).expect("pack"))
    }

    /// A sample of `channel` captured now, arriving `delay_ns` later.
    pub fn sample(&mut self, channel: ChannelId, applied: Option<Applied>, delay_ns: i64) {
        let spec = self.stream(channel);
        let seq = self.seqs.entry(channel).or_default();
        *seq += 1;
        let payload = match applied {
            Some(a) if !a.present => Bytes::new(),
            _ => Self::payload(&spec, (self.now / MS) as f32 * 0.001),
        };
        let sample = Sample {
            channel,
            epoch: self.auth.epoch(),
            seq: *seq,
            capture_ns: self.now,
            publish_ns: self.now,
            tick: 0,
            payload,
            applied,
        };
        self.record(
            Input::Sample {
                sample,
                receive_ns: self.now + delay_ns,
                capture_local_ns: self.now,
            },
            self.now + delay_ns,
        );
    }

    /// Skips the next seq of `channel` (a loss on the way).
    pub fn lose(&mut self, channel: ChannelId) {
        *self.seqs.entry(channel).or_default() += 1;
    }

    /// The policy's chunk for the arm, published now, arriving 2 ms later.
    pub fn chunk(&mut self, source: &str, token: LeaseToken) {
        self.chunk_seq += 1;
        let arm = self.channel("arm");
        let chunk = ActionChunk {
            channel: arm,
            epoch: self.auth.epoch(),
            lease_token: token,
            seq: self.chunk_seq,
            source: pid(source),
            binding: self
                .auth
                .binding(&pid(source))
                .map(|b| b.fingerprint)
                .unwrap_or_default(),
            based_on: vec![],
            t0_ns: self.now,
            dt_ns: 0,
            replace_from_ns: self.now,
            valid_until_ns: self.now + 1_000 * MS,
            count: 20,
            actions: Bytes::new(),
            publish_ns: self.now,
            tick: 0,
        };
        self.record(
            Input::Chunk {
                chunk,
                receive_ns: self.now + 2 * MS,
            },
            self.now + 2 * MS,
        );
    }

    fn record(&mut self, input: Input, at: i64) {
        self.recorded.push((at, input.clone()));
        self.model.ingest(input, at);
    }

    fn heartbeat(&mut self, who: &str, seen: Vec<(ChannelId, u64)>, queue_depth: u32) {
        let heartbeat = Heartbeat {
            participant: pid(who),
            epoch: self.auth.epoch(),
            mode: self.auth.mode(),
            seq: 0,
            t1_ns: self.now,
            previous: None,
            last_seq_seen: seen,
            last_chunk_accepted: self.chunk_seq,
            last_chunk_applied: self.chunk_seq,
            queue_depth,
            health: Health::Ok,
        };
        self.model.ingest(
            Input::Heartbeat {
                heartbeat,
                receive_ns: self.now,
            },
            self.now,
        );
    }

    /// `ms` milliseconds of a healthy loop: arm, ft and `_applied` at
    /// 500 Hz, the camera at 20 Hz (the publisher producing 30 and
    /// coalescing 10), the policy's 50 Hz chunks, and every 250 ms the
    /// heartbeats, the reports and a republished snapshot.
    pub fn run(&mut self, ms: i64) {
        let (arm, ft, cam) = (
            self.channel("arm"),
            self.channel("wrist/ft"),
            self.channel("wrist/cam"),
        );
        let applied = self
            .auth
            .part(arm)
            .and_then(|p| p.applied_channel)
            .expect("applied");
        let end = self.now + ms * MS;
        while self.now < end {
            let t = self.now / MS;
            if t % 2 == 0 {
                self.sample(arm, None, MS / 2);
                self.sample(ft, None, MS / 2);
                if self.now >= self.stall_until {
                    let token = self.auth.lease(arm).map(|l| l.token).unwrap_or_default();
                    let present = matches!(self.edge_mode, EdgeMode::Driving | EdgeMode::Hold);
                    let a = if present {
                        Applied {
                            mode: self.edge_mode,
                            present: true,
                            lease_token: token,
                            chunk_seq: self.chunk_seq,
                            action_index: 0,
                            reason: ReasonCode::Unspecified,
                        }
                    } else {
                        Applied::absent(self.edge_mode)
                    };
                    self.sample(applied, Some(a), MS / 2);
                }
            }
            if t % 50 == 0 {
                self.sample(cam, None, 3 * MS);
            }
            if t % 100 == 50 {
                self.coalesced += 1;
            }
            if t % 20 == 0 {
                self.run.starts += 1;
                self.run.completions += 1;
                self.run.chunks_sent += 1;
                if let Some(lease) = self.auth.lease(arm).filter(|l| l.holder == pid("policy")) {
                    let token = lease.token;
                    self.chunk("policy", token);
                }
            }
            if t % 250 == 0 {
                let seen: Vec<_> = [arm, cam]
                    .iter()
                    .map(|c| (*c, self.seqs.get(c).copied().unwrap_or(0)))
                    .collect();
                self.heartbeat("policy", seen, 0);
                self.heartbeat("ur10e", vec![], 18);
                let epoch = self.auth.epoch();
                let streams = [arm, ft, cam]
                    .iter()
                    .map(|c| PublisherCounters {
                        channel: *c,
                        published: self.seqs.get(c).copied().unwrap_or(0),
                        coalesced: if *c == cam { self.coalesced } else { 0 },
                    })
                    .collect();
                for (who, streams, run) in
                    [("ur10e", streams, None), ("policy", vec![], Some(self.run))]
                {
                    self.model.ingest(
                        Input::Report {
                            report: ParticipantReport {
                                participant: pid(who),
                                epoch: Epoch(epoch.0),
                                sent_ns: self.now,
                                streams,
                                run,
                            },
                            receive_ns: self.now,
                        },
                        self.now,
                    );
                }
            }
            if t % 50 == 0 {
                // Holders' heartbeats renew their leases; the host
                // republishes while leases are held (every lease_grace/4).
                let epoch = self.auth.epoch();
                for who in ["policy", "operator"] {
                    self.auth.heartbeat(&pid(who), epoch, self.now);
                }
                self.publish_snapshot();
            }
            self.now += MS;
        }
    }
}
