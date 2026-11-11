//! The edge core's tests, by topic, and the fixture they share.

mod dispatch;
mod fence;
mod group;

use crate::wire::version::{PROTOCOL_MINOR, SPEC_VERSION};
use crate::wire::{
    AuthoritySnapshot, ClockMapping, Command, Fingerprint, InputRange, Lease, PriorityClass,
    ReasonCode, Request, Role, SessionId,
};
use crate::{Authority, AuthorityConfig, VirtualConfig};

use super::*;

const GRACE: i64 = 100_000_000;
const MS: i64 = 1_000_000;

fn pid(id: &str) -> ParticipantId {
    ParticipantId::new(id).expect("id")
}

/// `cmd`: 6 float joints at 1 kHz, hold 10 ms; `grip`: binary, limited;
/// `arm` + `arm/tcp`: a group.
fn robot_map(hold_last: bool) -> String {
    // A core joint set holds its target; a custom kind runs safe.
    let kind = if hold_last {
        r#""kind": "joints", "count": 6, "fields": ["pos"]"#
    } else {
        r#""kind": "x/joints", "fields": {"pos": ["float32", [6]]}"#
    };
    format!(
        r#"{{
            "cmd": {{{kind}, "hz": 1000,
                    "command": "position", "hold_max_ms": 10, "safe": "brake",
                    "max_action_age_ms": 200}},
            "grip": {{"kind": "gripper", "hz": 1000, "command": "position", "safe": "stop", "binary": true}},
            "arm": {{"kind": "joints", "hz": 1000, "count": 2, "fields": ["pos"],
                    "command": "position", "safe": "stop", "group": "arm", "limits": {{"pos": [-1, 1]}}}},
            "arm/tcp": {{"kind": "x/tcp", "hz": 1000, "fields": {{"dpos": ["float32", [3]]}},
                        "command": "delta", "safe": "stop", "group": "arm"}}
        }}"#
    )
}

struct Fixture {
    auth: Authority,
    edge: EdgeCore,
    snapshot: Option<AuthoritySnapshot>,
    observed: i64,
    /// Mappings (offset, error) of the robot and the controllers.
    mapping: Option<(i64, u64)>,
}

impl Fixture {
    fn new() -> Self {
        Self::with(EdgeConfig::default(), true, &["cmd"], None)
    }

    fn with(
        config: EdgeConfig,
        hold_last: bool,
        paths: &[&str],
        virtual_clock: Option<VirtualConfig>,
    ) -> Self {
        let mut auth = Authority::new(
            SessionId::new(),
            AuthorityConfig {
                lease_grace_ns: GRACE as u64,
                virtual_clock,
                ..AuthorityConfig::default()
            },
        );
        for (id, roles, priority) in [
            ("robot", vec![Role::Embodiment], PriorityClass::Scripted),
            ("policy", vec![Role::Controller], PriorityClass::Model),
            ("operator", vec![Role::Controller], PriorityClass::Operator),
        ] {
            Self::run(
                &mut auth,
                id,
                Command::Attach {
                    name: id.into(),
                    roles,
                    priority,
                    protocol_minor: PROTOCOL_MINOR,
                    supports: Vec::new(),
                    requires: Vec::new(),
                    workflow_edition: String::new(),
                    workflow_editions: Vec::new(),
                    estop: None,
                },
            );
        }
        Self::run(
            &mut auth,
            "robot",
            Command::DeclareParts {
                spec_version: SPEC_VERSION,
                robot_json: robot_map(hold_last),
                seal: Vec::new(),
                board_id: None,
            },
        );
        let channels: Vec<ChannelId> = auth
            .parts()
            .filter(|p| p.command.is_some())
            .map(|p| p.channel)
            .collect();
        for (id, fp) in [("policy", 1), ("operator", 2)] {
            Self::run(
                &mut auth,
                id,
                Command::DeclareBinding {
                    fingerprint: Fingerprint::from(fp),
                    revision: 1,
                    table_json: "{}".into(),
                    channels: channels.clone(),
                    replan_hz: 0.0,
                    contracts: Vec::new(),
                    grids: Vec::new(),
                },
            );
        }
        Self::run(&mut auth, "operator", Command::Arm);
        let parts = paths
            .iter()
            .map(|path| {
                auth.parts()
                    .find(|p| p.path.as_str() == *path)
                    .cloned()
                    .expect(path)
            })
            .collect();
        let edge = EdgeCore::new(parts, pid("robot"), config).expect("edge");
        Self {
            auth,
            edge,
            snapshot: None,
            observed: 0,
            mapping: Some((0, 0)),
        }
    }

    fn run(auth: &mut Authority, from: &str, command: Command) {
        let epoch = Some(auth.epoch());
        auth.apply(
            Request {
                from: pid(from),
                epoch,
                request_id: 0,
                command,
            },
            0,
        )
        .expect("apply");
    }

    fn ch(&self, path: &str) -> ChannelId {
        self.auth
            .parts()
            .find(|p| p.path.as_str() == path)
            .map(|p| p.channel)
            .expect(path)
    }

    fn grant_with(&mut self, from: &str, path: &str, release: &[&str], now: i64) -> Lease {
        let fingerprint = self.auth.binding(&pid(from)).expect("binding").fingerprint;
        let epoch = Some(self.auth.epoch());
        let command = Command::GrantLease {
            channels: vec![self.ch(path)],
            release: release.iter().map(|p| self.ch(p)).collect(),
            binding: fingerprint,
            ttl_ns: None,
        };
        self.auth
            .apply(
                Request {
                    from: pid(from),
                    epoch,
                    request_id: 0,
                    command,
                },
                now,
            )
            .expect("grant")
            .leases
            .remove(0)
    }

    fn grant(&mut self, from: &str, now: i64) -> Lease {
        self.grant_with(from, "cmd", &[], now)
    }

    /// The edge observes the host's snapshot taken at `now`.
    fn publish(&mut self, now: i64) {
        let clocks = self
            .mapping
            .map(|(offset_ns, error_ns)| {
                ["robot", "policy", "operator"]
                    .into_iter()
                    .map(|id| ClockMapping {
                        participant: pid(id),
                        offset_ns,
                        error_ns,
                        rtt_ns: 0,
                        sample_ns: now,
                        drift_bound_ppm: 100.0,
                    })
                    .collect()
            })
            .unwrap_or_default();
        self.observed = now;
        self.snapshot = Some(self.auth.snapshot(clocks, now));
    }

    /// `n` actions at 1 ms from `t0` (session time) for `path`, each
    /// value `fill`.
    #[allow(clippy::too_many_arguments)]
    fn chunk_for(
        &self,
        path: &str,
        source: &str,
        lease: &Lease,
        seq: u64,
        t0: i64,
        n: u32,
        fill: f32,
    ) -> ActionChunk {
        let part = self
            .auth
            .parts()
            .find(|p| p.path.as_str() == path)
            .expect(path);
        let command = part.command.as_ref().expect("commandable");
        let stride = command.layout().expect("layout").stride().expect("stride");
        let elements: usize = command
            .fields
            .iter()
            .map(|f| f.elements().unwrap_or(0) as usize)
            .sum();
        let mut action: Vec<u8> = core::iter::repeat_n(fill.to_le_bytes(), elements)
            .flatten()
            .collect();
        action.resize(stride, 0);
        ActionChunk {
            channel: part.channel,
            epoch: self.auth.epoch(),
            lease_token: lease.token,
            seq,
            source: pid(source),
            binding: lease.binding,
            based_on: vec![InputRange::one(ChannelId(99), seq)],
            t0_ns: t0,
            dt_ns: MS as u64,
            replace_from_ns: t0,
            valid_until_ns: t0 + 150 * MS,
            count: n,
            actions: Bytes::from(action.repeat(n as usize)),
            publish_ns: t0,
            tick: 0,
        }
    }

    fn chunk(&self, source: &str, lease: &Lease, seq: u64, t0: i64, n: u32) -> ActionChunk {
        self.chunk_for("cmd", source, lease, seq, t0, n, 0.0)
    }

    fn accept(&mut self, chunk: ActionChunk, at: i64) -> Result<SpliceOutcome, Rejection> {
        self.edge
            .accept(chunk, at, self.snapshot.as_ref(), self.observed)
    }

    fn tick(&mut self, at: i64) -> EdgeTick {
        self.edge.tick(at, self.snapshot.as_ref(), self.observed)
    }
}
