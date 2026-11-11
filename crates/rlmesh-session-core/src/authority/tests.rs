//! The authority's tests, by topic, and the fixture they share.

mod barrier;
mod demand;
mod leases;
mod mode;
mod parts;
mod roster;
#[cfg(feature = "fleet")]
mod scene;

use super::*;

use crate::wire::version::SPEC_VERSION;
use crate::wire::{Command, CtlError, CtlStatus, PriorityClass, ReasonCode, Request, Role};
use core::fmt;

const GRACE: u64 = 100;

fn pid(id: &str) -> ParticipantId {
    ParticipantId::new(id).expect("id")
}

/// Four ungrouped commandable parts, an `arm` group of two, and a
/// sensor.
fn robot_map(extra: &str) -> String {
    let cmd = |path: &str, group: &str| {
        format!(
            r#""{path}": {{"kind": "joints", "hz": 500, "count": 6, "fields": ["pos"],
                   "command": "position", "safe": "stop"{group}}}"#
        )
    };
    let parts = [
        cmd("cmd1", ""),
        cmd("cmd2", ""),
        cmd("cmd3", ""),
        cmd("cmd4", ""),
        cmd("arm", r#", "group": "arm""#),
        cmd("arm/tcp", r#", "group": "arm""#),
        r#""ft": {"kind": "x/ft", "hz": 1000, "fields": {"wrench": ["float32", [6]]}}"#.to_string(),
    ];
    format!("{{{}{extra}}}", parts.join(","))
}

struct Fixture {
    auth: Authority,
}

impl Fixture {
    /// robot (embodiment), policy (model), operator, script, viewer; each
    /// controller bound to every commandable part.
    fn new() -> Self {
        let mut auth = Authority::new(
            SessionId::new(),
            AuthorityConfig {
                lease_grace_ns: GRACE,
                ..AuthorityConfig::default()
            },
        );
        for (id, roles, priority) in [
            ("robot", vec![Role::Embodiment], PriorityClass::Scripted),
            ("policy", vec![Role::Controller], PriorityClass::Model),
            ("operator", vec![Role::Controller], PriorityClass::Operator),
            ("script", vec![Role::Controller], PriorityClass::Scripted),
            ("viewer", vec![Role::Observer], PriorityClass::Scripted),
        ] {
            auth.apply(
                Request {
                    from: pid(id),
                    epoch: None,
                    request_id: 0,
                    command: attach(id, roles, priority),
                },
                0,
            )
            .expect("attach");
        }
        let mut f = Self { auth };
        f.run("robot", declare(&robot_map("")), 0).expect("declare");
        for (i, id) in ["policy", "operator", "script"].into_iter().enumerate() {
            f.bind(
                id,
                i as u64 + 1,
                &["cmd1", "cmd2", "cmd3", "cmd4", "arm", "arm/tcp"],
            )
            .expect("bind");
        }
        f
    }

    fn ch(&self, path: &str) -> ChannelId {
        self.auth
            .parts()
            .find(|p| p.path.as_str() == path)
            .map(|p| p.channel)
            .expect(path)
    }

    fn run(&mut self, from: &str, command: Command, now: i64) -> Result<Applied, CtlError> {
        let epoch = Some(self.auth.epoch());
        self.auth.apply(
            Request {
                from: pid(from),
                epoch,
                request_id: 0,
                command,
            },
            now,
        )
    }

    fn bind(&mut self, from: &str, fp: u64, paths: &[&str]) -> Result<Applied, CtlError> {
        let channels = paths.iter().map(|p| self.ch(p)).collect();
        self.run(
            from,
            Command::DeclareBinding {
                fingerprint: Fingerprint::from(fp),
                revision: 1,
                table_json: "{}".into(),
                channels,
                replan_hz: 0.0,
                contracts: Vec::new(),
                grids: Vec::new(),
            },
            0,
        )
    }

    fn fingerprint(&self, from: &str) -> Fingerprint {
        self.auth
            .binding(&pid(from))
            .map_or_else(Fingerprint::default, |b| b.fingerprint)
    }

    fn grant_with(
        &mut self,
        from: &str,
        paths: &[&str],
        release: &[&str],
        now: i64,
    ) -> Result<Vec<Lease>, CtlError> {
        let command = Command::GrantLease {
            channels: paths.iter().map(|p| self.ch(p)).collect(),
            release: release.iter().map(|p| self.ch(p)).collect(),
            binding: self.fingerprint(from),
            ttl_ns: None,
        };
        self.run(from, command, now).map(|a| a.leases)
    }

    fn grant(&mut self, from: &str, path: &str, now: i64) -> Result<Lease, CtlError> {
        self.grant_with(from, &[path], &[], now)
            .map(|mut leases| leases.remove(0))
    }

    fn release(&mut self, from: &str, lease: &Lease, now: i64) -> Result<Applied, CtlError> {
        self.run(
            from,
            Command::ReleaseLease {
                channels: vec![lease.channel],
                token: lease.token,
            },
            now,
        )
    }

    fn status(result: Result<impl fmt::Debug, CtlError>) -> CtlStatus {
        result.expect_err("refused").status
    }
}

fn attach(name: &str, roles: Vec<Role>, priority: PriorityClass) -> Command {
    Command::Attach {
        name: name.into(),
        roles,
        priority,
        protocol_minor: PROTOCOL_MINOR,
        supports: Vec::new(),
        requires: Vec::new(),
        workflow_edition: String::new(),
        workflow_editions: Vec::new(),
        estop: None,
    }
}

fn declare(json: &str) -> Command {
    Command::DeclareParts {
        spec_version: SPEC_VERSION,
        robot_json: json.into(),
        seal: Vec::new(),
        board_id: None,
    }
}
