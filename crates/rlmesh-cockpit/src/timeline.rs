//! The authority timeline: what changed between consecutive authority
//! snapshots, and lanes of state over time (session mode, each part's lease
//! holder, each edge's mode).
//!
//! The host publishes snapshots, not its events, so the cockpit derives
//! the events by diffing them. A lease that disappeared is classed by what
//! else changed in the same revision (§11.1): revoked when the mode left
//! ARMED/ACTIVE or the epoch moved, expired when the snapshot is past its
//! expiry, voided when the session went SAFE (an edge fell back), released
//! otherwise. That inference is the cockpit's: the snapshot carries no
//! reasons.

use std::collections::{BTreeMap, VecDeque};

use rlmesh_session::wire::{AuthoritySnapshot, Lease, Mode};

/// What an entry is about, for filtering and colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// The session mode or epoch.
    Mode,
    /// An e-stop latched or cleared.
    Estop,
    /// A lease granted, moved or ended.
    Lease,
    /// A binding declared, changed or removed.
    Binding,
    /// A participant attached or left.
    Roster,
    /// An edge changed mode (from its `_applied` stream).
    Edge,
    /// A verb this cockpit sent, and the host's answer.
    Command,
}

impl Kind {
    /// A short label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Mode => "mode",
            Self::Estop => "estop",
            Self::Lease => "lease",
            Self::Binding => "binding",
            Self::Roster => "roster",
            Self::Edge => "edge",
            Self::Command => "command",
        }
    }
}

/// How a state reads at a glance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tone {
    /// Nothing moves (disarmed, idle, free).
    Off,
    /// Ready or neutral.
    Neutral,
    /// Moving normally (active, driving, held lease).
    Good,
    /// Degraded but within the contract (hold).
    Warn,
    /// Fallen back or stopped (safe, estop).
    Bad,
}

/// One timeline entry.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// Cockpit clock when it was seen.
    pub at_ns: i64,
    /// Session time, when known (the snapshot's `host_ns`).
    pub session_ns: Option<i64>,
    /// The epoch it happened in.
    pub epoch: u64,
    /// What it is about.
    pub kind: Kind,
    /// How bad it is.
    pub tone: Tone,
    /// What happened.
    pub text: String,
}

/// The tone of a session mode.
pub fn mode_tone(mode: Mode) -> Tone {
    match mode {
        Mode::Disarmed => Tone::Off,
        Mode::Armed => Tone::Neutral,
        Mode::Active => Tone::Good,
        Mode::Safe | Mode::Estop => Tone::Bad,
    }
}

fn path_of(s: &AuthoritySnapshot, lease: &Lease) -> String {
    s.part(lease.channel).map_or_else(
        || lease.channel.to_string(),
        |p| format!("{}/{}", p.owner, p.path),
    )
}

fn class(lease: &Lease) -> &'static str {
    match lease.priority {
        rlmesh_session::wire::PriorityClass::Scripted => "scripted",
        rlmesh_session::wire::PriorityClass::Model => "model",
        rlmesh_session::wire::PriorityClass::Operator => "operator",
        rlmesh_session::wire::PriorityClass::Safety => "safety",
    }
}

/// What changed from `old` to `new`, as `(kind, tone, text)`. With no
/// `old` (the cockpit just attached), the state found.
pub fn diff(old: Option<&AuthoritySnapshot>, new: &AuthoritySnapshot) -> Vec<(Kind, Tone, String)> {
    let mut out = Vec::new();
    let Some(old) = old else {
        out.push((
            Kind::Mode,
            mode_tone(new.mode),
            format!(
                "attached: {} in epoch {}{}",
                new.mode,
                new.epoch.0,
                if new.estop_latched {
                    ", e-stop latched"
                } else {
                    ""
                }
            ),
        ));
        for lease in &new.leases {
            out.push((
                Kind::Lease,
                Tone::Good,
                format!(
                    "{} held by {} ({}, {})",
                    path_of(new, lease),
                    lease.holder,
                    class(lease),
                    lease.token
                ),
            ));
        }
        return out;
    };

    if old.epoch != new.epoch {
        out.push((
            Kind::Mode,
            Tone::Neutral,
            format!("epoch {} -> {}", old.epoch.0, new.epoch.0),
        ));
    }
    if old.mode != new.mode {
        out.push((
            Kind::Mode,
            mode_tone(new.mode),
            format!("{} -> {}", old.mode, new.mode),
        ));
    }
    match (old.estop_latched, new.estop_latched) {
        (false, true) => out.push((Kind::Estop, Tone::Bad, "e-stop latched".into())),
        (true, false) => out.push((
            Kind::Estop,
            Tone::Warn,
            "e-stop cleared locally: disarm, then arm".into(),
        )),
        _ => {}
    }

    for p in &new.roster {
        if old.participant(&p.id).is_none() {
            let roles: Vec<String> = p
                .roles
                .iter()
                .map(|r| format!("{r:?}").to_lowercase())
                .collect();
            out.push((
                Kind::Roster,
                Tone::Neutral,
                format!("{} attached ({})", p.id, roles.join(", ")),
            ));
        }
    }
    for p in &old.roster {
        if new.participant(&p.id).is_none() {
            out.push((Kind::Roster, Tone::Warn, format!("{} left", p.id)));
        }
    }

    let revoked = new.epoch != old.epoch || !matches!(new.mode, Mode::Armed | Mode::Active);
    for lease in &old.leases {
        let path = path_of(old, lease);
        match new.lease(lease.channel) {
            Some(now) if now.token == lease.token => {}
            Some(now) if now.holder == lease.holder => out.push((
                Kind::Lease,
                Tone::Good,
                format!(
                    "{path} rebound by {}: {} -> {}",
                    now.holder, lease.token, now.token
                ),
            )),
            Some(now) if now.priority > lease.priority => out.push((
                Kind::Lease,
                Tone::Warn,
                format!(
                    "{path} preempted: {} ({}) took it from {} ({}), {}",
                    now.holder,
                    class(now),
                    lease.holder,
                    class(lease),
                    now.token
                ),
            )),
            Some(now) => out.push((
                Kind::Lease,
                Tone::Good,
                format!(
                    "{path} moved {} -> {} ({})",
                    lease.holder, now.holder, now.token
                ),
            )),
            None => {
                let (tone, why) = if new.participant(&lease.holder).is_none() {
                    (Tone::Warn, "holder left".to_string())
                } else if new.host_ns >= lease.expires_ns {
                    (Tone::Bad, "expired".to_string())
                } else if revoked && new.mode != Mode::Safe {
                    (Tone::Warn, format!("revoked ({})", new.mode))
                } else if new.mode == Mode::Safe {
                    (Tone::Bad, "voided: the session went SAFE".to_string())
                } else {
                    (Tone::Neutral, "released".to_string())
                };
                out.push((
                    Kind::Lease,
                    tone,
                    format!("{path} {why}: {} {}", lease.holder, lease.token),
                ));
            }
        }
    }
    for lease in &new.leases {
        if old.lease(lease.channel).is_none() {
            out.push((
                Kind::Lease,
                Tone::Good,
                format!(
                    "{} granted to {} ({}, {})",
                    path_of(new, lease),
                    lease.holder,
                    class(lease),
                    lease.token
                ),
            ));
        }
    }

    for b in &new.bindings {
        match old.binding(&b.participant) {
            None => out.push((
                Kind::Binding,
                Tone::Neutral,
                format!(
                    "{} bound: {} (rev {}, {} parts, replan {} Hz)",
                    b.participant,
                    b.fingerprint,
                    b.revision,
                    b.channels.len(),
                    b.replan_hz
                ),
            )),
            Some(was) if was.fingerprint != b.fingerprint => out.push((
                Kind::Binding,
                Tone::Neutral,
                format!(
                    "{} rebound: {} -> {} (rev {})",
                    b.participant, was.fingerprint, b.fingerprint, b.revision
                ),
            )),
            Some(_) => {}
        }
    }
    for b in &old.bindings {
        if new.binding(&b.participant).is_none() {
            out.push((
                Kind::Binding,
                Tone::Neutral,
                format!("{} unbound ({})", b.participant, b.fingerprint),
            ));
        }
    }
    out
}

/// One stretch of a lane in one state.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    /// Cockpit clock at its start.
    pub start_ns: i64,
    /// The state.
    pub label: String,
    /// Its tone.
    pub tone: Tone,
}

/// A state over time: consecutive segments, the last one still open.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lane {
    segments: VecDeque<Segment>,
}

impl Lane {
    /// The state is `label` from `at_ns` on (a repeat extends the open
    /// segment).
    pub fn set(&mut self, at_ns: i64, label: impl Into<String>, tone: Tone) {
        let label = label.into();
        if self
            .segments
            .back()
            .is_some_and(|s| s.label == label && s.tone == tone)
        {
            return;
        }
        self.segments.push_back(Segment {
            start_ns: at_ns,
            label,
            tone,
        });
    }

    /// Drops segments that ended before `horizon_ns`.
    pub fn prune(&mut self, horizon_ns: i64) {
        while self.segments.len() > 1 && self.segments[1].start_ns <= horizon_ns {
            self.segments.pop_front();
        }
    }

    /// The segments, oldest first.
    pub fn segments(&self) -> impl Iterator<Item = &Segment> {
        self.segments.iter()
    }

    /// The current state.
    pub fn current(&self) -> Option<&Segment> {
        self.segments.back()
    }
}

/// Every entry, bounded, and the lanes.
#[derive(Clone, Debug, Default)]
pub struct Timeline {
    cap: usize,
    entries: VecDeque<Entry>,
    /// The session mode.
    pub mode: Lane,
    /// Each lease's holder, by part label.
    pub leases: BTreeMap<String, Lane>,
    /// Each edge's mode, by part label.
    pub edges: BTreeMap<String, Lane>,
}

impl Timeline {
    /// A timeline keeping the newest `cap` entries.
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            ..Self::default()
        }
    }

    /// Appends an entry.
    pub fn push(&mut self, entry: Entry) {
        if self.entries.len() == self.cap {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// The entries, oldest first.
    pub fn entries(&self) -> impl DoubleEndedIterator<Item = &Entry> {
        self.entries.iter()
    }

    /// Takes in a new snapshot seen at `at_ns`: its diff from `old` as
    /// entries, and the lanes.
    pub fn observe(
        &mut self,
        old: Option<&AuthoritySnapshot>,
        new: &AuthoritySnapshot,
        at_ns: i64,
    ) {
        for (kind, tone, text) in diff(old, new) {
            self.push(Entry {
                at_ns,
                session_ns: Some(new.host_ns),
                epoch: new.epoch.0,
                kind,
                tone,
                text,
            });
        }
        self.mode.set(
            at_ns,
            format!("{} e{}", new.mode, new.epoch.0),
            mode_tone(new.mode),
        );
        for part in new.parts.iter().filter(|p| p.command.is_some()) {
            let label = format!("{}/{}", part.owner, part.path);
            let lane = self.leases.entry(label).or_default();
            match new.lease(part.channel) {
                Some(lease) => lane.set(
                    at_ns,
                    format!("{} {}", lease.holder, lease.token),
                    Tone::Good,
                ),
                None => lane.set(at_ns, "free", Tone::Off),
            }
        }
    }

    /// Drops lane history older than `horizon_ns`.
    pub fn prune(&mut self, horizon_ns: i64) {
        self.mode.prune(horizon_ns);
        for lane in self.leases.values_mut().chain(self.edges.values_mut()) {
            lane.prune(horizon_ns);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_session::wire::{
        ChannelId, ClockMode, Epoch, Fingerprint, LastStop, LeaseToken, ParticipantId,
        ParticipantInfo, PriorityClass, Reason, Role, SessionId,
    };

    fn pid(s: &str) -> ParticipantId {
        ParticipantId::new(s).expect("id")
    }

    fn snapshot(mode: Mode, epoch: u64, leases: Vec<Lease>, host_ns: i64) -> AuthoritySnapshot {
        AuthoritySnapshot {
            session: SessionId::new(),
            epoch: Epoch(epoch),
            revision: 1,
            mode,
            estop_latched: mode == Mode::Estop,
            roster: ["policy", "operator"]
                .iter()
                .map(|id| ParticipantInfo {
                    id: pid(id),
                    name: (*id).into(),
                    roles: vec![Role::Controller],
                    priority: PriorityClass::Model,
                    attached_ns: 0,
                    protocol_minor: 0,
                    supports: vec![],
                    requires: vec![],
                    workflow_edition: String::new(),
                    workflow_editions: vec![],
                    present: true,
                    left: None,
                    device: "local".into(),
                    generation: 1,
                })
                .collect(),
            leases,
            clocks: vec![],
            host_ns,
            parts: vec![],
            protocol_minor: 0,
            supports: vec![],
            clock_mode: ClockMode::Wall,
            tick_ns: 0,
            bindings: vec![],
            contract: BTreeMap::new(),
            demands: vec![],
            mode_reason: Reason::default(),
            ended: vec![],
            last_stop: LastStop::default(),
        }
    }

    fn lease(holder: &str, priority: PriorityClass, token: u64, expires_ns: i64) -> Lease {
        Lease {
            channel: ChannelId(3),
            holder: pid(holder),
            priority,
            token: LeaseToken(token),
            ttl_ns: 250,
            granted_ns: 0,
            expires_ns,
            binding: Fingerprint::from(1_u64),
        }
    }

    fn texts(old: &AuthoritySnapshot, new: &AuthoritySnapshot) -> Vec<String> {
        diff(Some(old), new)
            .into_iter()
            .map(|(_, _, t)| t)
            .collect()
    }

    #[test]
    fn mode_epoch_and_estop_changes_are_named() {
        let a = snapshot(Mode::Disarmed, 1, vec![], 0);
        let b = snapshot(Mode::Armed, 2, vec![], 10);
        assert_eq!(texts(&a, &b), ["epoch 1 -> 2", "DISARMED -> ARMED"]);
        let e = snapshot(Mode::Estop, 2, vec![], 20);
        assert_eq!(texts(&b, &e), ["ARMED -> ESTOP", "e-stop latched"]);
        let first = diff(None, &e);
        assert_eq!(first[0].2, "attached: ESTOP in epoch 2, e-stop latched");
    }

    #[test]
    fn lease_endings_are_classed_by_what_else_changed() {
        let held = snapshot(
            Mode::Active,
            2,
            vec![lease("policy", PriorityClass::Model, 7, 100)],
            50,
        );
        // Preempted by a higher class.
        let taken = snapshot(
            Mode::Active,
            2,
            vec![lease("operator", PriorityClass::Operator, 8, 300)],
            60,
        );
        assert!(
            texts(&held, &taken)[0]
                .contains("preempted: operator (operator) took it from policy (model), t8")
        );
        // Rebound by the same holder.
        let rebound = snapshot(
            Mode::Active,
            2,
            vec![lease("policy", PriorityClass::Model, 9, 300)],
            60,
        );
        assert!(texts(&held, &rebound)[0].contains("rebound by policy: t7 -> t9"));
        // Gone past its expiry: expired.
        let lapsed = snapshot(Mode::Safe, 2, vec![], 150);
        assert!(
            texts(&held, &lapsed)
                .iter()
                .any(|t| t.contains("expired: policy t7"))
        );
        // Gone early as the session went SAFE: an edge fell back.
        let fell_back = snapshot(Mode::Safe, 2, vec![], 60);
        assert!(
            texts(&held, &fell_back)
                .iter()
                .any(|t| t.contains("voided"))
        );
        // Gone early, nothing else changed: released.
        let released = snapshot(Mode::Active, 2, vec![], 60);
        assert!(texts(&held, &released)[0].contains("released"));
        // Gone with a disarm: revoked.
        let disarmed = snapshot(Mode::Disarmed, 2, vec![], 60);
        assert!(
            texts(&held, &disarmed)
                .iter()
                .any(|t| t.contains("revoked (DISARMED)"))
        );
        // A grant.
        let free = snapshot(Mode::Armed, 2, vec![], 0);
        assert!(
            texts(&free, &held)
                .iter()
                .any(|t| t.contains("granted to policy (model, t7)"))
        );
    }

    #[test]
    fn lanes_merge_repeats_and_prune_old_segments() {
        let mut lane = Lane::default();
        lane.set(0, "ACTIVE", Tone::Good);
        lane.set(5, "ACTIVE", Tone::Good);
        lane.set(10, "SAFE", Tone::Bad);
        lane.set(20, "ARMED", Tone::Neutral);
        assert_eq!(lane.segments().count(), 3);
        lane.prune(15);
        let labels: Vec<_> = lane.segments().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, ["SAFE", "ARMED"], "SAFE was still on at 15");
        let mut timeline = Timeline::new(2);
        for i in 0..3 {
            timeline.push(Entry {
                at_ns: i,
                session_ns: None,
                epoch: 1,
                kind: Kind::Command,
                tone: Tone::Neutral,
                text: i.to_string(),
            });
        }
        assert_eq!(timeline.entries().count(), 2, "bounded");
    }
}
