//! The edge of one command group, sans-IO: chunk intake, the action queues,
//! the fallback ladder and the applied stream.
//!
//! A group is the robot's commandable parts that drive the same actuators
//! (`command.group`); an ungrouped part is a group of one.
//! [`EdgeCore::accept`] validates each chunk on arrival against the edge's
//! last-known authority view and its own clock, and splices it into its
//! part's queue; the embodiment's control loop calls [`EdgeCore::tick`] once
//! per control period. Neither waits on the network: the edge keeps acting on
//! its last-known lease until that lease expires by its own clock, then
//! falls back.
//!
//! # Intake
//!
//! A chunk is rejected whole when: it is for another part; the robot
//! process latched a local e-stop ([`EdgeCore::set_local_estop`]) or the
//! session is in ESTOP; the epoch is wrong; the token is not the part's current lease's;
//! its source is not the lease holder; its binding is not the lease's;
//! the lease expired by the edge clock; it is past `valid_until`; it fails
//! the transit check; its seq is not above the last accepted for its token;
//! its payload does not match `count` × the command stride; or any value is
//! invalid ([`crate::edge::command`]). Valid actions are clamped to the
//! part's limits before they are queued, so the clamped value is what the
//! edge dispatches and what `_applied` shows.
//!
//! # Clock safety
//!
//! Chunk times are session time. With the edge's clock mapping
//! `session = local + θ` within `± ε`:
//!
//! - slot positions (`t0`, `replace_from`) convert as `t − θ`;
//! - **deadlines** (`valid_until`, the lease expiry) convert conservatively,
//!   `t − θ − ε`, so an offset error can only shorten them;
//! - a chunk may be up to `ε_ctrl + ε_edge` older than it reads, so its
//!   life ends at `min(valid_until − θ, receipt + max_action_age) − ε_ctrl −
//!   ε_edge`;
//! - the transit check: `receipt_session − publish_session + ε_ctrl +
//!   ε_edge` (the controller's `publish_ns` through its own mapping) above
//!   `max_action_age` rejects the chunk;
//! - with no mapping, chunks are refused and the lease cannot be timed,
//!   unless [`EdgeConfig::require_clock_mapping`] is off (every participant
//!   on one clock);
//! - either `ε` above the part's `max_clock_error` (a group's smallest)
//!   refuses the chunk, and [`EdgeCore::clock_refused`] says by how much.
//!
//! In virtual and lockstep sessions every time is session time and no
//! mapping is needed: the host times leases, and the edge's clock is the
//! tick time it is driven at.

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::sync::Arc;
use core::time::Duration;

use crate::wire::{
    ActionChunk, Applied, AuthorityView, ChannelId, ChunkTiming, ClockMode, ClockRefusal, EdgeMode,
    Epoch, FenceError, LeaseToken, Mode, PartSpec, ParticipantId, ReasonCode,
};
use bytes::Bytes;

use crate::edge::GroupError;
use crate::edge::command::CommandCheck;
use crate::edge::fallback::{
    Decision, Fallback, LeaseView, MemberLadder, ModeStats, SafeCause, TickView,
};
use crate::edge::queue::{
    ActionQueue, ChunkMeta, Delivery, QueueStats, QueuedAction, Splice, SpliceOutcome,
};

/// How an edge runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeConfig {
    /// Refuse chunks until this participant (and the chunk's source) has a
    /// clock mapping. Turn off only when every participant shares one clock
    /// (one machine, or PTP with `SystemMonotonicClock`): without a mapping,
    /// times are read as edge time and a delayed snapshot can extend the
    /// lease.
    pub require_clock_mapping: bool,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            require_clock_mapping: true,
        }
    }
}

/// Why the edge refused a chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rejection {
    /// Not a decodable chunk, or its payload is not `count` actions.
    Malformed,
    /// For a part outside this group.
    WrongChannel,
    /// A local e-stop is latched, or the session is in ESTOP.
    Estop,
    /// From another epoch.
    StaleEpoch,
    /// The part has no lease.
    NoLease,
    /// Fenced: not the current lease's token (a preempted or rebound
    /// holder).
    StaleToken,
    /// Its source is not the holder of the lease its token names.
    WrongSource,
    /// Its binding is not the lease's.
    WrongBinding,
    /// The lease expired by the edge's own clock.
    LeaseExpired,
    /// Its seq is not newer than one already accepted under the token.
    StaleSeq,
    /// Already past `valid_until` on arrival.
    Expired,
    /// The transit check: older than `max_action_age` on arrival.
    TooOld,
    /// No actions.
    Empty,
    /// A value is invalid (not finite, a bad bool or rotation).
    InvalidValue,
    /// No clock mapping yet ([`EdgeConfig::require_clock_mapping`]).
    NoClockMapping,
    /// A clock's error bound exceeds the part's `max_clock_error`.
    ClockUncertain,
    /// Not sealed under a command key this robot holds for the chunk's
    /// part, token and source: not the lease holder's.
    Unauthenticated,
}

impl Rejection {
    /// Every rejection, in stats order.
    pub const ALL: [Self; 17] = [
        Self::Malformed,
        Self::WrongChannel,
        Self::Estop,
        Self::StaleEpoch,
        Self::NoLease,
        Self::StaleToken,
        Self::WrongSource,
        Self::WrongBinding,
        Self::LeaseExpired,
        Self::StaleSeq,
        Self::Expired,
        Self::TooOld,
        Self::Empty,
        Self::InvalidValue,
        Self::NoClockMapping,
        Self::ClockUncertain,
        Self::Unauthenticated,
    ];

    /// Its place in [`Self::ALL`] (stats order).
    pub fn index(self) -> usize {
        self as usize
    }
}

impl From<FenceError> for Rejection {
    fn from(err: FenceError) -> Self {
        match err {
            FenceError::StaleEpoch { .. } => Self::StaleEpoch,
            FenceError::NoLease { .. } => Self::NoLease,
            FenceError::StaleToken { .. } => Self::StaleToken,
            FenceError::Expired { .. } => Self::LeaseExpired,
        }
    }
}

/// Counts for one edge.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EdgeStats {
    /// Chunks accepted.
    pub accepted: u64,
    /// Chunks refused, by reason (non-zero only).
    pub rejected: Vec<(Rejection, u64)>,
    /// What splicing and draining did to individual actions, over every
    /// member.
    pub queue: QueueStats,
    /// Entries, ticks and time per edge mode.
    pub modes: Vec<(EdgeMode, ModeStats)>,
    /// Times the edge went safe on its own account (and told the host).
    pub safe_reports: u64,
    /// Accepted actions a limit, a norm bound or the binary snap changed.
    pub clamped: u64,
    /// Handovers between group members.
    pub handovers: u64,
    /// Set while the newest clock judgement refused a chunk.
    pub clock_refused: Option<ClockRefusal>,
    /// How each member's chunks arrived.
    pub timing: Vec<ChunkTiming>,
}

impl EdgeStats {
    /// How `channel`'s chunks arrived, if it is a member.
    pub fn timing(&self, channel: ChannelId) -> Option<&ChunkTiming> {
        self.timing.iter().find(|t| t.channel == channel)
    }

    /// Chunks refused for `reason`.
    pub fn rejected(&self, reason: Rejection) -> u64 {
        self.rejected
            .iter()
            .find(|(r, _)| *r == reason)
            .map_or(0, |(_, n)| *n)
    }

    /// Chunks refused for any reason.
    pub fn rejected_total(&self) -> u64 {
        self.rejected.iter().map(|(_, n)| n).sum()
    }

    /// Ticks, entries and time in `mode`.
    pub fn mode(&self, mode: EdgeMode) -> ModeStats {
        self.modes
            .iter()
            .find(|(m, _)| *m == mode)
            .map_or_else(ModeStats::default, |(_, s)| *s)
    }
}

/// One sample of a part's applied stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppliedSample {
    /// The applied stream's id.
    pub stream: ChannelId,
    /// The commandable part.
    pub part: ChannelId,
    /// The envelope metadata.
    pub applied: Applied,
    /// The dispatched value (empty when nothing was).
    pub value: Bytes,
}

/// The lease moved between members of the group: the robot switches its
/// vendor control mode, neutralises the old member's target, then calls
/// [`EdgeCore::handover_done`]. Until then the group dispatches nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupHandover {
    /// The group.
    pub group: String,
    /// The part that held the lease.
    pub from: ChannelId,
    /// The part that holds it now.
    pub to: ChannelId,
}

/// What the embodiment does this tick.
#[derive(Clone, Debug)]
pub struct EdgeTick {
    /// The mode.
    pub mode: EdgeMode,
    /// The part the tick concerns (the active or last-driven member).
    pub channel: Option<ChannelId>,
    /// The action to dispatch to that part, if any.
    pub action: Option<QueuedAction>,
    /// The declared safe behaviour to run, if it must run.
    pub safe_behavior: Option<Arc<str>>,
    /// Set on the tick the edge went safe on its own account, with the
    /// epoch it happened in: the host must be told.
    pub report: Option<(Epoch, SafeCause)>,
    /// Set on the tick a handover between members starts.
    pub handover: Option<GroupHandover>,
    /// One applied sample per member.
    pub applied: Vec<AppliedSample>,
}

fn duration_i64(d: Duration) -> i64 {
    i64::try_from(d.as_nanos()).unwrap_or(i64::MAX)
}

/// The edge's clock relation to the session, from the authority view.
#[derive(Clone, Copy, Debug)]
struct ClockView {
    /// session = local + offset.
    offset_ns: i64,
    error_ns: i64,
    mapped: bool,
    /// Virtual or lockstep: local time is session time, leases are timed by
    /// the host.
    ticked: bool,
}

/// One commandable part of the group.
#[derive(Debug)]
struct Member {
    part: PartSpec,
    applied: ChannelId,
    check: CommandCheck,
    /// The control period, for chunks that carry `dt_ns = 0` under a
    /// binding that declared no action grid for the part.
    period_ns: i64,
    max_action_age_ns: i64,
    /// (epoch, token, seq) of the newest accepted chunk.
    last: Option<(Epoch, LeaseToken, u64)>,
    /// How its chunks arrived.
    timing: ChunkTiming,
    /// When the newest accepted chunk arrived (edge clock).
    last_at: Option<i64>,
}

impl Member {
    /// It accepted chunk `id` (epoch, token, seq), received at `receive_ns`
    /// and `late_ns` past its first slot: counted, and its newest. A skip in
    /// the seqs under one token is a loss, and only there is the time since
    /// the last one a gap.
    fn took(&mut self, id: (Epoch, LeaseToken, u64), receive_ns: i64, late_ns: i64) {
        let timing = &mut self.timing;
        timing.accepted += 1;
        timing.late(late_ns);
        if let Some((epoch, token, seq)) = self.last
            && (epoch, token) == (id.0, id.1)
        {
            let skipped = id.2.saturating_sub(seq).saturating_sub(1);
            timing.lost = timing
                .lost
                .saturating_add(u32::try_from(skipped).unwrap_or(u32::MAX));
            if let Some(at) = self.last_at {
                timing.gap(receive_ns.saturating_sub(at));
            }
        }
        self.last = Some(id);
        self.last_at = Some(receive_ns);
    }
}

/// The edge of one command group.
#[derive(Debug)]
pub struct EdgeCore {
    me: ParticipantId,
    group: Option<String>,
    members: Vec<Member>,
    queues: Vec<ActionQueue>,
    config: EdgeConfig,
    /// The smallest `max_clock_error` of the members.
    max_clock_error_ns: u64,
    /// Set while the newest clock judgement refused a chunk.
    clock_refused: Option<ClockRefusal>,
    safe_behavior: Arc<str>,
    fallback: Fallback,
    accepted: u64,
    rejected: [u64; Rejection::ALL.len()],
    safe_reports: u64,
    /// A fall not reported yet, waiting to see whether its holder is alive.
    unjudged: Option<Unjudged>,
    clamped: u64,
    handovers: u64,
}

/// A dry queue or a lapsed lease is what a dead holder looks like from here,
/// and the edge sees it before the host does. The group is safe either way;
/// only the report waits. A renewal from after the fall means the holder is
/// alive, so the fall is the robot's to report; the lease going away means
/// the host judged the holder lost, and only its parts stop.
#[derive(Clone, Copy, Debug)]
struct Unjudged {
    epoch: Epoch,
    cause: SafeCause,
    token: LeaseToken,
    /// An expiry at or past this came from a heartbeat after the fall.
    renewed_by_ns: i64,
}

impl EdgeCore {
    /// The edge for `parts`: commandable parts of `me`, either one part or
    /// every member of one command group (which share `safe` and
    /// `hold_max`).
    pub fn new(
        parts: Vec<PartSpec>,
        me: ParticipantId,
        config: EdgeConfig,
    ) -> Result<Self, GroupError> {
        let first = parts.first().ok_or(GroupError::NoParts)?;
        let group = first.command.as_ref().and_then(|c| c.group.clone());
        let first_command = first
            .command
            .clone()
            .ok_or(GroupError::NotCommandable(first.channel))?;
        let mut members = Vec::with_capacity(parts.len());
        let mut ladders = Vec::with_capacity(parts.len());
        let mut queues = Vec::with_capacity(parts.len());
        let mut max_clock_error = first_command.max_clock_error;
        for part in parts {
            let command = part
                .command
                .clone()
                .ok_or(GroupError::NotCommandable(part.channel))?;
            if part.owner != me {
                return Err(GroupError::NotOwner {
                    channel: part.channel,
                    me,
                });
            }
            let same_group = command.group == group && (group.is_some() || members.is_empty());
            if !same_group
                || command.safe_behavior != first_command.safe_behavior
                || command.hold_max != first_command.hold_max
            {
                return Err(GroupError::NotOneGroup(part.channel));
            }
            max_clock_error = max_clock_error.min(command.max_clock_error);
            let check =
                CommandCheck::new(&command).ok_or(GroupError::NotCommandable(part.channel))?;
            let delivery = if command.is_delta() {
                Delivery::Once
            } else {
                Delivery::EveryTick
            };
            ladders.push(MemberLadder {
                hold_last: command.hold_last,
                delivery,
                latch: command.latch,
            });
            queues.push(ActionQueue::new(delivery));
            members.push(Member {
                applied: part
                    .applied_channel
                    .ok_or(GroupError::NotCommandable(part.channel))?,
                check,
                period_ns: duration_i64(part.control_period()).max(1),
                max_action_age_ns: duration_i64(command.max_action_age),
                last: None,
                timing: ChunkTiming {
                    channel: part.channel,
                    ..ChunkTiming::default()
                },
                last_at: None,
                part,
            });
        }
        Ok(Self {
            me,
            group,
            members,
            queues,
            config,
            max_clock_error_ns: u64::try_from(max_clock_error.as_nanos()).unwrap_or(u64::MAX),
            clock_refused: None,
            safe_behavior: Arc::from(first_command.safe_behavior.as_str()),
            fallback: Fallback::new(duration_i64(first_command.hold_max), ladders),
            accepted: 0,
            rejected: [0; Rejection::ALL.len()],
            safe_reports: 0,
            unjudged: None,
            clamped: 0,
            handovers: 0,
        })
    }

    /// The group's parts.
    pub fn parts(&self) -> impl Iterator<Item = &PartSpec> {
        self.members.iter().map(|m| &m.part)
    }

    /// The group's name, if the parts are grouped.
    pub fn group(&self) -> Option<&str> {
        self.group.as_deref()
    }

    /// Queued actions, over every member.
    pub fn queue_depth(&self) -> usize {
        self.queues.iter().map(ActionQueue::len).sum()
    }

    /// Counts a chunk refused before [`Self::accept`] (one that did not
    /// decode).
    pub fn count_rejection(&mut self, reason: Rejection) {
        self.rejected[reason.index()] += 1;
    }

    /// Counts a chunk for member `channel` refused before [`Self::accept`]
    /// (one that did not open or decode): it reached the edge all the same.
    pub fn count_unread(&mut self, channel: ChannelId, reason: Rejection) {
        self.count_rejection(reason);
        if let Some(member) = self.members.iter_mut().find(|m| m.part.channel == channel) {
            member.timing.received += 1;
        }
    }

    /// How each member's chunks arrived, without building [`Self::stats`]:
    /// a board reports these on every heartbeat.
    pub fn timing(&self) -> impl Iterator<Item = &ChunkTiming> {
        self.members.iter().map(|m| &m.timing)
    }

    /// Chunks refused for `reason`, without building [`Self::stats`]: a
    /// board reports these on every heartbeat.
    pub fn rejected(&self, reason: Rejection) -> u64 {
        self.rejected[reason.index()]
    }

    /// Why the clocks are refusing chunks, while the newest judgement did.
    pub fn clock_refused(&self) -> Option<&ClockRefusal> {
        self.clock_refused.as_ref()
    }

    /// The robot finished a handover: the group may dispatch again.
    pub fn handover_done(&mut self) {
        self.fallback.handover_done();
    }

    /// Latches (`Some`, with its code) or clears the robot process's local
    /// e-stop: while latched every chunk is refused
    /// ([`Rejection::Estop`]) and every tick purges the queues and runs
    /// Estop, with or without a host.
    pub fn set_local_estop(&mut self, latch: Option<ReasonCode>) {
        self.fallback.set_latch(latch);
    }

    fn clock<V: AuthorityView + ?Sized>(&self, view: &V, who: &ParticipantId) -> ClockView {
        if view.clock_mode() != ClockMode::Wall {
            return ClockView {
                offset_ns: 0,
                error_ns: 0,
                mapped: true,
                ticked: true,
            };
        }
        match view.clock(who) {
            Some(mapping) => ClockView {
                offset_ns: mapping.offset_ns,
                error_ns: i64::try_from(mapping.error_ns).unwrap_or(i64::MAX),
                mapped: true,
                ticked: false,
            },
            None => ClockView {
                offset_ns: 0,
                error_ns: 0,
                mapped: false,
                ticked: false,
            },
        }
    }

    /// The lease on the group's active member, in the edge clock: the
    /// host's expiry converted conservatively. Unmapped, it is untimeable
    /// unless the mapping is not required, in which case the remaining ttl
    /// at snapshot time counts from when the edge observed the snapshot.
    fn lease_view<V: AuthorityView + ?Sized>(
        &self,
        view: &V,
        observed_local_ns: i64,
    ) -> Option<LeaseView> {
        let (member, lease) = self
            .members
            .iter()
            .enumerate()
            .filter_map(|(i, m)| view.lease(m.part.channel).map(|l| (i, l)))
            .max_by_key(|(_, l)| l.token)?;
        let clock = self.clock(view, &self.me);
        let expires_ns = if clock.ticked {
            Some(i64::MAX)
        } else if clock.mapped {
            Some(
                lease
                    .expires_ns
                    .saturating_sub(clock.offset_ns)
                    .saturating_sub(clock.error_ns),
            )
        } else if self.config.require_clock_mapping {
            None
        } else {
            Some(observed_local_ns.saturating_add(lease.expires_ns.saturating_sub(view.host_ns())))
        };
        Some(LeaseView {
            member,
            token: lease.token,
            expires_ns,
        })
    }

    /// Validates `chunk`, received at `receive_ns` (edge clock), against the
    /// view (observed at `observed_local_ns`) and splices it.
    pub fn accept<V: AuthorityView + ?Sized>(
        &mut self,
        chunk: ActionChunk,
        receive_ns: i64,
        view: Option<&V>,
        observed_local_ns: i64,
    ) -> Result<SpliceOutcome, Rejection> {
        if let Some(member) = self
            .members
            .iter_mut()
            .find(|m| m.part.channel == chunk.channel)
        {
            member.timing.received += 1;
        }
        let result = self.validate_and_splice(chunk, receive_ns, view, observed_local_ns);
        match result {
            Ok(_) => self.accepted += 1,
            Err(reason) => self.rejected[reason.index()] += 1,
        }
        result
    }

    fn validate_and_splice<V: AuthorityView + ?Sized>(
        &mut self,
        chunk: ActionChunk,
        receive_ns: i64,
        view: Option<&V>,
        observed_local_ns: i64,
    ) -> Result<SpliceOutcome, Rejection> {
        let index = self
            .members
            .iter()
            .position(|m| m.part.channel == chunk.channel)
            .ok_or(Rejection::WrongChannel)?;
        if self.fallback.latch().is_some() {
            return Err(Rejection::Estop);
        }
        let view = view.ok_or(Rejection::NoLease)?;
        if view.mode() == Mode::Estop {
            return Err(Rejection::Estop);
        }
        // A closing scene's run is over: no lease in it drives again.
        if view.closing().is_some() {
            return Err(Rejection::NoLease);
        }
        let lease = view.check_fence(chunk.channel, chunk.lease_token, chunk.epoch)?;
        if lease.holder != chunk.source {
            return Err(Rejection::WrongSource);
        }
        if lease.binding != chunk.binding {
            return Err(Rejection::WrongBinding);
        }
        let edge = self.clock(view, &self.me);
        let source = self.clock(view, &chunk.source);
        if !edge.ticked {
            if self.config.require_clock_mapping && !(edge.mapped && source.mapped) {
                return Err(Rejection::NoClockMapping);
            }
            self.clock_refused = ClockRefusal::judge(
                chunk.channel,
                self.max_clock_error_ns,
                [
                    (&self.me, edge.error_ns.unsigned_abs()),
                    (&chunk.source, source.error_ns.unsigned_abs()),
                ],
            );
            if self.clock_refused.is_some() {
                return Err(Rejection::ClockUncertain);
            }
        }
        let expires = self
            .lease_view(view, observed_local_ns)
            .filter(|l| l.member == index)
            .and_then(|l| l.expires_ns)
            .ok_or(Rejection::NoLease)?;
        if receive_ns > expires {
            return Err(Rejection::LeaseExpired);
        }
        let member = &self.members[index];
        if !edge.ticked {
            // Unmapped clocks (allowed only when not required) are one clock.
            let (edge_offset, source_offset) = if edge.mapped && source.mapped {
                (edge.offset_ns, source.offset_ns)
            } else {
                (0, 0)
            };
            let age = receive_ns
                .saturating_add(edge_offset)
                .saturating_sub(chunk.publish_ns.saturating_add(source_offset))
                .saturating_add(edge.error_ns)
                .saturating_add(source.error_ns);
            if age > member.max_action_age_ns {
                return Err(Rejection::TooOld);
            }
        }
        if chunk.count == 0 {
            return Err(Rejection::Empty);
        }
        let count = chunk.count as usize;
        if Some(chunk.actions.len()) != count.checked_mul(member.check.stride()) {
            return Err(Rejection::Malformed);
        }
        let checked = member
            .check
            .check(&chunk.actions, count)
            .map_err(|_| Rejection::InvalidValue)?;
        let valid_until_ns = chunk
            .valid_until_ns
            .saturating_sub(edge.offset_ns)
            .min(receive_ns.saturating_add(member.max_action_age_ns))
            .saturating_sub(edge.error_ns)
            .saturating_sub(source.error_ns);
        if valid_until_ns <= receive_ns {
            return Err(Rejection::Expired);
        }
        // dt = 0 takes the action grid the lease's binding declared,
        // else the part's control period.
        let dt_ns = match i64::try_from(chunk.dt_ns) {
            Ok(0) => view
                .grid(chunk.channel)
                .map_or(member.period_ns, |dt| duration_i64(dt).max(1)),
            Ok(dt) => dt,
            Err(_) => return Err(Rejection::Malformed),
        };
        if let Some((epoch, token, seq)) = member.last
            && epoch == chunk.epoch
            && token == chunk.lease_token
            && chunk.seq <= seq
        {
            return Err(Rejection::StaleSeq);
        }
        let stride = member.check.stride();
        let t0_ns = chunk.t0_ns.saturating_sub(edge.offset_ns);
        let member = &mut self.members[index];
        member.took(
            (chunk.epoch, chunk.lease_token, chunk.seq),
            receive_ns,
            receive_ns.saturating_sub(t0_ns),
        );
        self.clamped += u64::from(checked.clamped);
        let meta = Arc::new(ChunkMeta {
            source: chunk.source,
            epoch: chunk.epoch,
            lease_token: chunk.lease_token,
            seq: chunk.seq,
            based_on: chunk.based_on,
            receive_ns,
            valid_until_ns,
        });
        let actions = (0..count)
            .map(|i| checked.actions.slice(i * stride..(i + 1) * stride))
            .collect();
        Ok(self.queues[index].splice(
            Splice {
                chunk: meta,
                t0_ns,
                dt_ns,
                replace_from_ns: chunk.replace_from_ns.saturating_sub(edge.offset_ns),
                actions,
            },
            receive_ns,
        ))
    }

    /// Whether the runtime is catching up after its own stall
    /// ([`Fallback::set_listening`]).
    pub fn set_listening(&mut self, listening: bool) {
        self.fallback.set_listening(listening);
    }

    /// How long the group waits for a valid action before it goes Safe.
    pub fn hold_max_ns(&self) -> i64 {
        self.fallback.hold_max_ns()
    }

    /// One control tick at edge time `now_ns`: what to dispatch, hold, or
    /// the safe behaviour to run, and one applied sample per member.
    pub fn tick<V: AuthorityView + ?Sized>(
        &mut self,
        now_ns: i64,
        view: Option<&V>,
        observed_local_ns: i64,
    ) -> EdgeTick {
        let tick_view = view.map(|v| TickView {
            epoch: v.epoch(),
            mode: v.mode(),
            lease: self.lease_view(v, observed_local_ns),
            reason: v.mode_reason(),
        });
        let decision = self.fallback.tick(now_ns, tick_view, &mut self.queues);
        let report = self.judge_fall(now_ns, &decision, tick_view, view);
        if report.is_some() {
            self.safe_reports += 1;
        }
        let channel = decision.member.map(|i| self.members[i].part.channel);
        let handover = decision.handover.map(|h| {
            self.handovers += 1;
            GroupHandover {
                group: self.group.clone().unwrap_or_default(),
                from: self.members[h.from].part.channel,
                to: self.members[h.to].part.channel,
            }
        });
        let group_wide =
            matches!(decision.mode, EdgeMode::Safe | EdgeMode::Estop) || decision.member.is_none();
        let applied = self
            .members
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let mine = decision.member == Some(i);
                let (mode, reason) = if mine || group_wide {
                    (decision.mode, decision.reason)
                } else {
                    (EdgeMode::Idle, ReasonCode::Unspecified)
                };
                match decision.action.as_ref().filter(|_| mine) {
                    Some(action) => AppliedSample {
                        stream: m.applied,
                        part: m.part.channel,
                        applied: Applied {
                            mode,
                            present: true,
                            lease_token: action.chunk.lease_token,
                            chunk_seq: action.chunk.seq,
                            action_index: action.index,
                            reason,
                        },
                        value: action.value.clone(),
                    },
                    None => AppliedSample {
                        stream: m.applied,
                        part: m.part.channel,
                        applied: Applied {
                            reason,
                            ..Applied::absent(mode)
                        },
                        value: Bytes::new(),
                    },
                }
            })
            .collect();
        EdgeTick {
            mode: decision.mode,
            channel,
            safe_behavior: decision.run_safe.then(|| self.safe_behavior.clone()),
            action: decision.action,
            report,
            handover,
            applied,
        }
    }

    /// What to tell the host this tick about the group's fall, if anything.
    fn judge_fall<V: AuthorityView + ?Sized>(
        &mut self,
        now_ns: i64,
        decision: &Decision,
        tick_view: Option<TickView>,
        view: Option<&V>,
    ) -> Option<(Epoch, SafeCause)> {
        let tick_view = tick_view?;
        if let Some(cause) = decision.went_safe.filter(|c| c.needs_report()) {
            let lease = tick_view
                .lease
                .filter(|_| matches!(cause, SafeCause::QueueDry | SafeCause::LeaseLapsed));
            let Some(lease) = lease else {
                return Some((tick_view.epoch, cause));
            };
            let ttl = decision
                .member
                .and_then(|i| view?.lease(self.members[i].part.channel))
                .map_or(0, |l| l.ttl_ns);
            self.unjudged = Some(Unjudged {
                epoch: tick_view.epoch,
                cause,
                token: lease.token,
                renewed_by_ns: now_ns.saturating_add_unsigned(ttl),
            });
        }
        let fall = self.unjudged?;
        let Some(lease) = tick_view.lease.filter(|l| l.token == fall.token) else {
            self.unjudged = None;
            return None;
        };
        if lease.expires_ns.is_some_and(|e| e < fall.renewed_by_ns) {
            return None;
        }
        self.unjudged = None;
        Some((fall.epoch, fall.cause))
    }

    /// The counters so far.
    pub fn stats(&self) -> EdgeStats {
        let mut queue = QueueStats::default();
        for q in &self.queues {
            let s = q.stats();
            queue.queued += s.queued;
            queue.dropped_past += s.dropped_past;
            queue.dropped_void += s.dropped_void;
            queue.replaced += s.replaced;
            queue.kept_committed += s.kept_committed;
            queue.fenced += s.fenced;
            queue.skipped += s.skipped;
            queue.expired += s.expired;
            queue.purged += s.purged;
        }
        EdgeStats {
            accepted: self.accepted,
            rejected: Rejection::ALL
                .iter()
                .map(|r| (*r, self.rejected[r.index()]))
                .filter(|(_, n)| *n > 0)
                .collect(),
            queue,
            modes: self.fallback.stats(),
            safe_reports: self.safe_reports,
            clamped: self.clamped,
            handovers: self.handovers,
            clock_refused: self.clock_refused.clone(),
            timing: self.members.iter().map(|m| m.timing).collect(),
        }
    }
}

#[cfg(test)]
mod tests;
