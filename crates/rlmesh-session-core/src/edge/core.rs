//! The edge of one command group, sans-IO: chunk intake, the action queues,
//! the fallback ladder, the applied stream and the applied-action log.
//!
//! A group is the robot's commandable parts that drive the same actuators
//! (`command.group`, §9.7); an ungrouped part is a group of one.
//! [`EdgeCore::accept`] validates each chunk on arrival against the edge's
//! last-known authority view and its own clock, and splices it into its
//! part's queue; the embodiment's control loop calls [`EdgeCore::tick`] once
//! per control period. Neither waits on the network: the edge keeps acting on
//! its last-known lease until that lease expires by its own clock, then
//! falls back.
//!
//! # Intake (§9.2)
//!
//! A chunk is rejected whole when: it is for another part; the session is in
//! ESTOP; the epoch is wrong; the token is not the part's current lease's;
//! its source is not the lease holder; its binding is not the lease's;
//! the lease expired by the edge clock; it is past `valid_until`; it fails
//! the transit check; its seq is not above the last accepted for its token;
//! its payload does not match `count` × the command stride; or any value is
//! invalid ([`crate::edge::command`], §9.6). Valid actions are clamped to the
//! part's limits before they are queued, so the clamped value is what the
//! edge dispatches and what `_applied` shows.
//!
//! # Clock safety (§9.1)
//!
//! Chunk times are session time. With the edge's clock mapping
//! `session = local + θ` within `± ε`:
//!
//! - slot positions (`t0`, `replace_from`) convert as `t − θ`;
//! - **deadlines** (`valid_until`, the lease expiry) convert conservatively,
//!   `t − θ − ε`, so an offset error can only shorten them;
//! - a chunk's life is capped at receipt + the part's `max_action_age`;
//! - the transit check: `receipt_session − publish_session + ε_ctrl +
//!   ε_edge` (the controller's `publish_ns` through its own mapping) above
//!   `max_action_age` rejects the chunk;
//! - with no mapping, or `ε` above [`EdgeConfig::max_clock_error`] (5 ms by
//!   default), chunks are refused and the lease cannot be timed, unless
//!   [`EdgeConfig::require_clock_mapping`] is off (every participant on one
//!   clock).
//!
//! In virtual and lockstep sessions every time is session time and no
//! mapping is needed: the host times leases, and the edge's clock is the
//! tick time it is driven at.

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use core::time::Duration;

use crate::wire::{
    ActionChunk, Applied, AuthoritySnapshot, ChannelId, ChunkRef, ClockMode, EdgeMode, Epoch,
    FenceError, LeaseToken, Mode, PartSpec, ParticipantId,
};
use bytes::Bytes;

use crate::edge::GroupError;
use crate::edge::command::CommandCheck;
use crate::edge::fallback::{Fallback, LeaseView, MemberLadder, ModeStats, SafeCause, TickView};
use crate::edge::queue::{
    ActionQueue, ChunkMeta, Delivery, QueueStats, QueuedAction, Splice, SpliceOutcome,
};

/// Chunks remembered as judged (accepted or rejected), for the virtual-mode
/// barrier.
const JUDGED: usize = 1024;

/// How an edge runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeConfig {
    /// Ticks kept in the applied-action log (oldest dropped first; 0 keeps
    /// none).
    pub log_capacity: usize,
    /// Refuse chunks until this participant (and the chunk's source) has a
    /// clock mapping. Turn off only when every participant shares one clock
    /// (one machine, or PTP with `SystemMonotonicClock`): without a mapping,
    /// times are read as edge time and a delayed snapshot can extend the
    /// lease.
    pub require_clock_mapping: bool,
    /// Refuse chunks while a mapping's error bound exceeds this (§9.1,
    /// `max_clock_error`).
    pub max_clock_error: Duration,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            log_capacity: 4096,
            require_clock_mapping: true,
            max_clock_error: Duration::from_millis(5),
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
    /// The session is in ESTOP.
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
    /// The clock error bound exceeds [`EdgeConfig::max_clock_error`].
    ClockUncertain,
}

impl Rejection {
    /// Every rejection, in stats order.
    pub const ALL: [Self; 16] = [
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
    ];

    fn index(self) -> usize {
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
}

impl EdgeStats {
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

/// One control tick, as the edge logs it: what was dispatched and on whose
/// authority (the action carries its chunk's source, token, seq and
/// `based_on`).
#[derive(Clone, Debug)]
pub struct AppliedAction {
    /// Edge clock at the tick.
    pub at_ns: i64,
    /// The mode.
    pub mode: EdgeMode,
    /// The part the tick concerned.
    pub channel: Option<ChannelId>,
    /// The action dispatched (`Driving`, or a position-type `Hold`).
    pub action: Option<QueuedAction>,
    /// Whether the safe behaviour ran.
    pub ran_safe: bool,
}

/// One sample of a part's applied stream (§9.4).
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
}

/// The edge of one command group.
#[derive(Debug)]
pub struct EdgeCore {
    me: ParticipantId,
    group: Option<String>,
    members: Vec<Member>,
    queues: Vec<ActionQueue>,
    config: EdgeConfig,
    safe_behavior: Arc<str>,
    fallback: Fallback,
    log: VecDeque<AppliedAction>,
    judged: VecDeque<ChunkRef>,
    accepted: u64,
    rejected: [u64; Rejection::ALL.len()],
    safe_reports: u64,
    clamped: u64,
    handovers: u64,
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
                part,
            });
        }
        Ok(Self {
            me,
            group,
            members,
            queues,
            config,
            safe_behavior: Arc::from(first_command.safe_behavior.as_str()),
            fallback: Fallback::new(duration_i64(first_command.hold_max), ladders),
            log: VecDeque::with_capacity(config.log_capacity.min(1 << 16)),
            judged: VecDeque::with_capacity(JUDGED),
            accepted: 0,
            rejected: [0; Rejection::ALL.len()],
            safe_reports: 0,
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

    /// Whether the edge has judged (accepted or rejected) `chunk`: what an
    /// embodiment waits for before stepping in virtual mode (§11.3).
    pub fn has_judged(&self, chunk: &ChunkRef) -> bool {
        self.judged.contains(chunk)
    }

    /// The robot finished a handover: the group may dispatch again.
    pub fn handover_done(&mut self) {
        self.fallback.handover_done();
    }

    fn clock(&self, view: &AuthoritySnapshot, who: &ParticipantId) -> ClockView {
        if view.clock_mode != ClockMode::Wall {
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
    fn lease_view(&self, view: &AuthoritySnapshot, observed_local_ns: i64) -> Option<LeaseView> {
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
            Some(observed_local_ns.saturating_add(lease.expires_ns.saturating_sub(view.host_ns)))
        };
        Some(LeaseView {
            member,
            token: lease.token,
            expires_ns,
        })
    }

    /// Validates `chunk`, received at `receive_ns` (edge clock), against the
    /// view (observed at `observed_local_ns`) and splices it.
    pub fn accept(
        &mut self,
        chunk: ActionChunk,
        receive_ns: i64,
        view: Option<&AuthoritySnapshot>,
        observed_local_ns: i64,
    ) -> Result<SpliceOutcome, Rejection> {
        let reference = chunk.reference();
        let result = self.validate_and_splice(chunk, receive_ns, view, observed_local_ns);
        match result {
            Ok(_) => self.accepted += 1,
            Err(reason) => self.rejected[reason.index()] += 1,
        }
        if result != Err(Rejection::WrongChannel) {
            if self.judged.len() == JUDGED {
                self.judged.pop_front();
            }
            self.judged.push_back(reference);
        }
        result
    }

    fn validate_and_splice(
        &mut self,
        chunk: ActionChunk,
        receive_ns: i64,
        view: Option<&AuthoritySnapshot>,
        observed_local_ns: i64,
    ) -> Result<SpliceOutcome, Rejection> {
        let index = self
            .members
            .iter()
            .position(|m| m.part.channel == chunk.channel)
            .ok_or(Rejection::WrongChannel)?;
        let view = view.ok_or(Rejection::NoLease)?;
        if view.mode == Mode::Estop {
            return Err(Rejection::Estop);
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
        let max_clock_error = duration_i64(self.config.max_clock_error);
        if !edge.ticked {
            if self.config.require_clock_mapping && !(edge.mapped && source.mapped) {
                return Err(Rejection::NoClockMapping);
            }
            if edge.error_ns > max_clock_error || source.error_ns > max_clock_error {
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
            .saturating_sub(edge.error_ns)
            .min(receive_ns.saturating_add(member.max_action_age_ns));
        if valid_until_ns <= receive_ns {
            return Err(Rejection::Expired);
        }
        // dt = 0 takes the action grid the lease's binding declared (§9.2),
        // else the part's control period.
        let dt_ns = match i64::try_from(chunk.dt_ns) {
            Ok(0) => view
                .binding(&lease.holder)
                .filter(|b| b.fingerprint == lease.binding)
                .and_then(|b| b.grid(chunk.channel))
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
        self.members[index].last = Some((chunk.epoch, chunk.lease_token, chunk.seq));
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
                t0_ns: chunk.t0_ns.saturating_sub(edge.offset_ns),
                dt_ns,
                replace_from_ns: chunk.replace_from_ns.saturating_sub(edge.offset_ns),
                actions,
            },
            receive_ns,
        ))
    }

    /// One control tick at edge time `now_ns`: what to dispatch, hold, or
    /// the safe behaviour to run, and one applied sample per member.
    pub fn tick(
        &mut self,
        now_ns: i64,
        view: Option<&AuthoritySnapshot>,
        observed_local_ns: i64,
    ) -> EdgeTick {
        let tick_view = view.map(|v| TickView {
            epoch: v.epoch,
            mode: v.mode,
            lease: self.lease_view(v, observed_local_ns),
        });
        let decision = self.fallback.tick(now_ns, tick_view, &mut self.queues);
        let report = decision
            .went_safe
            .filter(|cause| cause.needs_report())
            .zip(tick_view)
            .map(|(cause, v)| (v.epoch, cause));
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
                let mode = if mine || group_wide {
                    decision.mode
                } else {
                    EdgeMode::Idle
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
                        },
                        value: action.value.clone(),
                    },
                    None => AppliedSample {
                        stream: m.applied,
                        part: m.part.channel,
                        applied: Applied::absent(mode),
                        value: Bytes::new(),
                    },
                }
            })
            .collect();
        if self.config.log_capacity > 0 {
            if self.log.len() >= self.config.log_capacity {
                self.log.pop_front();
            }
            self.log.push_back(AppliedAction {
                at_ns: now_ns,
                mode: decision.mode,
                channel,
                action: decision.action.clone(),
                ran_safe: decision.run_safe,
            });
        }
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
        }
    }

    /// Takes the applied-action log: one entry per tick, oldest first.
    pub fn drain_log(&mut self) -> Vec<AppliedAction> {
        self.log.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::wire::version::{PROTOCOL_MINOR, SPEC_VERSION};
    use crate::wire::{
        ClockMapping, Command, Fingerprint, InputRange, Lease, PriorityClass, Request, Role,
        SessionId,
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
            "grip": {{"kind": "gripper", "hz": 1000, "command": "position", "binary": true}},
            "arm": {{"kind": "joints", "hz": 1000, "count": 2, "fields": ["pos"],
                    "command": "position", "group": "arm", "limits": {{"pos": [-1, 1]}}}},
            "arm/tcp": {{"kind": "x/tcp", "hz": 1000, "fields": {{"dpos": ["float32", [3]]}},
                        "command": "delta", "group": "arm"}}
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
                    },
                );
            }
            Self::run(
                &mut auth,
                "robot",
                Command::DeclareParts {
                    spec_version: SPEC_VERSION,
                    robot_json: robot_map(hold_last),
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

    #[test]
    fn a_preempted_controllers_queued_and_inflight_chunks_are_fenced() {
        let mut f = Fixture::with(EdgeConfig::default(), false, &["cmd"], None);
        let model = f.grant("policy", 0);
        f.publish(0);
        f.accept(f.chunk("policy", &model, 1, 0, 50), 0)
            .expect("accept");
        assert_eq!(f.tick(MS).mode, EdgeMode::Driving);

        let human = f.grant("operator", 2 * MS);
        f.publish(2 * MS);
        // The model's in-flight chunk is fenced; its queued actions go at the
        // next tick, and the edge waits (braking) until the operator
        // delivers.
        assert_eq!(
            f.accept(f.chunk("policy", &model, 2, 2 * MS, 50), 2 * MS),
            Err(Rejection::StaleToken)
        );
        let tick = f.tick(3 * MS);
        assert_eq!(tick.mode, EdgeMode::Hold);
        assert!(tick.action.is_none() && tick.safe_behavior.is_some());
        // A chunk claiming the operator's token but sent by someone else, or
        // under another binding.
        assert_eq!(
            f.accept(f.chunk("policy", &human, 1, 4 * MS, 50), 4 * MS),
            Err(Rejection::WrongSource)
        );
        let mut other = f.chunk("operator", &human, 1, 4 * MS, 50);
        other.binding = Fingerprint::from(1);
        assert_eq!(f.accept(other, 4 * MS), Err(Rejection::WrongBinding));
        f.accept(f.chunk("operator", &human, 1, 4 * MS, 50), 4 * MS)
            .expect("accept");
        let tick = f.tick(4 * MS);
        assert_eq!(tick.mode, EdgeMode::Driving);
        assert_eq!(tick.action.expect("due").chunk.source, pid("operator"));
        let stats = f.edge.stats();
        assert_eq!(stats.accepted, 2);
        assert_eq!(stats.rejected(Rejection::StaleToken), 1);
        assert_eq!(stats.rejected(Rejection::WrongSource), 1);
        assert_eq!(stats.rejected(Rejection::WrongBinding), 1);
        assert_eq!(stats.queue.fenced, 49, "{stats:?}");

        // The log attributes every tick.
        let log = f.edge.drain_log();
        let sources: Vec<_> = log
            .iter()
            .map(|a| {
                (
                    a.mode,
                    a.action.as_ref().map(|q| q.chunk.source.to_string()),
                    a.ran_safe,
                )
            })
            .collect();
        assert_eq!(
            sources,
            vec![
                (EdgeMode::Driving, Some("policy".into()), false),
                (EdgeMode::Hold, None, true),
                (EdgeMode::Driving, Some("operator".into()), false),
            ]
        );
    }

    #[test]
    fn a_chunk_without_dt_takes_the_bindings_action_grid() {
        let mut f = Fixture::new();
        let cmd = f.ch("cmd");
        Fixture::run(
            &mut f.auth,
            "policy",
            Command::DeclareBinding {
                fingerprint: Fingerprint::from(7),
                revision: 2,
                table_json: "{}".into(),
                channels: vec![cmd],
                replan_hz: 0.0,
                contracts: Vec::new(),
                grids: vec![crate::wire::ActionGrid {
                    channel: cmd,
                    dt: Duration::from_millis(10),
                }],
            },
        );
        let lease = f.grant("policy", 0);
        f.publish(0);
        let mut chunk = f.chunk("policy", &lease, 1, 0, 5);
        chunk.dt_ns = 0;
        f.accept(chunk, 0).expect("accept");
        let first = f.tick(9 * MS).action.expect("due");
        assert_eq!((first.index, first.dt_ns), (0, 10 * MS));
        assert_eq!(f.tick(10 * MS).action.expect("due").index, 1);

        // Without a grid for the part, dt = 0 is the control period (1 ms).
        Fixture::run(
            &mut f.auth,
            "policy",
            Command::DeclareBinding {
                fingerprint: Fingerprint::from(8),
                revision: 3,
                table_json: "{}".into(),
                channels: vec![cmd],
                replan_hz: 0.0,
                contracts: Vec::new(),
                grids: Vec::new(),
            },
        );
        let lease = f.grant("policy", 11 * MS);
        f.publish(11 * MS);
        let mut chunk = f.chunk("policy", &lease, 1, 11 * MS, 5);
        chunk.dt_ns = 0;
        f.accept(chunk, 11 * MS).expect("accept");
        assert_eq!(f.tick(12 * MS).action.expect("due").dt_ns, MS);
    }

    #[test]
    fn a_rebinding_fences_the_old_tokens_chunks() {
        let mut f = Fixture::new();
        let lease = f.grant("policy", 0);
        f.publish(0);
        f.accept(f.chunk("policy", &lease, 1, 0, 20), 0)
            .expect("accept");
        assert_eq!(f.tick(0).mode, EdgeMode::Driving);
        let channels = vec![f.ch("cmd")];
        Fixture::run(
            &mut f.auth,
            "policy",
            Command::DeclareBinding {
                fingerprint: Fingerprint::from(10),
                revision: 2,
                table_json: "{}".into(),
                channels,
                replan_hz: 0.0,
                contracts: Vec::new(),
                grids: Vec::new(),
            },
        );
        f.publish(MS);
        let rebound = f.auth.lease(f.ch("cmd")).expect("lease").clone();
        assert!(rebound.token > lease.token);
        assert_eq!(
            f.accept(f.chunk("policy", &lease, 2, MS, 20), MS),
            Err(Rejection::StaleToken),
            "an in-flight chunk of the old binding"
        );
        // The queue of the old token is purged on the next tick, and the new
        // token's chunks drive.
        let tick = f.tick(MS);
        assert_eq!(tick.mode, EdgeMode::Hold, "nothing of the new token yet");
        f.accept(f.chunk("policy", &rebound, 1, 2 * MS, 20), 2 * MS)
            .expect("accept");
        assert_eq!(
            f.tick(2 * MS).action.map(|a| a.chunk.lease_token),
            Some(rebound.token)
        );
        assert!(f.edge.stats().queue.fenced > 0);
    }

    #[test]
    fn seq_is_per_token_and_void_or_invalid_chunks_are_rejected() {
        let mut f = Fixture::new();
        let model = f.grant("policy", 0);
        f.publish(0);
        let mut late = f.chunk("policy", &model, 1, 0, 3);
        late.valid_until_ns = 5 * MS;
        assert_eq!(f.accept(late, 6 * MS), Err(Rejection::Expired));
        let nan = f.chunk_for("cmd", "policy", &model, 1, 0, 3, f32::NAN);
        assert_eq!(f.accept(nan, 0), Err(Rejection::InvalidValue));
        let mut short = f.chunk("policy", &model, 1, 0, 3);
        short.actions = short.actions.slice(..20);
        assert_eq!(f.accept(short, 0), Err(Rejection::Malformed));
        let mut empty = f.chunk("policy", &model, 1, 0, 3);
        empty.count = 0;
        empty.actions = Bytes::new();
        assert_eq!(f.accept(empty, 0), Err(Rejection::Empty));
        for (seq, ok) in [(2, true), (2, false), (1, false)] {
            assert_eq!(
                f.accept(f.chunk("policy", &model, seq, 0, 3), 0).is_ok(),
                ok
            );
        }
        // A new holder's token starts its own seq.
        let human = f.grant("operator", MS);
        f.publish(MS);
        f.accept(f.chunk("operator", &human, 1, MS, 3), MS)
            .expect("accept");
        let stats = f.edge.stats();
        assert_eq!(stats.accepted, 2);
        assert_eq!(stats.rejected(Rejection::StaleSeq), 2);
        assert!(f.edge.has_judged(&ChunkRef {
            channel: f.ch("cmd"),
            lease_token: human.token,
            seq: 1
        }));
    }

    #[test]
    fn clock_conversion_never_extends_a_deadline_and_transit_is_checked() {
        let mut f = Fixture::new();
        // The edge's clock reads 1 s behind session time (offset +1 s),
        // known within 2 ms; edge time 0 is session time 1 s.
        let session_now = 1_000 * MS;
        let lease = f.grant("policy", session_now);
        f.mapping = Some((1_000 * MS, 2 * MS as u64));
        f.publish(session_now);
        let mut chunk = f.chunk("policy", &lease, 1, session_now, 100);
        chunk.valid_until_ns = session_now + 30 * MS;
        chunk.publish_ns = 0;
        f.accept(chunk, 0).expect("accept");
        // valid_until converts to 30 − 2 = 28 ms local.
        assert_eq!(f.tick(27 * MS).mode, EdgeMode::Driving);
        let tick = f.tick(29 * MS);
        assert_eq!(
            tick.mode,
            EdgeMode::Hold,
            "void after the conservative deadline"
        );

        // A chunk that spent longer than max_action_age (200 ms) in transit,
        // with the error bounds added, is refused.
        let mut stale = f.chunk("policy", &lease, 2, session_now, 10);
        stale.publish_ns = -197 * MS;
        assert_eq!(
            f.accept(stale, 0),
            Err(Rejection::TooOld),
            "197 + 2 + 2 > 200"
        );
        let mut fresh = f.chunk("policy", &lease, 3, session_now, 10);
        fresh.publish_ns = -195 * MS;
        f.accept(fresh, 0).expect("195 + 4 <= 200");

        // The lease (granted at session 1 s with a 100 ms grace) expires at
        // 100 ms local, less the error bound.
        assert_eq!(
            f.accept(f.chunk("policy", &lease, 4, session_now, 10), 99 * MS),
            Err(Rejection::LeaseExpired)
        );

        // An uncertain or missing mapping refuses chunks (5 ms by default).
        let mut f = Fixture::new();
        let lease = f.grant("policy", 0);
        f.mapping = Some((0, 6 * MS as u64));
        f.publish(0);
        assert_eq!(
            f.accept(f.chunk("policy", &lease, 1, 0, 10), 0),
            Err(Rejection::ClockUncertain)
        );
        f.mapping = None;
        f.publish(0);
        assert_eq!(
            f.accept(f.chunk("policy", &lease, 2, 0, 10), 0),
            Err(Rejection::NoClockMapping)
        );
        assert_eq!(
            f.tick(0).mode,
            EdgeMode::Idle,
            "untimed and nothing applied"
        );
    }

    #[test]
    fn the_lease_lapses_by_the_edge_clock_alone_and_the_edge_reports_safe() {
        let mut f = Fixture::new();
        let lease = f.grant("policy", 0);
        // One snapshot, then the host goes quiet: no renewal ever arrives.
        f.publish(0);
        f.accept(f.chunk("policy", &lease, 1, 0, 140), 0)
            .expect("accept");
        assert_eq!(f.tick(MS).mode, EdgeMode::Driving);
        let tick = f.tick(GRACE + MS);
        assert_eq!(tick.mode, EdgeMode::Safe, "safe at once, no hold");
        assert_eq!(tick.safe_behavior.as_deref(), Some("brake"));
        let (epoch, cause) = tick.report.expect("a report for the host");
        assert_eq!((epoch, cause), (f.auth.epoch(), SafeCause::LeaseLapsed));
        let mut late = f.chunk("policy", &lease, 2, GRACE + 2 * MS, 10);
        late.publish_ns = GRACE + 2 * MS;
        assert_eq!(f.accept(late, GRACE + 2 * MS), Err(Rejection::LeaseExpired));
    }

    #[test]
    fn a_dry_queue_holds_then_goes_safe_and_estop_is_immediate() {
        let mut f = Fixture::new();
        let lease = f.grant("policy", 0);
        f.publish(0);
        f.accept(f.chunk("policy", &lease, 1, 0, 5), 0)
            .expect("accept");
        let ticks: Vec<EdgeTick> = (0..20).map(|t| f.tick(t * MS)).collect();
        let count = |m| ticks.iter().filter(|t| t.mode == m).count();
        assert_eq!(count(EdgeMode::Driving), 5);
        assert_eq!(count(EdgeMode::Hold), 10, "hold_max is 10 ms");
        assert_eq!(count(EdgeMode::Safe), 5);
        assert_eq!(f.edge.stats().safe_reports, 1);
        assert_eq!(
            ticks.iter().filter_map(|t| t.report).collect::<Vec<_>>(),
            vec![(f.auth.epoch(), SafeCause::QueueDry)]
        );

        let mut f = Fixture::new();
        let lease = f.grant("policy", 0);
        f.publish(0);
        f.accept(f.chunk("policy", &lease, 1, 0, 50), 0)
            .expect("accept");
        assert_eq!(f.tick(0).mode, EdgeMode::Driving);
        Fixture::run(
            &mut f.auth,
            "operator",
            Command::Estop {
                reason: "test".into(),
            },
        );
        f.publish(MS);
        let tick = f.tick(MS);
        assert_eq!(tick.mode, EdgeMode::Estop);
        assert_eq!(tick.safe_behavior.as_deref(), Some("brake"));
        assert!(tick.report.is_none(), "the host already knows");
        assert_eq!(
            f.accept(f.chunk("policy", &lease, 2, MS, 50), MS),
            Err(Rejection::Estop)
        );
    }

    #[test]
    fn the_applied_stream_shows_what_was_dispatched_each_tick() {
        let mut f = Fixture::new();
        let lease = f.grant("policy", 0);
        f.publish(0);
        let applied_id = f
            .auth
            .part(f.ch("cmd"))
            .and_then(|p| p.applied_channel)
            .expect("applied");
        let tick = f.tick(0);
        assert_eq!(tick.applied.len(), 1);
        assert_eq!(tick.applied[0].stream, applied_id);
        assert_eq!(tick.applied[0].applied, Applied::absent(EdgeMode::Idle));
        f.accept(f.chunk_for("cmd", "policy", &lease, 7, MS, 2, 0.5), MS)
            .expect("accept");
        let tick = f.tick(2 * MS);
        let sample = &tick.applied[0];
        assert_eq!(
            sample.applied,
            Applied {
                mode: EdgeMode::Driving,
                present: true,
                lease_token: lease.token,
                chunk_seq: 7,
                action_index: 1,
            }
        );
        assert_eq!(&sample.value[..4], &0.5_f32.to_le_bytes());
        // Hold (position): the held value, mode HOLD, present.
        let tick = f.tick(4 * MS);
        assert_eq!(tick.applied[0].applied.mode, EdgeMode::Hold);
        assert!(tick.applied[0].applied.present);
        // Safe: nothing.
        let tick = f.tick(20 * MS);
        assert_eq!(tick.applied[0].applied, Applied::absent(EdgeMode::Safe));
        assert!(tick.applied[0].value.is_empty());
    }

    #[test]
    fn limits_and_the_binary_snap_apply_before_queueing() {
        let mut f = Fixture::with(EdgeConfig::default(), true, &["grip"], None);
        let lease = f.grant_with("policy", "grip", &[], 0);
        f.publish(0);
        f.accept(f.chunk_for("grip", "policy", &lease, 1, 0, 2, 0.8), 0)
            .expect("accept");
        let tick = f.tick(0);
        assert_eq!(
            &tick.action.expect("due").value[..4],
            &1.0_f32.to_le_bytes()
        );
        assert_eq!(&tick.applied[0].value[..4], &1.0_f32.to_le_bytes());
        assert_eq!(f.edge.stats().clamped, 2);
    }

    #[test]
    fn a_group_dispatches_only_its_leased_member_and_hands_over() {
        let mut f = Fixture::with(EdgeConfig::default(), true, &["arm", "arm/tcp"], None);
        assert_eq!(f.edge.group(), Some("arm"));
        let joint = f.grant_with("policy", "arm", &[], 0);
        f.publish(0);
        f.accept(f.chunk_for("arm", "policy", &joint, 1, 0, 50, 3.0), 0)
            .expect("accept");
        let tick = f.tick(0);
        assert_eq!(tick.channel, Some(f.ch("arm")));
        assert_eq!(
            &tick.action.expect("due").value[..4],
            &1.0_f32.to_le_bytes(),
            "clamped to the arm's limit"
        );
        assert_eq!(tick.applied.len(), 2);
        assert_eq!(tick.applied[1].applied, Applied::absent(EdgeMode::Idle));

        // The model switches to the Cartesian member in one revision.
        let tcp = f.grant_with("policy", "arm/tcp", &["arm"], MS);
        f.publish(MS);
        f.accept(
            f.chunk_for("arm/tcp", "policy", &tcp, 1, 2 * MS, 5, 0.01),
            MS,
        )
        .expect("accept");
        assert_eq!(
            f.accept(f.chunk_for("arm", "policy", &joint, 2, MS, 5, 0.0), MS),
            Err(Rejection::NoLease)
        );
        let tick = f.tick(MS);
        let handover = tick.handover.expect("a handover");
        assert_eq!((handover.from, handover.to), (f.ch("arm"), f.ch("arm/tcp")));
        assert!(tick.action.is_none() && tick.safe_behavior.is_none());
        f.edge.handover_done();
        let tick = f.tick(2 * MS);
        assert_eq!(
            (tick.mode, tick.channel),
            (EdgeMode::Driving, Some(f.ch("arm/tcp")))
        );
        assert!(tick.action.is_some(), "a delta's first tick in its slot");
        assert!(f.tick(2 * MS + MS / 2).action.is_none(), "dispatched once");
        assert_eq!(f.edge.stats().handovers, 1);
    }

    #[test]
    fn a_virtual_session_runs_the_edge_on_session_ticks() {
        let mut f = Fixture::with(
            EdgeConfig::default(),
            true,
            &["cmd"],
            Some(VirtualConfig::default()),
        );
        let lease = f.grant("policy", 0);
        f.mapping = None;
        f.publish(0);
        // No mapping is needed and the lease is timed by the host: a chunk
        // received at tick 1000 (1 s) with an old publish stamp is fine.
        let mut chunk = f.chunk("policy", &lease, 1, 1_000 * MS, 5);
        chunk.publish_ns = -10_000 * MS;
        f.accept(chunk, 1_000 * MS).expect("accept");
        assert_eq!(f.tick(1_002 * MS).mode, EdgeMode::Driving);
    }

    #[test]
    fn only_commandable_parts_of_one_group_and_owner_get_an_edge() {
        let f = Fixture::new();
        let part = |path: &str| {
            f.auth
                .parts()
                .find(|p| p.path.as_str() == path)
                .cloned()
                .expect(path)
        };
        let mut sense = part("cmd");
        sense.command = None;
        assert!(matches!(
            EdgeCore::new(vec![sense], pid("robot"), EdgeConfig::default()),
            Err(GroupError::NotCommandable(_))
        ));
        assert!(matches!(
            EdgeCore::new(vec![part("cmd")], pid("other"), EdgeConfig::default()),
            Err(GroupError::NotOwner { .. })
        ));
        assert!(matches!(
            EdgeCore::new(
                vec![part("cmd"), part("grip")],
                pid("robot"),
                EdgeConfig::default()
            ),
            Err(GroupError::NotOneGroup(_))
        ));
        assert!(matches!(
            EdgeCore::new(Vec::new(), pid("robot"), EdgeConfig::default()),
            Err(GroupError::NoParts)
        ));
    }
}
