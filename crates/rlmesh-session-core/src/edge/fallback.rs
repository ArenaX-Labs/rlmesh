//! The fallback ladder of one command group, run once per control tick on edge-local time.
//!
//! A group is one or more commandable parts that drive the same actuators (an ungrouped part is
//! a group of one). The host leases at most one member at a time; only that **active** member
//! produces output, and every other member's queue is purged on every tick.
//!
//! - `Idle`: no lease, or nothing applied under it yet. The wait timer is not armed, so a fresh
//!   grant does not trip SAFE before the first chunk; a preempting holder's wait is not timed
//!   from the old holder's last action either.
//! - `Driving`: an action is due and dispatched. A latched part (`latch`) stays here on its last
//!   target once its queue runs out, for as long as the lease that commanded it is held; when
//!   that lease ends, the ladder below takes over, timed from then.
//! - `Hold`: nothing valid is due. The edge waits at most `hold_max` for a valid action. A
//!   position-type part (`hold_last`) repeats its last dispatched target meanwhile, after a
//!   preemption too; any other mode runs the safe behaviour instead (an expired or preempted
//!   velocity or torque command is never continued), and a delta mode dispatches nothing.
//! - `Safe`: the safe behaviour, latched until the epoch moves on (a session re-arm) or the host
//!   grants the group a lease newer than any it had seen when it latched. Entered when the wait
//!   exceeds `hold_max`, when the lease lapses by the edge's clock, when a handover overruns
//!   `hold_max`, or when the host says SAFE. A released (not lapsed) lease still holds and then
//!   runs the safe behaviour, but unlatched: the next lease on the group resumes it.
//! - `Estop`: the robot process latched a local e-stop ([`Fallback::set_latch`]), or the session
//!   is in ESTOP. The local latch works with no network and takes precedence over having no host
//!   view; clearing it leaves the ladder to the host's view (an edge with none stays Idle).
//!
//! **Handover.** When the lease moves from member A to member B, A's queue is purged and the
//! group dispatches nothing until the robot has switched its control mode and neutralised A's
//! target ([`Fallback::handover_done`]); if that takes longer than `hold_max`, the group goes
//! Safe.
//!
//! Every deadline is absolute: the wait is measured from when the queue's coverage actually
//! ended, so after a CPU stall the edge acts on the real elapsed time at its next tick. Behaviour
//! during a full process stall cannot be bounded by the edge itself; that needs an external or
//! device watchdog.
//!
//! **Listening.** A stalled process was not receiving either. A runtime that sees its own ticks
//! stall may say so ([`Fallback::set_listening`]): while it does, a dry queue or a lapsed lease
//! holds as in `Hold` without latching Safe, and nothing drives under a lease the edge cannot see
//! as live.

#[allow(unused_imports)]
use crate::prelude::*;

use core::fmt;

use crate::wire::{EdgeMode, Epoch, LeaseToken, Mode, ReasonCode};

use crate::edge::queue::{ActionQueue, Delivery, QueuedAction};

fn mode_index(mode: EdgeMode) -> usize {
    match mode {
        EdgeMode::Idle => 0,
        EdgeMode::Driving => 1,
        EdgeMode::Hold => 2,
        EdgeMode::Safe => 3,
        EdgeMode::Estop => 4,
    }
}

/// Why the edge went safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SafeCause {
    /// No valid action arrived within `hold_max`.
    QueueDry,
    /// The lease expired by the edge's own clock (the host's renewals stopped reaching it), or
    /// could no longer be timed.
    LeaseLapsed,
    /// The robot's handover between group members took longer than `hold_max`.
    HandoverTimeout,
    /// The host's view says SAFE.
    HostSafe,
    /// The active member's lease was released and nothing replaced it within `hold_max`
    /// (unlatched).
    Released,
}

impl SafeCause {
    /// Causes the host has not seen yet: the edge reports them.
    pub fn needs_report(self) -> bool {
        matches!(
            self,
            Self::QueueDry | Self::LeaseLapsed | Self::HandoverTimeout
        )
    }

    /// Its code on the wire.
    pub fn code(self) -> ReasonCode {
        match self {
            Self::QueueDry => ReasonCode::QueueDry,
            Self::LeaseLapsed => ReasonCode::EdgeLeaseLapsed,
            Self::HandoverTimeout => ReasonCode::HandoverTimeout,
            Self::HostSafe => ReasonCode::Unspecified,
            Self::Released => ReasonCode::Released,
        }
    }
}

impl fmt::Display for SafeCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::QueueDry => "no valid action within hold_max",
            Self::LeaseLapsed => "lease lapsed by the edge clock",
            Self::HandoverTimeout => "group handover longer than hold_max",
            Self::HostSafe => "session SAFE",
            Self::Released => "lease released",
        })
    }
}

/// Entries, ticks and time in one mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModeStats {
    /// Times the mode was entered.
    pub entries: u64,
    /// Ticks spent in it.
    pub ticks: u64,
    /// Edge time from each tick in this mode to the next tick.
    pub time_ns: u64,
}

/// The group's active member and its lease, as the edge's clock sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeaseView {
    /// The leased member (an index into the group).
    pub member: usize,
    /// The current token.
    pub token: LeaseToken,
    /// Its conservative expiry in the edge clock; `None` when it cannot be timed (no clock
    /// mapping).
    pub expires_ns: Option<i64>,
}

/// What the edge's authority view says this tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TickView {
    /// The epoch.
    pub epoch: Epoch,
    /// The mode.
    pub mode: Mode,
    /// The lease on the group's active member, if any.
    pub lease: Option<LeaseView>,
    /// Why the host's mode is what it is (`mode_reason`).
    pub reason: ReasonCode,
}

/// How one member's actions reach the robot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemberLadder {
    /// Whether Hold repeats its last target (position-type modes).
    pub hold_last: bool,
    /// Every tick (absolute) or once per slot (delta).
    pub delivery: Delivery,
    /// Whether its last target stands while its lease is held.
    pub latch: bool,
}

/// The lease moved between members of the group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handover {
    /// The member that held it.
    pub from: usize,
    /// The member that holds it now.
    pub to: usize,
}

/// One tick's decision.
#[derive(Clone, Debug)]
pub struct Decision {
    /// The mode.
    pub mode: EdgeMode,
    /// The member the decision concerns (the active or last-driven one).
    pub member: Option<usize>,
    /// The action to dispatch: the one due (`Driving`; under [`Delivery::Once`] only on its
    /// first tick), or the held target (`Hold`, position-type).
    pub action: Option<QueuedAction>,
    /// Whether to run the safe behaviour this tick.
    pub run_safe: bool,
    /// Set on the tick the edge went safe.
    pub went_safe: Option<SafeCause>,
    /// Set on the tick a handover starts: the robot switches its control mode, then calls
    /// [`Fallback::handover_done`].
    pub handover: Option<Handover>,
    /// Why the edge is in its mode (Safe, Estop): its own cause, or the host's.
    pub reason: ReasonCode,
}

/// The ladder of one group.
#[derive(Debug)]
pub struct Fallback {
    hold_max_ns: i64,
    members: Vec<MemberLadder>,
    mode: EdgeMode,
    safe_epoch: Option<Epoch>,
    safe_token: LeaseToken,
    newest_token: LeaseToken,
    safe_cause: Option<SafeCause>,
    safe_reason: ReasonCode,
    latch: Option<ReasonCode>,
    stopped: bool,
    stop_epoch: Option<Epoch>,
    parked: bool,
    awaiting: Option<LeaseToken>,
    last: Option<(usize, QueuedAction)>,
    dry_since_ns: Option<i64>,
    latched_ns: Option<i64>,
    active: Option<usize>,
    handover: Option<(Handover, i64, bool)>,
    last_tick_ns: Option<i64>,
    listening: bool,
    modes: [ModeStats; 5],
}

impl Fallback {
    /// A ladder starting `Idle` for a group with `members`.
    pub fn new(hold_max_ns: i64, members: Vec<MemberLadder>) -> Self {
        Self {
            hold_max_ns: hold_max_ns.max(0),
            members,
            mode: EdgeMode::Idle,
            safe_epoch: None,
            safe_token: LeaseToken(0),
            newest_token: LeaseToken(0),
            safe_cause: None,
            safe_reason: ReasonCode::Unspecified,
            latch: None,
            stopped: false,
            stop_epoch: None,
            parked: false,
            awaiting: None,
            last: None,
            dry_since_ns: None,
            latched_ns: None,
            active: None,
            handover: None,
            last_tick_ns: None,
            listening: false,
            modes: [ModeStats::default(); 5],
        }
    }

    /// The mode of the last tick.
    pub fn mode(&self) -> EdgeMode {
        self.mode
    }

    /// Why the edge is latched safe, if it is.
    pub fn safe_cause(&self) -> Option<SafeCause> {
        self.safe_cause
    }

    /// Latches (`Some`, with its code) or clears the robot process's local e-stop: from the
    /// next tick every queue is purged and the group runs Estop until it is cleared. A clear
    /// leaves the edge idle until the host's authority moves to a later epoch, so a lease from
    /// before the stop never drives again; a latch cleared before the next tick still purges.
    pub fn set_latch(&mut self, latch: Option<ReasonCode>) {
        self.stopped |= latch.is_some();
        self.latch = latch;
    }

    /// The local e-stop, if latched.
    pub fn latch(&self) -> Option<ReasonCode> {
        self.latch
    }

    /// Entries, ticks and time per mode.
    pub fn stats(&self) -> Vec<(EdgeMode, ModeStats)> {
        EdgeMode::ALL
            .iter()
            .map(|m| (*m, self.modes[mode_index(*m)]))
            .collect()
    }

    /// The robot finished a handover: the group may dispatch again.
    pub fn handover_done(&mut self) {
        self.handover = None;
    }

    /// Whether the runtime is catching up after its own stall (see the module docs): a dry
    /// queue or a lapsed lease then holds instead of latching Safe.
    pub fn set_listening(&mut self, listening: bool) {
        self.listening = listening;
    }

    /// How long the group waits for a valid action before it goes Safe.
    pub fn hold_max_ns(&self) -> i64 {
        self.hold_max_ns
    }

    /// Runs one tick at edge time `now_ns`: purges inactive members, fences and drains the
    /// active member's queue against `view`, and decides what the robot dispatches.
    pub fn tick(
        &mut self,
        now_ns: i64,
        view: Option<TickView>,
        queues: &mut [ActionQueue],
    ) -> Decision {
        if let Some(previous) = self.last_tick_ns {
            let elapsed = u64::try_from(now_ns.saturating_sub(previous)).unwrap_or(0);
            self.modes[mode_index(self.mode)].time_ns += elapsed;
        }
        self.last_tick_ns = Some(now_ns);
        let decision = self.decide(now_ns, view, queues);
        if decision.mode != self.mode {
            self.modes[mode_index(decision.mode)].entries += 1;
            self.mode = decision.mode;
        }
        self.modes[mode_index(decision.mode)].ticks += 1;
        decision
    }

    fn decide(
        &mut self,
        now_ns: i64,
        view: Option<TickView>,
        queues: &mut [ActionQueue],
    ) -> Decision {
        let host_estop = view.filter(|v| v.mode == Mode::Estop).map(|v| v.reason);
        if let Some(reason) = self.latch.or(host_estop) {
            clear(queues);
            self.last = None;
            self.dry_since_ns = None;
            self.handover = None;
            self.parked = false;
            if (self.latch.is_some() || self.stopped)
                && let Some(view) = view
            {
                self.stop_epoch = self.stop_epoch.max(Some(view.epoch));
            }
            let mut decision = self.decision(EdgeMode::Estop, None, None, true);
            decision.reason = reason;
            return decision;
        }
        let Some(view) = view else {
            clear(queues);
            return self.idle(None);
        };
        if self.stopped {
            let fence = *self.stop_epoch.get_or_insert(view.epoch);
            if view.epoch <= fence {
                clear(queues);
                return self.idle(None);
            }
            self.stopped = false;
            self.stop_epoch = None;
        }
        let granted_after = view.lease.is_some_and(|l| l.token > self.safe_token);
        if let Some(lease) = view.lease {
            self.newest_token = self.newest_token.max(lease.token);
        }
        if let Some(epoch) = self.safe_epoch {
            if view.epoch <= epoch && !granted_after {
                clear(queues);
                let mut decision = self.decision(EdgeMode::Safe, None, None, true);
                decision.reason = self.safe_reason;
                return decision;
            }
            self.safe_epoch = None;
            self.safe_cause = None;
        }
        if view.mode == Mode::Safe {
            return self.go_safe(view, SafeCause::HostSafe, queues);
        }
        let active = view.lease.map(|l| l.member);
        for (i, queue) in queues.iter_mut().enumerate() {
            if Some(i) != active {
                queue.clear();
            }
        }
        if let (Some(from), Some(to)) = (self.active, active)
            && from != to
        {
            self.handover = Some((Handover { from, to }, now_ns, false));
            self.last = None;
            self.dry_since_ns = None;
        }
        if active.is_some() {
            self.active = active;
            self.parked = false;
        }
        if let Some((handover, since, announced)) = self.handover.as_mut() {
            let handover = *handover;
            if now_ns.saturating_sub(*since) >= self.hold_max_ns {
                self.handover = None;
                return self.go_safe(view, SafeCause::HandoverTimeout, queues);
            }
            let first = !core::mem::replace(announced, true);
            let mut decision = self.decision(EdgeMode::Hold, Some(handover.to), None, false);
            decision.handover = first.then_some(handover);
            return decision;
        }
        let Some(lease) = view.lease else {
            if self.parked {
                let mut decision = self.decision(EdgeMode::Safe, self.active, None, true);
                decision.reason = SafeCause::Released.code();
                return decision;
            }
            if self.last.is_none() {
                clear(queues);
                return self.idle(self.active);
            }
            return self.hold(now_ns, view, None, queues);
        };
        match lease.expires_ns {
            Some(expires) if now_ns > expires && self.listening && self.last.is_some() => {
                return self.hold(now_ns, view, Some(lease), queues);
            }
            Some(expires) if now_ns > expires => {
                return self.go_safe(view, SafeCause::LeaseLapsed, queues);
            }
            Some(_) => {}
            None if self.last.is_none() => {
                clear(queues);
                return self.idle(Some(lease.member));
            }
            None => return self.go_safe(view, SafeCause::LeaseLapsed, queues),
        }
        if self.last.as_ref().is_some_and(|(member, last)| {
            *member == lease.member && last.chunk.lease_token != lease.token
        }) && self.awaiting != Some(lease.token)
        {
            self.awaiting = Some(lease.token);
            self.dry_since_ns = None;
        }
        let Some(queue) = queues.get_mut(lease.member) else {
            return self.idle(None);
        };
        queue.fence(view.epoch, lease.token);
        if let Some(due) = queue.due(now_ns) {
            self.dry_since_ns = None;
            self.latched_ns = None;
            self.awaiting = None;
            let ladder = self.ladder(lease.member);
            let deliver = ladder.delivery == Delivery::EveryTick || due.first;
            self.last = Some((lease.member, due.action.clone()));
            return self.decision(
                EdgeMode::Driving,
                Some(lease.member),
                deliver.then_some(due.action),
                false,
            );
        }
        if self.last.is_none() {
            return self.decision(EdgeMode::Idle, Some(lease.member), None, false);
        }
        if let Some(latched) = self.latched(now_ns, view.epoch, lease) {
            return latched;
        }
        self.hold(now_ns, view, Some(lease), queues)
    }

    fn latched(&mut self, now_ns: i64, epoch: Epoch, lease: LeaseView) -> Option<Decision> {
        let (member, last) = self.last.clone()?;
        let held = member == lease.member
            && last.chunk.epoch == epoch
            && last.chunk.lease_token == lease.token;
        if !held || !self.ladder(member).latch {
            return None;
        }
        self.dry_since_ns = None;
        self.latched_ns = Some(now_ns);
        Some(self.decision(EdgeMode::Driving, Some(member), Some(last), false))
    }

    fn hold(
        &mut self,
        now_ns: i64,
        view: TickView,
        lease: Option<LeaseView>,
        queues: &mut [ActionQueue],
    ) -> Decision {
        let epoch = view.epoch;
        let Some((member, last)) = self.last.clone() else {
            return self.idle(self.active);
        };
        let current = lease.is_some_and(|l| {
            l.member == member && last.chunk.epoch == epoch && last.chunk.lease_token == l.token
        });
        let mut covered_until = last.end_ns().min(last.chunk.valid_until_ns);
        if let Some(latched) = self.latched_ns {
            covered_until = covered_until.max(latched);
        }
        if !current {
            covered_until = covered_until.min(now_ns);
        }
        let dry_since = *self.dry_since_ns.get_or_insert(covered_until.min(now_ns));
        let new_holder = lease.is_some_and(|l| self.awaiting == Some(l.token));
        if !new_holder && !self.listening && now_ns.saturating_sub(dry_since) >= self.hold_max_ns {
            if lease.is_some() {
                return self.go_safe(view, SafeCause::QueueDry, queues);
            }
            clear(queues);
            self.parked = true;
            self.last = None;
            self.dry_since_ns = None;
            let mut decision = self.decision(EdgeMode::Safe, Some(member), None, true);
            decision.went_safe = Some(SafeCause::Released);
            decision.reason = SafeCause::Released.code();
            return decision;
        }
        let ladder = self.ladder(member);
        let every_tick = ladder.delivery == Delivery::EveryTick;
        let repeat = ladder.hold_last && every_tick;
        self.decision(
            EdgeMode::Hold,
            Some(member),
            repeat.then_some(last),
            !repeat && every_tick,
        )
    }

    fn ladder(&self, member: usize) -> MemberLadder {
        self.members.get(member).copied().unwrap_or(MemberLadder {
            hold_last: false,
            delivery: Delivery::EveryTick,
            latch: false,
        })
    }

    fn decision(
        &self,
        mode: EdgeMode,
        member: Option<usize>,
        action: Option<QueuedAction>,
        run_safe: bool,
    ) -> Decision {
        Decision {
            mode,
            member,
            action,
            run_safe,
            went_safe: None,
            handover: None,
            reason: ReasonCode::Unspecified,
        }
    }

    fn idle(&mut self, member: Option<usize>) -> Decision {
        self.last = None;
        self.dry_since_ns = None;
        self.decision(EdgeMode::Idle, member, None, false)
    }

    fn go_safe(
        &mut self,
        view: TickView,
        cause: SafeCause,
        queues: &mut [ActionQueue],
    ) -> Decision {
        clear(queues);
        self.safe_epoch = Some(view.epoch);
        self.safe_token = self
            .newest_token
            .max(view.lease.map_or(LeaseToken(0), |l| l.token));
        self.safe_cause = Some(cause);
        self.safe_reason = match cause {
            SafeCause::HostSafe => view.reason,
            _ => cause.code(),
        };
        self.last = None;
        self.dry_since_ns = None;
        self.handover = None;
        self.parked = false;
        let mut decision = self.decision(EdgeMode::Safe, self.active, None, true);
        decision.went_safe = Some(cause);
        decision.reason = self.safe_reason;
        decision
    }
}

fn clear(queues: &mut [ActionQueue]) {
    for queue in queues {
        queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use crate::wire::{ChannelId, InputRange, ParticipantId};
    use bytes::Bytes;

    use super::*;
    use crate::edge::queue::{ChunkMeta, Splice};

    const MS: i64 = 1_000_000;

    fn splice(token: u64, t0: i64, n: u8, valid_until: i64) -> Splice {
        Splice {
            chunk: Arc::new(ChunkMeta {
                source: ParticipantId::new("policy").expect("id"),
                epoch: Epoch(1),
                lease_token: LeaseToken(token),
                seq: 1,
                based_on: vec![InputRange::one(ChannelId(1), 1)],
                receive_ns: 0,
                valid_until_ns: valid_until,
            }),
            t0_ns: t0,
            dt_ns: MS,
            replace_from_ns: t0,
            actions: (0..n).map(|i| Bytes::from(vec![i])).collect(),
        }
    }

    fn lease(member: usize, token: u64, expires: i64) -> LeaseView {
        LeaseView {
            member,
            token: LeaseToken(token),
            expires_ns: Some(expires),
        }
    }

    fn view(lease: Option<LeaseView>) -> Option<TickView> {
        Some(TickView {
            epoch: Epoch(1),
            mode: if lease.is_some() {
                Mode::Active
            } else {
                Mode::Armed
            },
            lease,
            reason: ReasonCode::Unspecified,
        })
    }

    fn held(token: u64) -> Option<TickView> {
        view(Some(lease(0, token, i64::MAX)))
    }

    fn ladder(hold_max: i64, hold_last: bool) -> (Fallback, Vec<ActionQueue>) {
        (
            Fallback::new(
                hold_max,
                vec![MemberLadder {
                    hold_last,
                    delivery: Delivery::EveryTick,
                    latch: false,
                }],
            ),
            vec![ActionQueue::default()],
        )
    }

    fn latched(hold_max: i64) -> (Fallback, Vec<ActionQueue>) {
        (
            Fallback::new(
                hold_max,
                vec![MemberLadder {
                    hold_last: true,
                    delivery: Delivery::EveryTick,
                    latch: true,
                }],
            ),
            vec![ActionQueue::default()],
        )
    }

    fn latched_for_two_seconds() -> (Fallback, Vec<ActionQueue>) {
        let (mut f, mut q) = latched(5 * MS);
        q[0].splice(splice(1, 0, 1, 3 * MS), 0);
        assert_eq!(f.tick(0, held(1), &mut q).mode, EdgeMode::Driving);
        for t in [MS, 10 * MS, 500 * MS, 2_000 * MS] {
            let d = f.tick(t, held(1), &mut q);
            assert_eq!(
                (d.mode, d.went_safe, d.run_safe),
                (EdgeMode::Driving, None, false),
                "at {t}"
            );
            assert_eq!(d.action.map(|a| a.index), Some(0), "at {t}");
        }
        (f, q)
    }

    #[test]
    fn a_latched_target_outlives_its_chunk_while_the_lease_is_held() {
        let (f, _) = latched_for_two_seconds();
        let stats: alloc::collections::BTreeMap<_, _> = f.stats().into_iter().collect();
        assert_eq!(stats[&EdgeMode::Hold].ticks, 0, "never waits");
        assert_eq!(stats[&EdgeMode::Safe].entries, 0, "never dry");
    }

    #[test]
    fn a_latched_part_follows_a_newer_target_and_latches_that() {
        let (mut f, mut q) = latched_for_two_seconds();
        let mut next = splice(1, 2_001 * MS, 1, i64::MAX);
        Arc::get_mut(&mut next.chunk).expect("unshared").seq = 2;
        q[0].splice(next, 2_001 * MS);
        let d = f.tick(2_001 * MS, held(1), &mut q);
        assert_eq!(d.action.map(|a| a.chunk.seq), Some(2));
        let d = f.tick(3_000 * MS, held(1), &mut q);
        assert_eq!(d.mode, EdgeMode::Driving);
        assert_eq!(d.action.map(|a| a.chunk.seq), Some(2));
    }

    #[test]
    fn a_released_latch_holds_for_hold_max_from_the_release_then_runs_safe() {
        let (mut f, mut q) = latched_for_two_seconds();
        let d = f.tick(2_001 * MS, view(None), &mut q);
        assert_eq!((d.mode, d.action.is_some()), (EdgeMode::Hold, true));
        let d = f.tick(2_005 * MS, view(None), &mut q);
        assert_eq!(
            (d.mode, d.went_safe, d.run_safe),
            (EdgeMode::Safe, Some(SafeCause::Released), true)
        );
    }

    #[test]
    fn a_lapsed_lease_takes_a_latch_safe_at_once() {
        let (mut f, mut q) = latched_for_two_seconds();
        let expiring = view(Some(lease(0, 1, 2_001 * MS)));
        assert_eq!(f.tick(2_001 * MS, expiring, &mut q).mode, EdgeMode::Driving);
        let d = f.tick(2_002 * MS, expiring, &mut q);
        assert_eq!(
            (d.mode, d.went_safe, d.run_safe),
            (EdgeMode::Safe, Some(SafeCause::LeaseLapsed), true)
        );
    }

    #[test]
    fn a_new_holder_does_not_inherit_a_latch() {
        let (mut f, mut q) = latched_for_two_seconds();
        let d = f.tick(2_001 * MS, held(2), &mut q);
        assert_eq!(d.mode, EdgeMode::Hold, "held as any position part is");
        assert_eq!(d.action.map(|a| a.chunk.lease_token), Some(LeaseToken(1)));
    }

    #[test]
    fn a_position_hold_repeats_the_last_target_then_goes_safe_until_a_new_epoch() {
        let (mut f, mut q) = ladder(5 * MS, true);
        assert_eq!(
            f.tick(0, held(1), &mut q).mode,
            EdgeMode::Idle,
            "nothing yet"
        );
        q[0].splice(splice(1, MS, 3, i64::MAX), 0);
        for t in 1..=3 {
            assert_eq!(f.tick(t * MS, held(1), &mut q).mode, EdgeMode::Driving);
        }
        for t in 4..9 {
            let d = f.tick(t * MS, held(1), &mut q);
            assert_eq!(d.mode, EdgeMode::Hold, "at {t} ms");
            assert_eq!(d.action.map(|a| a.index), Some(2));
            assert!(!d.run_safe);
        }
        let d = f.tick(9 * MS, held(1), &mut q);
        assert_eq!(
            (d.mode, d.went_safe),
            (EdgeMode::Safe, Some(SafeCause::QueueDry))
        );
        q[0].splice(splice(1, 10 * MS, 3, i64::MAX), 0);
        let d = f.tick(10 * MS, held(1), &mut q);
        assert_eq!(
            (d.mode, d.went_safe, d.run_safe),
            (EdgeMode::Safe, None, true)
        );
        let rearmed = Some(TickView {
            epoch: Epoch(2),
            mode: Mode::Active,
            lease: Some(lease(0, 3, i64::MAX)),
            reason: ReasonCode::FirstLease,
        });
        assert_eq!(f.tick(11 * MS, rearmed, &mut q).mode, EdgeMode::Idle);
        let stats: alloc::collections::BTreeMap<_, _> = f.stats().into_iter().collect();
        assert_eq!(stats[&EdgeMode::Hold].ticks, 5);
        assert_eq!(stats[&EdgeMode::Hold].time_ns, 5 * MS as u64);
        assert_eq!(stats[&EdgeMode::Safe].entries, 1);
        assert_eq!(stats[&EdgeMode::Driving].ticks, 3);
    }

    #[test]
    fn a_non_position_hold_runs_the_safe_behaviour_instead_of_repeating() {
        let (mut f, mut q) = ladder(5 * MS, false);
        q[0].splice(splice(1, 0, 1, i64::MAX), 0);
        assert_eq!(f.tick(0, held(1), &mut q).mode, EdgeMode::Driving);
        let d = f.tick(MS, held(1), &mut q);
        assert_eq!(d.mode, EdgeMode::Hold);
        assert!(d.action.is_none() && d.run_safe);
    }

    #[test]
    fn an_expired_target_is_held_only_by_a_position_part() {
        for (hold_last, repeats) in [(true, true), (false, false)] {
            let (mut f, mut q) = ladder(50 * MS, hold_last);
            q[0].splice(splice(1, 0, 10, 3 * MS), 0);
            for t in 0..3 {
                assert_eq!(f.tick(t * MS, held(1), &mut q).mode, EdgeMode::Driving);
            }
            let d = f.tick(4 * MS, held(1), &mut q);
            assert_eq!(d.mode, EdgeMode::Hold);
            assert_eq!(d.action.is_some(), repeats, "hold_last {hold_last}");
            assert_eq!(d.run_safe, !repeats);
            assert_eq!(f.tick(53 * MS, held(1), &mut q).mode, EdgeMode::Safe);
        }
    }

    #[test]
    fn a_preempted_velocity_command_is_never_continued() {
        let (mut f, mut q) = ladder(20 * MS, false);
        q[0].splice(splice(1, 0, 100, i64::MAX), 0);
        assert_eq!(f.tick(0, held(1), &mut q).mode, EdgeMode::Driving);
        let d = f.tick(MS, held(2), &mut q);
        assert_eq!((d.mode, q[0].stats().fenced), (EdgeMode::Hold, 100));
        assert!(d.action.is_none() && d.run_safe);
        q[0].splice(splice(2, 2 * MS, 5, i64::MAX), 2 * MS);
        let d = f.tick(2 * MS, held(2), &mut q);
        assert_eq!(d.mode, EdgeMode::Driving);
        assert_eq!(d.action.map(|a| a.chunk.lease_token), Some(LeaseToken(2)));
        let (mut f, mut q) = ladder(20 * MS, true);
        q[0].splice(splice(1, 0, 100, i64::MAX), 0);
        f.tick(0, held(1), &mut q);
        let d = f.tick(MS, held(2), &mut q);
        assert_eq!(d.mode, EdgeMode::Hold);
        assert_eq!(d.action.map(|a| a.chunk.lease_token), Some(LeaseToken(1)));
    }

    #[test]
    fn a_preempting_holders_wait_is_timed_from_its_own_first_action() {
        let (mut f, mut q) = ladder(20 * MS, true);
        q[0].splice(splice(1, 0, 1, i64::MAX), 0);
        assert_eq!(f.tick(0, held(1), &mut q).mode, EdgeMode::Driving);
        for t in [MS, 30 * MS, 60 * MS] {
            let d = f.tick(t, held(2), &mut q);
            assert_eq!((d.mode, d.went_safe), (EdgeMode::Hold, None), "at {t}");
        }
        q[0].splice(splice(2, 61 * MS, 1, i64::MAX), 61 * MS);
        assert_eq!(f.tick(61 * MS, held(2), &mut q).mode, EdgeMode::Driving);
        let d = f.tick(100 * MS, held(2), &mut q);
        assert_eq!(
            (d.mode, d.went_safe),
            (EdgeMode::Safe, Some(SafeCause::QueueDry))
        );
    }

    #[test]
    fn after_a_stall_the_edge_acts_on_the_real_elapsed_time() {
        let (mut f, mut q) = ladder(100 * MS, true);
        q[0].splice(splice(1, 0, 2, i64::MAX), 0);
        assert_eq!(f.tick(MS, held(1), &mut q).mode, EdgeMode::Driving);
        let d = f.tick(500 * MS, held(1), &mut q);
        assert_eq!(
            (d.mode, d.went_safe),
            (EdgeMode::Safe, Some(SafeCause::QueueDry))
        );
    }

    #[test]
    fn a_listening_edge_holds_through_what_it_has_not_taken_in_yet() {
        let (mut f, mut q) = ladder(100 * MS, true);
        q[0].splice(splice(1, 0, 2, i64::MAX), 0);
        assert_eq!(f.tick(MS, held(1), &mut q).mode, EdgeMode::Driving);
        f.set_listening(true);
        let d = f.tick(500 * MS, held(1), &mut q);
        assert_eq!((d.mode, d.went_safe), (EdgeMode::Hold, None));
        assert!(
            d.action.is_some(),
            "the last position target, as Hold gives"
        );
        f.set_listening(false);
        let d = f.tick(600 * MS, held(1), &mut q);
        assert_eq!(
            (d.mode, d.went_safe),
            (EdgeMode::Safe, Some(SafeCause::QueueDry))
        );

        let (mut f, mut q) = ladder(50 * MS, true);
        q[0].splice(splice(1, 0, 100, i64::MAX), 0);
        let expiring = view(Some(lease(0, 1, 20 * MS)));
        assert_eq!(f.tick(MS, expiring, &mut q).mode, EdgeMode::Driving);
        f.set_listening(true);
        let d = f.tick(21 * MS, expiring, &mut q);
        assert_eq!((d.mode, d.went_safe), (EdgeMode::Hold, None));
        let renewed = view(Some(lease(0, 1, 500 * MS)));
        assert_eq!(f.tick(22 * MS, renewed, &mut q).mode, EdgeMode::Driving);
        f.set_listening(false);
        let d = f.tick(501 * MS, renewed, &mut q);
        assert_eq!(
            (d.mode, d.went_safe),
            (EdgeMode::Safe, Some(SafeCause::LeaseLapsed))
        );
    }

    #[test]
    fn estop_and_a_local_lease_lapse_fall_back_at_once() {
        let (mut f, mut q) = ladder(50 * MS, true);
        q[0].splice(splice(1, 0, 100, i64::MAX), 0);
        let expiring = view(Some(lease(0, 1, 20 * MS)));
        assert_eq!(f.tick(MS, expiring, &mut q).mode, EdgeMode::Driving);
        let d = f.tick(21 * MS, expiring, &mut q);
        assert_eq!(
            (d.mode, d.went_safe),
            (EdgeMode::Safe, Some(SafeCause::LeaseLapsed))
        );
        assert!(q[0].is_empty());

        let (mut f, mut q) = ladder(50 * MS, true);
        q[0].splice(splice(1, 0, 100, i64::MAX), 0);
        assert_eq!(f.tick(MS, expiring, &mut q).mode, EdgeMode::Driving);
        let estop = Some(TickView {
            epoch: Epoch(1),
            mode: Mode::Estop,
            lease: None,
            reason: ReasonCode::EstopRemote,
        });
        let d = f.tick(2 * MS, estop, &mut q);
        assert_eq!(
            (d.mode, d.went_safe, d.run_safe, d.reason),
            (EdgeMode::Estop, None, true, ReasonCode::EstopRemote)
        );
        assert!(q[0].is_empty(), "ESTOP drops the queue");
    }

    #[test]
    fn a_local_latch_stops_with_no_host_view_and_clearing_it_leaves_the_host_in_charge() {
        let (mut f, mut q) = ladder(50 * MS, true);
        q[0].splice(splice(1, 0, 100, i64::MAX), 0);
        assert_eq!(f.tick(MS, held(1), &mut q).mode, EdgeMode::Driving);
        f.set_latch(Some(ReasonCode::EstopLocal));
        let d = f.tick(2 * MS, None, &mut q);
        assert_eq!(
            (d.mode, d.run_safe, d.reason),
            (EdgeMode::Estop, true, ReasonCode::EstopLocal)
        );
        assert!(q[0].is_empty());
        f.set_latch(None);
        assert_eq!(f.tick(3 * MS, None, &mut q).mode, EdgeMode::Idle);
        assert_eq!(f.tick(4 * MS, held(1), &mut q).mode, EdgeMode::Idle);
        q[0].splice(splice(1, 5 * MS, 100, i64::MAX), 0);
        assert_eq!(f.tick(5 * MS, held(1), &mut q).mode, EdgeMode::Idle);
        assert!(q[0].is_empty(), "the fence purges it");
        f.set_latch(Some(ReasonCode::EstopLocal));
        f.set_latch(None);
        q[0].splice(splice(1, 6 * MS, 100, i64::MAX), 0);
        assert_eq!(f.tick(6 * MS, held(1), &mut q).mode, EdgeMode::Idle);
        let rearmed = Some(TickView {
            epoch: Epoch(2),
            ..held(2).expect("view")
        });
        let mut next = splice(2, 7 * MS, 100, i64::MAX);
        Arc::get_mut(&mut next.chunk).expect("unshared").epoch = Epoch(2);
        q[0].splice(next, 0);
        assert_eq!(f.tick(7 * MS, rearmed, &mut q).mode, EdgeMode::Driving);

        let (mut f, mut q) = ladder(50 * MS, true);
        f.set_latch(Some(ReasonCode::EstopLocal));
        f.set_latch(None);
        let estop = Some(TickView {
            epoch: Epoch(1),
            mode: Mode::Estop,
            lease: None,
            reason: ReasonCode::EstopLocal,
        });
        assert_eq!(f.tick(MS, estop, &mut q).mode, EdgeMode::Estop);
        let mut next = splice(2, 7 * MS, 100, i64::MAX);
        Arc::get_mut(&mut next.chunk).expect("unshared").epoch = Epoch(2);
        q[0].splice(next, 0);
        assert_eq!(f.tick(7 * MS, rearmed, &mut q).mode, EdgeMode::Driving);
    }

    #[test]
    fn an_untimeable_lease_idles_before_the_first_action_and_lapses_after() {
        let untimed = view(Some(LeaseView {
            member: 0,
            token: LeaseToken(1),
            expires_ns: None,
        }));
        let (mut f, mut q) = ladder(50 * MS, true);
        assert_eq!(f.tick(0, untimed, &mut q).mode, EdgeMode::Idle);
        q[0].splice(splice(1, 0, 10, i64::MAX), 0);
        assert_eq!(f.tick(MS, held(1), &mut q).mode, EdgeMode::Driving);
        assert_eq!(
            f.tick(2 * MS, untimed, &mut q).went_safe,
            Some(SafeCause::LeaseLapsed)
        );
    }

    #[test]
    fn a_released_lease_holds_then_parks_safe_until_the_next_lease() {
        let (mut f, mut q) = ladder(5 * MS, true);
        q[0].splice(splice(1, 0, 3, i64::MAX), 0);
        assert_eq!(f.tick(0, held(1), &mut q).mode, EdgeMode::Driving);
        let d = f.tick(MS, view(None), &mut q);
        assert_eq!((d.mode, d.action.is_some()), (EdgeMode::Hold, true));
        assert!(q[0].is_empty(), "the released queue is purged");
        let d = f.tick(6 * MS, view(None), &mut q);
        assert_eq!(
            (d.mode, d.went_safe, d.run_safe),
            (EdgeMode::Safe, Some(SafeCause::Released), true)
        );
        assert!(!SafeCause::Released.needs_report());
        assert_eq!(f.tick(7 * MS, view(None), &mut q).mode, EdgeMode::Safe);
        assert_eq!(f.tick(8 * MS, held(2), &mut q).mode, EdgeMode::Idle);
        q[0].splice(splice(2, 9 * MS, 2, i64::MAX), 9 * MS);
        assert_eq!(f.tick(9 * MS, held(2), &mut q).mode, EdgeMode::Driving);
    }

    fn group() -> (Fallback, Vec<ActionQueue>) {
        (
            Fallback::new(
                10 * MS,
                vec![
                    MemberLadder {
                        hold_last: true,
                        delivery: Delivery::EveryTick,
                        latch: false,
                    },
                    MemberLadder {
                        hold_last: false,
                        delivery: Delivery::Once,
                        latch: false,
                    },
                ],
            ),
            vec![ActionQueue::default(), ActionQueue::new(Delivery::Once)],
        )
    }

    #[test]
    fn only_the_leased_member_dispatches_and_a_handover_waits_for_the_robot() {
        let (mut f, mut q) = group();
        q[0].splice(splice(1, 0, 50, i64::MAX), 0);
        q[1].splice(splice(1, 0, 50, i64::MAX), 0);
        let d = f.tick(0, view(Some(lease(0, 1, i64::MAX))), &mut q);
        assert_eq!((d.mode, d.member), (EdgeMode::Driving, Some(0)));
        assert!(q[1].is_empty(), "the inactive member is purged");
        q[1].splice(splice(2, MS, 50, i64::MAX), MS);
        let at_1 = view(Some(lease(1, 2, i64::MAX)));
        let d = f.tick(MS, at_1, &mut q);
        assert_eq!(d.handover, Some(Handover { from: 0, to: 1 }));
        assert_eq!(
            (d.mode, d.action.is_some(), d.run_safe),
            (EdgeMode::Hold, false, false)
        );
        assert!(q[0].is_empty());
        let d = f.tick(2 * MS, at_1, &mut q);
        assert_eq!(
            (d.mode, d.handover),
            (EdgeMode::Hold, None),
            "announced once"
        );
        f.handover_done();
        let d = f.tick(3 * MS, at_1, &mut q);
        assert_eq!((d.mode, d.member), (EdgeMode::Driving, Some(1)));
        let at_0 = view(Some(lease(0, 3, i64::MAX)));
        assert!(f.tick(4 * MS, at_0, &mut q).handover.is_some());
        assert_eq!(f.tick(13 * MS, at_0, &mut q).mode, EdgeMode::Hold);
        let d = f.tick(14 * MS, at_0, &mut q);
        assert_eq!(
            (d.mode, d.went_safe),
            (EdgeMode::Safe, Some(SafeCause::HandoverTimeout))
        );
        assert!(SafeCause::HandoverTimeout.needs_report());
    }

    #[test]
    fn a_group_that_loses_its_member_holds_on_it_then_runs_safe() {
        let (mut f, mut q) = group();
        assert_eq!(
            f.tick(0, view(None), &mut q).mode,
            EdgeMode::Idle,
            "never driven"
        );
        q[0].splice(splice(1, 0, 2, i64::MAX), 0);
        f.tick(0, view(Some(lease(0, 1, i64::MAX))), &mut q);
        let d = f.tick(MS, view(None), &mut q);
        assert_eq!(
            (d.mode, d.member, d.action.is_some()),
            (EdgeMode::Hold, Some(0), true)
        );
        assert_eq!(f.tick(12 * MS, view(None), &mut q).mode, EdgeMode::Safe);
    }
}
