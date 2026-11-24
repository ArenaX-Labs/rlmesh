//! The edge's timed action queue, sans-IO.
//!
//! Every time here is the edge's own clock, in nanoseconds: the intake converts a chunk's
//! session times before it splices, so the queue keeps working with the host unreachable.
//!
//! Each queued action owns the slot `[slot_ns, slot_ns + dt_ns)`. A new chunk is spliced under
//! the `replace` merge policy:
//!
//! - actions of any other `(epoch, lease token)` are purged (the fence);
//! - queued actions whose slot starts at or after `replace_from` are replaced; the committed ones
//!   before it are kept;
//! - the chunk's own actions before `replace_from` only fill slots no committed action covers;
//! - actions whose slot already ended on arrival are dropped, never shifted forward, and so are
//!   actions at or past the chunk's (clamped) `valid_until`.
//!
//! A committed action that straddles `replace_from` runs to the end of its slot or until a
//! replacing action starts, whichever is first; a gap of less than one period to the next queued
//! action is bridged by the earlier action. Longer gaps are dry.
//!
//! Under [`Delivery::Once`] (delta commands) an action is delivered on the first tick of its slot
//! and never again: a delivered slot stays committed whatever a later `replace_from` says, slots
//! that ended before any tick delivered them are dropped rather than replayed, and nothing is
//! bridged.

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::VecDeque;
use alloc::sync::Arc;

use crate::wire::{Epoch, InputRange, LeaseToken, ParticipantId};
use bytes::Bytes;

/// How each action reaches the embodiment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Delivery {
    /// Every tick of the slot applies the action (absolute targets).
    #[default]
    EveryTick,
    /// Only the first tick of the slot delivers it (delta commands): later ticks of the slot,
    /// and holds, deliver nothing.
    Once,
}

/// What every action of one accepted chunk shares: its provenance.
#[derive(Debug, PartialEq, Eq)]
pub struct ChunkMeta {
    /// The controller that sent it (checked against the lease holder).
    pub source: ParticipantId,
    /// The epoch it was accepted in.
    pub epoch: Epoch,
    /// Its lease token.
    pub lease_token: LeaseToken,
    /// Its seq under that token.
    pub seq: u64,
    /// The inputs it was computed from.
    pub based_on: Vec<InputRange>,
    /// Edge clock at receipt.
    pub receive_ns: i64,
    /// `valid_until` in the edge clock, clamped to receipt + `max_action_age`, both less the
    /// error of the two clocks.
    pub valid_until_ns: i64,
}

/// One timed action in the queue.
#[derive(Clone, Debug)]
pub struct QueuedAction {
    /// Its chunk.
    pub chunk: Arc<ChunkMeta>,
    /// Its index in the chunk.
    pub index: u32,
    /// Edge clock at the start of its slot.
    pub slot_ns: i64,
    /// Slot length.
    pub dt_ns: i64,
    /// The packed action (after validation and clamping), a slice of the chunk's payload when
    /// nothing was clamped.
    pub value: Bytes,
    delivered: bool,
}

impl QueuedAction {
    /// Edge clock at the end of its slot.
    pub fn end_ns(&self) -> i64 {
        self.slot_ns.saturating_add(self.dt_ns)
    }

    /// Whether a tick has delivered it.
    pub fn delivered(&self) -> bool {
        self.delivered
    }

    fn overlaps(&self, start: i64, end: i64) -> bool {
        self.slot_ns < end && start < self.end_ns()
    }
}

/// A chunk to splice, its times already in the edge clock.
#[derive(Debug)]
pub struct Splice {
    /// Its provenance.
    pub chunk: Arc<ChunkMeta>,
    /// Slot start of the first action.
    pub t0_ns: i64,
    /// Slot length (positive).
    pub dt_ns: i64,
    /// Replace queued actions from here.
    pub replace_from_ns: i64,
    /// The packed actions, in order.
    pub actions: Vec<Bytes>,
}

/// What one splice did, in actions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpliceOutcome {
    /// The chunk's actions queued.
    pub queued: u64,
    /// The chunk's actions whose slot had already ended on arrival.
    pub dropped_past: u64,
    /// The chunk's actions at or past its `valid_until`.
    pub dropped_void: u64,
    /// Queued actions at or after `replace_from` that the chunk replaced.
    pub replaced: u64,
    /// The chunk's actions whose slot a committed action already covered.
    pub kept_committed: u64,
    /// Queued actions of another epoch or lease token, purged.
    pub fenced: u64,
    /// The chunk's actions dropped because the heap had no room for them now: the queue is as
    /// the fence left it.
    pub no_memory: u64,
}

/// Running action counts for one queue.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueueStats {
    /// Actions queued.
    pub queued: u64,
    /// Dropped: their slot ended before arrival.
    pub dropped_past: u64,
    /// Dropped: at or past `valid_until` on arrival.
    pub dropped_void: u64,
    /// Replaced by a later chunk.
    pub replaced: u64,
    /// Not queued: a committed action covered their slot.
    pub kept_committed: u64,
    /// Purged by the fence.
    pub fenced: u64,
    /// Slots that ended before any tick delivered them (a missed tick, or an overlapping
    /// replacement).
    pub skipped: u64,
    /// Reached `valid_until` while queued.
    pub expired: u64,
    /// Dropped by a fallback (Safe, Estop) or because their command group switched to another
    /// member.
    pub purged: u64,
}

impl QueueStats {
    fn add(&mut self, o: SpliceOutcome) {
        self.queued += o.queued;
        self.dropped_past += o.dropped_past;
        self.dropped_void += o.dropped_void;
        self.replaced += o.replaced;
        self.kept_committed += o.kept_committed;
    }
}

/// What is due at one instant.
#[derive(Clone, Debug)]
pub struct Due {
    /// The action whose slot contains the instant.
    pub action: QueuedAction,
    /// Whether this is its first delivery.
    pub first: bool,
}

/// Timed actions in slot order.
#[derive(Debug, Default)]
pub struct ActionQueue {
    items: VecDeque<QueuedAction>,
    delivery: Delivery,
    stats: QueueStats,
}

impl ActionQueue {
    /// An empty queue delivering as `delivery` says.
    pub fn new(delivery: Delivery) -> Self {
        Self {
            items: VecDeque::new(),
            delivery,
            stats: QueueStats::default(),
        }
    }

    /// Queued actions.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether nothing is queued.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The counters so far.
    pub fn stats(&self) -> QueueStats {
        self.stats
    }

    /// The end of the last queued slot: how far ahead the queue reaches.
    pub fn horizon_ns(&self) -> Option<i64> {
        self.items.iter().map(QueuedAction::end_ns).max()
    }

    /// Purges every action not under `(epoch, token)`; returns how many.
    pub fn fence(&mut self, epoch: Epoch, token: LeaseToken) -> u64 {
        let before = self.items.len();
        self.items
            .retain(|a| a.chunk.epoch == epoch && a.chunk.lease_token == token);
        let fenced = (before - self.items.len()) as u64;
        self.stats.fenced += fenced;
        fenced
    }

    /// Drops everything (ESTOP, SAFE, an inactive group member); returns how many.
    pub fn clear(&mut self) -> u64 {
        let cleared = self.items.len() as u64;
        self.items.clear();
        self.stats.purged += cleared;
        cleared
    }

    /// Splices a chunk under the `replace` policy at edge time `now_ns`.
    ///
    /// The chunk's epoch and lease token fence the queue first, which may purge it. Then the
    /// chunk's actions whose slot already ended, or that start at or past its `valid_until`,
    /// are dropped, and the queued suffix is replaced only from where the first surviving
    /// action starts (or `replace_from`, if later). A chunk with nothing left leaves the
    /// suffix unchanged, never a gap.
    pub fn splice(&mut self, splice: Splice, now_ns: i64) -> SpliceOutcome {
        let mut outcome = SpliceOutcome {
            fenced: self.fence(splice.chunk.epoch, splice.chunk.lease_token),
            ..SpliceOutcome::default()
        };
        let dt = splice.dt_ns.max(1);
        let n = splice.actions.len();
        let mut candidates = Vec::new();
        let mut fresh = Vec::new();
        if candidates.try_reserve_exact(n).is_err()
            || fresh.try_reserve_exact(n).is_err()
            || self.items.try_reserve(n).is_err()
        {
            outcome.no_memory = n as u64;
            self.stats.add(outcome);
            return outcome;
        }
        for (i, value) in splice.actions.into_iter().enumerate() {
            let i = i64::try_from(i).unwrap_or(i64::MAX);
            let slot_ns = splice.t0_ns.saturating_add(i.saturating_mul(dt));
            if slot_ns.saturating_add(dt) <= now_ns {
                outcome.dropped_past += 1;
            } else if slot_ns >= splice.chunk.valid_until_ns {
                outcome.dropped_void += 1;
            } else {
                candidates.push((i, slot_ns, value));
            }
        }
        let Some(first_ns) = candidates.first().map(|(_, slot, _)| *slot) else {
            self.stats.add(outcome);
            return outcome;
        };
        let replace_from_ns = splice.replace_from_ns.max(first_ns);
        let once = self.delivery == Delivery::Once;
        let before = self.items.len();
        self.items
            .retain(|a| a.slot_ns < replace_from_ns || (once && a.delivered));
        outcome.replaced = (before - self.items.len()) as u64;

        for (i, slot_ns, value) in candidates {
            let end_ns = slot_ns.saturating_add(dt);
            if self.items.iter().any(|a| {
                a.overlaps(slot_ns, end_ns) && (slot_ns < replace_from_ns || (once && a.delivered))
            }) {
                outcome.kept_committed += 1;
            } else {
                fresh.push(QueuedAction {
                    chunk: splice.chunk.clone(),
                    index: u32::try_from(i).unwrap_or(u32::MAX),
                    slot_ns,
                    dt_ns: dt,
                    value,
                    delivered: false,
                });
            }
        }
        outcome.queued = fresh.len() as u64;
        self.items.extend(fresh);
        // Stable: kept actions start before replace_from, so a committed action stays ahead of a
        // replacement starting in the same slot.
        self.items.make_contiguous().sort_by_key(|a| a.slot_ns);
        self.stats.add(outcome);
        outcome
    }

    /// The action due at `now_ns`: the latest-starting one whose slot contains now, after
    /// dropping those whose slot ended or that went void.
    pub fn due(&mut self, now_ns: i64) -> Option<Due> {
        let bridge = self.delivery == Delivery::EveryTick;
        loop {
            let front = self.items.front()?;
            let next = self.items.get(1);
            let bridged = bridge
                && next.is_some_and(|next| {
                    now_ns < next.slot_ns
                        && next.slot_ns.saturating_sub(front.end_ns()) < front.dt_ns
                });
            let ended = front.end_ns() <= now_ns && !bridged;
            let void = now_ns >= front.chunk.valid_until_ns;
            let superseded = next.is_some_and(|next| next.slot_ns <= now_ns);
            if !(ended || void || superseded) {
                break;
            }
            if let Some(dropped) = self.items.pop_front() {
                if void && dropped.end_ns() > now_ns {
                    self.stats.expired += 1;
                } else if !dropped.delivered {
                    self.stats.skipped += 1;
                }
            }
        }
        let front = self.items.front_mut()?;
        if front.slot_ns > now_ns {
            return None;
        }
        let first = !front.delivered;
        front.delivered = true;
        Some(Due {
            action: front.clone(),
            first,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::ChannelId;

    const MS: i64 = 1_000_000;

    fn meta(token: u64, seq: u64, valid_until_ns: i64) -> Arc<ChunkMeta> {
        Arc::new(ChunkMeta {
            source: ParticipantId::new(if token == 1 { "policy" } else { "operator" }).expect("id"),
            epoch: Epoch(1),
            lease_token: LeaseToken(token),
            seq,
            based_on: vec![InputRange::one(ChannelId(1), seq)],
            receive_ns: 0,
            valid_until_ns,
        })
    }

    fn splice(token: u64, seq: u64, t0: i64, n: u8, replace_from: i64, tag: u8) -> Splice {
        Splice {
            chunk: meta(token, seq, i64::MAX),
            t0_ns: t0,
            dt_ns: MS,
            replace_from_ns: replace_from,
            actions: (0..n)
                .map(|i| Bytes::from(vec![tag.wrapping_add(i)]))
                .collect(),
        }
    }

    fn tag(due: &Due) -> u8 {
        due.action.value[0]
    }

    fn tags(q: &mut ActionQueue, ticks: core::ops::Range<i64>) -> Vec<u8> {
        ticks
            .filter_map(|t| q.due(t * MS).map(|d| tag(&d)))
            .collect()
    }

    #[test]
    fn a_splice_keeps_the_committed_prefix_and_replaces_after_replace_from() {
        let mut q = ActionQueue::default();
        q.splice(splice(1, 1, 0, 10, 0, 0), 0);
        let o = q.splice(splice(1, 2, 3 * MS, 10, 5 * MS, 100), 0);
        assert_eq!((o.replaced, o.kept_committed, o.queued), (5, 2, 8), "{o:?}");
        assert_eq!(
            tags(&mut q, 0..13),
            vec![0, 1, 2, 3, 4, 102, 103, 104, 105, 106, 107, 108, 109]
        );
    }

    #[test]
    fn a_late_prefix_is_dropped_never_shifted_forward() {
        let mut q = ActionQueue::default();
        let o = q.splice(splice(1, 1, 0, 10, 0, 0), 7 * MS + MS / 2);
        assert_eq!((o.dropped_past, o.queued), (7, 3));
        let due = q.due(7 * MS + MS / 2).expect("due");
        assert_eq!(
            (due.action.index, tag(&due)),
            (7, 7),
            "the action of now's slot"
        );
    }

    #[test]
    fn a_gap_is_dry_and_the_chunk_prefix_fills_uncovered_slots() {
        let mut q = ActionQueue::default();
        q.splice(splice(1, 1, 0, 3, 0, 0), 0);
        q.splice(splice(1, 2, 6 * MS, 3, 4 * MS, 50), 0);
        assert_eq!(q.due(2 * MS).map(|d| tag(&d)), Some(2));
        assert!(q.due(4 * MS).is_none(), "a gap is dry");
        assert_eq!(q.due(6 * MS).map(|d| tag(&d)), Some(50));
        let mut q = ActionQueue::default();
        q.splice(splice(1, 1, 0, 2, 0, 0), 0);
        let o = q.splice(splice(1, 2, 0, 6, 4 * MS, 10), 0);
        assert_eq!((o.kept_committed, o.queued), (2, 4));
        assert_eq!(tags(&mut q, 0..6), vec![0, 1, 12, 13, 14, 15]);
    }

    #[test]
    fn a_sub_slot_gap_between_chunk_grids_is_bridged() {
        let mut q = ActionQueue::default();
        q.splice(splice(1, 1, 0, 3, 0, 0), 0);
        q.splice(
            splice(1, 2, 3 * MS + 4 * MS / 10, 3, 2 * MS + MS / 2, 10),
            0,
        );
        assert_eq!(q.due(3 * MS + MS / 10).map(|d| tag(&d)), Some(2));
        assert_eq!(q.due(3 * MS + 5 * MS / 10).map(|d| tag(&d)), Some(10));
        let mut q = ActionQueue::default();
        q.splice(splice(1, 1, 0, 3, 0, 0), 0);
        q.splice(splice(1, 2, 4 * MS, 3, 3 * MS, 10), 0);
        assert!(q.due(3 * MS + MS / 2).is_none());
    }

    #[test]
    fn actions_expire_at_valid_until_and_missed_slots_are_counted() {
        let mut q = ActionQueue::default();
        let mut s = splice(1, 1, 0, 10, 0, 0);
        s.chunk = meta(1, 1, 3 * MS + MS / 2);
        let o = q.splice(s, 0);
        assert_eq!((o.queued, o.dropped_void), (4, 6));
        assert!(q.due(0).is_some());
        assert_eq!(q.due(2 * MS).map(|d| d.action.index), Some(2));
        assert!(q.due(3 * MS + 6 * MS / 10).is_none());
        let stats = q.stats();
        assert_eq!((stats.skipped, stats.expired), (1, 1), "{stats:?}");
    }

    #[test]
    fn a_new_token_or_epoch_fences_the_old_queue() {
        let mut q = ActionQueue::default();
        q.splice(splice(1, 1, 0, 10, 0, 0), 0);
        assert_eq!(q.fence(Epoch(1), LeaseToken(2)), 10);
        q.splice(splice(1, 2, 0, 10, 0, 0), 0);
        let o = q.splice(splice(2, 1, 5 * MS, 5, 5 * MS, 0), 0);
        assert_eq!((o.fenced, o.queued), (10, 5));
        assert_eq!(q.stats().fenced, 20);
        assert_eq!(
            q.due(5 * MS).map(|d| d.action.chunk.lease_token),
            Some(LeaseToken(2))
        );
        assert_eq!(
            q.fence(Epoch(2), LeaseToken(2)),
            5,
            "a new epoch fences too"
        );
    }

    #[test]
    fn delta_actions_are_delivered_once_and_a_replan_never_redelivers_a_slot() {
        let mut q = ActionQueue::new(Delivery::Once);
        q.splice(splice(1, 1, 0, 4, 0, 0), 0);
        let firsts: Vec<bool> = [0, MS / 3, 2 * MS / 3]
            .iter()
            .map(|t| q.due(*t).expect("due").first)
            .collect();
        assert_eq!(firsts, vec![true, false, false]);
        let o = q.splice(splice(1, 2, 0, 4, 0, 100), MS / 2);
        assert_eq!((o.kept_committed, o.replaced, o.queued), (1, 3, 3), "{o:?}");
        assert!(!q.due(MS / 2 + 1).expect("due").first);
        let d = q.due(MS).expect("slot 1");
        assert_eq!((tag(&d), d.first), (101, true));
        let d = q.due(3 * MS).expect("slot 3");
        assert_eq!((tag(&d), d.first), (103, true));
        assert_eq!(q.stats().skipped, 1);
        let mut q = ActionQueue::new(Delivery::Once);
        q.splice(splice(1, 1, 0, 1, 0, 0), 0);
        q.splice(splice(1, 2, MS + MS / 2, 1, MS, 10), 0);
        assert!(q.due(0).is_some());
        assert!(q.due(MS + MS / 4).is_none());
    }

    #[test]
    fn an_obsolete_chunk_is_a_no_op_not_a_gap() {
        let mut q = ActionQueue::default();
        q.splice(splice(1, 1, 0, 10, 0, 0), 0);
        let o = q.splice(splice(1, 2, 0, 5, 0, 100), 5 * MS + MS / 2);
        assert_eq!((o.dropped_past, o.replaced, o.queued), (5, 0, 0), "{o:?}");
        assert_eq!(tags(&mut q, 6..8), vec![6, 7]);
        let mut void = splice(1, 3, 8 * MS, 5, 0, 100);
        void.chunk = meta(1, 3, 8 * MS);
        let o = q.splice(void, 7 * MS);
        assert_eq!((o.dropped_void, o.replaced), (5, 0));
        assert_eq!(tags(&mut q, 8..10), vec![8, 9]);
        let mut q = ActionQueue::default();
        q.splice(splice(1, 1, 0, 10, 0, 0), 0);
        let o = q.splice(splice(1, 2, 2 * MS, 5, 0, 100), 4 * MS + MS / 2);
        assert_eq!((o.dropped_past, o.queued), (2, 3), "{o:?}");
        assert_eq!(
            tags(&mut q, 4..10),
            vec![102, 103, 104],
            "the new plan from 4 ms"
        );
    }

    #[test]
    fn an_absolute_action_already_applied_is_still_replaced_from_replace_from() {
        let mut q = ActionQueue::default();
        q.splice(splice(1, 1, 0, 1, 0, 0), 0);
        assert!(q.due(10_000).expect("due").first);
        let o = q.splice(splice(1, 2, MS - 5_000, 1, MS - 5_000, 50), 100_000);
        assert_eq!((o.queued, o.kept_committed), (1, 0), "{o:?}");
        assert_eq!(q.due(MS).map(|d| tag(&d)), Some(50));
    }
}
