//! The edge mode machine, run once per control tick on edge-local time.
//!
//! - `Idle`: no lease, or nothing applied under the current lease yet. The
//!   wait timer is not armed, so a fresh grant does not trip SAFE before the
//!   first chunk.
//! - `Driving`: an action is due and applied.
//! - `Hold`: nothing valid is due. The edge waits at most `hold_max` for a
//!   valid action. While it waits it repeats the last action **only** when
//!   the channel permits it (`Actuation::hold_last`) and that action is
//!   still valid (before its `valid_until`) and under the current epoch and
//!   lease token; otherwise it runs the safe behaviour. An expired or
//!   preempted command is never continued.
//! - `Safe`: the declared safe behaviour, latched until the epoch moves on (a
//!   re-arm). Entered when the wait exceeds `hold_max`, when the lease lapses
//!   by the edge's clock, or when the host says SAFE.
//! - `Estop`: the session is in ESTOP.
//!
//! Every deadline is absolute. The wait is measured from when the queue's
//! coverage actually ended (the last action's slot end, its `valid_until`, or
//! the preemption), not from when a tick first noticed: after a CPU stall the
//! edge acts on the real elapsed time at its next tick. Behaviour *during* a
//! full process stall cannot be bounded by the edge itself: that needs an
//! external or device watchdog.

use std::fmt;

use rlmesh_wire::{Epoch, LeaseToken, Mode};

use crate::queue::{ActionQueue, Delivery, QueuedAction};

/// The edge's mode, per control tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EdgeMode {
    /// No lease, or nothing applied under it yet.
    Idle,
    /// Applying the action due.
    Driving,
    /// Waiting (at most `hold_max`) for a valid action.
    Hold,
    /// Running the declared safe behaviour; latched until a new epoch.
    Safe,
    /// The session is in ESTOP.
    Estop,
}

impl EdgeMode {
    /// Every mode, in stats order.
    pub const ALL: [Self; 5] = [
        Self::Idle,
        Self::Driving,
        Self::Hold,
        Self::Safe,
        Self::Estop,
    ];

    fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for EdgeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Idle => "idle",
            Self::Driving => "driving",
            Self::Hold => "hold",
            Self::Safe => "safe",
            Self::Estop => "estop",
        })
    }
}

/// Why the edge went safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SafeCause {
    /// No valid action arrived within `hold_max`.
    QueueDry,
    /// The lease expired by the edge's own clock (the host's renewals
    /// stopped reaching it), or could no longer be timed.
    LeaseLapsed,
    /// The host's view says SAFE.
    HostSafe,
}

impl SafeCause {
    /// Causes the host has not seen yet: the edge reports them.
    pub fn needs_report(self) -> bool {
        !matches!(self, Self::HostSafe)
    }
}

impl fmt::Display for SafeCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::QueueDry => "no valid action within hold_max",
            Self::LeaseLapsed => "lease lapsed by the edge clock",
            Self::HostSafe => "session SAFE",
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

/// The lease on this channel as the edge's clock sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeaseView {
    /// The current token.
    pub token: LeaseToken,
    /// Its conservative expiry in the edge clock; `None` when it cannot be
    /// timed (no clock mapping).
    pub expires_ns: Option<i64>,
}

/// What the edge's authority view says this tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TickView {
    /// The epoch.
    pub epoch: Epoch,
    /// The mode.
    pub mode: Mode,
    /// The lease on this channel.
    pub lease: Option<LeaseView>,
}

/// One tick's decision.
#[derive(Clone, Debug)]
pub struct Decision {
    /// The mode.
    pub mode: EdgeMode,
    /// The action to apply: the one due (`Driving`; under
    /// [`Delivery::Once`] only on its first tick), or the permitted repeat
    /// of the last valid one (`Hold`).
    pub action: Option<QueuedAction>,
    /// Whether to run the safe behaviour this tick (`Safe`, `Estop`, and a
    /// `Hold` that may not repeat).
    pub run_safe: bool,
    /// Set on the tick the edge went safe.
    pub went_safe: Option<SafeCause>,
}

/// How the ladder behaves for one channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FallbackConfig {
    /// The longest wait for a valid action.
    pub hold_max_ns: i64,
    /// Whether a wait may repeat the last valid action.
    pub hold_last: bool,
    /// How actions are delivered.
    pub delivery: Delivery,
}

/// The fallback ladder.
#[derive(Debug)]
pub struct Fallback {
    config: FallbackConfig,
    mode: EdgeMode,
    /// The epoch the edge latched SAFE in.
    safe_epoch: Option<Epoch>,
    safe_cause: Option<SafeCause>,
    last: Option<QueuedAction>,
    /// When valid coverage ended (edge clock), while waiting.
    dry_since_ns: Option<i64>,
    last_tick_ns: Option<i64>,
    modes: [ModeStats; 5],
}

impl Fallback {
    /// A ladder starting `Idle`.
    pub fn new(config: FallbackConfig) -> Self {
        Self {
            config: FallbackConfig {
                hold_max_ns: config.hold_max_ns.max(0),
                ..config
            },
            mode: EdgeMode::Idle,
            safe_epoch: None,
            safe_cause: None,
            last: None,
            dry_since_ns: None,
            last_tick_ns: None,
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

    /// Entries, ticks and time per mode.
    pub fn stats(&self) -> Vec<(EdgeMode, ModeStats)> {
        EdgeMode::ALL
            .iter()
            .map(|m| (*m, self.modes[m.index()]))
            .collect()
    }

    /// Runs one tick at edge time `now_ns`: fences and drains `queue`
    /// against `view`, and decides what the embodiment applies.
    pub fn tick(
        &mut self,
        now_ns: i64,
        view: Option<TickView>,
        queue: &mut ActionQueue,
    ) -> Decision {
        if let Some(previous) = self.last_tick_ns {
            let elapsed = u64::try_from(now_ns.saturating_sub(previous)).unwrap_or(0);
            self.modes[self.mode.index()].time_ns += elapsed;
        }
        self.last_tick_ns = Some(now_ns);
        let decision = self.decide(now_ns, view, queue);
        if decision.mode != self.mode {
            self.modes[decision.mode.index()].entries += 1;
            self.mode = decision.mode;
        }
        self.modes[decision.mode.index()].ticks += 1;
        decision
    }

    fn decide(&mut self, now_ns: i64, view: Option<TickView>, queue: &mut ActionQueue) -> Decision {
        let Some(view) = view else {
            queue.clear();
            return self.idle();
        };
        if view.mode == Mode::Estop {
            queue.clear();
            self.last = None;
            self.dry_since_ns = None;
            return Decision {
                mode: EdgeMode::Estop,
                action: None,
                run_safe: true,
                went_safe: None,
            };
        }
        if let Some(epoch) = self.safe_epoch {
            if view.epoch <= epoch {
                queue.clear();
                return Decision {
                    mode: EdgeMode::Safe,
                    action: None,
                    run_safe: true,
                    went_safe: None,
                };
            }
            // Re-armed: a fresh epoch releases the latch.
            self.safe_epoch = None;
            self.safe_cause = None;
        }
        if view.mode == Mode::Safe {
            return self.go_safe(view.epoch, SafeCause::HostSafe, queue);
        }
        let Some(lease) = view.lease else {
            queue.clear();
            return self.idle();
        };
        match lease.expires_ns {
            Some(expires) if now_ns > expires => {
                return self.go_safe(view.epoch, SafeCause::LeaseLapsed, queue);
            }
            Some(_) => {}
            // Untimeable: harmless before anything was applied, a lapse after.
            None if self.last.is_none() => {
                queue.clear();
                return self.idle();
            }
            None => return self.go_safe(view.epoch, SafeCause::LeaseLapsed, queue),
        }
        queue.fence(view.epoch, lease.token);
        if let Some(due) = queue.due(now_ns) {
            self.dry_since_ns = None;
            let deliver = self.config.delivery == Delivery::EveryTick || due.first;
            self.last = Some(due.action.clone());
            return Decision {
                mode: EdgeMode::Driving,
                action: deliver.then_some(due.action),
                run_safe: false,
                went_safe: None,
            };
        }
        let Some(last) = self.last.clone() else {
            // A lease with nothing applied under it yet.
            return Decision {
                mode: EdgeMode::Idle,
                action: None,
                run_safe: false,
                went_safe: None,
            };
        };
        let current = last.chunk.epoch == view.epoch && last.chunk.lease_token == lease.token;
        let valid = current && now_ns < last.chunk.valid_until_ns;
        // Coverage ended when the last action's slot ended, when it went
        // void, or (preempted) by now at the latest.
        let mut covered_until = last.end_ns().min(last.chunk.valid_until_ns);
        if !current {
            covered_until = covered_until.min(now_ns);
        }
        let dry_since = *self.dry_since_ns.get_or_insert(covered_until.min(now_ns));
        if now_ns.saturating_sub(dry_since) >= self.config.hold_max_ns {
            return self.go_safe(view.epoch, SafeCause::QueueDry, queue);
        }
        let repeat = self.config.hold_last && self.config.delivery == Delivery::EveryTick && valid;
        Decision {
            mode: EdgeMode::Hold,
            action: repeat.then_some(last),
            // A delta channel delivers nothing while it waits (the robot keeps
            // its target); anything else may not keep its last command.
            run_safe: !repeat && self.config.delivery == Delivery::EveryTick,
            went_safe: None,
        }
    }

    fn idle(&mut self) -> Decision {
        self.last = None;
        self.dry_since_ns = None;
        Decision {
            mode: EdgeMode::Idle,
            action: None,
            run_safe: false,
            went_safe: None,
        }
    }

    fn go_safe(&mut self, epoch: Epoch, cause: SafeCause, queue: &mut ActionQueue) -> Decision {
        queue.clear();
        self.safe_epoch = Some(epoch);
        self.safe_cause = Some(cause);
        self.last = None;
        self.dry_since_ns = None;
        Decision {
            mode: EdgeMode::Safe,
            action: None,
            run_safe: true,
            went_safe: Some(cause),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use rlmesh_wire::{ObservationRef, ParticipantId};

    use super::*;
    use crate::queue::{ChunkMeta, Splice};

    const MS: i64 = 1_000_000;

    fn splice(token: u64, t0: i64, n: u8, valid_until: i64) -> Splice {
        Splice {
            chunk: Arc::new(ChunkMeta {
                source: ParticipantId::new("policy").expect("id"),
                epoch: Epoch(1),
                lease_token: LeaseToken(token),
                seq: 1,
                based_on: vec![ObservationRef {
                    channel: rlmesh_wire::ChannelId(1),
                    seq: 1,
                }],
                receive_ns: 0,
                valid_until_ns: valid_until,
            }),
            t0_ns: t0,
            dt_ns: MS,
            replace_from_ns: t0,
            actions: (0..n).map(|i| vec![Bytes::from(vec![i])]).collect(),
        }
    }

    fn view(token: Option<u64>, expires: i64) -> Option<TickView> {
        Some(TickView {
            epoch: Epoch(1),
            mode: if token.is_some() {
                Mode::Active
            } else {
                Mode::Armed
            },
            lease: token.map(|t| LeaseView {
                token: LeaseToken(t),
                expires_ns: Some(expires),
            }),
        })
    }

    fn ladder(hold_max: i64, hold_last: bool) -> Fallback {
        Fallback::new(FallbackConfig {
            hold_max_ns: hold_max,
            hold_last,
            delivery: Delivery::EveryTick,
        })
    }

    #[test]
    fn a_permitted_hold_repeats_the_last_action_then_goes_safe_until_a_new_epoch() {
        let mut q = ActionQueue::default();
        let mut f = ladder(5 * MS, true);
        let lease = view(Some(1), i64::MAX);
        assert_eq!(f.tick(0, lease, &mut q).mode, EdgeMode::Idle, "nothing yet");
        q.splice(splice(1, MS, 3, i64::MAX), 0);
        for t in 1..=3 {
            assert_eq!(f.tick(t * MS, lease, &mut q).mode, EdgeMode::Driving);
        }
        // Coverage ends at 4 ms: hold the last action until 9 ms.
        for t in 4..9 {
            let d = f.tick(t * MS, lease, &mut q);
            assert_eq!(d.mode, EdgeMode::Hold, "at {t} ms");
            assert_eq!(d.action.map(|a| a.index), Some(2));
            assert!(!d.run_safe);
        }
        let d = f.tick(9 * MS, lease, &mut q);
        assert_eq!(
            (d.mode, d.went_safe),
            (EdgeMode::Safe, Some(SafeCause::QueueDry))
        );
        // Latched: new chunks in the same epoch are not acted on.
        q.splice(splice(1, 10 * MS, 3, i64::MAX), 0);
        let d = f.tick(10 * MS, lease, &mut q);
        assert_eq!(
            (d.mode, d.went_safe, d.run_safe),
            (EdgeMode::Safe, None, true)
        );
        // Re-armed (epoch 2), new lease: idle until the first action.
        let rearmed = Some(TickView {
            epoch: Epoch(2),
            mode: Mode::Active,
            lease: Some(LeaseView {
                token: LeaseToken(3),
                expires_ns: Some(i64::MAX),
            }),
        });
        assert_eq!(f.tick(11 * MS, rearmed, &mut q).mode, EdgeMode::Idle);
        let stats: std::collections::BTreeMap<_, _> = f.stats().into_iter().collect();
        assert_eq!(stats[&EdgeMode::Hold].ticks, 5);
        assert_eq!(stats[&EdgeMode::Hold].time_ns, 5 * MS as u64);
        assert_eq!(stats[&EdgeMode::Safe].entries, 1);
        assert_eq!(stats[&EdgeMode::Driving].ticks, 3);
    }

    #[test]
    fn without_permission_a_hold_runs_the_safe_behaviour_instead_of_repeating() {
        let mut q = ActionQueue::default();
        let mut f = ladder(5 * MS, false);
        let lease = view(Some(1), i64::MAX);
        q.splice(splice(1, 0, 1, i64::MAX), 0);
        assert_eq!(f.tick(0, lease, &mut q).mode, EdgeMode::Driving);
        let d = f.tick(MS, lease, &mut q);
        assert_eq!(d.mode, EdgeMode::Hold);
        assert!(d.action.is_none() && d.run_safe);
    }

    #[test]
    fn an_expired_last_action_is_never_repeated() {
        let mut q = ActionQueue::default();
        let mut f = ladder(50 * MS, true);
        let lease = view(Some(1), i64::MAX);
        // Ten 1 ms actions, void from 3 ms (the last seven never queue).
        q.splice(splice(1, 0, 10, 3 * MS), 0);
        for t in 0..3 {
            assert_eq!(f.tick(t * MS, lease, &mut q).mode, EdgeMode::Driving);
        }
        let d = f.tick(4 * MS, lease, &mut q);
        assert_eq!(d.mode, EdgeMode::Hold);
        assert!(d.action.is_none(), "expired at 3 ms: not continued");
        assert!(d.run_safe);
        // The wait counts from the expiry (3 ms), not from this tick.
        assert_eq!(f.tick(53 * MS, lease, &mut q).mode, EdgeMode::Safe);
    }

    #[test]
    fn a_preempted_holders_last_action_is_never_continued() {
        let mut q = ActionQueue::default();
        let mut f = ladder(20 * MS, true);
        q.splice(splice(1, 0, 100, i64::MAX), 0);
        assert_eq!(
            f.tick(0, view(Some(1), i64::MAX), &mut q).mode,
            EdgeMode::Driving
        );
        // Token 2 takes over: the model's queued actions are fenced and its
        // last action is not held; the safe behaviour runs while the edge
        // waits for the operator.
        let human = view(Some(2), i64::MAX);
        let d = f.tick(MS, human, &mut q);
        assert_eq!((d.mode, q.stats().fenced), (EdgeMode::Hold, 100));
        assert!(d.action.is_none() && d.run_safe);
        q.splice(splice(2, 2 * MS, 5, i64::MAX), 2 * MS);
        let d = f.tick(2 * MS, human, &mut q);
        assert_eq!(d.mode, EdgeMode::Driving);
        assert_eq!(d.action.map(|a| a.chunk.lease_token), Some(LeaseToken(2)));
    }

    #[test]
    fn after_a_stall_the_edge_acts_on_the_real_elapsed_time() {
        let mut q = ActionQueue::default();
        let mut f = ladder(100 * MS, true);
        let lease = view(Some(1), i64::MAX);
        q.splice(splice(1, 0, 2, i64::MAX), 0);
        assert_eq!(f.tick(MS, lease, &mut q).mode, EdgeMode::Driving);
        // The process stalls from 1 ms to 500 ms. Coverage ended at 2 ms, so
        // the 100 ms wait is long over: SAFE at once, no hold.
        let d = f.tick(500 * MS, lease, &mut q);
        assert_eq!(
            (d.mode, d.went_safe),
            (EdgeMode::Safe, Some(SafeCause::QueueDry))
        );
    }

    #[test]
    fn estop_and_a_local_lease_lapse_fall_back_at_once() {
        let mut q = ActionQueue::default();
        let mut f = ladder(50 * MS, true);
        q.splice(splice(1, 0, 100, i64::MAX), 0);
        let lease = view(Some(1), 20 * MS);
        assert_eq!(f.tick(MS, lease, &mut q).mode, EdgeMode::Driving);
        // The edge's copy of the lease expires at 20 ms: it goes safe at once,
        // without holding, though its queue still reaches 100 ms.
        let d = f.tick(21 * MS, lease, &mut q);
        assert_eq!(
            (d.mode, d.went_safe),
            (EdgeMode::Safe, Some(SafeCause::LeaseLapsed))
        );
        assert!(q.is_empty());

        let mut q = ActionQueue::default();
        let mut f = ladder(50 * MS, true);
        q.splice(splice(1, 0, 100, i64::MAX), 0);
        assert_eq!(f.tick(MS, lease, &mut q).mode, EdgeMode::Driving);
        let estop = Some(TickView {
            epoch: Epoch(1),
            mode: Mode::Estop,
            lease: None,
        });
        let d = f.tick(2 * MS, estop, &mut q);
        assert_eq!(
            (d.mode, d.went_safe, d.run_safe),
            (EdgeMode::Estop, None, true)
        );
        assert!(q.is_empty(), "ESTOP drops the queue");
    }

    #[test]
    fn an_untimeable_lease_idles_before_the_first_action_and_lapses_after() {
        let untimed = Some(TickView {
            epoch: Epoch(1),
            mode: Mode::Active,
            lease: Some(LeaseView {
                token: LeaseToken(1),
                expires_ns: None,
            }),
        });
        let mut q = ActionQueue::default();
        let mut f = ladder(50 * MS, true);
        assert_eq!(f.tick(0, untimed, &mut q).mode, EdgeMode::Idle);
        q.splice(splice(1, 0, 10, i64::MAX), 0);
        assert_eq!(
            f.tick(MS, view(Some(1), i64::MAX), &mut q).mode,
            EdgeMode::Driving
        );
        assert_eq!(
            f.tick(2 * MS, untimed, &mut q).went_safe,
            Some(SafeCause::LeaseLapsed)
        );
    }
}
