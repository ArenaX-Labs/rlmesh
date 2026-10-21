//! The virtual-mode tick barrier (§11.3), sans-IO.
//!
//! Session time is `tick × tick_ns`, advanced by the host tick by tick:
//!
//! 1. **Ready.** After arm, and after every step, each embodiment publishes
//!    its samples for tick k, then sends `Ready{k, watermarks}`. Nothing
//!    advances before the first `Ready` of an epoch.
//! 2. **Observe.** Once every embodiment is ready, the host publishes
//!    `TickState{k, watermarks, due}`: `due` lists the controllers whose
//!    policy period falls on k (from their binding's replan rate).
//! 3. **Decide.** Each due controller waits until its inputs reach the
//!    watermarks, snapshots, sends its chunks and `Decided{k, chunks}`. Its
//!    decision is scheduled at tick k + d, where d is the session's latency
//!    in ticks.
//! 4. **Step.** The host publishes `Step{k, chunks}` with the chunks
//!    scheduled at k, never before every decision scheduled at or before k
//!    has arrived, so real computation is never raced.
//! 5. The embodiment waits until its edge has judged every listed chunk,
//!    applies what is due, steps physics by `tick_ns` and returns to 1 with
//!    k + 1.
//!
//! Controllers not due at k are not waited for. A barrier not met within the
//! timeout ends the run: the host goes SAFE. d is a constant here; seeded
//! distributions and recorded traces belong to the managed
//! timing-robustness suite.

use std::collections::{BTreeMap, BTreeSet};

use crate::wire::tick::{Step, TickState, VirtualTick};
use crate::wire::{ChannelId, ChunkRef, CtlError, CtlStatus, Epoch, ParticipantId, Watermark};

/// How a virtual session ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VirtualConfig {
    /// Session time per tick.
    pub tick_ns: u64,
    /// Ticks between a decision and its delivery (the inference cost d).
    pub latency_ticks: u64,
    /// How long a barrier may wait (host clock) before the run ends SAFE.
    pub barrier_timeout_ns: u64,
}

impl Default for VirtualConfig {
    fn default() -> Self {
        Self {
            tick_ns: 1_000_000,
            latency_ticks: 0,
            barrier_timeout_ns: 30_000_000_000,
        }
    }
}

/// Who takes part in the barrier, from the authority state.
#[derive(Clone, Debug, Default)]
pub struct Participants {
    /// Embodiments that declared parts: every one sends `Ready`.
    pub embodiments: BTreeSet<ParticipantId>,
    /// Controllers with a binding, and their period in ticks.
    pub controllers: Vec<(ParticipantId, u64)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Waiting for every embodiment's `Ready` for the current tick.
    Ready,
    /// Observed; waiting for the decisions scheduled at or before it.
    Decide,
}

/// The barrier state of one epoch.
#[derive(Clone, Debug)]
pub struct VirtualClock {
    config: VirtualConfig,
    epoch: Epoch,
    tick: u64,
    phase: Phase,
    ready: BTreeSet<ParticipantId>,
    watermarks: BTreeMap<ChannelId, u64>,
    /// Per decision tick, the due controllers and whether each decided.
    due: BTreeMap<u64, BTreeMap<ParticipantId, bool>>,
    /// Chunks scheduled per delivery tick.
    scheduled: BTreeMap<u64, Vec<ChunkRef>>,
    started: bool,
    timed_out: bool,
    phase_since_ns: i64,
}

impl VirtualClock {
    /// A barrier at tick 0 of `epoch`.
    pub fn new(config: VirtualConfig, epoch: Epoch, now_ns: i64) -> Self {
        Self {
            config,
            epoch,
            tick: 0,
            phase: Phase::Ready,
            ready: BTreeSet::new(),
            watermarks: BTreeMap::new(),
            due: BTreeMap::new(),
            scheduled: BTreeMap::new(),
            started: false,
            timed_out: false,
            phase_since_ns: now_ns,
        }
    }

    /// Its configuration.
    pub fn config(&self) -> VirtualConfig {
        self.config
    }

    /// The tick the barrier is on.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Starts over at tick 0 of a new epoch.
    pub fn reset(&mut self, epoch: Epoch, now_ns: i64) {
        *self = Self::new(self.config, epoch, now_ns);
    }

    /// An embodiment's `Ready` for `tick`.
    pub fn on_ready(
        &mut self,
        from: &ParticipantId,
        tick: u64,
        watermarks: &[Watermark],
        who: &Participants,
        now_ns: i64,
    ) -> Result<Vec<VirtualTick>, CtlError> {
        if !who.embodiments.contains(from) {
            return Err(CtlError::new(
                CtlStatus::Forbidden,
                format!("{from} declares no parts: it is not in the tick barrier"),
            ));
        }
        let current = self.phase == Phase::Ready && tick == self.tick;
        if !current {
            // A retry of a ready the barrier already counted is a no-op.
            if tick < self.tick || (tick == self.tick && self.ready.contains(from)) {
                return Ok(Vec::new());
            }
            return Err(CtlError::new(
                CtlStatus::InvalidTransition,
                format!(
                    "ready for tick {tick}; the barrier is at tick {}",
                    self.tick
                ),
            ));
        }
        self.started = true;
        self.ready.insert(from.clone());
        for w in watermarks {
            let seq = self.watermarks.entry(w.channel).or_default();
            *seq = (*seq).max(w.seq);
        }
        Ok(self.advance(who, now_ns))
    }

    /// A due controller's `Decided` for `tick`.
    pub fn on_decided(
        &mut self,
        from: &ParticipantId,
        tick: u64,
        chunks: &[ChunkRef],
        who: &Participants,
        now_ns: i64,
    ) -> Result<Vec<VirtualTick>, CtlError> {
        let Some(decided) = self.due.get_mut(&tick).and_then(|due| due.get_mut(from)) else {
            return Err(CtlError::new(
                CtlStatus::InvalidRequest,
                format!("{from} is not due at tick {tick}"),
            ));
        };
        if std::mem::replace(decided, true) {
            return Ok(Vec::new());
        }
        let at = tick.saturating_add(self.config.latency_ticks);
        self.scheduled
            .entry(at)
            .or_default()
            .extend_from_slice(chunks);
        Ok(self.advance(who, now_ns))
    }

    /// A participant left: it no longer holds the barrier up.
    pub fn on_gone(
        &mut self,
        id: &ParticipantId,
        who: &Participants,
        now_ns: i64,
    ) -> Vec<VirtualTick> {
        self.ready.remove(id);
        for due in self.due.values_mut() {
            if let Some(decided) = due.get_mut(id) {
                *decided = true;
            }
        }
        self.advance(who, now_ns)
    }

    /// The tick a barrier timed out on, once.
    pub fn check_timeout(&mut self, now_ns: i64) -> Option<u64> {
        let waited = now_ns.saturating_sub(self.phase_since_ns).unsigned_abs();
        if self.started && !self.timed_out && waited > self.config.barrier_timeout_ns {
            self.timed_out = true;
            return Some(self.tick);
        }
        None
    }

    /// Runs the barrier as far as it can go.
    fn advance(&mut self, who: &Participants, now_ns: i64) -> Vec<VirtualTick> {
        let mut out = Vec::new();
        if self.phase == Phase::Ready {
            let all_ready = !who.embodiments.is_empty()
                && who.embodiments.iter().all(|e| self.ready.contains(e));
            if !all_ready {
                return out;
            }
            let k = self.tick;
            let due: Vec<ParticipantId> = who
                .controllers
                .iter()
                .filter(|(_, period)| *period > 0 && k.is_multiple_of(*period))
                .map(|(id, _)| id.clone())
                .collect();
            self.due
                .insert(k, due.iter().map(|id| (id.clone(), false)).collect());
            out.push(VirtualTick::Observe(TickState {
                epoch: self.epoch,
                tick: k,
                watermarks: self
                    .watermarks
                    .iter()
                    .map(|(channel, seq)| Watermark {
                        channel: *channel,
                        seq: *seq,
                    })
                    .collect(),
                due,
            }));
            self.phase = Phase::Decide;
            self.phase_since_ns = now_ns;
        }
        // Step k once every decision scheduled at or before k has arrived:
        // decisions made at ticks up to k - d.
        let k = self.tick;
        let latest_decision = k.checked_sub(self.config.latency_ticks);
        let pending = latest_decision.is_some_and(|last| {
            self.due
                .range(..=last)
                .any(|(_, due)| due.values().any(|decided| !decided))
        });
        if !pending {
            out.push(VirtualTick::Step(Step {
                epoch: self.epoch,
                tick: k,
                chunks: self.scheduled.remove(&k).unwrap_or_default(),
            }));
            if let Some(last) = latest_decision {
                // Every decision up to `last` is in: forget them.
                self.due.retain(|tick, _| *tick > last);
            }
            self.tick = k + 1;
            self.phase = Phase::Ready;
            self.ready.clear();
            self.phase_since_ns = now_ns;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::LeaseToken;

    fn pid(id: &str) -> ParticipantId {
        ParticipantId::new(id).expect("id")
    }

    fn who(period: u64) -> Participants {
        Participants {
            embodiments: [pid("robot")].into(),
            controllers: vec![(pid("policy"), period)],
        }
    }

    fn chunk(seq: u64) -> ChunkRef {
        ChunkRef {
            channel: ChannelId(1),
            lease_token: LeaseToken(1),
            seq,
        }
    }

    fn kinds(ticks: &[VirtualTick]) -> Vec<String> {
        ticks
            .iter()
            .map(|t| match t {
                VirtualTick::Observe(s) => format!("observe{}{:?}", s.tick, s.due.len()),
                VirtualTick::Step(s) => format!("step{}:{}", s.tick, s.chunks.len()),
            })
            .collect()
    }

    #[test]
    fn a_due_controller_holds_the_step_until_it_decides() {
        let mut clock = VirtualClock::new(VirtualConfig::default(), Epoch(2), 0);
        let who = who(2);
        let wm = [Watermark {
            channel: ChannelId(5),
            seq: 1,
        }];
        let out = clock
            .on_ready(&pid("robot"), 0, &wm, &who, 0)
            .expect("ready");
        assert_eq!(kinds(&out), ["observe01"], "due at 0, so no step yet");
        let VirtualTick::Observe(state) = &out[0] else {
            panic!("observe");
        };
        assert_eq!(state.watermarks, wm);
        let out = clock
            .on_decided(&pid("policy"), 0, &[chunk(1)], &who, 1)
            .expect("decided");
        assert_eq!(kinds(&out), ["step0:1"]);
        // Tick 1: not due, so observe and step together.
        let out = clock
            .on_ready(&pid("robot"), 1, &[], &who, 2)
            .expect("ready");
        assert_eq!(kinds(&out), ["observe10", "step1:0"]);
        assert_eq!(clock.tick(), 2);
        // A retried ready and decision are no-ops; a future tick is refused.
        assert!(
            clock
                .on_ready(&pid("robot"), 1, &[], &who, 3)
                .expect("retry")
                .is_empty()
        );
        assert_eq!(
            clock
                .on_ready(&pid("robot"), 5, &[], &who, 3)
                .map_err(|e| e.status),
            Err(CtlStatus::InvalidTransition)
        );
        assert!(
            clock.on_decided(&pid("policy"), 1, &[], &who, 3).is_err(),
            "not due at 1"
        );
        assert_eq!(
            clock
                .on_ready(&pid("policy"), 2, &[], &who, 3)
                .map_err(|e| e.status),
            Err(CtlStatus::Forbidden)
        );
    }

    #[test]
    fn with_latency_a_decision_lands_d_ticks_later_and_physics_runs_meanwhile() {
        let config = VirtualConfig {
            latency_ticks: 2,
            ..VirtualConfig::default()
        };
        let mut clock = VirtualClock::new(config, Epoch(1), 0);
        let who = who(4);
        let mut steps = Vec::new();
        for k in 0..7 {
            let out = clock
                .on_ready(&pid("robot"), k, &[], &who, 0)
                .expect("ready");
            steps.extend(kinds(&out));
            if k == 1 {
                // Inference started at tick 0 and finishes during tick 1.
                let out = clock
                    .on_decided(&pid("policy"), 0, &[chunk(7)], &who, 0)
                    .expect("decided");
                assert!(out.is_empty(), "the barrier waits on tick 2's ready");
            }
        }
        assert_eq!(
            steps,
            [
                "observe01",
                "step0:0",
                "observe10",
                "step1:0",
                "observe20",
                "step2:1",
                "observe30",
                "step3:0",
                "observe41",
                "step4:0",
                "observe50",
                "step5:0",
                "observe60"
            ],
            "tick 0's decision lands at 2; tick 4's decision holds step 6, not 4 or 5"
        );
        let out = clock
            .on_decided(&pid("policy"), 4, &[chunk(8)], &who, 0)
            .expect("decided");
        assert_eq!(kinds(&out), ["step6:1"]);
    }

    #[test]
    fn a_vanished_participant_releases_the_barrier_and_a_stall_times_out() {
        let config = VirtualConfig {
            barrier_timeout_ns: 100,
            ..VirtualConfig::default()
        };
        let mut clock = VirtualClock::new(config, Epoch(1), 0);
        let mut who = who(1);
        assert_eq!(clock.check_timeout(1_000), None, "not started");
        let out = clock
            .on_ready(&pid("robot"), 0, &[], &who, 0)
            .expect("ready");
        assert_eq!(kinds(&out), ["observe01"]);
        who.controllers.clear();
        let out = clock.on_gone(&pid("policy"), &who, 10);
        assert_eq!(kinds(&out), ["step0:0"]);
        assert_eq!(clock.check_timeout(50), None);
        assert_eq!(clock.check_timeout(200), Some(1));
        assert_eq!(clock.check_timeout(300), None, "once");
        clock.reset(Epoch(2), 300);
        assert_eq!(clock.tick(), 0);
    }
}
