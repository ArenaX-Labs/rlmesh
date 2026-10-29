//! Policy starts (§8.9) and slowdown reporting (§8.10), sans-IO: every
//! time is passed in (`now_ns`), nothing sleeps or spawns.
//!
//! [`StartSchedule`] decides when a controller's policy starts:
//!
//! - **One inference in flight.** Starts fall on a fixed grid of slots, one
//!   per replan period. A slot that falls while `predict` is busy is
//!   skipped (`busy_skip`), never queued: when the inference completes, the
//!   next start is the first slot still ahead, with a fresh snapshot.
//!   Overdue starts are never accumulated or replayed in a burst.
//! - **Skipped starts** keep their reason: `waiting_for` (an input never
//!   received in this epoch), `stale_input`, `window_gap`, `decode` (a
//!   sample did not decode or fit its plan) and `no_lease` (a standby, or
//!   a preempted controller: its inputs stay warm, `predict` does not run).
//! - **`rate_drift`** (scope `policy`): the completion rate over a window of
//!   `max(1 s, 4 / R)`, counting silent intervals, outside `[0.8, 1.25] × R`,
//!   with hysteresis (one event per transition). It is suppressed until the
//!   first window after the first start, so warming is not drift; its cause
//!   is the dominant skip reason in the window (`busy` for a slow model).
//! - **Inference pending**: [`StartSchedule::pending`] is the age of the
//!   inference in flight, for a timer outside the policy thread (not an
//!   average over completions).
//!
//! [`PhaseAlign`] phases a wall-clock grid just after the publication of
//! one sampled input, so a snapshot reads a fresh sample.
//!
//! Nothing here renegotiates: rates, horizons and contracts never change
//! because something got slow.

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::VecDeque;
use core::fmt;
use core::time::Duration;

use crate::cache::Snapshot;
use crate::spec::Binding;
use crate::wire::ContractMode;

/// How many completion latencies the percentiles are taken over.
const LATENCIES: usize = 1024;

/// Why a slot passed without a start.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkipReason {
    /// `predict` was still running when the slot fell (`busy_skip`).
    Busy,
    /// A required input was never received in this epoch (`waiting_for`).
    Waiting,
    /// A required input was stale and `hold` is not allowed.
    StaleInput,
    /// A history window has a seq gap (`window_gap`).
    WindowGap,
    /// A sample did not decode, or did not fit its plan.
    Decode,
    /// The controller does not hold its leases (a standby, or preempted).
    NoLease,
}

impl SkipReason {
    /// Every reason, in report order.
    pub const ALL: [Self; 6] = [
        Self::Busy,
        Self::Waiting,
        Self::StaleInput,
        Self::WindowGap,
        Self::Decode,
        Self::NoLease,
    ];

    /// The report name (`busy_skip`, `waiting_for`, ...).
    pub fn name(self) -> &'static str {
        match self {
            Self::Busy => "busy_skip",
            Self::Waiting => "waiting_for",
            Self::StaleInput => "stale_input",
            Self::WindowGap => "window_gap",
            Self::Decode => "decode",
            Self::NoLease => "no_lease",
        }
    }

    /// The `rate_drift` cause it stands for.
    fn cause(self) -> &'static str {
        match self {
            Self::Busy => "busy",
            other => other.name(),
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Busy => 0,
            Self::Waiting => 1,
            Self::StaleInput => 2,
            Self::WindowGap => 3,
            Self::Decode => 4,
            Self::NoLease => 5,
        }
    }
}

impl fmt::Display for SkipReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Something a controller runtime reports (§8.10).
#[derive(Clone, Debug, PartialEq)]
pub enum RunEvent {
    /// The completion rate left (`drifting`) or re-entered `[0.8, 1.25] ×`
    /// the replan rate.
    RateDrift {
        /// Completions per second over the window.
        measured_hz: f64,
        /// The replan rate R.
        expected_hz: f64,
        /// The dominant skip reason in the window (`busy`, `stale_input`,
        /// ...), or `unknown`.
        cause: &'static str,
        /// Entered drift (`true`) or recovered (`false`).
        drifting: bool,
    },
    /// An inference has been running for `pending`.
    InferencePending {
        /// How long.
        pending: Duration,
    },
}

impl fmt::Display for RunEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RateDrift {
                measured_hz,
                expected_hz,
                cause,
                drifting: true,
            } => write!(
                f,
                "rate_drift (policy, cause {cause}): {measured_hz:.1} Hz against a replan rate of {expected_hz} Hz"
            ),
            Self::RateDrift {
                measured_hz,
                expected_hz,
                ..
            } => write!(
                f,
                "rate_drift cleared: {measured_hz:.1} Hz against a replan rate of {expected_hz} Hz"
            ),
            Self::InferencePending { pending } => {
                write!(f, "inference pending for {:.1} s", pending.as_secs_f64())
            }
        }
    }
}

/// A controller's start and completion counters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunStats {
    /// `predict` calls started.
    pub starts: u64,
    /// `predict` calls completed.
    pub completions: u64,
    /// Slots skipped, by reason ([`SkipReason::ALL`] order).
    pub skipped: [u64; 6],
    /// Chunks sent.
    pub chunks_sent: u64,
    /// Chunks that could not be packed or sent.
    pub send_errors: u64,
    /// Completion latency (snapshot to the end of `predict`), median.
    pub latency_p50: Option<Duration>,
    /// Completion latency, 99th percentile.
    pub latency_p99: Option<Duration>,
    /// Completion latency, longest.
    pub latency_max: Option<Duration>,
    /// Whether the completion rate is drifting now.
    pub drifting: bool,
}

impl RunStats {
    /// Slots skipped for `reason`.
    pub fn skipped(&self, reason: SkipReason) -> u64 {
        self.skipped[reason.index()]
    }
}

/// When a controller's policy starts; see the module docs.
#[derive(Clone, Debug)]
pub struct StartSchedule {
    period_ns: i64,
    window_ns: i64,
    expected_hz: f64,
    /// The first slot; the grid is `anchor + k × period`.
    anchor_ns: Option<i64>,
    next_ns: Option<i64>,
    /// The slot and the start time of the inference in flight.
    running: Option<(i64, i64)>,
    /// When drift detection starts counting (one window after the first
    /// start).
    armed_ns: Option<i64>,
    completions: VecDeque<i64>,
    skips: VecDeque<(i64, SkipReason)>,
    latencies: VecDeque<i64>,
    latency_max: i64,
    stats: RunStats,
}

impl StartSchedule {
    /// Starts at most `replan_hz` times per second (a finite, positive
    /// rate).
    pub fn new(replan_hz: f64) -> Self {
        let hz = if replan_hz.is_finite() && replan_hz > 0.0 {
            replan_hz
        } else {
            1.0
        };
        let period_ns = ((1e9 / hz).round() as i64).max(1);
        let window_ns = (4 * period_ns).max(1_000_000_000);
        Self {
            period_ns,
            window_ns,
            expected_hz: hz,
            anchor_ns: None,
            next_ns: None,
            running: None,
            armed_ns: None,
            completions: VecDeque::new(),
            skips: VecDeque::new(),
            latencies: VecDeque::with_capacity(LATENCIES),
            latency_max: 0,
            stats: RunStats::default(),
        }
    }

    /// The replan period.
    pub fn period(&self) -> Duration {
        Duration::from_nanos(self.period_ns.unsigned_abs())
    }

    /// The drift window, `max(1 s, 4 / R)`.
    pub fn window(&self) -> Duration {
        Duration::from_nanos(self.window_ns.unsigned_abs())
    }

    /// The next slot at or after `now_ns`: when to try a start. The first
    /// call anchors the grid at `now_ns`.
    pub fn next_start(&mut self, now_ns: i64) -> i64 {
        match self.next_ns {
            Some(next) => next,
            None => {
                self.anchor_ns = Some(now_ns);
                self.next_ns = Some(now_ns);
                now_ns
            }
        }
    }

    /// The first slot strictly after `now_ns`.
    fn slot_after(&self, now_ns: i64) -> i64 {
        let anchor = self.anchor_ns.unwrap_or(now_ns);
        let k = (now_ns - anchor).div_euclid(self.period_ns) + 1;
        anchor + k * self.period_ns
    }

    /// The slot `now_ns` falls in (the latest at or before it).
    fn slot_of(&self, now_ns: i64) -> i64 {
        self.slot_after(now_ns) - self.period_ns
    }

    /// The slot due at `now_ns` passed without a start, for `reason`. The
    /// next attempt is the following slot. Returns a drift transition, if
    /// any.
    pub fn skip(&mut self, now_ns: i64, reason: SkipReason) -> Option<RunEvent> {
        self.count_skip(now_ns, reason, 1);
        self.next_ns = Some(self.slot_after(now_ns));
        self.check_drift(now_ns)
    }

    fn count_skip(&mut self, now_ns: i64, reason: SkipReason, n: u64) {
        if n == 0 {
            return;
        }
        self.stats.skipped[reason.index()] += n;
        self.skips.push_back((now_ns, reason));
    }

    /// `predict` starts now, on the slot due. Slots that fell between the
    /// due slot and now (the loop woke late) count as busy skips.
    pub fn start(&mut self, now_ns: i64) {
        let due = self.next_ns.unwrap_or(now_ns);
        let slot = if now_ns >= due {
            self.slot_of(now_ns)
        } else {
            due
        };
        let late = u64::try_from((slot - due).max(0) / self.period_ns).unwrap_or(0);
        self.count_skip(now_ns, SkipReason::Busy, late);
        self.running = Some((slot, now_ns));
        self.armed_ns.get_or_insert(now_ns + self.window_ns);
        self.stats.starts += 1;
    }

    /// `predict` returned (or raised) now. Every slot that fell while it ran
    /// is a busy skip; the next start is the first slot still ahead.
    /// Returns a drift transition, if any.
    pub fn finish(&mut self, now_ns: i64) -> Option<RunEvent> {
        let (slot, started) = self.running.take()?;
        let passed = (now_ns - slot).div_euclid(self.period_ns).max(0);
        self.count_skip(now_ns, SkipReason::Busy, passed.unsigned_abs());
        self.next_ns = Some(self.slot_after(now_ns));
        self.stats.completions += 1;
        self.completions.push_back(now_ns);
        let latency = (now_ns - started).max(0);
        self.latency_max = self.latency_max.max(latency);
        if self.latencies.len() == LATENCIES {
            self.latencies.pop_front();
        }
        self.latencies.push_back(latency);
        self.check_drift(now_ns)
    }

    /// How long the inference in flight has been running.
    pub fn pending(&self, now_ns: i64) -> Option<Duration> {
        self.running
            .map(|(_, started)| Duration::from_nanos((now_ns - started).max(0).unsigned_abs()))
    }

    /// Counts a sent chunk, or one that could not be packed or sent.
    pub fn sent(&mut self, ok: bool) {
        if ok {
            self.stats.chunks_sent += 1;
        } else {
            self.stats.send_errors += 1;
        }
    }

    /// The counters, with latency percentiles over the last 1024
    /// completions.
    pub fn stats(&self) -> RunStats {
        let mut stats = self.stats.clone();
        if !self.latencies.is_empty() {
            let mut sorted: Vec<i64> = self.latencies.iter().copied().collect();
            sorted.sort_unstable();
            let at = |q: f64| {
                let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
                Duration::from_nanos(sorted[rank - 1].unsigned_abs())
            };
            stats.latency_p50 = Some(at(0.5));
            stats.latency_p99 = Some(at(0.99));
            stats.latency_max = Some(Duration::from_nanos(self.latency_max.unsigned_abs()));
        }
        stats
    }

    /// The completion rate over the window ending at `now_ns`, and a drift
    /// transition when it crosses the band.
    fn check_drift(&mut self, now_ns: i64) -> Option<RunEvent> {
        let since = now_ns - self.window_ns;
        while self.completions.front().is_some_and(|t| *t <= since) {
            self.completions.pop_front();
        }
        while self.skips.front().is_some_and(|(t, _)| *t <= since) {
            self.skips.pop_front();
        }
        if self.armed_ns.is_none_or(|armed| now_ns < armed) {
            return None;
        }
        let measured_hz = self.completions.len() as f64 * 1e9 / self.window_ns as f64;
        let ratio = measured_hz / self.expected_hz;
        // Hysteresis: leave the band to enter drift; come back inside it,
        // with margin, to leave.
        let drifting = if self.stats.drifting {
            !(0.85..=1.2).contains(&ratio)
        } else {
            !(0.8..=1.25).contains(&ratio)
        };
        if drifting == self.stats.drifting {
            return None;
        }
        self.stats.drifting = drifting;
        Some(RunEvent::RateDrift {
            measured_hz,
            expected_hz: self.expected_hz,
            cause: self.dominant_cause(),
            drifting,
        })
    }

    fn dominant_cause(&self) -> &'static str {
        let mut counts = [0_u64; 6];
        for (_, reason) in &self.skips {
            counts[reason.index()] += 1;
        }
        SkipReason::ALL
            .iter()
            .filter(|r| counts[r.index()] > 0)
            .max_by_key(|r| counts[r.index()])
            .map_or("unknown", |r| r.cause())
    }
}

/// Phases a clocked run loop's starts just after the publication of one
/// sampled input (wall clock only; sans-IO, times injected).
///
/// A `latest` input is published at about the replan rate, at a phase set
/// by its publisher, and the start grid's phase is set by when the run
/// began: the two lock wherever they happen to fall, often with each start
/// just before the next sample, so every snapshot reads a sample nearly a
/// full replan period old. The aligner measures the phase of the aligned
/// input's newest sample at each snapshot (`(snapshot − capture) mod P`),
/// takes the freshest tenth over a window of starts and moves the
/// grid so that phase is [`Self::margin`]: once in full (always later, so no slot
/// is skipped), then by half the remaining error per window, at most a
/// quarter period, to follow slow drift between the publisher's clock and
/// this one.
///
/// The aligned input is the first bound input with a `latest` contract
/// (its publication follows the replan rate). `every` and `history`
/// inputs are already fresh at any phase, and events have none.
#[derive(Clone, Debug)]
pub struct PhaseAlign {
    input: usize,
    period_ns: i64,
    /// How far starts sit after the schedule's grid.
    pub offset_ns: i64,
    errors: Vec<i64>,
    aligned: bool,
}

impl PhaseAlign {
    /// Starts per estimate.
    pub const WINDOW: usize = 32;
    /// Starts before the first estimate.
    pub const FIRST: usize = 8;
    /// Consecutive starts reading the previous sample that move the
    /// starts later before the window ends.
    pub const LATE: usize = 4;
    /// The target sample age at a start: transit plus the publisher's
    /// loop jitter (`min(1 ms, P / 4)`).
    const MARGIN_NS: i64 = 1_000_000;
    /// Corrections smaller than this are left alone.
    const DEADBAND_NS: i64 = 50_000;

    /// The input `binding`'s starts align to, by index.
    pub fn input(binding: &Binding) -> Option<usize> {
        binding.inputs.iter().position(|row| {
            row.channel.is_some_and(|channel| {
                binding
                    .contracts
                    .iter()
                    .any(|c| c.channel == channel && c.mode == ContractMode::Latest)
            })
        })
    }

    /// An aligner for `binding`, or `None` when no input sets a phase.
    pub fn for_binding(binding: &Binding) -> Option<Self> {
        let input = Self::input(binding)?;
        let hz = if binding.replan_hz.is_finite() && binding.replan_hz > 0.0 {
            binding.replan_hz
        } else {
            return None;
        };
        Some(Self::new(input, ((1e9 / hz).round() as i64).max(1)))
    }

    /// Aligns to input `input` on a grid of period `period_ns`.
    pub fn new(input: usize, period_ns: i64) -> Self {
        Self {
            input,
            period_ns,
            offset_ns: 0,
            errors: Vec::with_capacity(Self::WINDOW),
            aligned: false,
        }
    }

    /// The target phase.
    pub fn margin(&self) -> i64 {
        Self::MARGIN_NS.min(self.period_ns / 4)
    }

    /// Records the phase of the aligned input's newest sample in `snapshot`,
    /// taken at `snapshot_ns` (session time).
    pub fn sample(&mut self, snapshot: &Snapshot, snapshot_ns: i64) {
        let capture = snapshot
            .inputs
            .get(self.input)
            .and_then(|i| i.samples.last())
            .and_then(|s| s.capture_ns);
        if let Some(capture) = capture {
            self.record(snapshot_ns - capture);
        }
    }

    /// Records one sample age (snapshot minus capture).
    pub fn record(&mut self, age_ns: i64) {
        let phase = age_ns.rem_euclid(self.period_ns);
        self.errors.push(self.centered(phase - self.margin()));
    }

    /// Moves the grid once a window is full; returns the change.
    pub fn update(&mut self) -> i64 {
        // Every one of the last few starts read the previous sample: the
        // publisher's phase moved later (its loop stalled). Follow now
        // rather than at the end of the window, by the smallest of them.
        let recent = self.errors.len().checked_sub(Self::LATE);
        if let Some(from) = recent.filter(|_| self.aligned)
            && self.errors[from..]
                .iter()
                .all(|e| (-self.period_ns / 4..-Self::DEADBAND_NS).contains(e))
        {
            let change = self.errors[from..].iter().map(|e| -e).min().unwrap_or(0);
            self.errors.clear();
            self.offset_ns += change;
            return change;
        }
        let window = if self.aligned {
            Self::WINDOW
        } else {
            Self::FIRST
        };
        if self.errors.len() < window {
            return 0;
        }
        self.errors.sort_unstable();
        // The freshest tenth: the samples that land latest set the phase,
        // so jitter in their arrival never puts one after the start.
        let error = self.errors[self.errors.len() / 10];
        self.errors.clear();
        // A phase above the margin: start that much earlier, which is the
        // same as starting `P − error` later on the next sample.
        let p = self.period_ns;
        let change = if (-p / 4..0).contains(&error) {
            // The samples land just after the starts: start later, in full.
            // Each start reads a sample almost a period old until then, and
            // the gap this opens in the chunk coverage is under P/4.
            -error
        } else if error < 0 {
            // Far later is the same phase as earlier: start sooner (which
            // never opens a gap), at most P/2 per window.
            (-(error + p)).max(-p / 2)
        } else if self.aligned {
            // Starts later than they need be: close in by halves.
            (-error / 2).max(-p / 4)
        } else {
            -error
        };
        self.aligned = true;
        if change.abs() < Self::DEADBAND_NS {
            return 0;
        }
        self.offset_ns += change;
        change
    }

    /// `x` wrapped into `(−P/2, P/2]`.
    fn centered(&self, x: i64) -> i64 {
        let p = self.period_ns;
        let w = x.rem_euclid(p);
        if w > p / 2 { w - p } else { w }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    #[test]
    fn slots_falling_while_busy_are_skipped_never_replayed() {
        let mut s = StartSchedule::new(100.0);
        assert_eq!(s.next_start(0), 0);
        s.start(0);
        // predict took 35 ms: slots 10, 20 and 30 fell while it ran.
        assert_eq!(s.pending(20 * MS), Some(Duration::from_millis(20)));
        assert!(s.finish(35 * MS).is_none());
        assert_eq!(s.stats().skipped(SkipReason::Busy), 3);
        assert_eq!(s.next_start(35 * MS), 40 * MS, "the next slot ahead");
        s.start(40 * MS);
        s.finish(41 * MS);
        assert_eq!(s.next_start(41 * MS), 50 * MS);
        assert_eq!(s.stats().skipped(SkipReason::Busy), 3);
        assert_eq!((s.stats().starts, s.stats().completions), (2, 2));
        assert_eq!(s.stats().latency_max, Some(Duration::from_millis(35)));
    }

    #[test]
    fn skipped_slots_keep_their_reason_and_a_late_start_counts_busy() {
        let mut s = StartSchedule::new(100.0);
        s.next_start(0);
        s.skip(MS, SkipReason::Waiting);
        assert_eq!(s.next_start(MS), 10 * MS);
        s.skip(10 * MS, SkipReason::NoLease);
        // The loop woke 25 ms late for slot 20: slots 20 and 30 are gone.
        s.start(45 * MS);
        s.finish(46 * MS);
        let stats = s.stats();
        assert_eq!(stats.skipped(SkipReason::Waiting), 1);
        assert_eq!(stats.skipped(SkipReason::NoLease), 1);
        assert_eq!(stats.skipped(SkipReason::Busy), 2);
        assert_eq!(s.next_start(46 * MS), 50 * MS);
    }

    #[test]
    fn a_slow_model_drifts_once_then_recovers_once() {
        let mut s = StartSchedule::new(20.0);
        let mut events = Vec::new();
        let mut now = s.next_start(0);
        // Healthy for 2 s: 5 ms inferences at 20 Hz.
        while now < 2_000 * MS {
            s.start(now);
            events.extend(s.finish(now + 5 * MS));
            now = s.next_start(now + 5 * MS);
        }
        assert!(events.is_empty(), "{events:?}");
        // 80 ms inferences: one completion per two slots, 10 Hz.
        while now < 5_000 * MS {
            s.start(now);
            events.extend(s.finish(now + 80 * MS));
            now = s.next_start(now + 80 * MS);
        }
        // Back to healthy.
        while now < 8_000 * MS {
            s.start(now);
            events.extend(s.finish(now + 5 * MS));
            now = s.next_start(now + 5 * MS);
        }
        let drift: Vec<(bool, &str)> = events
            .iter()
            .map(|e| match e {
                RunEvent::RateDrift {
                    drifting, cause, ..
                } => (*drifting, *cause),
                RunEvent::InferencePending { .. } => (false, "pending"),
            })
            .collect();
        assert_eq!(drift.len(), 2, "one event per transition: {events:?}");
        assert_eq!(drift[0], (true, "busy"));
        assert!(!drift[1].0);
        assert!(!s.stats().drifting);
    }

    #[test]
    fn silent_intervals_count_and_warming_is_not_drift() {
        let mut s = StartSchedule::new(10.0);
        let mut now = s.next_start(0);
        // Waiting for inputs for 3 s: no drift before the first start.
        while now < 3_000 * MS {
            assert!(s.skip(now, SkipReason::Waiting).is_none());
            now = s.next_start(now);
        }
        while now < 5_000 * MS {
            s.start(now);
            assert!(s.finish(now + MS).is_none());
            now = s.next_start(now + MS);
        }
        // Inputs go stale: starts are skipped, and the silence is drift.
        let mut events = Vec::new();
        while now < 7_000 * MS {
            events.extend(s.skip(now, SkipReason::StaleInput));
            now = s.next_start(now);
        }
        assert!(matches!(
            events.as_slice(),
            [RunEvent::RateDrift {
                drifting: true,
                cause: "stale_input",
                ..
            }]
        ));
    }

    /// Starts on a 10 ms grid shifted by `offset`; samples published every
    /// 10 ms at `publish_phase` (+ `jitter(k)`): the newest sample's age at
    /// each start.
    fn ages(
        offset: i64,
        publish_phase: i64,
        jitter: impl Fn(i64) -> i64,
        from: i64,
    ) -> impl Iterator<Item = i64> {
        (from..).map(move |k| {
            let start = k * 10 * MS + offset;
            // The newest publication at or before the start.
            let mut j = (start - publish_phase).div_euclid(10 * MS) + 1;
            while j * 10 * MS + publish_phase + jitter(j) > start {
                j -= 1;
            }
            start - (j * 10 * MS + publish_phase + jitter(j))
        })
    }

    #[test]
    fn a_start_just_before_each_sample_moves_to_just_after_it() {
        let mut align = PhaseAlign::new(0, 10 * MS);
        // Published 0.4 ms after each start: snapshots read a 9.6 ms old sample.
        for age in ages(0, 400_000, |_| 0, 1).take(PhaseAlign::FIRST) {
            assert_eq!(age, 9_600_000);
            align.record(age);
        }
        let change = align.update();
        assert_eq!(change, 1_400_000, "later, the short way");
        for age in ages(align.offset_ns, 400_000, |_| 0, 100).take(PhaseAlign::WINDOW) {
            assert_eq!(age, MS, "one margin old");
            align.record(age);
        }
        assert_eq!(align.update(), 0, "aligned: nothing to correct");
    }

    #[test]
    fn jitter_is_ignored_and_slow_drift_is_followed() {
        let mut align = PhaseAlign::new(0, 10 * MS);
        // ±0.5 ms publication jitter around a 3 ms phase.
        let jitter = |j: i64| if j % 2 == 0 { 500_000 } else { -500_000 };
        for age in ages(0, 3 * MS, jitter, 1).take(PhaseAlign::WINDOW) {
            align.record(age);
        }
        align.update();
        let mut worst = 0;
        // The publisher drifts 20 µs per window; the aligner keeps up.
        for round in 0..50 {
            let phase = 3 * MS + round * 20_000;
            let from = 1000 + round * 100;
            for age in ages(align.offset_ns, phase, jitter, from).take(PhaseAlign::WINDOW) {
                worst = worst.max(age);
                align.record(age);
            }
            align.update();
        }
        assert!(worst < 3 * MS, "ages stay near the margin, worst {worst}");
    }

    #[test]
    fn nothing_moves_before_a_window_is_full() {
        let mut align = PhaseAlign::new(0, 10 * MS);
        for _ in 0..PhaseAlign::FIRST - 1 {
            align.record(9 * MS);
        }
        assert_eq!(align.update(), 0);
        assert_eq!(align.offset_ns, 0);
    }

    #[test]
    fn a_far_phase_is_reached_by_starting_earlier() {
        let mut align = PhaseAlign::new(0, 10 * MS);
        // 4 ms old: 3 ms earlier, not 7 ms later.
        for _ in 0..PhaseAlign::FIRST {
            align.record(4 * MS);
        }
        assert_eq!(align.update(), -3 * MS);
        // 6.5 ms after the margin is 3.5 ms before the next sample's: still
        // earlier, by at most half a period per window.
        let mut align = PhaseAlign::new(0, 10 * MS);
        for _ in 0..PhaseAlign::FIRST {
            align.record(7_500_000);
        }
        assert_eq!(align.update(), -5 * MS);
    }

    #[test]
    fn a_publisher_that_jumps_later_is_followed_within_a_few_starts() {
        let mut align = PhaseAlign::new(0, 10 * MS);
        for age in ages(0, 8 * MS, |_| 0, 1).take(PhaseAlign::FIRST) {
            align.record(age);
        }
        assert_eq!(align.update(), -MS, "2 ms old: 1 ms earlier");
        // The publisher's loop slips 2 ms: every start now reads the
        // previous sample, 9 ms old, until the aligner follows.
        let mut moved = 0;
        for (n, age) in ages(align.offset_ns, 10 * MS, |_| 0, 100)
            .take(PhaseAlign::LATE)
            .enumerate()
        {
            assert_eq!(age, 9 * MS);
            align.record(age);
            moved = align.update();
            if n + 1 < PhaseAlign::LATE {
                assert_eq!(moved, 0);
            }
        }
        assert_eq!(moved, 2 * MS, "later, in full, after a few starts");
        for age in ages(align.offset_ns, 10 * MS, |_| 0, 200).take(PhaseAlign::WINDOW) {
            assert_eq!(age, MS);
        }
    }
}
