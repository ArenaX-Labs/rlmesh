//! Clock domains and NTP-style offset estimation (design §3.1-3.2).
//!
//! Every participant stamps with its own monotonic clock. In wall mode the
//! session clock is the host's monotonic clock, and the host keeps, per
//! participant, an offset estimate with a rigorous error bound:
//! `session_ns = participant_ns + offset_ns`, within `± error_ns`.
//!
//! Virtual and lockstep session clocks are later work (wave 5).

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use rlmesh_wire::{ClockMapping, ParticipantId, TimingSample};

/// A monotonic clock in nanoseconds. Each participant owns one clock domain.
pub trait Clock: Send + Sync + 'static {
    /// The current time in this clock domain, in nanoseconds.
    fn now_ns(&self) -> i64;
}

/// `std::time::Instant` measured from an anchor, plus a fixed shift.
///
/// The shift exists so one process can host several clock domains with known
/// offsets between them (tests and the demo); production participants use
/// [`MonotonicClock::new`].
#[derive(Clone, Debug)]
pub struct MonotonicClock {
    anchor: Instant,
    shift_ns: i64,
}

impl MonotonicClock {
    /// A clock anchored now, reading 0 at creation.
    pub fn new() -> Self {
        Self::with_anchor(Instant::now(), 0)
    }

    /// Reads `anchor.elapsed() + shift_ns`: two clocks sharing an anchor differ
    /// by exactly the difference of their shifts.
    pub fn with_anchor(anchor: Instant, shift_ns: i64) -> Self {
        Self { anchor, shift_ns }
    }

    /// As a shared trait object.
    pub fn shared(self) -> Arc<dyn Clock> {
        Arc::new(self)
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MonotonicClock {
    fn now_ns(&self) -> i64 {
        let elapsed = i64::try_from(self.anchor.elapsed().as_nanos()).unwrap_or(i64::MAX);
        elapsed.saturating_add(self.shift_ns)
    }
}

/// The host's raw `CLOCK_MONOTONIC`: one clock domain shared by every
/// process on a machine, so cross-process latencies on one host need no
/// offset estimation (benchmarks, co-located edge and controller). Off Unix
/// it falls back to a per-process clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemMonotonicClock;

impl SystemMonotonicClock {
    /// As a shared trait object.
    pub fn shared(self) -> Arc<dyn Clock> {
        Arc::new(self)
    }
}

#[cfg(unix)]
impl Clock for SystemMonotonicClock {
    // clock_gettime into a local timespec; the field types differ by target
    // (`time_t` is 32 bits on some), hence the widening conversions.
    #[allow(unsafe_code, clippy::useless_conversion)]
    fn now_ns(&self) -> i64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `ts` is a valid, writable timespec and CLOCK_MONOTONIC is
        // always available on the platforms this crate builds for.
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts) };
        debug_assert_eq!(rc, 0);
        i64::from(ts.tv_sec)
            .saturating_mul(1_000_000_000)
            .saturating_add(i64::from(ts.tv_nsec))
    }
}

#[cfg(not(unix))]
impl Clock for SystemMonotonicClock {
    fn now_ns(&self) -> i64 {
        use std::sync::OnceLock;
        static ANCHOR: OnceLock<Instant> = OnceLock::new();
        let anchor = ANCHOR.get_or_init(Instant::now);
        i64::try_from(anchor.elapsed().as_nanos()).unwrap_or(i64::MAX)
    }
}

/// How offsets are filtered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OffsetEstimatorConfig {
    /// Samples kept; the estimate comes from the best of them.
    pub window: usize,
    /// Assumed worst-case relative drift between two monotonic clocks. Linux
    /// `CLOCK_MONOTONIC` is NTP-slewed by at most 500 ppm; free-running
    /// crystals are typically within ±50 ppm.
    pub drift_bound_ppm: f64,
}

impl Default for OffsetEstimatorConfig {
    fn default() -> Self {
        Self {
            window: 32,
            drift_bound_ppm: 100.0,
        }
    }
}

/// The current mapping of one participant's clock onto the session clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OffsetEstimate {
    /// session_ns = participant_ns + offset_ns.
    pub offset_ns: i64,
    /// |true offset − offset_ns| ≤ error_ns, provided drift stays within the
    /// configured bound.
    pub error_ns: u64,
    /// δ of the sample the estimate came from.
    pub rtt_ns: u64,
    /// Host clock (t3) of that sample.
    pub sample_ns: i64,
}

impl OffsetEstimate {
    /// The published mapping of `participant`'s clock.
    pub fn mapping(&self, participant: ParticipantId, drift_bound_ppm: f64) -> ClockMapping {
        ClockMapping {
            participant,
            offset_ns: self.offset_ns,
            error_ns: self.error_ns,
            rtt_ns: self.rtt_ns,
            sample_ns: self.sample_ns,
            drift_bound_ppm,
        }
    }

    /// The estimate a published mapping carries.
    pub fn from_mapping(mapping: &ClockMapping) -> Self {
        Self {
            offset_ns: mapping.offset_ns,
            error_ns: mapping.error_ns,
            rtt_ns: mapping.rtt_ns,
            sample_ns: mapping.sample_ns,
        }
    }
}

/// Minimum-δ filter (NTP clock-filter style) over a window of exchanges.
///
/// The error bound is ε = ⌈δ/2⌉ + 1 + ⌈drift × (age + δ)⌉: half the round trip
/// (delay asymmetry), one nanosecond of flooring, and the drift accumulated
/// since the sample. The best sample is the one with the smallest bound now,
/// so a low-δ sample is kept until its age makes a fresher one better.
// TODO: a participant declaring `ptp_synced` should take ε from PTP instead.
#[derive(Clone, Debug)]
pub struct OffsetEstimator {
    config: OffsetEstimatorConfig,
    window: VecDeque<TimingSample>,
}

impl OffsetEstimator {
    /// An empty estimator.
    pub fn new(config: OffsetEstimatorConfig) -> Self {
        Self {
            config,
            window: VecDeque::with_capacity(config.window.max(1)),
        }
    }

    /// Its configuration.
    pub fn config(&self) -> OffsetEstimatorConfig {
        self.config
    }

    /// Adds an exchange. Returns false (and ignores it) when it is invalid or
    /// a duplicate of the newest sample.
    pub fn add(&mut self, sample: TimingSample) -> bool {
        if !sample.is_valid()
            || self
                .window
                .back()
                .is_some_and(|last| last.seq >= sample.seq)
        {
            return false;
        }
        if self.window.len() == self.config.window.max(1) {
            self.window.pop_front();
        }
        self.window.push_back(sample);
        true
    }

    /// Samples in the window.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// Whether no sample has arrived.
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// The bound a sample gives at host time `now_ns`.
    fn bound_ns(&self, sample: &TimingSample, now_ns: i64) -> u64 {
        let rtt = sample.rtt_ns().max(0).unsigned_abs();
        let age = now_ns.saturating_sub(sample.t3_ns).max(0).unsigned_abs();
        let drift = (self.config.drift_bound_ppm.max(0.0) * 1e-6 * (age + rtt) as f64).ceil();
        rtt.div_ceil(2)
            .saturating_add(1)
            .saturating_add(drift as u64)
    }

    /// The best estimate at host time `now_ns`, if any sample has arrived.
    pub fn estimate(&self, now_ns: i64) -> Option<OffsetEstimate> {
        let best = self
            .window
            .iter()
            .min_by_key(|sample| self.bound_ns(sample, now_ns))?;
        Some(OffsetEstimate {
            offset_ns: best.offset_ns(),
            error_ns: self.bound_ns(best, now_ns),
            rtt_ns: best.rtt_ns().max(0).unsigned_abs(),
            sample_ns: best.t3_ns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A participant clock at `participant = host − true_offset (+ drift)`,
    /// exchanging over a link with the given one-way delays.
    fn exchange(
        seq: u64,
        host_send_ns: i64,
        true_offset_ns: i64,
        out_ns: i64,
        hold_ns: i64,
        back_ns: i64,
    ) -> TimingSample {
        // t1 participant clock at host time host_send_ns.
        let t1 = host_send_ns - true_offset_ns;
        let t2 = host_send_ns + out_ns;
        let t3 = t2 + hold_ns;
        let t4 = (t3 + back_ns) - true_offset_ns;
        TimingSample {
            seq,
            t1_ns: t1,
            t2_ns: t2,
            t3_ns: t3,
            t4_ns: t4,
        }
    }

    #[test]
    fn symmetric_delay_recovers_the_offset_exactly() {
        let s = exchange(1, 1_000_000, 5_000_000_000, 300_000, 20_000, 300_000);
        assert_eq!(s.offset_ns(), 5_000_000_000);
        assert_eq!(s.rtt_ns(), 600_000);
    }

    #[test]
    fn asymmetric_delay_stays_within_half_the_round_trip() {
        let true_offset = -7_654_321_i64;
        // Outbound 9 ms, back 1 ms: the midpoint is off by 4 ms, δ/2 = 5 ms.
        let s = exchange(1, 50_000_000, true_offset, 9_000_000, 50_000, 1_000_000);
        let mut est = OffsetEstimator::new(OffsetEstimatorConfig::default());
        assert!(est.add(s));
        let e = est.estimate(s.t3_ns).expect("estimate");
        assert_eq!(e.rtt_ns, 10_000_000);
        let err = (e.offset_ns - true_offset).unsigned_abs();
        assert_eq!(err, 4_000_000);
        assert!(err <= e.error_ns, "{err} > {}", e.error_ns);
    }

    #[test]
    fn the_window_minimum_wins_and_the_bound_holds_for_every_sample() {
        let true_offset = 123_456_789_i64;
        let mut est = OffsetEstimator::new(OffsetEstimatorConfig {
            window: 16,
            drift_bound_ppm: 0.0,
        });
        // Deterministic pseudo-random asymmetric delays, one clean exchange.
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 5_000_000) as i64
        };
        for seq in 1..=16 {
            let (out, back) = if seq == 9 {
                (40_000, 60_000)
            } else {
                (200_000 + next(), 200_000 + next())
            };
            let s = exchange(seq, seq as i64 * 10_000_000, true_offset, out, 10_000, back);
            assert!(est.add(s));
            let now = s.t3_ns;
            let e = est.estimate(now).expect("estimate");
            assert!(
                (e.offset_ns - true_offset).unsigned_abs() <= e.error_ns,
                "bound violated at seq {seq}: {e:?}"
            );
        }
        let e = est.estimate(170_000_000).expect("estimate");
        assert_eq!(e.rtt_ns, 100_000, "the δ-minimum sample is chosen");
        assert!((e.offset_ns - true_offset).unsigned_abs() <= 10_000);
    }

    #[test]
    fn drift_grows_the_bound_with_sample_age_and_still_covers_the_truth() {
        let ppm = 50.0;
        let config = OffsetEstimatorConfig {
            window: 8,
            drift_bound_ppm: ppm,
        };
        let mut est = OffsetEstimator::new(config);
        // The participant clock runs 50 ppm slow: offset grows 50 ns per ms.
        let offset_at = |host_ns: i64| 1_000_000 + (host_ns as f64 * ppm * 1e-6) as i64;
        let s = exchange(1, 0, offset_at(0), 100_000, 0, 100_000);
        assert!(est.add(s));
        let fresh = est.estimate(s.t3_ns).expect("estimate");
        let later_ns = 2_000_000_000; // 2 s later: +100 µs of drift.
        let old = est.estimate(later_ns).expect("estimate");
        // ~2 s × 50 ppm = 100 µs more (less the 100 µs the fresh sample had aged).
        assert!(
            old.error_ns - fresh.error_ns >= 99_990,
            "{old:?} vs {fresh:?}"
        );
        assert!((old.offset_ns - offset_at(later_ns)).unsigned_abs() <= old.error_ns);
    }

    #[test]
    fn fresher_samples_replace_aged_ones() {
        let mut est = OffsetEstimator::new(OffsetEstimatorConfig {
            window: 8,
            drift_bound_ppm: 100.0,
        });
        // A very clean sample at t=0, then a noisier one 10 s later.
        assert!(est.add(exchange(1, 0, 0, 10_000, 0, 10_000)));
        assert!(est.add(exchange(2, 10_000_000_000, 0, 100_000, 0, 100_000)));
        let e = est.estimate(10_000_200_000).expect("estimate");
        // Aged: 10_000 + 1 + 1 ms of drift; fresh: 100_000 + 1 + ~20 ns.
        assert_eq!(e.rtt_ns, 200_000);
    }

    #[test]
    fn invalid_and_duplicate_samples_are_ignored() {
        let mut est = OffsetEstimator::new(OffsetEstimatorConfig::default());
        let s = exchange(3, 0, 0, 10, 0, 10);
        assert!(est.add(s));
        assert!(!est.add(s), "duplicate seq");
        let mut backwards = exchange(4, 0, 0, 10, 0, 10);
        backwards.t4_ns = backwards.t1_ns - 1;
        assert!(!est.add(backwards));
        assert_eq!(est.len(), 1);
    }

    #[test]
    fn clocks_sharing_an_anchor_differ_by_their_shift() {
        let anchor = Instant::now();
        let a = MonotonicClock::with_anchor(anchor, 0);
        let b = MonotonicClock::with_anchor(anchor, -3_000_000_000);
        // b is read after a, so the difference can only fall short of 3 s.
        let (ta, tb) = (a.now_ns(), b.now_ns());
        let diff = ta - tb;
        assert!((2_990_000_000..=3_000_000_000).contains(&diff), "{diff}");
    }
}
