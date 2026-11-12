//! NTP-style clock offset estimation with an error bound,
//! sans-IO.
//!
//! In wall mode the session clock is the host's monotonic clock, and the
//! host keeps, per participant, an offset estimate with a rigorous error
//! bound: `session_ns = participant_ns + offset_ns`, within `± error_ns`.
//! Every exchange and every query is passed in; nothing here reads a clock.

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::VecDeque;

use crate::wire::{ClockMapping, ParticipantId, TimingSample};

/// How offsets are filtered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OffsetEstimatorConfig {
    /// Samples kept; the estimate comes from the best of them.
    pub window: usize,
    /// Assumed worst-case relative drift between two monotonic clocks.
    /// Linux `CLOCK_MONOTONIC` is NTP-slewed by at most 500 ppm, so that is
    /// the default; free-running crystals are typically within ±50 ppm.
    pub drift_bound_ppm: f64,
}

impl Default for OffsetEstimatorConfig {
    fn default() -> Self {
        Self {
            window: 32,
            drift_bound_ppm: 500.0,
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
/// so a low-δ sample is kept until its age makes a fresher one better: also
/// past the window, which a burst of slow exchanges (a link that stalled,
/// then delivered its queued heartbeats at once) would otherwise empty of
/// every good one. Its bound grows with its age, so it holds as it did.
// TODO: a participant declaring `ptp_synced` should take ε from PTP instead.
#[derive(Clone, Debug)]
pub struct OffsetEstimator {
    config: OffsetEstimatorConfig,
    window: VecDeque<TimingSample>,
    /// The best sample the window let go of, while none in it is better.
    held: Option<TimingSample>,
}

impl OffsetEstimator {
    /// An empty estimator.
    pub fn new(config: OffsetEstimatorConfig) -> Self {
        Self {
            config,
            window: VecDeque::with_capacity(config.window.max(1)),
            held: None,
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
        if self.window.len() == self.config.window.max(1)
            && let Some(out) = self.window.pop_front()
        {
            // Kept while it is the best there is, by its bound now.
            let now = sample.t3_ns;
            let bound = |s: &TimingSample| self.bound_ns(s, now);
            let best_left = self.window.iter().chain([&sample]).map(bound).min();
            let candidate = match self.held {
                Some(held) if bound(&held) <= bound(&out) => held,
                _ => out,
            };
            self.held = best_left
                .is_none_or(|b| bound(&candidate) < b)
                .then_some(candidate);
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
            .chain(&self.held)
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

    /// A link that stalled delivers its queued heartbeats at once, each
    /// with the stall in its round trip: more of them than the window holds.
    /// The clean exchange before the stall still bounds the offset, by its
    /// round trip and the drift since; the stalled ones never could.
    #[test]
    fn a_burst_of_slow_exchanges_keeps_the_best_one_before_it() {
        let true_offset = 42_000_000_i64;
        let mut est = OffsetEstimator::new(OffsetEstimatorConfig {
            window: 8,
            drift_bound_ppm: 500.0,
        });
        // 40 ms round trip, then 20 that each waited out a 1.5 s stall.
        assert!(est.add(exchange(1, 0, true_offset, 20_000_000, 0, 20_000_000)));
        for seq in 2..22 {
            let at = 100_000_000 + seq * 1_000_000;
            assert!(est.add(exchange(
                seq as u64,
                at,
                true_offset,
                1_500_000_000,
                0,
                1_000_000
            )));
        }
        let now = 2_000_000_000;
        let e = est.estimate(now).expect("estimate");
        // 20 ms of half the round trip and 1 ms of 2 s of drift, not 750.
        assert_eq!(e.rtt_ns, 40_000_000);
        assert!(e.error_ns < 22_000_000, "{e:?}");
        assert!((e.offset_ns - true_offset).unsigned_abs() <= e.error_ns);
        // A fresher clean one takes over from it.
        assert!(est.add(exchange(30, now, true_offset, 5_000_000, 0, 5_000_000)));
        assert_eq!(
            est.estimate(now + 10_000_000).expect("estimate").rtt_ns,
            10_000_000
        );
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
}
