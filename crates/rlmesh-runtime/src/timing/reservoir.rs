use std::time::Duration;

/// Default reservoir capacity for session-lifetime samples.
///
/// Sessions can run unbounded (`max_episodes: None`), so the cumulative
/// (`total_*`) sample series must not grow without limit. A reservoir keeps a
/// fixed-size, statistically representative subsample so that summary
/// percentiles stay accurate while memory stays constant.
pub(super) const DEFAULT_RESERVOIR_CAPACITY: usize = 8192;

/// Fixed-capacity reservoir of samples (Vitter's Algorithm R).
///
/// Up to `capacity` samples are retained verbatim; beyond that, each new sample
/// replaces a uniformly-random existing slot with probability `capacity / seen`,
/// yielding a uniform random subsample of the full stream. This bounds memory
/// to `capacity` while keeping percentile/average estimates representative of
/// the whole stream.
///
/// Randomness comes from a small deterministic xorshift generator so the crate
/// needs no `rand` dependency; the bias-free guarantee only requires the
/// replacement decisions be uniform, not cryptographically secure.
#[derive(Debug, Clone)]
pub(super) struct Reservoir<T> {
    samples: Vec<T>,
    capacity: usize,
    seen: u64,
    rng_state: u64,
}

/// Reservoir specialized to `Duration` latency samples.
pub(super) type DurationReservoir = Reservoir<Duration>;

/// Reservoir specialized to floating-point metric samples (byte counts /
/// generic numbers exposed via OperationTelemetry).
pub(super) type ValueReservoir = Reservoir<f64>;

impl<T: Copy> Reservoir<T> {
    pub(super) fn new(capacity: usize) -> Self {
        debug_assert!(capacity > 0, "reservoir capacity must be non-zero");
        Self {
            samples: Vec::new(),
            capacity: capacity.max(1),
            seen: 0,
            // Fixed non-zero seed: reproducible across runs, distinct from 0
            // (xorshift degenerates on a zero state).
            rng_state: 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub(super) fn push(&mut self, sample: T) {
        self.seen += 1;
        if self.samples.len() < self.capacity {
            self.samples.push(sample);
            return;
        }
        // Replace a random slot with probability capacity / seen.
        let slot = (self.next_u64() % self.seen) as usize;
        if slot < self.capacity {
            self.samples[slot] = sample;
        }
    }

    pub(super) fn samples(&self) -> &[T] {
        &self.samples
    }

    /// Total number of samples observed, including those evicted from the
    /// reservoir. Use this for reporting counts, not `samples().len()`.
    pub(super) fn seen(&self) -> u64 {
        self.seen
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

impl<T: Copy> Default for Reservoir<T> {
    fn default() -> Self {
        Self::new(DEFAULT_RESERVOIR_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_all_samples_below_capacity() {
        let mut reservoir = DurationReservoir::new(4);
        reservoir.push(Duration::from_millis(1));
        reservoir.push(Duration::from_millis(2));
        assert_eq!(reservoir.samples().len(), 2);
        assert_eq!(reservoir.seen(), 2);
    }

    #[test]
    fn bounds_memory_above_capacity() {
        let mut reservoir = DurationReservoir::new(4);
        for ms in 0..10_000u64 {
            reservoir.push(Duration::from_millis(ms));
        }
        assert_eq!(reservoir.samples().len(), 4);
        assert_eq!(reservoir.seen(), 10_000);
    }

    #[test]
    fn value_reservoir_bounds_memory() {
        let mut reservoir = ValueReservoir::new(4);
        for value in 0..10_000u64 {
            reservoir.push(value as f64);
        }
        assert_eq!(reservoir.samples().len(), 4);
        assert_eq!(reservoir.seen(), 10_000);
    }
}
