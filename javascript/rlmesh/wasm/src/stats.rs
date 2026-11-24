//! Per-stream delivery measurements in the class Worker: rate, latency and
//! age percentiles and the share of late samples. Measurement only: the
//! receive rules themselves (epochs, gaps, layout, freshness) are
//! `rlmesh_session_core::stream`'s.

const BUCKET_NS: i64 = 5_000;
/// Buckets: 5 µs each up to 50 ms; beyond is the overflow bucket.
const BUCKETS: usize = 10_000;

#[derive(Clone, Debug)]
pub struct Histogram {
    counts: Vec<u32>,
    overflow: u64,
    total: u64,
    max_ns: i64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            counts: vec![0; BUCKETS],
            overflow: 0,
            total: 0,
            max_ns: 0,
        }
    }
}

impl Histogram {
    pub fn record(&mut self, ns: i64) {
        let ns = ns.max(0);
        self.total += 1;
        self.max_ns = self.max_ns.max(ns);
        match usize::try_from(ns / BUCKET_NS) {
            Ok(i) if i < BUCKETS => self.counts[i] = self.counts[i].saturating_add(1),
            _ => self.overflow += 1,
        }
    }

    pub fn count(&self) -> u64 {
        self.total
    }

    /// The `q` quantile in ms (the bucket's upper edge; the max past the
    /// last bucket).
    pub fn quantile_ms(&self, q: f64) -> f64 {
        if self.total == 0 {
            return f64::NAN;
        }
        let rank = ((q * self.total as f64).ceil() as u64).clamp(1, self.total);
        let mut seen = 0_u64;
        for (i, n) in self.counts.iter().enumerate() {
            seen += u64::from(*n);
            if seen >= rank {
                let edge = (i as i64 + 1) * BUCKET_NS;
                return edge.min(self.max_ns.max(i as i64 * BUCKET_NS)) as f64 / 1e6;
            }
        }
        self.max_ns as f64 / 1e6
    }

    pub fn above_pct(&self, threshold_ns: i64) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let first = usize::try_from((threshold_ns / BUCKET_NS).max(0)).unwrap_or(BUCKETS);
        let above: u64 = self
            .counts
            .iter()
            .skip(first)
            .map(|n| u64::from(*n))
            .sum::<u64>()
            + self.overflow;
        100.0 * above as f64 / self.total as f64
    }

    pub fn max_ms(&self) -> f64 {
        self.max_ns as f64 / 1e6
    }
}

#[derive(Clone, Debug, Default)]
pub struct Delivery {
    pub latency: Histogram,
    pub age: Histogram,
    first_ns: Option<i64>,
    last_ns: i64,
}

impl Delivery {
    pub fn record(&mut self, receive_ns: i64, latency_ns: i64, age_ns: i64) {
        self.first_ns.get_or_insert(receive_ns);
        self.last_ns = receive_ns;
        self.latency.record(latency_ns);
        self.age.record(age_ns);
    }

    pub fn rate_hz(&self) -> f64 {
        let n = self.latency.count();
        match self.first_ns {
            Some(first) if n > 1 && self.last_ns > first => {
                (n - 1) as f64 * 1e9 / (self.last_ns - first) as f64
            }
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_late_share_and_rate() {
        let mut d = Delivery::default();
        for i in 0..1000 {
            let latency = if i % 100 == 0 { 3_000_000 } else { 200_000 };
            d.record(i * 1_000_000, latency, latency + 50_000);
        }
        assert!((d.rate_hz() - 1000.0).abs() < 1e-6);
        assert!((d.latency.quantile_ms(0.5) - 0.205).abs() < 0.006);
        assert!((d.latency.quantile_ms(0.999) - 3.0).abs() < 0.006);
        assert!((d.latency.above_pct(2_000_000) - 1.0).abs() < 1e-9);
        assert_eq!(d.latency.max_ms(), 3.0);
        assert!(Histogram::default().quantile_ms(0.5).is_nan());
    }
}
