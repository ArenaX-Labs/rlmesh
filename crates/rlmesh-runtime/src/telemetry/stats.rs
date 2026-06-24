/// Aggregated statistics over a sample set, all in the sample's unit.
pub(super) struct Summary {
    pub avg: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Average + the p50/p95/p99 nearest-rank percentiles over `samples`, computed
/// with a SINGLE sort (the percentiles share one sorted buffer). `None` for an
/// empty slice — distinct from a real `0.0`.
pub(super) fn summary(samples: &[f64]) -> Option<Summary> {
    if samples.is_empty() {
        return None;
    }
    let avg = samples.iter().sum::<f64>() / samples.len() as f64;
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(Summary {
        avg,
        p50: nearest_rank(&sorted, 0.50),
        p95: nearest_rank(&sorted, 0.95),
        p99: nearest_rank(&sorted, 0.99),
    })
}

/// Nearest-rank percentile over an already-sorted, non-empty slice.
fn nearest_rank(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * percentile).ceil() as usize;
    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_is_ordered_and_handles_empty() {
        assert!(summary(&[]).is_none());

        let s = summary(&[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert!((s.avg - 25.0).abs() < 1e-9);
        assert!(s.p50 <= s.p95 && s.p95 <= s.p99);
        assert_eq!(s.p95, 40.0);
        assert_eq!(s.p99, 40.0);
    }
}
