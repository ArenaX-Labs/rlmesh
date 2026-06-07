use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub(crate) struct PhaseTiming {
    pub(crate) count: u64,
    pub(crate) total: Duration,
    pub(crate) min: Option<Duration>,
    pub(crate) max: Duration,
}

impl PhaseTiming {
    pub(crate) fn record(&mut self, duration: Duration) {
        self.count += 1;
        self.total += duration;
        self.min = Some(match self.min {
            Some(min) => min.min(duration),
            None => duration,
        });
        self.max = self.max.max(duration);
    }

    pub(crate) fn avg_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total.as_secs_f64() * 1000.0 / self.count as f64
        }
    }

    pub(crate) fn total_ms(&self) -> f64 {
        self.total.as_secs_f64() * 1000.0
    }

    pub(crate) fn min_ms(&self) -> f64 {
        self.min
            .map(|duration| duration.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }

    pub(crate) fn max_ms(&self) -> f64 {
        self.max.as_secs_f64() * 1000.0
    }
}
