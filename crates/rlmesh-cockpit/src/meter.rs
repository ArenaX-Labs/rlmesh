//! Rate meters, sans-IO: every time is passed in.
//!
//! - [`RateMeter`]: events per second over a sliding window, on one clock,
//!   counting silence (a stream that stops reads as slowing to zero, not as
//!   its last rate).
//! - [`CounterRate`]: the rate of a cumulative counter someone else reports
//!   (publisher and controller counters, heartbeat seqs), on the reporter's
//!   clock, and unknown once its reports stop arriving.

use std::collections::VecDeque;
use std::time::Duration;

fn ns(d: Duration) -> i64 {
    i64::try_from(d.as_nanos()).unwrap_or(i64::MAX)
}

/// The §8.10 monitoring window for a stream expected at `hz`:
/// `max(1 s, 4 / hz)`.
pub fn window_for(hz: f64) -> Duration {
    let slow = if hz.is_finite() && hz > 0.0 {
        Duration::from_secs_f64((4.0 / hz).min(3600.0))
    } else {
        Duration::ZERO
    };
    slow.max(Duration::from_secs(1))
}

/// Events per second over a sliding window.
#[derive(Clone, Debug)]
pub struct RateMeter {
    window_ns: i64,
    times: VecDeque<i64>,
    first_ns: Option<i64>,
    total: u64,
}

impl RateMeter {
    /// A meter over `window`.
    pub fn new(window: Duration) -> Self {
        Self {
            window_ns: ns(window).max(1),
            times: VecDeque::new(),
            first_ns: None,
            total: 0,
        }
    }

    /// One event at `t_ns`. Times should not go backwards; one that does
    /// counts at the newest time seen.
    pub fn record(&mut self, t_ns: i64) {
        let t_ns = self.times.back().map_or(t_ns, |&last| t_ns.max(last));
        self.first_ns.get_or_insert(t_ns);
        self.times.push_back(t_ns);
        self.total += 1;
        while self
            .times
            .front()
            .is_some_and(|&front| front <= t_ns - self.window_ns)
        {
            self.times.pop_front();
        }
    }

    /// Events per second over the window ending at `now_ns` (silence
    /// included). Before a full window has passed since the first event,
    /// over the time since it; `None` before any event or within the first
    /// 100 ms, when no rate means anything yet.
    pub fn rate(&self, now_ns: i64) -> Option<f64> {
        let first = self.first_ns?;
        let span = (now_ns - first).min(self.window_ns);
        if span < 100_000_000 {
            return None;
        }
        let from = now_ns - span;
        let count = self.times.len() - self.times.partition_point(|&t| t <= from);
        Some(count as f64 / (span as f64 * 1e-9))
    }

    /// Events in the window ending at `now_ns`.
    pub fn in_window(&self, now_ns: i64) -> usize {
        let from = now_ns - self.window_ns;
        self.times.len() - self.times.partition_point(|&t| t <= from)
    }

    /// Every event recorded.
    pub fn total(&self) -> u64 {
        self.total
    }
}

/// The rate of a reported cumulative counter.
#[derive(Clone, Debug)]
pub struct CounterRate {
    window_ns: i64,
    /// `(reporter time, value)`.
    points: VecDeque<(i64, u64)>,
    /// Local time of the newest report.
    seen_ns: Option<i64>,
}

impl CounterRate {
    /// A rate over `window` of the reporter's clock.
    pub fn new(window: Duration) -> Self {
        Self {
            window_ns: ns(window).max(1),
            points: VecDeque::new(),
            seen_ns: None,
        }
    }

    /// The counter read `value` at `t_ns` on the reporter's clock, received
    /// at `local_ns` on ours. A counter that went backwards (the reporter
    /// restarted) starts over.
    pub fn record(&mut self, t_ns: i64, value: u64, local_ns: i64) {
        if self
            .points
            .back()
            .is_some_and(|&(t, v)| value < v || t_ns < t)
        {
            self.points.clear();
        }
        self.points.push_back((t_ns, value));
        while self.points.len() > 2
            && self
                .points
                .get(1)
                .is_some_and(|&(t, _)| t <= t_ns - self.window_ns)
        {
            self.points.pop_front();
        }
        self.seen_ns = Some(local_ns);
    }

    /// Per second of the reporter's clock over the window; `None` until two
    /// reports, and once none has arrived for two windows (`now_ns` is
    /// local).
    pub fn rate(&self, now_ns: i64) -> Option<f64> {
        if self
            .seen_ns
            .is_none_or(|seen| now_ns - seen > 2 * self.window_ns)
        {
            return None;
        }
        let (&(t0, v0), &(t1, v1)) = (self.points.front()?, self.points.back()?);
        (t1 > t0).then(|| (v1 - v0) as f64 / ((t1 - t0) as f64 * 1e-9))
    }

    /// Growth over the window.
    pub fn delta(&self) -> u64 {
        match (self.points.front(), self.points.back()) {
            (Some(&(_, v0)), Some(&(_, v1))) => v1 - v0,
            _ => 0,
        }
    }

    /// The newest value.
    pub fn value(&self) -> Option<u64> {
        self.points.back().map(|&(_, v)| v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    #[test]
    fn a_steady_stream_reads_its_rate_and_silence_reads_as_slowing() {
        let mut meter = RateMeter::new(Duration::from_secs(1));
        assert_eq!(meter.rate(0), None);
        for i in 0..2000 {
            meter.record(i * MS);
        }
        let hz = meter.rate(2000 * MS).expect("rate");
        assert!((hz - 1000.0).abs() < 2.0, "{hz}");
        // Half a second of silence halves the rate; a full second zeroes it.
        assert!((meter.rate(2500 * MS).expect("rate") - 500.0).abs() < 2.0);
        assert_eq!(meter.rate(3000 * MS), Some(0.0));
        assert_eq!(meter.total(), 2000);
    }

    #[test]
    fn a_young_meter_divides_by_the_time_since_its_first_event() {
        let mut meter = RateMeter::new(Duration::from_secs(1));
        meter.record(0);
        assert_eq!(meter.rate(50 * MS), None, "too early to mean anything");
        for i in 1..=20 {
            meter.record(i * 10 * MS);
        }
        let hz = meter.rate(200 * MS).expect("rate");
        assert!((hz - 100.0).abs() < 6.0, "{hz}");
    }

    #[test]
    fn the_window_follows_the_spec_for_slow_streams() {
        assert_eq!(window_for(1000.0), Duration::from_secs(1));
        assert_eq!(window_for(2.0), Duration::from_secs(2));
        assert_eq!(window_for(0.0), Duration::from_secs(1));
    }

    #[test]
    fn a_counter_rate_is_on_the_reporters_clock_and_goes_unknown_when_reports_stop() {
        let mut rate = CounterRate::new(Duration::from_secs(2));
        // The reporter's clock is 7 s ahead of ours; only its deltas count.
        rate.record(7_000 * MS, 0, 0);
        assert_eq!(rate.rate(0), None, "one report");
        for i in 1..=8 {
            rate.record(7_000 * MS + i * 250 * MS, (i * 25) as u64, i * 250 * MS);
        }
        let hz = rate.rate(2_000 * MS).expect("rate");
        assert!((hz - 100.0).abs() < 1e-9, "{hz}");
        assert_eq!(rate.value(), Some(200));
        assert_eq!(rate.rate(7_000 * MS), None, "stale");
        // A restart starts over.
        rate.record(9_500 * MS, 3, 2_250 * MS);
        assert_eq!((rate.value(), rate.delta()), (Some(3), 0));
    }
}
