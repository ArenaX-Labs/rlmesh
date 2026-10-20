//! Observer telemetry (§8.10): what a participant publishes so an observer
//! (the cockpit) can show the counters only the measuring side knows.
//!
//! Every [`REPORT_PERIOD`] a participant that publishes streams or runs a
//! controller sends one [`ParticipantReport`] on `p/{participant}/report`,
//! data class, best effort: its publishers' published and coalesced counts,
//! and, once a [`Controller`](crate::Controller) is bound, its run counters
//! ([`run_counters`]). A participant with neither (a pure observer, the
//! host) sends nothing. The counters are cumulative, so a dropped report
//! loses nothing, and nothing in the session acts on them.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rlmesh_wire::codec;
use rlmesh_wire::{Epoch, ParticipantId, ParticipantReport, PublisherCounters, RunCounters};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::clock::Clock;
use crate::schedule::{RunStats, SkipReason};
use crate::transport::{ClassQos, TrafficClass, Transport, declare_publisher};

/// How often a participant reports.
pub const REPORT_PERIOD: Duration = Duration::from_millis(250);

/// A controller runtime's counters at a time on the participant's clock.
pub type RunSource = Arc<dyn Fn(i64) -> RunCounters + Send + Sync>;

/// A controller's [`RunStats`] and the age of its inference in flight, as
/// reported.
pub fn run_counters(stats: &RunStats, pending: Option<Duration>) -> RunCounters {
    let ns = |d: Option<Duration>| d.map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
    RunCounters {
        starts: stats.starts,
        completions: stats.completions,
        busy_skips: stats.skipped(SkipReason::Busy),
        waiting_for: stats.skipped(SkipReason::Waiting),
        stale_input: stats.skipped(SkipReason::StaleInput),
        window_gap: stats.skipped(SkipReason::WindowGap),
        decode: stats.skipped(SkipReason::Decode),
        no_lease: stats.skipped(SkipReason::NoLease),
        chunks_sent: stats.chunks_sent,
        pending_ns: ns(pending),
        latency_p50_ns: ns(stats.latency_p50),
        latency_p99_ns: ns(stats.latency_p99),
        drifting: stats.drifting,
    }
}

/// What the report loop reads from its participant.
pub(crate) struct ReportInputs {
    pub(crate) transport: Transport,
    pub(crate) key: String,
    pub(crate) id: ParticipantId,
    pub(crate) clock: Arc<dyn Clock>,
    pub(crate) epoch: Arc<AtomicU64>,
    pub(crate) run: Arc<Mutex<Option<RunSource>>>,
}

/// Sends a report every [`REPORT_PERIOD`] while there is something to
/// report. `streams` lists this participant's publishers' counters.
pub(crate) async fn report_loop(
    inputs: ReportInputs,
    streams: impl Fn() -> Vec<PublisherCounters> + Send,
    cancel: CancellationToken,
) {
    let qos = ClassQos {
        reliability: zenoh::qos::Reliability::BestEffort,
        congestion: zenoh::qos::CongestionControl::Drop,
        ..TrafficClass::Data.qos()
    };
    let publisher =
        match declare_publisher(&inputs.transport, TrafficClass::Data, inputs.key, qos).await {
            Ok(publisher) => publisher,
            Err(err) => {
                debug!(participant = %inputs.id, %err, "no report publisher");
                return;
            }
        };
    let mut interval = tokio::time::interval(REPORT_PERIOD);
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            () = cancel.cancelled() => return,
            _ = interval.tick() => {}
        }
        let now_ns = inputs.clock.now_ns();
        let source = inputs.run.lock().ok().and_then(|run| run.clone());
        let report = ParticipantReport {
            participant: inputs.id.clone(),
            epoch: Epoch(inputs.epoch.load(Ordering::Relaxed)),
            sent_ns: now_ns,
            streams: streams(),
            run: source.map(|source| source(now_ns)),
        };
        if report.streams.is_empty() && report.run.is_none() {
            continue;
        }
        if let Err(err) = publisher.put(codec::encode_report(&report)).await {
            debug!(participant = %inputs.id, %err, "report not sent");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_stats_map_to_counters_by_reason() {
        let mut stats = RunStats {
            starts: 10,
            completions: 9,
            chunks_sent: 9,
            latency_p50: Some(Duration::from_millis(8)),
            drifting: true,
            ..RunStats::default()
        };
        stats.skipped = [3, 2, 1, 4, 0, 6];
        let c = run_counters(&stats, Some(Duration::from_secs(2)));
        assert_eq!((c.starts, c.completions, c.chunks_sent), (10, 9, 9));
        assert_eq!(
            (
                c.busy_skips,
                c.waiting_for,
                c.stale_input,
                c.window_gap,
                c.decode,
                c.no_lease
            ),
            (3, 2, 1, 4, 0, 6)
        );
        assert_eq!(
            (c.pending_ns, c.latency_p50_ns, c.latency_p99_ns),
            (2_000_000_000, 8_000_000, 0)
        );
        assert!(c.drifting);
        assert_eq!(run_counters(&RunStats::default(), None).pending_ns, 0);
    }
}
