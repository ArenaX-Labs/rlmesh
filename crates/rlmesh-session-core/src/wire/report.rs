//! Observer telemetry (§8.10): the counters only the measuring side knows.
//!
//! An observer sees what travels: samples, chunks, `_applied` ticks,
//! heartbeats and authority. Two §8.10 counters never travel on their own:
//! how many samples a publisher produced before its plan coalesced some, and
//! a controller runtime's starts, completions and skipped slots. A
//! participant publishes them as a [`ParticipantReport`] on
//! `p/{participant}/report` ([`crate::wire::SessionKeys::report`]) a few times per
//! second. The counters are cumulative since the participant attached, so a
//! lost report loses nothing; observers take rates from their deltas on the
//! sender's clock. Nothing in the session acts on a report.

use crate::wire::ids::{ChannelId, Epoch, ParticipantId};

/// What one participant reports to observers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParticipantReport {
    /// The sender (must match the key).
    pub participant: ParticipantId,
    /// Its view of the epoch.
    pub epoch: Epoch,
    /// Sender clock at send.
    pub sent_ns: i64,
    /// One per stream it publishes.
    pub streams: Vec<PublisherCounters>,
    /// A controller runtime's counters.
    pub run: Option<RunCounters>,
}

/// The publisher side of one stream: produced = published + coalesced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublisherCounters {
    /// The stream.
    pub channel: ChannelId,
    /// Samples sent (the stream's last seq).
    pub published: u64,
    /// Samples the publication plan or the `send` ceiling coalesced.
    pub coalesced: u64,
}

impl PublisherCounters {
    /// Samples the source produced.
    pub fn produced(&self) -> u64 {
        self.published.saturating_add(self.coalesced)
    }
}

/// A controller runtime's start and completion counters (§8.9, §8.10).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunCounters {
    /// `predict` calls started.
    pub starts: u64,
    /// `predict` calls completed.
    pub completions: u64,
    /// Slots that fell while `predict` was busy (`busy_skip`).
    pub busy_skips: u64,
    /// Slots skipped because a required input was never received
    /// (`waiting_for`).
    pub waiting_for: u64,
    /// Slots skipped on a stale input (`stale_input`).
    pub stale_input: u64,
    /// Slots skipped on a history gap (`window_gap`).
    pub window_gap: u64,
    /// Slots skipped because a sample did not decode or fit its plan.
    pub decode: u64,
    /// Slots skipped without the leases (a standby or preempted controller).
    pub no_lease: u64,
    /// Chunks sent.
    pub chunks_sent: u64,
    /// Age of the inference in flight, in ns; 0 when none is.
    pub pending_ns: u64,
    /// Completion latency, median, in ns (0: none yet).
    pub latency_p50_ns: u64,
    /// Completion latency, 99th percentile, in ns (0: none yet).
    pub latency_p99_ns: u64,
    /// The completion rate is drifting (`rate_drift`).
    pub drifting: bool,
}
