//! Heartbeats and clock pings.
//!
//! A participant heartbeats on the control class; the host echoes each one
//! at once with its receive and send times, and the participant's next
//! heartbeat carries the completed four-timestamp exchange
//! ([`TimingSample`]), so the host learns all four timestamps without a third
//! message.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::wire::authority::Mode;
use crate::wire::ids::{ChannelId, Epoch, ParticipantId};

/// One NTP-style exchange: `t1`/`t4` in the participant's clock, `t2`/`t3` in
/// the host's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimingSample {
    /// The heartbeat it belongs to.
    pub seq: u64,
    /// Participant clock at send.
    pub t1_ns: i64,
    /// Host clock at receipt.
    pub t2_ns: i64,
    /// Host clock at the echo's send.
    pub t3_ns: i64,
    /// Participant clock at the echo's receipt.
    pub t4_ns: i64,
}

impl TimingSample {
    /// θ = ((t2 − t1) + (t3 − t4)) / 2: host minus participant, floored.
    ///
    /// For any non-negative one-way delays d1 (out) and d2 (back) the true
    /// offset lies within δ/2 of the exact midpoint, since the midpoint is
    /// θ + (d1 − d2)/2 and δ = d1 + d2.
    pub fn offset_ns(&self) -> i64 {
        let sum = (i128::from(self.t2_ns) - i128::from(self.t1_ns))
            + (i128::from(self.t3_ns) - i128::from(self.t4_ns));
        i64::try_from(sum.div_euclid(2)).unwrap_or(if sum < 0 { i64::MIN } else { i64::MAX })
    }

    /// δ = (t4 − t1) − (t3 − t2): the round trip less the host's hold time.
    pub fn rtt_ns(&self) -> i64 {
        let rtt = (i128::from(self.t4_ns) - i128::from(self.t1_ns))
            - (i128::from(self.t3_ns) - i128::from(self.t2_ns));
        i64::try_from(rtt).unwrap_or(if rtt < 0 { i64::MIN } else { i64::MAX })
    }

    /// A sample is usable when both clocks moved forward across it.
    pub fn is_valid(&self) -> bool {
        self.t4_ns >= self.t1_ns && self.t3_ns >= self.t2_ns && self.rtt_ns() >= 0
    }
}

/// A participant's self-reported health.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Health {
    /// Unknown (an older peer, or not reported).
    #[default]
    Unknown,
    /// Fine.
    Ok,
    /// Working, but slow or lossy.
    Degraded,
    /// Should not be trusted with a lease.
    Unhealthy,
}

/// Participant to host, on the control class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Heartbeat {
    /// The sender (must match the key).
    pub participant: ParticipantId,
    /// The sender's view of the epoch.
    pub epoch: Epoch,
    /// The sender's view of the mode.
    pub mode: Mode,
    /// Per-sender sequence number.
    pub seq: u64,
    /// Participant clock at send.
    pub t1_ns: i64,
    /// The previous exchange, completed with its t4.
    pub previous: Option<TimingSample>,
    /// The newest sample seq received per subscribed channel.
    pub last_seq_seen: Vec<(ChannelId, u64)>,
    /// Edges: the seq of the newest accepted chunk (zero when none).
    pub last_chunk_accepted: u64,
    /// Edges: the seq of the newest applied chunk (zero when none).
    pub last_chunk_applied: u64,
    /// Edges: queued actions.
    pub queue_depth: u32,
    /// Self-reported health.
    pub health: Health,
    /// Edges: chunks refused since attach, by reason (nonzero only).
    pub rejected: Vec<(crate::edge::Rejection, u64)>,
    /// Edges: one per edge whose newest clock judgement refused a chunk.
    pub clock_refused: Vec<ClockRefusal>,
    /// Edges: how each commanded part's chunks arrived since attach.
    pub chunk_timing: Vec<ChunkTiming>,
    /// Edges: chunks accepted since attach.
    pub chunks_accepted: u64,
    /// A board on the datagram path: its end's counters.
    pub datagram: Option<DatagramStats>,
}

/// Lateness buckets of [`ChunkTiming::late_ms`]: `[0, 1)` ms, then
/// `[2^(i-1), 2^i)` ms for `i` in 1..7, and the last from 64 ms up.
pub const LATE_BUCKETS: usize = 8;

/// How one commanded part's chunks arrived at its edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkTiming {
    /// The commandable part.
    pub channel: ChannelId,
    /// Chunks that reached the edge, whatever became of them.
    pub received: u64,
    /// Chunks it accepted.
    pub accepted: u64,
    /// Gaps in the seqs of accepted chunks under one lease token.
    pub lost: u32,
    /// Accepted chunks by lateness on arrival against their first slot
    /// ([`LATE_BUCKETS`]).
    pub late_ms: [u32; LATE_BUCKETS],
    /// The latest any accepted chunk arrived, in microseconds.
    pub late_max_us: u32,
    /// The longest time between two accepted chunks under one lease token,
    /// in microseconds.
    pub gap_max_us: u32,
}

impl Default for ChunkTiming {
    fn default() -> Self {
        Self {
            channel: ChannelId(0),
            received: 0,
            accepted: 0,
            lost: 0,
            late_ms: [0; LATE_BUCKETS],
            late_max_us: 0,
            gap_max_us: 0,
        }
    }
}

impl ChunkTiming {
    /// The bucket of a chunk `late_ns` late.
    pub fn bucket(late_ns: i64) -> usize {
        let ms = late_ns.max(0).unsigned_abs() / 1_000_000;
        if ms == 0 {
            0
        } else {
            (u64::BITS - ms.leading_zeros()).min(LATE_BUCKETS as u32 - 1) as usize
        }
    }

    /// Counts an accepted chunk `late_ns` late.
    pub fn late(&mut self, late_ns: i64) {
        let slot = &mut self.late_ms[Self::bucket(late_ns)];
        *slot = slot.saturating_add(1);
        let us = u32::try_from(late_ns.max(0).unsigned_abs() / 1000).unwrap_or(u32::MAX);
        self.late_max_us = self.late_max_us.max(us);
    }

    /// Counts `gap_ns` between two accepted chunks.
    pub fn gap(&mut self, gap_ns: i64) {
        let us = u32::try_from(gap_ns.max(0).unsigned_abs() / 1000).unwrap_or(u32::MAX);
        self.gap_max_us = self.gap_max_us.max(us);
    }
}

/// One end of a robot's datagram path: what it sent, and what it received
/// by the other end's statistics sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DatagramStats {
    /// The path carries the robot's realtime parts now.
    pub active: bool,
    /// Datagrams sent.
    pub sent: u64,
    /// Datagrams received that opened.
    pub received: u64,
    /// Gaps in the statistics sequence of those received.
    pub lost: u64,
    /// Those that arrived after a later one.
    pub reordered: u64,
    /// Received but unreadable, or failed to open under the route's key.
    pub unopened: u64,
    /// Times this end gave the path up for the session's own.
    pub fallbacks: u32,
}

/// Why an edge is refusing chunks as
/// [`Rejection::ClockUncertain`](crate::edge::Rejection::ClockUncertain).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClockRefusal {
    /// The part whose chunk it refused.
    pub channel: ChannelId,
    /// Whose clock mapping is over the bound.
    pub participant: ParticipantId,
    /// The mapping's measured error.
    pub error_ns: u64,
    /// The bound the edge held it to.
    pub bound_ns: u64,
}

impl ClockRefusal {
    /// The refusal `clocks` (`(whose, error)`) earn a chunk for `channel`
    /// under `bound_ns`, naming the worst clock; `None` when every one is
    /// within it.
    pub fn judge(
        channel: ChannelId,
        bound_ns: u64,
        clocks: [(&ParticipantId, u64); 2],
    ) -> Option<Self> {
        let [first, second] = clocks;
        let (participant, error_ns) = if second.1 > first.1 { second } else { first };
        (error_ns > bound_ns).then(|| Self {
            channel,
            participant: participant.clone(),
            error_ns,
            bound_ns,
        })
    }

    /// The refusal in words, for `robot`'s part at `path`.
    pub fn explain(&self, robot: &ParticipantId, path: &str) -> String {
        let whose = if self.participant == *robot {
            String::new()
        } else {
            format!("{}'s ", self.participant)
        };
        format!(
            "{robot}:{path} refuses chunks: {whose}clock error {} ms is above its {} ms bound; \
             raise max_clock_error_ms or improve the link",
            millis_text(self.error_ns),
            millis_text(self.bound_ns),
        )
    }
}

/// Nanoseconds as milliseconds to one decimal, without a trailing `.0`.
fn millis_text(ns: u64) -> String {
    let tenths = ns.saturating_add(50_000) / 100_000;
    match tenths % 10 {
        0 => format!("{}", tenths / 10),
        decimal => format!("{}.{decimal}", tenths / 10),
    }
}

/// Host to participant, on the control class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeartbeatEcho {
    /// The heartbeat's seq.
    pub seq: u64,
    /// The heartbeat's t1.
    pub t1_ns: i64,
    /// Host clock at the heartbeat's receipt.
    pub t2_ns: i64,
    /// Host clock at the echo's send.
    pub t3_ns: i64,
    /// The host's epoch.
    pub epoch: Epoch,
}

impl HeartbeatEcho {
    /// Completes the exchange with the participant's receive time.
    pub fn complete(&self, t4_ns: i64) -> TimingSample {
        TimingSample {
            seq: self.seq,
            t1_ns: self.t1_ns,
            t2_ns: self.t2_ns,
            t3_ns: self.t3_ns,
            t4_ns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_delay_recovers_the_offset_exactly() {
        // Participant clock = host − 5 s; 300 µs each way, 20 µs hold.
        let s = TimingSample {
            seq: 1,
            t1_ns: 1_000_000 - 5_000_000_000,
            t2_ns: 1_300_000,
            t3_ns: 1_320_000,
            t4_ns: 1_620_000 - 5_000_000_000,
        };
        assert_eq!(s.offset_ns(), 5_000_000_000);
        assert_eq!(s.rtt_ns(), 600_000);
        assert!(s.is_valid());
    }

    #[test]
    fn a_refusal_names_the_worst_clock_and_reads_in_milliseconds() {
        let robot = ParticipantId::new("eye-1").expect("id");
        let policy = ParticipantId::new("policy").expect("id");
        let channel = ChannelId(4);
        assert_eq!(
            ClockRefusal::judge(channel, 5_000_000, [(&robot, 5_000_000), (&policy, 1)]),
            None
        );
        let refusal = ClockRefusal::judge(channel, 5_000_000, [(&robot, 19_500_000), (&policy, 1)])
            .expect("over");
        assert_eq!(
            refusal.explain(&robot, "eyes"),
            "eye-1:eyes refuses chunks: clock error 19.5 ms is above its 5 ms bound; \
             raise max_clock_error_ms or improve the link"
        );
        let refusal = ClockRefusal::judge(channel, 5_000_000, [(&robot, 1), (&policy, 7_000_000)])
            .expect("over");
        assert_eq!(refusal.participant, policy);
        assert!(
            refusal
                .explain(&robot, "eyes")
                .contains("policy's clock error 7 ms"),
        );
    }

    #[test]
    fn extreme_timestamps_saturate_instead_of_overflowing() {
        let s = TimingSample {
            seq: 1,
            t1_ns: i64::MIN,
            t2_ns: i64::MAX,
            t3_ns: i64::MAX,
            t4_ns: i64::MIN,
        };
        assert_eq!(s.offset_ns(), i64::MAX);
        let echo = HeartbeatEcho {
            seq: 4,
            t1_ns: 1,
            t2_ns: 2,
            t3_ns: 3,
            epoch: Epoch(1),
        };
        assert_eq!(echo.complete(5).t4_ns, 5);
    }
}
