//! Heartbeats and clock pings.
//!
//! A participant heartbeats on the control class; the host echoes each one
//! at once with its receive and send times, and the participant's next
//! heartbeat carries the completed four-timestamp exchange
//! ([`TimingSample`]), so the host learns all four timestamps without a third
//! message.

use crate::authority::Mode;
use crate::ids::{ChannelId, Epoch, ParticipantId};

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
