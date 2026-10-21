//! Link and lag metrics per participant (design §3.3), as the host sees them,
//! plus what each participant reports in its heartbeats.
//!
//! Edge-side lag metrics (`action_age_at_apply`, deadline misses, fallback
//! time) live in `rlmesh-edge`; the heartbeat carries its queue report.

use std::collections::VecDeque;
use std::fmt;

use crate::wire::{ChannelId, Health, ParticipantId};

use crate::clock::OffsetEstimate;

/// Heartbeat arrival gaps and round-trip jitter over a sliding window.
#[derive(Clone, Debug)]
pub struct LinkStats {
    window: usize,
    gaps_ns: VecDeque<u64>,
    last_arrival_ns: Option<i64>,
    last_rtt_ns: Option<u64>,
    /// RFC 3550 style smoothed |Δrtt|.
    jitter_ns: f64,
    heartbeats: u64,
    report: Report,
}

/// What the latest heartbeat reported.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// The newest sample seq received per subscribed channel.
    pub last_seq_seen: Vec<(ChannelId, u64)>,
    /// Edges: the newest accepted chunk's seq.
    pub last_chunk_accepted: u64,
    /// Edges: the newest applied chunk's seq.
    pub last_chunk_applied: u64,
    /// Edges: queued actions.
    pub queue_depth: u32,
    /// Self-reported health.
    pub health: Health,
}

impl LinkStats {
    /// Stats over the last `window` heartbeat gaps.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            gaps_ns: VecDeque::new(),
            last_arrival_ns: None,
            last_rtt_ns: None,
            jitter_ns: 0.0,
            heartbeats: 0,
            report: Report::default(),
        }
    }

    /// A heartbeat arrived at host time `arrival_ns`.
    pub fn on_heartbeat(&mut self, arrival_ns: i64) {
        if let Some(last) = self.last_arrival_ns {
            if self.gaps_ns.len() == self.window {
                self.gaps_ns.pop_front();
            }
            self.gaps_ns
                .push_back((arrival_ns - last).max(0).unsigned_abs());
        }
        self.last_arrival_ns = Some(arrival_ns);
        self.heartbeats += 1;
    }

    /// Keeps the latest heartbeat's channel and edge report.
    pub fn on_report(&mut self, report: Report) {
        self.report = report;
    }

    /// A completed exchange measured round trip `rtt_ns`.
    pub fn on_round_trip(&mut self, rtt_ns: u64) {
        if let Some(last) = self.last_rtt_ns {
            let d = rtt_ns.abs_diff(last) as f64;
            self.jitter_ns += (d - self.jitter_ns) / 16.0;
        }
        self.last_rtt_ns = Some(rtt_ns);
    }

    /// The health row at host time `now_ns`.
    pub fn health(
        &self,
        participant: &ParticipantId,
        estimate: Option<OffsetEstimate>,
        now_ns: i64,
    ) -> ParticipantHealth {
        ParticipantHealth {
            participant: participant.clone(),
            heartbeats: self.heartbeats,
            rtt_ns: self.last_rtt_ns,
            jitter_ns: self.jitter_ns.round() as u64,
            heartbeat_gap_max_ns: self.gaps_ns.iter().copied().max().unwrap_or(0),
            last_heartbeat_age_ns: self
                .last_arrival_ns
                .map(|at| (now_ns - at).max(0).unsigned_abs()),
            offset: estimate,
            report: self.report.clone(),
        }
    }
}

/// One participant's link health, as the host measured it.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticipantHealth {
    /// Whose link.
    pub participant: ParticipantId,
    /// Heartbeats received.
    pub heartbeats: u64,
    /// The latest round trip.
    pub rtt_ns: Option<u64>,
    /// Smoothed round-trip jitter.
    pub jitter_ns: u64,
    /// Largest gap between heartbeat arrivals in the window.
    pub heartbeat_gap_max_ns: u64,
    /// Time since the last heartbeat.
    pub last_heartbeat_age_ns: Option<u64>,
    /// The clock mapping; its `rtt_ns` is the window minimum.
    pub offset: Option<OffsetEstimate>,
    /// The latest heartbeat's report.
    pub report: Report,
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

impl fmt::Display for ParticipantHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} hb={}", self.participant, self.heartbeats)?;
        if let Some(rtt) = self.rtt_ns {
            write!(f, " rtt={:.3}ms", ms(rtt))?;
        }
        write!(
            f,
            " jitter={:.3}ms gap_max={:.1}ms",
            ms(self.jitter_ns),
            ms(self.heartbeat_gap_max_ns)
        )?;
        if let Some(age) = self.last_heartbeat_age_ns {
            write!(f, " last_hb={:.1}ms ago", ms(age))?;
        }
        if let Some(o) = self.offset {
            write!(
                f,
                " offset={:+.3}ms ±{:.3}ms (rtt_min={:.3}ms)",
                o.offset_ns as f64 / 1e6,
                ms(o.error_ns),
                ms(o.rtt_ns)
            )?;
        }
        for (channel, seq) in &self.report.last_seq_seen {
            write!(f, " {channel}#{seq}")?;
        }
        if self.report.last_chunk_accepted > 0 {
            write!(
                f,
                " chunks accepted#{} applied#{} queue={}",
                self.report.last_chunk_accepted,
                self.report.last_chunk_applied,
                self.report.queue_depth
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaps_and_jitter_track_the_window() {
        let mut stats = LinkStats::new(3);
        for at in [0, 10, 20, 75, 85, 95, 105] {
            stats.on_heartbeat(at);
        }
        let id = ParticipantId::new("p").expect("id");
        // The 55 ns gap has slid out of a three-gap window.
        assert_eq!(stats.health(&id, None, 105).heartbeat_gap_max_ns, 10);
        stats.on_round_trip(100);
        stats.on_round_trip(260);
        let health = stats.health(&id, None, 205);
        assert_eq!(health.jitter_ns, 10);
        assert_eq!(health.last_heartbeat_age_ns, Some(100));
        assert_eq!(health.rtt_ns, Some(260));
    }
}
