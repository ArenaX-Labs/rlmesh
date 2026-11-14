//! What the host tells observers about every participant's link and edge
//! (`host/health`, signed): the heartbeats and reports participants seal to
//! the host, so the cockpit and recordings can show why a robot is or isn't
//! moving.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::edge::Rejection;
use crate::wire::heartbeat::{ChunkTiming, ClockRefusal, DatagramStats, Health};
use crate::wire::ids::{ChannelId, Epoch, ParticipantId};
use crate::wire::report::ParticipantReport;

/// One participant's link and edge as the host sees them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LinkHealth {
    /// Whose.
    pub participant: Option<ParticipantId>,
    /// Heartbeats received.
    pub heartbeats: u64,
    /// The latest round trip (0: none yet).
    pub rtt_ns: u64,
    /// Smoothed round-trip jitter.
    pub jitter_ns: u64,
    /// The largest gap between heartbeat arrivals in the window.
    pub heartbeat_gap_max_ns: u64,
    /// Since the last heartbeat.
    pub last_heartbeat_age_ns: u64,
    /// The clock mapping, when there is one: `(offset_ns, error_ns)`.
    pub offset: Option<(i64, u64)>,
    /// Edges: the newest accepted chunk seq.
    pub last_chunk_accepted: u64,
    /// Edges: the newest applied chunk seq.
    pub last_chunk_applied: u64,
    /// Edges: queued actions.
    pub queue_depth: u32,
    /// Its self-reported health.
    pub health: Health,
    /// Edges: chunks refused since attach, by reason.
    pub rejected: Vec<(Rejection, u64)>,
    /// Edges: one per edge whose newest clock judgement refused a chunk.
    pub clock_refused: Vec<ClockRefusal>,
    /// Its latest observer report.
    pub report: Option<ParticipantReport>,
    /// The newest seq it has received per subscribed channel.
    pub last_seq_seen: Vec<(ChannelId, u64)>,
    /// Edges: how each commanded part's chunks arrived.
    pub chunk_timing: Vec<ChunkTiming>,
    /// Edges: chunks accepted since attach.
    pub chunks_accepted: u64,
    /// Robots: the path their realtime parts ride (unknown for others).
    pub data_path: DataPath,
    /// Robots on the datagram path: the board's counters, from its
    /// heartbeat.
    pub datagram_board: Option<DatagramStats>,
    /// Robots on the datagram path: the host's counters for it.
    pub datagram_host: Option<DatagramStats>,
}

/// Which path a robot's realtime parts ride.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DataPath {
    /// Not a robot, or not reported.
    #[default]
    Unknown,
    /// The session's own links (TCP).
    Session,
    /// The host's datagram path (UDP).
    Datagram,
}

impl DataPath {
    /// `"tcp"`, `"udp"`, or `"unknown"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Session => "tcp",
            Self::Datagram => "udp",
        }
    }
}

/// The host's health message.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostHealth {
    /// The epoch.
    pub epoch: Epoch,
    /// Host clock at publish.
    pub host_ns: i64,
    /// One per attached participant.
    pub links: Vec<LinkHealth>,
}

impl HostHealth {
    /// The link of `participant`, when it is attached.
    pub fn link(&self, participant: &ParticipantId) -> Option<&LinkHealth> {
        self.links
            .iter()
            .find(|l| l.participant.as_ref() == Some(participant))
    }
}
