//! What the host tells observers about every participant's link and edge
//! (`host/health`, signed): the heartbeats and reports participants seal to
//! the host, so the cockpit and recordings can show why a robot is or isn't
//! moving.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::edge::Rejection;
use crate::wire::heartbeat::Health;
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
    /// Its latest observer report.
    pub report: Option<ParticipantReport>,
    /// The newest seq it has received per subscribed channel.
    pub last_seq_seen: Vec<(ChannelId, u64)>,
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
