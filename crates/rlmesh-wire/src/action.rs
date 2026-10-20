//! Timed action chunks: a controller's commands for one commandable part
//! (§9.2, §10.6).
//!
//! All times are **session time** (the host's clock domain, or ticks ×
//! `tick_ns` in virtual and lockstep modes); the edge converts them into its
//! own clock on receipt, conservatively, and never acts on a chunk past its
//! own age cap, whatever `valid_until_ns` says.

use bytes::Bytes;

use crate::ids::{ChannelId, Epoch, Fingerprint, LeaseToken, ParticipantId};

/// The samples one input of a chunk was computed from: a seq range of one
/// stream (a window, or one sample when `seq_first == seq_last`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InputRange {
    /// The stream.
    pub channel: ChannelId,
    /// The oldest sample used.
    pub seq_first: u64,
    /// The newest sample used.
    pub seq_last: u64,
}

impl InputRange {
    /// One sample.
    pub fn one(channel: ChannelId, seq: u64) -> Self {
        Self {
            channel,
            seq_first: seq,
            seq_last: seq,
        }
    }
}

/// A chunk, by identity `(channel, lease_token, seq)` (§10.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkRef {
    /// The commandable part.
    pub channel: ChannelId,
    /// Its lease token.
    pub lease_token: LeaseToken,
    /// Its seq under that token.
    pub seq: u64,
}

/// A run of `count` actions for one commandable part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionChunk {
    /// The commandable part.
    pub channel: ChannelId,
    /// The epoch it was issued in.
    pub epoch: Epoch,
    /// The lease token: fences preempted or rebound controllers.
    pub lease_token: LeaseToken,
    /// Per lease token, increasing.
    pub seq: u64,
    /// The controller that computed it. The edge refuses a chunk whose
    /// source does not hold the lease its token names, so every applied
    /// action's attribution is checked, not just claimed.
    pub source: ParticipantId,
    /// The binding of the lease; the edge refuses any other.
    pub binding: Fingerprint,
    /// The inputs it was computed from.
    pub based_on: Vec<InputRange>,
    /// Session time of the first action's slot.
    pub t0_ns: i64,
    /// The action period; zero takes the part's control period.
    pub dt_ns: u64,
    /// Keep committed actions before this; replace from here.
    pub replace_from_ns: i64,
    /// The whole chunk is void at and after this.
    pub valid_until_ns: i64,
    /// The number of actions.
    pub count: u32,
    /// `count` actions at the command stride (§10.6). Decoded chunks slice
    /// the received buffer.
    pub actions: Bytes,
    /// Sender clock at publish.
    pub publish_ns: i64,
    /// The virtual or lockstep tick; 0 in wall mode.
    pub tick: u64,
}

impl ActionChunk {
    /// Its identity.
    pub fn reference(&self) -> ChunkRef {
        ChunkRef {
            channel: self.channel,
            lease_token: self.lease_token,
            seq: self.seq,
        }
    }

    /// Action `i` at `stride` (a slice of the payload, no copy).
    pub fn action(&self, i: usize, stride: usize) -> Option<Bytes> {
        let start = i.checked_mul(stride)?;
        let end = start.checked_add(stride)?;
        (end <= self.actions.len()).then(|| self.actions.slice(start..end))
    }
}
