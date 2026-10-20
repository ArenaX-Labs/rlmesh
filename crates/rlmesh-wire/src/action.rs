//! Timed action chunks: a controller's commands for one actuation channel.
//!
//! All times are **session time** (the host's clock domain); the edge
//! converts them into its own clock on receipt and never acts on a chunk
//! past its own staleness limit, whatever `valid_until_ns` says.

use bytes::Bytes;

use crate::ids::{ChannelId, Epoch, LeaseToken, ParticipantId};

/// One action: one buffer per leaf of the channel's layout.
pub type Action = Vec<Bytes>;

/// The observation a chunk was computed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObservationRef {
    /// The sense channel.
    pub channel: ChannelId,
    /// The sample's seq.
    pub seq: u64,
}

/// A run of actions for one actuation channel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionChunk {
    /// The actuation channel.
    pub channel: ChannelId,
    /// The controller that computed it. The edge refuses a chunk whose
    /// source is not the holder of the lease its token names, so every
    /// applied action's attribution is checked, not just claimed.
    pub source: ParticipantId,
    /// The lease token: fences preempted controllers.
    pub lease_token: LeaseToken,
    /// The epoch it was issued in.
    pub epoch: Epoch,
    /// Per lease token, increasing.
    pub seq: u64,
    /// Sender clock at publish.
    pub publish_ns: i64,
    /// The virtual or lockstep tick; 0 in wall mode.
    pub tick: u64,
    /// The observations it was computed from.
    pub based_on: Vec<ObservationRef>,
    /// Session time of `actions[0]`.
    pub t0_ns: i64,
    /// The action period; zero takes the channel's declared period.
    pub dt_ns: u64,
    /// Keep committed actions before this; replace from here.
    pub replace_from_ns: i64,
    /// The whole chunk is void at and after this.
    pub valid_until_ns: i64,
    /// The actions, each with one buffer per leaf of the channel's layout.
    pub actions: Vec<Action>,
}
