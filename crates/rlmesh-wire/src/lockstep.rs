//! The lockstep fast path (§12): one request/reply per step between the
//! driver (the controller side) and the env (the embodiment side), on
//! `l/{robot}`, with no host round trip and no pacing.
//!
//! ```text
//! driver ──LockstepStep{epoch, step, lanes, commands}──▶ env
//! driver ◀──LockstepReply{step, observations, reward, terminated, truncated, episode_ids}── env
//! ```
//!
//! Payloads use §10.6 packing with a leading lane axis: a [`PartValues`] is
//! an N-lane slab, `lanes × stride`. The Gym view that drives these is wave
//! 5; the messages live here so it builds on a fixed wire.

use bytes::Bytes;

use crate::ids::{ChannelId, Epoch, LeaseToken};

/// An N-lane slab of one part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartValues {
    /// The part (its stream id).
    pub channel: ChannelId,
    /// `lanes × stride` bytes.
    pub payload: Bytes,
}

/// Resets lanes (all when `lanes` is empty).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LockstepReset {
    /// The epoch.
    pub epoch: Epoch,
    /// The lanes; empty is every lane.
    pub lanes: Vec<u32>,
    /// Seeds, aligned with `lanes` when present.
    pub seeds: Vec<u64>,
    /// Gymnasium `options`, as JSON.
    pub options_json: String,
    /// Episode ids, minted by the driver (as v1).
    pub episode_ids: Vec<u64>,
}

/// One step: exactly one wrapped Gym step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LockstepStep {
    /// The epoch.
    pub epoch: Epoch,
    /// The step number.
    pub step: u64,
    /// The driver's lease token on the env's commandable parts.
    pub lease_token: LeaseToken,
    /// The lanes; empty is every lane.
    pub lanes: Vec<u32>,
    /// One slab per commanded part.
    pub commands: Vec<PartValues>,
}

/// The env's answer to one step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LockstepReply {
    /// The epoch.
    pub epoch: Epoch,
    /// The step it answers.
    pub step: u64,
    /// One slab per published part that changed this step.
    pub observations: Vec<PartValues>,
    /// `f32 [N]`.
    pub reward: Bytes,
    /// `bool [N]`.
    pub terminated: Bytes,
    /// `bool [N]`.
    pub truncated: Bytes,
    /// Per lane, after autoreset.
    pub episode_ids: Vec<u64>,
    /// Gymnasium `info`, as JSON (only when `record_info` is on).
    pub info_json: String,
}
