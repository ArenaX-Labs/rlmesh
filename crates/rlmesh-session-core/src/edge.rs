//! The sans-IO edge (the v0.2 spec §9): what runs next to an embodiment and
//! turns controllers' timed action chunks into one command per control
//! tick, whatever the network does. No transport, clock reads or threads:
//! `rlmesh-session` attaches it to a participant, and a device without
//! `std` can drive it directly.
//!
//! - [`queue`]: the timed action queue, splicing new chunks at their
//!   `replace_from` (validated before anything is removed), with
//!   [`Delivery::Once`] for delta commands (§9.3).
//! - [`command`]: command validation and limits (§9.6): finite values, valid
//!   rotations, clamping, norm bounds and the binary snap.
//! - [`fallback`]: the per-tick mode ladder of one command group (Idle,
//!   Driving, Hold, Safe, Estop) on edge-local deadlines (§9.5), with
//!   dispatch gated to the leased member and the handover rule (§9.7).
//! - [`core`]: [`EdgeCore`], the edge of one group: chunk fencing (epoch,
//!   lease token, source, binding, seq, the §9.1 clock rules and the transit
//!   check), validation, the queues, the ladder, the applied stream and the
//!   applied-action log with source attribution.

pub mod command;
pub mod core;
pub mod fallback;
pub mod queue;

pub use self::core::{
    AppliedAction, AppliedSample, EdgeConfig, EdgeCore, EdgeStats, EdgeTick, GroupHandover,
    Rejection,
};
pub use crate::wire::EdgeMode;
pub use command::ActionError;
pub use fallback::{ModeStats, SafeCause};
pub use queue::{Delivery, QueueStats, QueuedAction};

use crate::wire::{ChannelId, ParticipantId};

/// Why an edge's command group could not be built.
#[derive(Debug, thiserror::Error)]
pub enum GroupError {
    /// No part was given.
    #[error("an edge needs at least one commandable part")]
    NoParts,
    /// The part is not commandable.
    #[error("{0} is not a commandable part")]
    NotCommandable(ChannelId),
    /// The parts are not one command group (sharing `safe` and `hold_max`).
    #[error("{0} is not in the edge's command group")]
    NotOneGroup(ChannelId),
    /// The part belongs to another embodiment.
    #[error("{channel} is not owned by {me}")]
    NotOwner {
        /// The channel.
        channel: ChannelId,
        /// This participant.
        me: ParticipantId,
    },
}
