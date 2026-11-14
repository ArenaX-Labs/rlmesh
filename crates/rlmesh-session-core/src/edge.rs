//! The sans-IO edge: what runs next to an embodiment and
//! turns controllers' timed action chunks into one command per control
//! tick, whatever the network does. No transport, clock reads or threads:
//! `rlmesh-session` attaches it to a participant, and a device without
//! `std` can drive it directly.
//!
//! - [`queue`]: the timed action queue, splicing new chunks at their
//!   `replace_from` (validated before anything is removed), with
//!   [`Delivery::Once`] for delta commands.
//! - [`command`]: command validation and limits: finite values, valid
//!   rotations, clamping, norm bounds and the binary snap.
//! - [`fallback`]: the per-tick mode ladder of one command group (Idle,
//!   Driving, Hold, Safe, Estop) on edge-local deadlines, with
//!   dispatch gated to the leased member and the handover rule.
//! - [`core`]: [`EdgeCore`], the edge of one group: chunk fencing (epoch,
//!   lease token, source, binding, seq, the clock rules and the transit
//!   check), validation, the queues, the ladder, the applied stream and the
//!   applied-action log with source attribution.

pub mod command;
pub mod core;
pub mod fallback;
pub mod queue;

pub use self::core::{
    AppliedSample, EdgeConfig, EdgeCore, EdgeStats, EdgeTick, GroupHandover, Rejection,
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
