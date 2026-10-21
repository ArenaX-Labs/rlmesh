//! Errors.

use crate::SessionError;
use crate::wire::{ChannelId, ParticipantId};

/// Why an edge could not be set up.
#[derive(Debug, thiserror::Error)]
pub enum EdgeError {
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
    /// The session layer failed (an undeclared channel, the transport).
    #[error(transparent)]
    Session(#[from] SessionError),
    /// The transport failed to declare the intake.
    #[error("zenoh: {0}")]
    Zenoh(#[from] zenoh::Error),
}
