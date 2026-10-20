//! Errors.

use rlmesh_session::SessionError;
use rlmesh_wire::{ChannelId, ParticipantId};

/// Why an edge could not be set up.
#[derive(Debug, thiserror::Error)]
pub enum EdgeError {
    /// The channel is not an actuation channel.
    #[error("{0} is not an actuation channel")]
    NotActuation(ChannelId),
    /// The channel belongs to another embodiment.
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
