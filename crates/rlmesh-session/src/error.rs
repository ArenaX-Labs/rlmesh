//! Errors.

use std::time::Duration;

use crate::wire::{ChannelId, CtlError, IdError, WireError};

/// Everything a session call can fail with.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// The transport failed (opening a session, declaring, sending).
    #[error("zenoh: {0}")]
    Zenoh(#[from] zenoh::Error),
    /// A received message did not decode.
    #[error("wire: {0}")]
    Wire(#[from] WireError),
    /// An id is malformed.
    #[error(transparent)]
    Id(#[from] IdError),
    /// The host refused a verb.
    #[error("refused by the host: {0}")]
    Rejected(#[from] CtlError),
    /// No sealed answer came, only an unsealed refusal: a hint, since
    /// anyone can answer `ctl` (a key the host no longer knows gets one
    /// too), so the verb's outcome is unknown.
    #[error("no authenticated reply from the session host; an unauthenticated one said: {0}")]
    Unverified(CtlError),
    /// The host did not answer in time.
    #[error("no reply from the session host within {0:?}")]
    NoReply(Duration),
    /// Something expected did not happen in time.
    #[error("timed out after {0:?} waiting for {1}")]
    Timeout(Duration, String),
    /// The part is not in this participant's view of the table.
    #[error("{0} is not declared")]
    UnknownPart(String),
    /// The stream exists but cannot be used this way.
    #[error("{channel}: {reason}")]
    WrongChannel {
        /// The channel.
        channel: ChannelId,
        /// Why it cannot be used.
        reason: String,
    },
    /// A value does not fit its stream.
    #[error("{channel} value: {reason}")]
    InvalidValue {
        /// The channel.
        channel: ChannelId,
        /// What is wrong.
        reason: String,
    },
    /// The host reached serves other sessions, and never this one.
    #[error("{0}")]
    NotServed(String),
    /// A configuration value is unusable.
    #[error("config: {0}")]
    Config(String),
    /// A media frame could not be encoded or decoded.
    #[error("media: {0}")]
    Media(String),
    /// An I/O error (the host binary's files, local sockets).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The session (or one of its tasks) has stopped.
    #[error("the session has stopped")]
    Closed,
}

impl SessionError {
    /// The host's refusal, or an unsealed one ([`Self::Unverified`]), when
    /// this is one.
    pub fn refusal(&self) -> Option<&CtlError> {
        match self {
            Self::Rejected(err) | Self::Unverified(err) => Some(err),
            _ => None,
        }
    }
}
