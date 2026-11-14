//! Decoding errors.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::wire::ids::IdError;

/// Why a message could not be decoded. Receivers drop such a message (and
/// count it); none of these is fatal to a session.
#[derive(Debug, thiserror::Error)]
pub enum WireError {
    /// Not a decodable protobuf message.
    #[error("protobuf: {0}")]
    Protobuf(#[from] prost::DecodeError),
    /// An id in the message is malformed.
    #[error(transparent)]
    Id(#[from] IdError),
    /// A field is missing, unknown where safety depends on it, or out of
    /// range.
    #[error("malformed {message}: {reason}")]
    Malformed {
        /// The message.
        message: &'static str,
        /// What is wrong with it.
        reason: String,
    },
    /// Refused unread by a bounded decoder, which says why without
    /// allocating: a board counts it and keeps what it held.
    #[error("{message} unread: {why}")]
    Unread {
        /// The message.
        message: &'static str,
        /// Why.
        why: Unread,
    },
}

/// Why a bounded decoder refused a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Unread {
    /// Not well formed: what is wrong.
    #[error("malformed {0}")]
    Malformed(&'static str),
    /// More of something than the decoder takes: what.
    #[error("too many or too long: {0}")]
    TooLarge(&'static str),
    /// More than the heap can give it now.
    #[error("no memory for it")]
    NoMemory,
}

impl WireError {
    pub(crate) const fn unread(message: &'static str, why: Unread) -> Self {
        Self::Unread { message, why }
    }

    /// Why a bounded decoder refused it, if one did.
    pub fn why_unread(&self) -> Option<Unread> {
        match self {
            Self::Unread { why, .. } => Some(*why),
            _ => None,
        }
    }

    pub(crate) fn malformed(message: &'static str, reason: impl Into<String>) -> Self {
        Self::Malformed {
            message,
            reason: reason.into(),
        }
    }
}
