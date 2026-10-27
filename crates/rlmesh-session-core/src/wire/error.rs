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
}

impl WireError {
    pub(crate) fn malformed(message: &'static str, reason: impl Into<String>) -> Self {
        Self::Malformed {
            message,
            reason: reason.into(),
        }
    }
}
