//! The messages (spec §4): the errors and receivers every tether frame
//! shares, and, with the `board` feature (unreleased), typed views over a
//! board link's payloads and their encoding. Lengths are exact; reserved
//! bits must be zero.
//!
//! `STATE` and `COMMAND` carry a part's dense value, whose length only the
//! receiver knows (from the map), so `decode` takes it as a lookup.

#[cfg(feature = "board")]
mod board;
#[cfg(feature = "board")]
pub use board::*;

/// Why a frame's payload isn't a valid message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MsgError {
    /// An unknown `type`.
    UnknownType,
    /// A `HELLO` without the magic.
    Magic,
    /// A `HELLO` of another protocol version (checked before its length).
    Version(u8),
    /// Not the message's exact length.
    Length,
    /// A reserved bit is set.
    Reserved,
    /// A field out of range (`max` 0, a name or note too long, a slice too
    /// long).
    Range,
    /// `STATE` or `COMMAND` for a part the receiver doesn't know.
    UnknownPart,
    /// A message type that travels the other way (§3.2): a bridge hearing a
    /// bridge message (an echoing adapter), or a board hearing a board one.
    Direction,
    /// `HELLO.name` isn't UTF-8.
    Utf8,
}

/// Which end is receiving: it decides which message types are valid, and
/// what `dense_len` looks up (sample lengths on the bridge, command lengths
/// on the board).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Receiver {
    /// The computer: accepts board → bridge types (below 0x80).
    Bridge,
    /// The board: accepts bridge → board types (0x80 and up).
    Board,
}
