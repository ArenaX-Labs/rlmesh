//! The serial tether (spec `tether-spec.md`, frozen at v6): a board without a
//! network speaks framed messages over a byte link to a computer, and the
//! computer is the RLMesh robot for it.
//!
//! This crate is the byte layer both ends share, with no allocation and no
//! I/O:
//!
//! - [`crc`]: CRC-16/CCITT-FALSE over a frame, CRC-32 over the map;
//! - [`cobs`]: COBS encoding without a copy of the source, and a
//!   byte-at-a-time decoder into a fixed buffer;
//! - [`frame`]: `type | seq | payload | crc` frames on top of both;
//! - [`msg`]: the messages, with exact lengths and reserved-bit checks;
//! - [`pack`]: a part's dense cable form against the session's 8-byte
//!   aligned layout.

#![no_std]
#![warn(missing_docs)]

pub mod cobs;
pub mod crc;
pub mod frame;
pub mod msg;
pub mod pack;
