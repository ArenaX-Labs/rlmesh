//! The device runtime a networked board links: a static library with a C
//! header (`include/rlmesh_device.h`), built without std for the board
//! (`xtensa-esp32-none-elf`) and linked into its firmware (the Arduino
//! library in `integrations/esp32`).
//!
//! The board's clock and randomness come in as C callbacks, and memory
//! comes from the C library's `malloc`, so nothing here is specific to one
//! MCU.
//!
//! - [`timing`]: what the session's crypto costs on this board: the IK
//!   handshake, verifying a signed authority state, and opening and sealing
//!   the chunks and samples of a control loop.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod ffi;
pub mod robot;
#[cfg(all(not(feature = "std"), target_os = "none"))]
mod sys;
pub mod timing;
