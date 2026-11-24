//! The device runtime a networked board links: a static library with a C
//! header (`include/rlmesh_device.h`), built without std for the board
//! (`xtensa-esp32-none-elf`) and linked into its firmware (the Arduino
//! library in `integrations/esp32`).

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod address;
pub mod ffi;
pub mod heap;
pub mod map;
pub mod provision;
pub mod robot;
#[cfg(all(not(feature = "std"), target_os = "none"))]
mod sys;
pub mod timing;
