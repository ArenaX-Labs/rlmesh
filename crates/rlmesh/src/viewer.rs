//! Built-in live debug viewer (feature `viewer`, on by default).
//!
//! A synchronous `Viewer` a session loop feeds each step: it normalizes
//! role-addressed camera frames to RGB, encodes them (JPEG by default, PNG for
//! lossless), and draws each frame to a terminal (in-place ANSI half-blocks,
//! fine over bare SSH) and/or serves it over a small HTTP server for a
//! browser. Runs drive it through [`crate::View`].

mod frame;
mod graphics;
mod http;
mod live;
mod terminal;

pub use crate::view::Hud;
pub use frame::FrameFormat;
pub use live::{Backend, Viewer};
