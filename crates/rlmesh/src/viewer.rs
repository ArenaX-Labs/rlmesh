//! Built-in live debug viewer (feature `viewer`, on by default).
//!
//! A thin, synchronous viewer a session loop feeds each step: it normalizes
//! role-addressed camera frames to RGB, encodes them (JPEG by default, PNG for
//! lossless), draws each frame to a terminal (the primary backend: in-place
//! ANSI half-blocks over bare SSH) and/or serves it over a tiny HTTP server
//! for a browser. No async runtime; see [`Viewer`]. Runs drive it through
//! [`View`](crate::View).

mod frame;
mod graphics;
mod http;
mod live;
mod terminal;

pub use crate::view::Hud;
pub use frame::FrameFormat;
pub use live::{Backend, Viewer};
