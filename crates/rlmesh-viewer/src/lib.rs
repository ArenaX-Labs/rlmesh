//! Built-in live debug viewer for RLMesh evaluations.
//!
//! A thin, synchronous viewer the Python `Session` loop feeds each step. The
//! Session already decodes role-addressed camera frames via `session.read()`; this
//! crate normalizes them to RGB, encodes them (JPEG by default, PNG for lossless),
//! draws each frame to a terminal (the primary backend — in-place ANSI
//! half-blocks over bare SSH) and/or serves it over a tiny HTTP server for a
//! browser. No async runtime, no dependency on the RLMesh runtime — see
//! [`Viewer`].

mod frame;
mod graphics;
mod http;
mod terminal;
mod viewer;

pub use frame::FrameFormat;
pub use viewer::{Backend, Viewer};
