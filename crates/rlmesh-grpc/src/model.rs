//! ModelService transport: the [`ModelClient`] that drives a model endpoint's
//! Join stream.
//!
//! The served side (the `ModelService` implementation) lives in the `rlmesh`
//! facade, since it is parameterized over the public `ModelHandler` family. This
//! crate owns only the client and its request/response demux.

pub mod client;
mod stream;
mod validation;
mod wire;

pub use client::ModelClient;
