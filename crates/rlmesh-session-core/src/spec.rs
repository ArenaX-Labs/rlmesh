//! The physical-AI layer of an RLMesh real-time session: what turns a robot map and a
//! policy map into streams,
//! a binding and delivery contracts. Sans-IO, pure Rust.
//!
//! - [`kinds`], [`units`], [`rotation`]: the kind table, units
//!   and rotation encodings.
//! - [`normalise`](mod@normalise): the shared normaliser. The host lowers an
//!   embodiment's authored robot map (JSON) to
//!   [`PartSpec`](crate::wire::PartSpec)s with the kind table's fields and
//!   defaults, and fingerprints the canonical form.
//! - [`canonical`]: RFC 8785 JSON and the BLAKE3 fingerprints over it.
//! - [`policy`], [`selector`]: the policy map and run-time options, and the `at`
//!   selectors with scope and captures; [`inputs`] and [`actions`] build a
//!   policy map's entries by kind for [`Policy::builder`].
//! - [`bind`](mod@bind): resolution to a [`Binding`] (the table, its
//!   delivery contracts, action grids and fingerprint) or a [`Refusal`]
//!   listing every problem; [`plan`] is the conversion vocabulary its
//!   rows use, and [`explain`] renders refusals and tables.
//! - [`apply`]: the plans executed: a bound input's samples to what
//!   `predict` receives, and a chunk from `predict` to packed command
//!   values; [`image`] holds the image operations, with v1's kernels.
//! - [`contract`]: delivery contracts, freshness bounds, admission and the
//!   publication plan per stream, shared by the binder and the
//!   host.
//! - [`role`]: the binder's Gym door, which binds a v1 `ModelSpec` to a
//!   Gym env's tagged leaves by role, with the same conversions.
//! - [`view`](mod@view), [`frames`]: a scene's view, its placed robots'
//!   parts in one path tree, and the one helper over its places' static
//!   poses.
//!
//! The generic core ([`wire`](crate::wire), `rlmesh-session` and its edge)
//! works on streams and never interprets kinds; only this module does.

#[allow(unused_imports)]
use crate::prelude::*;

pub mod actions;
pub mod apply;
pub mod bind;
pub mod canonical;
pub mod contract;
pub mod convert;
pub mod explain;
pub mod frames;
pub mod image;
pub mod inputs;
pub mod kinds;
pub mod normalise;
pub mod plan;
pub mod policy;
pub mod projection;
pub mod role;
pub mod rotation;
pub mod selector;
pub mod units;
pub mod view;

pub use bind::{BindError, Binding, Refusal, Target, bind, bind_in};
pub use kinds::Kind;
pub use normalise::normalise;
pub use policy::{Policy, PolicyBuilder, RunOptions};
pub use projection::Projection;
pub use rotation::Rotation;
pub use units::Quantity;
