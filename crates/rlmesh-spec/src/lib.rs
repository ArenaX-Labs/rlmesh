//! The physical-AI layer of an RLMesh real-time session (protocol v2, the
//! v0.2 spec §2-§8): what turns a robot map and a policy map into streams,
//! a binding and delivery contracts. Sans-IO, pure Rust.
//!
//! - [`kinds`], [`units`], [`rotation`]: the kind table (§4), units (§5.1)
//!   and rotation encodings (§5.3).
//! - [`normalise`](mod@normalise): the shared normaliser. The host lowers an
//!   embodiment's authored robot map (JSON) to
//!   [`PartSpec`](rlmesh_wire::PartSpec)s with the kind table's fields and
//!   defaults, and fingerprints the canonical form (§3.3).
//! - [`canonical`]: RFC 8785 JSON and the BLAKE3 fingerprints over it.
//! - [`policy`], [`selector`]: the policy map and run-time options (§6.1-
//!   §6.4), and the `at` selectors with scope and captures.
//! - [`bind`](mod@bind): resolution (§6.5) to a [`Binding`] (the table, its
//!   delivery contracts, action grids and fingerprint) or a [`Refusal`]
//!   listing every problem; [`plan`] is the conversion vocabulary (§7) its
//!   rows use, and [`explain`] renders refusals (§6.7) and tables (§6.8).
//! - [`contract`]: delivery contracts, freshness bounds, admission and the
//!   publication plan per stream (§8.2, §8.7), shared by the binder and the
//!   host.
//!
//! The generic core (`rlmesh-wire`, `rlmesh-session`, `rlmesh-edge`) works on
//! streams and never interprets kinds; only this crate does.

pub mod bind;
pub mod canonical;
pub mod contract;
pub mod explain;
pub mod kinds;
pub mod normalise;
pub mod plan;
pub mod policy;
pub mod rotation;
pub mod selector;
pub mod units;

pub use bind::{BindError, Binding, Refusal, bind};
pub use kinds::Kind;
pub use normalise::normalise;
pub use policy::{Policy, RunOptions};
pub use rotation::Rotation;
pub use units::Quantity;
