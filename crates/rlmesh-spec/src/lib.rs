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
//!
//! The generic core (`rlmesh-wire`, `rlmesh-session`, `rlmesh-edge`) works on
//! streams and never interprets kinds; only this crate does.

pub mod canonical;
pub mod kinds;
pub mod normalise;
pub mod rotation;
pub mod units;

pub use kinds::Kind;
pub use normalise::normalise;
pub use rotation::Rotation;
pub use units::Quantity;
