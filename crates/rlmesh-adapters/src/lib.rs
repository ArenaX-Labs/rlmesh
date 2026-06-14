//! Declarative env/model IO adapter core for RLMesh.
//!
//! Environments and models describe their IO formats once as versioned,
//! pure-data specs; [`v1::resolve`] derives the concrete per-pairing
//! conversion plan by matching semantic roles. No code is ever evaluated
//! from spec data: custom transforms resolve to host-language callbacks
//! that bindings materialize themselves.
//!
//! The JSON wire format and resolution semantics are frozen per version by
//! the conformance vectors under this crate's `conformance/` directory;
//! every implementation and binding must pass them.

pub mod v1;
