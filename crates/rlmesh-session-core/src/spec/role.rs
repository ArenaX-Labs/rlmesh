//! The binder's Gym door: a model bound to a Gym env by role.
//!
//! - [`roles`]: the role vocabulary a Gym env and a model spec bind by, and
//!   its registry.
//! - [`frames`]: the `frame`, `reference` and `provenance` vocabularies.
//! - [`path`]: where a feature is read from and a model value lands.

pub mod frames;
pub mod path;
pub mod roles;

pub use path::{NodePath, PathSeg};
