//! Semantic role vocabulary for matching env features to model inputs.
//!
//! Roles are an open vocabulary: any string works as long as the env and
//! model specs agree, and the resolver itself treats every role as an
//! opaque string.
//!
//! Width conventions: roles do not imply dims mechanically -- specs pin
//! widths explicitly where they matter.

pub mod core;
pub mod manipulation;
