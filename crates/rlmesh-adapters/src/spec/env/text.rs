//! A text entry (typically the task instruction) in an observation.
//!
//! Internal post-`join` form; never serialized (see `spec::env`), so no serde.

use crate::path::NodePath;

/// A text entry (typically the task instruction) in an observation.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvText {
    /// Structured source path into the raw observation tree this text is read
    /// from (the env-side placement); empty (root) for a bare single-leaf obs.
    pub source: NodePath,
    pub role: String,
}
