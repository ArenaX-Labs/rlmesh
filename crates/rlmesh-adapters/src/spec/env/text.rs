//! A text entry (typically the task instruction) in an observation.
//!
//! Internal post-`join` form; never serialized (see `spec::env`), so no serde.

/// A text entry (typically the task instruction) in an observation.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvText {
    pub key: String,
    pub role: String,
}
