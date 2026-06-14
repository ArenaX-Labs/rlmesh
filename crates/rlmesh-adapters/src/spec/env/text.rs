//! A text entry (typically the task instruction) in an observation.

use serde::{Deserialize, Serialize};

/// A text entry (typically the task instruction) in an observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvText {
    pub key: String,
    pub role: String,
}
