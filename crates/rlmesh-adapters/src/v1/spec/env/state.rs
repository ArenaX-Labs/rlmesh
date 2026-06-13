//! A numeric proprioception entry in an environment observation.

use serde::{Deserialize, Serialize};

use super::super::rotations::RotationEncoding;

/// A numeric proprioception entry in an environment observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvState {
    pub key: String,
    pub role: String,
    #[serde(default)]
    pub dim: Option<u32>,
    #[serde(default)]
    pub encoding: Option<RotationEncoding>,
    #[serde(default)]
    pub range: Option<(f64, f64)>,
}
