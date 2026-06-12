//! A camera image entry in an environment observation.

use serde::{Deserialize, Serialize};

use super::super::layouts::ImageLayout;

/// A camera image entry in an environment observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvImage {
    pub key: String,
    pub role: String,
    #[serde(default)]
    pub layout: ImageLayout,
    #[serde(default)]
    pub upside_down: bool,
}
