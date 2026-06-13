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
    /// Pixel height of the image, derived from the observation space by
    /// [`join`](crate::v1::join). Lets a model that resizes only one axis
    /// fill the other from the env's native resolution.
    #[serde(default)]
    pub height: u32,
    /// Pixel width of the image, derived from the observation space.
    #[serde(default)]
    pub width: u32,
}
