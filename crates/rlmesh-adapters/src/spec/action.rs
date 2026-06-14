//! Action layout types shared by env and model declarations.

use serde::{Deserialize, Serialize};

use super::rotations::RotationEncoding;

/// One contiguous slice of an action vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionComponent {
    pub role: String,
    #[serde(default)]
    pub dim: u32,
    #[serde(default)]
    pub encoding: Option<RotationEncoding>,
    #[serde(default)]
    pub range: Option<(f64, f64)>,
    #[serde(default)]
    pub binary: bool,
}

/// Ordered action components plus optional clipping bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionLayout {
    pub components: Vec<ActionComponent>,
    #[serde(default)]
    pub clip: Option<(f64, f64)>,
}
