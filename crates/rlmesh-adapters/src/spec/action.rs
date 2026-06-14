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
    // scale/invert/threshold are additive env-side corrections; they are
    // omitted from serialization when unset so layouts that do not use them are
    // byte-identical to before (matching the Python serializer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub invert: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Ordered action components plus optional clipping bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionLayout {
    pub components: Vec<ActionComponent>,
    #[serde(default)]
    pub clip: Option<(f64, f64)>,
}
