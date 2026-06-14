//! An image input expected by a model.

use serde::{Deserialize, Serialize};

use crate::spec::layouts::ImageLayout;

fn default_uint8() -> String {
    "uint8".to_owned()
}

fn default_bilinear_aa() -> String {
    "bilinear_aa".to_owned()
}

/// An image input expected by a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageInput {
    pub key: String,
    pub role: String,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub layout: ImageLayout,
    #[serde(default = "default_uint8")]
    pub dtype: String,
    #[serde(default)]
    pub normalize: bool,
    #[serde(default)]
    pub lead_dims: u32,
    #[serde(default)]
    pub upside_down: bool,
    /// Resize algorithm the model's training pipeline used. A constrained
    /// string (not an enum) so future additive values degrade to a typed
    /// resolution error on older cores instead of a parse failure.
    #[serde(default = "default_bilinear_aa")]
    pub resample: String,
}
