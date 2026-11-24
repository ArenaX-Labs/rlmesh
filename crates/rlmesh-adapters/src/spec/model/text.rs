use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Container kind for a resolved text value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextContainer {
    #[default]
    Str,
    List,
}

/// A text input expected by a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Text {
    pub role: String,
    #[serde(default)]
    pub container: TextContainer,
    /// Fallback value used when the observation omits the feature; `None` omits
    /// the input from the payload instead.
    #[serde(default)]
    pub fill: Option<String>,
    /// Unrecognized additive fields, retained for round-trip (see the strict-v1 publish gate).
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}
