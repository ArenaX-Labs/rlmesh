//! A text input expected by a model.

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
pub struct TextInput {
    pub key: String,
    pub role: String,
    #[serde(default)]
    pub container: TextContainer,
    #[serde(default)]
    pub default: Option<String>,
}
