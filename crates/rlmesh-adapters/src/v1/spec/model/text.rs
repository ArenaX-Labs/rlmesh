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

impl TextContainer {
    /// Wire/display name (matches the JSON form).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Str => "str",
            Self::List => "list",
        }
    }
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
