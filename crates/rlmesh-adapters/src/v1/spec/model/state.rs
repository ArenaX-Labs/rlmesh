//! A numeric state input expected by a model.

use serde::{Deserialize, Serialize};

use super::super::rotations::RotationEncoding;

fn default_float32() -> String {
    "float32".to_owned()
}

/// One piece of a model state vector, sourced from an env state feature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateComponent {
    pub role: String,
    #[serde(default)]
    pub encoding: Option<RotationEncoding>,
    #[serde(default)]
    pub dim: Option<u32>,
    #[serde(default)]
    pub index: Option<u32>,
    #[serde(default)]
    pub optional: bool,
}

/// Container kind for a resolved state value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StateContainer {
    #[default]
    Array,
    List,
}

impl StateContainer {
    /// Wire/display name (matches the JSON form).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Array => "array",
            Self::List => "list",
        }
    }
}

/// A numeric state input expected by a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateInput {
    pub key: String,
    pub components: Vec<StateComponent>,
    #[serde(default)]
    pub pad_to: Option<u32>,
    #[serde(default = "default_float32")]
    pub dtype: String,
    #[serde(default)]
    pub reshape: Option<Vec<i64>>,
    #[serde(default)]
    pub container: StateContainer,
}
