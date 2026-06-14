//! A numeric state input expected by a model.

use serde::{Deserialize, Serialize};

use crate::spec::rotations::RotationEncoding;

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
    /// Target value range. When set and the env feature declares a (derived
    /// or tagged) source range, values are affinely mapped from the env
    /// range into this one — the state-side analogue of action range mapping.
    #[serde(default)]
    pub range: Option<(f64, f64)>,
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
