//! Space value types and validation.
//!
//! Runtime value representations for RLMesh spaces
//! and functions to validate that values belong to their spaces.

use std::collections::BTreeMap;

use crate::errors::{SpaceError, err_space};
use crate::v1::spaces::composite::{contains_dict, contains_tuple};
use crate::v1::spaces::fundamental::{
    contains_box, contains_discrete, contains_multibinary, contains_multidiscrete, contains_text,
};
use crate::v1::spaces::{SpaceSpec, SpaceType};
use crate::v1::tensor::Tensor;

/// A runtime value that can belong to a space.
///
/// This is the Rust representation of values that flow through the
/// environment interface (observations, actions, etc.).
#[derive(Debug, Clone, PartialEq)]
pub enum SpaceValue {
    /// Box space value - continuous tensor
    Box(Tensor),

    /// Discrete space value - single integer
    Discrete(i64),

    /// MultiBinary space value - boolean array
    MultiBinary(Vec<bool>),

    /// MultiDiscrete space value - integer array
    MultiDiscrete(Vec<i64>),

    /// Text space value - string
    Text(String),

    /// Dict space value - named sub-values
    Dict(BTreeMap<String, SpaceValue>),

    /// Tuple space value - ordered sub-values
    Tuple(Vec<SpaceValue>),
}

/// Check if a value belongs to a space.
///
/// Returns Ok(()) if the value is valid for the space, or an error describing
/// why the value doesn't fit.
pub fn contains(space: &SpaceSpec, value: &SpaceValue) -> Result<(), SpaceError> {
    contains_at(space, value, "$")
}

pub(crate) fn contains_at(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    match space.space_type() {
        SpaceType::Box => contains_box(space, value, path),
        SpaceType::Discrete => contains_discrete(space, value, path),
        SpaceType::MultiBinary => contains_multibinary(space, value, path),
        SpaceType::MultiDiscrete => contains_multidiscrete(space, value, path),
        SpaceType::Text => contains_text(space, value, path),
        SpaceType::Dict => contains_dict(space, value, path),
        SpaceType::Tuple => contains_tuple(space, value, path),
        SpaceType::Unspecified => err_space!(path, "space type not specified"),
    }
}
