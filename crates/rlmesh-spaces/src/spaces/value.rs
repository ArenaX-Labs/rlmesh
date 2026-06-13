//! Space value types and validation.
//!
//! Runtime values and validation for RLMesh spaces.

use std::collections::BTreeMap;

use crate::errors::{SpaceError, err_space};
use crate::spaces::composite::{contains_dict, contains_tuple};
use crate::spaces::fundamental::{
    contains_box, contains_discrete, contains_multibinary, contains_multidiscrete, contains_text,
};
use crate::spaces::{SpaceSpec, SpaceType};
use crate::tensor::Tensor;

/// Runtime value carried by an RLMesh space.
#[derive(Debug, Clone, PartialEq)]
pub enum SpaceValue {
    /// Continuous tensor.
    Box(Tensor),

    /// Single integer.
    Discrete(i64),

    /// Boolean array.
    MultiBinary(Vec<bool>),

    /// Integer array.
    MultiDiscrete(Vec<i64>),

    /// String value.
    Text(String),

    /// Named child values.
    Dict(BTreeMap<String, SpaceValue>),

    /// Ordered child values.
    Tuple(Vec<SpaceValue>),
}

/// Validate that `value` belongs to `space`.
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
