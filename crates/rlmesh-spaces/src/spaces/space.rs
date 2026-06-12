use crate::errors::{SpaceError, err_space};
use crate::spaces::composite::*;
use crate::spaces::fundamental::*;
use crate::{SpaceSpec, SpaceType};

/// Validate a space specification.
///
/// This recursively validates all nested spaces and ensures all constraints
/// are satisfied (shape matches bounds, dtype is valid, etc.).
pub fn validate_space(spec: &SpaceSpec) -> Result<(), SpaceError> {
    validate_space_at(spec, "$")
}

pub(crate) fn validate_space_at(space: &SpaceSpec, path: &str) -> Result<(), SpaceError> {
    match space.space_type() {
        SpaceType::Box => validate_box_at(space, path),
        SpaceType::Discrete => validate_discrete_at(space, path),
        SpaceType::MultiBinary => validate_multibinary_at(space, path),
        SpaceType::MultiDiscrete => validate_multidiscrete_at(space, path),
        SpaceType::Text => validate_text_at(space, path),
        SpaceType::Dict => validate_dict_at(space, path),
        SpaceType::Tuple => validate_tuple_at(space, path),
        SpaceType::Unspecified => err_space!(path, "spec must be set"),
    }
}
