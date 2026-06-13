//! Payload byte-size accounting for profiling.

use rlmesh_spaces::SpaceValue;

/// Byte size of an optional space value (0 when absent).
pub(crate) fn observation_size(value: Option<&SpaceValue>) -> usize {
    value.map_or(0, space_value_size)
}

/// Byte size of a single space value's payload.
pub(crate) fn space_value_size(value: &SpaceValue) -> usize {
    match value {
        SpaceValue::Box(value) => value.nbytes(),
        SpaceValue::Discrete(_) => std::mem::size_of::<i64>(),
        SpaceValue::MultiBinary(values) => values.len(),
        SpaceValue::MultiDiscrete(values) => values.len() * std::mem::size_of::<i64>(),
        SpaceValue::Text(value) => value.len(),
        SpaceValue::Dict(values) => values.values().map(space_value_size).sum(),
        SpaceValue::Tuple(values) => values.iter().map(space_value_size).sum(),
    }
}
