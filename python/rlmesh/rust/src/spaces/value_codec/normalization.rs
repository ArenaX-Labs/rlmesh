use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Normalize Python wrappers accepted at SpaceValue boundaries.
///
/// This is intentionally narrow: it unwraps namedtuples and array/scalar wrapper
/// types into plain Python containers or scalars before space-specific encoding
/// applies shape, dtype, and conformance checks.
pub(super) fn normalize_space_value_input<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    normalize_python_wrapper(value)
}

/// Normalize Python wrappers accepted in metadata maps.
///
/// Metadata has its own converter because its policy is more permissive than
/// runtime values: after this wrapper normalization it also accepts enum
/// `.value`/`.name` fallbacks in `metadata.rs`.
pub(super) fn normalize_metadata_value<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    normalize_python_wrapper(value)
}

fn normalize_python_wrapper<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if value.hasattr("_asdict")? {
        return value.call_method0("_asdict");
    }
    if value.hasattr("tolist")? {
        return value.call_method0("tolist");
    }
    if value.hasattr("item")? {
        return value.call_method0("item");
    }
    Ok(value.clone())
}
