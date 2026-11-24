//! Python binding for space sampling.
//!
//! Drives `rlmesh_spaces::sample` with the caller's RNG and returns the sample in the same
//! Python form as any decoded value: a native `Tensor` for array leaves, `int`/`str` for
//! scalars, `dict`/`tuple` for composites.

use pyo3::prelude::*;
use pyo3::types::PyAny;
use rlmesh_spaces::ChaCha12Rng;
use rlmesh_spaces::spaces::SpaceSpec;

use crate::spaces::space_value_to_py_neutral;

pub(super) fn sample_space_value<'py>(
    py: Python<'py>,
    space: &SpaceSpec,
    rng: &mut ChaCha12Rng,
) -> PyResult<Bound<'py, PyAny>> {
    let value = rlmesh_spaces::sample_with(space, rng)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    space_value_to_py_neutral(py, &value, space)
}
