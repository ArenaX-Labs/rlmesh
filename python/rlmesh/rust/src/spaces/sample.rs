//! Python binding for space sampling.
//!
//! The sampling logic itself lives in `rlmesh_spaces::sample` (the reusable,
//! deterministic, ChaCha12-pinned core). This shim only drives that core with
//! the route's RNG and marshals the resulting native `SpaceValue` into its
//! Python representation, reusing the value codec so a sample looks exactly like
//! any other decoded value (native `Tensor` for array leaves, `int`/`str` for
//! scalars, `dict`/`tuple` for composites).

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
    // `sample_with` advances `rng` in place, so successive samples on one Space
    // draw from the advanced stream (gym's seed-once-then-sample semantics).
    let value = rlmesh_spaces::sample_with(space, rng)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    space_value_to_py_neutral(py, &value, space)
}
