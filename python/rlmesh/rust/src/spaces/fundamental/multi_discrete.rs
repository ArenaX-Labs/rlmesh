use crate::spaces::utils::dtype_to_py;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rlmesh_spaces::spaces::*;

pub fn make_multidiscrete<'py>(
    py: Python<'py>,
    spaces: &Bound<'py, PyAny>,
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    let nvec_spec = match &space.spec {
        Some(SpaceKind::MultiDiscrete(spec)) => spec,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "spec.multi_discrete missing",
            ));
        }
    };

    let np = py.import("numpy")?;
    let dtype = dtype_to_py(py, space.dtype)?;
    // `nvec` is flat (row-major); reshape it back to the logical `shape` so a
    // rank-2 MultiDiscrete materializes as a 2-D nvec array for Gymnasium.
    let flat = np
        .getattr("array")?
        .call1((nvec_spec.nvec.clone(), &dtype))?;
    let nvec_value = if space.shape.len() > 1 {
        let shape: Vec<i64> = space.shape.clone();
        flat.call_method1("reshape", (shape,))?
    } else {
        flat
    };

    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype)?;
    spaces
        .getattr("MultiDiscrete")?
        .call((nvec_value,), Some(&kwargs))
}

pub fn parse_multidiscrete<'py>(space: &Bound<'py, PyAny>) -> PyResult<SpaceSpec> {
    let nvec_value = space.getattr("nvec")?;
    let nvec_value = if nvec_value.hasattr("tolist")? {
        nvec_value.call_method0("tolist")?
    } else {
        nvec_value
    };

    if let Ok(vector) = nvec_value.extract::<Vec<i64>>() {
        return MultiDiscreteBuilder::vector(vector)
            .build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
    }

    let matrix = nvec_value.extract::<Vec<Vec<i64>>>()?;
    MultiDiscreteBuilder::matrix(matrix)
        .build()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
