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
    // `nvec` (and `start`) are flat (row-major); reshape them back to the
    // logical `shape` so a rank-2 MultiDiscrete materializes as 2-D arrays for
    // Gymnasium.
    let shaped = |values: &[i64]| -> PyResult<Bound<'py, PyAny>> {
        let flat = np.getattr("array")?.call1((values.to_vec(), &dtype))?;
        if space.shape.len() > 1 {
            flat.call_method1("reshape", (space.shape.clone(),))
        } else {
            Ok(flat)
        }
    };
    let nvec_value = shaped(&nvec_spec.nvec)?;

    let kwargs = PyDict::new(py);
    if nvec_spec.has_start() {
        kwargs.set_item("start", shaped(&nvec_spec.start)?)?;
    }
    kwargs.set_item("dtype", dtype)?;
    spaces
        .getattr("MultiDiscrete")?
        .call((nvec_value,), Some(&kwargs))
}

pub fn parse_multidiscrete<'py>(space: &Bound<'py, PyAny>) -> PyResult<SpaceSpec> {
    let listed = |name: &str| -> PyResult<Option<Bound<'py, PyAny>>> {
        if !space.hasattr(name)? {
            return Ok(None);
        }
        let value = space.getattr(name)?;
        Ok(Some(if value.hasattr("tolist")? {
            value.call_method0("tolist")?
        } else {
            value
        }))
    };
    let flat = |value: &Bound<'py, PyAny>| -> PyResult<Vec<i64>> {
        match value.extract::<Vec<i64>>() {
            Ok(vector) => Ok(vector),
            Err(_) => Ok(value.extract::<Vec<Vec<i64>>>()?.concat()),
        }
    };
    let nvec_value = listed("nvec")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("MultiDiscrete has no nvec"))?;
    let builder = match nvec_value.extract::<Vec<i64>>() {
        Ok(vector) => MultiDiscreteBuilder::vector(vector),
        Err(_) => MultiDiscreteBuilder::matrix(nvec_value.extract::<Vec<Vec<i64>>>()?),
    };
    let start = listed("start")?.map(|v| flat(&v)).transpose()?;
    let builder = match start {
        Some(start) if start.iter().any(|s| *s != 0) => builder.start(start),
        _ => builder,
    };
    builder
        .build()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
