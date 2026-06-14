use crate::spaces::utils::dtype_to_py;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rlmesh_spaces::MultiDiscreteNvec;
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
    let nvec_value = match &nvec_spec.nvec {
        Some(MultiDiscreteNvec::Flat(vector)) => {
            np.getattr("array")?.call1((vector.clone(), &dtype))?
        }
        Some(MultiDiscreteNvec::Shaped(matrix)) => {
            let rows: Vec<Vec<i64>> = matrix.clone();
            np.getattr("array")?.call1((rows, &dtype))?
        }
        None => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "MultiDiscreteSpec.nvec missing",
            ));
        }
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
