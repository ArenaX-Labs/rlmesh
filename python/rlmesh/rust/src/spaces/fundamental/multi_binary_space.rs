use pyo3::prelude::*;
use pyo3::types::PyAny;
use rlmesh_spaces::v1::multi_binary_spec;
use rlmesh_spaces::v1::spaces::*;

pub fn make_multibinary<'py>(
    py: Python<'py>,
    spaces: &Bound<'py, PyAny>,
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    let multi_binary_spec = match &space.spec {
        Some(space_spec::Spec::MultiBinary(spec)) => spec,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "spec.multi_binary missing",
            ));
        }
    };

    let n_value = match &multi_binary_spec.n {
        Some(multi_binary_spec::N::Size(size)) => (*size).into_pyobject(py)?.unbind().into_any(),
        Some(multi_binary_spec::N::Dims(dims)) => {
            dims.data.clone().into_pyobject(py)?.unbind().into_any()
        }
        None => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "MultiBinarySpec.n missing",
            ));
        }
    };

    spaces.getattr("MultiBinary")?.call1((n_value,))
}

pub fn parse_multibinary<'py>(space: &Bound<'py, PyAny>) -> PyResult<SpaceSpec> {
    let n_value = space.getattr("n")?;
    if let Ok(size) = n_value.extract::<i64>() {
        return MultiBinaryBuilder::scalar(size)
            .build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
    }

    let dims = if let Ok(dims) = n_value.extract::<Vec<i64>>() {
        dims
    } else {
        let tolist = n_value.call_method0("tolist")?;
        tolist.extract::<Vec<i64>>()?
    };

    MultiBinaryBuilder::shape(dims)
        .build()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
