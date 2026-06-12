use crate::spaces::space::{make_space, parse_space};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use rlmesh_spaces::spaces::*;

pub(crate) fn make_tuple<'py>(
    py: Python<'py>,
    spaces: &Bound<'py, PyAny>,
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    let tuple_spec = match &space.spec {
        Some(space_spec::Spec::Tuple(t)) => t,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "missing tuple detail",
            ));
        }
    };

    let values = tuple_spec
        .spaces
        .iter()
        .map(|child| make_space(py, child))
        .collect::<PyResult<Vec<_>>>()?;

    let py_tuple = PyTuple::new(py, values)?;
    spaces.getattr("Tuple")?.call((py_tuple.as_any(),), None)
}

pub(crate) fn parse_tuple<'py>(space: &Bound<'py, PyAny>) -> PyResult<SpaceSpec> {
    let spaces_any = space.getattr("spaces")?;
    let py_tuple = spaces_any.cast::<PyTuple>()?;

    let spaces = py_tuple
        .iter()
        .map(|v| parse_space(&v))
        .collect::<Result<Vec<_>, _>>()?;

    TupleSpaceBuilder::new()
        .extend(spaces)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
