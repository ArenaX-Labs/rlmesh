use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rlmesh_spaces::v1::spaces::*;
use rlmesh_spaces::v1::{DType, DiscreteSpec};

pub fn make_discrete<'py>(
    py: Python<'py>,
    spaces: &Bound<'py, PyAny>,
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    let d = match &space.spec {
        Some(space_spec::Spec::Discrete(d)) => d,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "spec.discrete missing",
            ));
        }
    };

    let kwargs = PyDict::new(py);
    kwargs.set_item("n", d.n)?;

    if d.start != 0 {
        kwargs.set_item("start", d.start)?;
    }

    let ctor = spaces.getattr("Discrete")?;
    if let Ok(obj) = ctor.call((), Some(&kwargs)) {
        return Ok(obj);
    }

    ctor.call1((d.n,))
}

pub fn parse_discrete<'py>(space: &Bound<'py, PyAny>) -> PyResult<SpaceSpec> {
    let n: i64 = space.getattr("n")?.extract()?;
    let start: i64 = match space.getattr("start") {
        Ok(v) => v.extract().unwrap_or(0),
        Err(_) => 0,
    };

    Ok(SpaceSpec {
        shape: vec![],
        dtype: DType::Int64,
        spec: Some(space_spec::Spec::Discrete(DiscreteSpec { n, start })),
    })
}
