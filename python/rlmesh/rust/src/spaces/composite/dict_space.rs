use crate::spaces::space::{make_space, parse_space};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rlmesh_spaces::v1::spaces::*;

pub(crate) fn make_dict<'py>(
    py: Python<'py>,
    spaces: &Bound<'py, PyAny>,
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    let dict_spec = match &space.spec {
        Some(space_spec::Spec::Dict(d)) => d,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "missing dict detail",
            ));
        }
    };

    if dict_spec.keys.len() != dict_spec.spaces.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "DictSpec.keys/spaces length mismatch",
        ));
    }

    let py_dict = PyDict::new(py);
    for (key, child_spec) in dict_spec.keys.iter().zip(dict_spec.spaces.iter()) {
        let child_space = make_space(py, child_spec)?;
        py_dict.set_item(key, child_space)?;
    }

    spaces.getattr("Dict")?.call((py_dict,), None)
}

pub(crate) fn parse_dict<'py>(space: &Bound<'py, PyAny>) -> PyResult<SpaceSpec> {
    let spaces_any = space.getattr("spaces")?;
    let py_dict = spaces_any.cast::<PyDict>()?;

    let entries = py_dict
        .iter()
        .map(|(k, v)| {
            let key = match k.extract::<String>() {
                Ok(s) => s,
                Err(_) => k
                    .str()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            };

            let value = parse_space(&v)?;
            Ok::<_, PyErr>((key, value))
        })
        .collect::<Result<Vec<_>, _>>()?;

    DictSpaceBuilder::new()
        .extend(entries)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
