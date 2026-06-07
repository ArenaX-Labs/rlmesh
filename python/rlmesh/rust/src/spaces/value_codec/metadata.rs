use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyList, PyTuple};
use rlmesh_spaces::v1::{MetaMap, MetaValue};

pub(crate) fn meta_map_to_pydict<'py>(
    py: Python<'py>,
    value: &MetaMap,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, item) in value {
        dict.set_item(key, meta_value_to_py(py, item)?)?;
    }
    Ok(dict)
}

pub(super) fn normalize_py_value<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
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

pub(crate) fn py_any_to_meta_map(value: &Bound<'_, PyAny>) -> PyResult<MetaMap> {
    let normalized = normalize_py_value(value)?;
    let dict = normalized.cast::<PyDict>()?;
    let mut fields = BTreeMap::new();
    for (key, item) in dict.iter() {
        fields.insert(key.extract::<String>()?, py_any_to_meta_value(&item)?);
    }
    Ok(fields)
}

fn py_any_to_meta_value(value: &Bound<'_, PyAny>) -> PyResult<MetaValue> {
    let normalized = normalize_py_value(value)?;

    if normalized.is_none() {
        return Ok(MetaValue::Null);
    }
    if let Ok(dict) = normalized.cast::<PyDict>() {
        let mut fields = BTreeMap::new();
        for (key, item) in dict.iter() {
            fields.insert(key.extract::<String>()?, py_any_to_meta_value(&item)?);
        }
        return Ok(MetaValue::Map(fields));
    }
    if let Ok(list) = normalized.cast::<PyList>() {
        let values = list
            .iter()
            .map(|item| py_any_to_meta_value(&item))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(MetaValue::List(values));
    }
    if let Ok(tuple) = normalized.cast::<PyTuple>() {
        let values = tuple
            .iter()
            .map(|item| py_any_to_meta_value(&item))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(MetaValue::List(values));
    }
    if let Ok(flag) = normalized.extract::<bool>() {
        return Ok(MetaValue::Bool(flag));
    }
    if let Ok(number) = normalized.extract::<i64>() {
        return Ok(MetaValue::Int(number));
    }
    if let Ok(number) = normalized.extract::<f64>() {
        return Ok(MetaValue::Float(number));
    }
    if let Ok(text) = normalized.extract::<String>() {
        return Ok(MetaValue::String(text));
    }
    if normalized.hasattr("value")? {
        let enum_value = normalized.getattr("value")?;
        if !enum_value.is(&normalized)
            && let Ok(meta_value) = py_any_to_meta_value(&enum_value)
        {
            return Ok(meta_value);
        }
    }
    if normalized.hasattr("name")? {
        let enum_name = normalized.getattr("name")?;
        if !enum_name.is(&normalized)
            && let Ok(text) = enum_name.extract::<String>()
        {
            return Ok(MetaValue::String(text));
        }
    }

    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "value of type '{}' is not supported in RLMesh metadata",
        normalized.get_type().name()?
    )))
}

fn meta_value_to_py<'py>(py: Python<'py>, value: &MetaValue) -> PyResult<Py<PyAny>> {
    Ok(match value {
        MetaValue::Null => py.None(),
        MetaValue::Bool(value) => PyBool::new(py, *value).to_owned().into_any().unbind(),
        MetaValue::Int(value) => value.into_pyobject(py)?.into_any().unbind(),
        MetaValue::Float(value) => value.into_pyobject(py)?.into_any().unbind(),
        MetaValue::String(value) => value.into_pyobject(py)?.into_any().unbind(),
        MetaValue::List(values) => {
            let values = values
                .iter()
                .map(|value| meta_value_to_py(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, values)?.into_any().unbind()
        }
        MetaValue::Map(values) => meta_map_to_pydict(py, values)?.into_any().unbind(),
    })
}
