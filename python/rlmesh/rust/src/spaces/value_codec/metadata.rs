use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyList, PyTuple};
use rlmesh_spaces::{MetaMap, MetaValue};

use super::normalization::normalize_metadata_value;

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

pub(crate) fn py_any_to_meta_map(value: &Bound<'_, PyAny>) -> PyResult<MetaMap> {
    let normalized = normalize_metadata_value(value)?;
    let dict = normalized.cast::<PyDict>()?;
    let mut fields = BTreeMap::new();
    for (key, item) in dict.iter() {
        fields.insert(key.extract::<String>()?, py_any_to_meta_value(&item)?);
    }
    Ok(fields)
}

fn py_any_to_meta_value(value: &Bound<'_, PyAny>) -> PyResult<MetaValue> {
    let normalized = normalize_metadata_value(value)?;

    if normalized.is_none() {
        return Ok(MetaValue::Null);
    }
    // `bytes` must be matched before the sequence and integer paths: it is
    // neither a list nor a scalar, and carries raw binary exactly.
    if let Ok(bytes) = normalized.cast::<PyBytes>() {
        return Ok(MetaValue::Bytes(bytes.as_bytes().to_vec()));
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
        MetaValue::Bytes(value) => PyBytes::new(py, value).into_any().unbind(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyBytesMethods;

    #[test]
    fn meta_preserves_int_float_bytes_types_exactly() {
        Python::attach(|py| {
            // 2^53 + 1 (inexact as f64), a whole-number float, and raw bytes:
            // the types and exact values must survive the Python round-trip.
            let value = py
                .eval(
                    pyo3::ffi::c_str!(
                        "{'big': 9007199254740993, 'lr': 2.0, 'blob': b'\\x00\\x01\\xff'}"
                    ),
                    None,
                    None,
                )
                .unwrap();
            let native = py_any_to_meta_map(&value).unwrap();
            assert_eq!(native.get("big"), Some(&MetaValue::Int((1i64 << 53) + 1)));
            assert_eq!(native.get("lr"), Some(&MetaValue::Float(2.0)));
            assert_eq!(native.get("blob"), Some(&MetaValue::Bytes(vec![0, 1, 255])));

            let roundtrip = meta_map_to_pydict(py, &native).unwrap();
            assert_eq!(
                roundtrip
                    .get_item("big")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                (1i64 << 53) + 1
            );
            // The whole-number float stays a Python float, not an int.
            let lr = roundtrip.get_item("lr").unwrap().unwrap();
            assert!(lr.is_instance_of::<pyo3::types::PyFloat>());
            // Bytes stay bytes, byte-exact.
            let blob = roundtrip.get_item("blob").unwrap().unwrap();
            let blob = blob.cast::<PyBytes>().unwrap();
            assert_eq!(blob.as_bytes(), &[0, 1, 255]);
        });
    }
}
