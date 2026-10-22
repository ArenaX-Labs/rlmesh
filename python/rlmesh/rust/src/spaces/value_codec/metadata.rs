use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyInt, PyList, PyMapping, PyTuple};
use rlmesh_spaces::{DType, MetaMap, MetaValue};

use super::normalization::normalize_metadata_value;
use crate::spaces::tensor::extract_tensor;

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
    if let Ok(dict) = normalized.cast::<PyDict>() {
        return fields(dict, "");
    }
    // Any other mapping (a client's read-only `mappingproxy`) as a dict.
    let mapping = normalized.cast::<PyMapping>()?;
    let dict = PyDict::new(value.py());
    dict.update(mapping)?;
    fields(&dict, "")
}

/// A dict's entries, each refusal naming its key path (`['a'][0]`).
fn fields(dict: &Bound<'_, PyDict>, path: &str) -> PyResult<MetaMap> {
    let mut fields = BTreeMap::new();
    for (key, item) in dict.iter() {
        let path = format!("{path}[{}]", key.repr()?);
        fields.insert(
            key.extract::<String>()?,
            py_any_to_meta_value(&item, &path)?,
        );
    }
    Ok(fields)
}

/// A NumPy array or scalar as an array value: a numeric or bool dtype as a
/// tensor of its bytes, any other as its elements (`None`: not NumPy).
fn numpy_value(value: &Bound<'_, PyAny>, path: &str) -> PyResult<Option<MetaValue>> {
    let class = value.get_type();
    if class.getattr("__module__")?.extract::<String>()? != "numpy" || !value.hasattr("dtype")? {
        return Ok(None);
    }
    let np = value.py().import("numpy")?;
    let array = np.call_method1("asarray", (value,))?;
    let shape: Vec<i64> = array.getattr("shape")?.extract()?;
    let dtype = array.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let name: String = dtype.getattr("name")?.extract()?;
    if let (true, Some(native)) = ("biuf".contains(kind.as_str()), DType::from_name(&name)) {
        // The wire is little-endian.
        let little = np.call_method1(
            "ascontiguousarray",
            (array, dtype.call_method1("newbyteorder", ("<",))?),
        )?;
        let bytes: Vec<u8> = little.call_method0("tobytes")?.extract()?;
        let tensor = rlmesh_spaces::Tensor::from_vec(bytes, shape, native)
            .map_err(|err| PyValueError::new_err(format!("metadata value at {path}: {err}")))?;
        return Ok(Some(MetaValue::Tensor(tensor)));
    }
    let items = array
        .call_method0("ravel")?
        .call_method0("tolist")?
        .cast::<PyList>()?
        .iter()
        .enumerate()
        .map(|(i, item)| py_any_to_meta_value(&item, &format!("{path}.flat[{i}]")))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Some(MetaValue::Array { shape, items }))
}

fn py_any_to_meta_value(value: &Bound<'_, PyAny>, path: &str) -> PyResult<MetaValue> {
    if let Some(array) = numpy_value(value, path)? {
        return Ok(array);
    }
    let normalized = normalize_metadata_value(value)?;

    if normalized.is_none() {
        return Ok(MetaValue::Null);
    }
    // An `rlmesh.Tensor` travels as an array, as a NumPy array does (COR-26,
    // COR-P10).
    if let Some(tensor) = extract_tensor(&normalized)? {
        let tensor = rlmesh::binding::tensor_from_core(tensor.inner.clone())
            .map_err(crate::types::to_py_err)?;
        return Ok(MetaValue::Tensor(
            rlmesh::binding::tensor_core(&tensor).clone(),
        ));
    }
    // `bytes` must be matched before the sequence and integer paths: it is
    // neither a list nor a scalar, and carries raw binary exactly.
    if let Ok(bytes) = normalized.cast::<PyBytes>() {
        return Ok(MetaValue::Bytes(bytes.as_bytes().to_vec()));
    }
    if let Ok(dict) = normalized.cast::<PyDict>() {
        return Ok(MetaValue::Map(fields(dict, path)?));
    }
    let items = |items: Vec<Bound<'_, PyAny>>| -> PyResult<MetaValue> {
        Ok(MetaValue::List(
            items
                .iter()
                .enumerate()
                .map(|(i, item)| py_any_to_meta_value(item, &format!("{path}[{i}]")))
                .collect::<PyResult<Vec<_>>>()?,
        ))
    };
    if let Ok(list) = normalized.cast::<PyList>() {
        return items(list.iter().collect());
    }
    if let Ok(tuple) = normalized.cast::<PyTuple>() {
        return items(tuple.iter().collect());
    }
    if let Ok(flag) = normalized.extract::<bool>() {
        return Ok(MetaValue::Bool(flag));
    }
    if let Ok(number) = normalized.extract::<i64>() {
        return Ok(MetaValue::Int(number));
    }
    // An integer that does not fit `i64` must not slide into the float branch:
    // that would silently round it (metadata `Int` is int64 on the wire).
    if normalized.is_instance_of::<PyInt>() {
        return Err(PyValueError::new_err(format!(
            "metadata integer {normalized} at {path} does not fit int64; RLMesh metadata \
             integers must be in [-2**63, 2**63) -- store a larger value as a string \
             instead"
        )));
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
            && let Ok(meta_value) = py_any_to_meta_value(&enum_value, path)
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
        "metadata value at {path} of type '{}' is not supported in RLMesh metadata; \
         pass the dict through rlmesh.sanitize_metadata(...) to stringify unsupported \
         values",
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
        // An array as a NumPy array of its dtype; shape `()` as a NumPy scalar.
        MetaValue::Tensor(tensor) => {
            let np = py.import("numpy")?;
            let bytes = PyBytes::new(py, &tensor.to_contiguous_bytes());
            let dtype = np.call_method1("dtype", (format!("<{}", numpy_code(tensor.dtype())),))?;
            let array = np
                .call_method1("frombuffer", (bytes, dtype))?
                .call_method1("reshape", (tensor.shape().to_vec(),))?
                .call_method0("copy")?;
            scalar_or_array(&array, tensor.shape())?
        }
        MetaValue::Array { shape, items } => {
            let np = py.import("numpy")?;
            let array = np.call_method1("empty", (items.len(), "O"))?;
            for (i, item) in items.iter().enumerate() {
                array.set_item(i, meta_value_to_py(py, item)?)?;
            }
            let array = array.call_method1("reshape", (shape.clone(),))?;
            scalar_or_array(&array, shape)?
        }
    })
}

fn scalar_or_array(array: &Bound<'_, PyAny>, shape: &[i64]) -> PyResult<Py<PyAny>> {
    Ok(if shape.is_empty() {
        array.get_item(())?.unbind()
    } else {
        array.clone().unbind()
    })
}

/// NumPy's type code for `dtype` (`i2`, `f4`, `?`).
fn numpy_code(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "?",
        DType::Uint8 => "u1",
        DType::Uint16 => "u2",
        DType::Uint32 => "u4",
        DType::Uint64 => "u8",
        DType::Int8 => "i1",
        DType::Int16 => "i2",
        DType::Int32 => "i4",
        DType::Int64 => "i8",
        DType::Float16 => "f2",
        DType::Float32 => "f4",
        DType::Float64 | DType::Unspecified => "f8",
    }
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

    #[test]
    fn meta_rejects_integers_wider_than_int64() {
        Python::attach(|py| {
            // 2^63 and -(2^63 + 1) are Python ints the wire's int64 cannot
            // carry; they must raise, not fall through to the float branch and
            // silently round.
            for source in [
                pyo3::ffi::c_str!("{'n': 2**63}"),
                pyo3::ffi::c_str!("{'n': -(2**63) - 1}"),
            ] {
                let value = py.eval(source, None, None).unwrap();
                let error = py_any_to_meta_map(&value).unwrap_err();
                assert!(error.is_instance_of::<PyValueError>(py));
                assert!(error.to_string().contains("at ['n'] does not fit int64"));
            }
            // The int64 edges still convert.
            let value = py
                .eval(
                    pyo3::ffi::c_str!("{'lo': -(2**63), 'hi': 2**63 - 1}"),
                    None,
                    None,
                )
                .unwrap();
            let native = py_any_to_meta_map(&value).unwrap();
            assert_eq!(native.get("lo"), Some(&MetaValue::Int(i64::MIN)));
            assert_eq!(native.get("hi"), Some(&MetaValue::Int(i64::MAX)));
        });
    }
}
