use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rlmesh_spaces::v1::SpaceValue;
use rlmesh_spaces::v1::spaces::{SpaceSpec, space_spec};

use super::array_codec::encode_i64_sequence_bytes;
use crate::spaces::tensor::make_tensor;
use crate::spaces::utils::dtype_name;

pub(crate) fn space_value_to_py_neutral<'py>(
    py: Python<'py>,
    value: &SpaceValue,
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    match (space.spec.as_ref(), value) {
        (Some(space_spec::Spec::Box(_)), SpaceValue::Box(value)) => tensor_from_array_bytes(
            py,
            value.to_contiguous_bytes().into_owned(),
            value.shape().to_vec(),
            value.dtype(),
        ),
        (Some(space_spec::Spec::Discrete(_)), SpaceValue::Discrete(value)) => {
            Ok(value.into_pyobject(py)?.into_any())
        }
        (Some(space_spec::Spec::MultiBinary(_)), SpaceValue::MultiBinary(values)) => {
            let bytes = values
                .iter()
                .map(|value| u8::from(*value))
                .collect::<Vec<_>>();
            tensor_from_array_bytes(py, bytes, space.shape.clone(), space.dtype)
        }
        (Some(space_spec::Spec::MultiDiscrete(_)), SpaceValue::MultiDiscrete(values)) => {
            let bytes = encode_i64_sequence_bytes(values, space.dtype)?;
            tensor_from_array_bytes(py, bytes, space.shape.clone(), space.dtype)
        }
        (Some(space_spec::Spec::Text(_)), SpaceValue::Text(value)) => {
            Ok(value.into_pyobject(py)?.into_any())
        }
        (Some(space_spec::Spec::Dict(spec)), SpaceValue::Dict(values)) => {
            let dict = PyDict::new(py);
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                let child_value = values.get(key).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "missing RLMesh dict key '{key}'"
                    ))
                })?;
                dict.set_item(
                    key,
                    space_value_to_py_neutral(py, child_value, child_space)?,
                )?;
            }
            Ok(dict.into_any())
        }
        (Some(space_spec::Spec::Tuple(spec)), SpaceValue::Tuple(values)) => {
            if values.len() != spec.spaces.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "tuple arity mismatch: expected {}, got {}",
                    spec.spaces.len(),
                    values.len()
                )));
            }
            let items = values
                .iter()
                .zip(spec.spaces.iter())
                .map(|(value, child_space)| {
                    space_value_to_py_neutral(py, value, child_space).map(|value| value.unbind())
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, items)?.into_any())
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "space/value kind mismatch",
        )),
    }
}

pub(crate) fn batched_space_values_to_py_neutral<'py>(
    py: Python<'py>,
    values: &[SpaceValue],
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    match space.spec.as_ref() {
        Some(space_spec::Spec::Box(_)) => {
            let mut bytes = Vec::new();
            for value in values {
                match value {
                    SpaceValue::Box(value) => bytes.extend_from_slice(&value.to_contiguous_bytes()),
                    _ => {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "batched value kind mismatch for Box space",
                        ));
                    }
                }
            }
            let mut shape = vec![values.len()];
            shape.extend(space.shape.iter().map(|dim| *dim as usize));
            tensor_from_shape(py, bytes, shape, dtype_name(space.dtype))
        }
        Some(space_spec::Spec::Discrete(_)) => {
            let items = values
                .iter()
                .map(|value| match value {
                    SpaceValue::Discrete(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),
                    _ => Err(pyo3::exceptions::PyTypeError::new_err(
                        "batched value kind mismatch for Discrete space",
                    )),
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.into_any())
        }
        Some(space_spec::Spec::MultiBinary(_)) => {
            let mut bytes = Vec::new();
            for value in values {
                match value {
                    SpaceValue::MultiBinary(bits) => {
                        bytes.extend(bits.iter().map(|value| u8::from(*value)));
                    }
                    _ => {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "batched value kind mismatch for MultiBinary space",
                        ));
                    }
                }
            }
            let mut shape = vec![values.len()];
            shape.extend(space.shape.iter().map(|dim| *dim as usize));
            tensor_from_shape(py, bytes, shape, dtype_name(space.dtype))
        }
        Some(space_spec::Spec::MultiDiscrete(_)) => {
            let mut bytes = Vec::new();
            for value in values {
                match value {
                    SpaceValue::MultiDiscrete(items) => {
                        bytes.extend(encode_i64_sequence_bytes(items, space.dtype)?);
                    }
                    _ => {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "batched value kind mismatch for MultiDiscrete space",
                        ));
                    }
                }
            }
            let mut shape = vec![values.len()];
            shape.extend(space.shape.iter().map(|dim| *dim as usize));
            tensor_from_shape(py, bytes, shape, dtype_name(space.dtype))
        }
        Some(space_spec::Spec::Dict(spec)) => {
            let dict = PyDict::new(py);
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                let child_values = values
                    .iter()
                    .map(|value| match value {
                        SpaceValue::Dict(fields) => fields.get(key).cloned().ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(format!(
                                "missing RLMesh dict key '{key}'"
                            ))
                        }),
                        _ => Err(pyo3::exceptions::PyTypeError::new_err(
                            "batched value kind mismatch for Dict space",
                        )),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                dict.set_item(
                    key,
                    batched_space_values_to_py_neutral(py, &child_values, child_space)?,
                )?;
            }
            Ok(dict.into_any())
        }
        Some(space_spec::Spec::Tuple(spec)) => {
            let mut columns = vec![Vec::with_capacity(values.len()); spec.spaces.len()];
            for value in values {
                let items = match value {
                    SpaceValue::Tuple(items) => items,
                    _ => {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "batched value kind mismatch for Tuple space",
                        ));
                    }
                };
                if items.len() != spec.spaces.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "tuple arity mismatch: expected {}, got {}",
                        spec.spaces.len(),
                        items.len()
                    )));
                }
                for (column, item) in columns.iter_mut().zip(items.iter()) {
                    column.push(item.clone());
                }
            }
            let items = columns
                .iter()
                .zip(spec.spaces.iter())
                .map(|(child_values, child_space)| {
                    batched_space_values_to_py_neutral(py, child_values, child_space)
                        .map(|value| value.unbind())
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, items)?.into_any())
        }
        _ => {
            let items = values
                .iter()
                .map(|value| {
                    space_value_to_py_neutral(py, value, space).map(|value| value.unbind())
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.into_any())
        }
    }
}

pub(crate) fn tensor_from_array_bytes<'py>(
    py: Python<'py>,
    bytes: Vec<u8>,
    shape: Vec<i64>,
    dtype: impl Into<i32>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape
        .into_iter()
        .map(|dim| {
            usize::try_from(dim).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!("negative shape dimension: {dim}"))
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    tensor_from_shape(py, bytes, shape, dtype_name(dtype))
}

pub(crate) fn tensor_from_shape<'py>(
    py: Python<'py>,
    bytes: Vec<u8>,
    shape: Vec<usize>,
    dtype: impl Into<String>,
) -> PyResult<Bound<'py, PyAny>> {
    make_tensor(py, bytes, shape, dtype)
}
