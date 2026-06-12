use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};
use rlmesh_spaces::SpaceValue;
use rlmesh_spaces::spaces::{SpaceKind, SpaceSpec};

use super::ValueBackend;
use super::metadata::normalize_py_value;
use super::native_value_codec::{
    py_any_to_space_value_with_backend, space_value_to_py_with_backend,
};

pub(crate) fn batched_space_values_to_py_with_backend<'py>(
    py: Python<'py>,
    values: &[SpaceValue],
    space: &SpaceSpec,
    backend: ValueBackend,
) -> PyResult<Bound<'py, PyAny>> {
    match space.spec.as_ref() {
        Some(SpaceKind::Dict(spec)) => {
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
                    batched_space_values_to_py_with_backend(
                        py,
                        &child_values,
                        child_space,
                        backend,
                    )?,
                )?;
            }
            Ok(dict.into_any())
        }
        Some(SpaceKind::Tuple(spec)) => {
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
                    batched_space_values_to_py_with_backend(py, child_values, child_space, backend)
                        .map(|value| value.unbind())
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, items)?.into_any())
        }
        _ => batched_values_to_py(py, values, space, backend),
    }
}

pub(crate) fn py_any_to_batched_space_values_with_backend(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    space: &SpaceSpec,
    num_envs: usize,
    backend: ValueBackend,
) -> PyResult<Vec<SpaceValue>> {
    if num_envs == 0 {
        return Ok(vec![]);
    }

    match space.spec.as_ref() {
        Some(SpaceKind::Dict(spec)) => {
            let normalized = normalize_py_value(value)?;
            let dict = normalized.cast::<PyDict>()?;
            let child_batches = spec
                .keys
                .iter()
                .zip(spec.spaces.iter())
                .map(|(key, child_space)| {
                    let child = dict.get_item(key)?.ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!("missing dict key '{key}'"))
                    })?;
                    py_any_to_batched_space_values_with_backend(
                        py,
                        &child,
                        child_space,
                        num_envs,
                        backend,
                    )
                })
                .collect::<PyResult<Vec<_>>>()?;

            let mut values = vec![BTreeMap::new(); num_envs];
            for ((key, _), batch) in spec.keys.iter().zip(spec.spaces.iter()).zip(child_batches) {
                if batch.len() != num_envs {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "dict batch for key '{key}' expected {num_envs} values, got {}",
                        batch.len()
                    )));
                }
                for (index, child_value) in batch.into_iter().enumerate() {
                    values[index].insert(key.clone(), child_value);
                }
            }

            Ok(values.into_iter().map(SpaceValue::Dict).collect())
        }
        Some(SpaceKind::Tuple(spec)) => {
            let items = if let Ok(tuple) = value.cast::<PyTuple>() {
                tuple.iter().collect::<Vec<_>>()
            } else if let Ok(list) = value.cast::<PyList>() {
                list.iter().collect::<Vec<_>>()
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Tuple space batched values must be a tuple or list",
                ));
            };
            if items.len() != spec.spaces.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Tuple space expected {} items, got {}",
                    spec.spaces.len(),
                    items.len()
                )));
            }

            let child_batches = items
                .iter()
                .zip(spec.spaces.iter())
                .map(|(item, child_space)| {
                    py_any_to_batched_space_values_with_backend(
                        py,
                        item,
                        child_space,
                        num_envs,
                        backend,
                    )
                })
                .collect::<PyResult<Vec<_>>>()?;

            let mut values = vec![Vec::with_capacity(spec.spaces.len()); num_envs];
            for batch in child_batches {
                if batch.len() != num_envs {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "tuple batch expected {num_envs} values, got {}",
                        batch.len()
                    )));
                }
                for (index, child_value) in batch.into_iter().enumerate() {
                    values[index].push(child_value);
                }
            }

            Ok(values.into_iter().map(SpaceValue::Tuple).collect())
        }
        _ => batched_items(value, num_envs)?
            .iter()
            .map(|item| py_any_to_space_value_with_backend(py, item, space, backend))
            .collect(),
    }
}

fn batched_values_to_py<'py>(
    py: Python<'py>,
    values: &[SpaceValue],
    child_space: &SpaceSpec,
    backend: ValueBackend,
) -> PyResult<Bound<'py, PyAny>> {
    let items = values
        .iter()
        .map(|value| {
            space_value_to_py_with_backend(py, value, child_space, backend)
                .map(|value| value.unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;

    match child_space.spec.as_ref() {
        Some(SpaceKind::Box(_))
        | Some(SpaceKind::Discrete(_))
        | Some(SpaceKind::MultiBinary(_))
        | Some(SpaceKind::MultiDiscrete(_)) => {
            let items = PyList::new(py, items)?;
            if backend.prefers_numpy(py)? {
                // Propagate any np.array failure (e.g. inhomogeneous per-sample
                // shapes from a malformed batch) instead of silently flipping
                // the return type to a list of arrays.
                let numpy = py.import("numpy")?;
                return numpy.getattr("array")?.call1((items,));
            }
            Ok(items.into_any())
        }
        _ => Ok(PyList::new(py, items)?.into_any()),
    }
}

fn batched_items<'py>(
    value: &Bound<'py, PyAny>,
    num_envs: usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let normalized = normalize_py_value(value)?;
    let items = normalized.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    if items.len() != num_envs {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected {num_envs} batched values, got {}",
            items.len()
        )));
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::batched_space_values_to_py_with_backend;
    use crate::spaces::ValueBackend;
    use pyo3::Python;
    use rlmesh_spaces::spaces::BoxSpaceBuilder;
    use rlmesh_spaces::{DType, SpaceValue, Tensor};

    #[test]
    fn batched_box_decode_propagates_inhomogeneous_shape_error() {
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                return;
            }
            let space = BoxSpaceBuilder::scalar(-10.0, 10.0, vec![2])
                .dtype(DType::Float32)
                .build()
                .unwrap();

            // A malformed batch: one sample carries a differently-shaped Box
            // payload, so stacking with np.array raises on numpy >= 1.24.
            let values = vec![
                SpaceValue::Box(Tensor::from_slice(&[0u8; 8], &[2], DType::Float32).unwrap()),
                SpaceValue::Box(Tensor::from_slice(&[0u8; 12], &[3], DType::Float32).unwrap()),
            ];

            // Auto backend prefers numpy; the error must surface rather than
            // silently degrading to a list of arrays.
            let result =
                batched_space_values_to_py_with_backend(py, &values, &space, ValueBackend::Auto);
            assert!(result.is_err(), "expected np.array stacking to error");
        });
    }
}
