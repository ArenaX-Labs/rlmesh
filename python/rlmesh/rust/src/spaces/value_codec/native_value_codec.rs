use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};
use rlmesh_spaces::spaces::{SpaceSpec, space_spec};
use rlmesh_spaces::{SpaceValue, Tensor, contains};

use super::ValueBackend;
use super::array_codec::{
    decode_array_like_value_with_backend, decode_i64_sequence_bytes,
    encode_array_like_value_with_backend, encode_i64_sequence_bytes,
};
use super::metadata::normalize_py_value;

pub(crate) fn space_value_to_py_with_backend<'py>(
    py: Python<'py>,
    value: &SpaceValue,
    space: &SpaceSpec,
    backend: ValueBackend,
) -> PyResult<Bound<'py, PyAny>> {
    match (space.spec.as_ref(), value) {
        (Some(space_spec::Spec::Box(_)), SpaceValue::Box(value)) => {
            decode_array_like_value_with_backend(py, &value.to_contiguous_bytes(), space, backend)
        }
        (Some(space_spec::Spec::Discrete(_)), SpaceValue::Discrete(value)) => {
            Ok(value.into_pyobject(py)?.into_any())
        }
        (Some(space_spec::Spec::MultiBinary(_)), SpaceValue::MultiBinary(values)) => {
            let bytes = values
                .iter()
                .map(|value| u8::from(*value))
                .collect::<Vec<_>>();
            decode_array_like_value_with_backend(py, &bytes, space, backend)
        }
        (Some(space_spec::Spec::MultiDiscrete(_)), SpaceValue::MultiDiscrete(values)) => {
            let bytes = encode_i64_sequence_bytes(values, space.dtype)?;
            decode_array_like_value_with_backend(py, &bytes, space, backend)
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
                    space_value_to_py_with_backend(py, child_value, child_space, backend)?,
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
            let values = values
                .iter()
                .zip(spec.spaces.iter())
                .map(|(value, child_space)| {
                    space_value_to_py_with_backend(py, value, child_space, backend)
                        .map(|value| value.unbind())
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, values)?.into_any())
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "space/value kind mismatch",
        )),
    }
}

pub(crate) fn py_any_to_space_value_with_backend(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    space: &SpaceSpec,
    backend: ValueBackend,
) -> PyResult<SpaceValue> {
    let encoded = py_any_to_space_value_unchecked(py, value, space, backend)?;
    contains(space, &encoded)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    Ok(encoded)
}

fn py_any_to_space_value_unchecked(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    space: &SpaceSpec,
    backend: ValueBackend,
) -> PyResult<SpaceValue> {
    Ok(match space.spec.as_ref() {
        Some(space_spec::Spec::Box(_)) => SpaceValue::Box(
            Tensor::from_vec(
                encode_array_like_value_with_backend(py, value, space, backend)?,
                space.shape.clone(),
                space.dtype,
            )
            .map_err(|err| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid box value: {err}"))
            })?,
        ),
        Some(space_spec::Spec::Discrete(_)) => {
            let normalized = normalize_py_value(value)?;
            let value = if let Ok(flag) = normalized.extract::<bool>() {
                i64::from(flag)
            } else if let Ok(number) = normalized.extract::<i64>() {
                number
            } else {
                normalized.extract::<f64>()? as i64
            };
            SpaceValue::Discrete(value)
        }
        Some(space_spec::Spec::MultiBinary(_)) => {
            let bytes = encode_array_like_value_with_backend(py, value, space, backend)?;
            SpaceValue::MultiBinary(bytes.into_iter().map(|value| value != 0).collect())
        }
        Some(space_spec::Spec::MultiDiscrete(_)) => {
            let bytes = encode_array_like_value_with_backend(py, value, space, backend)?;
            SpaceValue::MultiDiscrete(decode_i64_sequence_bytes(&bytes, space.dtype)?)
        }
        Some(space_spec::Spec::Text(_)) => {
            SpaceValue::Text(normalize_py_value(value)?.extract::<String>()?)
        }
        Some(space_spec::Spec::Dict(spec)) => {
            let normalized = normalize_py_value(value)?;
            let dict = normalized.cast::<PyDict>()?;
            let mut values = BTreeMap::new();
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                let child = dict.get_item(key)?.ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!("missing dict key '{key}'"))
                })?;
                values.insert(
                    key.clone(),
                    py_any_to_space_value_unchecked(py, &child, child_space, backend)?,
                );
            }
            SpaceValue::Dict(values)
        }
        Some(space_spec::Spec::Tuple(spec)) => {
            let items = if let Ok(tuple) = value.cast::<PyTuple>() {
                tuple.iter().collect::<Vec<_>>()
            } else if let Ok(list) = value.cast::<PyList>() {
                list.iter().collect::<Vec<_>>()
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Tuple space values must be a tuple or list",
                ));
            };
            if items.len() != spec.spaces.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Tuple space expected {} items, got {}",
                    spec.spaces.len(),
                    items.len()
                )));
            }
            let values = items
                .iter()
                .zip(spec.spaces.iter())
                .map(|(item, child_space)| {
                    py_any_to_space_value_unchecked(py, item, child_space, backend)
                })
                .collect::<PyResult<Vec<_>>>()?;
            SpaceValue::Tuple(values)
        }
        None => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "space spec is missing",
            ));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::super::{
        batched_space_values_to_py_with_backend, meta_map_to_pydict,
        py_any_to_batched_space_values_with_backend, py_any_to_meta_map,
    };
    use super::{py_any_to_space_value_with_backend, space_value_to_py_with_backend};
    use crate::spaces::ValueBackend;
    use pyo3::Python;
    use pyo3::types::{PyAnyMethods, PyDictMethods};
    use rlmesh_spaces::MetaValue;
    use rlmesh_spaces::spaces::{DictSpaceBuilder, DiscreteBuilder, TextBuilder};

    #[test]
    fn metadata_roundtrips_without_protobuf() {
        Python::attach(|py| {
            let metadata = py
                .eval(
                    pyo3::ffi::c_str!("{'seed': 7, 'nested': {'ok': True}, 'values': [1, 2, 3]}"),
                    None,
                    None,
                )
                .unwrap();
            let native = py_any_to_meta_map(&metadata).unwrap();
            let roundtrip = meta_map_to_pydict(py, &native).unwrap();
            assert_eq!(
                roundtrip
                    .get_item("seed")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                7
            );
            let nested = roundtrip.get_item("nested").unwrap().unwrap();
            assert!(nested.get_item("ok").unwrap().extract::<bool>().unwrap());
        });
    }

    #[test]
    fn metadata_accepts_enum_like_values() {
        Python::attach(|py| {
            let locals = pyo3::types::PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
class AutoresetMode:
    value = "next_step"
    name = "NEXT_STEP"
"#
                ),
                None,
                Some(&locals),
            )
            .unwrap();
            let metadata = py
                .eval(
                    pyo3::ffi::c_str!(r#"{"autoreset_mode": AutoresetMode()}"#),
                    None,
                    Some(&locals),
                )
                .unwrap();

            let native = py_any_to_meta_map(&metadata).unwrap();
            assert_eq!(
                native.get("autoreset_mode"),
                Some(&MetaValue::String("next_step".to_string()))
            );
        });
    }

    #[test]
    fn dict_space_roundtrips_without_protobuf() {
        Python::attach(|py| {
            let space = DictSpaceBuilder::new()
                .insert("choice", DiscreteBuilder::new(3).build().unwrap())
                .build()
                .unwrap();
            let value = py
                .eval(pyo3::ffi::c_str!("{'choice': 2}"), None, None)
                .unwrap();

            let native =
                py_any_to_space_value_with_backend(py, &value, &space, ValueBackend::Native)
                    .unwrap();
            let roundtrip =
                space_value_to_py_with_backend(py, &native, &space, ValueBackend::Native).unwrap();

            assert_eq!(
                roundtrip
                    .get_item("choice")
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                2
            );
        });
    }

    #[test]
    fn nested_space_validation_reports_full_path() {
        Python::attach(|py| {
            let space = DictSpaceBuilder::new()
                .insert(
                    "instruction",
                    TextBuilder::new(16).charset("abc").build().unwrap(),
                )
                .build()
                .unwrap();
            let value = py
                .eval(pyo3::ffi::c_str!("{'instruction': 'a b'}"), None, None)
                .unwrap();

            let err = py_any_to_space_value_with_backend(py, &value, &space, ValueBackend::Native)
                .unwrap_err();

            assert!(err.to_string().contains("$.instruction"));
            assert!(err.to_string().contains("character ' ' not in charset"));
        });
    }

    #[test]
    fn batched_dict_space_roundtrips_without_protobuf() {
        Python::attach(|py| {
            let space = DictSpaceBuilder::new()
                .insert("choice", DiscreteBuilder::new(3).build().unwrap())
                .build()
                .unwrap();
            let value = py
                .eval(pyo3::ffi::c_str!("{'choice': [0, 2]}"), None, None)
                .unwrap();

            let native = py_any_to_batched_space_values_with_backend(
                py,
                &value,
                &space,
                2,
                ValueBackend::Native,
            )
            .unwrap();
            let roundtrip =
                batched_space_values_to_py_with_backend(py, &native, &space, ValueBackend::Native)
                    .unwrap();

            let choices = roundtrip.get_item("choice").unwrap();
            assert_eq!(choices.get_item(0).unwrap().extract::<i64>().unwrap(), 0);
            assert_eq!(choices.get_item(1).unwrap().extract::<i64>().unwrap(), 2);
        });
    }
}
