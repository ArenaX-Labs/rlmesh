use super::ValueBackend;
use crate::spaces::tensor::{extract_tensor, make_tensor};
use crate::spaces::utils::dtype_name;
use half::{bf16, f16};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString, PyTuple};
use rlmesh_spaces::{
    DType, Scalar,
    spaces::{SpaceKind, SpaceSpec},
};

pub(crate) fn encode_array_like_value_with_backend(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    space: &SpaceSpec,
    backend: ValueBackend,
) -> PyResult<Vec<u8>> {
    if backend.prefers_numpy(py)? {
        return encode_with_numpy(py, value, space);
    }
    encode_without_numpy(value, space)
}

pub(crate) fn decode_array_like_value_with_backend<'py>(
    py: Python<'py>,
    bytes: &[u8],
    space: &SpaceSpec,
    backend: ValueBackend,
) -> PyResult<Bound<'py, PyAny>> {
    if backend.prefers_numpy(py)? {
        return decode_with_numpy(py, bytes, space);
    }

    let base_shape = shape_to_usize(&space.shape)?;
    let base_numel = element_count(&base_shape);
    let dtype = resolve_dtype(space.dtype);
    let item_size = dtype_size(dtype);
    let item_count = bytes.len() / item_size;

    if item_count == base_numel {
        if matches!(space.spec.as_ref(), Some(SpaceKind::Discrete(_))) || base_shape.is_empty() {
            let scalars = decode_scalars(bytes, dtype)?;
            return scalar_to_bound(
                py,
                scalars.first().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("expected one decoded scalar")
                })?,
            );
        }
        return make_tensor(
            py,
            bytes.to_vec(),
            base_shape,
            dtype_name(space.dtype as i32),
        );
    }

    if base_numel > 0 && item_count.is_multiple_of(base_numel) {
        let batch_size = item_count / base_numel;
        let mut batch_shape = vec![batch_size];
        batch_shape.extend(base_shape);
        return make_tensor(
            py,
            bytes.to_vec(),
            batch_shape,
            dtype_name(space.dtype as i32),
        );
    }

    make_tensor(
        py,
        bytes.to_vec(),
        vec![item_count],
        dtype_name(space.dtype as i32),
    )
}

pub(crate) fn encode_i64_sequence_bytes(values: &[i64], dtype: DType) -> PyResult<Vec<u8>> {
    rlmesh_spaces::encode_i64_scalars(values, normalize_dtype(dtype))
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))
}

pub(crate) fn decode_i64_sequence_bytes(bytes: &[u8], dtype: DType) -> PyResult<Vec<i64>> {
    Ok(decode_scalars(bytes, dtype)?
        .into_iter()
        .map(Scalar::as_i64)
        .collect())
}

fn encode_with_numpy(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    space: &SpaceSpec,
) -> PyResult<Vec<u8>> {
    let numpy = py.import("numpy")?;
    let dtype_name = dtype_name(space.dtype as i32);

    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype_name)?;
    let array = numpy.getattr("asarray")?.call((value,), Some(&kwargs))?;

    let bytes_obj = array.call_method0("tobytes")?;
    bytes_obj.extract::<Vec<u8>>()
}

fn encode_without_numpy(value: &Bound<'_, PyAny>, space: &SpaceSpec) -> PyResult<Vec<u8>> {
    if let Some(tensor) = extract_tensor(value)? {
        let expected = normalize_dtype(resolve_dtype(space.dtype));
        let actual = normalize_dtype(tensor.inner.dtype());
        if actual != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "tensor dtype {} does not match space dtype {}",
                actual.name(),
                expected.name(),
            )));
        }

        // The bytes are reinterpreted against the space's element layout, so a
        // tensor that is neither a single sample nor a whole number of samples
        // would silently misdecode. Reject those instead of shipping garbage.
        let base_numel = element_count(&shape_to_usize(&space.shape)?);
        let tensor_numel = tensor.inner.numel();
        if base_numel == 0 {
            if tensor_numel != 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "tensor has {tensor_numel} elements but the space is empty",
                )));
            }
        } else if !tensor_numel.is_multiple_of(base_numel) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "tensor element count {tensor_numel} is not a multiple of the space \
                 element count {base_numel}",
            )));
        }

        return Ok(tensor.inner.to_contiguous_bytes().into_owned());
    }
    let mut flattened = Vec::new();
    flatten_scalars(value, &mut flattened)?;
    let dtype = resolve_dtype(space.dtype);
    let mut bytes = Vec::with_capacity(flattened.len() * dtype_size(dtype));
    for item in flattened {
        pack_scalar_bytes(item.bind(value.py()), dtype, &mut bytes)?;
    }
    Ok(bytes)
}

fn decode_with_numpy<'py>(
    py: Python<'py>,
    bytes: &[u8],
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let raw_array = numpy.call_method1("frombuffer", (bytes, dtype_name(space.dtype as i32)))?;

    let item_count: usize = raw_array.getattr("size")?.extract()?;
    let base_shape: Vec<i64> = space.shape.clone();
    // Mirror the native backend's element_count: an empty shape is a single
    // scalar, otherwise the product of the dims (which is 0 for a (0,) space).
    // Using .max(1) here missized zero-size spaces into a (0, 0) batch.
    let base_numel = if base_shape.is_empty() {
        1
    } else {
        base_shape.iter().product::<i64>() as usize
    };

    if item_count == base_numel {
        if matches!(space.spec.as_ref(), Some(SpaceKind::Discrete(_))) || base_shape.is_empty() {
            // Scalar () spaces decode to a Python scalar, matching the native
            // backend instead of leaving frombuffer's stray (1,) array.
            return raw_array.call_method0("item");
        }
        return raw_array.call_method1("reshape", (base_shape,));
    }

    if base_numel > 0 && item_count.is_multiple_of(base_numel) {
        let batch_size = item_count / base_numel;
        let mut batch_shape = vec![batch_size as i64];
        batch_shape.extend(base_shape);
        return raw_array.call_method1("reshape", (batch_shape,));
    }

    Ok(raw_array)
}

fn flatten_scalars(value: &Bound<'_, PyAny>, out: &mut Vec<Py<PyAny>>) -> PyResult<()> {
    if value.cast::<PyDict>().is_ok() {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "array-like values cannot be dicts",
        ));
    }

    if value.cast::<PyString>().is_ok() || value.cast::<PyBytes>().is_ok() {
        out.push(value.clone().unbind());
        return Ok(());
    }

    if value.hasattr("tolist")? {
        let normalized = value.call_method0("tolist")?;
        return flatten_scalars(&normalized, out);
    }

    if let Ok(list) = value.cast::<PyList>() {
        for item in list.iter() {
            flatten_scalars(&item, out)?;
        }
        return Ok(());
    }

    if let Ok(tuple) = value.cast::<PyTuple>() {
        for item in tuple.iter() {
            flatten_scalars(&item, out)?;
        }
        return Ok(());
    }

    if value.hasattr("__iter__")?
        && let Ok(iter) = value.try_iter()
    {
        let mut seen_any = false;
        for item in iter {
            seen_any = true;
            flatten_scalars(&item?, out)?;
        }
        if seen_any {
            return Ok(());
        }
    }

    out.push(value.clone().unbind());
    Ok(())
}

fn pack_scalar_bytes(value: &Bound<'_, PyAny>, dtype: DType, out: &mut Vec<u8>) -> PyResult<()> {
    match dtype {
        DType::Bool => {
            let flag = if let Ok(flag) = value.extract::<bool>() {
                flag
            } else if let Ok(number) = value.extract::<i64>() {
                number != 0
            } else {
                value.extract::<f64>()? != 0.0
            };
            out.push(if flag { 1 } else { 0 });
        }
        DType::Uint8 => out.push(value.extract::<u8>()?),
        DType::Int8 => out.extend(value.extract::<i8>()?.to_le_bytes()),
        DType::Int16 => out.extend(value.extract::<i16>()?.to_le_bytes()),
        DType::Int32 => out.extend((value.extract::<i64>()? as i32).to_le_bytes()),
        DType::Int64 => out.extend(value.extract::<i64>()?.to_le_bytes()),
        DType::Uint16 => out.extend(value.extract::<u16>()?.to_le_bytes()),
        DType::Uint32 => out.extend(value.extract::<u32>()?.to_le_bytes()),
        DType::Uint64 => out.extend(value.extract::<u64>()?.to_le_bytes()),
        DType::Float32 | DType::Unspecified => {
            out.extend((value.extract::<f64>()? as f32).to_le_bytes());
        }
        DType::Float64 => out.extend(value.extract::<f64>()?.to_le_bytes()),
        DType::Float16 => out.extend(f16::from_f32(value.extract::<f64>()? as f32).to_le_bytes()),
        DType::Bfloat16 => out.extend(bf16::from_f64(value.extract::<f64>()?).to_le_bytes()),
    }
    Ok(())
}

fn decode_scalars(bytes: &[u8], dtype: DType) -> PyResult<Vec<Scalar>> {
    rlmesh_spaces::decode_scalars(bytes, normalize_dtype(dtype))
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))
}

/// The extension treats `Unspecified` specs as float32 throughout.
fn normalize_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Unspecified => DType::Float32,
        other => other,
    }
}

fn resolve_dtype<T>(dtype: T) -> DType
where
    T: Into<i32>,
{
    DType::try_from(dtype.into()).unwrap_or(DType::Float32)
}

fn dtype_size(dtype: DType) -> usize {
    // Unspecified follows the extension-wide float32 fallback.
    match dtype {
        DType::Unspecified => 4,
        other => rlmesh_spaces::dtype_size(other),
    }
}

fn shape_to_usize(shape: &[i64]) -> PyResult<Vec<usize>> {
    shape
        .iter()
        .map(|dim| {
            usize::try_from(*dim).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!("negative shape dimension: {dim}"))
            })
        })
        .collect()
}

fn element_count(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().copied().product()
    }
}

fn scalar_to_bound<'py>(py: Python<'py>, scalar: &Scalar) -> PyResult<Bound<'py, PyAny>> {
    Ok(scalar_to_object(py, scalar)?.bind(py).clone())
}

fn scalar_to_object(py: Python<'_>, scalar: &Scalar) -> PyResult<Py<PyAny>> {
    match scalar {
        Scalar::Bool(flag) => Ok(PyBool::new(py, *flag).to_owned().into_any().unbind()),
        Scalar::Int(number) => Ok(number.into_pyobject(py)?.into_any().unbind()),
        Scalar::Float(number) => Ok(number.into_pyobject(py)?.into_any().unbind()),
    }
}

#[cfg(test)]
mod tests {
    use super::super::ValueBackend;
    use super::{decode_array_like_value_with_backend, encode_array_like_value_with_backend};
    use crate::spaces::tensor::wrap_native_tensor;
    use pyo3::Python;
    use pyo3::types::PyAnyMethods;
    use rlmesh_spaces::Tensor;
    use rlmesh_spaces::spaces::BoxSpaceBuilder;

    fn numpy_available(py: Python<'_>) -> bool {
        py.import("numpy").is_ok()
    }

    fn box_spec(shape: Vec<i64>) -> rlmesh_spaces::spaces::SpaceSpec {
        use rlmesh_spaces::spaces::{SpaceKind, SpaceSpec};
        use rlmesh_spaces::{BoxBounds, BoxSpec, DType, UniformBounds};
        SpaceSpec {
            shape,
            dtype: DType::Float32,
            spec: Some(SpaceKind::Box(BoxSpec {
                bounds: Some(BoxBounds::Uniform(UniformBounds {
                    low: -1.0,
                    high: 1.0,
                })),
            })),
        }
    }

    #[test]
    fn encode_without_numpy_rejects_dtype_mismatched_tensor() {
        Python::attach(|py| {
            let space = BoxSpaceBuilder::scalar(-10.0, 10.0, vec![3])
                .dtype(rlmesh_spaces::DType::Int32)
                .build()
                .unwrap();

            let tensor = Tensor::from_slice(
                &1.0f32.to_le_bytes().repeat(3),
                &[3],
                rlmesh_spaces::DType::Float32,
            )
            .unwrap();
            let value = wrap_native_tensor(py, tensor).unwrap();

            let err =
                encode_array_like_value_with_backend(py, &value, &space, ValueBackend::Native)
                    .unwrap_err();
            assert!(
                err.to_string().contains("does not match space dtype"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn encode_without_numpy_accepts_matching_tensor() {
        Python::attach(|py| {
            let space = BoxSpaceBuilder::scalar(-10.0, 10.0, vec![3])
                .dtype(rlmesh_spaces::DType::Int32)
                .build()
                .unwrap();
            let tensor = Tensor::from_slice(
                &1i32.to_le_bytes().repeat(3),
                &[3],
                rlmesh_spaces::DType::Int32,
            )
            .unwrap();
            let value = wrap_native_tensor(py, tensor).unwrap();

            let bytes =
                encode_array_like_value_with_backend(py, &value, &space, ValueBackend::Native)
                    .unwrap();
            assert_eq!(bytes.len(), 3 * 4);
        });
    }

    #[test]
    fn decode_with_numpy_scalar_box_returns_scalar() {
        Python::attach(|py| {
            if !numpy_available(py) {
                return;
            }
            let space = box_spec(Vec::<i64>::new());
            let bytes = 0.5f32.to_le_bytes();

            let decoded =
                decode_array_like_value_with_backend(py, &bytes, &space, ValueBackend::Auto)
                    .unwrap();
            // A scalar () space decodes to a Python float, not a (1,) array.
            assert!(
                !decoded.hasattr("shape").unwrap(),
                "expected a scalar, got an array-like: {decoded:?}"
            );
            assert!((decoded.extract::<f64>().unwrap() - 0.5).abs() < 1e-6);
        });
    }

    #[test]
    fn decode_with_numpy_zero_size_box_keeps_rank_one() {
        Python::attach(|py| {
            if !numpy_available(py) {
                return;
            }
            let space = box_spec(vec![0]);

            let decoded =
                decode_array_like_value_with_backend(py, &[], &space, ValueBackend::Auto).unwrap();
            let shape: Vec<i64> = decoded.getattr("shape").unwrap().extract().unwrap();
            assert_eq!(shape, vec![0]);
        });
    }
}
