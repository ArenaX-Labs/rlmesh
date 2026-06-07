use super::ValueBackend;
use crate::spaces::tensor::{extract_tensor, make_tensor};
use crate::spaces::utils::dtype_name;
use half::f16;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString, PyTuple};
use rlmesh_spaces::v1::{
    DType,
    spaces::{SpaceSpec, space_spec},
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
    let item_size = dtype_size(dtype)?;
    let item_count = bytes.len() / item_size;

    if item_count == base_numel {
        if matches!(space.spec.as_ref(), Some(space_spec::Spec::Discrete(_)))
            || base_shape.is_empty()
        {
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
    let mut bytes = Vec::with_capacity(values.len() * dtype_size(dtype)?);
    for value in values {
        match dtype {
            DType::Bool => bytes.push(u8::from(*value != 0)),
            DType::Uint8 => bytes.push(*value as u8),
            DType::Int32 => bytes.extend((*value as i32).to_le_bytes()),
            DType::Int64 => bytes.extend(value.to_le_bytes()),
            DType::Float32 | DType::Unspecified => {
                bytes.extend((*value as f32).to_le_bytes());
            }
            DType::Float64 => bytes.extend((*value as f64).to_le_bytes()),
            DType::Float16 => bytes.extend(f16::from_f32(*value as f32).to_le_bytes()),
        }
    }
    Ok(bytes)
}

pub(crate) fn decode_i64_sequence_bytes(bytes: &[u8], dtype: DType) -> PyResult<Vec<i64>> {
    decode_scalars(bytes, dtype)?
        .into_iter()
        .map(|value| match value {
            ScalarValue::Bool(flag) => Ok(i64::from(flag)),
            ScalarValue::Int(number) => Ok(number),
            ScalarValue::Float(number) => Ok(number as i64),
        })
        .collect()
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
        return Ok(tensor.data.clone());
    }
    let mut flattened = Vec::new();
    flatten_scalars(value, &mut flattened)?;
    let dtype = resolve_dtype(space.dtype);
    let mut bytes = Vec::with_capacity(flattened.len() * dtype_size(dtype)?);
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
    let base_numel = if base_shape.is_empty() {
        1
    } else {
        base_shape.iter().product::<i64>().max(1) as usize
    };

    if item_count == base_numel {
        if matches!(space.spec.as_ref(), Some(space_spec::Spec::Discrete(_))) {
            return raw_array.call_method0("item");
        }
        if base_shape.is_empty() {
            return Ok(raw_array);
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
        DType::Int32 => out.extend((value.extract::<i64>()? as i32).to_le_bytes()),
        DType::Int64 => out.extend(value.extract::<i64>()?.to_le_bytes()),
        DType::Float32 | DType::Unspecified => {
            out.extend((value.extract::<f64>()? as f32).to_le_bytes());
        }
        DType::Float64 => out.extend(value.extract::<f64>()?.to_le_bytes()),
        DType::Float16 => out.extend(f16::from_f32(value.extract::<f64>()? as f32).to_le_bytes()),
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum ScalarValue {
    Bool(bool),
    Int(i64),
    Float(f64),
}

fn decode_scalars(bytes: &[u8], dtype: DType) -> PyResult<Vec<ScalarValue>> {
    match dtype {
        DType::Bool => Ok(bytes
            .iter()
            .map(|value| ScalarValue::Bool(*value != 0))
            .collect()),
        DType::Uint8 => Ok(bytes
            .iter()
            .map(|value| ScalarValue::Int(*value as i64))
            .collect()),
        DType::Int32 => decode_chunks(bytes, 4, |chunk| {
            ScalarValue::Int(i32::from_le_bytes(chunk.try_into().expect("chunk")) as i64)
        }),
        DType::Int64 => decode_chunks(bytes, 8, |chunk| {
            ScalarValue::Int(i64::from_le_bytes(chunk.try_into().expect("chunk")))
        }),
        DType::Float32 | DType::Unspecified => decode_chunks(bytes, 4, |chunk| {
            ScalarValue::Float(f32::from_le_bytes(chunk.try_into().expect("chunk")) as f64)
        }),
        DType::Float64 => decode_chunks(bytes, 8, |chunk| {
            ScalarValue::Float(f64::from_le_bytes(chunk.try_into().expect("chunk")))
        }),
        DType::Float16 => decode_chunks(bytes, 2, |chunk| {
            ScalarValue::Float(f16::from_le_bytes(chunk.try_into().expect("chunk")).to_f64())
        }),
    }
}

fn decode_chunks<F>(bytes: &[u8], chunk_size: usize, f: F) -> PyResult<Vec<ScalarValue>>
where
    F: Fn(&[u8]) -> ScalarValue,
{
    if !bytes.len().is_multiple_of(chunk_size) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "byte length {} is not a multiple of element size {}",
            bytes.len(),
            chunk_size
        )));
    }

    Ok(bytes.chunks_exact(chunk_size).map(f).collect())
}

fn resolve_dtype<T>(dtype: T) -> DType
where
    T: Into<i32>,
{
    DType::try_from(dtype.into()).unwrap_or(DType::Float32)
}

fn dtype_size(dtype: DType) -> PyResult<usize> {
    match dtype {
        DType::Bool | DType::Uint8 => Ok(1),
        DType::Float16 => Ok(2),
        DType::Int32 | DType::Float32 | DType::Unspecified => Ok(4),
        DType::Int64 | DType::Float64 => Ok(8),
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

fn scalar_to_bound<'py>(py: Python<'py>, scalar: &ScalarValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(scalar_to_object(py, scalar)?.bind(py).clone())
}

fn scalar_to_object(py: Python<'_>, scalar: &ScalarValue) -> PyResult<Py<PyAny>> {
    match scalar {
        ScalarValue::Bool(flag) => Ok(PyBool::new(py, *flag).to_owned().into_any().unbind()),
        ScalarValue::Int(number) => Ok(number.into_pyobject(py)?.into_any().unbind()),
        ScalarValue::Float(number) => Ok(number.into_pyobject(py)?.into_any().unbind()),
    }
}
