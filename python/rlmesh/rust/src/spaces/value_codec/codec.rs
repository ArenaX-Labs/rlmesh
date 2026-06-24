use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString, PyTuple};
use rlmesh_spaces::spaces::{SpaceKind, SpaceSpec};
use rlmesh_spaces::{Conformance, DType, Scalar, SpaceValue, Tensor, conform};

use super::ValueBackend;
use super::metadata::normalize_py_value;
use crate::spaces::tensor::{extract_tensor, make_tensor, wrap_native_tensor};
use crate::spaces::utils::dtype_name;

pub(crate) fn space_value_to_py_with_backend<'py>(
    py: Python<'py>,
    value: &SpaceValue,
    space: &SpaceSpec,
    backend: ValueBackend,
) -> PyResult<Bound<'py, PyAny>> {
    space_value_to_py(py, value, space, &move |py, value, space| {
        array_leaf_to_py_with_backend(py, value, space, backend)
    })
}

pub(crate) fn space_value_to_py_neutral<'py>(
    py: Python<'py>,
    value: &SpaceValue,
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    space_value_to_py(py, value, space, &array_leaf_to_py_neutral)
}

// Encodes the array-like leaves (Box, MultiBinary, MultiDiscrete); only the leaf
// behavior differs between the backend-aware and neutral (always-native) paths.
type LeafEncoder<'py> = dyn Fn(Python<'py>, &SpaceValue, &SpaceSpec) -> PyResult<Bound<'py, PyAny>>;

// Shared scalar/composite dispatch parameterized by the leaf encoder.
fn space_value_to_py<'py>(
    py: Python<'py>,
    value: &SpaceValue,
    space: &SpaceSpec,
    leaf: &LeafEncoder<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    match (space.spec.as_ref(), value) {
        (Some(SpaceKind::Box(_)), SpaceValue::Box(_))
        | (Some(SpaceKind::MultiBinary(_)), SpaceValue::MultiBinary(_))
        | (Some(SpaceKind::MultiDiscrete(_)), SpaceValue::MultiDiscrete(_)) => {
            leaf(py, value, space)
        }
        (Some(SpaceKind::Discrete(_)), SpaceValue::Discrete(value)) => {
            Ok(value.into_pyobject(py)?.into_any())
        }
        (Some(SpaceKind::Text(_)), SpaceValue::Text(value)) => {
            Ok(value.into_pyobject(py)?.into_any())
        }
        (Some(SpaceKind::Dict(spec)), SpaceValue::Dict(values)) => {
            let dict = PyDict::new(py);
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                let child_value = values.get(key).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "missing RLMesh dict key '{key}'"
                    ))
                })?;
                dict.set_item(key, space_value_to_py(py, child_value, child_space, leaf)?)?;
            }
            Ok(dict.into_any())
        }
        (Some(SpaceKind::Tuple(spec)), SpaceValue::Tuple(values)) => {
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
                    space_value_to_py(py, value, child_space, leaf).map(|value| value.unbind())
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, items)?.into_any())
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "space/value kind mismatch",
        )),
    }
}

fn array_leaf_to_py_with_backend<'py>(
    py: Python<'py>,
    value: &SpaceValue,
    space: &SpaceSpec,
    backend: ValueBackend,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        SpaceValue::Box(value) => {
            decode_array_like_value_with_backend(py, &value.to_contiguous_bytes(), space, backend)
        }
        SpaceValue::MultiBinary(values) => {
            let bytes = values
                .iter()
                .map(|value| u8::from(*value))
                .collect::<Vec<_>>();
            decode_array_like_value_with_backend(py, &bytes, space, backend)
        }
        SpaceValue::MultiDiscrete(values) => {
            let bytes = encode_i64_sequence_bytes(values, space.dtype)?;
            decode_array_like_value_with_backend(py, &bytes, space, backend)
        }
        _ => unreachable!("array_leaf only dispatched for array-like kinds"),
    }
}

fn array_leaf_to_py_neutral<'py>(
    py: Python<'py>,
    value: &SpaceValue,
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        // Hand the native tensor over directly: shares the (aligned) wire
        // storage instead of copying into a fresh unaligned buffer.
        SpaceValue::Box(value) => wrap_native_tensor(py, value.clone()),
        SpaceValue::MultiBinary(values) => {
            let bytes = values
                .iter()
                .map(|value| u8::from(*value))
                .collect::<Vec<_>>();
            tensor_from_array_bytes(py, bytes, space.shape.clone(), space.dtype)
        }
        SpaceValue::MultiDiscrete(values) => {
            let bytes = encode_i64_sequence_bytes(values, space.dtype)?;
            tensor_from_array_bytes(py, bytes, space.shape.clone(), space.dtype)
        }
        _ => unreachable!("array_leaf only dispatched for array-like kinds"),
    }
}

pub(crate) fn py_any_to_space_value_with_backend(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    space: &SpaceSpec,
    backend: ValueBackend,
) -> PyResult<SpaceValue> {
    let encoded = py_any_to_space_value_unchecked(py, value, space, backend)?;
    // Structural deviations (wrong shape/dtype/arity/domain, NaN, a missing key)
    // are rejected at encode regardless of policy. Range deviations (Box bounds,
    // Text charset/length) pass through so the serving side can apply its
    // validation policy (warn by default); see the env server's enforcement.
    if let Conformance::Structural(err) = conform(space, &encoded) {
        return Err(pyo3::exceptions::PyValueError::new_err(err.to_string()));
    }
    Ok(encoded)
}

fn py_any_to_space_value_unchecked(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    space: &SpaceSpec,
    backend: ValueBackend,
) -> PyResult<SpaceValue> {
    Ok(match space.spec.as_ref() {
        Some(SpaceKind::Box(_)) => SpaceValue::Box(
            Tensor::from_vec(
                encode_array_like_value_with_backend(py, value, space, backend)?,
                space.shape.clone(),
                space.dtype,
            )
            .map_err(|err| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid box value: {err}"))
            })?,
        ),
        Some(SpaceKind::Discrete(_)) => {
            let normalized = normalize_py_value(value)?;
            let value = if let Ok(flag) = normalized.extract::<bool>() {
                i64::from(flag)
            } else if let Ok(number) = normalized.extract::<i64>() {
                number
            } else {
                // Reject non-integer floats instead of truncating toward zero,
                // matching gymnasium's Discrete.contains.
                let number = normalized.extract::<f64>()?;
                if !number.is_finite() || number.fract() != 0.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Discrete value must be an integer, got {number}"
                    )));
                }
                number as i64
            };
            SpaceValue::Discrete(value)
        }
        Some(SpaceKind::MultiBinary(_)) => {
            let bytes = encode_array_like_value_with_backend(py, value, space, backend)?;
            SpaceValue::MultiBinary(bytes.into_iter().map(|value| value != 0).collect())
        }
        Some(SpaceKind::MultiDiscrete(_)) => {
            let bytes = encode_array_like_value_with_backend(py, value, space, backend)?;
            SpaceValue::MultiDiscrete(decode_i64_sequence_bytes(&bytes, space.dtype)?)
        }
        Some(SpaceKind::Text(_)) => {
            SpaceValue::Text(normalize_py_value(value)?.extract::<String>()?)
        }
        Some(SpaceKind::Dict(spec)) => {
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
        Some(SpaceKind::Tuple(spec)) => {
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

pub(crate) fn encode_array_like_value_with_backend(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    space: &SpaceSpec,
    backend: ValueBackend,
) -> PyResult<Vec<u8>> {
    // A native rlmesh tensor is validated and encoded identically on either
    // backend; only non-tensor inputs (lists, numpy arrays) take the backend path.
    if let Some(bytes) = encode_native_tensor(value, space)? {
        return Ok(bytes);
    }
    if backend.prefers_numpy(py)? {
        return encode_with_numpy(py, value, space);
    }
    encode_scalars(value, space)
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
        if base_shape.is_empty() {
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

    // A float supplied for an integer dtype coerces only when every element is
    // finite, exactly integral, and within the target dtype's range; otherwise
    // reject rather than silently truncating or overflow-wrapping, which would
    // corrupt the value. (NumPy only warns on out-of-range integer casts.)
    let source = numpy.getattr("asarray")?.call1((value,))?;
    let kind = source
        .getattr("dtype")?
        .getattr("kind")?
        .extract::<String>()?;
    // An integer dtype must reject any out-of-range element on encode, mirroring
    // the native `check_int_in_dtype_range` guard — otherwise the `asarray` cast
    // below silently overflow-wraps (e.g. 300 -> 44 for uint8) and the numpy
    // backend disagrees byte-for-byte with the native path. We check both float
    // sources (which also must be exactly integral) and integer/bool sources.
    if resolve_dtype(space.dtype).is_integer() && matches!(kind.as_str(), "f" | "i" | "u" | "b") {
        if kind == "f" {
            let rint = numpy.getattr("rint")?.call1((&source,))?;
            let integral = source.call_method1("__eq__", (&rint,))?;
            let finite = numpy.getattr("isfinite")?.call1((&source,))?;
            let clean = integral.call_method1("__and__", (finite,))?;
            if !numpy.getattr("all")?.call1((&clean,))?.is_truthy()? {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "float value supplied for integer dtype {dtype_name} is not integral"
                )));
            }
        }

        let info = numpy.getattr("iinfo")?.call1((dtype_name,))?;
        // Use a strict `< max+1`, with max+1 computed in Python's arbitrary-
        // precision int (`iinfo.max` is a Python int, not a numpy scalar that
        // would wrap). iinfo.max such as 2**63-1 is not representable in f64 and
        // rounds *up* to 2**63, so `<= max` would wrongly accept 2**63 and the
        // int cast below would then overflow-wrap. max+1 is an exact power-of-two
        // float, so `< max+1` is an exact boundary.
        let max_exclusive = info.getattr("max")?.call_method1("__add__", (1,))?;
        let in_range = source
            .call_method1("__ge__", (info.getattr("min")?,))?
            .call_method1(
                "__and__",
                (source.call_method1("__lt__", (max_exclusive,))?,),
            )?;
        if !numpy.getattr("all")?.call1((in_range,))?.is_truthy()? {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "value supplied for integer dtype {dtype_name} is not representable in its range"
            )));
        }
    }

    // Reuse the already-materialized `source` so the data is parsed once.
    let array = numpy.getattr("asarray")?.call1((&source, dtype_name))?;

    let bytes_obj = array.call_method0("tobytes")?;
    bytes_obj.extract::<Vec<u8>>()
}

// Validate + encode a native rlmesh tensor; returns None when `value` is not a
// native tensor so the caller can fall back to a backend-specific encoder.
fn encode_native_tensor(value: &Bound<'_, PyAny>, space: &SpaceSpec) -> PyResult<Option<Vec<u8>>> {
    let Some(tensor) = extract_tensor(value)? else {
        return Ok(None);
    };
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

    Ok(Some(tensor.inner.to_contiguous_bytes().into_owned()))
}

fn encode_scalars(value: &Bound<'_, PyAny>, space: &SpaceSpec) -> PyResult<Vec<u8>> {
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
        if base_shape.is_empty() {
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
        DType::Int32 => out.extend(value.extract::<i32>()?.to_le_bytes()),
        DType::Int64 => out.extend(value.extract::<i64>()?.to_le_bytes()),
        DType::Uint16 => out.extend(value.extract::<u16>()?.to_le_bytes()),
        DType::Uint32 => out.extend(value.extract::<u32>()?.to_le_bytes()),
        DType::Uint64 => out.extend(value.extract::<u64>()?.to_le_bytes()),
        DType::Float32 | DType::Unspecified => {
            out.extend((value.extract::<f64>()? as f32).to_le_bytes());
        }
        DType::Float64 => out.extend(value.extract::<f64>()?.to_le_bytes()),
        // Single rounding f64 -> f16 via the native codec's portable conversion,
        // matching numpy. `half::f16::from_f64` cannot be used: on x86 it rounds
        // through f32 (F16C `f as f32`, and the software fallback truncates the
        // low 32 mantissa bits), double-rounding borderline values; only aarch64
        // hardware is correct. `f64_to_f16_bits` rounds directly on every arch.
        DType::Float16 => {
            out.extend(rlmesh_spaces::f64_to_f16_bits(value.extract::<f64>()?).to_le_bytes())
        }
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
    fn f16_pack_single_rounds_not_double_rounds() {
        // value-encoding-v1 sentinel (mirrors scalar.rs `value_encoding_v1_float_golden`):
        // 1.0 + 2^-11 + 2^-25 packs to f16 0x3C01 with single f64->f16 rounding;
        // the old `from_f32(x as f32)` double-rounded it to 0x3C00.
        Python::attach(|py| {
            let dr = 1.0_f64 + 1.0 / 2048.0 + 1.0 / 33_554_432.0;
            let value = pyo3::types::PyFloat::new(py, dr);
            let mut out = Vec::new();
            super::pack_scalar_bytes(value.as_any(), rlmesh_spaces::DType::Float16, &mut out)
                .unwrap();
            assert_eq!(
                out,
                vec![0x01, 0x3C],
                "f16 must single-round, got {out:02x?}"
            );
        });
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
    fn encode_rejects_dtype_mismatched_tensor_on_numpy_backend() {
        Python::attach(|py| {
            if !numpy_available(py) {
                return;
            }
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

            // A native tensor is validated ahead of the backend split, so the
            // numpy path rejects the dtype mismatch instead of silently coercing.
            let err = encode_array_like_value_with_backend(py, &value, &space, ValueBackend::Auto)
                .unwrap_err();
            assert!(
                err.to_string().contains("does not match space dtype"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn encode_rejects_non_integral_float_for_integer_dtype() {
        Python::attach(|py| {
            if !numpy_available(py) {
                return;
            }
            let numpy = py.import("numpy").unwrap();
            let space = BoxSpaceBuilder::scalar(-10.0, 10.0, vec![2])
                .dtype(rlmesh_spaces::DType::Int32)
                .build()
                .unwrap();

            // Integral floats coerce cleanly to the integer dtype.
            let integral = numpy
                .getattr("asarray")
                .unwrap()
                .call1((vec![1.0f64, 2.0],))
                .unwrap();
            assert!(
                encode_array_like_value_with_backend(py, &integral, &space, ValueBackend::Auto)
                    .is_ok()
            );

            // A non-integral float is rejected rather than silently truncated.
            let fractional = numpy
                .getattr("asarray")
                .unwrap()
                .call1((vec![1.5f64, 2.0],))
                .unwrap();
            let err =
                encode_array_like_value_with_backend(py, &fractional, &space, ValueBackend::Auto)
                    .unwrap_err();
            assert!(
                err.to_string().contains("not integral"),
                "unexpected error: {err}"
            );

            // An integral but out-of-range float is rejected rather than
            // silently overflow-wrapping to a garbage in-type value.
            let out_of_range = numpy
                .getattr("asarray")
                .unwrap()
                .call1((vec![3e9f64, 2.0],))
                .unwrap();
            let err =
                encode_array_like_value_with_backend(py, &out_of_range, &space, ValueBackend::Auto)
                    .unwrap_err();
            assert!(
                err.to_string().contains("not representable"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn encode_rejects_out_of_range_integer_array() {
        Python::attach(|py| {
            if !numpy_available(py) {
                return;
            }
            let numpy = py.import("numpy").unwrap();
            let space = BoxSpaceBuilder::scalar(0.0, 255.0, vec![2])
                .dtype(rlmesh_spaces::DType::Uint8)
                .build()
                .unwrap();

            // An int64 array whose element exceeds uint8 must be rejected, not
            // silently overflow-wrapped (300 -> 44) the way numpy's cast does —
            // matching the native `check_int_in_dtype_range` encode guard.
            let over = numpy
                .getattr("asarray")
                .unwrap()
                .call1((vec![300i64, 0],))
                .unwrap();
            let err = encode_array_like_value_with_backend(py, &over, &space, ValueBackend::Auto)
                .unwrap_err();
            assert!(
                err.to_string().contains("not representable"),
                "unexpected error: {err}"
            );

            // In-range integers still encode.
            let within = numpy
                .getattr("asarray")
                .unwrap()
                .call1((vec![255i64, 0],))
                .unwrap();
            assert!(
                encode_array_like_value_with_backend(py, &within, &space, ValueBackend::Auto)
                    .is_ok()
            );
        });
    }

    #[test]
    fn encode_rejects_float_at_int64_max_boundary() {
        Python::attach(|py| {
            if !numpy_available(py) {
                return;
            }
            let numpy = py.import("numpy").unwrap();
            // Wide float bounds so only the dtype-range check (not the box
            // bounds) decides; Int64::MAX is not representable in f64.
            let space = BoxSpaceBuilder::scalar(-1e19, 1e19, vec![1])
                .dtype(rlmesh_spaces::DType::Int64)
                .build()
                .unwrap();

            // 2**63 is an exact f64 but one past i64::MAX (2**63 - 1). i64::MAX
            // rounds *up* to 2**63 in f64, so a `<= max` check would wrongly
            // accept this and the int cast would then overflow-wrap.
            let over = numpy
                .getattr("asarray")
                .unwrap()
                .call1((vec![9223372036854775808.0f64],))
                .unwrap();
            let err = encode_array_like_value_with_backend(py, &over, &space, ValueBackend::Auto)
                .unwrap_err();
            assert!(
                err.to_string().contains("not representable"),
                "unexpected error: {err}"
            );

            // The largest in-range integral f64 (2**63 - 1024) still encodes.
            let within = numpy
                .getattr("asarray")
                .unwrap()
                .call1((vec![9223372036854774784.0f64],))
                .unwrap();
            assert!(
                encode_array_like_value_with_backend(py, &within, &space, ValueBackend::Auto)
                    .is_ok()
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

    use super::super::{
        batched_space_values_to_py_with_backend, meta_map_to_pydict,
        py_any_to_batched_space_values_with_backend, py_any_to_meta_map,
    };
    use super::{py_any_to_space_value_with_backend, space_value_to_py_with_backend};
    use pyo3::types::PyDictMethods;
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
    fn discrete_rejects_non_integer_float() {
        Python::attach(|py| {
            let space = DiscreteBuilder::new(3).build().unwrap();
            let value = py.eval(pyo3::ffi::c_str!("1.9"), None, None).unwrap();

            let err = py_any_to_space_value_with_backend(py, &value, &space, ValueBackend::Native)
                .unwrap_err();
            assert!(
                err.to_string().contains("must be an integer"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn discrete_accepts_integer_valued_float() {
        Python::attach(|py| {
            let space = DiscreteBuilder::new(3).build().unwrap();
            let value = py.eval(pyo3::ffi::c_str!("1.0"), None, None).unwrap();

            let encoded =
                py_any_to_space_value_with_backend(py, &value, &space, ValueBackend::Native)
                    .unwrap();
            assert_eq!(encoded, rlmesh_spaces::SpaceValue::Discrete(1));
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

            // A charset mismatch is a range deviation: it is no longer rejected at
            // encode (the serving side applies the validation policy), but `conform`
            // still classifies it as a range deviation and reports the nested path.
            let encoded =
                py_any_to_space_value_with_backend(py, &value, &space, ValueBackend::Native)
                    .unwrap();
            let rlmesh_spaces::Conformance::Range(err) = rlmesh_spaces::conform(&space, &encoded)
            else {
                panic!("expected a range deviation for the charset mismatch");
            };

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
