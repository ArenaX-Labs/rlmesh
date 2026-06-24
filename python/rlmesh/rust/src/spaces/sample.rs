use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};
use rand::RngExt;
use rand::rngs::StdRng;
use rlmesh_spaces::scalar::decode_scalars;
use rlmesh_spaces::spaces::{SpaceKind, SpaceSpec};
use rlmesh_spaces::{BoxBounds, BoxSpec, DType, Scalar, encode_scalars};

use crate::spaces::tensor::make_tensor;
use crate::spaces::utils::dtype_name;

pub(super) fn sample_space_value<'py>(
    py: Python<'py>,
    space: &SpaceSpec,
    rng: &mut StdRng,
) -> PyResult<Bound<'py, PyAny>> {
    match space
        .spec
        .as_ref()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("space spec is missing"))?
    {
        SpaceKind::Box(spec) => sample_box(py, space, spec, rng),
        SpaceKind::Discrete(spec) => {
            if spec.n <= 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "cannot sample Discrete space with n={}",
                    spec.n
                )));
            }
            let value = rng.random_range(spec.start..(spec.start + spec.n));
            Ok(value.into_pyobject(py)?.into_any())
        }
        SpaceKind::MultiBinary(_) => {
            let shape = multi_binary_shape(space);
            sample_boolean_array(py, space, &shape, rng)
        }
        SpaceKind::MultiDiscrete(spec) => sample_multi_discrete(py, space, spec, rng),
        SpaceKind::Text(spec) => sample_text(py, spec, rng),
        SpaceKind::Dict(spec) => sample_dict(py, spec, rng),
        SpaceKind::Tuple(spec) => sample_tuple(py, spec, rng),
    }
}

fn sample_box<'py>(
    py: Python<'py>,
    space: &SpaceSpec,
    spec: &BoxSpec,
    rng: &mut StdRng,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = space
        .shape
        .iter()
        .map(|dim| *dim as usize)
        .collect::<Vec<_>>();
    let numel = shape.iter().product::<usize>();
    let (low, high) = box_bounds(spec, numel.max(1), space.dtype);
    let scalars = low
        .iter()
        .zip(high.iter())
        .take(numel)
        .map(|(low, high)| sample_box_scalar(rng, *low, *high, space.dtype))
        .collect::<PyResult<Vec<_>>>()?;
    tensor_from_scalars(py, &scalars, shape, space.dtype)
}

/// Encode sampled scalars straight into a native Tensor buffer, avoiding a
/// round trip through per-element Python objects, nested lists, and a
/// Python-side struct.pack.
fn tensor_from_scalars<'py>(
    py: Python<'py>,
    scalars: &[Scalar],
    shape: Vec<usize>,
    dtype: DType,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype = normalize_dtype(dtype);
    let bytes = encode_scalars(scalars, dtype)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    make_tensor(py, bytes, shape, dtype_name(dtype as i32))
}

/// The sampler treats `Unspecified` specs as float32, matching the codecs.
fn normalize_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Unspecified => DType::Float32,
        other => other,
    }
}

fn box_bounds(spec: &BoxSpec, numel: usize, dtype: DType) -> (Vec<f64>, Vec<f64>) {
    match &spec.bounds {
        Some(BoxBounds::Uniform(bounds)) => (vec![bounds.low; numel], vec![bounds.high; numel]),
        Some(BoxBounds::Elementwise(bounds)) => (
            elementwise_or_default(bounds.low.as_slice(), numel, f64::NEG_INFINITY),
            elementwise_or_default(bounds.high.as_slice(), numel, f64::INFINITY),
        ),
        // Typed byte bounds are decoded with the space dtype into f64 for
        // sampling. The sampler only needs an approximate range, so the f64
        // view is acceptable here even for int64/uint64.
        Some(BoxBounds::TypedUniform(bounds)) => {
            let low = decode_typed_one(&bounds.low, dtype).unwrap_or(f64::NEG_INFINITY);
            let high = decode_typed_one(&bounds.high, dtype).unwrap_or(f64::INFINITY);
            (vec![low; numel], vec![high; numel])
        }
        Some(BoxBounds::TypedElementwise(bounds)) => (
            decode_typed_many(&bounds.low, dtype, numel, f64::NEG_INFINITY),
            decode_typed_many(&bounds.high, dtype, numel, f64::INFINITY),
        ),
        Some(BoxBounds::Unbounded(_)) | None => {
            (vec![f64::NEG_INFINITY; numel], vec![f64::INFINITY; numel])
        }
    }
}

fn decode_typed_one(bytes: &[u8], dtype: DType) -> Option<f64> {
    decode_scalars(bytes, dtype)
        .ok()
        .and_then(|scalars| scalars.first().copied().map(|s| s.to_f64(dtype)))
}

fn decode_typed_many(bytes: &[u8], dtype: DType, numel: usize, default: f64) -> Vec<f64> {
    match decode_scalars(bytes, dtype) {
        Ok(scalars) if scalars.len() == numel => {
            scalars.into_iter().map(|s| s.to_f64(dtype)).collect()
        }
        _ => vec![default; numel],
    }
}

/// Validated elementwise bounds always carry one value per element; anything
/// else falls back to the unbounded default for sampling purposes.
fn elementwise_or_default(values: &[f64], len: usize, default: f64) -> Vec<f64> {
    if values.len() == len {
        values.to_vec()
    } else {
        vec![default; len]
    }
}

fn sample_box_scalar(rng: &mut StdRng, low: f64, high: f64, dtype: DType) -> PyResult<Scalar> {
    if is_integer_dtype(dtype) {
        return sample_integer_box_scalar(rng, low, high, dtype);
    }

    let value = sample_continuous(rng, low, high)?;
    if matches!(dtype, DType::Bool) {
        return Ok(Scalar::Bool(value.round() != 0.0));
    }
    Ok(Scalar::Float(value))
}

/// Sample a continuous value within `[low, high]`, falling back to
/// exponential/normal sampling for half- and un-bounded ranges.
fn sample_continuous(rng: &mut StdRng, low: f64, high: f64) -> PyResult<f64> {
    let value = if low.is_finite() && high.is_finite() {
        if low > high {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cannot sample Box element with low {low} greater than high {high}"
            )));
        }
        rng.random_range(low..=high)
    } else if low.is_finite() {
        low + exp_sample(rng)
    } else if high.is_finite() {
        high - exp_sample(rng)
    } else {
        normal_sample(rng)
    };
    Ok(value)
}

/// Sample an integer-dtype Box element with a uniform integer distribution
/// over `[ceil(low), floor(high)]`, matching Gymnasium instead of rounding a
/// continuous uniform (which biases the endpoints).
fn sample_integer_box_scalar(
    rng: &mut StdRng,
    low: f64,
    high: f64,
    dtype: DType,
) -> PyResult<Scalar> {
    // Clamp the open ends to the dtype's representable range so an unbounded
    // integer Box still samples a finite value.
    let (dtype_lo, dtype_hi) = integer_dtype_bounds(dtype);
    let lo = if low.is_finite() {
        low.ceil()
    } else {
        dtype_lo
    };
    let hi = if high.is_finite() {
        high.floor()
    } else {
        dtype_hi
    };
    if lo > hi {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "cannot sample integer Box element with low {low} greater than high {high}"
        )));
    }
    let lo = lo as i64;
    let hi = hi as i64;
    Ok(Scalar::Int(rng.random_range(lo..=hi)))
}

fn is_integer_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Uint8
            | DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::Uint16
            | DType::Uint32
            | DType::Uint64
    )
}

/// Conservative finite sampling bounds for an unbounded integer Box element.
/// These stay well within `i64` so `random_range` cannot overflow.
fn integer_dtype_bounds(dtype: DType) -> (f64, f64) {
    match dtype {
        DType::Bool => (0.0, 1.0),
        DType::Uint8 => (0.0, u8::MAX as f64),
        DType::Int8 => (i8::MIN as f64, i8::MAX as f64),
        DType::Int16 => (i16::MIN as f64, i16::MAX as f64),
        DType::Uint16 => (0.0, u16::MAX as f64),
        DType::Int32 => (i32::MIN as f64, i32::MAX as f64),
        DType::Uint32 => (0.0, u32::MAX as f64),
        // Keep i64/u64 within a range that round-trips through f64 exactly.
        DType::Int64 => (-(1i64 << 53) as f64, (1i64 << 53) as f64),
        DType::Uint64 => (0.0, (1u64 << 53) as f64),
        _ => (-(1i64 << 53) as f64, (1i64 << 53) as f64),
    }
}

fn exp_sample(rng: &mut StdRng) -> f64 {
    let u = (1.0 - rng.random::<f64>()).max(f64::MIN_POSITIVE);
    -u.ln()
}

fn normal_sample(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn multi_binary_shape(space: &SpaceSpec) -> Vec<usize> {
    space.shape.iter().map(|dim| *dim as usize).collect()
}

fn sample_boolean_array<'py>(
    py: Python<'py>,
    space: &SpaceSpec,
    shape: &[usize],
    rng: &mut StdRng,
) -> PyResult<Bound<'py, PyAny>> {
    let numel = shape.iter().product::<usize>();
    let scalars = (0..numel)
        .map(|_| Scalar::Int(i64::from(rng.random::<bool>())))
        .collect::<Vec<_>>();
    tensor_from_scalars(py, &scalars, shape.to_vec(), space.dtype)
}

fn sample_multi_discrete<'py>(
    py: Python<'py>,
    space: &SpaceSpec,
    spec: &rlmesh_spaces::MultiDiscreteSpec,
    rng: &mut StdRng,
) -> PyResult<Bound<'py, PyAny>> {
    let scalars = spec
        .nvec
        .iter()
        .map(|n| {
            if *n <= 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "cannot sample MultiDiscrete space with a non-positive dimension",
                ));
            }
            Ok(Scalar::Int(rng.random_range(0..*n)))
        })
        .collect::<PyResult<Vec<_>>>()?;
    // The logical shape lives in `SpaceSpec.shape` (flat for rank-1, `[rows,
    // cols]` for a matrix).
    let shape: Vec<usize> = space.shape.iter().map(|dim| *dim as usize).collect();
    tensor_from_scalars(py, &scalars, shape, space.dtype)
}

fn sample_text<'py>(
    py: Python<'py>,
    spec: &rlmesh_spaces::TextSpec,
    rng: &mut StdRng,
) -> PyResult<Bound<'py, PyAny>> {
    let printable_ascii = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?";
    let chars = if spec.charset.is_empty() {
        printable_ascii.chars().collect::<Vec<_>>()
    } else {
        spec.charset.chars().collect::<Vec<_>>()
    };
    let length = if spec.max_length <= spec.min_length {
        spec.min_length
    } else {
        rng.random_range(spec.min_length..=spec.max_length)
    };
    let value = (0..length)
        .map(|_| chars[rng.random_range(0..chars.len())])
        .collect::<String>();
    Ok(value.into_pyobject(py)?.into_any())
}

fn sample_dict<'py>(
    py: Python<'py>,
    spec: &rlmesh_spaces::DictSpec,
    rng: &mut StdRng,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    for (key, child) in spec.keys.iter().zip(spec.spaces.iter()) {
        dict.set_item(key, sample_space_value(py, child, rng)?)?;
    }
    Ok(dict.into_any())
}

fn sample_tuple<'py>(
    py: Python<'py>,
    spec: &rlmesh_spaces::TupleSpec,
    rng: &mut StdRng,
) -> PyResult<Bound<'py, PyAny>> {
    let values = spec
        .spaces
        .iter()
        .map(|child| sample_space_value(py, child, rng).map(|value| value.unbind()))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, values)?.into_any())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rlmesh_spaces::{TypedUniformBounds, encode_i64_scalars};

    fn typed_uniform_spec(low: i64, high: i64, dtype: DType) -> BoxSpec {
        BoxSpec {
            bounds: Some(BoxBounds::TypedUniform(TypedUniformBounds {
                low: encode_i64_scalars(&[low], dtype).expect("encode low"),
                high: encode_i64_scalars(&[high], dtype).expect("encode high"),
            })),
        }
    }

    #[test]
    fn test_uint64_max_bound_decodes_to_positive_for_sampling() {
        let spec = typed_uniform_spec(0, -1, DType::Uint64);
        let (low, high) = box_bounds(&spec, 1, DType::Uint64);
        assert_eq!(low, vec![0.0]);
        assert_eq!(high, vec![u64::MAX as f64], "u64::MAX must decode positive");

        // Sampling the integer Box scalar must succeed (low <= high).
        let mut rng = StdRng::seed_from_u64(7);
        let sampled = sample_integer_box_scalar(&mut rng, low[0], high[0], DType::Uint64);
        assert!(
            sampled.is_ok(),
            "sampling a [0, u64::MAX] Uint64 box must not error: {sampled:?}"
        );
    }

    #[test]
    fn test_scalar_to_f64_via_to_f64_matches_dtype() {
        // The sampler now routes through Scalar::to_f64(dtype); Uint64 -1 bits
        // read as the large positive magnitude.
        assert_eq!(Scalar::Int(-1).to_f64(DType::Uint64), u64::MAX as f64);
        assert_eq!(Scalar::Int(-1).to_f64(DType::Int64), -1.0);
    }
}
