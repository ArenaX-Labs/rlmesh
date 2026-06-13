use crate::spaces::utils::*;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use rlmesh_spaces::spaces::*;
use rlmesh_spaces::{
    BoxBounds, BoxSpec, DType, ElementwiseBounds, TypedElementwiseBounds, TypedUniformBounds,
    UniformBounds,
};

pub fn make_box<'py>(
    py: Python<'py>,
    spaces: &Bound<'py, PyAny>,
    spec: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    let b = match &spec.spec {
        Some(SpaceKind::Box(b)) => b,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "missing box detail",
            ));
        }
    };

    let shape_i64 = spec.shape.clone();
    let shape: Vec<usize> = shape_i64.iter().map(|&x| x as usize).collect();

    let np = py.import("numpy")?;
    let dtype = dtype_to_py(py, spec.dtype)?;

    let (low_obj, high_obj) = match &b.bounds {
        Some(BoxBounds::Unbounded(_)) => {
            let inf = np.getattr("inf")?;
            let low = np.getattr("full")?.call1((
                &shape,
                inf.clone().call_method0("__neg__")?,
                &dtype,
            ))?;
            let high = np.getattr("full")?.call1((&shape, inf, &dtype))?;
            (low, high)
        }
        Some(BoxBounds::Uniform(s)) => {
            let low = np.getattr("full")?.call1((&shape, s.low, &dtype))?;
            let high = np.getattr("full")?.call1((&shape, s.high, &dtype))?;
            (low, high)
        }
        Some(BoxBounds::Elementwise(t)) => {
            let low = np
                .getattr("array")?
                .call1((t.low.clone(), &dtype))?
                .call_method1("reshape", (shape.clone(),))?;
            let high = np
                .getattr("array")?
                .call1((t.high.clone(), &dtype))?
                .call_method1("reshape", (shape.clone(),))?;
            (low, high)
        }
        Some(BoxBounds::TypedUniform(t)) => {
            // One scalar each, in the space's dtype. Broadcast to the shape.
            let low_scalar = typed_bytes_to_np(py, &np, &t.low, &dtype, &[1])?;
            let high_scalar = typed_bytes_to_np(py, &np, &t.high, &dtype, &[1])?;
            let low = np
                .getattr("full")?
                .call1((&shape, low_scalar.get_item(0)?, &dtype))?;
            let high = np
                .getattr("full")?
                .call1((&shape, high_scalar.get_item(0)?, &dtype))?;
            (low, high)
        }
        Some(BoxBounds::TypedElementwise(t)) => {
            let low = typed_bytes_to_np(py, &np, &t.low, &dtype, &shape)?;
            let high = typed_bytes_to_np(py, &np, &t.high, &dtype, &shape)?;
            (low, high)
        }
        None => return Err(pyo3::exceptions::PyValueError::new_err("missing bounds")),
    };

    let kwargs = PyDict::new(py);
    kwargs.set_item("shape", shape)?;
    kwargs.set_item("dtype", dtype)?;

    spaces
        .getattr("Box")?
        .call((low_obj, high_obj), Some(&kwargs))
}

/// Decode dtype-typed bound bytes into a writable numpy array of `shape`.
///
/// The bytes are interpreted as little-endian `dtype` scalars via
/// `np.frombuffer`; the result is copied (frombuffer yields a read-only view)
/// and reshaped so gymnasium.Box receives an owned, mutable array.
fn typed_bytes_to_np<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyAny>,
    bytes: &[u8],
    dtype: &Bound<'py, PyAny>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let buffer = PyBytes::new(py, bytes);
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype)?;
    np.getattr("frombuffer")?
        .call((buffer,), Some(&kwargs))?
        .call_method0("copy")?
        .call_method1("reshape", (shape.to_vec(),))
}

pub fn parse_box<'py>(space: &Bound<'py, PyAny>) -> PyResult<SpaceSpec> {
    let py = space.py();
    let shape_usize = extract_shape(&space.getattr("shape")?)?;
    let shape: Vec<i64> = shape_usize.iter().map(|&x| x as i64).collect();

    let dtype = extract_dtype(&space.getattr("dtype")?)?;
    let low = space.getattr("low")?;
    let high = space.getattr("high")?;

    let box_spec = build_box_bounds(py, &shape_usize, dtype, &low, &high)?;

    Ok(SpaceSpec {
        shape,
        dtype,
        spec: Some(SpaceKind::Box(box_spec)),
    })
}

/// dtype families that carry exact integer/boolean bounds. Their bounds are
/// emitted as dtype-typed bytes so values beyond 2^53 (notably int64/uint64)
/// round-trip exactly instead of degrading through `f64`.
fn is_integral_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Bool
            | DType::Uint8
            | DType::Int8
            | DType::Int16
            | DType::Uint16
            | DType::Int32
            | DType::Uint32
            | DType::Int64
            | DType::Uint64
    )
}

/// Build Box bounds from a gymnasium space's `low`/`high`.
///
/// Per-axis or scalar inputs are broadcast to the full shape with NumPy
/// semantics (`np.broadcast_to`) and flattened in row-major order, so the
/// stored bounds are always either uniform or fully elementwise — there is no
/// ambiguous per-axis form. Integer/boolean dtypes carry their bounds as
/// dtype-typed bytes (exact); float dtypes keep the `double` forms.
fn build_box_bounds<'py>(
    py: Python<'py>,
    shape: &[usize],
    dtype: DType,
    low: &Bound<'py, PyAny>,
    high: &Bound<'py, PyAny>,
) -> PyResult<BoxSpec> {
    let np = py.import("numpy")?;

    // Broadcast low/high to the full shape (matching gymnasium/NumPy), then
    // work with contiguous row-major arrays.
    let low = broadcast_to(&np, low, shape)?;
    let high = broadcast_to(&np, high, shape)?;

    if is_integral_dtype(dtype) {
        return integral_box_bounds(py, &np, dtype, &low, &high);
    }

    // Float dtypes: keep the historical double-based bounds.
    let lo_vec = flatten_f64(&low)?;
    let hi_vec = flatten_f64(&high)?;

    let all_neg_inf = lo_vec
        .iter()
        .all(|x| x.is_infinite() && x.is_sign_negative());
    let all_pos_inf = hi_vec
        .iter()
        .all(|x| x.is_infinite() && x.is_sign_positive());
    if all_neg_inf && all_pos_inf {
        return Ok(BoxSpec {
            bounds: Some(BoxBounds::Unbounded(true)),
        });
    }

    if let (Some(lo), Some(hi)) = (uniform_value(&lo_vec), uniform_value(&hi_vec)) {
        return Ok(BoxSpec {
            bounds: Some(BoxBounds::Uniform(UniformBounds { low: lo, high: hi })),
        });
    }

    Ok(BoxSpec {
        bounds: Some(BoxBounds::Elementwise(ElementwiseBounds {
            low: lo_vec,
            high: hi_vec,
        })),
    })
}

/// Integer/boolean Box bounds, encoded as dtype-typed little-endian bytes.
fn integral_box_bounds<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyAny>,
    dtype: DType,
    low: &Bound<'py, PyAny>,
    high: &Bound<'py, PyAny>,
) -> PyResult<BoxSpec> {
    let low_bytes = typed_bytes_from_np(py, np, dtype, low)?;
    let high_bytes = typed_bytes_from_np(py, np, dtype, high)?;

    let elem = rlmesh_spaces::dtype_size(dtype);
    // Detect a uniform bound (every element identical) to store a single
    // scalar instead of numel copies.
    if let (Some(low_one), Some(high_one)) = (
        uniform_chunk(&low_bytes, elem),
        uniform_chunk(&high_bytes, elem),
    ) {
        return Ok(BoxSpec {
            bounds: Some(BoxBounds::TypedUniform(TypedUniformBounds {
                low: low_one,
                high: high_one,
            })),
        });
    }

    Ok(BoxSpec {
        bounds: Some(BoxBounds::TypedElementwise(TypedElementwiseBounds {
            low: low_bytes,
            high: high_bytes,
        })),
    })
}

/// Cast a numpy array to `dtype` and return its little-endian C-order bytes.
///
/// `astype` copies into a freshly allocated array (so a broadcast view becomes
/// real per-element data), `ascontiguousarray` forces C order, and `tobytes`
/// defaults to C order; the result is `numel * dtype_size(dtype)` bytes.
fn typed_bytes_from_np<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyAny>,
    dtype: DType,
    array: &Bound<'py, PyAny>,
) -> PyResult<Vec<u8>> {
    let np_dtype = dtype_to_py(py, dtype)?;
    let typed = array.call_method1("astype", (np_dtype,))?;
    let contiguous = np.getattr("ascontiguousarray")?.call1((typed,))?;
    contiguous.call_method0("tobytes")?.extract::<Vec<u8>>()
}

/// `Some(scalar)` if every f64 in the slice is identical, else `None`.
fn uniform_value(values: &[f64]) -> Option<f64> {
    let first = *values.first()?;
    values
        .iter()
        .all(|v| v.to_bits() == first.to_bits())
        .then_some(first)
}

/// `Some(chunk)` if every `elem`-byte chunk is identical, else `None`.
fn uniform_chunk(bytes: &[u8], elem: usize) -> Option<Vec<u8>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(elem) {
        return None;
    }
    let first = &bytes[..elem];
    bytes
        .chunks_exact(elem)
        .all(|chunk| chunk == first)
        .then(|| first.to_vec())
}

/// `np.broadcast_to(np.asarray(obj), shape)` — expands scalar or per-axis
/// inputs to the full shape exactly as gymnasium/NumPy would.
fn broadcast_to<'py>(
    np: &Bound<'py, PyAny>,
    obj: &Bound<'py, PyAny>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let array = np.getattr("asarray")?.call1((obj,))?;
    np.getattr("broadcast_to")?.call1((array, shape.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::{make_box, parse_box};
    use crate::spaces::utils::import_gym;
    use pyo3::Python;
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
    use rlmesh_spaces::spaces::{SpaceKind, SpaceSpec};
    use rlmesh_spaces::{BoxBounds, BoxSpec, DType};

    fn gym_box<'py>(
        py: pyo3::Python<'py>,
        expr: &std::ffi::CStr,
    ) -> pyo3::Bound<'py, pyo3::types::PyAny> {
        let spaces = import_gym(py).unwrap().getattr("spaces").unwrap();
        let np = py.import("numpy").unwrap();
        let globals = PyDict::new(py);
        globals.set_item("spaces", spaces).unwrap();
        globals.set_item("np", np).unwrap();
        py.eval(expr, Some(&globals), None).unwrap()
    }

    #[test]
    fn parse_box_preserves_rank2_elementwise_bounds() {
        Python::attach(|py| {
            let space = gym_box(
                py,
                pyo3::ffi::c_str!(
                    "spaces.Box(low=np.array([[0.,0.],[5.,5.]]), high=np.array([[1.,1.],[10.,10.]]))"
                ),
            );

            let parsed = parse_box(&space).unwrap();
            let Some(SpaceKind::Box(spec)) = parsed.spec else {
                panic!("expected Box space");
            };
            let Some(BoxBounds::Elementwise(bounds)) = spec.bounds else {
                panic!("expected per-element bounds, got {:?}", spec.bounds);
            };
            assert_eq!(bounds.low, vec![0.0, 0.0, 5.0, 5.0]);
            assert_eq!(bounds.high, vec![1.0, 1.0, 10.0, 10.0]);
        });
    }

    #[test]
    fn make_box_keeps_unbounded_float32_bounds_at_float32() {
        Python::attach(|py| {
            let spaces = import_gym(py).unwrap();
            let spaces = spaces.getattr("spaces").unwrap();
            let spec = SpaceSpec {
                shape: vec![3],
                dtype: DType::Float32,
                spec: Some(SpaceKind::Box(BoxSpec {
                    bounds: Some(BoxBounds::Unbounded(true)),
                })),
            };

            let box_space = make_box(py, &spaces, &spec).unwrap();
            let low_dtype = box_space
                .getattr("low")
                .unwrap()
                .getattr("dtype")
                .unwrap()
                .getattr("name")
                .unwrap()
                .extract::<String>()
                .unwrap();
            let high_dtype = box_space
                .getattr("high")
                .unwrap()
                .getattr("dtype")
                .unwrap()
                .getattr("name")
                .unwrap()
                .extract::<String>()
                .unwrap();

            assert_eq!(low_dtype, "float32");
            assert_eq!(high_dtype, "float32");
        });
    }

    #[test]
    fn parse_box_int64_high_roundtrips_max_exactly() {
        Python::attach(|py| {
            // high = 2^63 - 1; an f64 bound would round this up to 2^63.
            let space = gym_box(
                py,
                pyo3::ffi::c_str!(
                    "spaces.Box(low=np.int64(0), high=np.int64(9223372036854775807), shape=(2,), dtype=np.int64)"
                ),
            );
            let parsed = parse_box(&space).unwrap();
            assert_eq!(parsed.dtype, DType::Int64);
            let Some(SpaceKind::Box(spec)) = parsed.spec else {
                panic!("expected Box space");
            };
            let Some(BoxBounds::TypedUniform(bounds)) = spec.bounds else {
                panic!("expected typed-uniform bounds, got {:?}", spec.bounds);
            };
            assert_eq!(bounds.low, 0i64.to_le_bytes());
            assert_eq!(bounds.high, i64::MAX.to_le_bytes());
        });
    }

    #[test]
    fn parse_box_int64_elementwise_roundtrips_exactly() {
        Python::attach(|py| {
            let space = gym_box(
                py,
                pyo3::ffi::c_str!(
                    "spaces.Box(low=np.array([0, 100], dtype=np.int64), \
                     high=np.array([10, 9223372036854775807], dtype=np.int64), dtype=np.int64)"
                ),
            );
            let parsed = parse_box(&space).unwrap();
            let Some(SpaceKind::Box(spec)) = parsed.spec else {
                panic!("expected Box space");
            };
            let Some(BoxBounds::TypedElementwise(bounds)) = spec.bounds else {
                panic!("expected typed-elementwise bounds, got {:?}", spec.bounds);
            };
            let expected_high: Vec<u8> = [10i64, i64::MAX]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(bounds.high, expected_high);
        });
    }

    #[test]
    fn parse_box_shape_2_1_is_elementwise_not_misclassified() {
        Python::attach(|py| {
            let space = gym_box(
                py,
                pyo3::ffi::c_str!(
                    "spaces.Box(low=np.array([[0.],[1.]]), high=np.array([[1.],[2.]]))"
                ),
            );
            let parsed = parse_box(&space).unwrap();
            let Some(SpaceKind::Box(spec)) = &parsed.spec else {
                panic!("expected Box space");
            };
            let Some(BoxBounds::Elementwise(bounds)) = &spec.bounds else {
                panic!("expected elementwise bounds, got {:?}", spec.bounds);
            };
            assert_eq!(bounds.low, vec![0.0, 1.0]);
            assert_eq!(bounds.high, vec![1.0, 2.0]);

            let spaces = import_gym(py).unwrap().getattr("spaces").unwrap();
            make_box(py, &spaces, &parsed).unwrap();
        });
    }

    #[test]
    fn parse_box_per_axis_input_broadcasts_like_gymnasium() {
        Python::attach(|py| {
            // gymnasium broadcasts a scalar low across a (2,3) Box; the parsed
            // elementwise bounds must match that row-major broadcast rather than
            // collapse to a global min/max or misclassify.
            let space = gym_box(
                py,
                pyo3::ffi::c_str!(
                    "spaces.Box(low=0.0, high=np.array([[1.,2.,3.],[4.,5.,6.]]), shape=(2,3))"
                ),
            );
            let parsed = parse_box(&space).unwrap();
            let Some(SpaceKind::Box(spec)) = parsed.spec else {
                panic!("expected Box space");
            };
            let Some(BoxBounds::Elementwise(bounds)) = spec.bounds else {
                panic!("expected elementwise bounds, got {:?}", spec.bounds);
            };
            assert_eq!(bounds.low, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            assert_eq!(bounds.high, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        });
    }

    #[test]
    fn make_box_reconstructs_typed_int64_bounds() {
        Python::attach(|py| {
            use rlmesh_spaces::TypedUniformBounds;
            let spaces = import_gym(py).unwrap().getattr("spaces").unwrap();
            let spec = SpaceSpec {
                shape: vec![2],
                dtype: DType::Int64,
                spec: Some(SpaceKind::Box(BoxSpec {
                    bounds: Some(BoxBounds::TypedUniform(TypedUniformBounds {
                        low: 0i64.to_le_bytes().to_vec(),
                        high: i64::MAX.to_le_bytes().to_vec(),
                    })),
                })),
            };
            let box_space = make_box(py, &spaces, &spec).unwrap();
            let high_max = box_space
                .getattr("high")
                .unwrap()
                .call_method0("max")
                .unwrap()
                .call_method0("item")
                .unwrap()
                .extract::<i64>()
                .unwrap();
            assert_eq!(high_max, i64::MAX);
        });
    }
}
