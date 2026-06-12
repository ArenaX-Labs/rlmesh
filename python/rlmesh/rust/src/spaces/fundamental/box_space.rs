use crate::spaces::utils::*;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rlmesh_spaces::spaces::*;
use rlmesh_spaces::{AxiswiseBounds, BoxBounds, BoxSpec, UniformBounds};

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
        Some(BoxBounds::Axiswise(v)) => {
            let low = np.getattr("array")?.call1((v.low.clone(), &dtype))?;
            let high = np.getattr("array")?.call1((v.high.clone(), &dtype))?;
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
        None => return Err(pyo3::exceptions::PyValueError::new_err("missing bounds")),
    };

    let kwargs = PyDict::new(py);
    kwargs.set_item("shape", shape)?;
    kwargs.set_item("dtype", dtype)?;

    spaces
        .getattr("Box")?
        .call((low_obj, high_obj), Some(&kwargs))
}

pub fn parse_box<'py>(space: &Bound<'py, PyAny>) -> PyResult<SpaceSpec> {
    let shape_usize = extract_shape(&space.getattr("shape")?)?;
    let shape: Vec<i64> = shape_usize.iter().map(|&x| x as i64).collect();

    let dtype = extract_dtype(&space.getattr("dtype")?)?;
    let low = space.getattr("low")?;
    let high = space.getattr("high")?;

    let box_spec = build_box_bounds(&shape_usize, &low, &high)?;

    Ok(SpaceSpec {
        shape,
        dtype,
        spec: Some(SpaceKind::Box(box_spec)),
    })
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn build_box_bounds<'py>(
    shape: &[usize],
    low: &Bound<'py, PyAny>,
    high: &Bound<'py, PyAny>,
) -> PyResult<BoxSpec> {
    if let (Ok(lo), Ok(hi)) = (low.extract::<f64>(), high.extract::<f64>()) {
        if is_unbounded_scalar(lo, hi) {
            return Ok(BoxSpec {
                bounds: Some(BoxBounds::Unbounded(true)),
            });
        }
        return Ok(BoxSpec {
            bounds: Some(BoxBounds::Uniform(UniformBounds { low: lo, high: hi })),
        });
    }

    let rank = shape.len();
    let n = numel(shape);

    // Flatten low/high in row-major order so per-element bounds survive for
    // Boxes of any rank, rather than collapsing rank>=2 arrays to a global
    // min/max (which would silently accept out-of-bounds values).
    let lo_vec = flatten_f64(low)?;
    let hi_vec = flatten_f64(high)?;

    if lo_vec.len() == rank && hi_vec.len() == rank && rank != n {
        return Ok(BoxSpec {
            bounds: Some(BoxBounds::Axiswise(AxiswiseBounds {
                low: lo_vec,
                high: hi_vec,
            })),
        });
    }

    if lo_vec.len() == n && hi_vec.len() == n {
        if lo_vec
            .iter()
            .all(|x| x.is_infinite() && x.is_sign_negative())
            && hi_vec
                .iter()
                .all(|x| x.is_infinite() && x.is_sign_positive())
        {
            return Ok(BoxSpec {
                bounds: Some(BoxBounds::Unbounded(true)),
            });
        }

        return Ok(BoxSpec {
            bounds: Some(BoxBounds::Elementwise(rlmesh_spaces::ElementwiseBounds {
                low: lo_vec,
                high: hi_vec,
            })),
        });
    }

    // Per-element bounds whose element count matches neither the rank nor the
    // numel cannot be represented without silently degrading them. Error out
    // rather than collapse to a lossy uniform range.
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "Box low/high have {} and {} elements, which match neither the rank ({rank}) \
         nor the element count ({n}) of shape {shape:?}",
        lo_vec.len(),
        hi_vec.len(),
    )))
}

fn is_unbounded_scalar(lo: f64, hi: f64) -> bool {
    lo.is_infinite() && lo.is_sign_negative() && hi.is_infinite() && hi.is_sign_positive()
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
}
