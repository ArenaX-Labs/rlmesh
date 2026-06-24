use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rlmesh_spaces::scalar::{Scalar, decode_scalars};
use rlmesh_spaces::spaces::{SpaceKind, SpaceSpec};
use rlmesh_spaces::{BoxBounds, BoxSpec, DType};

use super::spec_view::PySpaceSpec;
use crate::spaces::utils::dtype_name;

pub(super) fn space_kind_name(space: &SpaceSpec) -> &'static str {
    match space.spec.as_ref() {
        Some(SpaceKind::Box(_)) => "box",
        Some(SpaceKind::Discrete(_)) => "discrete",
        Some(SpaceKind::MultiBinary(_)) => "multi_binary",
        Some(SpaceKind::MultiDiscrete(_)) => "multi_discrete",
        Some(SpaceKind::Text(_)) => "text",
        Some(SpaceKind::Dict(_)) => "dict",
        Some(SpaceKind::Tuple(_)) => "tuple",
        None => "unknown",
    }
}

pub(super) fn space_spec_to_pydict<'py>(
    py: Python<'py>,
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("kind", space_kind_name(space))?;
    result.set_item("shape", space.shape.clone())?;
    result.set_item("dtype", dtype_name(space.dtype))?;
    result.set_item("details", space_spec_details_dict(py, space)?)?;
    Ok(result)
}

pub(super) fn space_spec_details_to_py<'py>(
    py: Python<'py>,
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(space_spec_details_impl(py, space, NestedSpaceMode::Spec)?.into_any())
}

fn space_spec_details_dict<'py>(
    py: Python<'py>,
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyDict>> {
    space_spec_details_impl(py, space, NestedSpaceMode::Dict)
}

enum NestedSpaceMode {
    Dict,
    Spec,
}

fn space_spec_details_impl<'py>(
    py: Python<'py>,
    space: &SpaceSpec,
    nested: NestedSpaceMode,
) -> PyResult<Bound<'py, PyDict>> {
    let details = PyDict::new(py);
    match space
        .spec
        .as_ref()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("space spec is missing"))?
    {
        SpaceKind::Box(spec) => add_box_details(&details, spec, space.dtype)?,
        SpaceKind::Discrete(spec) => {
            details.set_item("n", spec.n)?;
            details.set_item("start", spec.start)?;
        }
        // The MultiBinary marker carries no fields; dimensions come from
        // `shape`. A rank-1 shape surfaces as a scalar `size`, higher ranks as
        // `dims`, mirroring Gymnasium's scalar-vs-vector MultiBinary.
        SpaceKind::MultiBinary(_) => {
            if space.shape.len() == 1 {
                details.set_item("size", space.shape[0])?;
            } else {
                details.set_item("dims", space.shape.clone())?;
            }
        }
        // `nvec` is stored flat (row-major). A rank-2 shape is reshaped back to
        // the nested `[[...], [...]]` matrix form; lower ranks stay flat.
        SpaceKind::MultiDiscrete(spec) => {
            if space.shape.len() == 2 {
                let cols = space.shape[1].max(0) as usize;
                let rows: Vec<Vec<i64>> = if cols == 0 {
                    Vec::new()
                } else {
                    spec.nvec.chunks(cols).map(|chunk| chunk.to_vec()).collect()
                };
                details.set_item("nvec", rows)?;
            } else {
                details.set_item("nvec", spec.nvec.clone())?;
            }
        }
        SpaceKind::Text(spec) => {
            details.set_item("min_length", spec.min_length)?;
            details.set_item("max_length", spec.max_length)?;
            details.set_item("charset", spec.charset.clone())?;
        }
        SpaceKind::Dict(spec) => {
            let spaces = PyDict::new(py);
            for (key, child) in spec.keys.iter().zip(spec.spaces.iter()) {
                spaces.set_item(key, nested_space(py, child, &nested)?)?;
            }
            details.set_item("spaces", spaces)?;
        }
        SpaceKind::Tuple(spec) => {
            let spaces = spec
                .spaces
                .iter()
                .map(|child| nested_space(py, child, &nested).map(|value| value.unbind()))
                .collect::<PyResult<Vec<_>>>()?;
            details.set_item("spaces", PyList::new(py, spaces)?)?;
        }
    }
    Ok(details)
}

fn nested_space<'py>(
    py: Python<'py>,
    child: &SpaceSpec,
    nested: &NestedSpaceMode,
) -> PyResult<Bound<'py, PyAny>> {
    match nested {
        NestedSpaceMode::Dict => Ok(space_spec_to_pydict(py, child)?.into_any()),
        NestedSpaceMode::Spec => Py::new(
            py,
            PySpaceSpec {
                inner: child.clone(),
            },
        )
        .map(|value| value.into_bound(py).into_any()),
    }
}

fn add_box_details(details: &Bound<'_, PyDict>, spec: &BoxSpec, dtype: DType) -> PyResult<()> {
    match &spec.bounds {
        Some(BoxBounds::Unbounded(_)) => {
            details.set_item("bounds_kind", "unbounded")?;
        }
        Some(BoxBounds::Uniform(bounds)) => {
            details.set_item("bounds_kind", "uniform")?;
            details.set_item("low", bounds.low)?;
            details.set_item("high", bounds.high)?;
        }
        Some(BoxBounds::Elementwise(bounds)) => {
            details.set_item("bounds_kind", "elementwise")?;
            details.set_item("low", bounds.low.clone())?;
            details.set_item("high", bounds.high.clone())?;
        }
        // Dtype-typed byte bounds are decoded into native Python numbers in the
        // space's dtype, so the integer values stay exact (no f64 round-trip).
        Some(BoxBounds::TypedUniform(bounds)) => {
            details.set_item("bounds_kind", "typed_uniform")?;
            let py = details.py();
            details.set_item("low", typed_bounds_to_py(py, &bounds.low, dtype)?)?;
            details.set_item("high", typed_bounds_to_py(py, &bounds.high, dtype)?)?;
        }
        Some(BoxBounds::TypedElementwise(bounds)) => {
            details.set_item("bounds_kind", "typed_elementwise")?;
            let py = details.py();
            details.set_item("low", typed_bounds_to_py(py, &bounds.low, dtype)?)?;
            details.set_item("high", typed_bounds_to_py(py, &bounds.high, dtype)?)?;
        }
        None => {
            details.set_item("bounds_kind", details.py().None())?;
        }
    }
    Ok(())
}

/// Decode dtype-typed bound bytes into a Python list of native numbers,
/// preserving exact integer values (int64/uint64 do not round-trip through
/// f64). `Uint64` values above `i64::MAX` are returned as Python ints.
fn typed_bounds_to_py<'py>(
    py: Python<'py>,
    bytes: &[u8],
    dtype: DType,
) -> PyResult<Bound<'py, PyList>> {
    let scalars = decode_scalars(bytes, dtype)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let items = scalars
        .into_iter()
        .map(|scalar| scalar_to_py(py, scalar, dtype))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

fn scalar_to_py<'py>(py: Python<'py>, scalar: Scalar, dtype: DType) -> PyResult<Bound<'py, PyAny>> {
    Ok(match scalar {
        Scalar::Bool(value) => value.into_pyobject(py)?.to_owned().into_any(),
        // `Uint64` decodes into a wrapped i64; use the centralized reinterpret
        // so values above i64::MAX surface as the correct positive Python int.
        Scalar::Int(_) if dtype == DType::Uint64 => scalar.as_u64().into_pyobject(py)?.into_any(),
        Scalar::Int(value) => value.into_pyobject(py)?.into_any(),
        Scalar::Float(value) => value.into_pyobject(py)?.into_any(),
    })
}
