use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rlmesh_spaces::spaces::{SpaceKind, SpaceSpec};
use rlmesh_spaces::{BoxBounds, BoxSpec, MultiBinaryDims, MultiDiscreteNvec};

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
        SpaceKind::Box(spec) => add_box_details(&details, spec)?,
        SpaceKind::Discrete(spec) => {
            details.set_item("n", spec.n)?;
            details.set_item("start", spec.start)?;
        }
        SpaceKind::MultiBinary(spec) => match &spec.n {
            Some(MultiBinaryDims::Size(size)) => {
                details.set_item("size", *size)?;
            }
            Some(MultiBinaryDims::Dims(dims)) => {
                details.set_item("dims", dims.clone())?;
            }
            None => {}
        },
        SpaceKind::MultiDiscrete(spec) => match &spec.nvec {
            Some(MultiDiscreteNvec::Flat(vector)) => {
                details.set_item("nvec", vector.clone())?;
            }
            Some(MultiDiscreteNvec::Shaped(matrix)) => {
                let rows = matrix.clone();
                details.set_item("nvec", rows)?;
            }
            None => {}
        },
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

fn add_box_details(details: &Bound<'_, PyDict>, spec: &BoxSpec) -> PyResult<()> {
    match &spec.bounds {
        Some(BoxBounds::Unbounded(_)) => {
            details.set_item("bounds_kind", "unbounded")?;
        }
        Some(BoxBounds::Uniform(bounds)) => {
            details.set_item("bounds_kind", "uniform")?;
            details.set_item("low", bounds.low)?;
            details.set_item("high", bounds.high)?;
        }
        Some(BoxBounds::Axiswise(bounds)) => {
            details.set_item("bounds_kind", "axiswise")?;
            details.set_item("low", bounds.low.clone())?;
            details.set_item("high", bounds.high.clone())?;
        }
        Some(BoxBounds::Elementwise(bounds)) => {
            details.set_item("bounds_kind", "elementwise")?;
            details.set_item("low", bounds.low.clone())?;
            details.set_item("high", bounds.high.clone())?;
        }
        None => {
            details.set_item("bounds_kind", details.py().None())?;
        }
    }
    Ok(())
}
