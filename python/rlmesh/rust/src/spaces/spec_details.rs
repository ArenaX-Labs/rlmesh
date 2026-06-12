use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rlmesh_spaces::spaces::{SpaceSpec, space_spec};
use rlmesh_spaces::{BoxSpec, box_spec, multi_binary_spec, multi_discrete_spec};

use super::spec_view::PySpaceSpec;
use crate::spaces::utils::dtype_name;

pub(super) fn space_kind_name(space: &SpaceSpec) -> &'static str {
    match space.spec.as_ref() {
        Some(space_spec::Spec::Box(_)) => "box",
        Some(space_spec::Spec::Discrete(_)) => "discrete",
        Some(space_spec::Spec::MultiBinary(_)) => "multi_binary",
        Some(space_spec::Spec::MultiDiscrete(_)) => "multi_discrete",
        Some(space_spec::Spec::Text(_)) => "text",
        Some(space_spec::Spec::Dict(_)) => "dict",
        Some(space_spec::Spec::Tuple(_)) => "tuple",
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
        space_spec::Spec::Box(spec) => add_box_details(&details, spec)?,
        space_spec::Spec::Discrete(spec) => {
            details.set_item("n", spec.n)?;
            details.set_item("start", spec.start)?;
        }
        space_spec::Spec::MultiBinary(spec) => match &spec.n {
            Some(multi_binary_spec::N::Size(size)) => {
                details.set_item("size", *size)?;
            }
            Some(multi_binary_spec::N::Dims(dims)) => {
                details.set_item("dims", dims.data.clone())?;
            }
            None => {}
        },
        space_spec::Spec::MultiDiscrete(spec) => match &spec.nvec {
            Some(multi_discrete_spec::Nvec::Flat(vector)) => {
                details.set_item("nvec", vector.data.clone())?;
            }
            Some(multi_discrete_spec::Nvec::Shaped(matrix)) => {
                let rows = matrix
                    .data
                    .iter()
                    .map(|row| row.data.clone())
                    .collect::<Vec<_>>();
                details.set_item("nvec", rows)?;
            }
            None => {}
        },
        space_spec::Spec::Text(spec) => {
            details.set_item("min_length", spec.min_length)?;
            details.set_item("max_length", spec.max_length)?;
            details.set_item("charset", spec.charset.clone())?;
        }
        space_spec::Spec::Dict(spec) => {
            let spaces = PyDict::new(py);
            for (key, child) in spec.keys.iter().zip(spec.spaces.iter()) {
                spaces.set_item(key, nested_space(py, child, &nested)?)?;
            }
            details.set_item("spaces", spaces)?;
        }
        space_spec::Spec::Tuple(spec) => {
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
        Some(box_spec::Bounds::Unbounded(_)) => {
            details.set_item("bounds_kind", "unbounded")?;
        }
        Some(box_spec::Bounds::Uniform(bounds)) => {
            details.set_item("bounds_kind", "uniform")?;
            details.set_item("low", bounds.low)?;
            details.set_item("high", bounds.high)?;
        }
        Some(box_spec::Bounds::Axiswise(bounds)) => {
            details.set_item("bounds_kind", "axiswise")?;
            details.set_item("low", bounds.low.clone())?;
            details.set_item("high", bounds.high.clone())?;
        }
        Some(box_spec::Bounds::Elementwise(bounds)) => {
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
