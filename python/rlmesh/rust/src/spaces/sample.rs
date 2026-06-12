use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};
use rand::RngExt;
use rand::rngs::StdRng;
use rlmesh_spaces::spaces::{SpaceKind, SpaceSpec};
use rlmesh_spaces::{BoxBounds, BoxSpec, DType, MultiBinaryDims, MultiDiscreteNvec};

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
        SpaceKind::MultiBinary(spec) => {
            let shape = multi_binary_shape(space, spec);
            sample_boolean_array(py, &shape, rng)
        }
        SpaceKind::MultiDiscrete(spec) => sample_multi_discrete(py, spec, rng),
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
    let numel = shape.iter().product::<usize>().max(1);
    let (low, high) = box_bounds(spec, numel);
    let values = low
        .iter()
        .zip(high.iter())
        .map(|(low, high)| {
            sample_box_scalar(py, rng, *low, *high, space.dtype).map(|value| value.unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    reshape_values(py, &values, &shape)
}

fn box_bounds(spec: &BoxSpec, numel: usize) -> (Vec<f64>, Vec<f64>) {
    match &spec.bounds {
        Some(BoxBounds::Uniform(bounds)) => (vec![bounds.low; numel], vec![bounds.high; numel]),
        Some(BoxBounds::Axiswise(bounds)) => (
            repeat_or_truncate(bounds.low.as_slice(), numel, f64::NEG_INFINITY),
            repeat_or_truncate(bounds.high.as_slice(), numel, f64::INFINITY),
        ),
        Some(BoxBounds::Elementwise(bounds)) => (
            repeat_or_truncate(bounds.low.as_slice(), numel, f64::NEG_INFINITY),
            repeat_or_truncate(bounds.high.as_slice(), numel, f64::INFINITY),
        ),
        Some(BoxBounds::Unbounded(_)) | None => {
            (vec![f64::NEG_INFINITY; numel], vec![f64::INFINITY; numel])
        }
    }
}

fn repeat_or_truncate(values: &[f64], len: usize, default: f64) -> Vec<f64> {
    match values.len() {
        0 => vec![default; len],
        1 => vec![values[0]; len],
        current if current >= len => values[..len].to_vec(),
        current => values
            .iter()
            .copied()
            .cycle()
            .take(len.max(current))
            .take(len)
            .collect(),
    }
}

fn sample_box_scalar<'py>(
    py: Python<'py>,
    rng: &mut StdRng,
    low: f64,
    high: f64,
    dtype: DType,
) -> PyResult<Bound<'py, PyAny>> {
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

    match dtype {
        DType::Bool => Ok(pyo3::types::PyBool::new(py, value.round() != 0.0)
            .to_owned()
            .into_any()),
        DType::Uint8
        | DType::Int8
        | DType::Int16
        | DType::Int32
        | DType::Int64
        | DType::Uint16
        | DType::Uint32
        | DType::Uint64 => Ok((value.round() as i64).into_pyobject(py)?.into_any()),
        _ => Ok(value.into_pyobject(py)?.into_any()),
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

fn multi_binary_shape(space: &SpaceSpec, spec: &rlmesh_spaces::MultiBinarySpec) -> Vec<usize> {
    if !space.shape.is_empty() {
        return space.shape.iter().map(|dim| *dim as usize).collect();
    }
    match &spec.n {
        Some(MultiBinaryDims::Size(size)) => vec![*size as usize],
        Some(MultiBinaryDims::Dims(dims)) => dims.iter().map(|dim| *dim as usize).collect(),
        None => vec![],
    }
}

fn sample_boolean_array<'py>(
    py: Python<'py>,
    shape: &[usize],
    rng: &mut StdRng,
) -> PyResult<Bound<'py, PyAny>> {
    let numel = shape.iter().product::<usize>().max(1);
    let values = (0..numel)
        .map(|_| {
            Ok(((rng.random::<bool>()) as i64)
                .into_pyobject(py)?
                .into_any()
                .unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    reshape_values(py, &values, shape)
}

fn sample_multi_discrete<'py>(
    py: Python<'py>,
    spec: &rlmesh_spaces::MultiDiscreteSpec,
    rng: &mut StdRng,
) -> PyResult<Bound<'py, PyAny>> {
    let nvec = match &spec.nvec {
        Some(MultiDiscreteNvec::Flat(vector)) => vector
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>(),
        Some(MultiDiscreteNvec::Shaped(matrix)) => matrix
            .iter()
            .flat_map(|row| row.iter().map(|value| *value as usize))
            .collect::<Vec<_>>(),
        None => vec![],
    };
    let values = nvec
        .iter()
        .map(|n| {
            if *n == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "cannot sample MultiDiscrete space with a zero-sized dimension",
                ));
            }
            Ok(rng
                .random_range(0..(*n as i64))
                .into_pyobject(py)?
                .into_any()
                .unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    if let Some(MultiDiscreteNvec::Shaped(matrix)) = &spec.nvec {
        let shape = [matrix.len(), matrix.first().map_or(0, |row| row.len())];
        return reshape_values(py, &values, &shape);
    }
    reshape_values(py, &values, &[nvec.len()])
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

fn reshape_values<'py>(
    py: Python<'py>,
    values: &[Py<PyAny>],
    shape: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    if shape.is_empty() {
        return Ok(values
            .first()
            .map(|value| value.clone_ref(py))
            .unwrap_or_else(|| py.None())
            .bind(py)
            .clone());
    }
    let mut index = 0;
    reshape_values_recursive(py, values, shape, &mut index)
}

fn reshape_values_recursive<'py>(
    py: Python<'py>,
    values: &[Py<PyAny>],
    shape: &[usize],
    index: &mut usize,
) -> PyResult<Bound<'py, PyAny>> {
    if shape.is_empty() {
        let value = values
            .get(*index)
            .map(|value| value.clone_ref(py))
            .unwrap_or_else(|| py.None());
        *index += 1;
        return Ok(value.bind(py).clone());
    }

    let items = (0..shape[0])
        .map(|_| {
            reshape_values_recursive(py, values, &shape[1..], index).map(|value| value.unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, items)?.into_any())
}
