use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use rlmesh_spaces::DType;

/// Parse a tensor dtype name, rejecting anything unknown.
///
/// Unlike [`extract_dtype`], there is no `Unspecified` fallback: tensors
/// must carry a concrete dtype.
pub fn parse_dtype_strict(dtype: &str) -> PyResult<DType> {
    DType::from_name(dtype).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unsupported tensor dtype {dtype:?}"))
    })
}

pub fn extract_shape<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(t) = obj.cast::<PyTuple>() {
        return t.iter().map(|x| x.extract::<usize>()).collect();
    }
    obj.try_iter()?.map(|x| x?.extract::<usize>()).collect()
}

pub fn extract_dtype<'py>(obj: &Bound<'py, PyAny>) -> PyResult<DType> {
    let name = if let Ok(n) = obj.getattr("name").and_then(|x| x.extract::<String>()) {
        n
    } else {
        obj.str()?.to_string()
    };
    let norm = name.to_lowercase();
    Ok(DType::from_name(&norm).unwrap_or(DType::Unspecified))
}

pub fn dtype_to_py<'py, T>(py: Python<'py>, dt: T) -> PyResult<Bound<'py, PyAny>>
where
    T: Into<i32>,
{
    let np = py.import("numpy")?;
    let name = dtype_name(dt);
    np.getattr("dtype")?.call1((name,))
}

pub fn dtype_name<T>(dt: T) -> &'static str
where
    T: Into<i32>,
{
    match DType::try_from(dt.into()).unwrap_or(DType::Unspecified) {
        // Legacy display fallback: unspecified specs surface as float32.
        DType::Unspecified => "float32",
        dtype => dtype.name(),
    }
}

pub fn extract_1d_f64<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
    obj.try_iter()?.map(|x| x?.extract::<f64>()).collect()
}

pub fn deep_min_f64<'py>(obj: &Bound<'py, PyAny>) -> PyResult<f64> {
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(v);
    }
    let mut m = f64::INFINITY;
    for item in obj.try_iter()? {
        let v = deep_min_f64(&item?)?;
        if v < m {
            m = v;
        }
    }
    Ok(m)
}

pub fn deep_max_f64<'py>(obj: &Bound<'py, PyAny>) -> PyResult<f64> {
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(v);
    }
    let mut m = f64::NEG_INFINITY;
    for item in obj.try_iter()? {
        let v = deep_max_f64(&item?)?;
        if v > m {
            m = v;
        }
    }
    Ok(m)
}

pub fn import_gym<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    py.import("gymnasium").or_else(|_| py.import("gym"))
}
