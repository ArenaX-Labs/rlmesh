use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyMemoryView};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::ffi::{CString, c_int, c_void};
use std::ptr;

#[gen_stub_pyclass]
#[pyclass(
    module = "rlmesh._rlmesh",
    name = "Tensor",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyTensor {
    pub(crate) data: Vec<u8>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: String,
}

impl PyTensor {
    fn from_parts(data: Vec<u8>, shape: Vec<usize>, dtype: String) -> PyResult<Self> {
        let expected_len = expected_nbytes(&shape, &dtype)?;
        if data.len() != expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "tensor byte length mismatch: expected {expected_len} bytes for shape {shape:?} and dtype {dtype:?}, got {}",
                data.len()
            )));
        }
        Ok(Self { data, shape, dtype })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (buffer, shape, dtype))]
    fn new(
        #[gen_stub(override_type(type_repr = "object", imports = ()))] buffer: &Bound<'_, PyAny>,
        shape: Vec<usize>,
        dtype: String,
    ) -> PyResult<Self> {
        Self::from_parts(extract_buffer_bytes(buffer)?, shape, dtype)
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[getter]
    fn dtype(&self) -> String {
        self.dtype.clone()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[getter]
    fn size(&self) -> usize {
        element_count(&self.shape)
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.data.len()
    }

    #[getter]
    fn strides(&self) -> PyResult<Vec<usize>> {
        c_contiguous_strides(&self.shape, dtype_size(&self.dtype)?)
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "memoryview", imports = ()))]
    fn buffer<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let owner = slf.into_any();
        Ok(PyMemoryView::from(&owner)?.into_any())
    }

    fn tobytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.data)
    }

    #[gen_stub(skip)]
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(pyo3::exceptions::PyBufferError::new_err("view is null"));
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Tensor buffer is read-only",
            ));
        }

        let borrowed = slf.borrow();
        let data_ptr = borrowed.data.as_ptr();
        let data_len = borrowed.data.len();
        drop(borrowed);
        unsafe {
            (*view).obj = slf.into_any().into_ptr();
            (*view).buf = data_ptr as *mut c_void;
            (*view).len = data_len as isize;
            (*view).readonly = 1;
            (*view).itemsize = 1;
            (*view).format = if (flags & pyo3::ffi::PyBUF_FORMAT) == pyo3::ffi::PyBUF_FORMAT {
                CString::new("B").expect("static format").into_raw()
            } else {
                ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = if (flags & pyo3::ffi::PyBUF_ND) == pyo3::ffi::PyBUF_ND {
                &mut (*view).len
            } else {
                ptr::null_mut()
            };
            (*view).strides = if (flags & pyo3::ffi::PyBUF_STRIDES) == pyo3::ffi::PyBUF_STRIDES {
                &mut (*view).itemsize
            } else {
                ptr::null_mut()
            };
            (*view).suboffsets = ptr::null_mut();
            (*view).internal = ptr::null_mut();
        }
        Ok(())
    }

    #[gen_stub(skip)]
    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        unsafe {
            if !view.is_null() && !(*view).format.is_null() {
                drop(CString::from_raw((*view).format));
                (*view).format = ptr::null_mut();
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(dtype={:?}, shape={:?}, bytes={})",
            self.dtype,
            self.shape,
            self.data.len()
        )
    }
}

pub(crate) fn make_tensor<'py>(
    py: Python<'py>,
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: impl Into<String>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(
        Py::new(py, PyTensor::from_parts(data, shape, dtype.into())?)?
            .into_bound(py)
            .into_any(),
    )
}

pub(crate) fn extract_tensor<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<Option<PyRef<'py, PyTensor>>> {
    match value.extract::<PyRef<'py, PyTensor>>() {
        Ok(leaf) => Ok(Some(leaf)),
        Err(_) => Ok(None),
    }
}

pub(crate) fn extract_buffer_bytes(value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = value.extract::<Vec<u8>>() {
        return Ok(bytes);
    }
    if value.hasattr("tobytes")? {
        return value.call_method0("tobytes")?.extract::<Vec<u8>>();
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a Tensor, bytes-like object, or value with tobytes()",
    ))
}

fn element_count(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().copied().product()
    }
}

fn expected_nbytes(shape: &[usize], dtype: &str) -> PyResult<usize> {
    element_count_checked(shape)?
        .checked_mul(dtype_size(dtype)?)
        .ok_or_else(|| pyo3::exceptions::PyOverflowError::new_err("tensor byte length overflow"))
}

fn element_count_checked(shape: &[usize]) -> PyResult<usize> {
    if shape.is_empty() {
        return Ok(1);
    }
    shape.iter().try_fold(1usize, |count, dim| {
        count.checked_mul(*dim).ok_or_else(|| {
            pyo3::exceptions::PyOverflowError::new_err("tensor element count overflow")
        })
    })
}

fn c_contiguous_strides(shape: &[usize], item_size: usize) -> PyResult<Vec<usize>> {
    let mut stride = item_size;
    let mut strides = Vec::with_capacity(shape.len());
    for dim in shape.iter().rev() {
        strides.push(stride);
        stride = stride.checked_mul(*dim).ok_or_else(|| {
            pyo3::exceptions::PyOverflowError::new_err("tensor strides overflow usize")
        })?;
    }
    strides.reverse();
    Ok(strides)
}

fn dtype_size(dtype: &str) -> PyResult<usize> {
    match dtype {
        "bool" | "uint8" => Ok(1),
        "float16" => Ok(2),
        "int32" | "float32" => Ok(4),
        "int64" | "float64" => Ok(8),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported tensor dtype {other:?}"
        ))),
    }
}
