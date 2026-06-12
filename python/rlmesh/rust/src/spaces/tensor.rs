// The Python buffer protocol (`__getbuffer__`/`__releasebuffer__`) is a raw FFI
// contract over `Py_buffer`, so this module needs `unsafe`. It is the only place
// in the crate that does; the workspace otherwise denies `unsafe_code`.
#![allow(unsafe_code)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyMemoryView};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rlmesh_spaces::v1::{Tensor, TensorError, dtype_size};
use std::ffi::{CString, c_int, c_void};
use std::ptr;

use super::utils::{dtype_name, parse_dtype_strict};

#[gen_stub_pyclass]
#[pyclass(
    module = "rlmesh._rlmesh",
    name = "Tensor",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyTensor {
    pub(crate) inner: Tensor,
}

impl PyTensor {
    fn from_parts(data: Vec<u8>, shape: Vec<usize>, dtype: String) -> PyResult<Self> {
        let dtype_id = parse_dtype_strict(&dtype)?;
        let dims: Vec<i64> = shape.iter().map(|&dim| dim as i64).collect();
        match Tensor::from_vec(data, dims, dtype_id) {
            Ok(inner) => Ok(Self { inner }),
            Err(TensorError::ByteLengthMismatch { expected, actual }) => {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "tensor byte length mismatch: expected {expected} bytes for shape {shape:?} and dtype {dtype:?}, got {actual}"
                )))
            }
            Err(TensorError::Overflow) => Err(pyo3::exceptions::PyOverflowError::new_err(
                "tensor element count overflow",
            )),
            Err(err) => Err(pyo3::exceptions::PyValueError::new_err(err.to_string())),
        }
    }

    fn shape_usize(&self) -> Vec<usize> {
        self.inner.shape().iter().map(|&dim| dim as usize).collect()
    }
}

impl From<Tensor> for PyTensor {
    fn from(inner: Tensor) -> Self {
        Self { inner }
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
        self.shape_usize()
    }

    #[getter]
    fn dtype(&self) -> String {
        dtype_name(self.inner.dtype()).to_string()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.shape().len()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.numel()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    /// Strides in bytes per dimension, C-order.
    #[getter]
    fn strides(&self) -> Vec<usize> {
        let item_size = dtype_size(self.inner.dtype());
        self.inner
            .effective_strides()
            .iter()
            .map(|&stride| stride as usize * item_size)
            .collect()
    }

    /// Device holding the tensor data. Always `"cpu"`.
    #[getter]
    fn device(&self) -> &'static str {
        "cpu"
    }

    /// Whether the elements are laid out C-contiguously.
    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    /// A tensor with the same elements and a new shape. Shares the
    /// underlying data when this tensor is contiguous.
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        let dims: Vec<i64> = shape.iter().map(|&dim| dim as i64).collect();
        self.inner
            .reshape(&dims)
            .map(Self::from)
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))
    }

    /// A deep copy backed by fresh storage.
    fn copy(&self) -> PyResult<Self> {
        Tensor::from_slice(
            &self.inner.to_contiguous_bytes(),
            self.inner.shape(),
            self.inner.dtype(),
        )
        .map(Self::from)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "memoryview", imports = ()))]
    fn buffer<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let owner = slf.into_any();
        Ok(PyMemoryView::from(&owner)?.into_any())
    }

    fn tobytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_contiguous_bytes())
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
        if !borrowed.inner.is_contiguous() {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Tensor buffer requires a contiguous layout; call copy() first",
            ));
        }
        let data_len = borrowed.inner.nbytes();
        let storage = borrowed.inner.storage().as_slice();
        let data_ptr = if data_len == 0 {
            storage.as_ptr()
        } else {
            storage[borrowed.inner.byte_offset()..].as_ptr()
        };
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
            self.dtype(),
            self.shape_usize(),
            self.inner.nbytes()
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
