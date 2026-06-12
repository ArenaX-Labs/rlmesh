// The Python buffer protocol (`__getbuffer__`/`__releasebuffer__`) is a raw FFI
// contract over `Py_buffer`, so this module needs `unsafe`. It is the only place
// in the crate that does; the workspace otherwise denies `unsafe_code`.
#![allow(unsafe_code)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyMemoryView};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rlmesh_spaces::{DType, Tensor, TensorError, dtype_size};
use std::ffi::{CString, c_int, c_void};
use std::ptr;

use super::utils::{dtype_name, parse_dtype_strict};

/// Heap state kept alive for the duration of an exported buffer view,
/// referenced from `Py_buffer.internal` and freed in `__releasebuffer__`.
struct ViewState {
    shape: Vec<isize>,
    strides: Vec<isize>,
    format: Option<CString>,
}

/// Python `struct`-module format code for a dtype. `bfloat16` has no code.
///
/// 64-bit integers use `"q"`/`"Q"`: the `"l"`/`"L"` codes are platform
/// `long`, which is 32-bit on LLP64 targets such as Windows.
fn buffer_format(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::Bool => Some("?"),
        DType::Int8 => Some("b"),
        DType::Uint8 => Some("B"),
        DType::Int16 => Some("h"),
        DType::Uint16 => Some("H"),
        DType::Int32 => Some("i"),
        DType::Uint32 => Some("I"),
        DType::Int64 => Some("q"),
        DType::Uint64 => Some("Q"),
        DType::Float16 => Some("e"),
        DType::Float32 => Some("f"),
        DType::Float64 => Some("d"),
        DType::Bfloat16 | DType::Unspecified => None,
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
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
    /// Copies `data` into fresh 64-byte-aligned storage so DLPack consumers
    /// with alignment requirements (modern XLA) can share it zero-copy.
    fn from_parts(data: &[u8], shape: Vec<usize>, dtype: String) -> PyResult<Self> {
        let dtype_id = parse_dtype_strict(&dtype)?;
        let dims: Vec<i64> = shape.iter().map(|&dim| dim as i64).collect();
        match Tensor::from_slice(data, &dims, dtype_id) {
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

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (buffer, shape, dtype))]
    fn new(
        #[gen_stub(override_type(type_repr = "object", imports = ()))] buffer: &Bound<'_, PyAny>,
        shape: Vec<usize>,
        dtype: String,
    ) -> PyResult<Self> {
        // Borrow bytes objects directly so construction costs exactly one
        // (aligned) copy; other buffer-likes go through tobytes() first.
        if let Ok(bytes) = buffer.cast::<pyo3::types::PyBytes>() {
            return Self::from_parts(bytes.as_bytes(), shape, dtype);
        }
        Self::from_parts(&extract_buffer_bytes(buffer)?, shape, dtype)
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

    /// A tensor with the same elements and a new shape. One dimension may
    /// be ``-1`` to infer its size from the element count. Shares the
    /// underlying data when this tensor is contiguous.
    fn reshape(&self, shape: Vec<i64>) -> PyResult<Self> {
        self.inner
            .reshape(&shape)
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

    /// Export the tensor as a DLPack capsule.
    ///
    /// With `max_version` of `(1, 0)` or newer the capsule is a DLPack 1.0
    /// `DLManagedTensorVersioned` flagged read-only; otherwise it is a
    /// legacy `DLManagedTensor`. `copy=True` exports a fresh writable
    /// buffer. Only `stream=None` and CPU `dl_device` are accepted.
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    #[gen_stub(override_return_type(type_repr = "object", imports = ()))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "object | None", imports = ()))] stream: Option<
            Bound<'py, PyAny>,
        >,
        max_version: Option<(i64, i64)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if stream.is_some() {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "stream must be None for CPU tensors",
            ));
        }
        if let Some(device) = dl_device
            && device != (i32::from(self.inner.device()), 0)
        {
            return Err(pyo3::exceptions::PyBufferError::new_err(format!(
                "cannot export to device {device:?}; only CPU (1, 0) is supported",
            )));
        }

        let copied;
        let (tensor, is_copy) = if copy == Some(true) {
            copied = Tensor::from_slice(
                &self.inner.to_contiguous_bytes(),
                self.inner.shape(),
                self.inner.dtype(),
            )
            .map_err(|err| pyo3::exceptions::PyBufferError::new_err(err.to_string()))?;
            (&copied, true)
        } else {
            (&self.inner, false)
        };

        match max_version {
            // A fresh copy is exclusively owned by the consumer, so it is
            // exported writable; shared exports are flagged read-only.
            Some((major, _)) if major >= 1 => super::dlpack::export_versioned(py, tensor, !is_copy),
            _ => super::dlpack::export_legacy(py, tensor),
        }
    }

    /// DLPack device of the tensor data: `(kDLCPU, 0)`.
    fn __dlpack_device__(&self) -> (i32, i32) {
        (i32::from(self.inner.device()), 0)
    }

    /// Import a tensor from a DLPack capsule or any object implementing
    /// `__dlpack__`. Accepts both legacy and versioned capsules; the
    /// elements are always copied into fresh storage and the source
    /// capsule is consumed.
    #[staticmethod]
    fn from_dlpack(
        #[gen_stub(override_type(type_repr = "object", imports = ()))] obj: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        super::dlpack::import_tensor(obj).map(Self::from)
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
        // On failure the exporter must leave view.obj null.
        unsafe {
            (*view).obj = ptr::null_mut();
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Tensor buffer is read-only",
            ));
        }

        let borrowed = slf.borrow();
        let dtype = borrowed.inner.dtype();
        let Some(format) = buffer_format(dtype) else {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "bfloat16 tensors do not support the buffer protocol; use __dlpack__ or tobytes()",
            ));
        };
        if !borrowed.inner.is_contiguous()
            && (flags & pyo3::ffi::PyBUF_STRIDES) != pyo3::ffi::PyBUF_STRIDES
        {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "tensor is not C-contiguous; the buffer request requires strides",
            ));
        }

        let item_size = dtype_size(dtype) as isize;
        let state = Box::new(ViewState {
            shape: borrowed
                .inner
                .shape()
                .iter()
                .map(|&dim| dim as isize)
                .collect(),
            strides: borrowed
                .inner
                .effective_strides()
                .iter()
                .map(|&stride| stride as isize * item_size)
                .collect(),
            format: if (flags & pyo3::ffi::PyBUF_FORMAT) == pyo3::ffi::PyBUF_FORMAT {
                Some(CString::new(format).expect("static format"))
            } else {
                None
            },
        });

        let data_len = borrowed.inner.nbytes();
        let storage = borrowed.inner.storage().as_slice();
        let data_ptr = if data_len == 0 {
            storage.as_ptr()
        } else {
            storage[borrowed.inner.byte_offset()..].as_ptr()
        };
        let ndim = borrowed.inner.shape().len();
        drop(borrowed);
        unsafe {
            (*view).buf = data_ptr as *mut c_void;
            (*view).len = data_len as isize;
            (*view).readonly = 1;
            (*view).itemsize = item_size;
            (*view).format = state
                .format
                .as_ref()
                .map_or(ptr::null_mut(), |format| format.as_ptr() as *mut _);
            // Without PyBUF_ND the protocol requires shape = NULL, and the
            // consumer treats the buffer as a 1-D byte stream.
            if (flags & pyo3::ffi::PyBUF_ND) == pyo3::ffi::PyBUF_ND {
                (*view).ndim = ndim as c_int;
                (*view).shape = state.shape.as_ptr() as *mut isize;
            } else {
                (*view).ndim = 1;
                (*view).shape = ptr::null_mut();
            }
            (*view).strides = if (flags & pyo3::ffi::PyBUF_STRIDES) == pyo3::ffi::PyBUF_STRIDES {
                state.strides.as_ptr() as *mut isize
            } else {
                ptr::null_mut()
            };
            (*view).suboffsets = ptr::null_mut();
            (*view).internal = Box::into_raw(state) as *mut c_void;
            (*view).obj = slf.into_any().into_ptr();
        }
        Ok(())
    }

    #[gen_stub(skip)]
    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        unsafe {
            if view.is_null() {
                return;
            }
            let state = (*view).internal as *mut ViewState;
            if !state.is_null() {
                drop(Box::from_raw(state));
                (*view).internal = ptr::null_mut();
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
        Py::new(py, PyTensor::from_parts(&data, shape, dtype.into())?)?
            .into_bound(py)
            .into_any(),
    )
}

/// Wrap an existing native tensor without copying; the Python object shares
/// the tensor's storage (and its alignment).
pub(crate) fn wrap_native_tensor<'py>(
    py: Python<'py>,
    tensor: Tensor,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(Py::new(py, PyTensor::from(tensor))?
        .into_bound(py)
        .into_any())
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
