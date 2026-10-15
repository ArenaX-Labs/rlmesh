//! Opaque tensors backed by the Rust facade; all storage stays read-only.
#![allow(unsafe_code)] // FFI: raw pointers and DLPack ownership transfer.

use super::dtype::{dtype_from_c, dtype_to_c};
use super::handle::write_out;
use crate::abi::status::{CapiError, RLMeshStatus, guard, guard_ptr, guard_value};
use crate::codec::RLMeshBytes;
use crate::spaces::slice_arg;
use rlmesh::Tensor;
use std::ffi::c_void;

pub struct RLMeshTensor(pub(crate) Tensor);
/// # Safety
/// A non-NULL tensor handle stays live for the returned reference's lifetime.
pub(crate) unsafe fn tensor_ref<'a>(tensor: *const RLMeshTensor) -> Result<&'a Tensor, CapiError> {
    Ok(&unsafe { tensor.as_ref() }
        .ok_or_else(|| CapiError::invalid_arg("null tensor"))?
        .0)
}
pub(crate) fn output(out: *mut *mut RLMeshTensor, tensor: Tensor) -> Result<(), CapiError> {
    if out.is_null() {
        return Err(CapiError::invalid_arg("null out"));
    }
    unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(out, Box::into_raw(Box::new(RLMeshTensor(tensor))))
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_tensor_new(
    buffer: *const c_void,
    buffer_len: usize,
    shape: *const usize,
    ndim: usize,
    dtype: i32,
    out: *mut *mut RLMeshTensor,
) -> RLMeshStatus {
    guard(|| {
        let dtype =
            dtype_from_c(dtype).ok_or_else(|| CapiError::invalid_arg("invalid tensor dtype"))?;
        let buffer = unsafe { slice_arg(buffer.cast::<u8>(), buffer_len, "buffer") }?;
        let shape = unsafe { slice_arg(shape, ndim, "shape") }?;
        output(out, Tensor::new(buffer, shape, dtype)?)
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_tensor_clone(tensor: *const RLMeshTensor) -> *mut RLMeshTensor {
    guard_ptr(|| {
        Ok(Box::into_raw(Box::new(RLMeshTensor(
            unsafe {
                // SAFETY: the FFI contract supplies live handles and writable output storage.
                tensor_ref(tensor)
            }?
            .clone(),
        ))))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_tensor_free(tensor: *mut RLMeshTensor) {
    guard_value((), || {
        if !tensor.is_null() {
            drop(unsafe { Box::from_raw(tensor) });
        }
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_tensor_shape(
    tensor: *const RLMeshTensor,
    out: *mut usize,
    cap: usize,
    out_len: *mut usize,
) -> RLMeshStatus {
    guard(|| {
        crate::space_handle::copy_buffer(
            unsafe {
                // SAFETY: the FFI contract supplies live handles and writable output storage.
                tensor_ref(tensor)
            }?
            .shape(),
            out,
            cap,
            out_len,
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_tensor_strides(
    tensor: *const RLMeshTensor,
    out: *mut usize,
    cap: usize,
    out_len: *mut usize,
) -> RLMeshStatus {
    guard(|| {
        crate::space_handle::copy_buffer(
            &unsafe {
                // SAFETY: the FFI contract supplies live handles and writable output storage.
                tensor_ref(tensor)
            }?
            .strides(),
            out,
            cap,
            out_len,
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_tensor_dtype(tensor: *const RLMeshTensor) -> i32 {
    guard_value(0, || {
        unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            tensor_ref(tensor)
        }
        .map_or(0, |t| dtype_to_c(t.dtype()))
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_tensor_device(tensor: *const RLMeshTensor) -> i32 {
    guard_value(0, || {
        unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            tensor_ref(tensor)
        }
        .map_or(0, |_| 1)
    })
}
macro_rules! size_read {
    ($name:ident, $method:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(tensor: *const RLMeshTensor, out: *mut usize) -> RLMeshStatus {
            guard(|| unsafe {
                // SAFETY: the FFI contract supplies live handles and writable output storage.
                write_out(out, tensor_ref(tensor)?.$method())
            })
        }
    };
}
size_read!(rlmesh_tensor_ndim, ndim);
size_read!(rlmesh_tensor_size, size);
size_read!(rlmesh_tensor_nbytes, nbytes);
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_tensor_buffer(
    tensor: *const RLMeshTensor,
    out_data: *mut *const c_void,
    out_len: *mut usize,
) -> RLMeshStatus {
    guard(|| {
        let data = unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            tensor_ref(tensor)
        }?
        .buffer();
        if out_data.is_null() || out_len.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out_data, data.as_ptr().cast())
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out_len, data.len())
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_tensor_is_contiguous(tensor: *const RLMeshTensor) -> bool {
    guard_value(false, || {
        unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            tensor_ref(tensor)
        }
        .is_ok_and(Tensor::is_contiguous)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_tensor_reshape(
    tensor: *const RLMeshTensor,
    shape: *const i64,
    ndim: usize,
    out: *mut *mut RLMeshTensor,
) -> RLMeshStatus {
    guard(|| {
        output(
            out,
            unsafe {
                // SAFETY: the FFI contract supplies live handles and writable output storage.
                tensor_ref(tensor)
            }?
            .reshape(unsafe { slice_arg(shape, ndim, "shape") }?)?,
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_tensor_copy(
    tensor: *const RLMeshTensor,
    out: *mut *mut RLMeshTensor,
) -> RLMeshStatus {
    guard(|| {
        output(
            out,
            unsafe {
                // SAFETY: the FFI contract supplies live handles and writable output storage.
                tensor_ref(tensor)
            }?
            .copy(),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_tensor_tobytes(
    tensor: *const RLMeshTensor,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, RLMeshBytes::from_vec(tensor_ref(tensor)?.tobytes()))
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_tensor_repr(
    tensor: *const RLMeshTensor,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                RLMeshBytes::from_vec(tensor_ref(tensor)?.to_string().into_bytes()),
            )
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_device_str(device: i32) -> *const std::ffi::c_char {
    if device == 1 {
        super::dtype::static_str("cpu")
    } else {
        std::ptr::null()
    }
}
