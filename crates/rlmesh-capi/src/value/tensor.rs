//! `RlmeshTensor` — the minimal DLPack-shaped POD tensor (data + shape + dtype).
//!
//! A tensor returned by value (e.g. from `rlmesh_value_as_tensor`) is a borrowed
//! view: `deleter == NULL`, valid only while its source `RlmeshValue` lives. A
//! by-value out-param tensor's `deleter` (when present) releases only
//! `manager_ctx` and never frees `self` — that diverges from DLPack, whose
//! "deleter frees self" applies only to the heap `DLManagedTensorVersioned`.
#![allow(unsafe_code)] // FFI: raw pointers + repr(C) struct.

use std::ffi::c_void;

use super::dtype::RlmeshDType;

/// Host CPU device (`kDLCPU`).
pub const RLMESH_DEVICE_CPU: i32 = 1;
/// The buffer must not be written.
pub const RLMESH_TENSOR_FLAG_READ_ONLY: u64 = 1;

/// A DLPack-shaped tensor view. `strides` is in element counts (NULL = row-major
/// contiguous), matching DLPack. `data` points at element 0 (offset folded in).
#[repr(C)]
pub struct RlmeshTensor {
    /// Element 0 (aligned, but do not assume 256-byte alignment).
    pub data: *mut c_void,
    pub ndim: i32,
    /// Dimension sizes, length `ndim`.
    pub shape: *const i64,
    /// Element-count strides, or NULL for row-major contiguous.
    pub strides: *const i64,
    pub dtype: RlmeshDType,
    pub device_type: i32,
    pub device_id: i32,
    pub flags: u64,
    /// Producer-owned context dropped by `deleter`; NULL for a borrowed view.
    pub manager_ctx: *mut c_void,
    /// Releases `manager_ctx` only (never frees `self`); NULL for a borrowed view.
    pub deleter: Option<unsafe extern "C" fn(*mut RlmeshTensor)>,
}

/// Release a tensor's backing resource. A no-op for a borrowed view (NULL
/// `deleter`). Never frees the `RlmeshTensor` itself.
///
/// # Safety
/// `tensor` must be a valid pointer to an `RlmeshTensor` this thread owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_tensor_release(tensor: *mut RlmeshTensor) {
    if tensor.is_null() {
        return;
    }
    if let Some(deleter) = unsafe { (*tensor).deleter } {
        unsafe { deleter(tensor) };
    }
}
