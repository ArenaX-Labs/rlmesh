// DLPack is a raw C ABI: exporting requires #[repr(C)] structs, raw capsule
// pointers, and an extern "C" deleter, so this module allows `unsafe`.
//
// Invariants that keep the exchange sound:
// - The capsule pointer is a `DLManagedTensor`/`DLManagedTensorVersioned`
//   embedded at offset 0 of a boxed `ExportHolder`, whose `manager_ctx` is
//   the box pointer itself. shape/strides pointers point into Vecs owned by
//   the same holder, and `_storage` keeps the element bytes alive.
// - Exactly one party frees the holder: the consumer's call to `deleter`
//   after it renames the capsule to `used_*`, or the capsule destructor if
//   the capsule is dropped unconsumed (name still `dltensor*`).
// - The deleter only drops plain Rust data (Storage, Vecs). Consumers may
//   invoke it on any thread without the GIL, so the holder must never
//   contain `Py<T>` or other GIL-bound state.
#![allow(unsafe_code)]

use std::ffi::{CStr, c_void};

use pyo3::prelude::*;
use rlmesh_spaces::v1::{Storage, Tensor, dlpack_type};

pub(crate) static DLTENSOR_NAME: &CStr = c"dltensor";
pub(crate) static DLTENSOR_VERSIONED_NAME: &CStr = c"dltensor_versioned";

/// `DLPACK_FLAG_BITMASK_READ_ONLY` from dlpack.h.
const FLAG_READ_ONLY: u64 = 1;

#[repr(C)]
pub(crate) struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

#[repr(C)]
pub(crate) struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
pub(crate) struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
pub(crate) struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
pub(crate) struct DLPackVersion {
    pub major: u32,
    pub minor: u32,
}

#[repr(C)]
pub(crate) struct DLManagedTensorVersioned {
    pub version: DLPackVersion,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    pub flags: u64,
    pub dl_tensor: DLTensor,
}

/// Owns everything a live export needs: the managed-tensor struct handed to
/// the consumer (at offset 0), the storage keeping the bytes alive, and the
/// shape/strides arrays the `DLTensor` points into.
#[repr(C)]
struct ExportHolder<M> {
    managed: M,
    _storage: Storage,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

/// Export `tensor` as a legacy `"dltensor"` capsule.
pub(crate) fn export_legacy<'py>(py: Python<'py>, tensor: &Tensor) -> PyResult<Bound<'py, PyAny>> {
    let holder = new_holder(
        tensor,
        DLManagedTensor {
            dl_tensor: empty_dl_tensor(),
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(legacy_deleter),
        },
    )?;
    let raw = Box::into_raw(holder);
    unsafe {
        (*raw).managed.dl_tensor = dl_tensor_for(raw, tensor);
        (*raw).managed.manager_ctx = raw as *mut c_void;
        new_capsule(
            py,
            &raw mut (*raw).managed as *mut c_void,
            DLTENSOR_NAME,
            legacy_capsule_destructor,
            || legacy_deleter(&raw mut (*raw).managed),
        )
    }
}

/// Export `tensor` as a versioned `"dltensor_versioned"` (DLPack 1.0)
/// capsule. `read_only` sets `DLPACK_FLAG_BITMASK_READ_ONLY`.
pub(crate) fn export_versioned<'py>(
    py: Python<'py>,
    tensor: &Tensor,
    read_only: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let holder = new_holder(
        tensor,
        DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 0 },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(versioned_deleter),
            flags: if read_only { FLAG_READ_ONLY } else { 0 },
            dl_tensor: empty_dl_tensor(),
        },
    )?;
    let raw = Box::into_raw(holder);
    unsafe {
        (*raw).managed.dl_tensor = dl_tensor_for(raw, tensor);
        (*raw).managed.manager_ctx = raw as *mut c_void;
        new_capsule(
            py,
            &raw mut (*raw).managed as *mut c_void,
            DLTENSOR_VERSIONED_NAME,
            versioned_capsule_destructor,
            || versioned_deleter(&raw mut (*raw).managed),
        )
    }
}

fn new_holder<M>(tensor: &Tensor, managed: M) -> PyResult<Box<ExportHolder<M>>> {
    if dlpack_type(tensor.dtype()).is_none() {
        return Err(pyo3::exceptions::PyBufferError::new_err(format!(
            "dtype {:?} has no DLPack representation",
            tensor.dtype()
        )));
    }
    Ok(Box::new(ExportHolder {
        managed,
        _storage: tensor.storage().clone(),
        shape: tensor.shape().to_vec(),
        strides: tensor.effective_strides().into_owned(),
    }))
}

fn empty_dl_tensor() -> DLTensor {
    DLTensor {
        data: std::ptr::null_mut(),
        device: DLDevice {
            device_type: 1,
            device_id: 0,
        },
        ndim: 0,
        dtype: DLDataType {
            code: 0,
            bits: 0,
            lanes: 0,
        },
        shape: std::ptr::null_mut(),
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    }
}

/// Build the `DLTensor` view for a holder that has already been moved to its
/// final heap address. The byte offset is folded into the data pointer.
unsafe fn dl_tensor_for<M>(raw: *mut ExportHolder<M>, tensor: &Tensor) -> DLTensor {
    let ty = dlpack_type(tensor.dtype()).expect("checked in new_holder");
    unsafe {
        DLTensor {
            data: (*raw)
                ._storage
                .as_slice()
                .as_ptr()
                .wrapping_add(tensor.byte_offset()) as *mut c_void,
            device: DLDevice {
                device_type: i32::from(tensor.device()),
                device_id: 0,
            },
            ndim: (*raw).shape.len() as i32,
            dtype: DLDataType {
                code: ty.code,
                bits: ty.bits,
                lanes: ty.lanes,
            },
            shape: (*raw).shape.as_mut_ptr(),
            strides: (*raw).strides.as_mut_ptr(),
            byte_offset: 0,
        }
    }
}

/// Wrap a managed-tensor pointer in a capsule; on capsule-creation failure
/// run `cleanup` so the holder is not leaked.
unsafe fn new_capsule<'py>(
    py: Python<'py>,
    pointer: *mut c_void,
    name: &'static CStr,
    destructor: unsafe extern "C" fn(*mut pyo3::ffi::PyObject),
    cleanup: impl FnOnce(),
) -> PyResult<Bound<'py, PyAny>> {
    unsafe {
        let capsule = pyo3::ffi::PyCapsule_New(pointer, name.as_ptr(), Some(destructor));
        if capsule.is_null() {
            cleanup();
            return Err(PyErr::fetch(py));
        }
        Ok(Bound::from_owned_ptr(py, capsule))
    }
}

unsafe extern "C" fn legacy_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    unsafe {
        let holder = (*managed).manager_ctx as *mut ExportHolder<DLManagedTensor>;
        drop(Box::from_raw(holder));
    }
}

unsafe extern "C" fn versioned_deleter(managed: *mut DLManagedTensorVersioned) {
    if managed.is_null() {
        return;
    }
    unsafe {
        let holder = (*managed).manager_ctx as *mut ExportHolder<DLManagedTensorVersioned>;
        drop(Box::from_raw(holder));
    }
}

unsafe extern "C" fn legacy_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        // A renamed (`used_dltensor`) capsule belongs to the consumer; only
        // free when the export was never consumed.
        let pointer = pyo3::ffi::PyCapsule_GetPointer(capsule, DLTENSOR_NAME.as_ptr());
        if pointer.is_null() {
            pyo3::ffi::PyErr_Clear();
            return;
        }
        let managed = pointer as *mut DLManagedTensor;
        if let Some(deleter) = (*managed).deleter {
            deleter(managed);
        }
    }
}

unsafe extern "C" fn versioned_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        let pointer = pyo3::ffi::PyCapsule_GetPointer(capsule, DLTENSOR_VERSIONED_NAME.as_ptr());
        if pointer.is_null() {
            pyo3::ffi::PyErr_Clear();
            return;
        }
        let managed = pointer as *mut DLManagedTensorVersioned;
        if let Some(deleter) = (*managed).deleter {
            deleter(managed);
        }
    }
}
