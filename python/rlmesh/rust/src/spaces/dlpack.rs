//! Python capsule negotiation around the shared Rust DLPack implementation.
#![allow(unsafe_code)]

use std::ffi::CStr;
use std::ptr::NonNull;

use pyo3::prelude::*;
use rlmesh::binding::dlpack::{DLManagedTensor, export_legacy, import_legacy};
use rlmesh::ext::dlpack::{DLManagedTensorVersioned, DlpackTensor};
use rlmesh_spaces::Tensor;

pub(crate) static DLTENSOR_NAME: &CStr = c"dltensor";
pub(crate) static DLTENSOR_VERSIONED_NAME: &CStr = c"dltensor_versioned";
pub(crate) static USED_DLTENSOR_NAME: &CStr = c"used_dltensor";
pub(crate) static USED_DLTENSOR_VERSIONED_NAME: &CStr = c"used_dltensor_versioned";

/// Wrap the facade's owned export in the capsule flavor requested by Python.
pub(crate) fn export<'py>(
    py: Python<'py>,
    tensor: &Tensor,
    options: rlmesh::ToDlpackOptions,
    versioned: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let facade = rlmesh::binding::tensor_from_core(tensor.clone())
        .map_err(|e| pyo3::exceptions::PyBufferError::new_err(e.to_string()))?;
    let owned = facade
        .to_dlpack(options)
        .map_err(|e| pyo3::exceptions::PyBufferError::new_err(e.to_string()))?;
    // SAFETY: ownership passes to the capsule on success. On allocation
    // failure, reconstruct the owner / call the legacy deleter exactly once.
    unsafe {
        let (ptr, name, destructor): (_, _, unsafe extern "C" fn(*mut pyo3::ffi::PyObject)) =
            if versioned {
                (
                    owned.into_raw().as_ptr().cast(),
                    DLTENSOR_VERSIONED_NAME,
                    versioned_capsule_destructor,
                )
            } else {
                (
                    export_legacy(owned).as_ptr().cast(),
                    DLTENSOR_NAME,
                    legacy_capsule_destructor,
                )
            };
        let capsule = pyo3::ffi::PyCapsule_New(ptr, name.as_ptr(), Some(destructor));
        if capsule.is_null() {
            if versioned {
                drop(DlpackTensor::from_raw(NonNull::new_unchecked(ptr.cast())));
            } else {
                let managed = ptr.cast::<DLManagedTensor>();
                if let Some(deleter) = (*managed).deleter {
                    deleter(managed);
                }
            }
            return Err(PyErr::fetch(py));
        }
        Ok(Bound::from_owned_ptr(py, capsule))
    }
}

/// Import either capsule flavor. The facade copies and consumes on success;
/// failed imports leave the capsule and its producer ownership unchanged.
pub(crate) fn import_tensor(obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let capsule = if is_capsule(obj, DLTENSOR_NAME) || is_capsule(obj, DLTENSOR_VERSIONED_NAME) {
        obj.clone()
    } else if obj.hasattr("__dlpack__")? {
        let capsule = obj.call_method0("__dlpack__")?;
        if !is_capsule(&capsule, DLTENSOR_NAME) && !is_capsule(&capsule, DLTENSOR_VERSIONED_NAME) {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "__dlpack__ did not return a DLPack capsule",
            ));
        }
        capsule
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a DLPack capsule or an object implementing __dlpack__",
        ));
    };
    let versioned = is_capsule(&capsule, DLTENSOR_VERSIONED_NAME);
    let (name, used) = if versioned {
        (DLTENSOR_VERSIONED_NAME, USED_DLTENSOR_VERSIONED_NAME)
    } else {
        (DLTENSOR_NAME, USED_DLTENSOR_NAME)
    };
    // SAFETY: valid capsules contain live managed tensors supplied by the
    // producer. Rename before invoking its deleter so no capsule destructor
    // can later free the consumed pointer. Restore the name on import error.
    unsafe {
        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), name.as_ptr());
        let Some(ptr) = NonNull::new(ptr) else {
            return Err(PyErr::fetch(capsule.py()));
        };
        if pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), used.as_ptr()) != 0 {
            return Err(PyErr::fetch(capsule.py()));
        }
        let result = if versioned {
            rlmesh::Tensor::from_dlpack(ptr.cast())
        } else {
            import_legacy(ptr.cast())
        };
        match result {
            Ok(tensor) => Ok(rlmesh::binding::tensor_core(&tensor).clone()),
            Err(err) => {
                if pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), name.as_ptr()) != 0 {
                    return Err(PyErr::fetch(capsule.py()));
                }
                Err(pyo3::exceptions::PyValueError::new_err(err.to_string()))
            }
        }
    }
}

fn is_capsule(obj: &Bound<'_, PyAny>, name: &CStr) -> bool {
    // SAFETY: obj is a live Python object and name is a static C string.
    unsafe { pyo3::ffi::PyCapsule_IsValid(obj.as_ptr(), name.as_ptr()) == 1 }
}

unsafe extern "C" fn legacy_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    // SAFETY: Python invokes this destructor on a live capsule. Only an
    // unconsumed capsule owns its managed tensor and may call its deleter.
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
    // SAFETY: as above; a consumed capsule never releases producer data.
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
