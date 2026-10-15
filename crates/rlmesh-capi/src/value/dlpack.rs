//! DLPack 1.x exchange through the facade, with success-only ownership transfer.
#![allow(unsafe_code)] // Foreign descriptors and output pointers follow SPC-19.

use super::tensor::{RLMeshTensor, output, tensor_ref};
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard},
};
use crate::value::handle::write_out;
use rlmesh::ext::dlpack::DLManagedTensorVersioned;
use rlmesh::{Tensor, ToDlpackOptions};
use std::ptr::{NonNull, null_mut};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshToDlpackOptions {
    pub struct_size: usize,
    pub copy: bool,
    pub has_dl_device: bool,
    pub _reserved0: u16,
    pub dl_device_type: i32,
    pub dl_device_id: i32,
    pub _reserved1: u32,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_tensor_from_dlpack(
    obj: *mut DLManagedTensorVersioned,
    out: *mut *mut RLMeshTensor,
) -> RLMeshStatus {
    guard(|| {
        // Validate the output before the facade can consume the producer.
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, null_mut())
        }?;
        let obj = NonNull::new(obj).ok_or_else(|| CapiError::invalid_arg("null DLPack tensor"))?;
        // SAFETY: the C caller supplies an exclusively owned live descriptor
        // and readable CPU data/metadata. The facade copies and releases it
        // only after validation succeeds; errors leave ownership with C.
        let tensor = unsafe { Tensor::from_dlpack(obj) }?;
        output(out, tensor)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_tensor_to_dlpack(
    tensor: *const RLMeshTensor,
    options: *const RLMeshToDlpackOptions,
    out: *mut *mut DLManagedTensorVersioned,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, null_mut())
        }?;
        let options = match unsafe { sizing::read(options, "RLMeshToDlpackOptions") }? {
            None => ToDlpackOptions::default(),
            Some(o) => {
                if o._reserved0 != 0 || o._reserved1 != 0 {
                    return Err(CapiError::type_error(
                        "RLMeshToDlpackOptions reserved bytes must be zero",
                    ));
                }
                ToDlpackOptions {
                    copy: o.copy,
                    dl_device: o
                        .has_dl_device
                        .then_some((o.dl_device_type, o.dl_device_id)),
                }
            }
        };
        let export = unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            tensor_ref(tensor)
        }?
        .to_dlpack(options)?;
        // Only the successful output gives up the facade's deleter-owning RAII
        // handle. No fallible operations remain after into_raw.
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, export.into_raw().as_ptr())
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_tensor_dlpack_device(
    tensor: *const RLMeshTensor,
    out_type: *mut i32,
    out_id: *mut i32,
) -> RLMeshStatus {
    guard(|| {
        if out_type.is_null() || out_id.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let (device_type, device_id) = unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            tensor_ref(tensor)
        }?
        .dlpack_device();
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out_type, device_type)
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out_id, device_id)
        }
    })
}
