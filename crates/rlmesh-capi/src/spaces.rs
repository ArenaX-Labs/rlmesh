//! Immutable space spec handles and checked FFI slices.
#![allow(unsafe_code)]
use crate::abi::status::{CapiError, guard_value};
use rlmesh_spaces::SpaceSpec;

/// An opaque space specification.
#[repr(transparent)]
pub struct RLMeshSpaceSpec(pub(crate) SpaceSpec);

pub(crate) fn spec_ref<'a>(spec: *const RLMeshSpaceSpec) -> Option<&'a SpaceSpec> {
    unsafe { spec.cast::<SpaceSpec>().as_ref() }
}

/// # Safety
/// `ptr` is NULL only when `len == 0`; otherwise it points at `len` readable `T`s.
pub(crate) unsafe fn slice_arg<'a, T>(
    ptr: *const T,
    len: usize,
    what: &str,
) -> Result<&'a [T], CapiError> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(CapiError::invalid_arg(format!("null {what}")));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

///
/// # Safety
/// `spec` is NULL or an owned handle this thread has not freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_space_spec_free(spec: *mut RLMeshSpaceSpec) {
    guard_value((), || {
        if !spec.is_null() {
            drop(unsafe { Box::from_raw(spec) });
        }
    });
}
