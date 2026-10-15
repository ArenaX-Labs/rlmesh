//! Callback error reporting shared by all ABI 5 vtables.
#![allow(unsafe_code)]
use crate::abi::status::RLMeshStatus;
use std::ffi::{CStr, c_char};

/// Set the current callback's error message + recoverability (read by the capi
/// on this thread when the callback returns nonzero).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_callback_set_error(message: *const c_char, recoverable: bool) {
    let message = if message.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    };
    // A callback failure surfaces as `Error::Model`, so record the status the
    // outermost export would map it to (the callback has no status channel).
    crate::abi::status::store_last_error(&message, recoverable, RLMeshStatus::Model);
}

/// Set the current callback's error as an exact Python class (R-12), as a
/// Python author raises `ValueError` or `TypeError` from a hook. An
/// Environment or Model class carries code INTERNAL and is recoverable only as
/// `RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION`; INVALID or an
/// unknown value records `RLMeshException`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_callback_set_error_class(error_class: i32, message: *const c_char) {
    let message = if message.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    };
    crate::abi::status::store_last_error_class(&message, error_class);
}
