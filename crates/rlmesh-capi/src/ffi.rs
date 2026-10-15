//! Common binding conversions and callback ownership.
#![allow(unsafe_code)] // C input pointers and callback-owned handles.
use crate::abi::status::{
    CapiError, error_of_class, last_error_message, last_error_recoverable, take_callback_class,
};
use crate::codec::RLMeshBytes;
use crate::value::handle::{RLMeshValue, value_ref, write_out};
use rlmesh::{Error, Metadata, Value};
use std::ffi::{CStr, c_char};
use std::time::Duration;

pub(crate) unsafe fn text(ptr: *const c_char) -> Result<Option<String>, CapiError> {
    if ptr.is_null() {
        return Ok(None);
    }
    Ok(Some(
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .map_err(|_| CapiError::invalid_arg("text must be UTF-8"))?
            .to_owned(),
    ))
}
pub(crate) fn dict(ptr: *const RLMeshValue) -> Result<Option<Metadata>, CapiError> {
    if ptr.is_null() {
        Ok(None)
    } else {
        Ok(Some(value_ref(ptr)?.as_dict()?.clone()))
    }
}
pub(crate) fn bytes(out: *mut RLMeshBytes, text: String) -> Result<(), CapiError> {
    if out.is_null() {
        return Err(CapiError::invalid_arg("null out"));
    }
    write_out(out, RLMeshBytes::from_vec(text.into_bytes()))
}
pub(crate) fn duration(value: f64, name: &str) -> Result<Duration, CapiError> {
    Duration::try_from_secs_f64(value).map_err(|_| {
        CapiError::invalid_arg(format!(
            "{name} must be finite, non-negative and representable"
        ))
    })
}
pub(crate) fn callback_error(kind: &str) -> Error {
    let message = last_error_message();
    let message = if message.is_empty() {
        format!("{kind} callback failed")
    } else {
        message
    };
    if let Some(class) = take_callback_class() {
        return error_of_class(class, &message);
    }
    match (kind, last_error_recoverable()) {
        ("env", true) => Error::environment_recoverable(message),
        ("env", false) => Error::environment(message),
        (_, true) => Error::model_recoverable(message),
        (_, false) => Error::model(message),
    }
}
/// Take a callback out-handle even on failure; NULL is an absent value.
pub(crate) unsafe fn take(ptr: *mut RLMeshValue) -> Option<Value> {
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { Box::from_raw(ptr) }.value().clone())
    }
}
pub(crate) fn info(value: Option<Value>) -> rlmesh::Result<Metadata> {
    value.map_or_else(|| Ok(Metadata::new()), |v| v.as_dict().cloned())
}
