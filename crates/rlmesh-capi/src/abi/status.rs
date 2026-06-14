//! Status codes and by-value error marshaling.
//!
//! A callback runs on a tokio worker thread, not the export caller's, so a
//! thread-local set inside a callback is read on the wrong thread. Errors
//! therefore travel by value (`CapiError`) to the outermost export, which
//! writes the thread-local last-error slot on the caller's return thread as the
//! final hop only.
#![allow(unsafe_code)] // FFI: C string out-pointer + panic guard.

use std::any::Any;
use std::cell::RefCell;
use std::ffi::{CString, c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Result of a fallible C ABI call. Integer values are stable.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RlmeshStatus {
    /// Success.
    Ok = 0,
    /// A pointer/length/argument was invalid.
    InvalidArgument = 1,
    /// A value did not satisfy its space (shape/dtype/bounds).
    InvalidValue = 2,
    /// The environment reported a failure.
    Environment = 3,
    /// The model declined the request.
    Model = 4,
    /// A transport/connection/handshake failure.
    Transport = 5,
    /// An operation exceeded its deadline.
    Timeout = 6,
    /// A Rust panic was caught at the boundary.
    Panic = 7,
    /// An internal error that should not normally occur.
    Internal = 99,
}

/// An error carried by value from where it occurs to the caller's return thread.
pub(crate) struct CapiError {
    pub status: RlmeshStatus,
    pub message: String,
    pub recoverable: bool,
}

impl CapiError {
    pub(crate) fn new(status: RlmeshStatus, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            recoverable: false,
        }
    }
    pub(crate) fn invalid_arg(message: impl Into<String>) -> Self {
        Self::new(RlmeshStatus::InvalidArgument, message)
    }
    pub(crate) fn invalid_value(message: impl Into<String>) -> Self {
        Self::new(RlmeshStatus::InvalidValue, message)
    }
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(RlmeshStatus::Internal, message)
    }
}

impl From<rlmesh::Error> for CapiError {
    fn from(err: rlmesh::Error) -> Self {
        let recoverable = err.is_recoverable();
        let status = match &err {
            rlmesh::Error::Address(_) => RlmeshStatus::InvalidArgument,
            rlmesh::Error::Connection(_) | rlmesh::Error::Server(_) => RlmeshStatus::Transport,
            rlmesh::Error::Timeout(_) => RlmeshStatus::Timeout,
            rlmesh::Error::Environment(_) => RlmeshStatus::Environment,
            rlmesh::Error::Model(_) => RlmeshStatus::Model,
            _ => RlmeshStatus::Internal,
        };
        Self {
            status,
            message: err.to_string(),
            recoverable,
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<(CString, bool)>> = const { RefCell::new(None) };
}

/// Store an error in the calling thread's last-error slot (final hop only).
pub(crate) fn store_last_error(message: &str, recoverable: bool) {
    let safe: Vec<u8> = message.bytes().filter(|&b| b != 0).collect();
    let cstr = CString::new(safe).unwrap_or_default();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some((cstr, recoverable)));
}

/// Clear this thread's last-error slot. Called before invoking a callback so a
/// decline that does not set an error cannot report a previous request's stale
/// message on a reused pool thread (20-review-spec B3).
pub(crate) fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Read this thread's last-error message, or an empty string if none. Used to
/// fold a callback's error into an `Error` value on the callback's own thread.
pub(crate) fn last_error_message() -> String {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|(cstr, _)| cstr.to_string_lossy().into_owned())
            .unwrap_or_default()
    })
}

/// Whether this thread's last error is marked recoverable.
pub(crate) fn last_error_recoverable() -> bool {
    LAST_ERROR.with(|slot| slot.borrow().as_ref().is_some_and(|(_, r)| *r))
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic in rlmesh-capi".to_string()
    }
}

/// Run a fallible FFI body with a panic guard, writing the last-error slot on
/// this thread and returning a status. Use for status-returning exports.
pub(crate) fn guard<F>(f: F) -> RlmeshStatus
where
    F: FnOnce() -> Result<(), CapiError>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => RlmeshStatus::Ok,
        Ok(Err(err)) => {
            store_last_error(&err.message, err.recoverable);
            err.status
        }
        Err(payload) => {
            store_last_error(&panic_message(payload), false);
            RlmeshStatus::Panic
        }
    }
}

/// Run a panic-guarded FFI body that yields a pointer, returning null on panic
/// or error (with the last-error slot set).
pub(crate) fn guard_ptr<T, F>(f: F) -> *mut T
where
    F: FnOnce() -> Result<*mut T, CapiError>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(err)) => {
            store_last_error(&err.message, err.recoverable);
            std::ptr::null_mut()
        }
        Err(payload) => {
            store_last_error(&panic_message(payload), false);
            std::ptr::null_mut()
        }
    }
}

/// The most recent failing call's message on this thread. Valid until the next
/// RLMesh call on this thread; NULL if none. Read only after a nonzero status.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| match &*slot.borrow() {
        Some((cstr, _)) => cstr.as_ptr(),
        None => std::ptr::null(),
    })
}

/// Whether the most recent failing call on this thread is recoverable.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_last_error_is_recoverable() -> c_int {
    c_int::from(last_error_recoverable())
}
