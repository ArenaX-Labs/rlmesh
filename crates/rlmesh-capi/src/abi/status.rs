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

use rlmesh::{ErrorClass, ErrorCode, ModelErrorCode};

/// Integer values are stable (ABI). Values 0-8 and 99 keep their original
/// discriminants; 9-12 were added with the cross-language error classes.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RLMeshStatus {
    Ok = 0,
    InvalidArgument = 1,
    /// Deprecated: never returned; a wrong-kind read is [`RLMeshStatus::Type`].
    #[allow(dead_code)] // Kept for its ABI discriminant.
    InvalidValue = 2,
    Environment = 3,
    Model = 4,
    /// `RLMESH_ERR_CONNECTION`; `RLMESH_ERR_TRANSPORT` is its deprecated alias.
    Connection = 5,
    Timeout = 6,
    Panic = 7,
    Cancelled = 8,
    Protocol = 9,
    Type = 10,
    Runtime = 11,
    Os = 12,
    Internal = 99,
}

/// The exact class of the last error: `INVALID` (0) for none, then the 13
/// classes in the order of the cross-language error table.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RLMeshErrorClass {
    Invalid = 0,
    RLMeshException = 1,
    ProtocolException = 2,
    EnvironmentException = 3,
    RecoverableEnvironmentException = 4,
    ModelException = 5,
    ValueError = 6,
    AdapterResolutionError = 7,
    TypeError = 8,
    RuntimeError = 9,
    OSError = 10,
    ConnectionError = 11,
    TimeoutError = 12,
    KeyboardInterrupt = 13,
}

impl RLMeshErrorClass {
    /// The class a C value names; `None` for INVALID or an unknown value.
    pub(crate) fn from_c(value: i32) -> Option<Self> {
        Some(match value {
            1 => Self::RLMeshException,
            2 => Self::ProtocolException,
            3 => Self::EnvironmentException,
            4 => Self::RecoverableEnvironmentException,
            5 => Self::ModelException,
            6 => Self::ValueError,
            7 => Self::AdapterResolutionError,
            8 => Self::TypeError,
            9 => Self::RuntimeError,
            10 => Self::OSError,
            11 => Self::ConnectionError,
            12 => Self::TimeoutError,
            13 => Self::KeyboardInterrupt,
            _ => return None,
        })
    }

    fn from_rust(class: ErrorClass) -> Self {
        match class {
            ErrorClass::RLMeshException => Self::RLMeshException,
            ErrorClass::ProtocolException => Self::ProtocolException,
            ErrorClass::EnvironmentException => Self::EnvironmentException,
            ErrorClass::RecoverableEnvironmentException => Self::RecoverableEnvironmentException,
            ErrorClass::ModelException => Self::ModelException,
            ErrorClass::ValueError => Self::ValueError,
            ErrorClass::AdapterResolutionError => Self::AdapterResolutionError,
            ErrorClass::TypeError => Self::TypeError,
            ErrorClass::RuntimeError => Self::RuntimeError,
            ErrorClass::OSError => Self::OSError,
            ErrorClass::ConnectionError => Self::ConnectionError,
            ErrorClass::TimeoutError => Self::TimeoutError,
            ErrorClass::KeyboardInterrupt => Self::KeyboardInterrupt,
            // ErrorClass is #[non_exhaustive]; a class added later without a C
            // value reports as its nearest documented base.
            _ => Self::RLMeshException,
        }
    }

    /// The Rust class for a C value; `None` for `INVALID` or an unknown value.
    fn to_rust(value: c_int) -> Option<ErrorClass> {
        let index = usize::try_from(value).ok()?.checked_sub(1)?;
        ErrorClass::ALL.get(index).copied()
    }

    /// The class a status reports when nothing more specific is known.
    fn for_status(status: RLMeshStatus) -> Self {
        match status {
            RLMeshStatus::Ok => Self::Invalid,
            RLMeshStatus::InvalidArgument => Self::ValueError,
            RLMeshStatus::InvalidValue | RLMeshStatus::Type => Self::TypeError,
            RLMeshStatus::Environment => Self::EnvironmentException,
            RLMeshStatus::Model => Self::ModelException,
            RLMeshStatus::Connection => Self::ConnectionError,
            RLMeshStatus::Timeout => Self::TimeoutError,
            RLMeshStatus::Panic | RLMeshStatus::Internal => Self::RLMeshException,
            RLMeshStatus::Cancelled => Self::KeyboardInterrupt,
            RLMeshStatus::Protocol => Self::ProtocolException,
            RLMeshStatus::Runtime => Self::RuntimeError,
            RLMeshStatus::Os => Self::OSError,
        }
    }
}

/// The status an error of `class` returns.
fn status_for_class(class: ErrorClass) -> RLMeshStatus {
    match class {
        ErrorClass::ProtocolException => RLMeshStatus::Protocol,
        ErrorClass::EnvironmentException | ErrorClass::RecoverableEnvironmentException => {
            RLMeshStatus::Environment
        }
        ErrorClass::ModelException => RLMeshStatus::Model,
        ErrorClass::ValueError | ErrorClass::AdapterResolutionError => {
            RLMeshStatus::InvalidArgument
        }
        ErrorClass::TypeError => RLMeshStatus::Type,
        ErrorClass::RuntimeError => RLMeshStatus::Runtime,
        ErrorClass::OSError => RLMeshStatus::Os,
        ErrorClass::ConnectionError => RLMeshStatus::Connection,
        ErrorClass::TimeoutError => RLMeshStatus::Timeout,
        ErrorClass::KeyboardInterrupt => RLMeshStatus::Cancelled,
        _ => RLMeshStatus::Internal,
    }
}

/// The detail code an error carries: an env code for the Environment family,
/// a model code for the Model class (both as wire numbers).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CapiCode {
    Env(ErrorCode),
    Model(ModelErrorCode),
}

impl CapiCode {
    fn wire(self) -> i32 {
        match self {
            Self::Env(code) => env_code_wire(code),
            Self::Model(code) => model_code_wire(code),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Env(code) => code.as_str(),
            Self::Model(code) => code.as_str(),
        }
    }
}

fn env_code_wire(code: ErrorCode) -> i32 {
    match code {
        ErrorCode::Unspecified => 0,
        ErrorCode::Timeout => 1,
        ErrorCode::InvalidAction => 2,
        ErrorCode::NotReady => 3,
        ErrorCode::Busy => 4,
        ErrorCode::Unsupported => 5,
        ErrorCode::Crashed => 11,
        ErrorCode::Cancelled => 12,
        ErrorCode::Closed => 13,
        // Internal, and any code added later (ErrorCode is #[non_exhaustive]).
        _ => 10,
    }
}

fn model_code_wire(code: ModelErrorCode) -> i32 {
    match code {
        ModelErrorCode::Unspecified => 0,
        ModelErrorCode::InvalidRequest => 2,
        ModelErrorCode::NotConfigured => 3,
        ModelErrorCode::Busy => 4,
        ModelErrorCode::Unsupported => 5,
        ModelErrorCode::Cancelled => 12,
        ModelErrorCode::Closed => 13,
        // Internal, and any code added later (#[non_exhaustive]).
        _ => 10,
    }
}

fn env_code_from_wire(value: c_int) -> Option<ErrorCode> {
    Some(match value {
        0 => ErrorCode::Unspecified,
        1 => ErrorCode::Timeout,
        2 => ErrorCode::InvalidAction,
        3 => ErrorCode::NotReady,
        4 => ErrorCode::Busy,
        5 => ErrorCode::Unsupported,
        10 => ErrorCode::Internal,
        11 => ErrorCode::Crashed,
        12 => ErrorCode::Cancelled,
        13 => ErrorCode::Closed,
        _ => return None,
    })
}

fn model_code_from_wire(value: c_int) -> Option<ModelErrorCode> {
    Some(match value {
        0 => ModelErrorCode::Unspecified,
        2 => ModelErrorCode::InvalidRequest,
        3 => ModelErrorCode::NotConfigured,
        4 => ModelErrorCode::Busy,
        5 => ModelErrorCode::Unsupported,
        10 => ModelErrorCode::Internal,
        12 => ModelErrorCode::Cancelled,
        13 => ModelErrorCode::Closed,
        _ => return None,
    })
}

pub(crate) struct CapiError {
    pub status: RLMeshStatus,
    pub class: RLMeshErrorClass,
    pub code: Option<CapiCode>,
    pub message: String,
    pub recoverable: bool,
}

impl CapiError {
    pub(crate) fn new(status: RLMeshStatus, message: impl Into<String>) -> Self {
        Self {
            status,
            class: RLMeshErrorClass::for_status(status),
            code: None,
            message: message.into(),
            recoverable: false,
        }
    }
    /// An invalid argument value (`ValueError`).
    pub(crate) fn invalid_arg(message: impl Into<String>) -> Self {
        Self::new(RLMeshStatus::InvalidArgument, message)
    }
    /// The wrong kind of value or object (`TypeError`), such as a read of
    /// another kind than the value holds.
    pub(crate) fn type_error(message: impl Into<String>) -> Self {
        Self::new(RLMeshStatus::Type, message)
    }
    /// Adapters cannot bridge env and model (`AdapterResolutionError`, a
    /// `ValueError`).
    pub(crate) fn adapter_resolution(message: impl Into<String>) -> Self {
        Self {
            class: RLMeshErrorClass::AdapterResolutionError,
            ..Self::invalid_arg(message)
        }
    }
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(RLMeshStatus::Internal, message)
    }
    /// The same error reported as a cancellation (`KeyboardInterrupt`).
    pub(crate) fn into_cancelled(self) -> Self {
        Self {
            status: RLMeshStatus::Cancelled,
            class: RLMeshErrorClass::KeyboardInterrupt,
            code: None,
            recoverable: false,
            ..self
        }
    }
}

impl From<rlmesh::Error> for CapiError {
    fn from(err: rlmesh::Error) -> Self {
        let class = err.class();
        let code = match &err {
            rlmesh::Error::Environment(error) => Some(CapiCode::Env(error.code)),
            rlmesh::Error::Model(error) => Some(CapiCode::Model(error.code)),
            _ => None,
        };
        Self {
            status: status_for_class(class),
            class: RLMeshErrorClass::from_rust(class),
            code,
            recoverable: err.is_recoverable(),
            message: err.to_string(),
        }
    }
}

struct LastError {
    message: CString,
    recoverable: bool,
    status: RLMeshStatus,
    class: RLMeshErrorClass,
    code: Option<CapiCode>,
    code_text: Option<CString>,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<LastError>> = const { RefCell::new(None) };
}

fn store(err: &CapiError) {
    let safe: Vec<u8> = err.message.bytes().filter(|&b| b != 0).collect();
    let message = CString::new(safe).unwrap_or_default();
    let code_text = err
        .code
        .map(|code| CString::new(code.as_str()).expect("code spellings contain no NUL"));
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(LastError {
            message,
            recoverable: err.recoverable,
            status: err.status,
            class: err.class,
            code: err.code,
            code_text,
        });
    });
}

/// Record a callback's failure (`rlmesh_callback_set_error`): read back on the
/// same thread when the callback returns nonzero.
pub(crate) fn store_last_error(message: &str, recoverable: bool, status: RLMeshStatus) {
    let mut err = CapiError::new(status, message);
    err.recoverable = recoverable;
    store(&err);
    CALLBACK_CLASS.with(|slot| slot.set(None));
}

/// Record a callback's failure of an exact class
/// (`rlmesh_callback_set_error_class`); INVALID or an unknown value records
/// `RLMeshException`. Only `RecoverableEnvironmentException` is recoverable.
pub(crate) fn store_last_error_class(message: &str, class: i32) {
    let class = RLMeshErrorClass::from_c(class).unwrap_or(RLMeshErrorClass::RLMeshException);
    let error = error_of_class(class, message);
    let mut err = CapiError::from(error);
    err.message = message.to_string();
    store(&err);
    CALLBACK_CLASS.with(|slot| slot.set(Some(class)));
}

thread_local! {
    /// The class the running callback set with
    /// `rlmesh_callback_set_error_class`, so a nested export's own failure
    /// is not mistaken for it.
    static CALLBACK_CLASS: std::cell::Cell<Option<RLMeshErrorClass>> =
        const { std::cell::Cell::new(None) };
}

/// The class the callback that just returned set, if it set one.
pub(crate) fn take_callback_class() -> Option<RLMeshErrorClass> {
    CALLBACK_CLASS.with(|slot| slot.take())
}

/// The facade error a callback's recorded class stands for.
pub(crate) fn error_of_class(class: RLMeshErrorClass, message: &str) -> rlmesh::Error {
    use rlmesh::Error;
    let message = message.to_string();
    match class {
        RLMeshErrorClass::ProtocolException => Error::Protocol(message),
        RLMeshErrorClass::EnvironmentException => Error::environment(message),
        RLMeshErrorClass::RecoverableEnvironmentException => {
            Error::environment_recoverable(message)
        }
        RLMeshErrorClass::ModelException => Error::model(message),
        RLMeshErrorClass::ValueError | RLMeshErrorClass::AdapterResolutionError => {
            Error::invalid_argument(message)
        }
        RLMeshErrorClass::TypeError => Error::Type(message),
        RLMeshErrorClass::RuntimeError => Error::Runtime(message),
        RLMeshErrorClass::OSError => Error::Os(message),
        RLMeshErrorClass::ConnectionError => Error::Connection(message),
        // Rust's Timeout carries the deadline, not a text; a callback's has none.
        RLMeshErrorClass::TimeoutError => Error::Timeout(std::time::Duration::ZERO),
        RLMeshErrorClass::KeyboardInterrupt => Error::Cancelled,
        RLMeshErrorClass::RLMeshException | RLMeshErrorClass::Invalid => Error::Internal(message),
    }
}

/// Called before invoking a callback so a decline that does not set an error
/// cannot report a previous request's stale message on a reused pool thread
/// (20-review-spec B3).
pub(crate) fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
    CALLBACK_CLASS.with(|slot| slot.set(None));
}

pub(crate) fn last_error_message() -> String {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|last| last.message.to_string_lossy().into_owned())
            .unwrap_or_default()
    })
}

pub(crate) fn last_error_recoverable() -> bool {
    LAST_ERROR.with(|slot| slot.borrow().as_ref().is_some_and(|last| last.recoverable))
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

fn store_panic(payload: Box<dyn Any + Send>) {
    store(&CapiError::new(RLMeshStatus::Panic, panic_message(payload)));
}

pub(crate) fn guard<F>(f: F) -> RLMeshStatus
where
    F: FnOnce() -> Result<(), CapiError>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => RLMeshStatus::Ok,
        Ok(Err(err)) => {
            store(&err);
            err.status
        }
        Err(payload) => {
            store_panic(payload);
            RLMeshStatus::Panic
        }
    }
}

pub(crate) fn guard_ptr<T, F>(f: F) -> *mut T
where
    F: FnOnce() -> Result<*mut T, CapiError>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(err)) => {
            store(&err);
            std::ptr::null_mut()
        }
        Err(payload) => {
            store_panic(payload);
            std::ptr::null_mut()
        }
    }
}

/// Panic boundary for value-returning exports (no status channel): on a panic,
/// record it as the last error and return `default` instead of unwinding into C.
pub(crate) fn guard_value<T, F>(default: T, f: F) -> T
where
    F: FnOnce() -> T,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(payload) => {
            store_panic(payload);
            default
        }
    }
}

/// Valid until the next RLMesh call on this thread; NULL if none. Read only
/// after a nonzero status.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| match &*slot.borrow() {
        Some(last) => last.message.as_ptr(),
        None => std::ptr::null(),
    })
}

/// The status the most recent failing call on this thread recorded — the code a
/// status-returning call would have returned. This is the only status channel
/// the pointer-returning exports have (they report failure as NULL).
/// `RLMESH_OK` when no error is recorded.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_last_error_status() -> RLMeshStatus {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(RLMeshStatus::Ok, |last| last.status)
    })
}

/// Python's `getattr(exc, "is_recoverable", False)`: nonzero only for a
/// recoverable Environment or Model error.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_last_error_is_recoverable() -> c_int {
    c_int::from(last_error_recoverable())
}

/// The exact class of the last error on this thread; `INVALID` when none.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_last_error_class() -> RLMeshErrorClass {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(RLMeshErrorClass::Invalid, |last| last.class)
    })
}

/// Write the last error's code (the wire number of an `RLMeshErrorCode` for the
/// Environment family, of an `RLMeshModelErrorCode` for the Model class) and
/// return true; return false with `*out` untouched when the class carries no
/// code. A NULL `out` only reports whether there is a code.
///
/// # Safety
/// `out` is NULL or points to a writable `int32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_last_error_code(out: *mut i32) -> bool {
    let code = LAST_ERROR.with(|slot| slot.borrow().as_ref().and_then(|last| last.code));
    match code {
        Some(code) => {
            if let Some(out) = unsafe { out.as_mut() } {
                *out = code.wire();
            }
            true
        }
        None => false,
    }
}

/// The last error's code as Python's `code` text (`"INVALID_ACTION"`); NULL
/// when the class carries no code. Valid until the next RLMesh call on this
/// thread.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_last_error_code_str() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|last| last.code_text.as_ref())
            .map_or(std::ptr::null(), |text| text.as_ptr())
    })
}

/// Static NUL-terminated spelling of a code, or NULL.
fn static_cstr(text: Option<&'static str>) -> *const c_char {
    macro_rules! table {
        ($($name:literal),*) => {
            match text {
                $(Some($name) => concat!($name, "\0").as_ptr().cast(),)*
                _ => std::ptr::null(),
            }
        };
    }
    table!(
        "UNSPECIFIED",
        "TIMEOUT",
        "INVALID_ACTION",
        "NOT_READY",
        "BUSY",
        "UNSUPPORTED",
        "INTERNAL",
        "CRASHED",
        "CANCELLED",
        "CLOSED",
        "INVALID_REQUEST",
        "NOT_CONFIGURED"
    )
}

/// The text of an env error code (`"BUSY"`); NULL for an unknown value. The
/// string is static.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_error_code_str(code: c_int) -> *const c_char {
    static_cstr(env_code_from_wire(code).map(ErrorCode::as_str))
}

/// The text of a model error code (`"INVALID_REQUEST"`); NULL for an unknown
/// value. The string is static.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_model_error_code_str(code: c_int) -> *const c_char {
    static_cstr(model_code_from_wire(code).map(ModelErrorCode::as_str))
}

/// Python's spelling of a class (`"RecoverableEnvironmentException"`); NULL for
/// `INVALID` or an unknown value. The string is static.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_error_class_name(class: c_int) -> *const c_char {
    macro_rules! names {
        ($($class:ident),*) => {
            match RLMeshErrorClass::to_rust(class) {
                $(Some(ErrorClass::$class) => concat!(stringify!($class), "\0").as_ptr().cast(),)*
                _ => std::ptr::null(),
            }
        };
    }
    names!(
        RLMeshException,
        ProtocolException,
        EnvironmentException,
        RecoverableEnvironmentException,
        ModelException,
        ValueError,
        AdapterResolutionError,
        TypeError,
        RuntimeError,
        OSError,
        ConnectionError,
        TimeoutError,
        KeyboardInterrupt
    )
}

/// Python's `issubclass(cls, base)`: reflexive and following Python's MRO
/// (`CONNECTION_ERROR` is a subclass of `OS_ERROR`). False when either value
/// is `INVALID` or unknown.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_error_class_is_subclass(class: c_int, base: c_int) -> bool {
    match (
        RLMeshErrorClass::to_rust(class),
        RLMeshErrorClass::to_rust(base),
    ) {
        (Some(class), Some(base)) => class.is_subclass_of(base),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::time::Duration;

    use super::*;

    fn text(ptr: *const c_char) -> Option<String> {
        (!ptr.is_null()).then(|| {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        })
    }

    fn fail_with(err: rlmesh::Error) -> RLMeshStatus {
        guard(|| Err(CapiError::from(err)))
    }

    #[test]
    fn each_rust_error_returns_its_status_class_and_text() {
        let cases = [
            (
                rlmesh::Error::Internal("boom".into()),
                RLMeshStatus::Internal,
                RLMeshErrorClass::RLMeshException,
                "boom",
            ),
            (
                rlmesh::Error::Protocol("bad".into()),
                RLMeshStatus::Protocol,
                RLMeshErrorClass::ProtocolException,
                "bad",
            ),
            (
                rlmesh::Error::environment_recoverable("busy"),
                RLMeshStatus::Environment,
                RLMeshErrorClass::RecoverableEnvironmentException,
                "[INTERNAL] busy",
            ),
            (
                rlmesh::Error::model("declined"),
                RLMeshStatus::Model,
                RLMeshErrorClass::ModelException,
                "[INTERNAL] declined",
            ),
            (
                rlmesh::Error::invalid_argument("bad seed"),
                RLMeshStatus::InvalidArgument,
                RLMeshErrorClass::ValueError,
                "bad seed",
            ),
            (
                rlmesh::Error::Type("kind".into()),
                RLMeshStatus::Type,
                RLMeshErrorClass::TypeError,
                "kind",
            ),
            (
                rlmesh::Error::Runtime("closed".into()),
                RLMeshStatus::Runtime,
                RLMeshErrorClass::RuntimeError,
                "closed",
            ),
            (
                rlmesh::Error::Os("exists".into()),
                RLMeshStatus::Os,
                RLMeshErrorClass::OSError,
                "exists",
            ),
            (
                rlmesh::Error::Connection("reset".into()),
                RLMeshStatus::Connection,
                RLMeshErrorClass::ConnectionError,
                "reset",
            ),
            (
                rlmesh::Error::Timeout(Duration::from_secs(5)),
                RLMeshStatus::Timeout,
                RLMeshErrorClass::TimeoutError,
                "operation timed out after 5s",
            ),
            (
                rlmesh::Error::Cancelled,
                RLMeshStatus::Cancelled,
                RLMeshErrorClass::KeyboardInterrupt,
                "cancelled",
            ),
        ];
        for (error, status, class, message) in cases {
            assert_eq!(fail_with(error), status);
            assert_eq!(rlmesh_last_error_status(), status);
            assert_eq!(rlmesh_last_error_class(), class);
            assert_eq!(text(rlmesh_last_error_message()).as_deref(), Some(message));
        }
    }

    #[test]
    fn codes_are_read_only_for_the_classes_that_carry_one() {
        fail_with(rlmesh::Error::Environment(rlmesh::EnvironmentError {
            code: ErrorCode::Busy,
            message: "later".into(),
            is_recoverable: true,
        }));
        let mut code = -1;
        assert!(unsafe { rlmesh_last_error_code(&mut code) });
        assert_eq!(code, 4);
        assert_eq!(text(rlmesh_last_error_code_str()).as_deref(), Some("BUSY"));
        assert_eq!(rlmesh_last_error_is_recoverable(), 1);

        fail_with(rlmesh::Error::model("declined"));
        assert!(unsafe { rlmesh_last_error_code(&mut code) });
        assert_eq!(code, 10);
        assert_eq!(
            text(rlmesh_last_error_code_str()).as_deref(),
            Some("INTERNAL")
        );
        assert_eq!(rlmesh_last_error_is_recoverable(), 0);

        // A transport failure is not recoverable in Python's sense (COR-4) and
        // carries no code.
        fail_with(rlmesh::Error::Connection("reset".into()));
        let mut untouched = 77;
        assert!(!unsafe { rlmesh_last_error_code(&mut untouched) });
        assert_eq!(untouched, 77);
        assert!(rlmesh_last_error_code_str().is_null());
        assert_eq!(rlmesh_last_error_is_recoverable(), 0);
    }

    #[test]
    fn code_and_class_spellings_are_static_text() {
        assert_eq!(
            text(rlmesh_error_code_str(2)).as_deref(),
            Some("INVALID_ACTION")
        );
        assert_eq!(
            text(rlmesh_error_code_str(5)).as_deref(),
            Some("UNSUPPORTED")
        );
        assert!(rlmesh_error_code_str(7).is_null());
        assert_eq!(
            text(rlmesh_model_error_code_str(2)).as_deref(),
            Some("INVALID_REQUEST")
        );
        assert!(rlmesh_model_error_code_str(1).is_null());
        assert_eq!(
            text(rlmesh_error_class_name(
                RLMeshErrorClass::RecoverableEnvironmentException as c_int
            ))
            .as_deref(),
            Some("RecoverableEnvironmentException")
        );
        assert_eq!(
            text(rlmesh_error_class_name(RLMeshErrorClass::OSError as c_int)).as_deref(),
            Some("OSError")
        );
        assert!(rlmesh_error_class_name(0).is_null());
        assert!(rlmesh_error_class_name(14).is_null());
        // Every C class value round-trips to the Rust class of the same name.
        for class in ErrorClass::ALL {
            let value = RLMeshErrorClass::from_rust(class) as c_int;
            assert_eq!(
                text(rlmesh_error_class_name(value)).as_deref(),
                Some(class.name())
            );
        }
    }

    #[test]
    fn is_subclass_follows_python_mro() {
        use RLMeshErrorClass as C;
        let sub = |a: C, b: C| rlmesh_error_class_is_subclass(a as c_int, b as c_int);
        assert!(sub(C::ConnectionError, C::OSError));
        assert!(sub(C::RecoverableEnvironmentException, C::RLMeshException));
        assert!(sub(C::RLMeshException, C::RuntimeError));
        assert!(sub(C::ValueError, C::ValueError));
        assert!(!sub(C::ValueError, C::RLMeshException));
        assert!(!sub(C::Invalid, C::Invalid));
    }

    #[test]
    fn a_panic_reports_the_rlmesh_exception_class() {
        let status = guard(|| panic!("boom"));
        assert_eq!(status, RLMeshStatus::Panic);
        assert_eq!(rlmesh_last_error_class(), RLMeshErrorClass::RLMeshException);
        assert!(rlmesh_error_class_is_subclass(
            rlmesh_last_error_class() as c_int,
            RLMeshErrorClass::RLMeshException as c_int
        ));
    }

    #[test]
    fn adapter_resolution_is_a_value_error_with_its_own_class() {
        let status = guard(|| Err(CapiError::adapter_resolution("no bridge")));
        assert_eq!(status, RLMeshStatus::InvalidArgument);
        assert_eq!(
            rlmesh_last_error_class(),
            RLMeshErrorClass::AdapterResolutionError
        );
    }
}
