//! Build identity, the current workflow edition and the warning handler:
//! `rlmesh_build_info`, `rlmesh_current_workflow_edition`,
//! `rlmesh_set_warning_handler` (Core, R-23).
#![allow(unsafe_code)] // FFI: out-structs, static C strings, callbacks.

use std::ffi::{CString, c_char, c_void};
use std::sync::LazyLock;

use crate::abi::status::{
    CapiError, RLMeshStatus, clear_last_error, guard, guard_value, last_error_message,
};
use crate::codec::RLMeshBytes;

/// `RLMeshBuildSource`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RLMeshBuildSource {
    #[allow(dead_code)] // the C enum's reserved 0, never produced
    Invalid = 0,
    Release = 1,
    Package = 2,
    Git = 3,
}

impl RLMeshBuildSource {
    fn of(source: rlmesh::BuildSource) -> Self {
        match source {
            rlmesh::BuildSource::Release => Self::Release,
            rlmesh::BuildSource::Git => Self::Git,
            _ => Self::Package,
        }
    }
}

/// `RLMeshBuildInfo` (an out-struct, R-19).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshBuildInfo {
    pub struct_size: usize,
    pub version: *const c_char,
    pub protocol_generation: *const c_char,
    pub workflow_edition: *const c_char,
    pub workflow_edition_base: *const c_char,
    pub supported_workflow_editions: *const *const c_char,
    pub supported_workflow_editions_len: usize,
    pub build_cohort: *const c_char,
    pub build_source: RLMeshBuildSource,
    pub _reserved: u32,
    pub git: *const c_char,
}

const BUILD_INFO_MIN_SIZE: usize = std::mem::size_of::<RLMeshBuildInfo>();

/// The build's strings as process-lifetime C strings.
struct StaticInfo {
    version: CString,
    protocol_generation: CString,
    workflow_edition: CString,
    workflow_edition_base: CString,
    supported: Vec<CString>,
    supported_ptrs: Vec<*const c_char>,
    build_cohort: CString,
    git: Option<CString>,
}

// SAFETY: the pointers point into `supported`, which is never mutated after
// construction and lives for the process.
unsafe impl Send for StaticInfo {}
unsafe impl Sync for StaticInfo {}

fn c_string(text: &str) -> CString {
    CString::new(text).unwrap_or_default()
}

static INFO: LazyLock<StaticInfo> = LazyLock::new(|| {
    let info = rlmesh::build_info();
    let supported: Vec<CString> = info
        .supported_workflow_editions
        .iter()
        .map(|edition| c_string(edition))
        .collect();
    let supported_ptrs = supported.iter().map(|edition| edition.as_ptr()).collect();
    StaticInfo {
        version: c_string(info.version),
        protocol_generation: c_string(info.protocol_generation),
        workflow_edition: c_string(info.workflow_edition),
        workflow_edition_base: c_string(info.workflow_edition_base),
        supported,
        supported_ptrs,
        build_cohort: c_string(info.build_cohort),
        git: info.git.map(c_string),
    }
});

/// Fill `out` with this library's build identity. `out->struct_size` must be
/// set (`RLMESH_BUILD_INFO_INIT`); only the whole fields it covers are
/// written. Every string is static: nothing to free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_build_info(out: *mut RLMeshBuildInfo) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let struct_size = unsafe { out.cast::<usize>().read_unaligned() };
        if struct_size < BUILD_INFO_MIN_SIZE {
            return Err(CapiError::invalid_arg(format!(
                "RLMeshBuildInfo.struct_size {struct_size} is below its minimum \
                 {BUILD_INFO_MIN_SIZE}"
            )));
        }
        let own = std::mem::size_of::<RLMeshBuildInfo>();
        if struct_size > own {
            let tail =
                unsafe { std::slice::from_raw_parts(out.cast::<u8>().add(own), struct_size - own) };
            if let Some(offset) = tail.iter().position(|&byte| byte != 0) {
                return Err(CapiError::type_error(format!(
                    "RLMeshBuildInfo sets byte {} past the {own} this library knows",
                    own + offset
                )));
            }
        }
        let info = &*INFO;
        let filled = RLMeshBuildInfo {
            struct_size,
            version: info.version.as_ptr(),
            protocol_generation: info.protocol_generation.as_ptr(),
            workflow_edition: info.workflow_edition.as_ptr(),
            workflow_edition_base: info.workflow_edition_base.as_ptr(),
            supported_workflow_editions: info.supported_ptrs.as_ptr(),
            supported_workflow_editions_len: info.supported.len(),
            build_cohort: info.build_cohort.as_ptr(),
            build_source: RLMeshBuildSource::of(rlmesh::build_info().build_source),
            _reserved: 0,
            git: info
                .git
                .as_ref()
                .map_or(std::ptr::null(), |git| git.as_ptr()),
        };
        unsafe { out.write_unaligned(filled) };
        Ok(())
    })
}

/// Python's `repr(build_info())` into `out` (UTF-8, not NUL-terminated; free
/// with `rlmesh_bytes_free`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_build_info_repr(out: *mut RLMeshBytes) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let text = rlmesh::build_info().to_string();
        unsafe { *out = RLMeshBytes::from_vec(text.into_bytes()) };
        Ok(())
    })
}

/// Python's spelling of a build source (`"release"`), static; NULL for
/// INVALID or an unknown value.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_build_source_str(source: i32) -> *const c_char {
    match source {
        1 => c"release".as_ptr(),
        2 => c"package".as_ptr(),
        3 => c"git".as_ptr(),
        _ => std::ptr::null(),
    }
}

/// The bare `YYYY.MM` workflow edition this build runs at, static.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_current_workflow_edition() -> *const c_char {
    static EDITION: LazyLock<CString> =
        LazyLock::new(|| c_string(rlmesh::current_workflow_edition()));
    EDITION.as_ptr()
}

// ---- warnings ---------------------------------------------------------------

/// `RLMeshWarningCategory`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RLMeshWarningCategory {
    #[allow(dead_code)] // the C enum's reserved 0, never produced
    Invalid = 0,
    UserWarning = 1,
    RuntimeWarning = 2,
    WorkflowEditionWarning = 3,
}

impl RLMeshWarningCategory {
    fn of(category: rlmesh::WarningCategory) -> Self {
        match category {
            rlmesh::WarningCategory::RuntimeWarning => Self::RuntimeWarning,
            rlmesh::WarningCategory::WorkflowEditionWarning => Self::WorkflowEditionWarning,
            _ => Self::UserWarning,
        }
    }

    fn to_rust(value: i32) -> Option<rlmesh::WarningCategory> {
        match value {
            1 => Some(rlmesh::WarningCategory::UserWarning),
            2 => Some(rlmesh::WarningCategory::RuntimeWarning),
            3 => Some(rlmesh::WarningCategory::WorkflowEditionWarning),
            _ => None,
        }
    }
}

/// Python's class name of a warning category, static; NULL for INVALID or an
/// unknown value.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_warning_category_name(category: i32) -> *const c_char {
    match RLMeshWarningCategory::to_rust(category) {
        Some(rlmesh::WarningCategory::UserWarning) => c"UserWarning".as_ptr(),
        Some(rlmesh::WarningCategory::RuntimeWarning) => c"RuntimeWarning".as_ptr(),
        Some(rlmesh::WarningCategory::WorkflowEditionWarning) => c"WorkflowEditionWarning".as_ptr(),
        _ => std::ptr::null(),
    }
}

/// Python's `issubclass(cls, base)` on warning categories; false for INVALID
/// or an unknown value.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_warning_category_is_subclass(cls: i32, base: i32) -> bool {
    match (
        RLMeshWarningCategory::to_rust(cls),
        RLMeshWarningCategory::to_rust(base),
    ) {
        (Some(cls), Some(base)) => cls.is_subclass_of(base),
        _ => false,
    }
}

type WarningHandlerFn =
    unsafe extern "C" fn(user_data: *mut c_void, category: i32, message: *const c_char) -> i32;

/// The C handler and its `user_data`, freed once with the last copy.
struct CHandler {
    handler: WarningHandlerFn,
    user_data: *mut c_void,
    free_user_data: Option<unsafe extern "C" fn(*mut c_void)>,
}

// SAFETY: R-23 requires the handler and its `user_data` to be thread-safe.
unsafe impl Send for CHandler {}
unsafe impl Sync for CHandler {}

impl Drop for CHandler {
    fn drop(&mut self) {
        if let Some(free) = self.free_user_data {
            unsafe { free(self.user_data) };
        }
    }
}

impl CHandler {
    fn call(&self, warning: &rlmesh::Warning) -> rlmesh::Result<()> {
        let message = c_string(&warning.message.replace('\0', ""));
        let category = RLMeshWarningCategory::of(warning.category) as i32;
        clear_last_error();
        let status = guard_value(-1, || unsafe {
            (self.handler)(self.user_data, category, message.as_ptr())
        });
        if status == 0 {
            return Ok(());
        }
        let detail = last_error_message();
        let detail = if detail.is_empty() {
            format!("warning handler failed with status {status}")
        } else {
            detail
        };
        // A class the handler set (`rlmesh_callback_set_error_class`) is the
        // error's class; otherwise it is an RLMeshException (R-23).
        Err(match crate::abi::status::take_callback_class() {
            Some(class) => crate::abi::status::error_of_class(class, &detail),
            None => rlmesh::Error::Internal(detail),
        })
    }
}

/// Install the process-wide warning handler (R-23). A NULL `handler`
/// restores the default (stderr, each distinct warning once); a NULL
/// `handler` with a non-NULL `user_data` or `free_user_data` is
/// RLMESH_ERR_INVALID_ARGUMENT and keeps the installed handler. On RLMESH_OK
/// the library owns `user_data` and calls `free_user_data` once, after the
/// handler is replaced and every call of it has returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_set_warning_handler(
    handler: Option<WarningHandlerFn>,
    user_data: *mut c_void,
    free_user_data: Option<unsafe extern "C" fn(*mut c_void)>,
) -> RLMeshStatus {
    guard(|| {
        let Some(handler) = handler else {
            if !user_data.is_null() || free_user_data.is_some() {
                return Err(CapiError::invalid_arg(
                    "a NULL warning handler takes no user_data or free_user_data",
                ));
            }
            rlmesh::set_warning_handler(None);
            return Ok(());
        };
        let installed = CHandler {
            handler,
            user_data,
            free_user_data,
        };
        rlmesh::set_warning_handler(Some(Box::new(move |warning: &rlmesh::Warning| {
            installed.call(warning)
        })));
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn build_info_fills_the_static_fields() {
        let mut info = RLMeshBuildInfo {
            struct_size: std::mem::size_of::<RLMeshBuildInfo>(),
            version: std::ptr::null(),
            protocol_generation: std::ptr::null(),
            workflow_edition: std::ptr::null(),
            workflow_edition_base: std::ptr::null(),
            supported_workflow_editions: std::ptr::null(),
            supported_workflow_editions_len: 0,
            build_cohort: std::ptr::null(),
            build_source: RLMeshBuildSource::Invalid,
            _reserved: 0,
            git: std::ptr::null(),
        };
        assert_eq!(unsafe { rlmesh_build_info(&mut info) }, RLMeshStatus::Ok);
        let text = |ptr: *const c_char| unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        let rust = rlmesh::build_info();
        assert_eq!(text(info.version), rust.version);
        assert_eq!(text(info.protocol_generation), rust.protocol_generation);
        assert_eq!(text(info.workflow_edition_base), rust.workflow_edition_base);
        let editions = unsafe {
            std::slice::from_raw_parts(
                info.supported_workflow_editions,
                info.supported_workflow_editions_len,
            )
        };
        assert_eq!(text(editions[0]), rust.workflow_edition);
        assert_eq!(info.git.is_null(), rust.git.is_none());
        assert_eq!(
            text(rlmesh_build_source_str(info.build_source as i32)),
            rust.build_source.as_str()
        );
        assert_eq!(
            text(rlmesh_current_workflow_edition()),
            rlmesh::current_workflow_edition()
        );

        info.struct_size = 8;
        assert_eq!(
            unsafe { rlmesh_build_info(&mut info) },
            RLMeshStatus::InvalidArgument
        );
    }

    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static FREED: AtomicUsize = AtomicUsize::new(0);
    static SEEN: Mutex<Vec<(i32, String)>> = Mutex::new(Vec::new());

    unsafe extern "C" fn record(_: *mut c_void, category: i32, message: *const c_char) -> i32 {
        CALLS.fetch_add(1, Ordering::SeqCst);
        let message = unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned();
        SEEN.lock().unwrap().push((category, message));
        unsafe { crate::model::rlmesh_callback_set_error(c"as error".as_ptr(), false) };
        1
    }

    unsafe extern "C" fn free(_: *mut c_void) {
        FREED.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn a_failing_c_handler_fails_the_warning_and_is_freed_once() {
        let status =
            unsafe { rlmesh_set_warning_handler(Some(record), std::ptr::null_mut(), Some(free)) };
        assert_eq!(status, RLMeshStatus::Ok);
        let err = rlmesh::binding::resolve_workflow_edition(Default::default(), true);
        assert_eq!(
            unsafe { rlmesh_set_warning_handler(None, std::ptr::null_mut(), None) },
            RLMeshStatus::Ok
        );
        assert_eq!(FREED.load(Ordering::SeqCst), 1);
        if std::env::var("RLMESH_WORKFLOW_EDITION").is_err() {
            let err = err.unwrap_err();
            assert_eq!(err.to_string(), "as error");
            let seen = SEEN.lock().unwrap();
            assert_eq!(
                seen[0].0,
                RLMeshWarningCategory::WorkflowEditionWarning as i32
            );
            assert!(
                seen[0].1.starts_with("no workflow edition declared"),
                "{}",
                seen[0].1
            );
        }
        let status = unsafe { rlmesh_set_warning_handler(None, std::ptr::null_mut(), Some(free)) };
        assert_eq!(status, RLMeshStatus::InvalidArgument);
        assert!(rlmesh_warning_category_is_subclass(3, 1));
        assert!(!rlmesh_warning_category_is_subclass(1, 3));
        assert!(rlmesh_warning_category_name(0).is_null());
    }
}
