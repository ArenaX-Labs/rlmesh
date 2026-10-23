//! Serialized recorder calls; capture hooks own the facade's shared state.
#![allow(unsafe_code)]
pub(crate) mod records;
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard, guard_value},
};
use crate::codec::RLMeshBytes;
use crate::ffi;
use crate::run_hooks::{RLMeshRunContext, RLMeshRunHooks};
use crate::run_results::{self, RLMeshRunResult};
use crate::value::handle::{RLMeshValue, into_handle, write_out};
use records::RLMeshWorkloadRecord;
use rlmesh::Value;
use rlmesh::recorder::{
    Recorder, RecorderAddOptions, RecorderCaptureOptions, RecorderExportOptions, RecorderOptions,
    RecorderToDictOptions,
};
use std::ffi::c_char;
use std::ptr::null_mut;
use std::sync::{Arc, Mutex, MutexGuard};

pub struct RLMeshRecorder(Mutex<Recorder>);
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshRecorderOptions {
    pub struct_size: usize,
    pub result_set_id: *const c_char,
    pub fps: u64,
    pub quality: u8,
    pub _reserved: [u8; 7],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshRecorderAddOptions {
    pub struct_size: usize,
    pub task: *const c_char,
    pub config: *const RLMeshValue,
    pub included_in_metrics: bool,
    pub _reserved: [u8; std::mem::size_of::<usize>() - 1],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshRecorderCaptureOptions {
    pub struct_size: usize,
    pub task: *const c_char,
    pub config: *const RLMeshValue,
    pub cameras: *const *const c_char,
    pub cameras_len: usize,
    pub binding: *const RLMeshRunContext,
    pub video_info_keys: *const *const c_char,
    pub video_info_keys_len: usize,
    pub included_in_metrics: bool,
    pub _reserved: [u8; std::mem::size_of::<usize>() - 1],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshRecorderToDictOptions {
    pub struct_size: usize,
    pub recorded_at: *const c_char,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshRecorderExportOptions {
    pub struct_size: usize,
    pub recorded_at: *const c_char,
    pub has_archive: bool,
    pub archive: bool,
    pub _reserved: [u8; std::mem::size_of::<usize>() - 2],
}
fn lock<'a>(ptr: *const RLMeshRecorder) -> Result<MutexGuard<'a, Recorder>, CapiError> {
    Ok(unsafe { ptr.as_ref() }
        .ok_or_else(|| CapiError::invalid_arg("null recorder"))?
        .0
        .lock()
        .unwrap_or_else(|e| e.into_inner()))
}
unsafe fn text(ptr: *const c_char, name: &str) -> Result<String, CapiError> {
    unsafe { ffi::text(ptr) }?.ok_or_else(|| CapiError::invalid_arg(format!("null {name}")))
}
unsafe fn strings(
    ptr: *const *const c_char,
    len: usize,
    name: &str,
) -> Result<Option<Vec<String>>, CapiError> {
    if ptr.is_null() && len == 0 {
        return Ok(None);
    }
    unsafe { crate::spaces::slice_arg(ptr, len, name) }?
        .iter()
        .map(|&p| unsafe { text(p, name) })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}
fn reserved(bytes: &[u8]) -> Result<(), CapiError> {
    if bytes.iter().any(|&b| b != 0) {
        Err(CapiError::type_error(
            "reserved recorder option bytes must be zero",
        ))
    } else {
        Ok(())
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_recorder_new(
    options: *const RLMeshRecorderOptions,
    out: *mut *mut RLMeshRecorder,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, null_mut())
        }?;
        let mut options_out = RecorderOptions::default();
        if let Some(o) = unsafe { sizing::read(options, "RLMeshRecorderOptions") }? {
            reserved(&o._reserved)?;
            options_out = RecorderOptions {
                result_set_id: unsafe { ffi::text(o.result_set_id) }?,
                fps: o.fps,
                quality: o.quality,
            };
        }
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                Box::into_raw(Box::new(RLMeshRecorder(Mutex::new(Recorder::new(
                    options_out,
                )?)))),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_recorder_add(
    ptr: *mut RLMeshRecorder,
    result: *const RLMeshRunResult,
    model: *const c_char,
    env: *const c_char,
    options: *const RLMeshRecorderAddOptions,
    out: *mut *mut RLMeshWorkloadRecord,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, null_mut())
        }?;
        let mut opts = RecorderAddOptions::default();
        if let Some(o) = unsafe { sizing::read(options, "RLMeshRecorderAddOptions") }? {
            reserved(&o._reserved)?;
            opts = RecorderAddOptions {
                task: unsafe { ffi::text(o.task) }?,
                config: ffi::dict(o.config)?,
                included_in_metrics: o.included_in_metrics,
            };
        }
        let result = unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            run_results::require(result)
        }?;
        let model = unsafe { text(model, "model") }?;
        let env = unsafe { text(env, "env") }?;
        let record = lock(ptr)?.add(result, &model, &env, opts)?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                Box::into_raw(Box::new(RLMeshWorkloadRecord(Arc::new(record)))),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_recorder_capture(
    ptr: *mut RLMeshRecorder,
    model: *const c_char,
    env: *const c_char,
    options: *const RLMeshRecorderCaptureOptions,
    out: *mut *mut RLMeshRunHooks,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, null_mut())
        }?;
        let mut opts = RecorderCaptureOptions::default();
        if let Some(o) = unsafe { sizing::read(options, "RLMeshRecorderCaptureOptions") }? {
            reserved(&o._reserved)?;
            opts.task = unsafe { ffi::text(o.task) }?;
            opts.config = ffi::dict(o.config)?;
            opts.cameras = unsafe { strings(o.cameras, o.cameras_len, "cameras") }?;
            opts.binding = unsafe { o.binding.as_ref() }.map(|c| Arc::clone(&c.0));
            if let Some(keys) =
                unsafe { strings(o.video_info_keys, o.video_info_keys_len, "video_info_keys") }?
            {
                opts.video_info_keys = keys;
            }
            opts.included_in_metrics = o.included_in_metrics;
        }
        let model = unsafe { text(model, "model") }?;
        let env = unsafe { text(env, "env") }?;
        let hooks = lock(ptr)?.capture(&model, &env, opts)?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                Box::into_raw(Box::new(crate::run_hooks::native(hooks))),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_recorder_result_set_id(
    ptr: *const RLMeshRecorder,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, lock(ptr)?.result_set_id().to_owned()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_recorder_workloads_len(ptr: *const RLMeshRecorder) -> usize {
    guard_value(0, || lock(ptr).map_or(0, |r| r.workloads().len()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_recorder_workloads_get(
    ptr: *const RLMeshRecorder,
    i: usize,
    out: *mut *mut RLMeshWorkloadRecord,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, null_mut())
        }?;
        let record = lock(ptr)?
            .workloads()
            .get(i)
            .cloned()
            .ok_or_else(|| CapiError::invalid_arg("workload index out of range"))?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                Box::into_raw(Box::new(RLMeshWorkloadRecord(Arc::new(record)))),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_recorder_to_dict(
    ptr: *const RLMeshRecorder,
    options: *const RLMeshRecorderToDictOptions,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, null_mut())
        }?;
        let recorded_at = unsafe { sizing::read(options, "RLMeshRecorderToDictOptions") }?
            .map(|o| unsafe { ffi::text(o.recorded_at) })
            .transpose()?
            .flatten();
        let value = lock(ptr)?.to_dict(RecorderToDictOptions { recorded_at });
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, into_handle(Value::Dict(value)))
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_recorder_export(
    ptr: *const RLMeshRecorder,
    path: *const c_char,
    options: *const RLMeshRecorderExportOptions,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        // An invalid output must not create files.
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let path = unsafe { text(path, "path") }?;
        let mut opts = RecorderExportOptions::default();
        if let Some(o) = unsafe { sizing::read(options, "RLMeshRecorderExportOptions") }? {
            reserved(&o._reserved)?;
            opts = RecorderExportOptions {
                recorded_at: unsafe { ffi::text(o.recorded_at) }?,
                archive: o.has_archive.then_some(o.archive),
            };
        }
        let path = lock(ptr)?.export_(path, opts)?;
        ffi::bytes(out, path.to_string_lossy().into_owned())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_recorder_close(ptr: *mut RLMeshRecorder) -> RLMeshStatus {
    guard(|| lock(ptr)?.close().map_err(Into::into))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_recorder_free(ptr: *mut RLMeshRecorder) {
    guard_value((), || {
        if !ptr.is_null() {
            let r = unsafe { Box::from_raw(ptr) };
            let _ = r.0.lock().unwrap_or_else(|e| e.into_inner()).close();
        }
    });
}
