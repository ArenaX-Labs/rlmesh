//! Plain View data and facade parsing/validation.
#![allow(unsafe_code)]
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard},
};
use crate::codec::RLMeshBytes;
use crate::{ffi, value::handle::write_out};
use rlmesh::{View, ViewBackend, ViewFormat, ViewHold};
use std::ffi::c_char;
use std::time::Duration;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshView {
    pub struct_size: usize,
    pub source: *const c_char,
    pub fps: u64,
    pub hold_seconds: f64,
    pub step_hz: f64,
    pub backend: i32,
    pub format: i32,
    pub hold: i32,
    pub port: u16,
    pub quality: u8,
    pub has_step_hz: bool,
}
pub(crate) unsafe fn read(ptr: *const RLMeshView) -> Result<Option<View>, CapiError> {
    let Some(v) = (unsafe { sizing::read(ptr, "RLMeshView") })? else {
        return Ok(None);
    };
    Ok(Some(View {
        backend: match v.backend {
            1 => ViewBackend::Terminal,
            2 => ViewBackend::Http,
            3 => ViewBackend::Both,
            _ => return Err(CapiError::invalid_arg("invalid View backend")),
        },
        format: match v.format {
            1 => ViewFormat::Jpeg,
            2 => ViewFormat::Png,
            _ => return Err(CapiError::invalid_arg("invalid View format")),
        },
        hold: match v.hold {
            1 => ViewHold::Off,
            2 => ViewHold::UntilQuit,
            3 => {
                if !v.hold_seconds.is_finite() || v.hold_seconds < 0.0 {
                    return Err(CapiError::invalid_arg(
                        "hold_seconds must be finite and non-negative",
                    ));
                }
                ViewHold::For(Duration::try_from_secs_f64(v.hold_seconds).unwrap_or(Duration::MAX))
            }
            _ => return Err(CapiError::invalid_arg("invalid View hold")),
        },
        source: unsafe { ffi::text(v.source) }?,
        fps: v.fps,
        port: v.port,
        quality: v.quality,
        step_hz: v.has_step_hz.then_some(v.step_hz),
    }))
}
unsafe fn required(ptr: *const RLMeshView) -> Result<View, CapiError> {
    unsafe { read(ptr) }?.ok_or_else(|| CapiError::invalid_arg("null View"))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_view_validate(ptr: *const RLMeshView) -> RLMeshStatus {
    guard(|| unsafe { required(ptr) }?.validate().map_err(Into::into))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_view_parse(
    text: *const c_char,
    out: *mut RLMeshView,
) -> RLMeshStatus {
    guard(|| {
        let text =
            unsafe { ffi::text(text) }?.ok_or_else(|| CapiError::invalid_arg("null text"))?;
        let v = View::parse(&text)?;
        // Shorthand parsing never supplies source or hold/step rate options.
        let parsed = RLMeshView {
            struct_size: std::mem::size_of::<RLMeshView>(),
            source: std::ptr::null(),
            fps: v.fps,
            hold_seconds: 0.0,
            step_hz: 0.0,
            backend: match v.backend {
                ViewBackend::Terminal => 1,
                ViewBackend::Http => 2,
                ViewBackend::Both => 3,
            },
            format: match v.format {
                ViewFormat::Jpeg => 1,
                ViewFormat::Png => 2,
            },
            hold: 1,
            port: v.port,
            quality: v.quality,
            has_step_hz: false,
        };
        unsafe { crate::adapters::out_struct(out, parsed, "RLMeshView") }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_view_equals(
    a: *const RLMeshView,
    b: *const RLMeshView,
    out: *mut bool,
) -> RLMeshStatus {
    guard(|| unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(out, { required(a) }? == { required(b) }?)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_view_repr(
    ptr: *const RLMeshView,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, unsafe { required(ptr) }?.to_string()))
}
