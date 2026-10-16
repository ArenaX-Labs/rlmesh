//! Sized C Describe options over the facade's shared envelope implementation.
#![allow(unsafe_code)]
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard},
};
use crate::codec::RLMeshBytes;
use crate::env_api::{Claim, RLMeshEnv, RLMeshVectorEnv};
use crate::env_factory::{RLMeshEnvFactory, borrow_factory};
use crate::env_server::RLMeshEnvServer;
use crate::ffi;
use crate::model_api::{self, RLMeshModel, RLMeshModelVtable};
use crate::value::handle::{RLMeshValue, into_handle, write_out};
use std::ffi::c_char;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshDescribeOptions {
    pub struct_size: usize,
    pub generated_at: *const c_char,
    pub kind: i32,
    pub has_kind: bool,
    pub _reserved: [u8; 3],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshDescribeJsonOptions {
    pub struct_size: usize,
    pub generated_at: *const c_char,
    pub served_env: *const *const RLMeshEnv,
    pub served_env_len: usize,
    pub served_vector_env: *const RLMeshVectorEnv,
    pub workflow_edition: *const c_char,
    pub kind: i32,
    pub has_kind: bool,
    pub _reserved: [u8; 3],
}
fn kind(has: bool, k: i32) -> Result<Option<rlmesh::DescribeKind>, CapiError> {
    if !has {
        return Ok(None);
    }
    match k {
        1 => Ok(Some(rlmesh::DescribeKind::Env)),
        2 => Ok(Some(rlmesh::DescribeKind::Model)),
        _ => Err(CapiError::invalid_arg("invalid DescribeKind")),
    }
}
unsafe fn options(o: *const RLMeshDescribeOptions) -> Result<rlmesh::DescribeOptions, CapiError> {
    let Some(o) = unsafe { sizing::read(o, "RLMeshDescribeOptions") }? else {
        return Ok(Default::default());
    };
    if o._reserved != [0; 3] {
        return Err(CapiError::type_error(
            "RLMeshDescribeOptions reserved bytes must be zero",
        ));
    }
    Ok(rlmesh::DescribeOptions {
        kind: kind(o.has_kind, o.kind)?,
        generated_at: unsafe { ffi::text(o.generated_at) }?,
    })
}
// A served view reads attributes only and never invokes reset/step/close. The
// lease and try-lock refuse concurrent/reentrant use instead of deadlocking.
unsafe fn with_served<R>(
    o: Option<RLMeshDescribeJsonOptions>,
    f: impl FnOnce(Option<rlmesh::ServedEnv<'_>>) -> Result<R, CapiError>,
) -> Result<R, CapiError> {
    let Some(o) = o else { return f(None) };
    if !o.served_env.is_null() && !o.served_vector_env.is_null() {
        return Err(CapiError::invalid_arg(
            "served_env and served_vector_env are mutually exclusive",
        ));
    }
    if !o.served_env.is_null() || o.served_env_len != 0 {
        let lanes =
            unsafe { crate::spaces::slice_arg(o.served_env, o.served_env_len, "served_env") }?;
        if lanes.is_empty() {
            return Err(CapiError::invalid_arg(
                "served_env must hold at least one lane",
            ));
        }
        for &lane in lanes {
            if lane.is_null() {
                return Err(CapiError::invalid_arg("null served_env lane"));
            }
        }
        let env = unsafe { &*lanes[0] };
        let _claim = Claim::new(&env.leased)?;
        let guard = env.inner.try_lock().map_err(|_| {
            CapiError::from(rlmesh::Error::Runtime(
                "served env is already active".into(),
            ))
        })?;
        return f(Some(rlmesh::ServedEnv::Env(guard.as_ref())));
    }
    if let Some(env) = unsafe { o.served_vector_env.as_ref() } {
        let _claim = Claim::new(&env.leased)?;
        let guard = env.inner.try_lock().map_err(|_| {
            CapiError::from(rlmesh::Error::Runtime(
                "served vector env is already active".into(),
            ))
        })?;
        return f(Some(rlmesh::ServedEnv::Vector(guard.as_ref())));
    }
    f(None)
}
unsafe fn metadata(
    obj: rlmesh::Describable<'_>,
    o: *const RLMeshDescribeOptions,
    out: *mut *mut RLMeshValue,
) -> Result<(), CapiError> {
    unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(out, std::ptr::null_mut())
    }?;
    let result = rlmesh::binding::describe_native(obj, unsafe { options(o) }?)?;
    unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(out, into_handle(rlmesh::Value::Dict(result)))
    }
}
unsafe fn json(
    obj: rlmesh::Describable<'_>,
    o: *const RLMeshDescribeJsonOptions,
    out: *mut RLMeshBytes,
) -> Result<(), CapiError> {
    unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(out, RLMeshBytes::from_vec(Vec::new()))
    }?;
    let o = unsafe { sizing::read(o, "RLMeshDescribeJsonOptions") }?;
    let mut opts = rlmesh::DescribeJsonOptions::default();
    if let Some(o) = o {
        if o._reserved != [0; 3] {
            return Err(CapiError::type_error(
                "RLMeshDescribeJsonOptions reserved bytes must be zero",
            ));
        }
        opts.kind = kind(o.has_kind, o.kind)?;
        opts.generated_at = unsafe { ffi::text(o.generated_at) }?;
        opts.workflow_edition = unsafe { ffi::text(o.workflow_edition) }?;
    }
    let text = unsafe {
        with_served(o, |served_env| {
            Ok(rlmesh::binding::describe_native_json(
                obj,
                rlmesh::DescribeJsonOptions { served_env, ..opts },
            )?)
        })
    }?;
    ffi::bytes(out, text)
}
unsafe fn declaration(
    v: *const RLMeshModelVtable,
) -> Result<rlmesh::Describable<'static>, CapiError> {
    let v = unsafe { sizing::read(v, "RLMeshModelVtable") }?
        .ok_or_else(|| CapiError::type_error("null model vtable"))?;
    Ok(rlmesh::binding::model_declaration_describable(unsafe {
        model_api::declaration(v)
    }?)?)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_vtable_describe(
    v: *const RLMeshModelVtable,
    o: *const RLMeshDescribeOptions,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        unsafe { metadata(declaration(v)?, o, out) }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_vtable_describe_json(
    v: *const RLMeshModelVtable,
    o: *const RLMeshDescribeJsonOptions,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, RLMeshBytes::from_vec(Vec::new()))
        }?;
        unsafe { json(declaration(v)?, o, out) }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_describe(
    v: *const RLMeshModel,
    o: *const RLMeshDescribeOptions,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        unsafe { metadata((&model_api::require(v)?.model).into(), o, out) }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_describe_json(
    v: *const RLMeshModel,
    o: *const RLMeshDescribeJsonOptions,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, RLMeshBytes::from_vec(Vec::new()))
        }?;
        unsafe { json((&model_api::require(v)?.model).into(), o, out) }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_factory_describe(
    v: *mut RLMeshEnvFactory,
    o: *const RLMeshDescribeOptions,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let mut f = borrow_factory(v)?;
        let _scope = f.scope();
        unsafe { metadata(f.describable(), o, out) }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_factory_describe_json(
    v: *mut RLMeshEnvFactory,
    o: *const RLMeshDescribeJsonOptions,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, RLMeshBytes::from_vec(Vec::new()))
        }?;
        let mut f = borrow_factory(v)?;
        let _scope = f.scope();
        unsafe { json(f.describable(), o, out) }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_env_server_describe_json(
    v: *const RLMeshEnvServer,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, RLMeshBytes::from_vec(Vec::new()))
        }?;
        let server =
            unsafe { v.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null env server"))?;
        ffi::bytes(
            out,
            rlmesh::binding::env_server_native_describe_json(server.facade())?,
        )
    })
}
