//! Remote clients delegate I/O and address/timeout rules to the facade.
#![allow(unsafe_code)]
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard, guard_value},
};
use crate::cancel::RLMeshExtCancelToken;
use crate::codec::RLMeshBytes;
use crate::contract::RLMeshEnvContract;
use crate::ffi;
use crate::run_api::RLMeshEnvTarget;
use crate::space_handle::RLMeshSpace;
use crate::spaces::RLMeshSpaceSpec;
use crate::value::handle::{RLMeshValue, into_handle, value_ref, write_out};
use rlmesh::{RemoteEnv, RemoteEnvOptions, RemoteVectorEnv};
use std::ffi::c_char;
use std::sync::{Arc, Mutex};
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshRemoteEnvOptions {
    pub struct_size: usize,
    pub address: *const c_char,
    pub host: *const c_char,
    pub has_port: bool,
    pub port: u16,
    pub path: *const c_char,
    pub has_transport: bool,
    pub transport: i32,
    pub has_connect_timeout_seconds: bool,
    pub connect_timeout_seconds: f64,
    pub has_request_timeout_seconds: bool,
    pub request_timeout_seconds: f64,
    pub workflow_edition: *const c_char,
    /// Appended after ABI 5; starts at the ABI-5 sizeof on every target.
    pub _reserved: u64,
    /// Appended after ABI 5: read only when `struct_size` covers it.
    pub api_key: *const c_char,
    /// Appended after ABI 5: read only when `struct_size` covers it.
    pub tls_ca_file: *const c_char,
}
fn transport(has: bool, t: i32) -> Result<Option<rlmesh::Transport>, CapiError> {
    if !has {
        return Ok(None);
    }
    match t {
        1 => Ok(Some(rlmesh::Transport::Tcp)),
        2 => Ok(Some(rlmesh::Transport::Unix)),
        _ => Err(CapiError::invalid_arg("invalid transport")),
    }
}
unsafe fn env_options(p: *const RLMeshRemoteEnvOptions) -> Result<RemoteEnvOptions, CapiError> {
    let Some(o) = unsafe { sizing::read(p, "RLMeshRemoteEnvOptions") }? else {
        return Ok(RemoteEnvOptions::default());
    };
    Ok(RemoteEnvOptions {
        address: unsafe { ffi::text(o.address) }?,
        host: unsafe { ffi::text(o.host) }?,
        port: o.has_port.then_some(o.port),
        path: unsafe { ffi::text(o.path) }?,
        transport: transport(o.has_transport, o.transport)?,
        connect_timeout: rlmesh::binding::env::client_timeout_from_secs(
            o.has_connect_timeout_seconds
                .then_some(o.connect_timeout_seconds),
            "connect_timeout_seconds",
        )?,
        request_timeout: rlmesh::binding::env::client_timeout_from_secs(
            o.has_request_timeout_seconds
                .then_some(o.request_timeout_seconds),
            "request_timeout_seconds",
        )?,
        workflow_edition: unsafe { ffi::text(o.workflow_edition) }?,
        api_key: unsafe { ffi::text(o.api_key) }?,
        tls_ca_file: unsafe { ffi::text(o.tls_ca_file) }?.map(Into::into),
    })
}
struct Common {
    address: String,
    env_id: String,
    render_mode: Option<String>,
    contract: RLMeshEnvContract,
    metadata: RLMeshValue,
    observation: RLMeshSpace,
    action: RLMeshSpace,
    observation_spec: RLMeshSpaceSpec,
    action_spec: RLMeshSpaceSpec,
    num_envs: u64,
    edition: rlmesh::binding::env::EditionCell,
    cancel: rlmesh::ext::CancelToken,
}
#[allow(clippy::too_many_arguments)]
fn common(
    address: &str,
    env_id: &str,
    c: &rlmesh::EnvContract,
    metadata: &rlmesh::Metadata,
    observation: &rlmesh::spaces::Space,
    action: &rlmesh::spaces::Space,
    edition: rlmesh::binding::env::EditionCell,
    cancel: rlmesh::ext::CancelToken,
) -> Result<Common, CapiError> {
    Ok(Common {
        address: address.into(),
        env_id: env_id.into(),
        render_mode: c.render_mode().map(str::to_owned),
        contract: RLMeshEnvContract::new(rlmesh::binding::env_contract_to_wire(c)?),
        metadata: RLMeshValue::new(rlmesh::Value::Dict(metadata.clone())),
        observation: RLMeshSpace(observation.clone()),
        action: RLMeshSpace(action.clone()),
        observation_spec: RLMeshSpaceSpec(
            rlmesh::binding::space_spec_wire(c.observation_space()).clone(),
        ),
        action_spec: RLMeshSpaceSpec(rlmesh::binding::space_spec_wire(c.action_space()).clone()),
        num_envs: c.num_envs(),
        edition,
        cancel,
    })
}
pub struct RLMeshRemoteEnv {
    common: Common,
    client: Arc<Mutex<Option<RemoteEnv>>>,
}
pub struct RLMeshRemoteVectorEnv {
    common: Common,
    client: Arc<Mutex<Option<RemoteVectorEnv>>>,
}
// A short operation and a run use the same exclusive borrow. Immutable
// accessors live outside it; the edition is the facade's shared pin cell.
pub(crate) struct Lease<T> {
    pub(crate) client: Option<Box<T>>,
    owner: Arc<Mutex<Option<T>>>,
}
impl<T> Lease<T> {
    fn new(owner: &Arc<Mutex<Option<T>>>) -> Result<Self, CapiError> {
        let client = owner
            .try_lock()
            .map_err(|_| busy())?
            .take()
            .ok_or_else(busy)?;
        Ok(Self {
            client: Some(Box::new(client)),
            owner: Arc::clone(owner),
        })
    }
    pub(crate) fn get(&mut self) -> &mut T {
        self.client.as_deref_mut().expect("live lease owns client")
    }
}
impl<T> Drop for Lease<T> {
    fn drop(&mut self) {
        *self.owner.lock().unwrap_or_else(|e| e.into_inner()) = self.client.take().map(|p| *p);
    }
}
fn busy() -> CapiError {
    rlmesh::Error::Runtime("remote client I/O is already active or borrowed by a run".into()).into()
}
pub(crate) unsafe fn borrow_env(p: *const RLMeshRemoteEnv) -> Result<Lease<RemoteEnv>, CapiError> {
    Lease::new(
        &unsafe { p.as_ref() }
            .ok_or_else(|| CapiError::invalid_arg("null remote env"))?
            .client,
    )
}
pub(crate) unsafe fn borrow_vector(
    p: *const RLMeshRemoteVectorEnv,
) -> Result<Lease<RemoteVectorEnv>, CapiError> {
    Lease::new(
        &unsafe { p.as_ref() }
            .ok_or_else(|| CapiError::invalid_arg("null remote vector env"))?
            .client,
    )
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_env_new(
    o: *const RLMeshRemoteEnvOptions,
    out: *mut *mut RLMeshRemoteEnv,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let c = RemoteEnv::new(unsafe { env_options(o) }?)?;
        env_output(c, out)
    })
}
fn env_output(c: RemoteEnv, out: *mut *mut RLMeshRemoteEnv) -> Result<(), CapiError> {
    let meta = common(
        c.address(),
        c.env_id(),
        c.env_contract(),
        c.metadata(),
        c.observation_space(),
        c.action_space(),
        rlmesh::binding::env::env_core(&c)?.edition_cell(),
        c.cancel_handle(),
    )?;
    unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshRemoteEnv {
                common: meta,
                client: Arc::new(Mutex::new(Some(c))),
            })),
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_vector_env_new(
    o: *const RLMeshRemoteEnvOptions,
    out: *mut *mut RLMeshRemoteVectorEnv,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let c = RemoteVectorEnv::new(unsafe { env_options(o) }?)?;
        vector_output(c, out)
    })
}
fn vector_output(
    c: RemoteVectorEnv,
    out: *mut *mut RLMeshRemoteVectorEnv,
) -> Result<(), CapiError> {
    let meta = common(
        c.address(),
        c.env_id(),
        c.env_contract(),
        c.metadata(),
        c.single_observation_space(),
        c.single_action_space(),
        rlmesh::binding::env::vector_env_core(&c)?.edition_cell(),
        c.cancel_handle(),
    )?;
    unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshRemoteVectorEnv {
                common: meta,
                client: Arc::new(Mutex::new(Some(c))),
            })),
        )
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_remote_env_target(p: *mut RLMeshRemoteEnv) -> RLMeshEnvTarget {
    RLMeshEnvTarget {
        kind: 3,
        handle: p.cast(),
        address: std::ptr::null(),
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_remote_vector_env_target(
    p: *mut RLMeshRemoteVectorEnv,
) -> RLMeshEnvTarget {
    RLMeshEnvTarget {
        kind: 4,
        handle: p.cast(),
        address: std::ptr::null(),
    }
}
macro_rules! free {
    ($fn:ident,$ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn(p: *mut $ty) {
            guard_value((), || {
                if !p.is_null() {
                    drop(unsafe { Box::from_raw(p) });
                }
            });
        }
    };
}
free!(rlmesh_remote_env_free, RLMeshRemoteEnv);
free!(rlmesh_remote_vector_env_free, RLMeshRemoteVectorEnv);
macro_rules! common_text {
    ($f:ident,$ty:ty,$field:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $f(p: *const $ty, out: *mut RLMeshBytes) -> RLMeshStatus {
            guard(|| {
                ffi::bytes(
                    out,
                    unsafe { p.as_ref() }
                        .ok_or_else(|| CapiError::invalid_arg("null remote env"))?
                        .common
                        .$field
                        .clone(),
                )
            })
        }
    };
}
macro_rules! common_ptr {
    ($f:ident,$ty:ty,$ret:ty,$field:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $f(p: *const $ty) -> *const $ret {
            guard_value(std::ptr::null(), || {
                unsafe { p.as_ref() }
                    .map_or(std::ptr::null(), |h| std::ptr::from_ref(&h.common.$field))
            })
        }
    };
}
common_text!(rlmesh_remote_env_address, RLMeshRemoteEnv, address);
common_text!(rlmesh_remote_env_env_id, RLMeshRemoteEnv, env_id);
common_ptr!(
    rlmesh_remote_env_env_contract,
    RLMeshRemoteEnv,
    RLMeshEnvContract,
    contract
);
common_ptr!(
    rlmesh_remote_env_metadata,
    RLMeshRemoteEnv,
    RLMeshValue,
    metadata
);
common_ptr!(
    rlmesh_remote_env_observation_space,
    RLMeshRemoteEnv,
    RLMeshSpace,
    observation
);
common_ptr!(
    rlmesh_remote_env_action_space,
    RLMeshRemoteEnv,
    RLMeshSpace,
    action
);
common_ptr!(
    rlmesh_remote_env_observation_space_spec,
    RLMeshRemoteEnv,
    RLMeshSpaceSpec,
    observation_spec
);
common_ptr!(
    rlmesh_remote_env_action_space_spec,
    RLMeshRemoteEnv,
    RLMeshSpaceSpec,
    action_spec
);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_env_selected_workflow_edition(
    p: *mut RLMeshRemoteEnv,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        let h = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null remote env"))?;
        ffi::bytes(out, h.common.edition.get())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_env_render_mode(
    p: *mut RLMeshRemoteEnv,
    has: *mut bool,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        let h = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null remote env"))?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(has, h.common.render_mode.is_some())
        }?;
        if let Some(v) = &h.common.render_mode {
            ffi::bytes(out, v.clone())?;
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_env_cancel_handle(
    p: *mut RLMeshRemoteEnv,
    out: *mut *mut RLMeshExtCancelToken,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let h = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null remote env"))?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                Box::into_raw(Box::new(RLMeshExtCancelToken(h.common.cancel.clone()))),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_env_render(
    p: *mut RLMeshRemoteEnv,
    index: u64,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let r = unsafe { borrow_env(p) }?.get().render(Some(index))?;
        if let Some(v) = r {
            unsafe {
                /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
                write_out(out, into_handle(v))
            }?;
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_env_close(p: *mut RLMeshRemoteEnv) -> RLMeshStatus {
    guard(|| unsafe { borrow_env(p) }?.get().close().map_err(Into::into))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_env_shutdown(
    p: *mut RLMeshRemoteEnv,
    reason: *const c_char,
    out: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let reason = unsafe { ffi::text(reason) }?;
        let v = unsafe { borrow_env(p) }?
            .get()
            .shutdown(reason.as_deref())?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, v)
        }
    })
}
common_text!(
    rlmesh_remote_vector_env_address,
    RLMeshRemoteVectorEnv,
    address
);
common_text!(
    rlmesh_remote_vector_env_env_id,
    RLMeshRemoteVectorEnv,
    env_id
);
common_ptr!(
    rlmesh_remote_vector_env_env_contract,
    RLMeshRemoteVectorEnv,
    RLMeshEnvContract,
    contract
);
common_ptr!(
    rlmesh_remote_vector_env_metadata,
    RLMeshRemoteVectorEnv,
    RLMeshValue,
    metadata
);
common_ptr!(
    rlmesh_remote_vector_env_observation_space,
    RLMeshRemoteVectorEnv,
    RLMeshSpace,
    observation
);
common_ptr!(
    rlmesh_remote_vector_env_action_space,
    RLMeshRemoteVectorEnv,
    RLMeshSpace,
    action
);
common_ptr!(
    rlmesh_remote_vector_env_observation_space_spec,
    RLMeshRemoteVectorEnv,
    RLMeshSpaceSpec,
    observation_spec
);
common_ptr!(
    rlmesh_remote_vector_env_action_space_spec,
    RLMeshRemoteVectorEnv,
    RLMeshSpaceSpec,
    action_spec
);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_vector_env_selected_workflow_edition(
    p: *mut RLMeshRemoteVectorEnv,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        let h = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null remote env"))?;
        ffi::bytes(out, h.common.edition.get())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_vector_env_render_mode(
    p: *mut RLMeshRemoteVectorEnv,
    has: *mut bool,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        let h = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null remote env"))?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(has, h.common.render_mode.is_some())
        }?;
        if let Some(v) = &h.common.render_mode {
            ffi::bytes(out, v.clone())?;
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_vector_env_cancel_handle(
    p: *mut RLMeshRemoteVectorEnv,
    out: *mut *mut RLMeshExtCancelToken,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let h = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null remote env"))?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                Box::into_raw(Box::new(RLMeshExtCancelToken(h.common.cancel.clone()))),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_vector_env_render(
    p: *mut RLMeshRemoteVectorEnv,
    index: u64,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let r = unsafe { borrow_vector(p) }?.get().render(Some(index))?;
        if let Some(v) = r {
            unsafe {
                /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
                write_out(out, into_handle(v))
            }?;
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_vector_env_close(
    p: *mut RLMeshRemoteVectorEnv,
) -> RLMeshStatus {
    guard(|| {
        unsafe { borrow_vector(p) }?
            .get()
            .close()
            .map_err(Into::into)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_vector_env_shutdown(
    p: *mut RLMeshRemoteVectorEnv,
    reason: *const c_char,
    out: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let reason = unsafe { ffi::text(reason) }?;
        let v = unsafe { borrow_vector(p) }?
            .get()
            .shutdown(reason.as_deref())?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, v)
        }
    })
}
common_ptr!(
    rlmesh_remote_vector_env_single_observation_space,
    RLMeshRemoteVectorEnv,
    RLMeshSpace,
    observation
);
common_ptr!(
    rlmesh_remote_vector_env_single_action_space,
    RLMeshRemoteVectorEnv,
    RLMeshSpace,
    action
);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_vector_env_num_envs(p: *const RLMeshRemoteVectorEnv) -> u64 {
    guard_value(0, || unsafe { p.as_ref() }.map_or(0, |h| h.common.num_envs))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_env_reset(
    p: *mut RLMeshRemoteEnv,
    seed: *const i64,
    options: *const RLMeshValue,
    obs: *mut *mut RLMeshValue,
    info: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    unsafe { rlmesh_ext_remote_env_reset(p, seed, options, std::ptr::null(), obs, info) }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_remote_env_reset(
    p: *mut RLMeshRemoteEnv,
    seed: *const i64,
    options: *const RLMeshValue,
    cancel: *const RLMeshExtCancelToken,
    obs: *mut *mut RLMeshValue,
    info: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(obs, std::ptr::null_mut())
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(info, std::ptr::null_mut())
        }?;
        let options = ffi::dict(options)?;
        let (v, i) = rlmesh::binding::calls::remote_env_reset(
            unsafe { borrow_env(p) }?.get(),
            unsafe { seed.as_ref() }.copied(),
            options.as_ref(),
            unsafe { cancel.as_ref() }.map(|c| &c.0),
        )?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(obs, into_handle(v))
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(info, into_handle(rlmesh::Value::Dict(i)))
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_env_step(
    p: *mut RLMeshRemoteEnv,
    action: *const RLMeshValue,
    obs: *mut *mut RLMeshValue,
    reward: *mut f64,
    terminated: *mut bool,
    truncated: *mut bool,
    info: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    unsafe {
        rlmesh_ext_remote_env_step(
            p,
            action,
            std::ptr::null(),
            obs,
            reward,
            terminated,
            truncated,
            info,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_remote_env_step(
    p: *mut RLMeshRemoteEnv,
    action: *const RLMeshValue,
    cancel: *const RLMeshExtCancelToken,
    obs: *mut *mut RLMeshValue,
    reward: *mut f64,
    terminated: *mut bool,
    truncated: *mut bool,
    info: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(obs, std::ptr::null_mut())
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(info, std::ptr::null_mut())
        }?;
        if reward.is_null() || terminated.is_null() || truncated.is_null() {
            return Err(CapiError::invalid_arg("null step output"));
        }
        let (v, r, t, u, i) = rlmesh::binding::calls::remote_env_step(
            unsafe { borrow_env(p) }?.get(),
            unsafe {
                // SAFETY: the FFI contract supplies live handles and writable output storage.
                value_ref(action)
            }?
            .clone(),
            unsafe { cancel.as_ref() }.map(|c| &c.0),
        )?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(obs, into_handle(v))
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(reward, r)
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(terminated, t)
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(truncated, u)
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(info, into_handle(rlmesh::Value::Dict(i)))
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_vector_env_reset(
    p: *mut RLMeshRemoteVectorEnv,
    seeds: *const i64,
    len: usize,
    options: *const RLMeshValue,
    obs: *mut *mut RLMeshValue,
    info: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    unsafe {
        rlmesh_ext_remote_vector_env_reset(p, seeds, len, options, std::ptr::null(), obs, info)
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_remote_vector_env_reset(
    p: *mut RLMeshRemoteVectorEnv,
    seeds: *const i64,
    len: usize,
    options: *const RLMeshValue,
    cancel: *const RLMeshExtCancelToken,
    obs: *mut *mut RLMeshValue,
    info: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(obs, std::ptr::null_mut())
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(info, std::ptr::null_mut())
        }?;
        let seed = if seeds.is_null() {
            if len != 0 {
                return Err(CapiError::invalid_arg("null seeds with nonzero length"));
            }
            None
        } else {
            let seeds = unsafe { crate::spaces::slice_arg(seeds, len, "seeds") }?;
            Some(if len == 1 {
                rlmesh::VectorSeed::One(seeds[0])
            } else {
                rlmesh::VectorSeed::Each(seeds.to_vec())
            })
        };
        let options = ffi::dict(options)?;
        let (v, i) = rlmesh::binding::calls::remote_vector_env_reset(
            unsafe { borrow_vector(p) }?.get(),
            seed,
            options.as_ref(),
            unsafe { cancel.as_ref() }.map(|c| &c.0),
        )?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(obs, into_handle(v))
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(info, into_handle(rlmesh::Value::Dict(i)))
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_remote_vector_env_step(
    p: *mut RLMeshRemoteVectorEnv,
    actions: *const RLMeshValue,
    obs: *mut *mut RLMeshValue,
    rewards: *mut f64,
    terminated: *mut bool,
    truncated: *mut bool,
    cap: usize,
    len: *mut usize,
    info: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    unsafe {
        rlmesh_ext_remote_vector_env_step(
            p,
            actions,
            std::ptr::null(),
            obs,
            rewards,
            terminated,
            truncated,
            cap,
            len,
            info,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_remote_vector_env_step(
    p: *mut RLMeshRemoteVectorEnv,
    actions: *const RLMeshValue,
    cancel: *const RLMeshExtCancelToken,
    obs: *mut *mut RLMeshValue,
    rewards: *mut f64,
    terminated: *mut bool,
    truncated: *mut bool,
    cap: usize,
    len: *mut usize,
    info: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        let h = unsafe { p.as_ref() }
            .ok_or_else(|| CapiError::invalid_arg("null remote vector env"))?;
        let n = usize::try_from(h.common.num_envs)
            .map_err(|_| CapiError::invalid_arg("num_envs exceeds size_t"))?;
        if cap < n
            || rewards.is_null()
            || terminated.is_null()
            || truncated.is_null()
            || len.is_null()
            || obs.is_null()
            || info.is_null()
        {
            return Err(CapiError::invalid_arg("step buffers must cover num_envs"));
        }
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(obs, std::ptr::null_mut())
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(info, std::ptr::null_mut())
        }?;
        let (v, r, t, u, i) = rlmesh::binding::calls::remote_vector_env_step(
            unsafe { borrow_vector(p) }?.get(),
            unsafe {
                // SAFETY: the FFI contract supplies live handles and writable output storage.
                value_ref(actions)
            }?
            .clone(),
            unsafe { cancel.as_ref() }.map(|c| &c.0),
        )?;
        if r.len() != n || t.len() != n || u.len() != n {
            return Err(rlmesh::Error::Protocol("vector reply has wrong lane count".into()).into());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(r.as_ptr(), rewards, n);
            std::ptr::copy_nonoverlapping(t.as_ptr(), terminated, n);
            std::ptr::copy_nonoverlapping(u.as_ptr(), truncated, n);
        }
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(len, n)
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(obs, into_handle(v))
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(info, into_handle(rlmesh::Value::Dict(i)))
        }
    })
}
