//! Owning C shell around the facade's borrowed binding: a model bound to one
//! env and stepped by hand, as Python's `Model.bind`.
#![allow(unsafe_code)]
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard},
};
use crate::codec::RLMeshBytes;
use crate::contract::RLMeshEnvContract;
use crate::env_api::{BorrowedEnv, BorrowedVectorEnv, Claim, RLMeshEnv, RLMeshVectorEnv};
use crate::env_factory::{FactoryBorrow, Scope, borrow_factory};
use crate::ffi;
use crate::model_api::RLMeshModel;
use crate::run_api::{RLMeshEnvTarget, RLMeshModelTarget};
use crate::value::handle::{RLMeshValue, into_handle, value_ref, write_out};
use std::ffi::c_char;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, Ordering},
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshBindOptions {
    pub struct_size: usize,
    pub workflow_edition: *const c_char,
    pub execution_horizon: u32,
    pub close_env: bool,
    pub _reserved: [u8; 3],
    pub hooks: *mut crate::run_hooks::RLMeshRunHooks,
}
// Stable allocations hold the local envs a Binding<'static> borrows. Only
// the binding touches them; it drops first, then the gates stop and drain
// detached callbacks before the state is released.
enum Backing {
    None,
    Env {
        ptr: *mut BorrowedEnv,
        _claim: Claim,
    },
    Vector {
        ptr: *mut BorrowedVectorEnv,
        _claim: Claim,
    },
    Factory(*mut FactoryBorrow),
}
// These pointers own Send allocations, driven only under the binding's lock.
unsafe impl Send for Backing {}
impl Backing {
    unsafe fn new(t: RLMeshEnvTarget) -> Result<Self, CapiError> {
        Ok(match t.kind {
            5 => {
                let h = unsafe { t.handle.cast::<RLMeshEnv>().as_ref() }
                    .ok_or_else(|| CapiError::invalid_arg("null env"))?;
                let claim = Claim::new(&h.leased)?;
                Self::Env {
                    ptr: Box::into_raw(Box::new(BorrowedEnv {
                        inner: Arc::clone(&h.inner),
                        stopped: Arc::new(AtomicBool::new(false)),
                    })),
                    _claim: claim,
                }
            }
            6 => {
                let h = unsafe { t.handle.cast::<RLMeshVectorEnv>().as_ref() }
                    .ok_or_else(|| CapiError::invalid_arg("null vector env"))?;
                let claim = Claim::new(&h.leased)?;
                Self::Vector {
                    ptr: Box::into_raw(Box::new(BorrowedVectorEnv {
                        inner: Arc::clone(&h.inner),
                        stopped: Arc::new(AtomicBool::new(false)),
                    })),
                    _claim: claim,
                }
            }
            7 => Self::Factory(Box::into_raw(Box::new(borrow_factory(t.handle.cast())?))),
            _ => Self::None,
        })
    }
    fn scope(&self) -> Option<Scope> {
        if let Self::Factory(p) = self {
            Some(unsafe { FactoryBorrow::scope_static(*p) })
        } else {
            None
        }
    }
    unsafe fn target(&self, t: RLMeshEnvTarget) -> Result<rlmesh::EnvTarget<'static>, CapiError> {
        match self {
            Self::Env { ptr: p, .. } => Ok(unsafe { &mut **p }.into()),
            Self::Vector { ptr: p, .. } => Ok(rlmesh::EnvTarget::vector(unsafe { &mut **p })),
            Self::Factory(p) => Ok(unsafe { FactoryBorrow::target_static(*p) }),
            Self::None => match t.kind {
                1 => Ok(unsafe { ffi::text(t.address) }?
                    .ok_or_else(|| CapiError::invalid_arg("null address"))?
                    .into()),
                2 => Ok(unsafe {
                    t.handle
                        .cast::<crate::env_server::RLMeshEnvServer>()
                        .as_ref()
                }
                .ok_or_else(|| CapiError::invalid_arg("null server"))?
                .facade()
                .into()),
                3 | 4 => Err(CapiError::type_error(
                    "binding a RemoteEnv is gone in rlmesh 0.2: an open RemoteEnv holds its \
                     env's lease, so close it and bind its address",
                )),
                _ => Err(CapiError::invalid_arg("invalid or unsupported env target")),
            },
        }
    }
}
impl Drop for Backing {
    fn drop(&mut self) {
        match self {
            Self::None => {}
            Self::Env { ptr: p, .. } => {
                let e = unsafe { Box::from_raw(*p) };
                e.stopped.store(true, Ordering::Release);
                e.finish(false);
            }
            Self::Vector { ptr: p, .. } => {
                let e = unsafe { Box::from_raw(*p) };
                e.stopped.store(true, Ordering::Release);
                e.finish(false);
            }
            Self::Factory(p) => drop(unsafe { Box::from_raw(*p) }),
        }
    }
}
/// The hooks a binding borrows, freed after the binding drops.
struct HooksBox(*mut crate::run_hooks::BoundHooks);
// SAFETY: BoundHooks is Send; only the binding (under its lock) touches it.
unsafe impl Send for HooksBox {}
impl Drop for HooksBox {
    fn drop(&mut self) {
        drop(unsafe { Box::from_raw(self.0) });
    }
}
struct State {
    binding: Option<rlmesh::Binding<'static>>,
    backing: Backing,
    /// Dropped after `binding` (whose close fires `on_run_end`).
    _hooks: Option<HooksBox>,
}
impl Drop for State {
    fn drop(&mut self) {
        let _scope = self.backing.scope();
        drop(self.binding.take());
    }
}
pub struct RLMeshBinding {
    state: Arc<Mutex<State>>,
    done: AtomicBool,
}
fn runtime(message: &str) -> CapiError {
    rlmesh::Error::Runtime(message.into()).into()
}
fn with<R>(
    p: *const RLMeshBinding,
    f: impl FnOnce(&mut rlmesh::Binding<'static>) -> Result<R, CapiError>,
) -> Result<R, CapiError> {
    let h = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null binding"))?;
    with_state(&h.state, |b| {
        let result = f(b);
        h.done.store(b.done(), Ordering::Release);
        result
    })
}
fn with_state<R>(
    state: &Mutex<State>,
    f: impl FnOnce(&mut rlmesh::Binding<'static>) -> Result<R, CapiError>,
) -> Result<R, CapiError> {
    let mut state = state
        .try_lock()
        .map_err(|_| runtime("a binding operation is already active"))?;
    let _scope = state.backing.scope();
    f(state
        .binding
        .as_mut()
        .expect("a live C binding owns its facade"))
}
/// A binding's render source that never keeps its env alive: it upgrades
/// the binding's state only for one render.
pub(crate) struct BindingRender(Weak<Mutex<State>>);
impl BindingRender {
    pub(crate) fn call(&self) -> rlmesh::Result<rlmesh::Value> {
        let state = self
            .0
            .upgrade()
            .ok_or_else(|| rlmesh::Error::Runtime("the binding has been freed".into()))?;
        let mut state = state
            .try_lock()
            .map_err(|_| rlmesh::Error::Runtime("a binding operation is already active".into()))?;
        let _scope = state.backing.scope();
        state
            .binding
            .as_mut()
            .expect("a live C binding owns its facade")
            .render()
    }
}
pub(crate) fn render_source(p: *const RLMeshBinding) -> Result<BindingRender, CapiError> {
    let h = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null binding"))?;
    Ok(BindingRender(Arc::downgrade(&h.state)))
}
pub(crate) fn read(
    p: *const RLMeshBinding,
    observation: &rlmesh::Value,
    item: &rlmesh::ReadItem,
) -> Result<rlmesh::Value, CapiError> {
    with(p, |b| Ok(b.read(observation, item.clone())?))
}
pub(crate) fn frame_sources(
    p: *const RLMeshBinding,
) -> Result<(Vec<String>, Option<String>, bool), CapiError> {
    with(p, |b| {
        let sources = b.frame_sources()?;
        Ok((
            sources.roles,
            sources.render_label,
            sources.render.is_some(),
        ))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_bind(
    m: *mut RLMeshModel,
    e: RLMeshEnvTarget,
    o: *const RLMeshBindOptions,
    out: *mut *mut RLMeshBinding,
) -> RLMeshStatus {
    unsafe { rlmesh_bind(crate::run_api::rlmesh_model_target(m), e, o, out) }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_bind(
    m: RLMeshModelTarget,
    e: RLMeshEnvTarget,
    o: *const RLMeshBindOptions,
    out: *mut *mut RLMeshBinding,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let mut options = rlmesh::BindOptions::default();
        let mut hooks = None;
        if let Some(o) = unsafe { sizing::read(o, "RLMeshBindOptions") }? {
            if o._reserved != [0; 3] {
                return Err(CapiError::type_error("reserved bind bytes must be zero"));
            }
            options.close_env = o.close_env;
            options.execution_horizon = o.execution_horizon;
            options.workflow_edition = unsafe { ffi::text(o.workflow_edition) }?;
            if !o.hooks.is_null() {
                let bound = HooksBox(Box::into_raw(Box::new(unsafe {
                    crate::run_hooks::BoundHooks::new(o.hooks)
                })));
                // SAFETY: the box outlives the binding (State drops it after).
                options.hooks = Some(unsafe { &mut *bound.0 });
                hooks = Some(bound);
            }
        }
        let model = unsafe { crate::run_api::model(m) }?;
        let backing = unsafe { Backing::new(e) }?;
        let _scope = backing.scope();
        let binding = rlmesh::bind(model, unsafe { backing.target(e) }?, options)?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                Box::into_raw(Box::new(RLMeshBinding {
                    done: AtomicBool::new(binding.done()),
                    state: Arc::new(Mutex::new(State {
                        binding: Some(binding),
                        backing,
                        _hooks: hooks,
                    })),
                })),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_binding_reset(
    p: *mut RLMeshBinding,
    seed: *const i64,
    options: *const RLMeshValue,
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
        let (v, i) = with(p, |b| {
            Ok(b.reset(unsafe { seed.as_ref() }.copied(), options.as_ref())?)
        })?;
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
pub unsafe extern "C" fn rlmesh_binding_step(
    p: *mut RLMeshBinding,
    action: *const RLMeshValue,
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
        let action = if action.is_null() {
            None
        } else {
            Some(
                unsafe {
                    // SAFETY: the FFI contract supplies live handles and writable output storage.
                    value_ref(action)
                }?
                .clone(),
            )
        };
        let (v, r, t, u, i) = with(p, |b| Ok(b.step(action)?))?;
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
pub extern "C" fn rlmesh_binding_done(p: *const RLMeshBinding, out: *mut bool) -> RLMeshStatus {
    guard(|| unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(
            out,
            { p.as_ref() }
                .ok_or_else(|| CapiError::invalid_arg("null binding"))?
                .done
                .load(Ordering::Acquire),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_binding_contract(
    p: *mut RLMeshBinding,
    out: *mut *mut RLMeshEnvContract,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let c = with(p, |b| Ok(b.contract().clone()))?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                Box::into_raw(Box::new(RLMeshEnvContract::new(
                    rlmesh::binding::env_contract_to_wire(&c)?,
                ))),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_binding_num_envs(p: *mut RLMeshBinding, out: *mut u64) -> RLMeshStatus {
    guard(|| {
        let n = with(p, |b| Ok(b.num_envs()))?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, n)
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_binding_render_mode(
    p: *mut RLMeshBinding,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let mode = with(p, |b| Ok(b.render_mode().to_string()))?;
        ffi::bytes(out, mode)
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_binding_render(
    p: *mut RLMeshBinding,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let frame = with(p, |b| Ok(b.render()?))?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, into_handle(frame))
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_binding_close(p: *mut RLMeshBinding) -> RLMeshStatus {
    guard(|| with(p, |b| Ok(b.close()?)))
}
/// NULL no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_binding_free(p: *mut RLMeshBinding) {
    let _ = guard(|| {
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) });
        }
        Ok(())
    });
}
