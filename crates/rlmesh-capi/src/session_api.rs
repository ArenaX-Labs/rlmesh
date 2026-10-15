//! Owning C shell around the facade's borrowed session.
#![allow(unsafe_code)]
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard},
};
use crate::cancel::RLMeshExtCancelToken;
use crate::codec::RLMeshBytes;
use crate::contract::RLMeshEnvContract;
use crate::env_api::{BorrowedEnv, BorrowedVectorEnv, Claim, RLMeshEnv, RLMeshVectorEnv};
use crate::env_factory::{FactoryBorrow, Scope, borrow_factory};
use crate::ffi;
use crate::model_api::RLMeshModel;
use crate::run_api::{RLMeshEnvTarget, RLMeshModelTarget};
use crate::run_hooks::{self, RLMeshRunContext, RLMeshRunHooks};
use crate::run_results::{self, RLMeshRunResult};
use crate::value::handle::{RLMeshValue, into_handle, value_ref, write_out};
use rlmesh::{RunContext, RunHooks};
use std::cell::RefCell;
use std::ffi::c_char;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshSessionOptions {
    pub struct_size: usize,
    pub workflow_edition: *const c_char,
    pub cancel: *const RLMeshExtCancelToken,
    pub execution_horizon: u32,
    pub close_env: bool,
    pub has_trust_entrypoints: bool,
    pub trust_entrypoints: bool,
    pub _reserved: u8,
    pub instruction: *const c_char,
    pub view: *const crate::view::RLMeshView,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshSessionRunOptions {
    pub struct_size: usize,
    pub seeds: *const i64,
    pub seeds_len: usize,
    pub hooks: *mut RLMeshRunHooks,
    pub cancel: *const RLMeshExtCancelToken,
    pub episodes: u64,
    pub max_episode_steps: u64,
    pub max_episode_seconds: f64,
    pub trial_index_base: u64,
    pub has_episodes: bool,
    pub has_max_episode_steps: bool,
    pub has_max_episode_seconds: bool,
    pub _reserved: [u8; 5],
}
// Stable allocations hold the local adapters referenced by Session<'static>.
// No allocation is accessed mutably except by its session. Session is dropped
// first; then the gates stop and drain detached callbacks before releasing state.
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
    Remote(*mut crate::remote_api::Lease<rlmesh::RemoteEnv>),
    RemoteVector(*mut crate::remote_api::Lease<rlmesh::RemoteVectorEnv>),
}
// These pointers own Send allocations, exclusively driven under the session lock.
unsafe impl Send for Backing {}
impl Backing {
    unsafe fn new(t: RLMeshEnvTarget) -> Result<Self, CapiError> {
        Ok(match t.kind {
            3 => Self::Remote(Box::into_raw(Box::new(unsafe {
                crate::remote_api::borrow_env(t.handle.cast())
            }?))),
            4 => Self::RemoteVector(Box::into_raw(Box::new(unsafe {
                crate::remote_api::borrow_vector(t.handle.cast())
            }?))),
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
            Self::Remote(p) => Ok(unsafe { (*(*p)).client.as_deref_mut() }
                .expect("session lease owns client")
                .into()),
            Self::RemoteVector(p) => Ok(unsafe { (*(*p)).client.as_deref_mut() }
                .expect("session lease owns client")
                .into()),
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
            Self::Remote(p) => drop(unsafe { Box::from_raw(*p) }),
            Self::RemoteVector(p) => drop(unsafe { Box::from_raw(*p) }),
        }
    }
}
struct State {
    session: Option<rlmesh::Session<'static>>,
    backing: Backing,
}
impl Drop for State {
    fn drop(&mut self) {
        let _scope = self.backing.scope();
        drop(self.session.take());
    }
}
pub struct RLMeshSession {
    pub(crate) roles: std::sync::OnceLock<crate::run_read::RolesView>,
    state: Mutex<State>,
    done: AtomicBool,
    cancel: rlmesh::ext::CancelToken,
}
thread_local! {static HOOKS:RefCell<Vec<(usize,RLMeshRunContext)>>=const{RefCell::new(Vec::new())};}
struct HookScope;
impl Drop for HookScope {
    fn drop(&mut self) {
        HOOKS.with(|s| s.borrow_mut().pop());
    }
}
pub(crate) fn own_hook(p: *const RLMeshSession) -> Option<RLMeshRunContext> {
    HOOKS.with(|s| {
        s.borrow()
            .iter()
            .rev()
            .find(|(id, _)| *id == p as usize)
            .map(|(_, c)| c.clone())
    })
}
fn runtime(message: &str) -> CapiError {
    rlmesh::Error::Runtime(message.into()).into()
}
pub(crate) fn with<R>(
    p: *const RLMeshSession,
    f: impl FnOnce(&mut rlmesh::Session<'static>) -> Result<R, CapiError>,
) -> Result<R, CapiError> {
    if own_hook(p).is_some() {
        return Err(runtime(
            "session member is unavailable from its own run hooks",
        ));
    }
    let h = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null session"))?;
    let mut state = h
        .state
        .try_lock()
        .map_err(|_| runtime("session operation is already active"))?;
    let _scope = state.backing.scope();
    let s = state
        .session
        .as_mut()
        .expect("live C session owns its facade");
    let result = f(s);
    h.done.store(s.done(), Ordering::Release);
    result
}
struct SessionHooks<'a> {
    hooks: &'a mut dyn RunHooks,
    owner: usize,
    context: Option<RLMeshRunContext>,
}
impl SessionHooks<'_> {
    fn scope(&self) -> HookScope {
        HOOKS.with(|s| {
            s.borrow_mut().push((
                self.owner,
                self.context
                    .as_ref()
                    .expect("on_run_start precedes hooks")
                    .clone(),
            ))
        });
        HookScope
    }
}
impl RunHooks for SessionHooks<'_> {
    fn on_run_start(&mut self, c: Arc<dyn RunContext>) -> rlmesh::Result<()> {
        self.context = Some(RLMeshRunContext(Arc::clone(&c)));
        let _scope = self.scope();
        self.hooks.on_run_start(c)
    }
    fn on_episode_start(&mut self, e: i64, s: Option<i64>) -> rlmesh::Result<()> {
        let _scope = self.scope();
        self.hooks.on_episode_start(e, s)
    }
    fn on_step(&mut self, e: &rlmesh::StepEvent) -> rlmesh::Result<()> {
        let _scope = self.scope();
        self.hooks.on_step(e)
    }
    fn on_episode_end(&mut self, e: &rlmesh::EpisodeResult) -> rlmesh::Result<()> {
        let _scope = self.scope();
        self.hooks.on_episode_end(e)
    }
    fn on_run_end(&mut self, r: &rlmesh::RunResult) -> rlmesh::Result<()> {
        let _scope = self.scope();
        self.hooks.on_run_end(r)
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_session_new(
    m: RLMeshModelTarget,
    e: RLMeshEnvTarget,
    o: *const RLMeshSessionOptions,
    out: *mut *mut RLMeshSession,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let mut options = rlmesh::SessionOptions::default();
        if let Some(o) = unsafe { sizing::read(o, "RLMeshSessionOptions") }? {
            if o._reserved != 0 {
                return Err(CapiError::type_error("reserved session byte must be zero"));
            }
            options.close_env = o.close_env;
            options.trust_entrypoints = o.has_trust_entrypoints.then_some(o.trust_entrypoints);
            options.execution_horizon = o.execution_horizon;
            options.workflow_edition = unsafe { ffi::text(o.workflow_edition) }?;
            options.instruction = unsafe { ffi::text(o.instruction) }?;
            options.view = unsafe { crate::view::read(o.view) }?;
            options.cancel = unsafe { o.cancel.as_ref() }.map(|c| c.0.clone());
        }
        let model = unsafe { crate::run_api::model(m) }?;
        let backing = unsafe { Backing::new(e) }?;
        let _scope = backing.scope();
        let session = rlmesh::session(model, unsafe { backing.target(e) }?, options)?;
        let cancel = session.cancel_handle();
        let done = session.done();
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshSession {
                roles: std::sync::OnceLock::new(),
                state: Mutex::new(State {
                    session: Some(session),
                    backing,
                }),
                done: AtomicBool::new(done),
                cancel,
            })),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_session(
    m: *mut RLMeshModel,
    e: RLMeshEnvTarget,
    o: *const RLMeshSessionOptions,
    out: *mut *mut RLMeshSession,
) -> RLMeshStatus {
    unsafe { rlmesh_session_new(crate::run_api::rlmesh_model_target(m), e, o, out) }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_session_reset(
    p: *mut RLMeshSession,
    seed: *const i64,
    trial: *const i64,
    obs: *mut *mut RLMeshValue,
    info: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        write_out(obs, std::ptr::null_mut())?;
        write_out(info, std::ptr::null_mut())?;
        let (v, i) = with(p, |s| {
            Ok(s.reset(
                unsafe { seed.as_ref() }.copied(),
                unsafe { trial.as_ref() }.copied(),
            )?)
        })?;
        write_out(obs, into_handle(v))?;
        write_out(info, into_handle(rlmesh::Value::Dict(i)))
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_session_predict(
    p: *mut RLMeshSession,
    obs: *const RLMeshValue,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let v = with(p, |s| Ok(s.predict(value_ref(obs)?.clone())?))?;
        write_out(out, into_handle(v))
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_session_step(
    p: *mut RLMeshSession,
    action: *const RLMeshValue,
    obs: *mut *mut RLMeshValue,
    reward: *mut f64,
    terminated: *mut bool,
    truncated: *mut bool,
    info: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        write_out(obs, std::ptr::null_mut())?;
        write_out(info, std::ptr::null_mut())?;
        if reward.is_null() || terminated.is_null() || truncated.is_null() {
            return Err(CapiError::invalid_arg("null step output"));
        }
        let (v, r, t, u, i) = with(p, |s| Ok(s.step(value_ref(action)?.clone())?))?;
        write_out(obs, into_handle(v))?;
        write_out(reward, r)?;
        write_out(terminated, t)?;
        write_out(truncated, u)?;
        write_out(info, into_handle(rlmesh::Value::Dict(i)))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_session_run(
    p: *mut RLMeshSession,
    o: *const RLMeshSessionRunOptions,
    out: *mut *mut RLMeshRunResult,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let mut options = rlmesh::SessionRunOptions::default();
        let o = unsafe { sizing::read(o, "RLMeshSessionRunOptions") }?;
        let mut hooks = unsafe { run_hooks::borrow(o.map_or(std::ptr::null_mut(), |o| o.hooks)) }?;
        if let Some(o) = o {
            if o._reserved != [0; 5] {
                return Err(CapiError::type_error(
                    "reserved session run bytes must be zero",
                ));
            }
            options.episodes = o.has_episodes.then_some(o.episodes);
            options.seeds = if o.seeds.is_null() {
                if o.seeds_len != 0 {
                    return Err(CapiError::invalid_arg("null seeds with nonzero length"));
                }
                None
            } else {
                Some(unsafe { crate::spaces::slice_arg(o.seeds, o.seeds_len, "seeds") }?.to_vec())
            };
            options.max_episode_steps = o.has_max_episode_steps.then_some(o.max_episode_steps);
            options.max_episode = if o.has_max_episode_seconds {
                crate::run_api::seconds(o.max_episode_seconds)?
            } else {
                None
            };
            options.trial_index_base = o.trial_index_base;
            options.cancel = unsafe { o.cancel.as_ref() }.map(|c| c.0.clone());
        }
        let mut wrapper = hooks.as_deref_mut().map(|h| SessionHooks {
            hooks: h,
            owner: p as usize,
            context: None,
        });
        options.hooks = wrapper.as_mut().map(|h| h as &mut dyn RunHooks);
        let result = with(p, |s| Ok(s.run(options)?))?;
        run_results::output(out, result)
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_session_done(p: *const RLMeshSession, out: *mut bool) -> RLMeshStatus {
    guard(|| {
        if own_hook(p).is_some() {
            return Err(runtime("done is unavailable from this session's hooks"));
        }
        write_out(
            out,
            unsafe { p.as_ref() }
                .ok_or_else(|| CapiError::invalid_arg("null session"))?
                .done
                .load(Ordering::Acquire),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_session_contract(
    p: *mut RLMeshSession,
    out: *mut *mut RLMeshEnvContract,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let c = if let Some(c) = own_hook(p) {
            c.0.contract()?
                .ok_or_else(|| runtime("session is not connected"))?
        } else {
            with(p, |s| Ok(s.contract()?.clone()))?
        };
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshEnvContract::new(
                rlmesh::binding::env_contract_to_wire(&c)?,
            ))),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_session_num_envs(p: *mut RLMeshSession, out: *mut u64) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let n = if let Some(c) = own_hook(p) {
            c.0.num_envs()?
        } else {
            with(p, |s| Ok(s.num_envs()?))?
        };
        write_out(out, n)
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_session_selected_workflow_edition(
    p: *mut RLMeshSession,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        ffi::bytes(out, with(p, |s| Ok(s.selected_workflow_edition()?))?)
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_session_run_context(
    p: *mut RLMeshSession,
    out: *mut *mut RLMeshRunContext,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let c = with(p, |s| Ok(s.run_context()?))?;
        write_out(out, Box::into_raw(Box::new(RLMeshRunContext(c))))
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_session_cancel_handle(
    p: *const RLMeshSession,
    out: *mut *mut RLMeshExtCancelToken,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let h = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null session"))?;
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshExtCancelToken(h.cancel.clone()))),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_session_close(p: *mut RLMeshSession) -> RLMeshStatus {
    guard(|| with(p, |s| Ok(s.close()?)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_session_free(p: *mut RLMeshSession) {
    let _ = guard(|| {
        if own_hook(p).is_some() {
            return Err(runtime("cannot free a session from its own run hooks"));
        }
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) });
        }
        Ok(())
    });
}
