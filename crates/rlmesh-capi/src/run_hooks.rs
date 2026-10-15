//! Run observers and retained facade contexts.
#![allow(unsafe_code)]
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard, guard_ptr, guard_value},
};
use crate::codec::RLMeshBytes;
use crate::contract::RLMeshEnvContract;
use crate::ffi;
use crate::run_results::{RLMeshEpisodeResult, RLMeshRunResult};
use crate::value::handle::{RLMeshValue, value_ref, write_out};
use rlmesh::{RunContext, RunHooks, StepEvent, Value};
use std::ffi::c_void;
use std::sync::{Arc, Mutex, MutexGuard};
#[derive(Clone)]
pub struct RLMeshRunContext(pub(crate) Arc<dyn RunContext>);
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshRunContextVtable {
    pub struct_size: usize,
    pub contract: Option<unsafe extern "C" fn(*mut c_void, *mut *mut RLMeshEnvContract) -> i32>,
    pub num_envs: Option<unsafe extern "C" fn(*mut c_void, *mut u64) -> i32>,
    pub read: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const RLMeshValue,
            *const crate::run_read::RLMeshReadItem,
            *mut *mut RLMeshValue,
        ) -> i32,
    >,
    pub frame_sources: Option<
        unsafe extern "C" fn(*mut c_void, *mut *mut crate::run_read::RLMeshFrameSources) -> i32,
    >,
    pub free_user_data: Option<unsafe extern "C" fn(*mut c_void)>,
}
struct Context {
    v: RLMeshRunContextVtable,
    user: usize,
    busy: Mutex<()>,
}
impl Drop for Context {
    fn drop(&mut self) {
        if let Some(f) = self.v.free_user_data {
            unsafe { f(self.user as *mut c_void) }
        }
    }
}
impl RunContext for Context {
    fn read(&self, observation: &Value, item: &rlmesh::ReadItem) -> rlmesh::Result<Value> {
        let _lock = self
            .busy
            .try_lock()
            .map_err(|_| rlmesh::Error::Runtime("context callback is already active".into()))?;
        let observation = RLMeshValue::new(observation.clone());
        let item = crate::run_read::RLMeshReadItem(item.clone());
        let mut out = std::ptr::null_mut();
        crate::abi::status::clear_last_error();
        let status = unsafe {
            self.v.read.expect("validated read slot")(
                self.user as *mut c_void,
                &observation,
                &item,
                &mut out,
            )
        };
        let value = unsafe { ffi::take(out) };
        if status != 0 {
            return Err(ffi::callback_error("context"));
        }
        value.ok_or_else(|| rlmesh::Error::Type("read callback returned no Value".into()))
    }
    fn frame_sources(&self) -> rlmesh::Result<rlmesh::FrameSources<'_>> {
        let _lock = self
            .busy
            .try_lock()
            .map_err(|_| rlmesh::Error::Runtime("context callback is already active".into()))?;
        let mut out = std::ptr::null_mut();
        crate::abi::status::clear_last_error();
        let status = unsafe {
            self.v.frame_sources.expect("validated frame_sources slot")(
                self.user as *mut c_void,
                &mut out,
            )
        };
        let value = if out.is_null() {
            None
        } else {
            Some(unsafe { Box::from_raw(out) })
        };
        if status != 0 {
            return Err(ffi::callback_error("context"));
        }
        Ok(value
            .ok_or_else(|| rlmesh::Error::Type("frame_sources callback returned no handle".into()))?
            .into_facade())
    }
    fn contract(&self) -> rlmesh::Result<Option<rlmesh::EnvContract>> {
        let _lock = self
            .busy
            .try_lock()
            .map_err(|_| rlmesh::Error::Runtime("context callback is already active".into()))?;
        let mut out = std::ptr::null_mut();
        crate::abi::status::clear_last_error();
        let status = unsafe {
            self.v.contract.expect("validated context callback")(self.user as *mut c_void, &mut out)
        };
        let out = if out.is_null() {
            None
        } else {
            Some(unsafe { Box::from_raw(out) })
        };
        if status != 0 {
            return Err(ffi::callback_error("context"));
        }
        out.map(|p| {
            rlmesh::binding::env_contract_from_wire(
                unsafe { RLMeshEnvContract::as_ref(&*p) }.expect("owned nonnull contract"),
            )
        })
        .transpose()
    }
    fn num_envs(&self) -> rlmesh::Result<u64> {
        let _lock = self
            .busy
            .try_lock()
            .map_err(|_| rlmesh::Error::Runtime("context callback is already active".into()))?;
        let mut n = 0;
        crate::abi::status::clear_last_error();
        if unsafe {
            self.v.num_envs.expect("validated context callback")(self.user as *mut c_void, &mut n)
        } != 0
        {
            return Err(ffi::callback_error("context"));
        }
        Ok(n)
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_context_new(
    v: *const RLMeshRunContextVtable,
    user: *mut c_void,
    out: *mut *mut RLMeshRunContext,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let v = unsafe { sizing::read(v, "RLMeshRunContextVtable") }?
            .ok_or_else(|| CapiError::invalid_arg("null vtable"))?;
        if v.contract.is_none()
            || v.num_envs.is_none()
            || v.read.is_none()
            || v.frame_sources.is_none()
        {
            return Err(CapiError::invalid_arg(
                "context requires contract, num_envs, read and frame_sources callbacks",
            ));
        }
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshRunContext(Arc::new(Context {
                v,
                user: user as usize,
                busy: Mutex::new(()),
            })))),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_context_clone(
    p: *const RLMeshRunContext,
) -> *mut RLMeshRunContext {
    guard_ptr(|| {
        Ok(Box::into_raw(Box::new(
            unsafe { p.as_ref() }
                .ok_or_else(|| CapiError::invalid_arg("null context"))?
                .clone(),
        )))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_context_free(p: *mut RLMeshRunContext) {
    guard_value((), || {
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) });
        }
    });
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_context_contract(
    p: *const RLMeshRunContext,
    out: *mut *mut RLMeshEnvContract,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let c = unsafe { p.as_ref() }
            .ok_or_else(|| CapiError::invalid_arg("null context"))?
            .0
            .contract()?;
        if let Some(c) = c {
            write_out(
                out,
                Box::into_raw(Box::new(RLMeshEnvContract::new(
                    rlmesh::binding::env_contract_to_wire(&c)?,
                ))),
            )?;
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_context_num_envs(
    p: *const RLMeshRunContext,
    out: *mut u64,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        write_out(
            out,
            unsafe { p.as_ref() }
                .ok_or_else(|| CapiError::invalid_arg("null context"))?
                .0
                .num_envs()?,
        )
    })
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshStepEvent {
    pub struct_size: usize,
    pub episode: i64,
    pub has_seed: bool,
    pub seed: i64,
    pub step: i64,
    pub observation: *const RLMeshValue,
    pub action: *const RLMeshValue,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info: *const RLMeshValue,
    pub predict_ms: f64,
    pub step_ms: f64,
    pub context: *const RLMeshRunContext,
}
unsafe fn event(p: *const RLMeshStepEvent) -> Result<StepEvent, CapiError> {
    let e = unsafe { sizing::read(p, "RLMeshStepEvent") }?
        .ok_or_else(|| CapiError::invalid_arg("null event"))?;
    Ok(StepEvent {
        episode: e.episode,
        seed: e.has_seed.then_some(e.seed),
        step: e.step,
        observation: value_ref(e.observation)?.clone(),
        action: value_ref(e.action)?.clone(),
        reward: e.reward,
        terminated: e.terminated,
        truncated: e.truncated,
        info: value_ref(e.info)?.as_dict()?.clone(),
        predict_ms: e.predict_ms,
        step_ms: e.step_ms,
        read: {
            let context = unsafe { e.context.as_ref() }.cloned();
            let observation = value_ref(e.observation)?.clone();
            Arc::new(move |item| {
                context
                    .as_ref()
                    .ok_or_else(|| rlmesh::Error::Runtime("event has no read context".into()))?
                    .0
                    .read(&observation, item)
            })
        },
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_step_event_repr(
    p: *const RLMeshStepEvent,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, unsafe { event(p) }?.to_string()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_step_event_equals(
    a: *const RLMeshStepEvent,
    b: *const RLMeshStepEvent,
    out: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        let left = unsafe { event(a) }?;
        let mut right = unsafe { event(b) }?;
        // A caller-built C event's read callable is represented by its context
        // and observation. Preserve that identity while comparing two views.
        if unsafe { (*a).context == (*b).context && (*a).observation == (*b).observation } {
            right.read = Arc::clone(&left.read);
        }
        write_out(out, left == right)
    })
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshRunHooksVtable {
    pub struct_size: usize,
    pub on_run_start: Option<unsafe extern "C" fn(*mut c_void, *const RLMeshRunContext) -> i32>,
    pub on_episode_start: Option<unsafe extern "C" fn(*mut c_void, i64, *const i64) -> i32>,
    pub on_step: Option<unsafe extern "C" fn(*mut c_void, *const RLMeshStepEvent) -> i32>,
    pub on_episode_end:
        Option<unsafe extern "C" fn(*mut c_void, *const RLMeshEpisodeResult) -> i32>,
    pub on_run_end: Option<unsafe extern "C" fn(*mut c_void, *const RLMeshRunResult) -> i32>,
    pub free_user_data: Option<unsafe extern "C" fn(*mut c_void)>,
}
pub struct RLMeshRunHooks(Mutex<Hooks>);
pub(crate) struct Hooks {
    v: RLMeshRunHooksVtable,
    user: usize,
    context: Option<RLMeshRunContext>,
    native: Option<Box<dyn RunHooks>>,
}
impl Drop for Hooks {
    fn drop(&mut self) {
        if let Some(f) = self.v.free_user_data {
            unsafe { f(self.user as *mut c_void) }
        }
    }
}
fn callback(s: i32) -> rlmesh::Result<()> {
    if s == 0 {
        Ok(())
    } else {
        Err(ffi::callback_error("hook"))
    }
}
impl RunHooks for Hooks {
    fn on_run_start(&mut self, context: Arc<dyn RunContext>) -> rlmesh::Result<()> {
        if let Some(h) = &mut self.native {
            return h.on_run_start(context);
        }
        self.context = Some(RLMeshRunContext(context));
        if let Some(f) = self.v.on_run_start {
            callback(unsafe {
                f(
                    self.user as *mut c_void,
                    self.context.as_ref().expect("context was just installed"),
                )
            })?;
        }
        Ok(())
    }
    fn on_episode_start(&mut self, e: i64, s: Option<i64>) -> rlmesh::Result<()> {
        if let Some(h) = &mut self.native {
            return h.on_episode_start(e, s);
        }
        if let Some(f) = self.v.on_episode_start {
            callback(unsafe {
                f(
                    self.user as *mut c_void,
                    e,
                    s.as_ref().map_or(std::ptr::null(), std::ptr::from_ref),
                )
            })?;
        }
        Ok(())
    }
    fn on_step(&mut self, e: &StepEvent) -> rlmesh::Result<()> {
        if let Some(h) = &mut self.native {
            return h.on_step(e);
        }
        let obs = RLMeshValue::new(e.observation.clone());
        let action = RLMeshValue::new(e.action.clone());
        let info = RLMeshValue::new(Value::Dict(e.info.clone()));
        let lent = RLMeshStepEvent {
            struct_size: std::mem::size_of::<RLMeshStepEvent>(),
            episode: e.episode,
            has_seed: e.seed.is_some(),
            seed: e.seed.unwrap_or_default(),
            step: e.step,
            observation: &obs,
            action: &action,
            reward: e.reward,
            terminated: e.terminated,
            truncated: e.truncated,
            info: &info,
            predict_ms: e.predict_ms,
            step_ms: e.step_ms,
            context: self
                .context
                .as_ref()
                .map_or(std::ptr::null(), std::ptr::from_ref),
        };
        if let Some(f) = self.v.on_step {
            callback(unsafe { f(self.user as *mut c_void, &lent) })?;
        }
        Ok(())
    }
    fn on_episode_end(&mut self, e: &rlmesh::EpisodeResult) -> rlmesh::Result<()> {
        if let Some(h) = &mut self.native {
            return h.on_episode_end(e);
        }
        if let Some(f) = self.v.on_episode_end {
            callback(unsafe { f(self.user as *mut c_void, &RLMeshEpisodeResult::from_core(e)) })?;
        }
        Ok(())
    }
    fn on_run_end(&mut self, r: &rlmesh::RunResult) -> rlmesh::Result<()> {
        if let Some(h) = &mut self.native {
            return h.on_run_end(r);
        }
        if let Some(f) = self.v.on_run_end {
            callback(unsafe { f(self.user as *mut c_void, &RLMeshRunResult::new(r.clone())) })?;
        }
        Ok(())
    }
}
pub(crate) unsafe fn borrow<'a>(
    p: *mut RLMeshRunHooks,
) -> Result<Option<MutexGuard<'a, Hooks>>, CapiError> {
    unsafe { p.as_ref() }
        .map(|h| {
            h.0.try_lock().map_err(|_| {
                CapiError::from(rlmesh::Error::Runtime(
                    "run hooks are already active".into(),
                ))
            })
        })
        .transpose()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_hooks_new(
    v: *const RLMeshRunHooksVtable,
    user: *mut c_void,
    out: *mut *mut RLMeshRunHooks,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let v = unsafe { sizing::read(v, "RLMeshRunHooksVtable") }?
            .ok_or_else(|| CapiError::invalid_arg("null vtable"))?;
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshRunHooks(Mutex::new(Hooks {
                v,
                user: user as usize,
                context: None,
                native: None,
            })))),
        )
    })
}
pub(crate) fn native(hooks: Box<dyn RunHooks>) -> RLMeshRunHooks {
    RLMeshRunHooks(Mutex::new(Hooks {
        v: RLMeshRunHooksVtable {
            struct_size: std::mem::size_of::<RLMeshRunHooksVtable>(),
            on_run_start: None,
            on_episode_start: None,
            on_step: None,
            on_episode_end: None,
            on_run_end: None,
            free_user_data: None,
        },
        user: 0,
        context: None,
        native: Some(hooks),
    }))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_hooks_free(p: *mut RLMeshRunHooks) {
    guard_value((), || {
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) });
        }
    });
}
fn hooks<'a>(p: *mut RLMeshRunHooks) -> Result<MutexGuard<'a, Hooks>, CapiError> {
    unsafe { borrow(p) }?.ok_or_else(|| CapiError::invalid_arg("null hooks"))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_hooks_on_run_start(
    p: *mut RLMeshRunHooks,
    c: *const RLMeshRunContext,
) -> RLMeshStatus {
    guard(|| {
        hooks(p)?
            .on_run_start(Arc::clone(
                &unsafe { c.as_ref() }
                    .ok_or_else(|| CapiError::invalid_arg("null context"))?
                    .0,
            ))
            .map_err(Into::into)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_hooks_on_episode_start(
    p: *mut RLMeshRunHooks,
    e: i64,
    s: *const i64,
) -> RLMeshStatus {
    guard(|| {
        hooks(p)?
            .on_episode_start(e, unsafe { s.as_ref() }.copied())
            .map_err(Into::into)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_hooks_on_step(
    p: *mut RLMeshRunHooks,
    e: *const RLMeshStepEvent,
) -> RLMeshStatus {
    guard(|| {
        let core = unsafe { event(e) }?;
        let mut h = hooks(p)?;
        let context = unsafe { (*e).context.as_ref() }.cloned();
        h.context = context;
        h.on_step(&core).map_err(Into::into)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_hooks_on_episode_end(
    p: *mut RLMeshRunHooks,
    e: *const RLMeshEpisodeResult,
) -> RLMeshStatus {
    guard(|| {
        hooks(p)?
            .on_episode_end(&unsafe { RLMeshEpisodeResult::read(e) }?)
            .map_err(Into::into)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_hooks_on_run_end(
    p: *mut RLMeshRunHooks,
    r: *const RLMeshRunResult,
) -> RLMeshStatus {
    guard(|| {
        hooks(p)?
            .on_run_end(crate::run_results::require(r)?)
            .map_err(Into::into)
    })
}
