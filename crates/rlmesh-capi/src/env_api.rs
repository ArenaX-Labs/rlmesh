//! Gymnasium-shaped C environments, projected through the Rust facade.
#![allow(unsafe_code)] // FFI callbacks, caller pointers and handle ownership.
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard, guard_value},
};
use crate::ffi;
use crate::space_handle::RLMeshSpace;
use crate::value::handle::{RLMeshValue, write_out};
use rlmesh::spaces::Space;
use rlmesh::{Env, Error, InfoDict, Metadata, Value, VectorEnv, VectorSeed};
use std::ffi::{c_char, c_void};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

type Reset = unsafe extern "C" fn(
    *mut c_void,
    *const i64,
    *const RLMeshValue,
    *mut *mut RLMeshValue,
    *mut *mut RLMeshValue,
) -> i32;
type Step = unsafe extern "C" fn(
    *mut c_void,
    *const RLMeshValue,
    *mut *mut RLMeshValue,
    *mut f64,
    *mut bool,
    *mut bool,
    *mut *mut RLMeshValue,
) -> i32;
type VectorReset = unsafe extern "C" fn(
    *mut c_void,
    *const i64,
    usize,
    *const RLMeshValue,
    *mut *mut RLMeshValue,
    *mut *mut RLMeshValue,
) -> i32;
type VectorStep = unsafe extern "C" fn(
    *mut c_void,
    *const RLMeshValue,
    *mut *mut RLMeshValue,
    *mut f64,
    *mut bool,
    *mut bool,
    *mut *mut RLMeshValue,
) -> i32;
type Render = unsafe extern "C" fn(*mut c_void, *mut *mut RLMeshValue) -> i32;
type Close = unsafe extern "C" fn(*mut c_void) -> i32;
type Free = unsafe extern "C" fn(*mut c_void);
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshEnvVtable {
    pub struct_size: usize,
    pub reset: Option<Reset>,
    pub step: Option<Step>,
    pub render: Option<Render>,
    pub close: Option<Close>,
    pub free_user_data: Option<Free>,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshVectorEnvVtable {
    pub struct_size: usize,
    pub reset: Option<VectorReset>,
    pub step: Option<VectorStep>,
    pub render: Option<Render>,
    pub close: Option<Close>,
    pub free_user_data: Option<Free>,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshEnvAttributes {
    pub struct_size: usize,
    pub observation_space: *const RLMeshSpace,
    pub action_space: *const RLMeshSpace,
    pub render_mode: *const c_char,
    pub metadata: *const RLMeshValue,
    pub id: *const c_char,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshVectorEnvAttributes {
    pub struct_size: usize,
    pub num_envs: u64,
    pub single_observation_space: *const RLMeshSpace,
    pub single_action_space: *const RLMeshSpace,
    pub render_mode: *const c_char,
    pub metadata: *const RLMeshValue,
    pub id: *const c_char,
}
#[derive(Clone)]
struct Attributes {
    obs: Space,
    act: Space,
    render_mode: Option<String>,
    metadata: Option<Metadata>,
    id: Option<String>,
}
unsafe fn attributes(
    obs: *const RLMeshSpace,
    act: *const RLMeshSpace,
    render_mode: *const c_char,
    metadata: *const RLMeshValue,
    id: *const c_char,
) -> Result<Attributes, CapiError> {
    let obs = unsafe { obs.as_ref() }
        .ok_or_else(|| CapiError::type_error("env must have 'observation_space'"))?
        .0
        .clone();
    let act = unsafe { act.as_ref() }
        .ok_or_else(|| CapiError::type_error("env must have 'action_space'"))?
        .0
        .clone();
    Ok(Attributes {
        obs,
        act,
        render_mode: unsafe { ffi::text(render_mode) }?,
        metadata: ffi::dict(metadata)?,
        id: unsafe { ffi::text(id) }?,
    })
}
struct UserData {
    ptr: *mut c_void,
    free: Option<Free>,
}
// SAFETY: constructors require that callback state may move between threads;
// a handle's mutex and exclusive lease serialize every callback.
unsafe impl Send for UserData {}
impl Drop for UserData {
    fn drop(&mut self) {
        if let Some(f) = self.free {
            unsafe { f(self.ptr) };
        }
    }
}
pub(crate) struct CEnv {
    vt: RLMeshEnvVtable,
    ud: UserData,
    attrs: Attributes,
    closed: bool,
}
pub(crate) struct CVectorEnv {
    vt: RLMeshVectorEnvVtable,
    ud: UserData,
    attrs: Attributes,
    num_envs: u64,
    closed: bool,
}
pub struct RLMeshEnv {
    pub(crate) inner: Arc<Mutex<Box<dyn Env + Send>>>,
    pub(crate) leased: Arc<AtomicBool>,
}
pub struct RLMeshVectorEnv {
    pub(crate) inner: Arc<Mutex<Box<dyn VectorEnv + Send>>>,
    pub(crate) leased: Arc<AtomicBool>,
}
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
fn open(closed: bool) -> rlmesh::Result<()> {
    if closed {
        Err(Error::Runtime("env is closed".into()))
    } else {
        Ok(())
    }
}
fn callback(status: i32) -> rlmesh::Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(ffi::callback_error("env"))
    }
}
fn read_outputs(
    status: i32,
    obs: *mut RLMeshValue,
    info: *mut RLMeshValue,
) -> rlmesh::Result<(Value, InfoDict)> {
    // Reclaim both handles before inspecting callback status.
    let (obs, info) = unsafe { (ffi::take(obs), ffi::take(info)) };
    callback(status)?;
    let obs = obs.ok_or_else(|| Error::Type("env callback returned no observation".into()))?;
    Ok((obs, ffi::info(info)?))
}
fn render(ud: &UserData, slot: Option<Render>, closed: bool) -> rlmesh::Result<Option<Value>> {
    open(closed)?;
    let Some(f) = slot else {
        return Ok(None);
    };
    let mut out = std::ptr::null_mut();
    let _callback_frame = crate::abi::status::CallbackFrame::enter();
    let status = unsafe { f(ud.ptr, &mut out) };
    let value = unsafe { ffi::take(out) };
    callback(status)?;
    Ok(value)
}
fn close(ud: &UserData, slot: Option<Close>, closed: &mut bool) -> rlmesh::Result<()> {
    if std::mem::replace(closed, true) {
        return Ok(());
    }
    if let Some(f) = slot {
        let _callback_frame = crate::abi::status::CallbackFrame::enter();
        callback(unsafe { f(ud.ptr) })?;
    }
    Ok(())
}
macro_rules! attrs {
    () => {
        fn render_mode(&self) -> Option<String> {
            self.attrs.render_mode.clone()
        }
        fn metadata(&self) -> Option<Metadata> {
            self.attrs.metadata.clone()
        }
        fn id(&self) -> Option<String> {
            self.attrs.id.clone()
        }
        fn render(&mut self) -> rlmesh::Result<Option<Value>> {
            render(&self.ud, self.vt.render, self.closed)
        }
        fn close(&mut self) -> rlmesh::Result<()> {
            close(&self.ud, self.vt.close, &mut self.closed)
        }
    };
}
impl Env for CEnv {
    fn observation_space(&self) -> Space {
        self.attrs.obs.clone()
    }
    fn action_space(&self) -> Space {
        self.attrs.act.clone()
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> rlmesh::Result<(Value, InfoDict)> {
        open(self.closed)?;
        let options = options.map(|o| RLMeshValue::new(Value::Dict(o.clone())));
        let (mut obs, mut info) = (std::ptr::null_mut(), std::ptr::null_mut());
        let _callback_frame = crate::abi::status::CallbackFrame::enter();
        let status = unsafe {
            self.vt.reset.expect("validated reset")(
                self.ud.ptr,
                seed.as_ref().map_or(std::ptr::null(), std::ptr::from_ref),
                options
                    .as_ref()
                    .map_or(std::ptr::null(), std::ptr::from_ref),
                &mut obs,
                &mut info,
            )
        };
        read_outputs(status, obs, info)
    }
    fn step(
        &mut self,
        action: Option<Value>,
    ) -> rlmesh::Result<(Value, f64, bool, bool, InfoDict)> {
        open(self.closed)?;
        let action = action.map(RLMeshValue::new);
        let (mut obs, mut info) = (std::ptr::null_mut(), std::ptr::null_mut());
        let (mut reward, mut terminated, mut truncated) = (0.0, false, false);
        let _callback_frame = crate::abi::status::CallbackFrame::enter();
        let status = unsafe {
            self.vt.step.expect("validated step")(
                self.ud.ptr,
                action.as_ref().map_or(std::ptr::null(), std::ptr::from_ref),
                &mut obs,
                &mut reward,
                &mut terminated,
                &mut truncated,
                &mut info,
            )
        };
        let (obs, info) = read_outputs(status, obs, info)?;
        Ok((obs, reward, terminated, truncated, info))
    }
    attrs!();
}
impl VectorEnv for CVectorEnv {
    fn num_envs(&self) -> u64 {
        self.num_envs
    }
    fn single_observation_space(&self) -> Space {
        self.attrs.obs.clone()
    }
    fn single_action_space(&self) -> Space {
        self.attrs.act.clone()
    }
    fn reset(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> rlmesh::Result<(Value, InfoDict)> {
        open(self.closed)?;
        let seed = seed.map(|s| match s {
            VectorSeed::One(n) => vec![n],
            VectorSeed::Each(ns) => ns,
        });
        let options = options.map(|o| RLMeshValue::new(Value::Dict(o.clone())));
        let (mut obs, mut info) = (std::ptr::null_mut(), std::ptr::null_mut());
        let _callback_frame = crate::abi::status::CallbackFrame::enter();
        let status = unsafe {
            self.vt.reset.expect("validated reset")(
                self.ud.ptr,
                seed.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
                seed.as_ref().map_or(0, Vec::len),
                options
                    .as_ref()
                    .map_or(std::ptr::null(), std::ptr::from_ref),
                &mut obs,
                &mut info,
            )
        };
        read_outputs(status, obs, info)
    }
    fn step(
        &mut self,
        actions: Value,
    ) -> rlmesh::Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        open(self.closed)?;
        let n = usize::try_from(self.num_envs)
            .map_err(|_| Error::invalid_argument("num_envs is too large"))?;
        let mut rewards = vec![0.0; n];
        let mut terminated = vec![false; n];
        let mut truncated = vec![false; n];
        let actions = RLMeshValue::new(actions);
        let (mut obs, mut info) = (std::ptr::null_mut(), std::ptr::null_mut());
        let _callback_frame = crate::abi::status::CallbackFrame::enter();
        let status = unsafe {
            self.vt.step.expect("validated step")(
                self.ud.ptr,
                &actions,
                &mut obs,
                rewards.as_mut_ptr(),
                terminated.as_mut_ptr(),
                truncated.as_mut_ptr(),
                &mut info,
            )
        };
        let (obs, info) = read_outputs(status, obs, info)?;
        Ok((obs, rewards, terminated, truncated, info))
    }
    attrs!();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_new(
    vt: *const RLMeshEnvVtable,
    user_data: *mut c_void,
    attrs: *const RLMeshEnvAttributes,
    out: *mut *mut RLMeshEnv,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let vt = unsafe { sizing::read(vt, "RLMeshEnvVtable") }?
            .ok_or_else(|| CapiError::type_error("env vtable is required"))?;
        if vt.reset.is_none() || vt.step.is_none() {
            return Err(CapiError::type_error("env must define reset and step"));
        }
        let a = unsafe { sizing::read(attrs, "RLMeshEnvAttributes") }?
            .ok_or_else(|| CapiError::type_error("env must have 'observation_space'"))?;
        let attrs = unsafe {
            attributes(
                a.observation_space,
                a.action_space,
                a.render_mode,
                a.metadata,
                a.id,
            )
        }?;
        crate::abi::ignore_sigpipe();
        let value = RLMeshEnv {
            inner: Arc::new(Mutex::new(Box::new(CEnv {
                vt,
                ud: UserData {
                    ptr: user_data,
                    free: vt.free_user_data,
                },
                attrs,
                closed: false,
            }))),
            leased: Arc::new(AtomicBool::new(false)),
        };
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, Box::into_raw(Box::new(value)))
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_vector_env_new(
    vt: *const RLMeshVectorEnvVtable,
    user_data: *mut c_void,
    attrs: *const RLMeshVectorEnvAttributes,
    out: *mut *mut RLMeshVectorEnv,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let vt = unsafe { sizing::read(vt, "RLMeshVectorEnvVtable") }?
            .ok_or_else(|| CapiError::type_error("vector env vtable is required"))?;
        if vt.reset.is_none() || vt.step.is_none() {
            return Err(CapiError::type_error(
                "vector env must define reset and step",
            ));
        }
        let a = unsafe { sizing::read(attrs, "RLMeshVectorEnvAttributes") }?
            .ok_or_else(|| CapiError::type_error("vector env must have single spaces"))?;
        let attrs = unsafe {
            attributes(
                a.single_observation_space,
                a.single_action_space,
                a.render_mode,
                a.metadata,
                a.id,
            )
        }?;
        crate::abi::ignore_sigpipe();
        let value = RLMeshVectorEnv {
            inner: Arc::new(Mutex::new(Box::new(CVectorEnv {
                vt,
                ud: UserData {
                    ptr: user_data,
                    free: vt.free_user_data,
                },
                attrs,
                num_envs: a.num_envs,
                closed: false,
            }))),
            leased: Arc::new(AtomicBool::new(false)),
        };
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, Box::into_raw(Box::new(value)))
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_close(env: *mut RLMeshEnv) -> RLMeshStatus {
    guard(|| {
        let e = unsafe { env.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null env"))?;
        let _lease = Claim::new(&e.leased)?;
        lock(&e.inner).close()?;
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_vector_env_close(env: *mut RLMeshVectorEnv) -> RLMeshStatus {
    guard(|| {
        let e = unsafe { env.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null vector env"))?;
        let _lease = Claim::new(&e.leased)?;
        lock(&e.inner).close()?;
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_free(env: *mut RLMeshEnv) {
    guard_value((), || {
        if !env.is_null() {
            drop(unsafe { Box::from_raw(env) });
        }
    });
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_vector_env_free(env: *mut RLMeshVectorEnv) {
    guard_value((), || {
        if !env.is_null() {
            drop(unsafe { Box::from_raw(env) });
        }
    });
}
pub(crate) struct Claim(Arc<AtomicBool>);
impl Claim {
    pub(crate) fn new(leased: &Arc<AtomicBool>) -> Result<Self, CapiError> {
        if leased
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(CapiError::from(Error::Runtime(
                "env is already serving or being driven".into(),
            )));
        }
        Ok(Self(Arc::clone(leased)))
    }
}
impl Drop for Claim {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// A per-server gate prevents callbacks detached by the facade's timeout from
/// starting after the C server has been freed. Free also drains each mutex.
#[derive(Clone)]
pub(crate) struct BorrowedEnv {
    pub(crate) inner: Arc<Mutex<Box<dyn Env + Send>>>,
    pub(crate) stopped: Arc<AtomicBool>,
}
#[derive(Clone)]
pub(crate) struct BorrowedVectorEnv {
    pub(crate) inner: Arc<Mutex<Box<dyn VectorEnv + Send>>>,
    pub(crate) stopped: Arc<AtomicBool>,
}
impl BorrowedEnv {
    pub(crate) fn finish(&self, close_env: bool) {
        let mut env = lock(&self.inner);
        if close_env {
            let _ = env.close();
        }
    }
}
impl BorrowedVectorEnv {
    pub(crate) fn finish(&self, close_env: bool) {
        let mut env = lock(&self.inner);
        if close_env {
            let _ = env.close();
        }
    }
}
macro_rules! borrowed_attrs {
    () => {
        fn render_mode(&self) -> Option<String> {
            lock(&self.inner).render_mode()
        }
        fn metadata(&self) -> Option<Metadata> {
            lock(&self.inner).metadata()
        }
        fn id(&self) -> Option<String> {
            lock(&self.inner).id()
        }
        fn render(&mut self) -> rlmesh::Result<Option<Value>> {
            let mut e = lock(&self.inner);
            open(self.stopped.load(Ordering::Acquire))?;
            e.render()
        }
        fn close(&mut self) -> rlmesh::Result<()> {
            let mut e = lock(&self.inner);
            open(self.stopped.load(Ordering::Acquire))?;
            e.close()
        }
    };
}
impl Env for BorrowedEnv {
    fn observation_space(&self) -> Space {
        lock(&self.inner).observation_space()
    }
    fn action_space(&self) -> Space {
        lock(&self.inner).action_space()
    }
    fn reset(&mut self, s: Option<i64>, o: Option<&InfoDict>) -> rlmesh::Result<(Value, InfoDict)> {
        let mut e = lock(&self.inner);
        open(self.stopped.load(Ordering::Acquire))?;
        e.reset(s, o)
    }
    fn step(&mut self, a: Option<Value>) -> rlmesh::Result<(Value, f64, bool, bool, InfoDict)> {
        let mut e = lock(&self.inner);
        open(self.stopped.load(Ordering::Acquire))?;
        e.step(a)
    }
    borrowed_attrs!();
}
impl VectorEnv for BorrowedVectorEnv {
    fn num_envs(&self) -> u64 {
        lock(&self.inner).num_envs()
    }
    fn single_observation_space(&self) -> Space {
        lock(&self.inner).single_observation_space()
    }
    fn single_action_space(&self) -> Space {
        lock(&self.inner).single_action_space()
    }
    fn reset(
        &mut self,
        s: Option<VectorSeed>,
        o: Option<&InfoDict>,
    ) -> rlmesh::Result<(Value, InfoDict)> {
        let mut e = lock(&self.inner);
        open(self.stopped.load(Ordering::Acquire))?;
        e.reset(s, o)
    }
    fn step(
        &mut self,
        a: Value,
    ) -> rlmesh::Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        let mut e = lock(&self.inner);
        open(self.stopped.load(Ordering::Acquire))?;
        e.step(a)
    }
    borrowed_attrs!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::RLMeshBytes;
    use crate::env_server::*;
    use crate::value::handle::{into_handle, value_ref};
    use rlmesh::spaces::{Discrete, DiscreteOptions};
    use std::ptr::{null, null_mut};
    use std::sync::atomic::AtomicUsize;
    #[derive(Default)]
    struct Probe {
        resets: AtomicUsize,
        steps: AtomicUsize,
        closes: AtomicUsize,
        frees: AtomicUsize,
        seed: Mutex<Option<i64>>,
        slow_close: AtomicBool,
        close_entered: Mutex<Option<std::sync::mpsc::Sender<()>>>,
        close_release: Mutex<Option<std::sync::mpsc::Receiver<()>>>,
    }
    unsafe extern "C" fn reset(
        ud: *mut c_void,
        seed: *const i64,
        options: *const RLMeshValue,
        obs: *mut *mut RLMeshValue,
        info: *mut *mut RLMeshValue,
    ) -> i32 {
        let p = unsafe { &*ud.cast::<Probe>() };
        p.resets.fetch_add(1, Ordering::Relaxed);
        *lock(&p.seed) = unsafe { seed.as_ref() }.copied();
        if !options.is_null() {
            assert!(
                unsafe {
                    // SAFETY: the FFI contract supplies live handles and writable output storage.
                    value_ref(options)
                }
                .ok()
                .unwrap()
                .as_dict()
                .is_ok()
            );
        }
        unsafe {
            *obs = into_handle(Value::Int(0));
            *info = into_handle(Value::Dict(
                [("blob", Value::Bytes(vec![0, 255]))].into_iter().collect(),
            ));
        }
        0
    }
    unsafe extern "C" fn step(
        ud: *mut c_void,
        action: *const RLMeshValue,
        obs: *mut *mut RLMeshValue,
        reward: *mut f64,
        terminated: *mut bool,
        _truncated: *mut bool,
        _info: *mut *mut RLMeshValue,
    ) -> i32 {
        let p = unsafe { &*ud.cast::<Probe>() };
        p.steps.fetch_add(1, Ordering::Relaxed);
        let n = unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            value_ref(action)
        }
        .ok()
        .unwrap()
        .as_int()
        .unwrap();
        unsafe {
            *obs = into_handle(Value::Int(n));
            *reward = 2.0;
            *terminated = true;
        }
        0
    }
    unsafe extern "C" fn close(ud: *mut c_void) -> i32 {
        let probe = unsafe { &*ud.cast::<Probe>() };
        if let Some(entered) = lock(&probe.close_entered).as_ref() {
            let _ = entered.send(());
        }
        if let Some(release) = lock(&probe.close_release).as_ref() {
            let _ = release.recv();
        }
        if unsafe { &*ud.cast::<Probe>() }
            .slow_close
            .load(Ordering::Relaxed)
        {
            std::thread::sleep(std::time::Duration::from_millis(80));
        }
        unsafe { &*ud.cast::<Probe>() }
            .closes
            .fetch_add(1, Ordering::Relaxed);
        0
    }
    unsafe extern "C" fn free(ud: *mut c_void) {
        unsafe { &*ud.cast::<Probe>() }
            .frees
            .fetch_add(1, Ordering::Relaxed);
    }
    fn new_env(p: &Probe) -> *mut RLMeshEnv {
        let space = RLMeshSpace(Discrete::new(4, DiscreteOptions::default()).unwrap().into());
        let attrs = RLMeshEnvAttributes {
            struct_size: std::mem::size_of::<RLMeshEnvAttributes>(),
            observation_space: &space,
            action_space: &space,
            render_mode: null(),
            metadata: null(),
            id: c"C-v1".as_ptr(),
        };
        let vt = RLMeshEnvVtable {
            struct_size: std::mem::size_of::<RLMeshEnvVtable>(),
            reset: Some(reset),
            step: Some(step),
            render: None,
            close: Some(close),
            free_user_data: Some(free),
        };
        let mut env = null_mut();
        assert_eq!(
            unsafe {
                rlmesh_env_new(
                    &vt,
                    std::ptr::from_ref(p).cast_mut().cast(),
                    &attrs,
                    &mut env,
                )
            },
            RLMeshStatus::Ok
        );
        env
    }
    #[test]
    fn facade_server_roundtrips_typed_callbacks_over_grpc() {
        let p = Probe::default();
        let env = new_env(&p);
        let mut server = null_mut();
        assert_eq!(
            unsafe { rlmesh_env_server_new(&env, 1, null(), &mut server) },
            RLMeshStatus::Ok
        );
        assert_eq!(rlmesh_env_server_start(server), RLMeshStatus::Ok);
        let mut addr = RLMeshBytes::from_vec(vec![]);
        assert_eq!(
            rlmesh_env_server_address(server, &mut addr),
            RLMeshStatus::Ok
        );
        let address = String::from_utf8(unsafe { addr.into_vec() }).unwrap();
        let mut client = rlmesh::RemoteEnv::new(rlmesh::RemoteEnvOptions {
            address: Some(address),
            ..Default::default()
        })
        .unwrap();
        let (obs, info) = client.reset(Some(7), Some(&Metadata::new())).unwrap();
        assert_eq!(obs, Value::Int(0));
        assert_eq!(info.get("blob"), Some(&Value::Bytes(vec![0, 255])));
        assert_eq!(*lock(&p.seed), Some(7));
        let (obs, reward, done, _, _) = client.step(Value::Int(2)).unwrap();
        assert_eq!((obs, reward, done), (Value::Int(2), 2.0, true));
        client.close().unwrap();
        assert_eq!(rlmesh_env_server_shutdown(server), RLMeshStatus::Ok);
        unsafe {
            rlmesh_env_server_free(server);
            rlmesh_env_free(env);
        }
        assert_eq!(p.closes.load(Ordering::Relaxed), 1);
        assert_eq!(p.frees.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn server_free_drains_a_callback_past_its_facade_close_deadline() {
        use std::time::Duration;
        let p = Probe::default();
        let (closing, entered) = std::sync::mpsc::channel();
        let (release, blocked) = std::sync::mpsc::channel();
        *lock(&p.close_entered) = Some(closing);
        *lock(&p.close_release) = Some(blocked);
        let env = new_env(&p);
        let serve = RLMeshServeOptions {
            struct_size: std::mem::size_of::<RLMeshServeOptions>(),
            allow_remote_shutdown: false,
            has_idle_timeout_seconds: false,
            has_drain_timeout_seconds: false,
            has_close_timeout_seconds: true,
            compress_responses: false,
            _reserved: [0; 3],
            idle_timeout_seconds: 0.0,
            drain_timeout_seconds: 0.0,
            close_timeout_seconds: 0.001,
            workflow_edition: null(),
        };
        let opts = RLMeshEnvServerOptions {
            struct_size: std::mem::size_of::<RLMeshEnvServerOptions>(),
            address: null(),
            host: null(),
            path: null(),
            options: &serve,
            tags: null(),
            ext: null(),
            port: 0,
            has_port: false,
            has_transport: false,
            transport: 0,
            close_env_on_shutdown: true,
            _reserved: [0; 7],
        };
        let mut server = null_mut();
        assert_eq!(
            unsafe { rlmesh_env_server_new(&env, 1, &opts, &mut server) },
            RLMeshStatus::Ok
        );
        // Never started: shutdown times out while the close stays wedged.
        assert_ne!(rlmesh_env_server_shutdown(server), RLMeshStatus::Ok);
        entered.recv_timeout(Duration::from_secs(5)).unwrap();
        let pointer = server as usize;
        let (done, joined) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            unsafe {
                rlmesh_env_server_free(pointer as *mut RLMeshEnvServer);
            }
            done.send(()).unwrap();
        });
        let early = joined.recv_timeout(Duration::from_millis(100));
        release.send(()).unwrap();
        worker.join().unwrap();
        assert!(
            early.is_err(),
            "C free must wait for the abandoned callback"
        );
        joined.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(p.closes.load(Ordering::Relaxed), 1);
        unsafe {
            rlmesh_env_free(env);
        }
    }
}
