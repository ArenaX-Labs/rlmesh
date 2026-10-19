//! A C policy vtable over the facade Model and its episode store.
#![allow(unsafe_code)] // FFI callbacks, borrowed contexts and ownership transfer.
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard, guard_value},
};
use crate::cancel::RLMeshExtCancelToken;
use crate::codec::RLMeshBytes;
use crate::env_server::RLMeshServeOptions;
use crate::ffi;
use crate::value::handle::{RLMeshValue, value_ref, write_out};
use rlmesh::binding::model::{DynPolicy, PolicyDecl, corners_from_bits, new_model};
use rlmesh::types::{EpisodeState, PredictContext};
use rlmesh::{Error, Metadata, Model, ModelOptions, Value};
use std::ffi::{CString, c_char, c_void};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub(crate) use crate::adapters::RLMeshModelSpec;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshSpecArg {
    pub kind: i32,
    pub spec: *const RLMeshModelSpec,
}
impl RLMeshSpecArg {
    fn read(self) -> Result<Option<rlmesh::types::SpecArg>, CapiError> {
        match self.kind {
            1 => Ok(None),
            3 => Ok(Some(rlmesh::NO_ADAPTER)),
            2 => Ok(Some(
                unsafe { self.spec.as_ref() }
                    .ok_or_else(|| CapiError::invalid_arg("null model spec"))?
                    .0
                    .clone()
                    .into(),
            )),
            _ => Err(CapiError::invalid_arg("invalid spec kind")),
        }
    }
}
#[repr(transparent)]
pub struct RLMeshEpisodeState(EpisodeState);
struct Foreign {
    ptr: *mut c_void,
    free: Option<unsafe extern "C" fn(*mut c_void)>,
}
// SAFETY: an episode's C state is documented as transferable between threads;
// the facade serializes every corner/hook accessing this episode's state.
unsafe impl Send for Foreign {}
impl Drop for Foreign {
    fn drop(&mut self) {
        if let Some(f) = self.free {
            unsafe { f(self.ptr) };
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_episode_state_get(
    state: *const RLMeshEpisodeState,
    key: *const c_char,
    out: *mut *mut c_void,
) -> bool {
    guard_value(false, || {
        let Some(state) = (unsafe { state.as_ref() }) else {
            return false;
        };
        let Ok(Some(key)) = (unsafe { ffi::text(key) }) else {
            return false;
        };
        let Some(v) = state.0.get::<Foreign>(&key) else {
            return false;
        };
        if out.is_null() {
            return false;
        }
        unsafe {
            *out = v.ptr;
        }
        true
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_episode_state_set(
    state: *mut RLMeshEpisodeState,
    key: *const c_char,
    value: *mut c_void,
    free_value: Option<unsafe extern "C" fn(*mut c_void)>,
) -> RLMeshStatus {
    guard(|| {
        let state =
            unsafe { state.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null state"))?;
        let key = unsafe { ffi::text(key) }?.ok_or_else(|| CapiError::invalid_arg("null key"))?;
        if let Some(existing) = state.0.get_mut::<Foreign>(&key)
            && existing.ptr == value
        {
            existing.free = free_value;
            return Ok(());
        }
        drop(state.0.remove(&key));
        state.0.insert(
            key,
            Foreign {
                ptr: value,
                free: free_value,
            },
        );
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_episode_state_remove(
    state: *mut RLMeshEpisodeState,
    key: *const c_char,
) -> bool {
    guard_value(false, || {
        let Some(state) = (unsafe { state.as_mut() }) else {
            return false;
        };
        let Ok(Some(key)) = (unsafe { ffi::text(key) }) else {
            return false;
        };
        state.0.remove(&key).is_some()
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_episode_state_len(state: *const RLMeshEpisodeState) -> usize {
    guard_value(0, || unsafe { state.as_ref() }.map_or(0, |s| s.0.len()))
}
#[repr(C)]
pub struct RLMeshPredictContext {
    pub struct_size: usize,
    pub episode_id: *const c_char,
    pub has_episode_seed: bool,
    pub _reserved0: [u8; 7],
    pub episode_seed: i64,
    pub predict_index: u64,
    pub has_predict_seed: bool,
    pub _reserved1: [u8; 7],
    pub predict_seed: i64,
    pub state: *mut RLMeshEpisodeState,
}
struct Context {
    _id: CString,
    raw: RLMeshPredictContext,
}
impl Context {
    fn new(c: &mut PredictContext<'_>) -> Self {
        let id = CString::new(c.episode_id).unwrap_or_default();
        let raw = RLMeshPredictContext {
            struct_size: std::mem::size_of::<RLMeshPredictContext>(),
            episode_id: id.as_ptr(),
            has_episode_seed: c.episode_seed.is_some(),
            _reserved0: [0; 7],
            episode_seed: c.episode_seed.unwrap_or_default(),
            predict_index: c.predict_index,
            has_predict_seed: c.predict_seed.is_some(),
            _reserved1: [0; 7],
            predict_seed: c.predict_seed.unwrap_or_default(),
            state: std::ptr::from_mut(c.state).cast(),
        };
        Self { _id: id, raw }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshLoadKey {
    pub name: *const c_char,
    pub required: bool,
    pub _reserved: [u8; std::mem::size_of::<usize>() - 1],
}
type Predict = unsafe extern "C" fn(
    *mut c_void,
    *const RLMeshValue,
    *const RLMeshPredictContext,
    *mut *mut RLMeshValue,
) -> i32;
type Chunk = unsafe extern "C" fn(
    *mut c_void,
    *const RLMeshValue,
    u32,
    *const RLMeshPredictContext,
    *mut *mut RLMeshValue,
) -> i32;
type Batch = unsafe extern "C" fn(
    *mut c_void,
    *const *const RLMeshValue,
    usize,
    *const *const RLMeshPredictContext,
    *mut *mut RLMeshValue,
) -> i32;
type ChunkBatch = unsafe extern "C" fn(
    *mut c_void,
    *const *const RLMeshValue,
    usize,
    u32,
    *const *const RLMeshPredictContext,
    *mut *mut RLMeshValue,
) -> i32;
type EpisodeHook = unsafe extern "C" fn(*mut c_void, *const c_char) -> i32;
type CloseHook = unsafe extern "C" fn(*mut c_void) -> i32;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshModelVtable {
    pub struct_size: usize,
    pub predict: Option<Predict>,
    pub predict_chunk: Option<Chunk>,
    pub predict_batch: Option<Batch>,
    pub predict_chunk_batch: Option<ChunkBatch>,
    pub on_episode_end: Option<EpisodeHook>,
    pub close: Option<CloseHook>,
    pub load: Option<unsafe extern "C" fn(*mut c_void, *const RLMeshValue) -> i32>,
    pub loaded_native_chunk: Option<unsafe extern "C" fn(*mut c_void, *mut bool, *mut u32) -> i32>,
    pub free_user_data: Option<unsafe extern "C" fn(*mut c_void)>,
    pub context_corners: u32,
    pub _reserved0: u32,
    pub spec: RLMeshSpecArg,
    pub allow_fusion: bool,
    pub has_native_chunk: bool,
    pub _reserved1: [u8; 2],
    pub native_chunk: u32,
    pub workflow_edition: *const c_char,
    pub load_keys: *const RLMeshLoadKey,
    pub load_keys_len: usize,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshModelOptions {
    pub struct_size: usize,
    pub spec: RLMeshSpecArg,
    pub on_episode_end: Option<EpisodeHook>,
    pub on_close: Option<CloseHook>,
    pub hook_user_data: *mut c_void,
    pub free_hook_user_data: Option<unsafe extern "C" fn(*mut c_void)>,
    pub trust_entrypoints: bool,
    pub _reserved: [u8; std::mem::size_of::<usize>() - 1],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshExtModelServeOptions {
    pub struct_size: usize,
    pub token: *const c_char,
    pub predict_concurrency: usize,
    pub has_predict_concurrency: bool,
    pub _reserved: [u8; std::mem::size_of::<usize>() - 1],
    pub no_signal_handlers: bool,
    pub _reserved1: [u8; std::mem::size_of::<usize>() - 1],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshModelServeOptions {
    pub struct_size: usize,
    pub options: *const RLMeshServeOptions,
    pub cancel: *const RLMeshExtCancelToken,
    pub ext: *const RLMeshExtModelServeOptions,
}
struct Owner {
    ptr: *mut c_void,
    free: Option<unsafe extern "C" fn(*mut c_void)>,
    owned: AtomicBool,
}
// SAFETY: owners are transferred between threads, and callbacks are serialized
// by the facade's model queue. Drop only runs after the final reference.
unsafe impl Send for Owner {}
unsafe impl Sync for Owner {}
impl Owner {
    fn release(&self) {
        if self.owned.swap(false, Ordering::AcqRel)
            && let Some(f) = self.free
        {
            unsafe { f(self.ptr) };
        }
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.release();
    }
}
struct CPolicy {
    vt: RLMeshModelVtable,
    owner: Arc<Owner>,
    native_chunk: Option<u32>,
}
// SAFETY: vtable pointer fields are never read after construction; strings,
// specs and load keys have been deep-copied into the facade's PolicyDecl.
unsafe impl Send for CPolicy {}
fn status(s: i32) -> rlmesh::Result<()> {
    if s == 0 {
        Ok(())
    } else {
        Err(ffi::callback_error("model"))
    }
}
fn action(s: i32, out: *mut RLMeshValue) -> rlmesh::Result<Value> {
    let value = unsafe { ffi::take(out) };
    status(s)?;
    value.ok_or_else(|| Error::Type("policy returned no action".into()))
}
impl CPolicy {
    fn one(
        &self,
        observation: Value,
        context: Option<&mut PredictContext<'_>>,
        horizon: Option<u32>,
    ) -> rlmesh::Result<Value> {
        let obs = RLMeshValue::new(observation);
        let context = context.map(Context::new);
        let ctx = context
            .as_ref()
            .map_or(std::ptr::null(), |c| std::ptr::from_ref(&c.raw));
        let mut out = std::ptr::null_mut();
        let _callback_frame = crate::abi::status::CallbackFrame::enter();
        let s = unsafe {
            match horizon {
                None => self
                    .vt
                    .predict
                    .ok_or_else(|| Error::Runtime("predict is not defined".into()))?(
                    self.owner.ptr,
                    &obs,
                    ctx,
                    &mut out,
                ),
                Some(h) => self
                    .vt
                    .predict_chunk
                    .ok_or_else(|| Error::Runtime("predict_chunk is not defined".into()))?(
                    self.owner.ptr,
                    &obs,
                    h,
                    ctx,
                    &mut out,
                ),
            }
        };
        action(s, out)
    }
    fn batch(
        &self,
        observations: Vec<Value>,
        context: Option<&mut [PredictContext<'_>]>,
        horizon: Option<u32>,
    ) -> rlmesh::Result<Vec<Value>> {
        let observations: Vec<_> = observations.into_iter().map(RLMeshValue::new).collect();
        let obs: Vec<_> = observations.iter().map(std::ptr::from_ref).collect();
        let context = context.map(|c| c.iter_mut().map(Context::new).collect::<Vec<_>>());
        let contexts = context.as_ref().map(|cs| {
            cs.iter()
                .map(|c| std::ptr::from_ref(&c.raw))
                .collect::<Vec<_>>()
        });
        let ctx = contexts.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let mut out = vec![std::ptr::null_mut(); obs.len()];
        let _callback_frame = crate::abi::status::CallbackFrame::enter();
        let s = unsafe {
            match horizon {
                None => self
                    .vt
                    .predict_batch
                    .ok_or_else(|| Error::Runtime("predict_batch is not defined".into()))?(
                    self.owner.ptr,
                    obs.as_ptr(),
                    obs.len(),
                    ctx,
                    out.as_mut_ptr(),
                ),
                Some(h) => self
                    .vt
                    .predict_chunk_batch
                    .ok_or_else(|| Error::Runtime("predict_chunk_batch is not defined".into()))?(
                    self.owner.ptr,
                    obs.as_ptr(),
                    obs.len(),
                    h,
                    ctx,
                    out.as_mut_ptr(),
                ),
            }
        };
        let values: Vec<_> = out.into_iter().map(|p| unsafe { ffi::take(p) }).collect();
        status(s)?;
        values
            .into_iter()
            .map(|v| v.ok_or_else(|| Error::Type("policy left a batch action unset".into())))
            .collect()
    }
}
impl DynPolicy for CPolicy {
    fn load(&mut self, config: &Metadata) -> rlmesh::Result<()> {
        if let Some(load) = self.vt.load {
            let config = RLMeshValue::new(Value::Dict(config.clone()));
            let _callback_frame = crate::abi::status::CallbackFrame::enter();
            status(unsafe { load(self.owner.ptr, &config) })?;
        } else {
            struct DefaultLoad;
            impl rlmesh::Policy for DefaultLoad {}
            rlmesh::Policy::load(&mut DefaultLoad, config)?;
        }
        if let Some(f) = self.vt.loaded_native_chunk {
            let (mut has, mut value) = (false, 0);
            let _callback_frame = crate::abi::status::CallbackFrame::enter();
            status(unsafe { f(self.owner.ptr, &mut has, &mut value) })?;
            self.native_chunk = has.then_some(value);
        }
        Ok(())
    }
    fn allow_fusion(&self) -> bool {
        self.vt.allow_fusion
    }
    fn native_chunk(&self) -> Option<u32> {
        self.native_chunk
    }
    fn predict(&mut self, o: Value, c: Option<&mut PredictContext<'_>>) -> rlmesh::Result<Value> {
        self.one(o, c, None)
    }
    fn predict_chunk(
        &mut self,
        o: Value,
        h: u32,
        c: Option<&mut PredictContext<'_>>,
    ) -> rlmesh::Result<Value> {
        self.one(o, c, Some(h))
    }
    fn predict_batch(
        &mut self,
        o: Vec<Value>,
        c: Option<&mut [PredictContext<'_>]>,
    ) -> rlmesh::Result<Vec<Value>> {
        self.batch(o, c, None)
    }
    fn predict_chunk_batch(
        &mut self,
        o: Vec<Value>,
        h: u32,
        c: Option<&mut [PredictContext<'_>]>,
    ) -> rlmesh::Result<Vec<Value>> {
        self.batch(o, c, Some(h))
    }
    fn on_episode_end(&mut self, id: &str) -> rlmesh::Result<()> {
        if let Some(f) = self.vt.on_episode_end {
            let id = CString::new(id).unwrap_or_default();
            let _callback_frame = crate::abi::status::CallbackFrame::enter();
            status(unsafe { f(self.owner.ptr, id.as_ptr()) })?;
        }
        Ok(())
    }
    fn close(&mut self) -> rlmesh::Result<()> {
        if self.owner.owned.load(Ordering::Acquire)
            && let Some(f) = self.vt.close
        {
            let _callback_frame = crate::abi::status::CallbackFrame::enter();
            status(unsafe { f(self.owner.ptr) })?;
        }
        Ok(())
    }
}
struct Info {
    _policy_owner: Arc<Owner>,
    spec: Option<RLMeshModelSpec>,
    _hook_owner: Option<Arc<Owner>>,
}
#[derive(Clone)]
pub struct RLMeshModel {
    pub(crate) model: Model,
    info: Arc<Info>,
}
/// # Safety
/// A non-NULL handle stays live for the returned reference's lifetime.
pub(crate) unsafe fn require<'a>(ptr: *const RLMeshModel) -> Result<&'a RLMeshModel, CapiError> {
    unsafe { ptr.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null model"))
}
pub(crate) unsafe fn declaration(vt: RLMeshModelVtable) -> Result<PolicyDecl, CapiError> {
    let bits = u32::from(vt.predict.is_some())
        | u32::from(vt.predict_chunk.is_some()) << 1
        | u32::from(vt.predict_batch.is_some()) << 2
        | u32::from(vt.predict_chunk_batch.is_some()) << 3;
    let keys = if vt.load_keys.is_null() {
        if vt.load_keys_len != 0 {
            return Err(CapiError::type_error("NULL load_keys with nonzero length"));
        }
        None
    } else {
        Some(
            unsafe { crate::spaces::slice_arg(vt.load_keys, vt.load_keys_len, "load_keys") }?
                .iter()
                .map(|k| {
                    Ok((
                        unsafe { ffi::text(k.name) }?
                            .ok_or_else(|| CapiError::type_error("null load key"))?,
                        k.required,
                    ))
                })
                .collect::<Result<Vec<_>, CapiError>>()?,
        )
    };
    Ok(PolicyDecl {
        corners: corners_from_bits(bits)?,
        context: corners_from_bits(vt.context_corners)?,
        native_chunk: vt.has_native_chunk.then_some(vt.native_chunk),
        load_keys: keys,
        spec: vt.spec.read()?,
        workflow_edition: unsafe { ffi::text(vt.workflow_edition) }?,
        authored: true,
        type_name: "CPolicy".into(),
    })
}
unsafe fn construct(
    vt: *const RLMeshModelVtable,
    ud: *mut c_void,
    options: *const RLMeshModelOptions,
    config: Option<Metadata>,
    out: *mut *mut RLMeshModel,
) -> Result<(), CapiError> {
    unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(out, std::ptr::null_mut())
    }?;
    let vt = unsafe { sizing::read(vt, "RLMeshModelVtable") }?
        .ok_or_else(|| CapiError::type_error("null model vtable"))?;
    let decl = unsafe { declaration(vt) }?;
    let options = unsafe { sizing::read(options, "RLMeshModelOptions") }?;
    let owner = Arc::new(Owner {
        ptr: ud,
        free: vt.free_user_data,
        owned: AtomicBool::new(false),
    });
    let mut hook_owner = None;
    let mut opts = ModelOptions::default();
    if let Some(o) = options {
        opts.spec = o.spec.read()?;
        opts.trust_entrypoints = o.trust_entrypoints;
        let hooks = Arc::new(Owner {
            ptr: o.hook_user_data,
            free: o.free_hook_user_data,
            owned: AtomicBool::new(false),
        });
        if let Some(f) = o.on_episode_end {
            let hooks = Arc::clone(&hooks);
            opts.on_episode_end = Some(Box::new(move |id| {
                let id = CString::new(id).unwrap_or_default();
                let _callback_frame = crate::abi::status::CallbackFrame::enter();
                status(unsafe { f(hooks.ptr(), id.as_ptr()) })
            }));
        }
        if let Some(f) = o.on_close {
            let hooks = Arc::clone(&hooks);
            opts.on_close = Some(Box::new(move || {
                let _callback_frame = crate::abi::status::CallbackFrame::enter();
                status(unsafe { f(hooks.ptr()) })
            }));
        }
        hook_owner = Some(hooks);
    }
    crate::abi::ignore_sigpipe();
    let policy = CPolicy {
        vt,
        owner: Arc::clone(&owner),
        native_chunk: decl.native_chunk,
    };
    let model = new_model(Box::new(policy), decl, opts, config)?;
    let spec = match model.spec() {
        Some(rlmesh::types::SpecArg::ModelSpec(s)) => Some(RLMeshModelSpec::new(s.clone())),
        _ => None,
    };
    owner.owned.store(true, Ordering::Release);
    if let Some(h) = &hook_owner {
        h.owned.store(true, Ordering::Release);
    }
    // Keep hook ownership even if neither hook slot was provided.
    let info = Arc::new(Info {
        _policy_owner: owner,
        spec,
        _hook_owner: hook_owner,
    });
    let handle = RLMeshModel { model, info };
    unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(out, Box::into_raw(Box::new(handle)))
    }
}
impl Owner {
    fn ptr(&self) -> *mut c_void {
        self.ptr
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_new(
    vt: *const RLMeshModelVtable,
    ud: *mut c_void,
    opts: *const RLMeshModelOptions,
    out: *mut *mut RLMeshModel,
) -> RLMeshStatus {
    guard(|| unsafe { construct(vt, ud, opts, None, out) })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_from_config(
    vt: *const RLMeshModelVtable,
    ud: *mut c_void,
    config: *const RLMeshValue,
    out: *mut *mut RLMeshModel,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let config = unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            value_ref(config)
        }?
        .as_dict()?
        .clone();
        unsafe { construct(vt, ud, std::ptr::null(), Some(config), out) }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_model_clone(
    model: *const RLMeshModel,
    out: *mut *mut RLMeshModel,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, Box::into_raw(Box::new(require(model)?.clone())))
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_model_close(model: *mut RLMeshModel) -> RLMeshStatus {
    guard(|| {
        unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            require(model)
        }?
        .model
        .close()?;
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_free(model: *mut RLMeshModel) {
    guard_value((), || {
        if !model.is_null() {
            let model = unsafe { Box::from_raw(model) };
            if Arc::strong_count(&model.info) == 1 {
                // The worker can retain an inert policy briefly after close's
                // reply. Release C state synchronously once the callback queue
                // drained; its later Arc drop is then a no-op. In an async or
                // reentrant context leave ownership to the facade's teardown.
                let closed = model.model.close().is_ok() || model.model.close().is_ok();
                if closed {
                    model.info._policy_owner.release();
                    if let Some(owner) = &model.info._hook_owner {
                        owner.release();
                    }
                }
            }
            drop(model);
        }
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_model_allow_fusion(model: *const RLMeshModel) -> bool {
    guard_value(false, || {
        unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            require(model)
        }
        .is_ok_and(|m| m.model.allow_fusion())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_model_native_chunk(
    model: *const RLMeshModel,
    out_has: *mut bool,
    out: *mut u32,
) {
    let _ = guard(|| {
        let k = unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            require(model)
        }?
        .model
        .native_chunk();
        if out.is_null() || out_has.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out_has, k.is_some())
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, k.unwrap_or_default())
        }
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_model_workflow_edition(
    model: *const RLMeshModel,
    out_has: *mut bool,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() || out_has.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let edition = unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            require(model)
        }?
        .model
        .workflow_edition();
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out_has, edition.is_some())
        }?;
        if let Some(e) = edition {
            ffi::bytes(out, e.to_string())?;
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_model_spec(
    model: *const RLMeshModel,
    out: *mut RLMeshSpecArg,
) -> RLMeshStatus {
    guard(|| {
        let m = unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            require(model)
        }?;
        let spec = match m.model.spec() {
            None => RLMeshSpecArg {
                kind: 1,
                spec: std::ptr::null(),
            },
            Some(rlmesh::types::SpecArg::NoAdapter) => RLMeshSpecArg {
                kind: 3,
                spec: std::ptr::null(),
            },
            Some(rlmesh::types::SpecArg::ModelSpec(_)) => RLMeshSpecArg {
                kind: 2,
                spec: m
                    .info
                    .spec
                    .as_ref()
                    .map_or(std::ptr::null(), std::ptr::from_ref),
            },
        };
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, spec)
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_serve(
    model: *mut RLMeshModel,
    address: *const c_char,
    options: *const RLMeshModelServeOptions,
) -> RLMeshStatus {
    guard(|| {
        let model = &unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            require(model)
        }?
        .model;
        let address =
            unsafe { ffi::text(address) }?.ok_or_else(|| CapiError::invalid_arg("null address"))?;
        let options = unsafe { sizing::read(options, "RLMeshModelServeOptions") }?;
        let mut no_signal_handlers = false;
        let opts = if let Some(o) = options {
            let ext = unsafe { sizing::read(o.ext, "RLMeshExtModelServeOptions") }?;
            let ext = ext
                .map(|e| -> Result<_, CapiError> {
                    no_signal_handlers = e.no_signal_handlers;
                    Ok(rlmesh::ext::ModelServeOptions {
                        token: unsafe { ffi::text(e.token) }?,
                        predict_concurrency: e
                            .has_predict_concurrency
                            .then_some(e.predict_concurrency),
                    })
                })
                .transpose()?;
            rlmesh::ModelServeOptions {
                options: unsafe { crate::env_server::serve_options(o.options) }?,
                cancel: unsafe { o.cancel.as_ref() }.map(|t| t.0.clone()),
                ext,
            }
        } else {
            rlmesh::ModelServeOptions::default()
        };
        rlmesh::binding::model::serve_model(model, &address, opts, !no_signal_handlers)?;
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_predict_seed(seed: i64, index: u64) -> i64 {
    rlmesh::predict_seed(seed, index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::handle::into_handle;
    use rlmesh::spaces::{Discrete, DiscreteOptions, Space};
    use rlmesh::{Env, InfoDict};
    use std::ptr::{null, null_mut};
    use std::sync::{Mutex, atomic::AtomicUsize};
    #[derive(Default)]
    struct Probe {
        predicts: AtomicUsize,
        ends: AtomicUsize,
        closes: AtomicUsize,
        frees: AtomicUsize,
        states_freed: AtomicUsize,
        seeds: Mutex<Vec<i64>>,
        fail: AtomicBool,
    }
    struct Held {
        probe: *const Probe,
        index: u64,
    }
    unsafe extern "C" fn free_state(v: *mut c_void) {
        let v = unsafe { Box::from_raw(v.cast::<Held>()) };
        unsafe { &*v.probe }
            .states_freed
            .fetch_add(1, Ordering::Relaxed);
    }
    unsafe extern "C" fn predict(
        ud: *mut c_void,
        obs: *const RLMeshValue,
        ctx: *const RLMeshPredictContext,
        out: *mut *mut RLMeshValue,
    ) -> i32 {
        let p = unsafe { &*ud.cast::<Probe>() };
        p.predicts.fetch_add(1, Ordering::Relaxed);
        if p.fail.load(Ordering::Relaxed) {
            unsafe {
                *out = into_handle(Value::Int(0));
                crate::model::rlmesh_callback_set_error_class(8, c"bad observation".as_ptr());
            }
            return RLMeshStatus::Type as i32;
        }
        if let Some(c) = unsafe { ctx.as_ref() } {
            assert_eq!(c.struct_size, std::mem::size_of::<RLMeshPredictContext>());
            assert!(c.has_episode_seed && c.has_predict_seed);
            assert_eq!(
                c.predict_seed,
                rlmesh::predict_seed(c.episode_seed, c.predict_index)
            );
            p.seeds.lock().unwrap().push(c.episode_seed);
            let mut ptr = null_mut();
            if !unsafe { rlmesh_episode_state_get(c.state, c"held".as_ptr(), &mut ptr) } {
                assert_eq!(c.predict_index, 0);
                ptr = Box::into_raw(Box::new(Held { probe: p, index: 0 })).cast();
                assert_eq!(
                    unsafe {
                        rlmesh_episode_state_set(c.state, c"held".as_ptr(), ptr, Some(free_state))
                    },
                    RLMeshStatus::Ok
                );
                assert_eq!(
                    unsafe {
                        rlmesh_episode_state_set(c.state, c"null".as_ptr(), null_mut(), None)
                    },
                    RLMeshStatus::Ok
                );
            }
            let held = unsafe { &mut *ptr.cast::<Held>() };
            assert_eq!(held.index, c.predict_index);
            held.index += 1;
            let mut null_value = ptr;
            assert!(unsafe {
                rlmesh_episode_state_get(c.state, c"null".as_ptr(), &mut null_value)
            });
            assert!(null_value.is_null());
            assert_eq!(unsafe { rlmesh_episode_state_len(c.state) }, 2);
        }
        assert_eq!(
            unsafe {
                // SAFETY: the FFI contract supplies live handles and writable output storage.
                value_ref(obs)
            }
            .ok()
            .unwrap()
            .as_int()
            .unwrap(),
            0
        );
        unsafe {
            *out = into_handle(Value::Int(0));
        }
        0
    }
    unsafe extern "C" fn end(ud: *mut c_void, _id: *const c_char) -> i32 {
        unsafe { &*ud.cast::<Probe>() }
            .ends
            .fetch_add(1, Ordering::Relaxed);
        0
    }
    unsafe extern "C" fn close(ud: *mut c_void) -> i32 {
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
    fn vt(context: bool) -> RLMeshModelVtable {
        RLMeshModelVtable {
            struct_size: std::mem::size_of::<RLMeshModelVtable>(),
            predict: Some(predict),
            predict_chunk: None,
            predict_batch: None,
            predict_chunk_batch: None,
            on_episode_end: Some(end),
            close: Some(close),
            load: None,
            loaded_native_chunk: None,
            free_user_data: Some(free),
            context_corners: u32::from(context),
            _reserved0: 0,
            spec: RLMeshSpecArg {
                kind: 3,
                spec: null(),
            },
            allow_fusion: true,
            has_native_chunk: false,
            _reserved1: [0; 2],
            native_chunk: 0,
            workflow_edition: null(),
            load_keys: null(),
            load_keys_len: 0,
        }
    }
    struct Tiny {
        space: Space,
        steps: u32,
    }
    impl Tiny {
        fn new() -> Self {
            Self {
                space: Discrete::new(4, DiscreteOptions::default()).unwrap().into(),
                steps: 0,
            }
        }
    }
    impl Env for Tiny {
        fn observation_space(&self) -> Space {
            self.space.clone()
        }
        fn action_space(&self) -> Space {
            self.space.clone()
        }
        fn reset(
            &mut self,
            _seed: Option<i64>,
            _opts: Option<&InfoDict>,
        ) -> rlmesh::Result<(Value, InfoDict)> {
            self.steps = 0;
            Ok((Value::Int(0), Metadata::new()))
        }
        fn step(
            &mut self,
            _action: Option<Value>,
        ) -> rlmesh::Result<(Value, f64, bool, bool, InfoDict)> {
            self.steps += 1;
            Ok((Value::Int(0), 1.0, self.steps == 2, false, Metadata::new()))
        }
    }
    #[test]
    fn facade_runs_c_policy_contexts_and_owns_episode_state() {
        let p = Probe::default();
        let mut m = null_mut();
        assert_eq!(
            unsafe {
                rlmesh_model_new(
                    &vt(true),
                    std::ptr::from_ref(&p).cast_mut().cast(),
                    null(),
                    &mut m,
                )
            },
            RLMeshStatus::Ok
        );
        let mut env = Tiny::new();
        let model = &unsafe { &*m }.model;
        let result = model
            .run(
                &mut env,
                rlmesh::RunOptions {
                    seeds: Some(vec![7, 8]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(result.num_episodes(), 2);
        assert_eq!(p.predicts.load(Ordering::Relaxed), 4);
        assert_eq!(p.ends.load(Ordering::Relaxed), 2);
        assert_eq!(p.states_freed.load(Ordering::Relaxed), 2);
        assert_eq!(*p.seeds.lock().unwrap(), vec![7, 7, 8, 8]);
        assert_eq!(rlmesh_model_close(m), RLMeshStatus::Ok);
        assert_eq!(p.frees.load(Ordering::Relaxed), 0);
        unsafe {
            rlmesh_model_free(m);
        }
        assert_eq!(p.closes.load(Ordering::Relaxed), 1);
        assert_eq!(p.frees.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn callback_error_class_and_context_opt_out_use_facade_behavior() {
        let p = Probe::default();
        let mut m = null_mut();
        assert_eq!(
            unsafe {
                rlmesh_model_new(
                    &vt(false),
                    std::ptr::from_ref(&p).cast_mut().cast(),
                    null(),
                    &mut m,
                )
            },
            RLMeshStatus::Ok
        );
        let mut env = Tiny::new();
        let model = &unsafe { &*m }.model;
        model.run(&mut env, Default::default()).unwrap();
        assert!(p.seeds.lock().unwrap().is_empty());
        p.fail.store(true, Ordering::Relaxed);
        let error = model.run(&mut env, Default::default()).unwrap_err();
        assert_eq!(error.class(), rlmesh::ErrorClass::TypeError);
        assert!(error.to_string().contains("bad observation"));
        unsafe {
            rlmesh_model_free(m);
        }
    }
}
