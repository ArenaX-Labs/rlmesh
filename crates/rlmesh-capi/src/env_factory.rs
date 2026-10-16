//! C factory hooks projected through the facade's factory implementation.
#![allow(unsafe_code)] // FFI pointers, callbacks and owned handles.
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard, guard_value},
};
use crate::cancel::RLMeshExtCancelToken;
use crate::codec::RLMeshBytes;
use crate::env_api::{BorrowedEnv, BorrowedVectorEnv, RLMeshEnv, RLMeshVectorEnv};
use crate::env_server::RLMeshEnvTags;
use crate::ffi;
use crate::value::handle::{RLMeshValue, into_handle, value_ref, write_out};
use crate::variant::{RLMeshVariant, RLMeshVariantList, RLMeshVariantSink};
use rlmesh::{EnvFactory, EnvFactoryExt, Error, MadeEnv, Metadata, TagParam, Value, Variant};
use std::cell::RefCell;
use std::ffi::{CString, c_char, c_void};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

type Hook = unsafe extern "C" fn(*mut c_void) -> i32;
type Make = unsafe extern "C" fn(
    *mut c_void,
    *const RLMeshValue,
    *mut *mut RLMeshEnv,
    *mut *mut RLMeshVectorEnv,
) -> i32;
type TagsFor =
    unsafe extern "C" fn(*mut c_void, *const RLMeshValue, *mut *mut RLMeshEnvTags) -> i32;
type Enumerate = unsafe extern "C" fn(*mut c_void, *mut RLMeshVariantSink) -> i32;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshEnvFactoryVtable {
    pub struct_size: usize,
    pub prepare: Option<Hook>,
    pub make: Option<Make>,
    pub tags_for: Option<TagsFor>,
    pub enumerate_variants: Option<Enumerate>,
    pub close: Option<Hook>,
    pub free_user_data: Option<unsafe extern "C" fn(*mut c_void)>,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshTagParam {
    pub struct_size: usize,
    pub name: *const c_char,
    pub default_: *const RLMeshValue,
    pub choices: *const RLMeshValue,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshEnvFactoryAttributes {
    pub struct_size: usize,
    pub tags: *const RLMeshEnvTags,
    pub tag_params: *const *const RLMeshTagParam,
    pub tag_params_len: usize,
    pub reset_options: *const *const c_char,
    pub reset_options_len: usize,
    pub workflow_edition: *const c_char,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshEnvFactoryServeOptions {
    pub struct_size: usize,
    pub num_envs: i64,
    pub workflow_edition: *const c_char,
    pub make_kwargs: *const RLMeshValue,
    pub cancel: *const RLMeshExtCancelToken,
}
type GateCollection = Arc<Mutex<Vec<MadeGate>>>;
struct Data {
    vt: RLMeshEnvFactoryVtable,
    ud: *mut c_void,
    owned: AtomicBool,
    tags: Option<rlmesh::adapters::EnvTags>,
    params: Vec<TagParam>,
    reset_options: Vec<String>,
    edition: Option<String>,
    serve_gates: Mutex<Option<GateCollection>>,
}
// SAFETY: callback state is allowed to move between threads. Exclusive factory
// admission serializes every hook; attribute reads access only immutable copies.
unsafe impl Send for Data {}
unsafe impl Sync for Data {}
impl Drop for Data {
    fn drop(&mut self) {
        if self.owned.load(Ordering::Acquire)
            && let Some(f) = self.vt.free_user_data
        {
            unsafe { f(self.ud) };
        }
    }
}
pub struct RLMeshEnvFactory {
    data: Arc<Data>,
    busy: Arc<AtomicBool>,
}
struct Busy(Arc<AtomicBool>);
impl Drop for Busy {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
/// # Safety
/// A non-NULL handle stays live for the returned reference's lifetime.
unsafe fn require<'a>(ptr: *const RLMeshEnvFactory) -> Result<&'a RLMeshEnvFactory, CapiError> {
    unsafe { ptr.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null factory"))
}
fn enter(f: &RLMeshEnvFactory) -> Result<Busy, CapiError> {
    if f.busy
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(CapiError::from(Error::Runtime(
            "factory is already running a hook".into(),
        )));
    }
    Ok(Busy(Arc::clone(&f.busy)))
}
// EnvFactory's attributes are associated functions. A scoped, reentrant TLS
// binding supplies this C instance while the facade invokes those functions.
// Every C factory operation is sync; the facade runs its pre-serve hooks on
// this calling thread. Nested calls on another factory restore the old scope.
thread_local! {static CURRENT:RefCell<Vec<Arc<Data>>>=const {RefCell::new(Vec::new())};}
pub(crate) struct Scope;
impl Scope {
    fn new(data: Arc<Data>) -> Self {
        CURRENT.with(|s| s.borrow_mut().push(data));
        Self
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|s| s.borrow_mut().pop());
    }
}
fn current() -> Arc<Data> {
    CURRENT.with(|s| {
        Arc::clone(
            s.borrow()
                .last()
                .expect("factory operation has a binding scope"),
        )
    })
}
#[derive(Default)]
struct Factory;
fn check(status: i32) -> rlmesh::Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(ffi::callback_error("env"))
    }
}
impl EnvFactory for Factory {
    fn tags() -> Option<rlmesh::adapters::EnvTags> {
        current().tags.clone()
    }
    fn tag_params() -> Vec<TagParam> {
        current().params.clone()
    }
    fn reset_options() -> Vec<String> {
        current().reset_options.clone()
    }
    fn workflow_edition() -> Option<String> {
        current().edition.clone()
    }
    fn tags_for(params: &Metadata) -> rlmesh::Result<Option<rlmesh::adapters::EnvTags>> {
        let data = current();
        let Some(callback) = data.vt.tags_for else {
            return Ok(data.tags.clone());
        };
        let params = RLMeshValue::new(Value::Dict(params.clone()));
        let mut out = std::ptr::null_mut();
        let _callback_frame = crate::abi::status::CallbackFrame::enter();
        let status = unsafe { callback(data.ud, &params, &mut out) };
        let error = (status != 0).then(|| ffi::callback_error("env"));
        let tags = if out.is_null() {
            None
        } else {
            Some(unsafe { Box::from_raw(out) }.0)
        };
        if let Some(error) = error {
            return Err(error);
        }
        Ok(tags)
    }
    fn enumerate_variants() -> rlmesh::Result<Option<Vec<Variant>>> {
        let data = current();
        let Some(callback) = data.vt.enumerate_variants else {
            return Ok(None);
        };
        let mut sink = RLMeshVariantSink(Vec::new());
        let _callback_frame = crate::abi::status::CallbackFrame::enter();
        check(unsafe { callback(data.ud, &mut sink) })?;
        Ok(Some(sink.0))
    }
    fn prepare(&mut self) -> rlmesh::Result<()> {
        let d = current();
        if let Some(f) = d.vt.prepare {
            let _callback_frame = crate::abi::status::CallbackFrame::enter();
            check(unsafe { f(d.ud) })?;
        }
        Ok(())
    }
    fn close(&mut self) -> rlmesh::Result<()> {
        let d = current();
        if let Some(f) = d.vt.close {
            let _callback_frame = crate::abi::status::CallbackFrame::enter();
            check(unsafe { f(d.ud) })?;
        }
        Ok(())
    }
    fn make(&mut self, kwargs: &Metadata) -> rlmesh::Result<MadeEnv> {
        let d = current();
        let kwargs = RLMeshValue::new(Value::Dict(kwargs.clone()));
        let (mut env, mut vector) = (std::ptr::null_mut(), std::ptr::null_mut());
        let _callback_frame = crate::abi::status::CallbackFrame::enter();
        let status =
            unsafe { d.vt.make.expect("validated make")(d.ud, &kwargs, &mut env, &mut vector) };
        let error = (status != 0).then(|| ffi::callback_error("env"));
        // Reclaim every output on failure too. Dropping a handle never closes
        // its env, matching factory's ownership rule.
        let env = if env.is_null() {
            None
        } else {
            Some(unsafe { Box::from_raw(env) })
        };
        let vector = if vector.is_null() {
            None
        } else {
            Some(unsafe { Box::from_raw(vector) })
        };
        if let Some(error) = error {
            return Err(error);
        }
        match (env, vector) {
            (Some(env), None) => {
                if env.leased.load(Ordering::Acquire) {
                    return Err(Error::Runtime("factory returned a busy env".into()));
                }
                let gates = d
                    .serve_gates
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                match gates {
                    None => Ok(MadeEnv::env(env.inner)),
                    Some(gates) => {
                        let borrowed = BorrowedEnv {
                            inner: env.inner,
                            stopped: Arc::new(AtomicBool::new(false)),
                        };
                        gates
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(MadeGate::Env(borrowed.clone()));
                        Ok(MadeEnv::env(borrowed))
                    }
                }
            }
            (None, Some(env)) => {
                if env.leased.load(Ordering::Acquire) {
                    return Err(Error::Runtime("factory returned a busy vector env".into()));
                }
                let gates = d
                    .serve_gates
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                match gates {
                    None => Ok(MadeEnv::vector(env.inner)),
                    Some(gates) => {
                        let borrowed = BorrowedVectorEnv {
                            inner: env.inner,
                            stopped: Arc::new(AtomicBool::new(false)),
                        };
                        gates
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(MadeGate::Vector(borrowed.clone()));
                        Ok(MadeEnv::vector(borrowed))
                    }
                }
            }
            _ => Err(Error::Type(
                "factory make must return exactly one env or vector env".into(),
            )),
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_factory_new(
    vt: *const RLMeshEnvFactoryVtable,
    ud: *mut c_void,
    attrs: *const RLMeshEnvFactoryAttributes,
    out: *mut *mut RLMeshEnvFactory,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let vt = unsafe { sizing::read(vt, "RLMeshEnvFactoryVtable") }?
            .ok_or_else(|| CapiError::type_error("factory vtable is required"))?;
        if vt.make.is_none() {
            return Err(CapiError::type_error("factory must define make"));
        }
        let attrs = unsafe { sizing::read(attrs, "RLMeshEnvFactoryAttributes") }?;
        let (mut tags, mut params, mut resets, mut edition) = (None, Vec::new(), Vec::new(), None);
        if let Some(a) = attrs {
            tags = unsafe { a.tags.as_ref() }.map(|t| t.0.clone());
            for &ptr in
                unsafe { crate::spaces::slice_arg(a.tag_params, a.tag_params_len, "tag_params") }?
            {
                let p = unsafe { sizing::read(ptr, "RLMeshTagParam") }?
                    .ok_or_else(|| CapiError::invalid_arg("null tag param"))?;
                params.push(TagParam {
                    name: unsafe { ffi::text(p.name) }?
                        .ok_or_else(|| CapiError::invalid_arg("null tag param name"))?,
                    default_: unsafe {
                        // SAFETY: the FFI contract supplies live handles and writable output storage.
                        value_ref(p.default_)
                    }?
                    .clone(),
                    choices: unsafe {
                        // SAFETY: the FFI contract supplies live handles and writable output storage.
                        value_ref(p.choices)
                    }?
                    .as_list()?
                    .to_vec(),
                });
            }
            for &ptr in unsafe {
                crate::spaces::slice_arg(a.reset_options, a.reset_options_len, "reset_options")
            }? {
                resets.push(
                    unsafe { ffi::text(ptr) }?
                        .ok_or_else(|| CapiError::invalid_arg("null reset option"))?,
                );
            }
            edition = unsafe { ffi::text(a.workflow_edition) }?;
        }
        let data = Arc::new(Data {
            vt,
            ud,
            owned: AtomicBool::new(false),
            tags,
            params,
            reset_options: resets,
            edition,
            serve_gates: Mutex::new(None),
        });
        let _scope = Scope::new(Arc::clone(&data));
        rlmesh::binding::env_factory_check::<Factory>()?;
        data.owned.store(true, Ordering::Release);
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                Box::into_raw(Box::new(RLMeshEnvFactory {
                    data,
                    busy: Arc::new(AtomicBool::new(false)),
                })),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_factory_free(f: *mut RLMeshEnvFactory) {
    guard_value((), || {
        if !f.is_null() {
            drop(unsafe { Box::from_raw(f) });
        }
    });
}
macro_rules! hook {
    ($name:ident,$method:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(f: *mut RLMeshEnvFactory) -> RLMeshStatus {
            guard(|| {
                let f = unsafe {
                    // SAFETY: the FFI contract supplies live handles and writable output storage.
                    require(f)
                }?;
                let _busy = enter(f)?;
                let _scope = Scope::new(Arc::clone(&f.data));
                Factory.$method()?;
                Ok(())
            })
        }
    };
}
hook!(rlmesh_env_factory_prepare, prepare);
hook!(rlmesh_env_factory_close, close);
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_factory_make(
    f: *mut RLMeshEnvFactory,
    kwargs: *const RLMeshValue,
    out_env: *mut *mut RLMeshEnv,
    out_vector: *mut *mut RLMeshVectorEnv,
) -> RLMeshStatus {
    guard(|| {
        if out_env.is_null() || out_vector.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let f = unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            require(f)
        }?;
        let _busy = enter(f)?;
        let _scope = Scope::new(Arc::clone(&f.data));
        let kwargs = ffi::dict(kwargs)?.unwrap_or_default();
        let made = Factory.make_env(&kwargs)?;
        let (env, vector) = match made {
            MadeEnv::Env(e) => (
                Box::into_raw(Box::new(RLMeshEnv {
                    inner: Arc::new(Mutex::new(e)),
                    leased: Arc::new(AtomicBool::new(false)),
                })),
                std::ptr::null_mut(),
            ),
            MadeEnv::Vector(e) => (
                std::ptr::null_mut(),
                Box::into_raw(Box::new(RLMeshVectorEnv {
                    inner: Arc::new(Mutex::new(e)),
                    leased: Arc::new(AtomicBool::new(false)),
                })),
            ),
        };
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out_env, env)
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out_vector, vector)
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_factory_tags(
    f: *const RLMeshEnvFactory,
    out: *mut *mut RLMeshEnvTags,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                require(f)?
                    .data
                    .tags
                    .clone()
                    .map_or(std::ptr::null_mut(), |t| {
                        Box::into_raw(Box::new(RLMeshEnvTags::new(t)))
                    }),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_factory_tags_for(
    f: *mut RLMeshEnvFactory,
    params: *const RLMeshValue,
    out: *mut *mut RLMeshEnvTags,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let f = unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            require(f)
        }?;
        let _busy = enter(f)?;
        let _scope = Scope::new(Arc::clone(&f.data));
        let tags = Factory::tags_for(&ffi::dict(params)?.unwrap_or_default())?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                tags.map_or(std::ptr::null_mut(), |t| {
                    Box::into_raw(Box::new(RLMeshEnvTags::new(t)))
                }),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_factory_enumerate_variants(
    f: *mut RLMeshEnvFactory,
    out: *mut *mut RLMeshVariantList,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let f = unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            require(f)
        }?;
        let _busy = enter(f)?;
        let _scope = Scope::new(Arc::clone(&f.data));
        let variants = Factory::enumerate_variants()?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                variants.map_or(std::ptr::null_mut(), |v| {
                    Box::into_raw(Box::new(RLMeshVariantList(
                        v.into_iter().map(RLMeshVariant::new).collect(),
                    )))
                }),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_factory_reset_options(
    f: *const RLMeshEnvFactory,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let v = Value::List(
            unsafe {
                // SAFETY: the FFI contract supplies live handles and writable output storage.
                require(f)
            }?
            .data
            .reset_options
            .iter()
            .cloned()
            .map(Value::Text)
            .collect(),
        );
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, into_handle(v))
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_factory_workflow_edition(
    f: *const RLMeshEnvFactory,
    out_has: *mut bool,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() || out_has.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let edition = &unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            require(f)
        }?
        .data
        .edition;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out_has, edition.is_some())
        }?;
        ffi::bytes(out, edition.clone().unwrap_or_default())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_factory_serve(
    f: *mut RLMeshEnvFactory,
    address: *const c_char,
    options: *const RLMeshEnvFactoryServeOptions,
) -> RLMeshStatus {
    guard(|| {
        let f = unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            require(f)
        }?;
        let _busy = enter(f)?;
        let _scope = Scope::new(Arc::clone(&f.data));
        let address =
            unsafe { ffi::text(address) }?.ok_or_else(|| CapiError::invalid_arg("null address"))?;
        let o = unsafe { sizing::read(options, "RLMeshEnvFactoryServeOptions") }?;
        let opts = match o {
            None => rlmesh::EnvFactoryServeOptions::default(),
            Some(o) => rlmesh::EnvFactoryServeOptions {
                num_envs: o.num_envs,
                workflow_edition: unsafe { ffi::text(o.workflow_edition) }?,
                make_kwargs: ffi::dict(o.make_kwargs)?.unwrap_or_default(),
                cancel: unsafe { o.cancel.as_ref() }.map(|t| t.0.clone()),
            },
        };
        let _drain = ServeDrain::new(Arc::clone(&f.data));
        rlmesh::binding::env_factory_native_serve(&mut Factory, &address, opts)?;
        Ok(())
    })
}
struct Entry {
    raw: RLMeshTagParam,
    _name: CString,
    default_: RLMeshValue,
    choices: RLMeshValue,
}
#[allow(clippy::vec_box)] // Lent struct fields point into each stable heap allocation.
pub struct RLMeshTagParamList(Vec<Box<Entry>>);
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_env_factory_tag_params(
    f: *const RLMeshEnvFactory,
    out: *mut *mut RLMeshTagParamList,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let f = unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            require(f)
        }?;
        let mut entries = Vec::new();
        for p in &f.data.params {
            let name = CString::new(p.name.clone())
                .map_err(|_| CapiError::invalid_arg("tag param name contains NUL"))?;
            let mut entry = Box::new(Entry {
                raw: RLMeshTagParam {
                    struct_size: std::mem::size_of::<RLMeshTagParam>(),
                    name: name.as_ptr(),
                    default_: std::ptr::null(),
                    choices: std::ptr::null(),
                },
                _name: name,
                default_: RLMeshValue::new(p.default_.clone()),
                choices: RLMeshValue::new(Value::List(p.choices.clone())),
            });
            entry.raw.default_ = &entry.default_;
            entry.raw.choices = &entry.choices;
            entries.push(entry);
        }
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, Box::into_raw(Box::new(RLMeshTagParamList(entries))))
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_tag_param_list_len(p: *const RLMeshTagParamList) -> usize {
    guard_value(0, || unsafe { p.as_ref() }.map_or(0, |p| p.0.len()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_tag_param_list_get(
    p: *const RLMeshTagParamList,
    i: usize,
) -> *const RLMeshTagParam {
    guard_value(std::ptr::null(), || {
        unsafe { p.as_ref() }
            .and_then(|p| p.0.get(i))
            .map_or(std::ptr::null(), |e| std::ptr::from_ref(&e.raw))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_tag_param_list_free(p: *mut RLMeshTagParamList) {
    guard_value((), || {
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) });
        }
    });
}

enum MadeGate {
    Env(BorrowedEnv),
    Vector(BorrowedVectorEnv),
}
struct ServeDrain(Arc<Data>);
impl ServeDrain {
    fn new(data: Arc<Data>) -> Self {
        *data.serve_gates.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(Arc::new(Mutex::new(Vec::new())));
        Self(data)
    }
}
impl Drop for ServeDrain {
    fn drop(&mut self) {
        let gates = self
            .0
            .serve_gates
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .expect("serve owns its gate collection");
        let gates = gates.lock().unwrap_or_else(|e| e.into_inner());
        for gate in gates.iter() {
            match gate {
                MadeGate::Env(e) => e.stopped.store(true, Ordering::Release),
                MadeGate::Vector(e) => e.stopped.store(true, Ordering::Release),
            }
        }
        for gate in gates.iter() {
            match gate {
                MadeGate::Env(e) => e.finish(true),
                MadeGate::Vector(e) => e.finish(true),
            }
        }
    }
}

/// Owns exclusive factory admission while a facade run/session borrows it.
pub(crate) struct FactoryBorrow {
    factory: Factory,
    data: Arc<Data>,
    _drain: ServeDrain,
    _busy: Busy,
}
impl FactoryBorrow {
    pub(crate) fn scope(&self) -> Scope {
        Scope::new(Arc::clone(&self.data))
    }
    // Only the factory field is lent for the session lifetime; scope() reads
    // separate data. The binding drops the Session before this allocation.
    pub(crate) unsafe fn scope_static(ptr: *const Self) -> Scope {
        Scope::new(Arc::clone(unsafe { &(*ptr).data }))
    }
    pub(crate) unsafe fn target_static(ptr: *mut Self) -> rlmesh::EnvTarget<'static> {
        rlmesh::EnvTarget::factory(unsafe { &mut (*ptr).factory })
    }
    pub(crate) fn describable(&mut self) -> rlmesh::Describable<'_> {
        (&mut self.factory).into()
    }
    pub(crate) fn target(&mut self) -> rlmesh::EnvTarget<'_> {
        rlmesh::EnvTarget::factory(&mut self.factory)
    }
}
pub(crate) fn borrow_factory(ptr: *const RLMeshEnvFactory) -> Result<FactoryBorrow, CapiError> {
    let f = unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        require(ptr)
    }?;
    let busy = enter(f)?;
    Ok(FactoryBorrow {
        factory: Factory,
        data: Arc::clone(&f.data),
        _busy: busy,
        _drain: ServeDrain::new(Arc::clone(&f.data)),
    })
}
