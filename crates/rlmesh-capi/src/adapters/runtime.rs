//! Shared callables, adapter protocol and facade execution.
#![allow(unsafe_code)]
use super::*;
use a::AdapterBase;
use std::sync::atomic::{AtomicBool, Ordering};
type TransformFn =
    unsafe extern "C" fn(*mut c_void, *const RLMeshValue, *mut *mut RLMeshValue) -> i32;
type Free = unsafe extern "C" fn(*mut c_void);
struct Owner {
    user: usize,
    free: Option<Free>,
    owned: AtomicBool,
}
impl Drop for Owner {
    fn drop(&mut self) {
        if self.owned.load(Ordering::Acquire)
            && let Some(f) = self.free
        {
            unsafe { f(self.user as *mut c_void) }
        }
    }
}
fn value_call(f: TransformFn, user: usize, input: Value) -> rlmesh::Result<Value> {
    let input = RLMeshValue::new(input);
    let mut out = std::ptr::null_mut();
    clear_last_error();
    let status = unsafe { f(user as *mut c_void, &input, &mut out) };
    let value = unsafe { ffi::take(out) };
    if status != 0 {
        return Err(ffi::callback_error("context"));
    }
    value.ok_or_else(|| rlmesh::Error::Type("callback returned no Value".into()))
}
macro_rules! transform {
    ($ty:ident,$native:ident,$new:ident,$clone:ident,$free:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $new(
            f: Option<TransformFn>,
            user: *mut c_void,
            free: Option<Free>,
            out: *mut *mut $ty,
        ) -> RLMeshStatus {
            guard(|| {
                write_out(out, std::ptr::null_mut())?;
                let f = f.ok_or_else(|| CapiError::type_error("transform function is required"))?;
                let owner = Arc::new(Owner {
                    user: user as usize,
                    free,
                    owned: AtomicBool::new(false),
                });
                let captured = Arc::clone(&owner);
                let value = a::$native::from(move |input: &Value| {
                    value_call(f, captured.user, input.clone())
                });
                write_out(out, Box::into_raw(Box::new($ty(value))))?;
                owner.owned.store(true, Ordering::Release);
                Ok(())
            })
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $clone(p: *const $ty) -> *mut $ty {
            guard_ptr(|| {
                Ok(Box::into_raw(Box::new($ty(unsafe { p.as_ref() }
                    .ok_or_else(|| CapiError::invalid_arg("null transform"))?
                    .0
                    .clone()))))
            })
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $free(p: *mut $ty) {
            guard_value((), || {
                if !p.is_null() {
                    drop(unsafe { Box::from_raw(p) })
                }
            })
        }
    };
}
transform!(
    RLMeshObsTransform,
    ObsTransform,
    rlmesh_obs_transform_new,
    rlmesh_obs_transform_clone,
    rlmesh_obs_transform_free
);
transform!(
    RLMeshRotationTransform,
    RotationTransform,
    rlmesh_rotation_transform_new,
    rlmesh_rotation_transform_clone,
    rlmesh_rotation_transform_free
);
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshAdapterVtable {
    pub struct_size: usize,
    pub transform_obs: Option<TransformFn>,
    pub transform_action: Option<TransformFn>,
    pub reset: Option<unsafe extern "C" fn(*mut c_void) -> i32>,
    pub observe: Option<unsafe extern "C" fn(*mut c_void, *const RLMeshValue) -> i32>,
    pub explain: Option<unsafe extern "C" fn(*mut c_void, *mut *const c_char) -> i32>,
    pub free_user_data: Option<Free>,
}
struct CustomAdapter {
    v: RLMeshAdapterVtable,
    owner: Owner,
}
impl AdapterBase for CustomAdapter {
    fn transform_obs(&mut self, raw: Value) -> rlmesh::Result<Value> {
        value_call(
            self.v.transform_obs.expect("validated slot"),
            self.owner.user,
            raw,
        )
    }
    fn transform_action(&mut self, raw: Value) -> rlmesh::Result<Value> {
        value_call(
            self.v.transform_action.expect("validated slot"),
            self.owner.user,
            raw,
        )
    }
    fn reset(&mut self) -> rlmesh::Result<()> {
        if let Some(f) = self.v.reset {
            clear_last_error();
            if unsafe { f(self.owner.user as *mut c_void) } != 0 {
                return Err(ffi::callback_error("context"));
            }
        }
        Ok(())
    }
    fn observe(&mut self, raw: Value) -> rlmesh::Result<()> {
        if let Some(f) = self.v.observe {
            let raw = RLMeshValue::new(raw);
            clear_last_error();
            if unsafe { f(self.owner.user as *mut c_void, &raw) } != 0 {
                return Err(ffi::callback_error("context"));
            }
        }
        Ok(())
    }
    fn explain(&self) -> rlmesh::Result<String> {
        if let Some(f) = self.v.explain {
            let mut out = std::ptr::null();
            clear_last_error();
            if unsafe { f(self.owner.user as *mut c_void, &mut out) } != 0 {
                return Err(ffi::callback_error("context"));
            }
            return unsafe { required(out, "explain text") }
                .map_err(|e| rlmesh::Error::Type(e.message));
        }
        Ok("AdapterBase (custom adapter)".into())
    }
}
enum Kind {
    Resolved(a::Adapter),
    Custom(CustomAdapter),
}
struct Shared {
    resolved: Option<a::Adapter>,
    inner: Mutex<Kind>,
    notes: Vec<a::Advisory>,
    messages: Vec<std::ffi::CString>,
    keys: Vec<String>,
    windows: Vec<(String, u64, u64)>,
}
#[derive(Clone)]
pub struct RLMeshAdapter(Arc<Shared>);
impl RLMeshAdapter {
    pub(crate) fn native(&self) -> Result<a::Adapter, CapiError> {
        self.0
            .resolved
            .clone()
            .ok_or_else(|| CapiError::type_error("Reader requires a resolved Adapter"))
    }
    fn resolved(value: a::Adapter) -> Self {
        let notes = value.advisories();
        let keys = value.history_keys();
        let windows = value.history_windows();
        let messages = notes
            .iter()
            .map(|n| std::ffi::CString::new(n.message.as_str()).unwrap_or_default())
            .collect();
        Self(Arc::new(Shared {
            resolved: Some(value.clone()),
            inner: Mutex::new(Kind::Resolved(value)),
            notes,
            messages,
            keys,
            windows,
        }))
    }
    fn call<R>(
        &self,
        f: impl FnOnce(&mut dyn AdapterBase) -> rlmesh::Result<R>,
    ) -> rlmesh::Result<R> {
        thread_local! {static ACTIVE:std::cell::RefCell<Vec<usize>>=const{std::cell::RefCell::new(Vec::new())};}
        let id = Arc::as_ptr(&self.0) as usize;
        if ACTIVE.with(|a| a.borrow().contains(&id)) {
            return Err(rlmesh::Error::Runtime(
                "adapter callback is already active".into(),
            ));
        }
        struct Entry(usize);
        impl Drop for Entry {
            fn drop(&mut self) {
                ACTIVE.with(|a| {
                    let popped = a.borrow_mut().pop();
                    debug_assert_eq!(popped, Some(self.0));
                })
            }
        }
        let mut inner = self.0.inner.lock().unwrap_or_else(|e| e.into_inner());
        ACTIVE.with(|a| a.borrow_mut().push(id));
        let _entry = Entry(id);
        match &mut *inner {
            Kind::Resolved(a) => f(a),
            Kind::Custom(a) => f(a),
        }
    }
    fn resolved_only(&self) -> Result<&Shared, CapiError> {
        if self.0.resolved.is_none() {
            return Err(CapiError::type_error(
                "operation requires a resolved Adapter",
            ));
        }
        Ok(&self.0)
    }
}
impl AdapterBase for RLMeshAdapter {
    fn transform_obs(&mut self, v: Value) -> rlmesh::Result<Value> {
        self.call(|a| a.transform_obs(v))
    }
    fn transform_action(&mut self, v: Value) -> rlmesh::Result<Value> {
        self.call(|a| a.transform_action(v))
    }
    fn observe(&mut self, v: Value) -> rlmesh::Result<()> {
        self.call(|a| a.observe(v))
    }
    fn reset(&mut self) -> rlmesh::Result<()> {
        self.call(|a| a.reset())
    }
    fn explain(&self) -> rlmesh::Result<String> {
        self.call(|a| a.explain())
    }
}
fn require<'a>(p: *const RLMeshAdapter) -> Result<&'a RLMeshAdapter, CapiError> {
    unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null adapter"))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_new(
    v: *const RLMeshAdapterVtable,
    user: *mut c_void,
    out: *mut *mut RLMeshAdapter,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let v = unsafe { sized(v, "RLMeshAdapterVtable") }?;
        if v.transform_obs.is_none() || v.transform_action.is_none() {
            return Err(CapiError::type_error(
                "adapter requires transform_obs and transform_action",
            ));
        }
        let value = RLMeshAdapter(Arc::new(Shared {
            resolved: None,
            inner: Mutex::new(Kind::Custom(CustomAdapter {
                v,
                owner: Owner {
                    user: user as usize,
                    free: v.free_user_data,
                    owned: AtomicBool::new(true),
                },
            })),
            notes: Vec::new(),
            messages: Vec::new(),
            keys: Vec::new(),
            windows: Vec::new(),
        }));
        write_out(out, Box::into_raw(Box::new(value)))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_clone(p: *const RLMeshAdapter) -> *mut RLMeshAdapter {
    guard_ptr(|| Ok(Box::into_raw(Box::new(require(p)?.clone()))))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_free(p: *mut RLMeshAdapter) {
    guard_value((), || {
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) })
        }
    })
}
unsafe fn resolve_options(p: *const RLMeshResolveOptions) -> Result<a::ResolveOptions, CapiError> {
    if p.is_null() {
        Ok(Default::default())
    } else {
        unsafe { parse_resolve_options(p) }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapters_resolve(
    tags: *const RLMeshEnvTags,
    obs: *const crate::spaces::RLMeshSpaceSpec,
    act: *const crate::spaces::RLMeshSpaceSpec,
    spec: *const RLMeshModelSpec,
    options: *const RLMeshResolveOptions,
    out: *mut *mut RLMeshAdapter,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let tags = unsafe { tags.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null tags"))?;
        let spec = unsafe { spec.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null spec"))?;
        let obs = crate::space_handle::require_spec(obs)?;
        let act = crate::space_handle::require_spec(act)?;
        let adapter = a::resolve(&tags.0, &obs, &act, &spec.0, unsafe {
            resolve_options(options)
        }?)?;
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshAdapter::resolved(adapter))),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapters_resolve_from_contract(
    contract: *const crate::contract::RLMeshEnvContract,
    spec: *const RLMeshModelSpec,
    options: *const RLMeshResolveOptions,
    out: *mut *mut RLMeshAdapter,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let spec = unsafe { spec.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null spec"))?;
        let wire = unsafe { crate::contract::RLMeshEnvContract::as_ref(contract) }
            .ok_or_else(|| CapiError::invalid_arg("null contract"))?;
        let contract = rlmesh::binding::env_contract_from_wire(wire)?;
        let adapter =
            a::resolve_from_contract(&contract, &spec.0, unsafe { resolve_options(options) }?)?;
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshAdapter::resolved(adapter))),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_transform_obs(
    p: *mut RLMeshAdapter,
    v: *const RLMeshValue,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let value = require(p)?.clone().transform_obs(value_ref(v)?.clone())?;
        write_out(out, into_handle(value))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_transform_action(
    p: *mut RLMeshAdapter,
    v: *const RLMeshValue,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let value = require(p)?
            .clone()
            .transform_action(value_ref(v)?.clone())?;
        write_out(out, into_handle(value))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_observe(
    p: *mut RLMeshAdapter,
    v: *const RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        require(p)?.clone().observe(value_ref(v)?.clone())?;
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_reset(p: *mut RLMeshAdapter) -> RLMeshStatus {
    guard(|| {
        require(p)?.clone().reset()?;
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_explain(
    p: *const RLMeshAdapter,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, require(p)?.explain()?))
}
macro_rules! length {
    ($f:ident,$field:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $f(p: *const RLMeshAdapter, out: *mut usize) -> RLMeshStatus {
            guard(|| write_out(out, require(p)?.resolved_only()?.$field.len()))
        }
    };
}
length!(rlmesh_adapter_advisories_len, notes);
length!(rlmesh_adapter_history_keys_len, keys);
length!(rlmesh_adapter_history_windows_len, windows);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_history_keys_get(
    p: *const RLMeshAdapter,
    i: usize,
    out: *mut RLMeshStr,
) -> RLMeshStatus {
    guard(|| {
        let v = require(p)?.resolved_only()?.keys.get(i);
        write_out(
            out,
            v.map_or(
                RLMeshStr {
                    data: std::ptr::null(),
                    len: 0,
                },
                |s| RLMeshStr::borrowed(s),
            ),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_history_windows_get(
    p: *const RLMeshAdapter,
    i: usize,
    key: *mut RLMeshStr,
    span: *mut u64,
    bytes: *mut u64,
) -> RLMeshStatus {
    guard(|| {
        if key.is_null() || span.is_null() || bytes.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let v = require(p)?.resolved_only()?.windows.get(i);
        write_out(
            key,
            v.map_or(
                RLMeshStr {
                    data: std::ptr::null(),
                    len: 0,
                },
                |s| RLMeshStr::borrowed(&s.0),
            ),
        )?;
        write_out(span, v.map_or(0, |v| v.1))?;
        write_out(bytes, v.map_or(0, |v| v.2))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_advisories_get(
    p: *const RLMeshAdapter,
    i: usize,
    out: *mut RLMeshAdvisory,
) -> RLMeshStatus {
    guard(|| {
        let state = require(p)?.resolved_only()?;
        let n = state
            .notes
            .get(i)
            .ok_or_else(|| CapiError::invalid_arg("advisory index out of range"))?;
        let mut arena = Arena::default();
        let mut v = view_advisory(&mut arena, n);
        v.message = state.messages[i].as_ptr();
        unsafe { out_struct(out, v, "RLMeshAdvisory") }
    })
}
type Predict = Box<dyn FnMut(Value) -> rlmesh::Result<Value> + Send>;
pub struct RLMeshWrappedPredict(Mutex<a::WrappedPredict<RLMeshAdapter, Predict>>);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapter_wrap_predict(
    p: *mut RLMeshAdapter,
    f: Option<TransformFn>,
    user: *mut c_void,
    free: Option<Free>,
    out: *mut *mut RLMeshWrappedPredict,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let adapter = require(p)?.clone();
        let f = f.ok_or_else(|| CapiError::type_error("predict function is required"))?;
        let owner = Arc::new(Owner {
            user: user as usize,
            free,
            owned: AtomicBool::new(false),
        });
        let captured = Arc::clone(&owner);
        let predict: Predict = Box::new(move |value| value_call(f, captured.user, value));
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshWrappedPredict(Mutex::new(
                adapter.wrap_predict(predict),
            )))),
        )?;
        owner.owned.store(true, Ordering::Release);
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_wrapped_predict_call(
    p: *mut RLMeshWrappedPredict,
    v: *const RLMeshValue,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let p = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null predict"))?;
        let mut f = p.0.try_lock().map_err(|_| {
            CapiError::from(rlmesh::Error::Runtime(
                "wrapped predict is already active".into(),
            ))
        })?;
        let value = f.call(value_ref(v)?.clone())?;
        write_out(out, into_handle(value))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_wrapped_predict_free(p: *mut RLMeshWrappedPredict) {
    guard_value((), || {
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) })
        }
    })
}
