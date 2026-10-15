//! Adapter-dependent readers and owned frame-source shells over the facade.
#![allow(unsafe_code)] // C callbacks and documented owner-borrowing render thunks.
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, clear_last_error, guard, guard_value},
};
use crate::adapters::{
    self as ad, RLMeshAdaptersText, RLMeshImage, RLMeshObservationRoles, RLMeshState,
};
use crate::codec::RLMeshBytes;
use crate::ffi;
use crate::run_hooks::{RLMeshRunContext, RLMeshStepEvent};
use crate::session_api::{self, RLMeshSession};
use crate::value::handle::{RLMeshValue, into_handle, value_ref, write_out};
use rlmesh::{FrameSources, ReadItem, Reader, RenderFn, Value};
use std::ffi::{c_char, c_void};
use std::sync::{Arc, Mutex};
pub struct RLMeshReadItem(pub(crate) ReadItem);
pub struct RLMeshReader(Reader);
unsafe fn item<'a>(p: *const RLMeshReadItem) -> Result<&'a ReadItem, CapiError> {
    unsafe { p.as_ref() }
        .map(|p| &p.0)
        .ok_or_else(|| CapiError::invalid_arg("null read item"))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_read_item_role(
    role: *const c_char,
    out: *mut *mut RLMeshReadItem,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let role = unsafe { ad::required(role, "role") }?;
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshReadItem(ReadItem::Role(role)))),
        )
    })
}
macro_rules! leaf_item {
    ($f:ident,$ty:ident,$parse:ident,$variant:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $f(p: *const $ty, out: *mut *mut RLMeshReadItem) -> RLMeshStatus {
            guard(|| {
                write_out(out, std::ptr::null_mut())?;
                let value = unsafe { ad::$parse(p) }?;
                write_out(
                    out,
                    Box::into_raw(Box::new(RLMeshReadItem(ReadItem::$variant(value)))),
                )
            })
        }
    };
}
leaf_item!(rlmesh_read_item_image, RLMeshImage, parse_image, Image);
leaf_item!(rlmesh_read_item_state, RLMeshState, parse_state, State);
leaf_item!(rlmesh_read_item_text, RLMeshAdaptersText, parse_text, Text);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_read_item_free(p: *mut RLMeshReadItem) {
    guard_value((), || {
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) })
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_reader_new(
    p: *const ad::runtime::RLMeshAdapter,
    roles: *const *const c_char,
    n: usize,
    out: *mut *mut RLMeshReader,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let adapter = unsafe { p.as_ref() }
            .ok_or_else(|| CapiError::invalid_arg("null adapter"))?
            .native()?;
        let roles = unsafe { ad::strings(roles, n) }?.unwrap_or_default();
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshReader(Reader::new(adapter, roles)))),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_reader_free(p: *mut RLMeshReader) {
    guard_value((), || {
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) })
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_reader_roles_len(p: *const RLMeshReader) -> usize {
    guard_value(0, || unsafe { p.as_ref() }.map_or(0, |p| p.0.roles().len()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_reader_roles_get(
    p: *const RLMeshReader,
    i: usize,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        let p = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null reader"))?;
        ffi::bytes(
            out,
            p.0.roles()
                .get(i)
                .ok_or_else(|| CapiError::invalid_arg("role index out of range"))?
                .clone(),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_reader_call(
    p: *const RLMeshReader,
    obs: *const RLMeshValue,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let p = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null reader"))?;
        write_out(out, into_handle(Value::Dict(p.0.call(value_ref(obs)?)?)))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_session_reader(
    p: *mut RLMeshSession,
    items: *const *const RLMeshReadItem,
    n: usize,
    out: *mut *mut RLMeshReader,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let items = unsafe { crate::spaces::slice_arg(items, n, "read items") }?
            .iter()
            .map(|&p| unsafe { item(p) }.cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let reader = session_api::with(p, |s| Ok(s.reader(&items)?))?;
        write_out(out, Box::into_raw(Box::new(RLMeshReader(reader))))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_session_read(
    p: *mut RLMeshSession,
    obs: *const RLMeshValue,
    read: *const RLMeshReadItem,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let obs = value_ref(obs)?;
        let item = unsafe { item(read) }?;
        let value = if let Some(context) = session_api::own_hook(p) {
            context.0.read(obs, item)?
        } else {
            session_api::with(p, |s| Ok(s.read(obs, item.clone())?))?
        };
        write_out(out, into_handle(value))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_context_read(
    p: *const RLMeshRunContext,
    obs: *const RLMeshValue,
    read: *const RLMeshReadItem,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let context =
            unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null context"))?;
        let value = context.0.read(value_ref(obs)?, unsafe { item(read) }?)?;
        write_out(out, into_handle(value))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_step_event_read(
    p: *const RLMeshStepEvent,
    read: *const RLMeshReadItem,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let event = unsafe { sizing::read(p, "RLMeshStepEvent") }?
            .ok_or_else(|| CapiError::invalid_arg("null event"))?;
        let context = unsafe { event.context.as_ref() }
            .ok_or_else(|| CapiError::invalid_arg("event has no read context"))?;
        write_out(
            out,
            into_handle(
                context
                    .0
                    .read(value_ref(event.observation)?, unsafe { item(read) }?)?,
            ),
        )
    })
}
pub(crate) struct RolesView {
    roles: RLMeshObservationRoles,
    _arena: ad::Arena,
}
// SAFETY: fully built immutable lent records refer only to this view's owned arena.
unsafe impl Send for RolesView {}
unsafe impl Sync for RolesView {}
impl RolesView {
    pub(crate) fn new(roles: &rlmesh::adapters::ObservationRoles) -> Self {
        let mut arena = ad::Arena::default();
        let roles = ad::view_observation_roles(&mut arena, roles);
        Self {
            roles,
            _arena: arena,
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_session_observation_roles(
    p: *mut RLMeshSession,
    out: *mut RLMeshObservationRoles,
) -> RLMeshStatus {
    guard(|| {
        let roles = session_api::with(p, |s| Ok(s.observation_roles()?))?;
        let h = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null session"))?;
        let view = h.roles.get_or_init(|| RolesView::new(&roles));
        unsafe { ad::out_struct(out, view.roles, "RLMeshObservationRoles") }
    })
}

type Render = unsafe extern "C" fn(*mut c_void, *mut *mut RLMeshValue) -> i32;
struct Owner {
    user: usize,
    free: Option<unsafe extern "C" fn(*mut c_void)>,
}
impl Drop for Owner {
    fn drop(&mut self) {
        if let Some(f) = self.free {
            unsafe { f(self.user as *mut c_void) }
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    Absent,
    Caller(usize, usize),
    Library(u64),
}
pub struct RLMeshFrameSources {
    inner: Mutex<FrameSources<'static>>,
    // A frozen facade projection: getters need no render-callback lock.
    projection: FrameSources<'static>,
    repr: String,
    source: Source,
    _owner: Option<Arc<Owner>>,
    // Drop order matters: inner's borrowed render thunk drops before this Arc.
    _context: Option<Arc<dyn rlmesh::RunContext>>,
}
impl RLMeshFrameSources {
    fn library(
        roles: Vec<String>,
        label: Option<String>,
        render: Option<RenderFn<'static>>,
    ) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let source = if render.is_some() {
            Source::Library(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
        } else {
            Source::Absent
        };
        let projection = FrameSources {
            roles: roles.clone(),
            render_label: label.clone(),
            render: None,
        };
        let native = FrameSources {
            roles,
            render_label: label,
            render,
        };
        let repr = native.to_string();
        Self {
            inner: Mutex::new(native),
            projection,
            repr,
            source,
            _owner: None,
            _context: None,
        }
    }
    pub(crate) fn into_facade(self) -> FrameSources<'static> {
        let (roles, render_label, has_render) = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            (
                inner.roles.clone(),
                inner.render_label.clone(),
                inner.render.is_some(),
            )
        };
        let render = has_render.then(|| {
            RenderFn::new(move || {
                self.render()?
                    .ok_or_else(|| rlmesh::Error::Runtime("render source is absent".into()))
            })
        });
        FrameSources {
            roles,
            render_label,
            render,
        }
    }
    fn render(&self) -> rlmesh::Result<Option<Value>> {
        let mut inner = self
            .inner
            .try_lock()
            .map_err(|_| rlmesh::Error::Runtime("frame render is already active".into()))?;
        inner.render.as_mut().map(RenderFn::call).transpose()
    }
}
fn sources(p: *const RLMeshFrameSources) -> Result<&'static RLMeshFrameSources, CapiError> {
    unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null frame sources"))
}
fn context_sources(context: Arc<dyn rlmesh::RunContext>) -> rlmesh::Result<RLMeshFrameSources> {
    let native = context.frame_sources()?;
    // SAFETY: frame_sources borrows the RunContext pointee of this Arc, whose
    // address is stable. The handle retains that exact Arc allocation and drops
    // its native sources before the Arc. No session/env borrow is extended:
    // their availability remains checked by the facade's original render thunk.
    let native = unsafe { std::mem::transmute::<FrameSources<'_>, FrameSources<'static>>(native) };
    let mut handle = RLMeshFrameSources::library(native.roles, native.render_label, native.render);
    handle._context = Some(context);
    Ok(handle)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_run_context_frame_sources(
    p: *const RLMeshRunContext,
    out: *mut *mut RLMeshFrameSources,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let context =
            unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null context"))?;
        let value = context_sources(Arc::clone(&context.0))?;
        write_out(out, Box::into_raw(Box::new(value)))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_session_frame_sources(
    p: *mut RLMeshSession,
    out: *mut *mut RLMeshFrameSources,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        if let Some(context) = session_api::own_hook(p) {
            let value = context_sources(context.0)?;
            return write_out(out, Box::into_raw(Box::new(value)));
        }
        let (roles, label, has) = session_api::with(p, |s| {
            let sources = s.frame_sources()?;
            Ok((
                sources.roles,
                sources.render_label,
                sources.render.is_some(),
            ))
        })?;
        let render = if has {
            let owner = unsafe { session_api::SessionRender::new(p) }?;
            Some(RenderFn::new(move || owner.call()))
        } else {
            None
        };
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshFrameSources::library(roles, label, render))),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_frame_sources_new(
    roles: *const *const c_char,
    n: usize,
    label: *const c_char,
    render: Option<Render>,
    user: *mut c_void,
    free: Option<unsafe extern "C" fn(*mut c_void)>,
    out: *mut *mut RLMeshFrameSources,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let roles = unsafe { ad::strings(roles, n) }?.unwrap_or_default();
        let label = unsafe { ffi::text(label) }?;
        let owner = Arc::new(Owner {
            user: user as usize,
            free,
        });
        let captured = Arc::clone(&owner);
        let source = render.map_or(Source::Absent, |f| {
            Source::Caller(f as usize, user as usize)
        });
        let render = render.map(|f| {
            RenderFn::new(move || {
                clear_last_error();
                let mut out = std::ptr::null_mut();
                let status = unsafe { f(captured.user as *mut c_void, &mut out) };
                let value = unsafe { ffi::take(out) };
                if status != 0 {
                    return Err(ffi::callback_error("context"));
                }
                value.ok_or_else(|| rlmesh::Error::Type("render callback returned no Value".into()))
            })
        });
        let projection = FrameSources {
            roles: roles.clone(),
            render_label: label.clone(),
            render: None,
        };
        let native = FrameSources {
            roles,
            render_label: label,
            render,
        };
        let repr = native.to_string();
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshFrameSources {
                inner: Mutex::new(native),
                projection,
                repr,
                source,
                _owner: Some(owner),
                _context: None,
            })),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_frame_sources_free(p: *mut RLMeshFrameSources) {
    guard_value((), || {
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) })
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_frame_sources_roles_len(p: *const RLMeshFrameSources) -> usize {
    guard_value(0, || sources(p).map_or(0, |p| p.projection.roles.len()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_frame_sources_labels_len(p: *const RLMeshFrameSources) -> usize {
    guard_value(0, || sources(p).map_or(0, |p| p.projection.labels().len()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_frame_sources_roles_get(
    p: *const RLMeshFrameSources,
    i: usize,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        let inner = &sources(p)?.projection;
        ffi::bytes(
            out,
            inner
                .roles
                .get(i)
                .ok_or_else(|| CapiError::invalid_arg("role index out of range"))?
                .clone(),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_frame_sources_labels_get(
    p: *const RLMeshFrameSources,
    i: usize,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        let inner = &sources(p)?.projection;
        ffi::bytes(
            out,
            inner
                .labels()
                .get(i)
                .ok_or_else(|| CapiError::invalid_arg("label index out of range"))?
                .clone(),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_frame_sources_render_label(
    p: *const RLMeshFrameSources,
    out: *mut RLMeshBytes,
    has: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() || has.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let inner = &sources(p)?.projection;
        write_out(has, inner.render_label.is_some())?;
        if let Some(label) = &inner.render_label {
            ffi::bytes(out, label.clone())?;
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_frame_sources_default_label(
    p: *const RLMeshFrameSources,
    out: *mut RLMeshBytes,
    has: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() || has.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let inner = &sources(p)?.projection;
        let label = inner.default_label();
        write_out(has, label.is_some())?;
        if let Some(label) = label {
            ffi::bytes(out, label)?;
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_frame_sources_render(
    p: *const RLMeshFrameSources,
    out: *mut *mut RLMeshValue,
    has: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() || has.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let value = sources(p)?.render()?;
        write_out(has, value.is_some())?;
        if let Some(v) = value {
            write_out(out, into_handle(v))?;
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_frame_sources_repr(
    p: *const RLMeshFrameSources,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, sources(p)?.repr.clone()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_frame_sources_equals(
    p: *const RLMeshFrameSources,
    q: *const RLMeshFrameSources,
) -> bool {
    guard_value(false, || {
        let (Ok(p), Ok(q)) = (sources(p), sources(q)) else {
            return false;
        };
        p.projection == q.projection && p.source == q.source
    })
}
