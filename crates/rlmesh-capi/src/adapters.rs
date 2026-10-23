//! Stable typed adapters built on the public Rust facade.
#![allow(unsafe_code)] // C callbacks, borrowed immutable trees and owned handles.
mod inputs;
mod records;
pub(crate) mod runtime;
mod views;
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard, guard_ptr, guard_value},
};
use crate::codec::RLMeshBytes;
use crate::ffi;
use crate::space_handle::RLMeshStr;
use crate::spaces::slice_arg;
use crate::value::handle::{RLMeshValue, into_handle, value_ref, write_out};
pub(crate) use inputs::*;
pub(crate) use records::*;
use rlmesh::{Value, adapters as a};

use std::ffi::{c_char, c_void};
use std::sync::{Arc, Mutex};
pub(crate) use views::*;
pub(crate) struct RLMeshObsTransform(pub(crate) a::ObsTransform);
pub(crate) struct RLMeshRotationTransform(pub(crate) a::RotationTransform);
#[derive(Clone)]
pub struct RLMeshModelSpec(pub(crate) a::ModelSpec, Arc<SpecView>);
#[derive(Clone)]
pub struct RLMeshEnvTags(pub(crate) a::EnvTags, Arc<TagsView>);
impl RLMeshModelSpec {
    pub(crate) fn new(value: a::ModelSpec) -> Self {
        let view = SpecView::new(&value);
        Self(value, Arc::new(view))
    }
}
impl RLMeshEnvTags {
    pub(crate) fn new(value: a::EnvTags) -> Self {
        let view = TagsView::new(&value);
        Self(value, Arc::new(view))
    }
}
pub(crate) unsafe fn required(p: *const c_char, name: &str) -> Result<String, CapiError> {
    unsafe { ffi::text(p) }?.ok_or_else(|| CapiError::type_error(format!("{name} is required")))
}
unsafe fn sized<T: sizing::SizedInput>(p: *const T, name: &str) -> Result<T, CapiError> {
    unsafe { sizing::read(p, name) }?.ok_or_else(|| CapiError::invalid_arg(format!("null {name}")))
}
unsafe fn optional_slice<T: Copy>(
    p: *const T,
    n: usize,
    name: &str,
) -> Result<Option<Vec<T>>, CapiError> {
    let values = unsafe { slice_arg(p, n, name) }?;
    Ok((!p.is_null()).then(|| values.to_vec()))
}
pub(crate) unsafe fn strings(
    p: *const *const c_char,
    n: usize,
) -> Result<Option<Vec<String>>, CapiError> {
    let values = unsafe { slice_arg(p, n, "strings") }?;
    if p.is_null() {
        return Ok(None);
    }
    Ok(Some(
        values
            .iter()
            .map(|&p| unsafe { required(p, "string") })
            .collect::<Result<_, _>>()?,
    ))
}
unsafe fn accept<E: a::Vocabulary>(
    p: *const i32,
    n: usize,
    other: *const *const c_char,
    read: unsafe fn(i32, *const c_char) -> Result<a::Tolerant<E>, CapiError>,
) -> Result<Option<Vec<a::Tolerant<E>>>, CapiError> {
    let values = unsafe { slice_arg(p, n, "accept set") }?;
    if p.is_null() {
        return Ok(None);
    }
    Ok(Some(
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| unsafe {
                read(
                    v,
                    if v == 1000 && !other.is_null() {
                        *other.add(i)
                    } else {
                        std::ptr::null()
                    },
                )
            })
            .collect::<Result<_, _>>()?,
    ))
}
unsafe fn axis(
    has: bool,
    scalar: f64,
    p: *const f64,
    n: usize,
    name: &str,
) -> Result<Option<a::AxisValue>, CapiError> {
    let values = unsafe { optional_slice(p, n, name) }?;
    if has && values.is_some() {
        return Err(CapiError::invalid_arg(format!(
            "{name} and axis_{name} are mutually exclusive"
        )));
    }
    Ok(if has {
        Some(a::AxisValue::Scalar(scalar))
    } else {
        values.map(a::AxisValue::PerAxis)
    })
}
unsafe fn encoding_arm(arm: RLMeshEncodingArm) -> Result<Option<a::EncodingArm>, CapiError> {
    if !arm.transform.is_null() && !arm.entrypoint.is_null() {
        return Err(CapiError::invalid_arg(
            "transform and entrypoint are mutually exclusive",
        ));
    }
    Ok(if let Some(t) = unsafe { arm.transform.as_ref() } {
        Some(a::EncodingArm::Transform(t.0.clone()))
    } else {
        unsafe { ffi::text(arm.entrypoint) }?.map(a::EncodingArm::Entrypoint)
    })
}
unsafe fn state_encoding(
    p: *const i32,
    n: usize,
    other: *const *const c_char,
    custom: *const RLMeshCustomEncoding,
) -> Result<Option<a::StateEncoding>, CapiError> {
    if !custom.is_null() && (!p.is_null() || n != 0) {
        return Err(CapiError::invalid_arg(
            "encoding and custom_encoding are mutually exclusive",
        ));
    }
    Ok(if !custom.is_null() {
        Some(a::StateEncoding::Custom(unsafe {
            parse_custom_encoding(custom)
        }?))
    } else {
        unsafe { accept(p, n, other, tolerant_rotation_encoding) }?.map(a::StateEncoding::Native)
    })
}
unsafe fn action_encoding(
    has: bool,
    e: i32,
    custom: *const RLMeshCustomEncoding,
) -> Result<Option<a::ActionEncoding>, CapiError> {
    if has && !custom.is_null() {
        return Err(CapiError::invalid_arg(
            "encoding and custom_encoding are mutually exclusive",
        ));
    }
    Ok(if !custom.is_null() {
        Some(a::ActionEncoding::Custom(unsafe {
            parse_custom_encoding(custom)
        }?))
    } else if has {
        Some(a::ActionEncoding::Native(enum_rotation_encoding(e)?))
    } else {
        None
    })
}
macro_rules! value_api {
    ($ty:ident,$native:ident,$new:ident,$clone:ident,$free:ident,$eq:ident,$repr:ident,$dict:ident,$json:ident,$meta:ident,$fromdict:ident,$fromjson:ident,$frommeta:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $clone(p: *const $ty) -> *mut $ty {
            guard_ptr(|| {
                Ok(Box::into_raw(Box::new(
                    unsafe { p.as_ref() }
                        .ok_or_else(|| CapiError::invalid_arg("null handle"))?
                        .clone(),
                )))
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
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $eq(p: *const $ty, q: *const $ty) -> bool {
            guard_value(false, || {
                match (unsafe { p.as_ref() }, unsafe { q.as_ref() }) {
                    (Some(p), Some(q)) => p.0 == q.0,
                    _ => false,
                }
            })
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $repr(p: *const $ty, out: *mut RLMeshBytes) -> RLMeshStatus {
            guard(|| {
                ffi::bytes(
                    out,
                    unsafe { p.as_ref() }
                        .ok_or_else(|| CapiError::invalid_arg("null handle"))?
                        .0
                        .to_string(),
                )
            })
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $dict(p: *const $ty, out: *mut *mut RLMeshValue) -> RLMeshStatus {
            guard(|| {
                unsafe {
                    // SAFETY: the FFI contract supplies live handles and writable output storage.
                    write_out(out, std::ptr::null_mut())
                }?;
                let value = unsafe { p.as_ref() }
                    .ok_or_else(|| CapiError::invalid_arg("null handle"))?
                    .0
                    .to_dict()?;
                unsafe {
                    // SAFETY: the FFI contract supplies live handles and writable output storage.
                    write_out(out, into_handle(Value::Dict(value)))
                }
            })
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $meta(p: *const $ty, out: *mut *mut RLMeshValue) -> RLMeshStatus {
            guard(|| {
                unsafe {
                    // SAFETY: the FFI contract supplies live handles and writable output storage.
                    write_out(out, std::ptr::null_mut())
                }?;
                let value = unsafe { p.as_ref() }
                    .ok_or_else(|| CapiError::invalid_arg("null handle"))?
                    .0
                    .to_metadata()?;
                unsafe {
                    // SAFETY: the FFI contract supplies live handles and writable output storage.
                    write_out(out, into_handle(Value::Dict(value)))
                }
            })
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $json(p: *const $ty, out: *mut RLMeshBytes) -> RLMeshStatus {
            guard(|| {
                ffi::bytes(
                    out,
                    unsafe { p.as_ref() }
                        .ok_or_else(|| CapiError::invalid_arg("null handle"))?
                        .0
                        .to_json()?,
                )
            })
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fromdict(
            p: *const RLMeshValue,
            out: *mut *mut $ty,
        ) -> RLMeshStatus {
            guard(|| {
                unsafe {
                    // SAFETY: the FFI contract supplies live handles and writable output storage.
                    write_out(out, std::ptr::null_mut())
                }?;
                let value = a::$native::from_dict(
                    unsafe {
                        // SAFETY: the FFI contract supplies live handles and writable output storage.
                        value_ref(p)
                    }?
                    .as_dict()?,
                )?;
                unsafe {
                    // SAFETY: the FFI contract supplies live handles and writable output storage.
                    write_out(out, Box::into_raw(Box::new($ty::new(value))))
                }
            })
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $frommeta(
            p: *const RLMeshValue,
            out: *mut *mut $ty,
        ) -> RLMeshStatus {
            guard(|| {
                unsafe {
                    // SAFETY: the FFI contract supplies live handles and writable output storage.
                    write_out(out, std::ptr::null_mut())
                }?;
                if let Some(value) = a::$native::from_metadata(
                    unsafe {
                        // SAFETY: the FFI contract supplies live handles and writable output storage.
                        value_ref(p)
                    }?
                    .as_dict()?,
                )? {
                    unsafe {
                        // SAFETY: the FFI contract supplies live handles and writable output storage.
                        write_out(out, Box::into_raw(Box::new($ty::new(value))))
                    }?;
                }
                Ok(())
            })
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fromjson(p: *const c_char, out: *mut *mut $ty) -> RLMeshStatus {
            guard(|| {
                unsafe {
                    // SAFETY: the FFI contract supplies live handles and writable output storage.
                    write_out(out, std::ptr::null_mut())
                }?;
                let value = a::$native::from_json(&unsafe { required(p, "json") }?)?;
                unsafe {
                    // SAFETY: the FFI contract supplies live handles and writable output storage.
                    write_out(out, Box::into_raw(Box::new($ty::new(value))))
                }
            })
        }
    };
}
value_api!(
    RLMeshModelSpec,
    ModelSpec,
    rlmesh_model_spec_new,
    rlmesh_model_spec_clone,
    rlmesh_model_spec_free,
    rlmesh_model_spec_equals,
    rlmesh_model_spec_repr,
    rlmesh_model_spec_to_dict,
    rlmesh_model_spec_to_json,
    rlmesh_model_spec_to_metadata,
    rlmesh_model_spec_from_dict,
    rlmesh_model_spec_from_json,
    rlmesh_model_spec_from_metadata
);
value_api!(
    RLMeshEnvTags,
    EnvTags,
    rlmesh_env_tags_new,
    rlmesh_env_tags_clone,
    rlmesh_env_tags_free,
    rlmesh_env_tags_equals,
    rlmesh_env_tags_repr,
    rlmesh_env_tags_to_dict,
    rlmesh_env_tags_to_json,
    rlmesh_env_tags_to_metadata,
    rlmesh_env_tags_from_dict,
    rlmesh_env_tags_from_json,
    rlmesh_env_tags_from_metadata
);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_spec_new(
    input: *const RLMeshInputNode,
    output: *const RLMeshAction,
    out: *mut *mut RLMeshModelSpec,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let value = a::ModelSpec::new(unsafe { parse_input(input, &mut Vec::new()) }?, unsafe {
            parse_action(output)
        }?)?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, Box::into_raw(Box::new(RLMeshModelSpec::new(value))))
        }
    })
}
/// The same spec accepting the named projections in `allow` (Rust's
/// `ModelSpec::with_allow`); a fresh handle in `*out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_spec_with_allow(
    spec: *const RLMeshModelSpec,
    allow: *const *const c_char,
    allow_len: usize,
    out: *mut *mut RLMeshModelSpec,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let spec = unsafe { spec.as_ref() }
            .ok_or_else(|| CapiError::invalid_arg("null RLMeshModelSpec"))?;
        let allow = unsafe { strings(allow, allow_len) }?.unwrap_or_default();
        let value = spec.0.clone().with_allow(allow);
        unsafe {
            /* SAFETY: the FFI caller supplies valid writable output storage for this call. */
            write_out(out, Box::into_raw(Box::new(RLMeshModelSpec::new(value))))
        }
    })
}
/// The named projections the spec accepts, borrowed for the spec's lifetime;
/// `*out_len` receives their count. NULL spec yields NULL and 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_spec_allow(
    p: *const RLMeshModelSpec,
    out_len: *mut usize,
) -> *const *const c_char {
    guard_value(std::ptr::null(), || {
        let (items, len) =
            unsafe { p.as_ref() }.map_or((std::ptr::null(), 0), |p| (p.1.allow, p.1.allow_len));
        if let Some(out_len) = unsafe { out_len.as_mut() } {
            *out_len = len;
        }
        items
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_tags_new(
    input: *const RLMeshObsNode,
    output: *const RLMeshAction,
    out: *mut *mut RLMeshEnvTags,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, std::ptr::null_mut())
        }?;
        let value = a::EnvTags::new(unsafe { parse_obs(input, &mut Vec::new()) }?, unsafe {
            parse_action(output)
        }?)?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, Box::into_raw(Box::new(RLMeshEnvTags::new(value))))
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_spec_input(
    p: *const RLMeshModelSpec,
) -> *const RLMeshInputNode {
    guard_value(std::ptr::null(), || {
        unsafe { p.as_ref() }.map_or(std::ptr::null(), |p| p.1.input)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_spec_output(
    p: *const RLMeshModelSpec,
) -> *const RLMeshAction {
    guard_value(std::ptr::null(), || {
        unsafe { p.as_ref() }.map_or(std::ptr::null(), |p| p.1.action)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_tags_observation(
    p: *const RLMeshEnvTags,
) -> *const RLMeshObsNode {
    guard_value(std::ptr::null(), || {
        unsafe { p.as_ref() }.map_or(std::ptr::null(), |p| p.1.observation)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_tags_action(p: *const RLMeshEnvTags) -> *const RLMeshAction {
    guard_value(std::ptr::null(), || {
        unsafe { p.as_ref() }.map_or(std::ptr::null(), |p| p.1.action)
    })
}

/// Write only the known output prefix; preserve the caller's declared size and tail.
pub(crate) unsafe fn out_struct<T: Copy>(
    out: *mut T,
    mut value: T,
    name: &str,
) -> Result<(), CapiError> {
    if out.is_null() {
        return Err(CapiError::invalid_arg("null out"));
    }
    let size = unsafe { out.cast::<usize>().read_unaligned() };
    if size < std::mem::size_of::<T>() {
        return Err(CapiError::invalid_arg(format!(
            "{name}.struct_size is below its minimum"
        )));
    }
    unsafe {
        std::ptr::from_mut(&mut value)
            .cast::<usize>()
            .write_unaligned(size);
        out.write_unaligned(value)
    }
    Ok(())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_tags_observation_roles(
    p: *const RLMeshEnvTags,
    out: *mut RLMeshObservationRoles,
) -> RLMeshStatus {
    guard(|| {
        let p = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null tags"))?;
        unsafe { out_struct(out, p.1.roles, "RLMeshObservationRoles") }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_observation_roles_equals(
    p: *const RLMeshObservationRoles,
    q: *const RLMeshObservationRoles,
    out: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                { parse_observation_roles(p) }? == { parse_observation_roles(q) }?,
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_observation_roles_repr(
    p: *const RLMeshObservationRoles,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, unsafe { parse_observation_roles(p) }?.to_string()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_advisory_equals(
    p: *const RLMeshAdvisory,
    q: *const RLMeshAdvisory,
    out: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(out, { parse_advisory(p) }? == { parse_advisory(q) }?)
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_advisory_repr(
    p: *const RLMeshAdvisory,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| ffi::bytes(out, format!("{:?}", unsafe { parse_advisory(p) }?)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_action_dim(p: *const RLMeshAction, out: *mut u32) -> RLMeshStatus {
    guard(|| unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(out, { parse_action(p) }?.dim())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_custom_encoding_is_entrypoint(
    p: *const RLMeshCustomEncoding,
    out: *mut bool,
) -> RLMeshStatus {
    guard(|| unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(out, { parse_custom_encoding(p) }?.is_entrypoint())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_custom_encoding_width(
    p: *const RLMeshCustomEncoding,
    out: *mut u32,
) -> RLMeshStatus {
    guard(|| unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(out, { parse_custom_encoding(p) }?.width())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_rotation_encoding_dims(v: i32, out: *mut u32) -> RLMeshStatus {
    guard(|| unsafe {
        /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
        write_out(out, enum_rotation_encoding(v)?.dims())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_adapters_parts_len() -> usize {
    a::PARTS.len()
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_adapters_parts_get(i: usize) -> *const c_char {
    a::PARTS
        .get(i)
        .map_or(std::ptr::null(), |s| crate::value::dtype::static_str(s))
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_adapters_rotation_dims_len() -> usize {
    a::ROTATION_DIMS.len()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapters_rotation_dims_get(
    i: usize,
    e: *mut i32,
    d: *mut u32,
) -> RLMeshStatus {
    guard(|| {
        if e.is_null() || d.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let entry = a::ROTATION_DIMS.get(i);
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(e, entry.map_or(0, |e| number_rotation_encoding(e.0)))
        }?;
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(d, entry.map_or(0, |e| e.1))
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_image_with_size(p: *mut RLMeshImage, size: u32) -> RLMeshStatus {
    guard(|| {
        let mut v = unsafe { sized(p, "RLMeshImage") }?;
        let _ = unsafe { parse_image(p) }?.with_size(size)?;
        v.has_height = true;
        v.height = size;
        v.has_width = true;
        v.width = size;
        unsafe { out_struct(p, v, "RLMeshImage") }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_rotation_from_matrix(
    rows: *const [f64; 3],
    values: *mut f64,
    out: *mut RLMeshRotation,
) -> RLMeshStatus {
    guard(|| {
        if rows.is_null() || values.is_null() || out.is_null() {
            return Err(CapiError::invalid_arg("null matrix, value or out"));
        }
        let rows = unsafe { [rows.read(), rows.add(1).read(), rows.add(2).read()] };
        let value = a::Rotation::from_matrix(rows);
        // This is an output record: inspect only its size, and leave future
        // caller fields untouched rather than applying input-tail validation.
        let mut v: RLMeshRotation = unsafe { std::mem::zeroed() };
        v.struct_size = std::mem::size_of_val(&v);
        v.encoding = number_rotation_encoding(value.encoding);
        v.value = values;
        v.value_len = 6;
        unsafe { out_struct(out, v, "RLMeshRotation") }?;
        unsafe { std::ptr::copy_nonoverlapping(value.value.as_ptr(), values, 6) };
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapters_previous(
    role: *const c_char,
    o: *const RLMeshPreviousOptions,
    out: *mut RLMeshState,
) -> RLMeshStatus {
    guard(|| {
        unsafe { required(role, "role") }?;
        let options = if o.is_null() {
            a::PreviousOptions::default()
        } else {
            unsafe { parse_previous_options(o) }?
        };
        let value = a::Previous(unsafe { required(role, "role") }?, options);
        let mut arena = Arena::default();
        let mut view = view_state(&mut arena, &value);
        view.role = role;
        view.part = if o.is_null() {
            std::ptr::null()
        } else {
            unsafe { (*o).part }
        };
        unsafe { out_struct(out, view, "RLMeshState") }
    })
}
// Constructors store borrowed pointers. Composition validates and copies trees.
macro_rules! node_leaf {
    ($f:ident,$node:ident,$ty:ident,$kind:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $f(leaf: *const $ty) -> $node {
            $node {
                kind: $kind,
                leaf: leaf.cast(),
                keys: std::ptr::null(),
                children: std::ptr::null(),
                children_len: 0,
            }
        }
    };
}
node_leaf!(rlmesh_input_node_image, RLMeshInputNode, RLMeshImage, 1);
node_leaf!(rlmesh_input_node_state, RLMeshInputNode, RLMeshState, 2);
node_leaf!(rlmesh_input_node_concat, RLMeshInputNode, RLMeshConcat, 3);
node_leaf!(
    rlmesh_input_node_text,
    RLMeshInputNode,
    RLMeshAdaptersText,
    4
);
node_leaf!(rlmesh_input_node_custom, RLMeshInputNode, RLMeshCustom, 5);
node_leaf!(rlmesh_obs_node_image_tag, RLMeshObsNode, RLMeshImageTag, 1);
node_leaf!(rlmesh_obs_node_state_tag, RLMeshObsNode, RLMeshStateTag, 2);
node_leaf!(rlmesh_obs_node_text_tag, RLMeshObsNode, RLMeshTextTag, 3);
node_leaf!(rlmesh_obs_node_split, RLMeshObsNode, RLMeshSplit, 4);
macro_rules! composites {
    ($dict:ident,$tuple:ident,$ty:ident,$d:expr,$t:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $dict(keys: *const RLMeshStr, children: *const $ty, n: usize) -> $ty {
            $ty {
                kind: $d,
                leaf: std::ptr::null(),
                keys,
                children,
                children_len: n,
            }
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn $tuple(children: *const $ty, n: usize) -> $ty {
            $ty {
                kind: $t,
                leaf: std::ptr::null(),
                keys: std::ptr::null(),
                children,
                children_len: n,
            }
        }
    };
}
composites!(
    rlmesh_input_node_dict,
    rlmesh_input_node_tuple,
    RLMeshInputNode,
    6,
    7
);
composites!(
    rlmesh_obs_node_dict,
    rlmesh_obs_node_tuple,
    RLMeshObsNode,
    5,
    6
);
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_concat_part_role(role: *const c_char) -> RLMeshConcatPart {
    RLMeshConcatPart {
        kind: 1,
        role,
        leaf: std::ptr::null(),
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_concat_part_state(leaf: *const RLMeshState) -> RLMeshConcatPart {
    RLMeshConcatPart {
        kind: 2,
        role: std::ptr::null(),
        leaf: leaf.cast(),
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_concat_part_constant(leaf: *const RLMeshConstant) -> RLMeshConcatPart {
    RLMeshConcatPart {
        kind: 3,
        role: std::ptr::null(),
        leaf: leaf.cast(),
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_register_entrypoint(
    name: *const c_char,
    transform: *const RLMeshObsTransform,
) -> RLMeshStatus {
    guard(|| {
        let transform = unsafe { transform.as_ref() }
            .ok_or_else(|| CapiError::invalid_arg("null transform"))?;
        rlmesh::ext::register_entrypoint(
            &unsafe { required(name, "entrypoint") }?,
            transform.0.clone(),
        )?;
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_unregister_entrypoint(
    name: *const c_char,
    expected: *const RLMeshObsTransform,
    out_removed: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        unsafe { write_out(out_removed, false) }?;
        if name.is_null() {
            return Err(CapiError::invalid_arg("null entrypoint"));
        }
        let name = unsafe { required(name, "entrypoint") }?;
        let expected = unsafe { expected.as_ref() }.map(|transform| &transform.0);
        let removed = rlmesh::ext::unregister_entrypoint(&name, expected)?;
        unsafe { write_out(out_removed, removed) }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapters_tag(
    p: *mut crate::env_api::RLMeshEnv,
    tags: *const RLMeshEnvTags,
    o: *const RLMeshTagOptions,
) -> RLMeshStatus {
    guard(|| {
        let h = unsafe { p.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null env"))?;
        let _claim = crate::env_api::Claim::new(&h.leased)?;
        let tags = unsafe { tags.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null tags"))?;
        let options = if o.is_null() {
            a::TagOptions::default()
        } else {
            unsafe { parse_tag_options(o) }?
        };
        let tagged = a::tag(Arc::clone(&h.inner), &tags.0, options)
            .map_err(|e| CapiError::from(e.into_parts().1))?;
        h.inner = Arc::new(Mutex::new(Box::new(tagged)));
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_adapters_tag_vector(
    p: *mut crate::env_api::RLMeshVectorEnv,
    tags: *const RLMeshEnvTags,
    o: *const RLMeshTagOptions,
) -> RLMeshStatus {
    guard(|| {
        let h = unsafe { p.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null env"))?;
        let _claim = crate::env_api::Claim::new(&h.leased)?;
        let tags = unsafe { tags.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null tags"))?;
        let options = if o.is_null() {
            a::TagOptions::default()
        } else {
            unsafe { parse_tag_options(o) }?
        };
        let tagged = a::tag_vector(Arc::clone(&h.inner), &tags.0, options)
            .map_err(|e| CapiError::from(e.into_parts().1))?;
        h.inner = Arc::new(Mutex::new(Box::new(tagged)));
        Ok(())
    })
}
