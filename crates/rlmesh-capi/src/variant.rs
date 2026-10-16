//! Mutable Variant values and callback catalog sinks.
#![allow(unsafe_code)]
use crate::abi::{
    sizing,
    status::{CapiError, RLMeshStatus, guard, guard_value},
};
use crate::codec::RLMeshBytes;
use crate::ffi;
use crate::value::handle::{RLMeshValue, value_ref, write_out};
use rlmesh::{Metadata, Value, Variant, VariantOptions};
use std::ffi::c_char;

pub struct RLMeshVariant {
    pub(crate) value: Variant,
    params: RLMeshValue,
    metadata: RLMeshValue,
}
impl RLMeshVariant {
    pub(crate) fn new(value: Variant) -> Self {
        Self {
            params: RLMeshValue::new(Value::Dict(value.params.clone())),
            metadata: RLMeshValue::new(Value::Dict(value.metadata.clone())),
            value,
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshVariantOptions {
    pub struct_size: usize,
    pub metadata: *const RLMeshValue,
}
/// # Safety
/// A non-NULL handle stays live for the returned reference's lifetime.
unsafe fn require<'a>(ptr: *const RLMeshVariant) -> Result<&'a RLMeshVariant, CapiError> {
    unsafe { ptr.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null variant"))
}
fn mapping(ptr: *const RLMeshValue) -> Result<Metadata, CapiError> {
    unsafe {
        // SAFETY: the FFI contract supplies live handles and writable output storage.
        value_ref(ptr)
    }?
    .as_dict()
    .cloned()
    .map_err(|_| CapiError::invalid_arg("variant fields must be dicts"))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_variant_new(
    id: *const c_char,
    params: *const RLMeshValue,
    options: *const RLMeshVariantOptions,
    out: *mut *mut RLMeshVariant,
) -> RLMeshStatus {
    guard(|| {
        if out.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        let id = unsafe { ffi::text(id) }?.ok_or_else(|| CapiError::invalid_arg("null id"))?;
        let params = mapping(params)?;
        let o = unsafe { sizing::read(options, "RLMeshVariantOptions") }?;
        let metadata = match o {
            Some(o) if !o.metadata.is_null() => mapping(o.metadata)?,
            _ => Metadata::new(),
        };
        unsafe {
            /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
            write_out(
                out,
                Box::into_raw(Box::new(RLMeshVariant::new(Variant::new(
                    id,
                    params,
                    VariantOptions { metadata },
                )))),
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_variant_free(v: *mut RLMeshVariant) {
    guard_value((), || {
        if !v.is_null() {
            drop(unsafe { Box::from_raw(v) });
        }
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_variant_id(
    v: *const RLMeshVariant,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        ffi::bytes(
            out,
            unsafe {
                // SAFETY: the FFI contract supplies live handles and writable output storage.
                require(v)
            }?
            .value
            .id
            .clone(),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_variant_repr(
    v: *const RLMeshVariant,
    out: *mut RLMeshBytes,
) -> RLMeshStatus {
    guard(|| {
        ffi::bytes(
            out,
            unsafe {
                // SAFETY: the FFI contract supplies live handles and writable output storage.
                require(v)
            }?
            .value
            .to_string(),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_variant_params(v: *const RLMeshVariant) -> *const RLMeshValue {
    guard_value(std::ptr::null(), || {
        unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            require(v)
        }
        .map_or(std::ptr::null(), |v| std::ptr::from_ref(&v.params))
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_variant_metadata(v: *const RLMeshVariant) -> *const RLMeshValue {
    guard_value(std::ptr::null(), || {
        unsafe {
            // SAFETY: the FFI contract supplies live handles and writable output storage.
            require(v)
        }
        .map_or(std::ptr::null(), |v| std::ptr::from_ref(&v.metadata))
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_variant_equals(a: *const RLMeshVariant, b: *const RLMeshVariant) -> bool {
    guard_value(false, || {
        match (
            unsafe {
                /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
                require(a)
            },
            unsafe {
                /* SAFETY: the FFI caller supplies live input handles and valid writable output storage for this call. */
                require(b)
            },
        ) {
            (Ok(a), Ok(b)) => a.value == b.value,
            _ => false,
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_variant_set_id(
    v: *mut RLMeshVariant,
    id: *const c_char,
) -> RLMeshStatus {
    guard(|| {
        let id = unsafe { ffi::text(id) }?.ok_or_else(|| CapiError::invalid_arg("null id"))?;
        let v = unsafe { v.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null variant"))?;
        v.value.id = id;
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_variant_set_params(
    v: *mut RLMeshVariant,
    params: *const RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        let params = mapping(params)?;
        let v = unsafe { v.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null variant"))?;
        v.value.params = params.clone();
        v.params = RLMeshValue::new(Value::Dict(params));
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_variant_set_metadata(
    v: *mut RLMeshVariant,
    metadata: *const RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        let metadata = mapping(metadata)?;
        let v = unsafe { v.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null variant"))?;
        v.value.metadata = metadata.clone();
        v.metadata = RLMeshValue::new(Value::Dict(metadata));
        Ok(())
    })
}
pub struct RLMeshVariantList(pub(crate) Vec<RLMeshVariant>);
pub struct RLMeshVariantSink(pub(crate) Vec<Variant>);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_variant_sink_push(
    sink: *mut RLMeshVariantSink,
    variant: *mut RLMeshVariant,
) -> RLMeshStatus {
    guard(|| {
        let sink = unsafe { sink.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null sink"))?;
        if variant.is_null() {
            return Err(CapiError::invalid_arg("null variant"));
        }
        sink.0.push(unsafe { Box::from_raw(variant) }.value);
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_variant_list_len(list: *const RLMeshVariantList) -> usize {
    guard_value(0, || unsafe { list.as_ref() }.map_or(0, |l| l.0.len()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_variant_list_get(
    list: *const RLMeshVariantList,
    i: usize,
) -> *const RLMeshVariant {
    guard_value(std::ptr::null(), || {
        unsafe { list.as_ref() }
            .and_then(|l| l.0.get(i))
            .map_or(std::ptr::null(), std::ptr::from_ref)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_variant_list_free(list: *mut RLMeshVariantList) {
    guard_value((), || {
        if !list.is_null() {
            drop(unsafe { Box::from_raw(list) });
        }
    });
}
