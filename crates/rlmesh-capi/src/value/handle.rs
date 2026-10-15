//! Immutable core `Value` handles. Clones share the tree and borrowed children.
#![allow(unsafe_code)] // FFI: raw pointers and owned handles.

use std::sync::{Arc, OnceLock};

use rlmesh::{Metadata, Value};

use super::tensor::{RLMeshTensor, tensor_ref};
use crate::abi::status::{CapiError, RLMeshStatus, guard, guard_ptr, guard_value};
use crate::space_handle::RLMeshStr;

#[derive(Clone)]
pub struct RLMeshValue(Arc<Node>);
struct Node {
    value: Value,
    children: OnceLock<Vec<RLMeshValue>>,
}

impl RLMeshValue {
    pub(crate) fn new(value: Value) -> Self {
        Self(Arc::new(Node {
            value,
            children: OnceLock::new(),
        }))
    }
    pub(crate) fn value(&self) -> &Value {
        &self.0.value
    }
    fn children(&self) -> &[RLMeshValue] {
        self.0.children.get_or_init(|| match self.value() {
            Value::List(items) | Value::Tuple(items) => {
                items.iter().cloned().map(Self::new).collect()
            }
            Value::Dict(map) => map.values().cloned().map(Self::new).collect(),
            _ => Vec::new(),
        })
    }
}

#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RLMeshValueKind {
    Invalid = 0,
    None = 1,
    Bool = 2,
    Int = 3,
    Float = 4,
    Text = 5,
    Bytes = 6,
    Tensor = 7,
    List = 8,
    Tuple = 9,
    Dict = 10,
}

pub(crate) fn value_ref<'a>(value: *const RLMeshValue) -> Result<&'a Value, CapiError> {
    Ok(handle_ref(value)?.value())
}
fn handle_ref<'a>(value: *const RLMeshValue) -> Result<&'a RLMeshValue, CapiError> {
    unsafe { value.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null value"))
}

#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_kind(value: *const RLMeshValue) -> RLMeshValueKind {
    use rlmesh::ValueKind as K;
    guard_value(RLMeshValueKind::Invalid, || {
        match value_ref(value).map(Value::kind) {
            Ok(K::None) => RLMeshValueKind::None,
            Ok(K::Bool) => RLMeshValueKind::Bool,
            Ok(K::Int) => RLMeshValueKind::Int,
            Ok(K::Float) => RLMeshValueKind::Float,
            Ok(K::Text) => RLMeshValueKind::Text,
            Ok(K::Bytes) => RLMeshValueKind::Bytes,
            Ok(K::Tensor) => RLMeshValueKind::Tensor,
            Ok(K::List) => RLMeshValueKind::List,
            Ok(K::Tuple) => RLMeshValueKind::Tuple,
            Ok(K::Dict) => RLMeshValueKind::Dict,
            _ => RLMeshValueKind::Invalid,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_clone(value: *const RLMeshValue) -> *mut RLMeshValue {
    guard_ptr(|| Ok(Box::into_raw(Box::new(handle_ref(value)?.clone()))))
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_equals(a: *const RLMeshValue, b: *const RLMeshValue) -> bool {
    guard_value(false, || match (value_ref(a), value_ref(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_free(value: *mut RLMeshValue) {
    guard_value((), || {
        if !value.is_null() {
            drop(unsafe { Box::from_raw(value) });
        }
    });
}

pub(crate) fn into_handle(value: Value) -> *mut RLMeshValue {
    Box::into_raw(Box::new(RLMeshValue::new(value)))
}
fn output(out: *mut *mut RLMeshValue, value: Value) -> Result<(), CapiError> {
    if out.is_null() {
        return Err(CapiError::invalid_arg("null out"));
    }
    write_out(out, into_handle(value))
}

#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_none(out: *mut *mut RLMeshValue) -> RLMeshStatus {
    guard(|| output(out, Value::None))
}
macro_rules! scalar {
    ($ctor:ident, $read:ident, $ty:ty, $variant:ident, $access:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $ctor(v: $ty, out: *mut *mut RLMeshValue) -> RLMeshStatus {
            guard(|| output(out, Value::$variant(v)))
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn $read(value: *const RLMeshValue, out: *mut $ty) -> RLMeshStatus {
            guard(|| write_out(out, value_ref(value)?.$access()?))
        }
    };
}
scalar!(rlmesh_value_bool, rlmesh_value_as_bool, bool, Bool, as_bool);
scalar!(rlmesh_value_int, rlmesh_value_as_int, i64, Int, as_int);
scalar!(
    rlmesh_value_float,
    rlmesh_value_as_float,
    f64,
    Float,
    as_float
);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_text(
    v: RLMeshStr,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| output(out, Value::Text(unsafe { v.read("text") }?.to_owned())))
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_as_text(
    value: *const RLMeshValue,
    out: *mut RLMeshStr,
) -> RLMeshStatus {
    guard(|| write_out(out, RLMeshStr::borrowed(value_ref(value)?.as_text()?)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_bytes(
    data: *const u8,
    len: usize,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        output(
            out,
            Value::Bytes(unsafe { crate::spaces::slice_arg(data, len, "bytes") }?.to_vec()),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_as_bytes(
    value: *const RLMeshValue,
    out_data: *mut *const u8,
    out_len: *mut usize,
) -> RLMeshStatus {
    guard(|| {
        let bytes = value_ref(value)?.as_bytes()?;
        if out_data.is_null() || out_len.is_null() {
            return Err(CapiError::invalid_arg("null out"));
        }
        write_out(out_data, bytes.as_ptr())?;
        write_out(out_len, bytes.len())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_tensor(
    tensor: *const RLMeshTensor,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| output(out, Value::Tensor(tensor_ref(tensor)?.clone())))
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_as_tensor(
    value: *const RLMeshValue,
    out: *mut *mut RLMeshTensor,
) -> RLMeshStatus {
    guard(|| super::tensor::output(out, value_ref(value)?.as_tensor()?.clone()))
}

unsafe fn items(items: *const *const RLMeshValue, n: usize) -> Result<Vec<RLMeshValue>, CapiError> {
    unsafe { crate::spaces::slice_arg(items, n, "items") }?
        .iter()
        .map(|&p| handle_ref(p).cloned())
        .collect()
}
fn composite(
    out: *mut *mut RLMeshValue,
    value: Value,
    children: Vec<RLMeshValue>,
) -> Result<(), CapiError> {
    if out.is_null() {
        return Err(CapiError::invalid_arg("null out"));
    }
    let handle = RLMeshValue::new(value);
    let _ = handle.0.children.set(children);
    write_out(out, Box::into_raw(Box::new(handle)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_list(
    items_: *const *const RLMeshValue,
    n: usize,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        let children = unsafe { items(items_, n) }?;
        composite(
            out,
            Value::List(children.iter().map(|v| v.value().clone()).collect()),
            children,
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_tuple(
    items_: *const *const RLMeshValue,
    n: usize,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        let children = unsafe { items(items_, n) }?;
        composite(
            out,
            Value::Tuple(children.iter().map(|v| v.value().clone()).collect()),
            children,
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_dict(
    keys: *const RLMeshStr,
    values: *const *const RLMeshValue,
    n: usize,
    out: *mut *mut RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        let keys = unsafe { crate::spaces::slice_arg(keys, n, "keys") }?;
        let children = unsafe { items(values, n) }?;
        let mut map = Metadata::new();
        let mut handles = Vec::<(String, RLMeshValue)>::new();
        for (key, child) in keys.iter().zip(children) {
            let key = unsafe { key.read("dict key") }?;
            map.insert(key, child.value().clone());
            if let Some((_, existing)) = handles.iter_mut().find(|(k, _)| k == key) {
                *existing = child;
            } else {
                handles.push((key.to_owned(), child));
            }
        }
        composite(
            out,
            Value::Dict(map),
            handles.into_iter().map(|(_, v)| v).collect(),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_len(value: *const RLMeshValue, out: *mut usize) -> RLMeshStatus {
    guard(|| {
        let len = match value_ref(value)? {
            Value::List(v) | Value::Tuple(v) => v.len(),
            Value::Dict(v) => v.len(),
            _ => return Err(CapiError::type_error("value is not a list, tuple or dict")),
        };
        write_out(out, len)
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_get(
    value: *const RLMeshValue,
    i: usize,
    out: *mut *const RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        let handle = handle_ref(value)?;
        if !matches!(handle.value(), Value::List(_) | Value::Tuple(_)) {
            return Err(CapiError::type_error("value is not a list or tuple"));
        }
        write_out(
            out,
            handle
                .children()
                .get(i)
                .map_or(std::ptr::null(), std::ptr::from_ref),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_dict_key_at(
    value: *const RLMeshValue,
    i: usize,
    out: *mut RLMeshStr,
) -> RLMeshStatus {
    guard(|| {
        let key = value_ref(value)?.as_dict()?.keys().nth(i);
        write_out(
            out,
            key.map_or(
                RLMeshStr {
                    data: std::ptr::null(),
                    len: 0,
                },
                RLMeshStr::borrowed,
            ),
        )
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_dict_value_at(
    value: *const RLMeshValue,
    i: usize,
    out: *mut *const RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        let handle = handle_ref(value)?;
        handle.value().as_dict()?;
        write_out(
            out,
            handle
                .children()
                .get(i)
                .map_or(std::ptr::null(), std::ptr::from_ref),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_dict_get(
    value: *const RLMeshValue,
    key: RLMeshStr,
    out: *mut *const RLMeshValue,
) -> RLMeshStatus {
    guard(|| {
        let handle = handle_ref(value)?;
        let map = handle.value().as_dict()?;
        let key = unsafe { key.read("dict key") }?;
        let child = map
            .keys()
            .position(|k| k == key)
            .map(|i| &handle.children()[i]);
        write_out(out, child.map_or(std::ptr::null(), std::ptr::from_ref))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_dict_contains_key(
    value: *const RLMeshValue,
    key: RLMeshStr,
    out: *mut bool,
) -> RLMeshStatus {
    guard(|| {
        write_out(
            out,
            value_ref(value)?
                .as_dict()?
                .contains_key(unsafe { key.read("dict key") }?),
        )
    })
}

pub(crate) fn write_out<T>(out: *mut T, value: T) -> Result<(), CapiError> {
    let slot = unsafe { out.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null out"))?;
    *slot = value;
    Ok(())
}
