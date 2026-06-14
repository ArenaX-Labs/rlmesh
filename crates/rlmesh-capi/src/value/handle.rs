//! `RlmeshValue` — an opaque handle bridging the 7 `SpaceValue` variants. Only
//! `Box` carries a tensor; the scalar/array/composite kinds keep their natural
//! shape (the C side never forces them into a tensor).
#![allow(unsafe_code)] // FFI: raw pointers + repr(transparent) handle.

use std::ffi::{CStr, c_char, c_void};

use rlmesh_spaces::{SpaceValue, Tensor, dtype_size};

use super::dtype::RlmeshDType;
use super::tensor::{RLMESH_DEVICE_CPU, RLMESH_TENSOR_FLAG_READ_ONLY, RlmeshTensor};
use crate::abi::status::{CapiError, RlmeshStatus, guard, guard_ptr, guard_value};

/// An owned RLMesh value. `repr(transparent)` over `SpaceValue` so a borrowed
/// child (`&SpaceValue`) can be handed out as `*const RlmeshValue`.
#[repr(transparent)]
pub struct RlmeshValue(pub(crate) SpaceValue);

/// Value kind, with discriminants pinned to the core `SpaceType`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RlmeshValueKind {
    Box = 1,
    Discrete = 2,
    MultiBinary = 3,
    MultiDiscrete = 4,
    Text = 5,
    Dict = 10,
    Tuple = 11,
}

#[inline]
fn value_ref<'a>(value: *const RlmeshValue) -> Result<&'a SpaceValue, CapiError> {
    unsafe { value.cast::<SpaceValue>().as_ref() }
        .ok_or_else(|| CapiError::invalid_arg("null value"))
}

/// The kind of `value`. `value` must be non-NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_kind(value: *const RlmeshValue) -> RlmeshValueKind {
    guard_value(RlmeshValueKind::Tuple, || {
        match unsafe { value.cast::<SpaceValue>().as_ref() } {
            Some(SpaceValue::Box(_)) => RlmeshValueKind::Box,
            Some(SpaceValue::Discrete(_)) => RlmeshValueKind::Discrete,
            Some(SpaceValue::MultiBinary(_)) => RlmeshValueKind::MultiBinary,
            Some(SpaceValue::MultiDiscrete(_)) => RlmeshValueKind::MultiDiscrete,
            Some(SpaceValue::Text(_)) => RlmeshValueKind::Text,
            Some(SpaceValue::Dict(_)) => RlmeshValueKind::Dict,
            // A null pointer cannot signal an error through this return type; the
            // contract requires `value` to be non-NULL. Fall through to Tuple.
            Some(SpaceValue::Tuple(_)) | None => RlmeshValueKind::Tuple,
        }
    })
}

/// Fill `out` with a borrowed tensor view of a `Box` value (valid while `value`
/// lives). Returns `RLMESH_ERR_INVALID_VALUE` for any other kind.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_as_tensor(
    value: *const RlmeshValue,
    out: *mut RlmeshTensor,
) -> RlmeshStatus {
    guard(|| {
        let value = value_ref(value)?;
        let out = unsafe { out.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null out"))?;
        let SpaceValue::Box(tensor) = value else {
            return Err(CapiError::invalid_value("value is not a Box tensor"));
        };
        let dtype = RlmeshDType::from_core(tensor.dtype())
            .ok_or_else(|| CapiError::invalid_value("unsupported dtype"))?;
        let storage = tensor.storage().as_slice();
        let data = unsafe { storage.as_ptr().add(tensor.byte_offset()) } as *mut c_void;
        *out = RlmeshTensor {
            data,
            ndim: tensor.shape().len() as i32,
            shape: tensor.shape().as_ptr(),
            strides: tensor.strides().map_or(std::ptr::null(), <[i64]>::as_ptr),
            dtype,
            device_type: RLMESH_DEVICE_CPU,
            device_id: 0,
            flags: RLMESH_TENSOR_FLAG_READ_ONLY,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };
        Ok(())
    })
}

/// Construct a `Box` value by copying a contiguous tensor (`strides == NULL`).
/// Returns NULL on error (see `rlmesh_last_error_message`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_box(tensor: *const RlmeshTensor) -> *mut RlmeshValue {
    guard_ptr(|| {
        let tensor =
            unsafe { tensor.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null tensor"))?;
        if !tensor.strides.is_null() {
            return Err(CapiError::invalid_value(
                "rlmesh_value_box requires a contiguous tensor (strides == NULL)",
            ));
        }
        if tensor.ndim < 0 {
            return Err(CapiError::invalid_value("negative ndim"));
        }
        let dtype = tensor
            .dtype
            .to_core()
            .ok_or_else(|| CapiError::invalid_value("unsupported dtype"))?;
        let shape: Vec<i64> = if tensor.ndim == 0 {
            // A scalar carries no dims; shape may be NULL (an empty C++ vector's
            // data()), and from_raw_parts(NULL, 0) is UB — so do not form a slice.
            Vec::new()
        } else if tensor.shape.is_null() {
            return Err(CapiError::invalid_arg("null shape"));
        } else {
            unsafe { std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize) }.to_vec()
        };
        let numel = shape
            .iter()
            .try_fold(1usize, |acc, &dim| {
                if dim < 0 {
                    None
                } else {
                    acc.checked_mul(dim as usize)
                }
            })
            .ok_or_else(|| CapiError::invalid_value("invalid shape"))?;
        let nbytes = numel
            .checked_mul(dtype_size(dtype))
            .ok_or_else(|| CapiError::invalid_value("size overflow"))?;
        let bytes: &[u8] = if nbytes == 0 {
            &[]
        } else if tensor.data.is_null() {
            return Err(CapiError::invalid_arg("null tensor data"));
        } else {
            unsafe { std::slice::from_raw_parts(tensor.data.cast::<u8>(), nbytes) }
        };
        let core = Tensor::from_slice(bytes, &shape, dtype)
            .map_err(|err| CapiError::invalid_value(err.to_string()))?;
        Ok(into_handle(SpaceValue::Box(core)))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_value_discrete(value: i64) -> *mut RlmeshValue {
    into_handle(SpaceValue::Discrete(value))
}

/// Read a `Discrete` value into `out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_as_discrete(
    value: *const RlmeshValue,
    out: *mut i64,
) -> RlmeshStatus {
    guard(|| {
        let out = unsafe { out.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null out"))?;
        match value_ref(value)? {
            SpaceValue::Discrete(n) => {
                *out = *n;
                Ok(())
            }
            _ => Err(CapiError::invalid_value("value is not Discrete")),
        }
    })
}

/// Construct a `Text` value from `len` UTF-8 bytes (not NUL-terminated).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_text(data: *const c_char, len: usize) -> *mut RlmeshValue {
    guard_ptr(|| {
        let bytes = if len == 0 {
            &[][..]
        } else if data.is_null() {
            return Err(CapiError::invalid_arg("null text data"));
        } else {
            unsafe { std::slice::from_raw_parts(data.cast::<u8>(), len) }
        };
        let text = std::str::from_utf8(bytes)
            .map_err(|_| CapiError::invalid_value("text must be UTF-8"))?
            .to_owned();
        Ok(into_handle(SpaceValue::Text(text)))
    })
}

/// Borrow a `Text` value's UTF-8 bytes (valid while `value` lives).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_as_text(
    value: *const RlmeshValue,
    out_ptr: *mut *const c_char,
    out_len: *mut usize,
) -> RlmeshStatus {
    guard(|| {
        let out_ptr =
            unsafe { out_ptr.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null out_ptr"))?;
        let out_len =
            unsafe { out_len.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null out_len"))?;
        match value_ref(value)? {
            SpaceValue::Text(text) => {
                *out_ptr = text.as_ptr().cast::<c_char>();
                *out_len = text.len();
                Ok(())
            }
            _ => Err(CapiError::invalid_value("value is not Text")),
        }
    })
}

/// Construct a `MultiDiscrete` value by copying `n` integers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_multi_discrete(
    data: *const i64,
    n: usize,
) -> *mut RlmeshValue {
    guard_ptr(|| {
        let values = read_slice(data, n)?.to_vec();
        Ok(into_handle(SpaceValue::MultiDiscrete(values)))
    })
}

/// The element count of a `MultiBinary`/`MultiDiscrete` value, else 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_array_len(value: *const RlmeshValue) -> usize {
    guard_value(0, || match unsafe { value.cast::<SpaceValue>().as_ref() } {
        Some(SpaceValue::MultiBinary(v)) => v.len(),
        Some(SpaceValue::MultiDiscrete(v)) => v.len(),
        _ => 0,
    })
}

/// Copy a `MultiDiscrete` value's integers into `out` (capacity `cap`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_copy_multi_discrete(
    value: *const RlmeshValue,
    out: *mut i64,
    cap: usize,
) -> RlmeshStatus {
    guard(|| {
        let SpaceValue::MultiDiscrete(values) = value_ref(value)? else {
            return Err(CapiError::invalid_value("value is not MultiDiscrete"));
        };
        copy_out(values, out, cap, i64::clone)
    })
}

/// Construct a `MultiBinary` value by copying `n` bytes (each normalized to 0/1).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_multi_binary(data: *const u8, n: usize) -> *mut RlmeshValue {
    guard_ptr(|| {
        let bytes = if n == 0 {
            &[][..]
        } else if data.is_null() {
            return Err(CapiError::invalid_arg("null data"));
        } else {
            unsafe { std::slice::from_raw_parts(data, n) }
        };
        let bits = bytes.iter().map(|&b| b != 0).collect();
        Ok(into_handle(SpaceValue::MultiBinary(bits)))
    })
}

/// Copy a `MultiBinary` value's bits into `out` as 0/1 bytes (capacity `cap`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_copy_multi_binary(
    value: *const RlmeshValue,
    out: *mut u8,
    cap: usize,
) -> RlmeshStatus {
    guard(|| {
        let SpaceValue::MultiBinary(bits) = value_ref(value)? else {
            return Err(CapiError::invalid_value("value is not MultiBinary"));
        };
        copy_out(bits, out, cap, |&b| u8::from(b))
    })
}

/// The child count of a `Dict`/`Tuple` value, else 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_len(value: *const RlmeshValue) -> usize {
    guard_value(0, || match unsafe { value.cast::<SpaceValue>().as_ref() } {
        Some(SpaceValue::Tuple(items)) => items.len(),
        Some(SpaceValue::Dict(map)) => map.len(),
        _ => 0,
    })
}

/// Borrow a `Tuple` child by index (valid while `value` lives), or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_tuple_get(
    value: *const RlmeshValue,
    index: usize,
) -> *const RlmeshValue {
    guard_value(std::ptr::null(), || {
        match unsafe { value.cast::<SpaceValue>().as_ref() } {
            Some(SpaceValue::Tuple(items)) => items.get(index).map_or(std::ptr::null(), |child| {
                (child as *const SpaceValue).cast()
            }),
            _ => std::ptr::null(),
        }
    })
}

/// Borrow a `Dict` child by key (valid while `value` lives), or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_dict_get(
    value: *const RlmeshValue,
    key: *const c_char,
) -> *const RlmeshValue {
    guard_value(std::ptr::null(), || {
        if key.is_null() {
            return std::ptr::null();
        }
        let Some(SpaceValue::Dict(map)) = (unsafe { value.cast::<SpaceValue>().as_ref() }) else {
            return std::ptr::null();
        };
        let Ok(key) = (unsafe { CStr::from_ptr(key) }).to_str() else {
            return std::ptr::null();
        };
        map.get(key).map_or(std::ptr::null(), |child| {
            (child as *const SpaceValue).cast()
        })
    })
}

/// Construct a `Tuple` value, taking ownership of (and freeing) each child.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_tuple(
    children: *const *mut RlmeshValue,
    n: usize,
) -> *mut RlmeshValue {
    guard_ptr(|| {
        let items = take_children(children, n)?;
        Ok(into_handle(SpaceValue::Tuple(items)))
    })
}

/// Construct a `Dict` value from parallel `keys`/`values`, taking ownership of
/// each child value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_dict(
    keys: *const *const c_char,
    values: *const *mut RlmeshValue,
    n: usize,
) -> *mut RlmeshValue {
    guard_ptr(|| {
        if n != 0 && (keys.is_null() || values.is_null()) {
            return Err(CapiError::invalid_arg("null keys or values"));
        }
        let key_slice = if n == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(keys, n) }
        };
        // Decode and validate every key BEFORE taking ownership of any child, so an
        // invalid key leaves all children owned by the caller (no double-free).
        let keys: Vec<String> = key_slice
            .iter()
            .map(|&key| {
                if key.is_null() {
                    return Err(CapiError::invalid_arg("null dict key"));
                }
                unsafe { CStr::from_ptr(key) }
                    .to_str()
                    .map(str::to_owned)
                    .map_err(|_| CapiError::invalid_value("dict key must be UTF-8"))
            })
            .collect::<Result<_, _>>()?;
        let children = take_children(values, n)?;
        let map: std::collections::BTreeMap<String, SpaceValue> =
            keys.into_iter().zip(children).collect();
        Ok(into_handle(SpaceValue::Dict(map)))
    })
}

/// Free an owned value (from a constructor or `rlmesh_decode_batch`). Do not call
/// on a borrowed child (`*_get`) or a tensor view.
///
/// # Safety
/// `value` must be NULL or a pointer this thread owns and has not freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_value_free(value: *mut RlmeshValue) {
    if !value.is_null() {
        drop(unsafe { Box::from_raw(value) });
    }
}

pub(crate) fn into_handle(value: SpaceValue) -> *mut RlmeshValue {
    Box::into_raw(Box::new(RlmeshValue(value)))
}

fn read_slice<'a>(data: *const i64, n: usize) -> Result<&'a [i64], CapiError> {
    if n == 0 {
        Ok(&[])
    } else if data.is_null() {
        Err(CapiError::invalid_arg("null data"))
    } else {
        Ok(unsafe { std::slice::from_raw_parts(data, n) })
    }
}

fn copy_out<T, U>(
    src: &[T],
    out: *mut U,
    cap: usize,
    map: impl Fn(&T) -> U,
) -> Result<(), CapiError> {
    if src.len() > cap {
        return Err(CapiError::invalid_arg("output buffer too small"));
    }
    if out.is_null() {
        return Err(CapiError::invalid_arg("null out"));
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(out, src.len()) };
    for (slot, item) in dst.iter_mut().zip(src) {
        *slot = map(item);
    }
    Ok(())
}

fn take_children(
    children: *const *mut RlmeshValue,
    n: usize,
) -> Result<Vec<SpaceValue>, CapiError> {
    if n != 0 && children.is_null() {
        return Err(CapiError::invalid_arg("null children"));
    }
    let slice = if n == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(children, n) }
    };
    // Validate every pointer BEFORE adopting any, so a NULL anywhere leaves all
    // children owned by the caller (all-or-nothing transfer; no partial free).
    if slice.iter().any(|&child| child.is_null()) {
        return Err(CapiError::invalid_arg("null child value"));
    }
    Ok(slice
        .iter()
        .map(|&child| unsafe { Box::from_raw(child) }.0)
        .collect())
}
