//! ABI 5 Value/Tensor ownership, type checks and header discriminants.
#![allow(unsafe_code)]
use super::{handle::*, tensor::*};
use crate::abi::status::{RLMeshStatus as S, rlmesh_last_error_status};
use crate::space_handle::RLMeshStr;
use rlmesh::{DType, Value};
use std::ptr::{null, null_mut};

#[test]
fn write_out_initializes_storage_without_dropping_old_bytes() {
    struct CountDrop<'a>(&'a std::cell::Cell<usize>);
    impl Drop for CountDrop<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let drops = std::cell::Cell::new(0);
    let mut output = std::mem::MaybeUninit::<CountDrop<'_>>::uninit();
    // SAFETY: output is aligned writable storage; no previous T exists there.
    assert!(unsafe { write_out(output.as_mut_ptr(), CountDrop(&drops)) }.is_ok());
    assert_eq!(drops.get(), 0);
    // SAFETY: write_out initialized the slot exactly once above.
    unsafe { output.assume_init_drop() };
    assert_eq!(drops.get(), 1);
}

fn text(s: &str) -> RLMeshStr {
    RLMeshStr::borrowed(s)
}
fn int(n: i64) -> *mut RLMeshValue {
    let mut out = null_mut();
    assert_eq!(rlmesh_value_int(n, &mut out), S::Ok);
    out
}

#[test]
fn abi5_header_kinds_and_removed_symbols() {
    let header = include_str!("../../include/rlmesh.h");
    assert!(header.contains("#define RLMESH_ABI_VERSION 5\n"));
    assert_eq!(crate::abi::rlmesh_abi_version(), 5);
    for name in [
        "ABI_VERSION_MAJOR",
        "ABI_VERSION_MINOR",
        "ABI_VERSION_PATCH",
        "RLMeshDLPackDType",
        "rlmesh_tensor_release",
        "rlmesh_value_box",
        "RLMESH_VALUE_DISCRETE",
    ] {
        assert!(!header.contains(name), "obsolete ABI 4 symbol {name}");
    }
    for (name, value) in [
        ("NONE", 1),
        ("BOOL", 2),
        ("INT", 3),
        ("FLOAT", 4),
        ("TEXT", 5),
        ("BYTES", 6),
        ("TENSOR", 7),
        ("LIST", 8),
        ("TUPLE", 9),
        ("DICT", 10),
    ] {
        assert!(header.contains(&format!("RLMESH_VALUE_{name} = {value}")));
    }
    assert_eq!(rlmesh_value_kind(null()), RLMeshValueKind::Invalid);
    for (value, kind) in [
        (Value::None, RLMeshValueKind::None),
        (Value::Bool(true), RLMeshValueKind::Bool),
        (Value::Int(1), RLMeshValueKind::Int),
        (Value::Float(1.0), RLMeshValueKind::Float),
        (Value::Text(String::new()), RLMeshValueKind::Text),
        (Value::Bytes(vec![]), RLMeshValueKind::Bytes),
        (Value::List(vec![]), RLMeshValueKind::List),
        (Value::Tuple(vec![]), RLMeshValueKind::Tuple),
        (Value::Dict(Default::default()), RLMeshValueKind::Dict),
        (
            Value::Tensor(rlmesh::Tensor::new(&[0], &[1], DType::Uint8).unwrap()),
            RLMeshValueKind::Tensor,
        ),
    ] {
        let v = into_handle(value);
        assert_eq!(rlmesh_value_kind(v), kind);
        unsafe {
            rlmesh_value_free(v);
        }
    }
}

#[test]
fn scalar_reads_are_strict_and_preserve_outputs_on_failure() {
    let mut b = null_mut();
    let mut f = null_mut();
    let mut n = null_mut();
    assert_eq!(rlmesh_value_bool(true, &mut b), S::Ok);
    assert_eq!(rlmesh_value_float(f64::NAN, &mut f), S::Ok);
    assert_eq!(rlmesh_value_none(&mut n), S::Ok);
    let i = int(1);
    let mut v = 91;
    assert_eq!(rlmesh_value_as_int(b, &mut v), S::Type);
    assert_eq!(v, 91);
    let mut f_out = 1.0;
    assert_eq!(rlmesh_value_as_float(i, &mut f_out), S::Type);
    let mut b_out = false;
    assert_eq!(rlmesh_value_as_bool(b, &mut b_out), S::Ok);
    assert!(b_out);
    assert!(!rlmesh_value_equals(b, i));
    assert!(!rlmesh_value_equals(f, f));
    assert_eq!(rlmesh_value_as_int(null(), &mut v), S::InvalidArgument);
    assert_eq!(rlmesh_value_int(1, null_mut()), S::InvalidArgument);
    unsafe {
        for v in [b, f, n, i] {
            rlmesh_value_free(v);
        }
    }
}

#[test]
fn text_bytes_and_keys_preserve_nuls_and_empty_presence() {
    let mut v = null_mut();
    let mut bytes = null_mut();
    assert_eq!(unsafe { rlmesh_value_text(text("a\0b"), &mut v) }, S::Ok);
    let copy = rlmesh_value_clone(v);
    let mut view = text("");
    let mut other = text("");
    assert_eq!(rlmesh_value_as_text(v, &mut view), S::Ok);
    assert_eq!(rlmesh_value_as_text(copy, &mut other), S::Ok);
    assert_eq!(view.data, other.data);
    assert_eq!(view.len, 3);
    assert_eq!(
        unsafe { rlmesh_value_bytes(b"a\0b".as_ptr(), 3, &mut bytes) },
        S::Ok
    );
    let mut data = null();
    let mut len = 0;
    assert_eq!(rlmesh_value_as_bytes(bytes, &mut data, &mut len), S::Ok);
    assert_eq!(unsafe { std::slice::from_raw_parts(data, len) }, b"a\0b");
    let keys = [text(""), text("a\0b")];
    let values = [v.cast_const(), bytes.cast_const()];
    let mut dict = null_mut();
    assert_eq!(
        unsafe { rlmesh_value_dict(keys.as_ptr(), values.as_ptr(), 2, &mut dict) },
        S::Ok
    );
    assert_eq!(rlmesh_value_dict_key_at(dict, 0, &mut view), S::Ok);
    assert!(!view.data.is_null());
    assert_eq!(view.len, 0);
    assert_eq!(rlmesh_value_dict_key_at(dict, 2, &mut view), S::Ok);
    assert!(view.data.is_null());
    let mut child = null();
    assert_eq!(
        unsafe { rlmesh_value_dict_get(dict, text("a\0b"), &mut child) },
        S::Ok
    );
    assert_eq!(rlmesh_value_kind(child), RLMeshValueKind::Bytes);
    let mut present = false;
    assert_eq!(
        unsafe { rlmesh_value_dict_contains_key(dict, text(""), &mut present) },
        S::Ok
    );
    assert!(present);
    let invalid = RLMeshStr {
        data: b"\xff".as_ptr().cast(),
        len: 1,
    };
    let mut out = null_mut();
    assert_eq!(
        unsafe { rlmesh_value_text(invalid, &mut out) },
        S::InvalidArgument
    );
    unsafe {
        for v in [v, copy, bytes, dict] {
            rlmesh_value_free(v);
        }
    }
}

#[test]
fn composites_share_children_and_duplicate_dict_keys_replace_in_place() {
    let a = int(1);
    let b = int(2);
    let values = [a.cast_const(), b.cast_const(), b.cast_const()];
    let keys = [text("z"), text("a"), text("z")];
    let mut dict = null_mut();
    assert_eq!(
        unsafe { rlmesh_value_dict(keys.as_ptr(), values.as_ptr(), 3, &mut dict) },
        S::Ok
    );
    let mut len = 0;
    assert_eq!(rlmesh_value_len(dict, &mut len), S::Ok);
    assert_eq!(len, 2);
    let mut key = text("");
    assert_eq!(rlmesh_value_dict_key_at(dict, 0, &mut key), S::Ok);
    assert_eq!(
        unsafe { std::slice::from_raw_parts(key.data.cast::<u8>(), key.len) },
        b"z"
    );
    let mut child = null();
    assert_eq!(rlmesh_value_dict_value_at(dict, 0, &mut child), S::Ok);
    let mut n = 0;
    assert_eq!(rlmesh_value_as_int(child, &mut n), S::Ok);
    assert_eq!(n, 2);
    let mut tuple = null_mut();
    let mut list = null_mut();
    assert_eq!(
        unsafe { rlmesh_value_tuple(values.as_ptr(), 3, &mut tuple) },
        S::Ok
    );
    assert_eq!(
        unsafe { rlmesh_value_list(values.as_ptr(), 3, &mut list) },
        S::Ok
    );
    assert!(!rlmesh_value_equals(tuple, list));
    unsafe {
        rlmesh_value_free(a);
        rlmesh_value_free(b);
    }
    assert_eq!(rlmesh_value_get(tuple, 2, &mut child), S::Ok);
    assert_eq!(rlmesh_value_as_int(child, &mut n), S::Ok);
    assert_eq!(rlmesh_value_get(tuple, 3, &mut child), S::Ok);
    assert!(child.is_null());
    assert_eq!(rlmesh_value_get(dict, 0, &mut child), S::Type);
    let retained = rlmesh_value_clone(tuple);
    unsafe {
        rlmesh_value_free(tuple);
    }
    assert_eq!(rlmesh_value_get(retained, 0, &mut child), S::Ok);
    assert_eq!(rlmesh_value_as_int(child, &mut n), S::Ok);
    assert_eq!(n, 1);
    unsafe {
        for v in [dict, list, retained] {
            rlmesh_value_free(v);
        }
    }
}

#[test]
fn failed_composites_keep_every_input_owned_and_outputs_untouched() {
    let value = int(4);
    let values = [value.cast_const(), null()];
    let mut out = value;
    assert_eq!(
        unsafe { rlmesh_value_tuple(values.as_ptr(), 2, &mut out) },
        S::InvalidArgument
    );
    assert_eq!(out, value);
    assert_eq!(
        unsafe {
            rlmesh_value_dict(
                [text("x"), text("y")].as_ptr(),
                values.as_ptr(),
                2,
                &mut out,
            )
        },
        S::InvalidArgument
    );
    assert_eq!(rlmesh_last_error_status(), S::InvalidArgument);
    let mut n = 0;
    assert_eq!(rlmesh_value_as_int(value, &mut n), S::Ok);
    assert_eq!(n, 4);
    unsafe {
        rlmesh_value_free(value);
    }
}

#[test]
fn tensor_uses_facade_validation_and_shares_only_when_requested() {
    let mut t = null_mut();
    let shape = [2, 3];
    let bytes = [0u8; 24];
    assert_eq!(
        unsafe { rlmesh_tensor_new(bytes.as_ptr().cast(), 24, shape.as_ptr(), 2, 11, &mut t) },
        S::Ok
    );
    let mut len = 0;
    assert_eq!(rlmesh_tensor_shape(t, null_mut(), 0, &mut len), S::Ok);
    assert_eq!(len, 2);
    let mut dims = [99; 2];
    assert_eq!(
        rlmesh_tensor_shape(t, dims.as_mut_ptr(), 1, &mut len),
        S::InvalidArgument
    );
    assert_eq!(dims, [99, 99]);
    assert_eq!(len, 2);
    assert_eq!(
        rlmesh_tensor_strides(t, dims.as_mut_ptr(), 2, &mut len),
        S::Ok
    );
    assert_eq!(dims, [12, 4]);
    let mut value = null_mut();
    assert_eq!(rlmesh_value_tensor(t, &mut value), S::Ok);
    let clone = rlmesh_tensor_clone(t);
    let mut copy = null_mut();
    let mut flat = null_mut();
    assert_eq!(rlmesh_tensor_copy(t, &mut copy), S::Ok);
    assert_eq!(
        unsafe { rlmesh_tensor_reshape(t, [-1i64].as_ptr(), 1, &mut flat) },
        S::Ok
    );
    let (mut data, mut clone_data, mut copy_data) = (null(), null(), null());
    assert_eq!(rlmesh_tensor_buffer(t, &mut data, &mut len), S::Ok);
    assert_eq!(len, 24);
    assert_eq!(
        rlmesh_tensor_buffer(clone, &mut clone_data, &mut len),
        S::Ok
    );
    assert_eq!(rlmesh_tensor_buffer(copy, &mut copy_data, &mut len), S::Ok);
    assert_eq!(data, clone_data);
    assert_ne!(data, copy_data);
    assert_eq!(rlmesh_tensor_size(t, &mut len), S::Ok);
    assert_eq!(len, 6);
    assert_eq!(rlmesh_tensor_dtype(t), 11);
    assert_eq!(rlmesh_tensor_device(t), 1);
    assert!(rlmesh_tensor_is_contiguous(t));
    unsafe {
        for t in [t, clone, copy, flat] {
            rlmesh_tensor_free(t);
        }
    }
    let mut from_value = null_mut();
    assert_eq!(rlmesh_value_as_tensor(value, &mut from_value), S::Ok);
    unsafe {
        rlmesh_value_free(value);
    }
    assert_eq!(rlmesh_tensor_nbytes(from_value, &mut len), S::Ok);
    assert_eq!(len, 24);
    unsafe {
        rlmesh_tensor_free(from_value);
    }
    for dtype in [0, 1000, 999] {
        assert_eq!(
            unsafe {
                rlmesh_tensor_new(bytes.as_ptr().cast(), 24, shape.as_ptr(), 2, dtype, &mut t)
            },
            S::InvalidArgument
        );
    }
    assert_eq!(
        unsafe { rlmesh_tensor_new(bytes.as_ptr().cast(), 23, shape.as_ptr(), 2, 11, &mut t) },
        S::InvalidArgument
    );
    assert_eq!(
        unsafe { rlmesh_tensor_new(null(), 0, [0usize].as_ptr(), 1, 11, &mut t) },
        S::Ok
    );
    unsafe {
        rlmesh_tensor_free(t);
    }
}
