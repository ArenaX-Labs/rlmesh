//! Round-trip tests exercising the FFI value bridge + codec against the real
//! core, without a live server.
#![allow(unsafe_code)] // exercising the raw FFI surface directly.

use std::ffi::c_void;

use rlmesh_spaces::spaces::{BoxSpaceBuilder, DiscreteBuilder, TextBuilder};
use rlmesh_spaces::{DType, SpaceSpec};

use crate::abi::status::RlmeshStatus;
use crate::codec::{
    RlmeshBytes, rlmesh_bytes_free, rlmesh_decode_batch, rlmesh_encode_batch, rlmesh_values_free,
};
use crate::spaces::RlmeshSpaceSpec;
use crate::value::bridge::{
    RlmeshValue, rlmesh_value_as_discrete, rlmesh_value_as_tensor, rlmesh_value_as_text,
    rlmesh_value_box, rlmesh_value_discrete, rlmesh_value_free, rlmesh_value_text,
};
use crate::value::dtype::RlmeshDType;
use crate::value::tensor::RlmeshTensor;

/// Encode one value, decode it back, and run `check` on the single decoded value.
fn round_trip(spec: &SpaceSpec, value: *mut RlmeshValue, check: impl FnOnce(*const RlmeshValue)) {
    let spec_ptr = std::ptr::from_ref(spec).cast::<RlmeshSpaceSpec>();
    let values: [*const RlmeshValue; 1] = [value];
    let mut encoded = RlmeshBytes::empty();
    assert_eq!(
        unsafe { rlmesh_encode_batch(values.as_ptr(), 1, spec_ptr, &mut encoded) },
        RlmeshStatus::Ok
    );
    let mut decoded: *mut *mut RlmeshValue = std::ptr::null_mut();
    let mut count = 0usize;
    assert_eq!(
        unsafe {
            rlmesh_decode_batch(
                encoded.data,
                encoded.len,
                spec_ptr,
                &mut decoded,
                &mut count,
            )
        },
        RlmeshStatus::Ok
    );
    assert_eq!(count, 1);
    check(unsafe { *decoded });
    unsafe {
        rlmesh_values_free(decoded, count);
        rlmesh_bytes_free(encoded);
        rlmesh_value_free(value);
    }
}

const F32: RlmeshDType = RlmeshDType {
    code: 2,
    bits: 32,
    lanes: 1,
};

fn box_spec() -> SpaceSpec {
    BoxSpaceBuilder::unbounded(vec![2, 2])
        .dtype(DType::Float32)
        .build()
        .expect("valid box spec")
}

fn tensor_view(data: &[f32], shape: &[i64]) -> RlmeshTensor {
    RlmeshTensor {
        data: data.as_ptr() as *mut c_void,
        ndim: shape.len() as i32,
        shape: shape.as_ptr(),
        strides: std::ptr::null(),
        dtype: F32,
        device_type: 1,
        device_id: 0,
        flags: 0,
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
    }
}

#[test]
fn box_value_round_trips_through_the_codec() {
    let spec = box_spec();
    let spec_ptr = std::ptr::from_ref(&spec).cast::<RlmeshSpaceSpec>();

    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let shape: [i64; 2] = [2, 2];
    let view = tensor_view(&data, &shape);

    let value = unsafe { rlmesh_value_box(&view) };
    assert!(!value.is_null(), "rlmesh_value_box returned null");

    // Encode the action value to wire bytes.
    let values: [*const RlmeshValue; 1] = [value];
    let mut encoded = RlmeshBytes::empty();
    let status = unsafe { rlmesh_encode_batch(values.as_ptr(), 1, spec_ptr, &mut encoded) };
    assert_eq!(status, RlmeshStatus::Ok);
    assert!(encoded.len > 0);

    // Decode them back (one sub-env).
    let mut decoded: *mut *mut RlmeshValue = std::ptr::null_mut();
    let mut count = 0usize;
    let status = unsafe {
        rlmesh_decode_batch(
            encoded.data,
            encoded.len,
            spec_ptr,
            &mut decoded,
            &mut count,
        )
    };
    assert_eq!(status, RlmeshStatus::Ok);
    assert_eq!(count, 1);

    // Read the tensor back and confirm the bytes survived the trip.
    let first = unsafe { *decoded };
    let mut out = RlmeshTensor {
        data: std::ptr::null_mut(),
        ndim: 0,
        shape: std::ptr::null(),
        strides: std::ptr::null(),
        dtype: F32,
        device_type: 0,
        device_id: 0,
        flags: 0,
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
    };
    let status = unsafe { rlmesh_value_as_tensor(first, &mut out) };
    assert_eq!(status, RlmeshStatus::Ok);
    assert_eq!(out.ndim, 2);
    let recovered = unsafe { std::slice::from_raw_parts(out.data.cast::<f32>(), 4) };
    assert_eq!(recovered, &data);

    unsafe {
        rlmesh_values_free(decoded, count);
        rlmesh_bytes_free(encoded);
        rlmesh_value_free(value);
    }
}

#[test]
fn discrete_value_round_trips() {
    let spec = DiscreteBuilder::new(8)
        .build()
        .expect("valid discrete spec");
    let value = rlmesh_value_discrete(5);
    round_trip(&spec, value, |decoded| {
        let mut out = 0i64;
        assert_eq!(
            unsafe { rlmesh_value_as_discrete(decoded, &mut out) },
            RlmeshStatus::Ok
        );
        assert_eq!(out, 5);
    });
}

#[test]
fn header_abi_version_macros_match_crate() {
    let header = include_str!("../include/rlmesh.h");
    let macro_value = |name: &str| -> String {
        let prefix = format!("#define {name} ");
        header
            .lines()
            .find_map(|line| line.strip_prefix(prefix.as_str()))
            .map(|rest| rest.trim().to_string())
            .expect("version macro present in header")
    };
    assert_eq!(
        macro_value("RLMESH_ABI_VERSION_MAJOR"),
        env!("CARGO_PKG_VERSION_MAJOR")
    );
    assert_eq!(
        macro_value("RLMESH_ABI_VERSION_MINOR"),
        env!("CARGO_PKG_VERSION_MINOR")
    );
    assert_eq!(
        macro_value("RLMESH_ABI_VERSION_PATCH"),
        env!("CARGO_PKG_VERSION_PATCH")
    );
}

#[test]
fn header_dtype_macros_match_core() {
    // The header's `RLMESH_<NAME>` dtype macros are hand-authored; assert each
    // `RLMESH_DTYPE_INIT(code, bits, lanes)` triple still matches what core's
    // `RlmeshDType::from_core` produces, so the C constants can't silently drift.
    let header = include_str!("../include/rlmesh.h");
    let triple = |name: &str| -> RlmeshDType {
        let prefix = format!("#define {name} RLMESH_DTYPE_INIT(");
        let rest = header
            .lines()
            .find_map(|line| line.trim().strip_prefix(prefix.as_str()))
            .unwrap_or_else(|| panic!("dtype macro {name} present in header"));
        let inner = rest.split(')').next().expect("closing paren");
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        assert_eq!(
            parts.len(),
            3,
            "{name} expects a (code, bits, lanes) triple"
        );
        RlmeshDType {
            code: parts[0].parse().expect("code"),
            bits: parts[1].parse().expect("bits"),
            lanes: parts[2].parse().expect("lanes"),
        }
    };
    for (name, dtype) in [
        ("RLMESH_F32", DType::Float32),
        ("RLMESH_F64", DType::Float64),
        ("RLMESH_I32", DType::Int32),
        ("RLMESH_I64", DType::Int64),
        ("RLMESH_U8", DType::Uint8),
        ("RLMESH_BOOL", DType::Bool),
    ] {
        assert_eq!(
            Some(triple(name)),
            RlmeshDType::from_core(dtype),
            "header dtype macro {name} drifted from core"
        );
    }
}

#[test]
fn text_value_round_trips() {
    let spec = TextBuilder::new(32).build().expect("valid text spec");
    let text = "pick up the cup";
    let value = unsafe { rlmesh_value_text(text.as_ptr().cast(), text.len()) };
    assert!(!value.is_null());
    round_trip(&spec, value, |decoded| {
        let mut ptr: *const std::ffi::c_char = std::ptr::null();
        let mut len = 0usize;
        assert_eq!(
            unsafe { rlmesh_value_as_text(decoded, &mut ptr, &mut len) },
            RlmeshStatus::Ok
        );
        let recovered = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) };
        assert_eq!(recovered, text.as_bytes());
    });
}
