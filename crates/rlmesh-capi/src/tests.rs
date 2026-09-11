//! Tests exercising the FFI value handle + header constants against the real
//! core, without a live server.
#![allow(unsafe_code)] // exercising the raw FFI surface directly.

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use rlmesh_spaces::spaces::{BoxSpaceBuilder, DiscreteBuilder, TextBuilder};
use rlmesh_spaces::{
    BoxBounds, BoxSpec, DType, DictSpec, DiscreteSpec, ElementwiseBounds, EnvContract,
    MultiDiscreteSpec, SpaceKind, SpaceSpec, TextSpec, TupleSpec, UniformBounds,
};

use crate::abi::status::{RlmeshStatus, rlmesh_last_error_is_recoverable};
use crate::spaces::{
    RlmeshContract, RlmeshSpaceSpec, rlmesh_contract_num_envs, rlmesh_contract_observation_space,
    rlmesh_space_box_bounds, rlmesh_space_copy_nvec, rlmesh_space_copy_shape,
    rlmesh_space_dict_get, rlmesh_space_dict_get_at, rlmesh_space_dict_key,
    rlmesh_space_discrete_n, rlmesh_space_len, rlmesh_space_text_charset, rlmesh_space_text_length,
    rlmesh_space_tuple_get, rlmesh_space_type,
};
use crate::value::dtype::RlmeshDType;
use crate::value::handle::{
    RlmeshValue, RlmeshValueKind, rlmesh_value_array_len, rlmesh_value_as_discrete,
    rlmesh_value_as_tensor, rlmesh_value_as_text, rlmesh_value_box, rlmesh_value_copy_multi_binary,
    rlmesh_value_copy_multi_discrete, rlmesh_value_dict, rlmesh_value_dict_get,
    rlmesh_value_dict_get_at, rlmesh_value_dict_key, rlmesh_value_discrete, rlmesh_value_free,
    rlmesh_value_kind, rlmesh_value_len, rlmesh_value_multi_binary, rlmesh_value_multi_discrete,
    rlmesh_value_text, rlmesh_value_tuple, rlmesh_value_tuple_get,
};
use crate::value::tensor::{RlmeshTensor, rlmesh_tensor_release};

/// Run `check` on a freshly built value, then free it.
fn round_trip(_spec: &SpaceSpec, value: *mut RlmeshValue, check: impl FnOnce(*const RlmeshValue)) {
    check(value);
    unsafe { rlmesh_value_free(value) };
}

const F32: RlmeshDType = RlmeshDType {
    code: 2,
    bits: 32,
    lanes: 1,
};

const HEADER: &str = include_str!("../include/rlmesh.h");

fn box_spec() -> SpaceSpec {
    BoxSpaceBuilder::unbounded(vec![2, 2])
        .dtype(DType::Float32)
        .build()
        .expect("valid box spec")
}

fn tensor_view(data: &[f32], shape: &[i64]) -> RlmeshTensor {
    RlmeshTensor {
        data: data.as_ptr().cast::<c_void>(),
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
fn box_value_round_trips() {
    let spec = box_spec();
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let shape: [i64; 2] = [2, 2];
    let view = tensor_view(&data, &shape);
    let value = unsafe { rlmesh_value_box(&view) };
    assert!(!value.is_null(), "rlmesh_value_box returned null");
    round_trip(&spec, value, |decoded| {
        let mut out = RlmeshTensor {
            data: std::ptr::null(),
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
        assert_eq!(
            unsafe { rlmesh_value_as_tensor(decoded, &mut out) },
            RlmeshStatus::Ok
        );
        assert_eq!(out.ndim, 2);
        let recovered = unsafe { std::slice::from_raw_parts(out.data.cast::<f32>(), 4) };
        assert_eq!(recovered, &data);
    });
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
    let macro_value = |name: &str| -> String {
        let prefix = format!("#define {name} ");
        HEADER
            .lines()
            .find_map(|line| line.strip_prefix(prefix.as_str()))
            .map(|rest| rest.trim().to_string())
            .expect("version macro present in header")
    };
    // The binary ABI generation: header macro, exported fn, and crate const must
    // all agree (this is the gate consumers compile against).
    assert_eq!(
        macro_value("RLMESH_ABI_VERSION"),
        crate::abi::RLMESH_ABI_VERSION.to_string()
    );
    assert_eq!(
        crate::abi::rlmesh_abi_version(),
        crate::abi::RLMESH_ABI_VERSION
    );
    // Package semver macros stay informational but honest vs the crate version.
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
    let triple = |name: &str| -> RlmeshDType {
        let prefix = format!("#define {name} RLMESH_DTYPE_INIT(");
        let rest = HEADER
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

#[test]
fn dict_get_rejects_null_key() {
    let key = c"only";
    let keys: [*const c_char; 1] = [key.as_ptr()];
    let values: [*mut RlmeshValue; 1] = [rlmesh_value_discrete(7)];
    let dict = unsafe { rlmesh_value_dict(keys.as_ptr(), values.as_ptr(), 1) };
    assert!(!dict.is_null());
    // A NULL key must return NULL, never dereference it.
    assert!(unsafe { rlmesh_value_dict_get(dict, std::ptr::null()) }.is_null());
    assert!(!unsafe { rlmesh_value_dict_get(dict, key.as_ptr()) }.is_null());
    unsafe { rlmesh_value_free(dict) };
}

#[test]
fn dict_with_a_null_child_takes_no_ownership() {
    // `[valid, NULL]` must fail without adopting `keep`, so the caller still owns
    // it — freeing it here is a single valid free (the pre-fix code adopted then
    // freed children on the error path, making this a double free).
    let keep = rlmesh_value_discrete(1);
    let keys: [*const c_char; 2] = [c"a".as_ptr(), c"b".as_ptr()];
    let values: [*mut RlmeshValue; 2] = [keep, std::ptr::null_mut()];
    let dict = unsafe { rlmesh_value_dict(keys.as_ptr(), values.as_ptr(), 2) };
    assert!(dict.is_null());
    unsafe { rlmesh_value_free(keep) };
}

#[test]
fn box_accepts_scalar_with_null_shape() {
    // A scalar Box (ndim == 0) may carry shape == NULL; constructing it must not
    // form a slice from the null pointer.
    let scalar: f32 = 4.0;
    let view = RlmeshTensor {
        data: std::ptr::from_ref(&scalar).cast::<c_void>(),
        ndim: 0,
        shape: std::ptr::null(),
        strides: std::ptr::null(),
        dtype: F32,
        device_type: 1,
        device_id: 0,
        flags: 0,
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
    };
    let value = unsafe { rlmesh_value_box(&view) };
    assert!(!value.is_null(), "scalar box with null shape must succeed");
    unsafe { rlmesh_value_free(value) };
}

#[test]
fn space_copy_shape_rejects_null_out() {
    let spec = box_spec();
    let spec_ptr = std::ptr::from_ref(&spec).cast::<RlmeshSpaceSpec>();
    // A NULL out with ample capacity for a non-empty shape must error, not deref.
    assert_eq!(
        unsafe { rlmesh_space_copy_shape(spec_ptr, std::ptr::null_mut(), 8) },
        RlmeshStatus::InvalidArgument
    );
}

// ---- kind / length accessors -------------------------------------------

#[test]
fn kind_reports_every_variant_and_invalid_for_null() {
    let cases: [(*mut RlmeshValue, RlmeshValueKind); 5] = [
        (rlmesh_value_discrete(1), RlmeshValueKind::Discrete),
        (
            unsafe { rlmesh_value_multi_binary([1u8, 0].as_ptr(), 2) },
            RlmeshValueKind::MultiBinary,
        ),
        (
            unsafe { rlmesh_value_multi_discrete([3i64, 4].as_ptr(), 2) },
            RlmeshValueKind::MultiDiscrete,
        ),
        (
            unsafe { rlmesh_value_text(c"hi".as_ptr(), 2) },
            RlmeshValueKind::Text,
        ),
        (
            unsafe { rlmesh_value_tuple([rlmesh_value_discrete(0)].as_ptr(), 1) },
            RlmeshValueKind::Tuple,
        ),
    ];
    for (value, kind) in cases {
        assert!(!value.is_null());
        assert_eq!(unsafe { rlmesh_value_kind(value) }, kind);
        unsafe { rlmesh_value_free(value) };
    }
    // A NULL handle has no kind — it must not masquerade as Tuple.
    assert_eq!(
        unsafe { rlmesh_value_kind(std::ptr::null()) },
        RlmeshValueKind::Invalid
    );
}

#[test]
fn len_separates_empty_from_wrong_kind_from_null() {
    let empty = unsafe { rlmesh_value_tuple(std::ptr::null(), 0) };
    let mut len = 9usize;
    assert_eq!(
        unsafe { rlmesh_value_len(empty, &mut len) },
        RlmeshStatus::Ok
    );
    assert_eq!(
        len, 0,
        "an empty Tuple has length 0, and that is not an error"
    );
    // Wrong kind, NULL value and NULL out are each distinguishable.
    let scalar = rlmesh_value_discrete(1);
    assert_eq!(
        unsafe { rlmesh_value_len(scalar, &mut len) },
        RlmeshStatus::InvalidValue
    );
    assert_eq!(
        unsafe { rlmesh_value_len(std::ptr::null(), &mut len) },
        RlmeshStatus::InvalidArgument
    );
    assert_eq!(
        unsafe { rlmesh_value_len(empty, std::ptr::null_mut()) },
        RlmeshStatus::InvalidArgument
    );
    unsafe { rlmesh_value_free(scalar) };
    unsafe { rlmesh_value_free(empty) };
}

#[test]
fn array_len_reports_length_and_rejects_other_kinds() {
    let bits = unsafe { rlmesh_value_multi_binary([1u8, 0, 1].as_ptr(), 3) };
    let mut len = 0usize;
    assert_eq!(
        unsafe { rlmesh_value_array_len(bits, &mut len) },
        RlmeshStatus::Ok
    );
    assert_eq!(len, 3);
    let scalar = rlmesh_value_discrete(1);
    assert_eq!(
        unsafe { rlmesh_value_array_len(scalar, &mut len) },
        RlmeshStatus::InvalidValue
    );
    unsafe { rlmesh_value_free(scalar) };
    unsafe { rlmesh_value_free(bits) };
}

// ---- array construct -> copy round trips -------------------------------

#[test]
fn multi_binary_round_trips_normalizing_to_bits() {
    let value = unsafe { rlmesh_value_multi_binary([0u8, 7, 0, 1].as_ptr(), 4) };
    let mut out = [9u8; 4];
    assert_eq!(
        unsafe { rlmesh_value_copy_multi_binary(value, out.as_mut_ptr(), out.len()) },
        RlmeshStatus::Ok
    );
    assert_eq!(out, [0, 1, 0, 1], "any nonzero byte becomes a set bit");
    // A buffer too small is rejected before any write.
    assert_eq!(
        unsafe { rlmesh_value_copy_multi_binary(value, out.as_mut_ptr(), 3) },
        RlmeshStatus::InvalidArgument
    );
    unsafe { rlmesh_value_free(value) };
}

#[test]
fn multi_discrete_round_trips() {
    let value = unsafe { rlmesh_value_multi_discrete([2i64, 0, 5].as_ptr(), 3) };
    let mut out = [0i64; 3];
    assert_eq!(
        unsafe { rlmesh_value_copy_multi_discrete(value, out.as_mut_ptr(), out.len()) },
        RlmeshStatus::Ok
    );
    assert_eq!(out, [2, 0, 5]);
    unsafe { rlmesh_value_free(value) };
}

#[test]
fn copy_out_accepts_a_null_buffer_for_an_empty_value() {
    // Same rule as rlmesh_space_copy_shape: nothing to write, nothing to reject.
    let empty = unsafe { rlmesh_value_multi_discrete(std::ptr::null(), 0) };
    assert_eq!(
        unsafe { rlmesh_value_copy_multi_discrete(empty, std::ptr::null_mut(), 0) },
        RlmeshStatus::Ok
    );
    unsafe { rlmesh_value_free(empty) };
    let bits = unsafe { rlmesh_value_multi_binary([1u8].as_ptr(), 1) };
    assert_eq!(
        unsafe { rlmesh_value_copy_multi_binary(bits, std::ptr::null_mut(), 4) },
        RlmeshStatus::InvalidArgument
    );
    unsafe { rlmesh_value_free(bits) };
}

#[test]
fn space_copy_shape_accepts_a_null_buffer_for_a_scalar_space() {
    let spec = SpaceSpec {
        shape: vec![],
        dtype: DType::Float32,
        spec: Some(SpaceKind::Box(BoxSpec { bounds: None })),
    };
    assert_eq!(
        unsafe { rlmesh_space_copy_shape(spec_ptr(&spec), std::ptr::null_mut(), 0) },
        RlmeshStatus::Ok
    );
}

// ---- composites --------------------------------------------------------

#[test]
fn tuple_round_trips_and_borrows_children_by_index() {
    let children: [*mut RlmeshValue; 2] = [rlmesh_value_discrete(3), unsafe {
        rlmesh_value_text(c"go".as_ptr(), 2)
    }];
    let tuple = unsafe { rlmesh_value_tuple(children.as_ptr(), 2) };
    assert!(!tuple.is_null());
    let mut len = 0usize;
    assert_eq!(
        unsafe { rlmesh_value_len(tuple, &mut len) },
        RlmeshStatus::Ok
    );
    assert_eq!(len, 2);
    let mut n = 0i64;
    assert_eq!(
        unsafe { rlmesh_value_as_discrete(rlmesh_value_tuple_get(tuple, 0), &mut n) },
        RlmeshStatus::Ok
    );
    assert_eq!(n, 3);
    assert_eq!(
        unsafe { rlmesh_value_kind(rlmesh_value_tuple_get(tuple, 1)) },
        RlmeshValueKind::Text
    );
    // Out of range and wrong kind both read as "no such child".
    assert!(unsafe { rlmesh_value_tuple_get(tuple, 2) }.is_null());
    unsafe { rlmesh_value_free(tuple) };
}

#[test]
fn dict_keys_are_discoverable_by_index() {
    let keys: [*const c_char; 2] = [c"pos".as_ptr(), c"grip".as_ptr()];
    let values: [*mut RlmeshValue; 2] = [rlmesh_value_discrete(1), rlmesh_value_discrete(2)];
    let dict = unsafe { rlmesh_value_dict(keys.as_ptr(), values.as_ptr(), 2) };
    assert!(!dict.is_null());
    // Keys come back in sorted order, parallel to nothing but themselves.
    assert_eq!(read_key(dict, 0), "grip");
    assert_eq!(read_key(dict, 1), "pos");
    let mut ptr: *const c_char = std::ptr::null();
    let mut len = 0usize;
    assert_eq!(
        unsafe { rlmesh_value_dict_key(dict, 2, &mut ptr, &mut len) },
        RlmeshStatus::InvalidArgument
    );
    // The children walk the SAME order, so key(i) names get_at(i) with no
    // NUL-terminated copy of the key in between.
    assert_eq!(discrete_at(unsafe { rlmesh_value_dict_get_at(dict, 0) }), 2);
    assert_eq!(discrete_at(unsafe { rlmesh_value_dict_get_at(dict, 1) }), 1);
    assert!(unsafe { rlmesh_value_dict_get_at(dict, 2) }.is_null());
    // Wrong kind and NULL both read as "no such child".
    let tuple_children: [*mut RlmeshValue; 1] = [rlmesh_value_discrete(1)];
    let tuple = unsafe { rlmesh_value_tuple(tuple_children.as_ptr(), 1) };
    assert!(unsafe { rlmesh_value_dict_get_at(tuple, 0) }.is_null());
    assert!(unsafe { rlmesh_value_dict_get_at(std::ptr::null(), 0) }.is_null());
    unsafe { rlmesh_value_free(tuple) };
    unsafe { rlmesh_value_free(dict) };
}

#[test]
fn dict_rejects_a_duplicate_key_without_taking_ownership() {
    // Last-wins would silently drop the first child; instead the call fails and
    // both children are still the caller's to free (a single valid free each).
    let first = rlmesh_value_discrete(1);
    let second = rlmesh_value_discrete(2);
    let keys: [*const c_char; 2] = [c"a".as_ptr(), c"a".as_ptr()];
    let values: [*mut RlmeshValue; 2] = [first, second];
    assert!(unsafe { rlmesh_value_dict(keys.as_ptr(), values.as_ptr(), 2) }.is_null());
    assert!(!crate::abi::status::rlmesh_last_error_message().is_null());
    unsafe { rlmesh_value_free(first) };
    unsafe { rlmesh_value_free(second) };
}

#[test]
fn nested_dict_of_box_and_tuple_round_trips() {
    let data: [f32; 2] = [1.0, 2.0];
    let shape: [i64; 1] = [2];
    let view = tensor_view(&data, &shape);
    let inner: [*mut RlmeshValue; 2] = [rlmesh_value_discrete(4), unsafe {
        rlmesh_value_text(c"lift".as_ptr(), 4)
    }];
    let children: [*mut RlmeshValue; 2] = [unsafe { rlmesh_value_box(&view) }, unsafe {
        rlmesh_value_tuple(inner.as_ptr(), 2)
    }];
    let keys: [*const c_char; 2] = [c"pos".as_ptr(), c"extra".as_ptr()];
    let dict = unsafe { rlmesh_value_dict(keys.as_ptr(), children.as_ptr(), 2) };
    assert!(!dict.is_null());

    let mut len = 0usize;
    assert_eq!(
        unsafe { rlmesh_value_len(dict, &mut len) },
        RlmeshStatus::Ok
    );
    assert_eq!(len, 2);

    let pos = unsafe { rlmesh_value_dict_get(dict, c"pos".as_ptr()) };
    let mut tensor = empty_tensor();
    assert_eq!(
        unsafe { rlmesh_value_as_tensor(pos, &mut tensor) },
        RlmeshStatus::Ok
    );
    assert_eq!(
        unsafe { std::slice::from_raw_parts(tensor.data.cast::<f32>(), 2) },
        &data
    );

    let extra = unsafe { rlmesh_value_dict_get(dict, c"extra".as_ptr()) };
    assert_eq!(unsafe { rlmesh_value_kind(extra) }, RlmeshValueKind::Tuple);
    let mut n = 0i64;
    assert_eq!(
        unsafe { rlmesh_value_as_discrete(rlmesh_value_tuple_get(extra, 0), &mut n) },
        RlmeshStatus::Ok
    );
    assert_eq!(n, 4);
    let mut text: *const c_char = std::ptr::null();
    let mut text_len = 0usize;
    assert_eq!(
        unsafe { rlmesh_value_as_text(rlmesh_value_tuple_get(extra, 1), &mut text, &mut text_len) },
        RlmeshStatus::Ok
    );
    assert_eq!(
        unsafe { std::slice::from_raw_parts(text.cast::<u8>(), text_len) },
        b"lift"
    );
    unsafe { rlmesh_value_free(dict) };
}

// ---- tensor release ----------------------------------------------------

static DELETER_CALLS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn counting_deleter(_tensor: *mut RlmeshTensor) {
    DELETER_CALLS.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn tensor_release_runs_the_deleter_exactly_once() {
    let mut tensor = empty_tensor();
    tensor.deleter = Some(counting_deleter);
    unsafe { rlmesh_tensor_release(&mut tensor) };
    assert_eq!(DELETER_CALLS.load(Ordering::SeqCst), 1);
    // A borrowed view (no deleter) and NULL are both no-ops.
    let mut borrowed = empty_tensor();
    unsafe { rlmesh_tensor_release(&mut borrowed) };
    unsafe { rlmesh_tensor_release(std::ptr::null_mut()) };
    assert_eq!(DELETER_CALLS.load(Ordering::SeqCst), 1);
}

// ---- contract + error + version accessors ------------------------------

#[test]
fn contract_exposes_num_envs_and_the_observation_space() {
    let contract = RlmeshContract(EnvContract {
        num_envs: 4,
        observation_space: Some(box_spec()),
        ..Default::default()
    });
    let ptr = std::ptr::from_ref(&contract);
    assert_eq!(unsafe { rlmesh_contract_num_envs(ptr) }, 4);
    let obs = unsafe { rlmesh_contract_observation_space(ptr) };
    assert_eq!(unsafe { rlmesh_space_type(obs) }, RlmeshValueKind::Box);
    // A NULL contract yields the empty answers, never a dereference.
    assert_eq!(unsafe { rlmesh_contract_num_envs(std::ptr::null()) }, 0);
    assert!(unsafe { rlmesh_contract_observation_space(std::ptr::null()) }.is_null());
}

#[test]
fn last_error_reports_an_unrecoverable_capi_failure() {
    let text = unsafe { rlmesh_value_text(c"x".as_ptr(), 1) };
    let mut out = 0i64;
    assert_eq!(
        unsafe { rlmesh_value_as_discrete(text, &mut out) },
        RlmeshStatus::InvalidValue
    );
    assert!(!crate::abi::status::rlmesh_last_error_message().is_null());
    // A capi-side argument error is never flagged recoverable (that is reserved
    // for runtime errors that carry the flag).
    assert_eq!(rlmesh_last_error_is_recoverable(), 0);
    unsafe { rlmesh_value_free(text) };
}

#[test]
fn package_version_accessors_match_the_crate() {
    let parse = |value: &str| value.parse::<u32>().expect("numeric version component");
    assert_eq!(
        crate::abi::rlmesh_abi_version_major(),
        parse(env!("CARGO_PKG_VERSION_MAJOR"))
    );
    assert_eq!(
        crate::abi::rlmesh_abi_version_minor(),
        parse(env!("CARGO_PKG_VERSION_MINOR"))
    );
    assert_eq!(
        crate::abi::rlmesh_abi_version_patch(),
        parse(env!("CARGO_PKG_VERSION_PATCH"))
    );
}

// ---- space introspection ------------------------------------------------

#[test]
fn space_type_is_invalid_for_null_and_unspecified() {
    assert_eq!(
        unsafe { rlmesh_space_type(std::ptr::null()) },
        RlmeshValueKind::Invalid
    );
    let unspecified = SpaceSpec::default();
    assert_eq!(
        unsafe { rlmesh_space_type(spec_ptr(&unspecified)) },
        RlmeshValueKind::Invalid
    );
}

#[test]
fn box_bounds_broadcast_uniform_and_report_per_element() {
    let uniform = SpaceSpec {
        shape: vec![2],
        dtype: DType::Float32,
        spec: Some(SpaceKind::Box(BoxSpec {
            bounds: Some(BoxBounds::Uniform(UniformBounds {
                low: -1.0,
                high: 1.0,
            })),
        })),
    };
    assert_eq!(bounds(&uniform, 0), (-1.0, 1.0));
    assert_eq!(bounds(&uniform, 1), (-1.0, 1.0));
    let mut low = 0.0;
    let mut high = 0.0;
    assert_eq!(
        unsafe { rlmesh_space_box_bounds(spec_ptr(&uniform), 2, &mut low, &mut high) },
        RlmeshStatus::InvalidArgument
    );

    let elementwise = SpaceSpec {
        shape: vec![2],
        dtype: DType::Float32,
        spec: Some(SpaceKind::Box(BoxSpec {
            bounds: Some(BoxBounds::Elementwise(ElementwiseBounds {
                low: vec![0.0, -5.0],
                high: vec![1.0, 5.0],
            })),
        })),
    };
    assert_eq!(bounds(&elementwise, 1), (-5.0, 5.0));

    // An unbounded Box reads as the full range, not as an error.
    assert_eq!(bounds(&box_spec(), 0), (f64::NEG_INFINITY, f64::INFINITY));
    // The wrong kind is rejected.
    let discrete = discrete_spec();
    assert_eq!(
        unsafe { rlmesh_space_box_bounds(spec_ptr(&discrete), 0, &mut low, &mut high) },
        RlmeshStatus::InvalidValue
    );
}

#[test]
fn discrete_text_and_nvec_expose_what_an_action_needs() {
    let discrete = discrete_spec();
    let mut n = 0i64;
    let mut start = 0i64;
    assert_eq!(
        unsafe { rlmesh_space_discrete_n(spec_ptr(&discrete), &mut n, &mut start) },
        RlmeshStatus::Ok
    );
    assert_eq!((n, start), (8, 2));
    // Either out-param may be skipped.
    assert_eq!(
        unsafe { rlmesh_space_discrete_n(spec_ptr(&discrete), &mut n, std::ptr::null_mut()) },
        RlmeshStatus::Ok
    );

    let text = SpaceSpec {
        shape: vec![],
        dtype: DType::Unspecified,
        spec: Some(SpaceKind::Text(TextSpec {
            min_length: 1,
            max_length: 16,
            charset: String::new(),
        })),
    };
    let (mut min, mut max) = (0i64, 0i64);
    assert_eq!(
        unsafe { rlmesh_space_text_length(spec_ptr(&text), &mut min, &mut max) },
        RlmeshStatus::Ok
    );
    assert_eq!((min, max), (1, 16));
    // An empty charset means "any character", and reads back as 0 bytes rather
    // than an error.
    assert_eq!(read_charset(spec_ptr(&text)), "");
    let pinned = TextBuilder::new(4)
        .charset("ab")
        .build()
        .expect("valid text spec");
    assert_eq!(read_charset(spec_ptr(&pinned)), "ab");
    let mut charset_ptr: *const c_char = std::ptr::null();
    let mut charset_len = 0usize;
    assert_eq!(
        unsafe {
            rlmesh_space_text_charset(spec_ptr(&discrete), &mut charset_ptr, &mut charset_len)
        },
        RlmeshStatus::InvalidValue
    );

    let multi = SpaceSpec {
        shape: vec![3],
        dtype: DType::Int64,
        spec: Some(SpaceKind::MultiDiscrete(MultiDiscreteSpec {
            nvec: vec![2, 3, 4],
        })),
    };
    let mut nvec = [0i64; 3];
    assert_eq!(
        unsafe { rlmesh_space_copy_nvec(spec_ptr(&multi), nvec.as_mut_ptr(), nvec.len()) },
        RlmeshStatus::Ok
    );
    assert_eq!(nvec, [2, 3, 4]);
    assert_eq!(
        unsafe { rlmesh_space_copy_nvec(spec_ptr(&multi), nvec.as_mut_ptr(), 2) },
        RlmeshStatus::InvalidArgument
    );
    assert_eq!(
        unsafe { rlmesh_space_copy_nvec(spec_ptr(&text), nvec.as_mut_ptr(), 3) },
        RlmeshStatus::InvalidValue
    );
}

#[test]
fn composite_spaces_are_walkable_from_c() {
    // Dict{ pos: Box, extra: Tuple(Discrete, Text) } — the walk a C model does to
    // build an action for a composite space.
    let tuple = SpaceSpec {
        shape: vec![],
        dtype: DType::Unspecified,
        spec: Some(SpaceKind::Tuple(TupleSpec {
            spaces: vec![
                discrete_spec(),
                SpaceSpec {
                    shape: vec![],
                    dtype: DType::Unspecified,
                    spec: Some(SpaceKind::Text(TextSpec::default())),
                },
            ],
        })),
    };
    let dict = SpaceSpec {
        shape: vec![],
        dtype: DType::Unspecified,
        spec: Some(SpaceKind::Dict(DictSpec {
            keys: vec!["pos".to_owned(), "extra".to_owned()],
            spaces: vec![box_spec(), tuple],
        })),
    };
    let root = spec_ptr(&dict);

    let mut len = 0usize;
    assert_eq!(
        unsafe { rlmesh_space_len(root, &mut len) },
        RlmeshStatus::Ok
    );
    assert_eq!(len, 2);
    // Dict keys keep declaration order, parallel to the children.
    assert_eq!(space_key(root, 0), "pos");
    assert_eq!(space_key(root, 1), "extra");

    let extra = unsafe { rlmesh_space_dict_get(root, c"extra".as_ptr()) };
    assert_eq!(unsafe { rlmesh_space_type(extra) }, RlmeshValueKind::Tuple);
    assert_eq!(
        unsafe { rlmesh_space_len(extra, &mut len) },
        RlmeshStatus::Ok
    );
    assert_eq!(len, 2);
    assert_eq!(
        unsafe { rlmesh_space_type(rlmesh_space_tuple_get(extra, 0)) },
        RlmeshValueKind::Discrete
    );
    assert_eq!(
        unsafe { rlmesh_space_type(rlmesh_space_tuple_get(extra, 1)) },
        RlmeshValueKind::Text
    );
    // Children by index walk the SAME declaration order as the keys, so key(i)
    // names get_at(i) without copying the key into a NUL-terminated buffer.
    assert_eq!(
        unsafe { rlmesh_space_type(rlmesh_space_dict_get_at(root, 0)) },
        RlmeshValueKind::Box
    );
    assert_eq!(unsafe { rlmesh_space_dict_get_at(root, 1) }, unsafe {
        rlmesh_space_dict_get(root, c"extra".as_ptr())
    });
    assert!(unsafe { rlmesh_space_dict_get_at(root, 2) }.is_null());
    assert!(unsafe { rlmesh_space_dict_get_at(extra, 0) }.is_null());
    assert!(unsafe { rlmesh_space_dict_get_at(std::ptr::null(), 0) }.is_null());
    // Absent key, out-of-range index and wrong kind all read as "no such child".
    assert!(unsafe { rlmesh_space_dict_get(root, c"nope".as_ptr()) }.is_null());
    assert!(unsafe { rlmesh_space_tuple_get(extra, 2) }.is_null());
    assert!(unsafe { rlmesh_space_tuple_get(root, 0) }.is_null());
    // A leaf space has no children.
    let leaf = box_spec();
    assert_eq!(
        unsafe { rlmesh_space_len(spec_ptr(&leaf), &mut len) },
        RlmeshStatus::InvalidValue
    );
}

// ---- test helpers -------------------------------------------------------

fn spec_ptr(spec: &SpaceSpec) -> *const RlmeshSpaceSpec {
    std::ptr::from_ref(spec).cast::<RlmeshSpaceSpec>()
}

fn discrete_spec() -> SpaceSpec {
    SpaceSpec {
        shape: vec![],
        dtype: DType::Int64,
        spec: Some(SpaceKind::Discrete(DiscreteSpec { n: 8, start: 2 })),
    }
}

fn empty_tensor() -> RlmeshTensor {
    RlmeshTensor {
        data: std::ptr::null(),
        ndim: 0,
        shape: std::ptr::null(),
        strides: std::ptr::null(),
        dtype: F32,
        device_type: 0,
        device_id: 0,
        flags: 0,
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
    }
}

fn bounds(spec: &SpaceSpec, index: usize) -> (f64, f64) {
    let (mut low, mut high) = (0.0, 0.0);
    assert_eq!(
        unsafe { rlmesh_space_box_bounds(spec_ptr(spec), index, &mut low, &mut high) },
        RlmeshStatus::Ok
    );
    (low, high)
}

/// The `index`-th dict key of a value, as a `&str` over the borrowed bytes.
fn read_key(value: *const RlmeshValue, index: usize) -> String {
    let mut ptr: *const c_char = std::ptr::null();
    let mut len = 0usize;
    assert_eq!(
        unsafe { rlmesh_value_dict_key(value, index, &mut ptr, &mut len) },
        RlmeshStatus::Ok
    );
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    String::from_utf8(bytes.to_vec()).expect("utf-8 key")
}

/// A borrowed `Discrete` child's value (the child must exist).
fn discrete_at(value: *const RlmeshValue) -> i64 {
    assert!(!value.is_null(), "dict child missing");
    let mut out = 0i64;
    assert_eq!(
        unsafe { rlmesh_value_as_discrete(value, &mut out) },
        RlmeshStatus::Ok
    );
    out
}

/// A Text space's charset, as a `String` over the borrowed bytes.
fn read_charset(spec: *const RlmeshSpaceSpec) -> String {
    let mut ptr: *const c_char = std::ptr::null();
    let mut len = 0usize;
    assert_eq!(
        unsafe { rlmesh_space_text_charset(spec, &mut ptr, &mut len) },
        RlmeshStatus::Ok
    );
    if len == 0 {
        return String::new();
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    String::from_utf8(bytes.to_vec()).expect("utf-8 charset")
}

/// The `index`-th dict key of a space spec.
fn space_key(spec: *const RlmeshSpaceSpec, index: usize) -> String {
    let mut ptr: *const c_char = std::ptr::null();
    let mut len = 0usize;
    assert_eq!(
        unsafe { rlmesh_space_dict_key(spec, index, &mut ptr, &mut len) },
        RlmeshStatus::Ok
    );
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    String::from_utf8(bytes.to_vec()).expect("utf-8 key")
}
