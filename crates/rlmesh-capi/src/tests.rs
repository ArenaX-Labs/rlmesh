//! Tests exercising the FFI value handle + header constants against the real
//! core, without a live server.
#![allow(unsafe_code)] // exercising the raw FFI surface directly.

use std::ffi::c_char;

use rlmesh_spaces::spaces::{BoxSpaceBuilder, TextBuilder};
use rlmesh_spaces::{
    BoxBounds, BoxSpec, DType, DictSpec, DiscreteSpec, ElementwiseBounds, MultiDiscreteSpec,
    SpaceKind, SpaceSpec, TextSpec, TupleSpec, UniformBounds,
};

use crate::abi::status::RLMeshStatus;
use crate::spaces::{
    RLMeshSpaceSpec, rlmesh_ext_legacy_space_box, rlmesh_ext_legacy_space_box_bounds,
    rlmesh_ext_legacy_space_box_elementwise, rlmesh_ext_legacy_space_copy_nvec,
    rlmesh_ext_legacy_space_dict, rlmesh_ext_legacy_space_dict_get,
    rlmesh_ext_legacy_space_dict_get_at, rlmesh_ext_legacy_space_dict_key,
    rlmesh_ext_legacy_space_discrete, rlmesh_ext_legacy_space_discrete_n,
    rlmesh_ext_legacy_space_len, rlmesh_ext_legacy_space_text_charset,
    rlmesh_ext_legacy_space_text_length, rlmesh_ext_legacy_space_tuple,
    rlmesh_ext_legacy_space_tuple_get, rlmesh_space_spec_free,
};

use crate::space_handle::rlmesh_space_spec_kind;

const HEADER: &str = include_str!("../include/rlmesh.h");
fn header_enum_value(name: &str) -> i64 {
    let prefix = format!("{name} = ");
    let value = HEADER
        .lines()
        .find_map(|line| line.trim().strip_prefix(prefix.as_str()))
        .unwrap_or_else(|| panic!("{name} present in header"));
    let digits: String = value.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits
        .parse()
        .unwrap_or_else(|_| panic!("{name} has a numeric value"))
}

#[test]
fn header_status_and_class_values_match_crate() {
    use crate::abi::status::RLMeshErrorClass as C;
    use RLMeshStatus as S;
    let statuses = [
        ("RLMESH_OK", S::Ok),
        ("RLMESH_ERR_INVALID_ARGUMENT", S::InvalidArgument),
        ("RLMESH_ERR_INVALID_VALUE", S::InvalidValue),
        ("RLMESH_ERR_ENVIRONMENT", S::Environment),
        ("RLMESH_ERR_MODEL", S::Model),
        ("RLMESH_ERR_CONNECTION", S::Connection),
        ("RLMESH_ERR_TIMEOUT", S::Timeout),
        ("RLMESH_ERR_PANIC", S::Panic),
        ("RLMESH_ERR_CANCELLED", S::Cancelled),
        ("RLMESH_ERR_PROTOCOL", S::Protocol),
        ("RLMESH_ERR_TYPE", S::Type),
        ("RLMESH_ERR_RUNTIME", S::Runtime),
        ("RLMESH_ERR_OS", S::Os),
        ("RLMESH_ERR_INTERNAL", S::Internal),
    ];
    for (name, status) in statuses {
        assert_eq!(header_enum_value(name), status as i64, "{name}");
    }
    assert!(HEADER.contains("RLMESH_ERR_TRANSPORT = RLMESH_ERR_CONNECTION"));
    let classes = [
        ("RLMESH_ERROR_CLASS_INVALID", C::Invalid),
        ("RLMESH_ERROR_CLASS_RLMESH_EXCEPTION", C::RLMeshException),
        (
            "RLMESH_ERROR_CLASS_PROTOCOL_EXCEPTION",
            C::ProtocolException,
        ),
        (
            "RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION",
            C::EnvironmentException,
        ),
        (
            "RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION",
            C::RecoverableEnvironmentException,
        ),
        ("RLMESH_ERROR_CLASS_MODEL_EXCEPTION", C::ModelException),
        ("RLMESH_ERROR_CLASS_VALUE_ERROR", C::ValueError),
        (
            "RLMESH_ERROR_CLASS_ADAPTER_RESOLUTION_ERROR",
            C::AdapterResolutionError,
        ),
        ("RLMESH_ERROR_CLASS_TYPE_ERROR", C::TypeError),
        ("RLMESH_ERROR_CLASS_RUNTIME_ERROR", C::RuntimeError),
        ("RLMESH_ERROR_CLASS_OS_ERROR", C::OSError),
        ("RLMESH_ERROR_CLASS_CONNECTION_ERROR", C::ConnectionError),
        ("RLMESH_ERROR_CLASS_TIMEOUT_ERROR", C::TimeoutError),
        (
            "RLMESH_ERROR_CLASS_KEYBOARD_INTERRUPT",
            C::KeyboardInterrupt,
        ),
    ];
    for (name, class) in classes {
        assert_eq!(header_enum_value(name), class as i64, "{name}");
    }
}

#[test]
fn space_type_is_invalid_for_null_and_unspecified() {
    assert_eq!(rlmesh_space_spec_kind(std::ptr::null()), 0);
    let unspecified = SpaceSpec::default();
    assert_eq!(rlmesh_space_spec_kind(spec_ptr(&unspecified)), 0);
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
        unsafe { rlmesh_ext_legacy_space_box_bounds(spec_ptr(&uniform), 2, &mut low, &mut high) },
        RLMeshStatus::InvalidArgument
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
        unsafe { rlmesh_ext_legacy_space_box_bounds(spec_ptr(&discrete), 0, &mut low, &mut high) },
        RLMeshStatus::Type
    );

    // Either out-param may be skipped, as the header says; a failure writes
    // through neither.
    let (mut low, mut high) = (7.0, 7.0);
    assert_eq!(
        unsafe {
            rlmesh_ext_legacy_space_box_bounds(
                spec_ptr(&uniform),
                0,
                std::ptr::null_mut(),
                &mut high,
            )
        },
        RLMeshStatus::Ok
    );
    assert_eq!(
        unsafe {
            rlmesh_ext_legacy_space_box_bounds(
                spec_ptr(&uniform),
                0,
                &mut low,
                std::ptr::null_mut(),
            )
        },
        RLMeshStatus::Ok
    );
    assert_eq!((low, high), (-1.0, 1.0));
    let (mut low, mut high) = (7.0, 7.0);
    assert_eq!(
        unsafe { rlmesh_ext_legacy_space_box_bounds(spec_ptr(&uniform), 2, &mut low, &mut high) },
        RLMeshStatus::InvalidArgument
    );
    assert_eq!((low, high), (7.0, 7.0), "a failed call writes nothing");
}

#[test]
fn discrete_text_and_nvec_expose_what_an_action_needs() {
    let discrete = discrete_spec();
    let mut n = 0i64;
    let mut start = 0i64;
    assert_eq!(
        unsafe { rlmesh_ext_legacy_space_discrete_n(spec_ptr(&discrete), &mut n, &mut start) },
        RLMeshStatus::Ok
    );
    assert_eq!((n, start), (8, 2));
    // Either out-param may be skipped.
    assert_eq!(
        unsafe {
            rlmesh_ext_legacy_space_discrete_n(spec_ptr(&discrete), &mut n, std::ptr::null_mut())
        },
        RLMeshStatus::Ok
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
        unsafe { rlmesh_ext_legacy_space_text_length(spec_ptr(&text), &mut min, &mut max) },
        RLMeshStatus::Ok
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
            rlmesh_ext_legacy_space_text_charset(
                spec_ptr(&discrete),
                &mut charset_ptr,
                &mut charset_len,
            )
        },
        RLMeshStatus::Type
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
        unsafe {
            rlmesh_ext_legacy_space_copy_nvec(spec_ptr(&multi), nvec.as_mut_ptr(), nvec.len())
        },
        RLMeshStatus::Ok
    );
    assert_eq!(nvec, [2, 3, 4]);
    assert_eq!(
        unsafe { rlmesh_ext_legacy_space_copy_nvec(spec_ptr(&multi), nvec.as_mut_ptr(), 2) },
        RLMeshStatus::InvalidArgument
    );
    assert_eq!(
        unsafe { rlmesh_ext_legacy_space_copy_nvec(spec_ptr(&text), nvec.as_mut_ptr(), 3) },
        RLMeshStatus::Type
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
        unsafe { rlmesh_ext_legacy_space_len(root, &mut len) },
        RLMeshStatus::Ok
    );
    assert_eq!(len, 2);
    // Dict keys keep declaration order, parallel to the children.
    assert_eq!(space_key(root, 0), "pos");
    assert_eq!(space_key(root, 1), "extra");

    let extra = unsafe { rlmesh_ext_legacy_space_dict_get(root, c"extra".as_ptr()) };
    assert_eq!(rlmesh_space_spec_kind(extra), 11);
    assert_eq!(
        unsafe { rlmesh_ext_legacy_space_len(extra, &mut len) },
        RLMeshStatus::Ok
    );
    assert_eq!(len, 2);
    assert_eq!(
        unsafe { rlmesh_space_spec_kind(rlmesh_ext_legacy_space_tuple_get(extra, 0)) },
        2
    );
    assert_eq!(
        unsafe { rlmesh_space_spec_kind(rlmesh_ext_legacy_space_tuple_get(extra, 1)) },
        5
    );
    // Children by index walk the SAME declaration order as the keys, so key(i)
    // names get_at(i) without copying the key into a NUL-terminated buffer.
    assert_eq!(
        unsafe { rlmesh_space_spec_kind(rlmesh_ext_legacy_space_dict_get_at(root, 0)) },
        1
    );
    assert_eq!(
        unsafe { rlmesh_ext_legacy_space_dict_get_at(root, 1) },
        unsafe { rlmesh_ext_legacy_space_dict_get(root, c"extra".as_ptr()) }
    );
    assert!(unsafe { rlmesh_ext_legacy_space_dict_get_at(root, 2) }.is_null());
    assert!(unsafe { rlmesh_ext_legacy_space_dict_get_at(extra, 0) }.is_null());
    assert!(unsafe { rlmesh_ext_legacy_space_dict_get_at(std::ptr::null(), 0) }.is_null());
    // Absent key, out-of-range index and wrong kind all read as "no such child".
    assert!(unsafe { rlmesh_ext_legacy_space_dict_get(root, c"nope".as_ptr()) }.is_null());
    assert!(unsafe { rlmesh_ext_legacy_space_tuple_get(extra, 2) }.is_null());
    assert!(unsafe { rlmesh_ext_legacy_space_tuple_get(root, 0) }.is_null());
    // A leaf space has no children.
    let leaf = box_spec();
    assert_eq!(
        unsafe { rlmesh_ext_legacy_space_len(spec_ptr(&leaf), &mut len) },
        RLMeshStatus::Type
    );
}

// ---- test helpers -------------------------------------------------------

fn spec_ptr(spec: &SpaceSpec) -> *const RLMeshSpaceSpec {
    std::ptr::from_ref(spec).cast::<RLMeshSpaceSpec>()
}

fn discrete_spec() -> SpaceSpec {
    SpaceSpec {
        shape: vec![],
        dtype: DType::Int64,
        spec: Some(SpaceKind::Discrete(DiscreteSpec { n: 8, start: 2 })),
    }
}

fn bounds(spec: &SpaceSpec, index: usize) -> (f64, f64) {
    let (mut low, mut high) = (0.0, 0.0);
    assert_eq!(
        unsafe { rlmesh_ext_legacy_space_box_bounds(spec_ptr(spec), index, &mut low, &mut high) },
        RLMeshStatus::Ok
    );
    (low, high)
}

/// A Text space's charset, as a `String` over the borrowed bytes.
fn read_charset(spec: *const RLMeshSpaceSpec) -> String {
    let mut ptr: *const c_char = std::ptr::null();
    let mut len = 0usize;
    assert_eq!(
        unsafe { rlmesh_ext_legacy_space_text_charset(spec, &mut ptr, &mut len) },
        RLMeshStatus::Ok
    );
    if len == 0 {
        return String::new();
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    String::from_utf8(bytes.to_vec()).expect("utf-8 charset")
}

/// The `index`-th dict key of a space spec.
fn space_key(spec: *const RLMeshSpaceSpec, index: usize) -> String {
    let mut ptr: *const c_char = std::ptr::null();
    let mut len = 0usize;
    assert_eq!(
        unsafe { rlmesh_ext_legacy_space_dict_key(spec, index, &mut ptr, &mut len) },
        RLMeshStatus::Ok
    );
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    String::from_utf8(bytes.to_vec()).expect("utf-8 key")
}

fn capi_dtype(dtype: DType) -> i32 {
    dtype as i32
}

/// The owned spec behind a builder handle.
fn built_spec<'a>(spec: *mut RLMeshSpaceSpec) -> &'a SpaceSpec {
    assert!(!spec.is_null(), "builder failed");
    unsafe { &(*spec).0 }
}

#[test]
fn a_dict_space_the_builder_rejects_takes_none_of_its_children() {
    // An empty key passes the capi's own key checks and is refused only by the
    // Dict builder; both children must still be the caller's afterwards.
    let first = rlmesh_ext_legacy_space_discrete(2, 0);
    let second = rlmesh_ext_legacy_space_discrete(3, 0);
    let keys: [*const c_char; 2] = [c"ok".as_ptr(), c"".as_ptr()];
    let children = [first, second];
    assert!(unsafe { rlmesh_ext_legacy_space_dict(keys.as_ptr(), children.as_ptr(), 2) }.is_null());
    assert_eq!(
        crate::abi::status::rlmesh_last_error_status(),
        RLMeshStatus::InvalidArgument
    );
    assert_eq!(rlmesh_space_spec_kind(first), 2);
    assert_eq!(rlmesh_space_spec_kind(second), 2);
    // The same children still build a valid Dict, which then owns them.
    let keys: [*const c_char; 2] = [c"a".as_ptr(), c"b".as_ptr()];
    let dict = unsafe { rlmesh_ext_legacy_space_dict(keys.as_ptr(), children.as_ptr(), 2) };
    let mut len = 0usize;
    assert_eq!(
        unsafe { rlmesh_ext_legacy_space_len(dict, &mut len) },
        RLMeshStatus::Ok
    );
    assert_eq!(len, 2);
    unsafe { rlmesh_space_spec_free(dict) };
}

#[test]
fn a_tuple_space_with_a_null_child_takes_none_of_the_others() {
    let child = rlmesh_ext_legacy_space_discrete(2, 0);
    let children = [child, std::ptr::null_mut()];
    assert!(unsafe { rlmesh_ext_legacy_space_tuple(children.as_ptr(), 2) }.is_null());
    assert_eq!(rlmesh_space_spec_kind(child), 2);
    let tuple = unsafe { rlmesh_ext_legacy_space_tuple([child].as_ptr(), 1) };
    assert_eq!(rlmesh_space_spec_kind(tuple), 11);
    unsafe { rlmesh_space_spec_free(tuple) };
}

#[test]
fn integer_box_bounds_outside_the_dtype_are_refused_not_clamped() {
    let shape = [2i64];
    let refused = |dtype: DType, low: f64, high: f64| {
        let spec =
            unsafe { rlmesh_ext_legacy_space_box(capi_dtype(dtype), shape.as_ptr(), 1, low, high) };
        assert!(spec.is_null(), "{dtype:?} [{low}, {high}] built");
        assert_eq!(
            crate::abi::status::rlmesh_last_error_status(),
            RLMeshStatus::InvalidArgument
        );
        let elementwise = unsafe {
            rlmesh_ext_legacy_space_box_elementwise(
                capi_dtype(dtype),
                shape.as_ptr(),
                1,
                [low, low].as_ptr(),
                [high, high].as_ptr(),
            )
        };
        assert!(
            elementwise.is_null(),
            "{dtype:?} elementwise [{low}, {high}] built"
        );
    };
    // Past 64 bits: previously both saturated to i64::MAX and built.
    refused(DType::Int64, 1e30, 1e31);
    refused(DType::Int64, -1e30, 0.0);
    // 2^63 is i64::MAX + 1 (i64::MAX itself rounds up to it as a double).
    refused(DType::Int64, 0.0, 9_223_372_036_854_775_808.0);
    refused(DType::Uint64, 0.0, 18_446_744_073_709_551_616.0);
    refused(DType::Uint64, -1.0, 1.0);
    // Inside 64 bits but outside the dtype.
    refused(DType::Int8, 0.0, 128.0);
    refused(DType::Uint8, 0.0, 256.0);
    refused(DType::Int32, -2_147_483_649.0, 0.0);
}

#[test]
fn unsigned_box_bounds_above_i64_max_round_trip_exactly() {
    // 2^63 and 2^64 - 2^11 (the largest double below 2^64) fit Uint64 but not
    // i64: they must reach the spec unchanged, never via a saturating i64.
    let low_value = 1u64 << 63;
    let high_value = u64::MAX - 2047;
    let typed_high = |spec: &SpaceSpec| -> Vec<u8> {
        let Some(SpaceKind::Box(BoxSpec {
            bounds: Some(BoxBounds::TypedUniform(rlmesh_spaces::TypedUniformBounds { low, high })),
        })) = &spec.spec
        else {
            panic!("expected typed-uniform bounds: {spec:?}");
        };
        assert_eq!(*low, low_value.to_le_bytes());
        high.clone()
    };
    let shape = [1i64];
    let uniform = unsafe {
        rlmesh_ext_legacy_space_box(
            capi_dtype(DType::Uint64),
            shape.as_ptr(),
            1,
            low_value as f64,
            high_value as f64,
        )
    };
    assert_eq!(typed_high(built_spec(uniform)), high_value.to_le_bytes());
    unsafe { rlmesh_space_spec_free(uniform) };

    let shape = [2i64];
    let elementwise = unsafe {
        rlmesh_ext_legacy_space_box_elementwise(
            capi_dtype(DType::Uint64),
            shape.as_ptr(),
            1,
            [0.0, low_value as f64].as_ptr(),
            [high_value as f64, high_value as f64].as_ptr(),
        )
    };
    let Some(SpaceKind::Box(BoxSpec {
        bounds: Some(BoxBounds::TypedElementwise(typed)),
    })) = &built_spec(elementwise).spec
    else {
        panic!("expected typed-elementwise bounds");
    };
    let expect = |values: [u64; 2]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    assert_eq!(typed.low, expect([0, low_value]));
    assert_eq!(typed.high, expect([high_value, high_value]));
    unsafe { rlmesh_space_spec_free(elementwise) };

    // The signed edge still builds: i64::MIN is exactly -2^63.
    let shape = [1i64];
    let signed = unsafe {
        rlmesh_ext_legacy_space_box(
            capi_dtype(DType::Int64),
            shape.as_ptr(),
            1,
            i64::MIN as f64,
            0.0,
        )
    };
    assert!(!signed.is_null());
    unsafe { rlmesh_space_spec_free(signed) };
}

fn box_spec() -> SpaceSpec {
    BoxSpaceBuilder::unbounded(vec![2, 2])
        .dtype(DType::Float32)
        .build()
        .expect("box")
}

#[test]
fn abi5_removes_the_old_space_symbol_names() {
    let header = include_str!("../include/rlmesh.h");
    for name in [
        "type",
        "ndim",
        "copy_shape",
        "len",
        "tuple_get",
        "dict_get",
        "dict_get_at",
        "dict_key",
        "box_bounds",
        "discrete_n",
        "text_length",
        "text_charset",
        "copy_nvec",
        "box",
        "box_elementwise",
        "discrete",
        "multi_binary",
        "multi_discrete",
        "text",
        "dict",
        "tuple",
    ] {
        assert!(
            !header.contains(&format!("rlmesh_space_{name}(")),
            "old space symbol {name}"
        );
    }
    assert!(header.contains("rlmesh_space_dtype(const RLMeshSpace* space)"));
}

#[test]
fn legacy_exports_are_only_the_cpp_compatibility_surface() {
    use std::collections::BTreeSet;
    let cpp = include_str!("../include/rlmesh.hpp");
    let names = |text: &str| {
        text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .filter(|s| s.starts_with("rlmesh_ext_legacy_"))
            .map(str::to_owned)
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(
        names(HEADER),
        names(cpp),
        "remove any legacy declaration the wrapper no longer needs"
    );
    for record in [
        "RLMeshExtLegacyEpisode",
        "RLMeshExtLegacyObservation",
        "RLMeshExtLegacyResetArgs",
        "RLMeshExtLegacyResetResult",
        "RLMeshExtLegacyStepResult",
        "RLMeshExtLegacyRunReport",
    ] {
        assert!(cpp.contains(record), "migrate the orphaned record {record}");
    }
    for removed in [
        "rlmesh_ext_legacy_env_new(",
        "rlmesh_ext_legacy_space_type(",
        "rlmesh_ext_legacy_space_dtype(",
        "rlmesh_ext_legacy_space_ndim(",
        "rlmesh_ext_legacy_space_copy_shape(",
    ] {
        assert!(!HEADER.contains(removed), "removed symbol {removed}");
    }
}
