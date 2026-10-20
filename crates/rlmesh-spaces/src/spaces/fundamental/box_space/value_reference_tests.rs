//! Box conformance against the reference implementation it replaced.
//!
//! `reference::conform_box` is the original element-at-a-time scan (one
//! `Scalar` and one bound pair materialised per element). The typed fast paths
//! in `value.rs` must reach the same decision with the same error for every
//! dtype, bound form and value, including NaN, infinities, the dtype extremes,
//! empty and block-spanning tensors, and corrupt (unvalidated) bounds.

use proptest::prelude::*;

use super::value::conform_box;
use crate::dtype::{DType, dtype_size};
use crate::scalar::{Scalar, encode_scalars};
use crate::spaces::{Conformance, SpaceKind, SpaceSpec, SpaceValue};
use crate::tensor::Tensor;
use crate::{
    BoxBounds, BoxSpec, ElementwiseBounds, TypedElementwiseBounds, TypedUniformBounds,
    UniformBounds,
};

mod reference {
    use std::cmp::Ordering;

    use crate::BoxBounds;
    use crate::dtype::DType;
    use crate::errors::{SpaceError, err_space};
    use crate::scalar::{Scalar, decode_scalars};
    use crate::spaces::{Conformance, SpaceKind, SpaceSpec, SpaceValue};

    type Bound = (Option<Scalar>, Option<Scalar>);

    fn scalar_gt(a: Scalar, b: Scalar, dtype: DType) -> bool {
        a.cmp_typed(b, dtype) == Some(Ordering::Greater)
    }

    pub(super) fn conform_box(space: &SpaceSpec, value: &SpaceValue, path: &str) -> Conformance {
        let tensor = match value {
            SpaceValue::Box(tensor) => tensor,
            _ => return Conformance::Structural(SpaceError::invalid(path, "expected Box value")),
        };
        if tensor.shape() != space.shape.as_slice() {
            return Conformance::Structural(SpaceError::invalid(
                path,
                format!(
                    "shape mismatch: expected {:?}, got {:?}",
                    space.shape,
                    tensor.shape()
                ),
            ));
        }
        if tensor.dtype() != space.dtype {
            return Conformance::Structural(SpaceError::invalid(
                path,
                format!(
                    "dtype mismatch: expected {:?}, got {:?}",
                    space.dtype,
                    tensor.dtype()
                ),
            ));
        }
        let dtype = tensor.dtype();
        let numel = tensor.numel();
        let bounds = match box_bounds(space, numel, dtype, path) {
            Ok(bounds) => bounds,
            Err(err) => return Conformance::Structural(err),
        };
        let data = tensor.to_contiguous_bytes();
        let values = match decode_scalars(&data, dtype) {
            Ok(values) => values,
            Err(err) => {
                return Conformance::Structural(SpaceError::invalid(
                    path,
                    format!("cannot decode Box value: {err}"),
                ));
            }
        };
        let mut range: Option<SpaceError> = None;
        for (index, value) in values.iter().enumerate() {
            if let Scalar::Float(v) = value
                && v.is_nan()
            {
                return Conformance::Structural(SpaceError::invalid(
                    path,
                    format!("Box value at element {index} is NaN"),
                ));
            }
            if range.is_none() {
                let (low, high) = &bounds[index];
                range = box_scalar_range(*value, low.as_ref(), high.as_ref(), dtype, path, index);
            }
        }
        match range {
            Some(err) => Conformance::Range(err),
            None => Conformance::Ok,
        }
    }

    fn box_bounds(
        space: &SpaceSpec,
        numel: usize,
        dtype: DType,
        path: &str,
    ) -> Result<Vec<Bound>, SpaceError> {
        let spec = match &space.spec {
            Some(SpaceKind::Box(spec)) => spec,
            _ => return err_space!(path, "space is not Box"),
        };
        Ok(match &spec.bounds {
            Some(BoxBounds::Uniform(bounds)) => {
                let bound = (finite_float(bounds.low), finite_float(bounds.high));
                vec![bound; numel]
            }
            Some(BoxBounds::Elementwise(bounds)) => {
                if bounds.low.len() != numel || bounds.high.len() != numel {
                    return err_space!(
                        path,
                        format!(
                            "Box elementwise bounds length mismatch: expected {numel} \
                             elements, got low={}, high={}",
                            bounds.low.len(),
                            bounds.high.len()
                        )
                    );
                }
                bounds
                    .low
                    .iter()
                    .zip(bounds.high.iter())
                    .map(|(lo, hi)| (finite_float(*lo), finite_float(*hi)))
                    .collect()
            }
            Some(BoxBounds::TypedUniform(bounds)) => {
                let low = decode_typed_bounds(&bounds.low, 1, dtype, path)?;
                let high = decode_typed_bounds(&bounds.high, 1, dtype, path)?;
                vec![(Some(low[0]), Some(high[0])); numel]
            }
            Some(BoxBounds::TypedElementwise(bounds)) => {
                let low = decode_typed_bounds(&bounds.low, numel, dtype, path)?;
                let high = decode_typed_bounds(&bounds.high, numel, dtype, path)?;
                low.into_iter()
                    .zip(high)
                    .map(|(lo, hi)| (Some(lo), Some(hi)))
                    .collect()
            }
            Some(BoxBounds::Unbounded(_)) | None => vec![(None, None); numel],
        })
    }

    fn finite_float(value: f64) -> Option<Scalar> {
        value.is_finite().then_some(Scalar::Float(value))
    }

    fn decode_typed_bounds(
        bytes: &[u8],
        count: usize,
        dtype: DType,
        path: &str,
    ) -> Result<Vec<Scalar>, SpaceError> {
        let scalars = decode_scalars(bytes, dtype).map_err(|err| SpaceError::Invalid {
            path: path.to_string(),
            msg: format!("cannot decode typed Box bounds: {err}"),
        })?;
        if scalars.len() != count {
            return err_space!(
                path,
                format!(
                    "Box typed bounds length mismatch: expected {count} elements, got {}",
                    scalars.len()
                )
            );
        }
        Ok(scalars)
    }

    fn box_scalar_range(
        value: Scalar,
        low: Option<&Scalar>,
        high: Option<&Scalar>,
        dtype: DType,
        path: &str,
        index: usize,
    ) -> Option<SpaceError> {
        if let Some(low) = low
            && scalar_gt(*low, value, dtype)
        {
            return Some(SpaceError::invalid(
                path,
                format!("Box value at element {index} out of bounds: below low bound"),
            ));
        }
        if let Some(high) = high
            && scalar_gt(value, *high, dtype)
        {
            return Some(SpaceError::invalid(
                path,
                format!("Box value at element {index} out of bounds: above high bound"),
            ));
        }
        None
    }
}

/// Both implementations' verdicts, rendered so a mismatch shows the class and
/// the exact error (message and path).
fn verdicts(space: &SpaceSpec, value: &SpaceValue) -> (String, String) {
    let render = |conformance: Conformance| format!("{conformance:?}");
    (
        render(conform_box(space, value, "$.obs")),
        render(reference::conform_box(space, value, "$.obs")),
    )
}

fn concrete_dtype() -> impl Strategy<Value = DType> {
    prop::sample::select(
        DType::ALL
            .into_iter()
            .filter(|&dtype| dtype != DType::Unspecified)
            .collect::<Vec<_>>(),
    )
}

/// Shapes from scalar and empty up to a few blocks, so block boundaries of the
/// vectorized scans are crossed.
fn shape() -> impl Strategy<Value = Vec<i64>> {
    prop_oneof![
        8 => prop::collection::vec(0i64..=5, 0..=3),
        1 => (4090i64..=4100).prop_map(|n| vec![n]),
        1 => (2i64..=3, 4095i64..=4097).prop_map(|(a, b)| vec![a, b]),
    ]
}

/// One element's bytes in `dtype`: mostly small in-range numbers, with raw
/// bit patterns (dtype extremes, odd float encodings), fractions, NaN and the
/// infinities mixed in.
fn element(dtype: DType) -> impl Strategy<Value = Vec<u8>> {
    let size = dtype_size(dtype);
    let number = move |scalar: Scalar| encode_scalars(&[scalar], dtype).expect("encode");
    prop_oneof![
        12 => (-3i64..=12).prop_map(move |v| number(Scalar::Int(v))),
        2 => (-3.0f64..12.0).prop_map(move |v| number(Scalar::Float(v))),
        2 => any::<u64>().prop_map(move |bits| bits.to_le_bytes()[..size].to_vec()),
        1 => prop::sample::select(vec![
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            1e30,
            -1e30,
        ])
        .prop_map(move |v| number(Scalar::Float(v))),
        1 => prop::sample::select(vec![0x00u8, 0x7f, 0x80, 0xff]).prop_map(move |fill| {
            let mut bytes = vec![0xffu8; size];
            bytes[size - 1] = fill;
            bytes
        }),
    ]
}

/// A `double` bound: in-range integers, fractions, the infinities, NaN, and
/// magnitudes past every dtype.
fn double_bound() -> impl Strategy<Value = f64> {
    prop_oneof![
        8 => (-4i64..=13).prop_map(|v| v as f64),
        2 => -4.0f64..13.0,
        1 => prop::sample::select(vec![
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            255.0,
            256.0,
            -129.0,
            9.3e18,
            -9.3e18,
            1.9e19,
            1e300,
            -1e300,
        ]),
    ]
}

/// Typed bound bytes for `count` elements of `dtype`, occasionally the wrong
/// length so the length checks are compared too.
fn typed_bound(dtype: DType, count: usize) -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        12 => prop::collection::vec(element(dtype), count).prop_map(|parts| parts.concat()),
        1 => prop::collection::vec(any::<u8>(), 0..=3 * dtype_size(dtype) + 1),
    ]
}

fn bounds(dtype: DType, numel: usize) -> BoxedStrategy<Option<BoxBounds>> {
    // Per-element strategies get expensive on block-sized tensors; cap them
    // and let the length checks reject the rest.
    let per_element = numel.min(64);
    prop_oneof![
        1 => Just(None),
        1 => any::<bool>().prop_map(|flag| Some(BoxBounds::Unbounded(flag))),
        4 => (double_bound(), double_bound())
            .prop_map(|(low, high)| Some(BoxBounds::Uniform(UniformBounds { low, high }))),
        2 => (
            prop::collection::vec(double_bound(), per_element),
            prop::collection::vec(double_bound(), per_element),
        )
            .prop_map(|(low, high)| Some(BoxBounds::Elementwise(ElementwiseBounds { low, high }))),
        4 => (typed_bound(dtype, 1), typed_bound(dtype, 1)).prop_map(|(low, high)| {
            Some(BoxBounds::TypedUniform(TypedUniformBounds { low, high }))
        }),
        2 => (typed_bound(dtype, per_element), typed_bound(dtype, per_element)).prop_map(
            |(low, high)| Some(BoxBounds::TypedElementwise(TypedElementwiseBounds { low, high }))
        ),
    ]
    .boxed()
}

/// A Box space (unvalidated, so corrupt bounds reach conformance) and a value
/// of its shape and dtype.
fn case() -> impl Strategy<Value = (SpaceSpec, SpaceValue)> {
    (concrete_dtype(), shape()).prop_flat_map(|(dtype, shape)| {
        let numel: usize = shape.iter().map(|&dim| dim as usize).product();
        (
            bounds(dtype, numel),
            prop::collection::vec(element(dtype), numel),
        )
            .prop_map(move |(bounds, elements)| {
                let space = SpaceSpec {
                    shape: shape.clone(),
                    dtype,
                    spec: Some(SpaceKind::Box(BoxSpec { bounds })),
                };
                let tensor = Tensor::from_vec(elements.concat(), shape.clone(), dtype)
                    .expect("valid tensor");
                (space, SpaceValue::Box(tensor))
            })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn prop_conform_box_matches_reference((space, value) in case()) {
        let (fast, reference) = verdicts(&space, &value);
        prop_assert_eq!(fast, reference);
    }
}

fn encode(dtype: DType, values: &[Scalar]) -> Vec<u8> {
    encode_scalars(values, dtype).expect("encode")
}

fn box_case(dtype: DType, bounds: Option<BoxBounds>, data: Vec<u8>) -> (SpaceSpec, SpaceValue) {
    let numel = (data.len() / dtype_size(dtype)) as i64;
    let space = SpaceSpec {
        shape: vec![numel],
        dtype,
        spec: Some(SpaceKind::Box(BoxSpec { bounds })),
    };
    let value = SpaceValue::Box(Tensor::from_vec(data, vec![numel], dtype).expect("tensor"));
    (space, value)
}

fn assert_agrees(space: &SpaceSpec, value: &SpaceValue) -> String {
    let (fast, reference) = verdicts(space, value);
    assert_eq!(fast, reference);
    fast
}

/// Bounds covering a whole integer dtype short-circuit; values at both
/// extremes stay members, and one bound step inside the range catches them.
#[test]
fn full_range_integer_bounds_accept_every_value() {
    let extremes: [(DType, i128, i128); 9] = [
        (DType::Bool, 0, 1),
        (DType::Uint8, 0, u8::MAX.into()),
        (DType::Int8, i8::MIN.into(), i8::MAX.into()),
        (DType::Uint16, 0, u16::MAX.into()),
        (DType::Int16, i16::MIN.into(), i16::MAX.into()),
        (DType::Uint32, 0, u32::MAX.into()),
        (DType::Int32, i32::MIN.into(), i32::MAX.into()),
        (DType::Uint64, 0, u64::MAX.into()),
        (DType::Int64, i64::MIN.into(), i64::MAX.into()),
    ];
    for (dtype, min, max) in extremes {
        let size = dtype_size(dtype);
        let le = |v: i128| v.to_le_bytes()[..size].to_vec();
        let data = [le(min), le(max), le(min), le(max)].concat();
        let full = Some(BoxBounds::TypedUniform(TypedUniformBounds {
            low: le(min),
            high: le(max),
        }));
        let (space, value) = box_case(dtype, full, data.clone());
        assert_eq!(assert_agrees(&space, &value), "Ok", "{dtype:?}");

        let inner = Some(BoxBounds::TypedUniform(TypedUniformBounds {
            low: le(min),
            high: le(max - 1),
        }));
        let (space, value) = box_case(dtype, inner, data);
        let verdict = assert_agrees(&space, &value);
        assert!(
            verdict.contains("element 1 out of bounds: above high bound"),
            "{dtype:?}: {verdict}"
        );
    }
}

/// A NaN in a later block still outranks an earlier range deviation, and the
/// reported index is the first NaN's across block boundaries.
#[test]
fn nan_after_range_deviation_across_blocks_is_structural() {
    for dtype in [DType::Float16, DType::Float32, DType::Float64] {
        let mut values = vec![Scalar::Float(0.5); 10_000];
        values[3] = Scalar::Float(2.0);
        values[9_001] = Scalar::Float(f64::NAN);
        values[9_500] = Scalar::Float(f64::NAN);
        let uniform = Some(BoxBounds::Uniform(UniformBounds {
            low: 0.0,
            high: 1.0,
        }));
        let (space, value) = box_case(dtype, uniform, encode(dtype, &values));
        let verdict = assert_agrees(&space, &value);
        assert!(
            verdict.starts_with("Structural") && verdict.contains("element 9001 is NaN"),
            "{dtype:?}: {verdict}"
        );
    }
}

/// The first range deviation is reported even when it sits deep in a later
/// block; infinities against infinite bounds are members.
#[test]
fn first_range_deviation_and_infinite_bounds() {
    for dtype in [DType::Float16, DType::Float32, DType::Float64] {
        let mut values = vec![Scalar::Float(f64::INFINITY); 9_000];
        values[0] = Scalar::Float(f64::NEG_INFINITY);
        let unbounded = Some(BoxBounds::Uniform(UniformBounds {
            low: f64::NEG_INFINITY,
            high: f64::INFINITY,
        }));
        let (space, value) = box_case(dtype, unbounded, encode(dtype, &values));
        assert_eq!(assert_agrees(&space, &value), "Ok");

        let mut values = vec![Scalar::Float(0.0); 9_000];
        values[8_999] = Scalar::Float(-1.0);
        values[8_998] = Scalar::Float(5.0);
        let bounded = Some(BoxBounds::Uniform(UniformBounds {
            low: 0.0,
            high: 1.0,
        }));
        let (space, value) = box_case(dtype, bounded, encode(dtype, &values));
        let verdict = assert_agrees(&space, &value);
        assert!(
            verdict.contains("element 8998 out of bounds: above high bound"),
            "{dtype:?}: {verdict}"
        );
    }
    for dtype in [DType::Uint8, DType::Int32, DType::Uint64] {
        let mut values = vec![Scalar::Int(3); 9_000];
        values[4_096] = Scalar::Int(0);
        let bounded = Some(BoxBounds::Uniform(UniformBounds {
            low: 1.0,
            high: 7.0,
        }));
        let (space, value) = box_case(dtype, bounded, encode(dtype, &values));
        let verdict = assert_agrees(&space, &value);
        assert!(
            verdict.contains("element 4096 out of bounds: below low bound"),
            "{dtype:?}: {verdict}"
        );
    }
}

/// Bounds outside an integer dtype's range: past the far end every element
/// deviates; a fractional or out-of-range bound compares exactly.
#[test]
fn integer_bounds_beyond_the_dtype() {
    let data = encode(DType::Uint8, &[Scalar::Int(0), Scalar::Int(255)]);
    for (low, high) in [
        (256.0, 300.0),
        (-10.0, -1.0),
        (0.5, 254.5),
        (-1e300, 1e300),
        (300.0, -300.0),
    ] {
        let uniform = Some(BoxBounds::Uniform(UniformBounds { low, high }));
        let (space, value) = box_case(DType::Uint8, uniform, data.clone());
        assert_agrees(&space, &value);
    }
}

/// Empty tensors are members whatever the bounds, but corrupt bounds are still
/// structural.
#[test]
fn empty_tensors() {
    for dtype in DType::ALL.into_iter().filter(|&d| d != DType::Unspecified) {
        let typed = Some(BoxBounds::TypedUniform(TypedUniformBounds {
            low: encode(dtype, &[Scalar::Int(5)]),
            high: encode(dtype, &[Scalar::Int(1)]),
        }));
        let (space, value) = box_case(dtype, typed, Vec::new());
        assert_eq!(assert_agrees(&space, &value), "Ok", "{dtype:?}");

        let corrupt = Some(BoxBounds::TypedUniform(TypedUniformBounds {
            low: Vec::new(),
            high: Vec::new(),
        }));
        let (space, value) = box_case(dtype, corrupt, Vec::new());
        assert!(assert_agrees(&space, &value).starts_with("Structural"));
    }
}

/// `cargo test -p rlmesh-spaces --release box_conformance_timing -- --ignored --nocapture`:
/// ms per camera frame through Box conformance, against the reference scan.
#[test]
#[ignore = "timing, not correctness: run with --release --nocapture"]
#[allow(clippy::print_stdout, reason = "the point of the test is the report")]
fn box_conformance_timing() {
    let time = |f: &dyn Fn() -> Conformance| {
        let reps = 50;
        let start = std::time::Instant::now();
        for _ in 0..reps {
            assert!(std::hint::black_box(f()).is_ok());
        }
        start.elapsed().as_secs_f64() * 1e3 / f64::from(reps)
    };
    for (w, h) in [(640usize, 480usize), (224, 224)] {
        let shape = vec![h as i64, w as i64, 3];
        let numel = w * h * 3;
        let cases = [
            (
                "uint8 0..255",
                DType::Uint8,
                Some(BoxBounds::TypedUniform(TypedUniformBounds {
                    low: vec![0],
                    high: vec![255],
                })),
                vec![7u8; numel],
            ),
            (
                "uint8 0..200",
                DType::Uint8,
                Some(BoxBounds::TypedUniform(TypedUniformBounds {
                    low: vec![0],
                    high: vec![200],
                })),
                vec![7u8; numel],
            ),
            (
                "float32 0..1",
                DType::Float32,
                Some(BoxBounds::Uniform(UniformBounds {
                    low: 0.0,
                    high: 1.0,
                })),
                encode(DType::Float32, &vec![Scalar::Float(0.5); numel]),
            ),
            (
                "uint8 elementwise",
                DType::Uint8,
                Some(BoxBounds::TypedElementwise(TypedElementwiseBounds {
                    low: vec![0; numel],
                    high: vec![255; numel],
                })),
                vec![7u8; numel],
            ),
        ];
        for (label, dtype, bounds, data) in cases {
            let space = SpaceSpec {
                shape: shape.clone(),
                dtype,
                spec: Some(SpaceKind::Box(BoxSpec { bounds })),
            };
            let value =
                SpaceValue::Box(Tensor::from_vec(data, shape.clone(), dtype).expect("tensor"));
            let fast = time(&|| conform_box(&space, &value, "$"));
            let reference = time(&|| reference::conform_box(&space, &value, "$"));
            println!(
                "{w}x{h}x3 {label}: conform {fast:.4} ms/frame, reference {reference:.3} ms/frame"
            );
        }
    }
}
