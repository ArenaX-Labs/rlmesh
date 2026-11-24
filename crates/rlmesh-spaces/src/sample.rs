//! Deterministic, seedable sampling of values from a space spec.
//!
//! The Python bindings sample through the same code, so a given seed gives byte-identical
//! samples in Rust and Python. The RNG is pinned to [`ChaCha12Rng`] rather than `rand`'s
//! `StdRng`, which `rand` may change between releases.
//!
//! As in Gymnasium, integer Box elements are uniform over `[ceil(low), floor(high)]`,
//! unbounded or half-bounded continuous elements draw from exponential or normal tails, Text
//! draws a length then characters from the charset, and composites recurse in declared order.

use std::collections::BTreeMap;

use rand::RngExt;

use crate::BoxBounds;
use crate::dtype::DType;
use crate::errors::SpaceError;
use crate::scalar::{Scalar, decode_scalars, encode_scalars};
use crate::spaces::SpaceValue;
use crate::tensor::Tensor;
use crate::types::{
    BoxSpec, DictSpec, MultiDiscreteSpec, SpaceKind, SpaceSpec, TextSpec, TupleSpec,
};

/// The pinned sampler RNG.
pub use chacha20::ChaCha12Rng;

/// Sample a value from `spec`, drawing from `rng`.
///
/// `rng` advances in place, so repeated calls on one RNG draw fresh samples. Use
/// [`sample_seeded`] for a one-shot reproducible draw.
///
/// # Errors
/// [`SpaceError`] for a spec that cannot be sampled, such as an empty `Discrete`, a
/// non-positive `MultiDiscrete` dimension or inverted bounds.
pub fn sample_with<R: rand::Rng + ?Sized>(
    spec: &SpaceSpec,
    rng: &mut R,
) -> Result<SpaceValue, SpaceError> {
    let kind = spec
        .spec
        .as_ref()
        .ok_or_else(|| SpaceError::invalid("$", "space spec is missing"))?;
    match kind {
        SpaceKind::Box(box_spec) => Ok(SpaceValue::Box(sample_box(spec, box_spec, rng)?)),
        SpaceKind::Discrete(discrete) => {
            if discrete.n <= 0 {
                return Err(SpaceError::invalid(
                    "$",
                    format!("cannot sample Discrete space with n={}", discrete.n),
                ));
            }
            let end = discrete
                .start
                .checked_add(discrete.n - 1)
                .ok_or_else(|| SpaceError::invalid("$", "Discrete range exceeds i64::MAX"))?;
            Ok(SpaceValue::Discrete(rng.random_range(discrete.start..=end)))
        }
        SpaceKind::MultiBinary(_) => Ok(SpaceValue::MultiBinary(sample_multi_binary(spec, rng))),
        SpaceKind::MultiDiscrete(spec) => {
            Ok(SpaceValue::MultiDiscrete(sample_multi_discrete(spec, rng)?))
        }
        SpaceKind::Text(text) => Ok(SpaceValue::Text(sample_text(text, rng))),
        SpaceKind::Dict(dict) => sample_dict(dict, rng),
        SpaceKind::Tuple(tuple) => sample_tuple(tuple, rng),
    }
}

/// Sample a value from `spec` using a fresh [`ChaCha12Rng`] seeded with `seed`.
///
/// The same `(spec, seed)` always yields the same value. For a stream of fresh samples,
/// call [`sample_with`] repeatedly on one RNG.
pub fn sample_seeded(spec: &SpaceSpec, seed: u64) -> Result<SpaceValue, SpaceError> {
    use rand::SeedableRng;
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    sample_with(spec, &mut rng)
}

fn sample_box<R: rand::Rng + ?Sized>(
    space: &SpaceSpec,
    spec: &BoxSpec,
    rng: &mut R,
) -> Result<Tensor, SpaceError> {
    let numel = numel(&space.shape);
    let dtype = normalize_dtype(space.dtype);
    let mut bytes = Vec::new();
    let mut block = Vec::with_capacity(numel.min(SAMPLE_BLOCK));
    let mut flush = |block: &mut Vec<Scalar>| -> Result<(), SpaceError> {
        let encoded = encode_scalars(block, dtype)
            .map_err(|err| SpaceError::invalid("$", err.to_string()))?;
        bytes.extend(encoded);
        block.clear();
        Ok(())
    };
    if is_integer_dtype(space.dtype) {
        // Integer bounds are decoded as integers, never through f64.
        let (low, high) = integer_box_bounds(spec, numel.max(1), space.dtype);
        for index in 0..numel {
            let scalar =
                sample_integer_box_scalar(rng, low.at(index), high.at(index), space.dtype)?;
            block.push(scalar);
            if block.len() == SAMPLE_BLOCK {
                flush(&mut block)?;
            }
        }
    } else {
        let (low, high) = box_bounds(spec, numel.max(1), space.dtype);
        for index in 0..numel {
            let scalar = sample_box_scalar(rng, low.at(index), high.at(index), space.dtype)?;
            block.push(scalar);
            if block.len() == SAMPLE_BLOCK {
                flush(&mut block)?;
            }
        }
    }
    flush(&mut block)?;
    // `from_slice` (not `from_vec`) lands the buffer in 64-byte-aligned storage, which zero-copy DLPack/numpy consumers rely on.
    Tensor::from_slice(&bytes, &space.shape, dtype)
        .map_err(|err| SpaceError::invalid("$", err.to_string()))
}

const SAMPLE_BLOCK: usize = 4096;

#[derive(Debug, Clone, PartialEq)]
enum Edges<T> {
    All(T),
    Each(Vec<T>),
}

impl<T: Copy> Edges<T> {
    fn at(&self, index: usize) -> T {
        match self {
            Edges::All(edge) => *edge,
            Edges::Each(edges) => edges[index],
        }
    }
}

fn box_bounds(spec: &BoxSpec, numel: usize, dtype: DType) -> (Edges<f64>, Edges<f64>) {
    match &spec.bounds {
        Some(BoxBounds::Uniform(bounds)) => (Edges::All(bounds.low), Edges::All(bounds.high)),
        Some(BoxBounds::Elementwise(bounds)) => (
            elementwise_or_default(bounds.low.as_slice(), numel, f64::NEG_INFINITY),
            elementwise_or_default(bounds.high.as_slice(), numel, f64::INFINITY),
        ),
        Some(BoxBounds::TypedUniform(bounds)) => {
            let low = decode_typed_one(&bounds.low, dtype).unwrap_or(f64::NEG_INFINITY);
            let high = decode_typed_one(&bounds.high, dtype).unwrap_or(f64::INFINITY);
            (Edges::All(low), Edges::All(high))
        }
        Some(BoxBounds::TypedElementwise(bounds)) => (
            decode_typed_many(&bounds.low, dtype, numel, f64::NEG_INFINITY),
            decode_typed_many(&bounds.high, dtype, numel, f64::INFINITY),
        ),
        Some(BoxBounds::Unbounded(_)) | None => {
            (Edges::All(f64::NEG_INFINITY), Edges::All(f64::INFINITY))
        }
    }
}

fn decode_typed_one(bytes: &[u8], dtype: DType) -> Option<f64> {
    decode_scalars(bytes, dtype)
        .ok()
        .and_then(|scalars| scalars.first().copied().map(|s| s.to_f64(dtype)))
}

fn decode_typed_many(bytes: &[u8], dtype: DType, numel: usize, default: f64) -> Edges<f64> {
    match decode_scalars(bytes, dtype) {
        Ok(scalars) if scalars.len() == numel => {
            Edges::Each(scalars.into_iter().map(|s| s.to_f64(dtype)).collect())
        }
        _ => Edges::All(default),
    }
}

type IntEdge = Option<i128>;

fn integer_box_bounds(
    spec: &BoxSpec,
    numel: usize,
    dtype: DType,
) -> (Edges<IntEdge>, Edges<IntEdge>) {
    fn float_low(low: f64) -> IntEdge {
        low.is_finite().then(|| low.ceil() as i128)
    }
    fn float_high(high: f64) -> IntEdge {
        high.is_finite().then(|| high.floor() as i128)
    }
    fn float_edges(values: &[f64], numel: usize, edge: fn(f64) -> IntEdge) -> Edges<IntEdge> {
        if values.len() == numel {
            Edges::Each(values.iter().map(|value| edge(*value)).collect())
        } else {
            Edges::All(None)
        }
    }
    let typed_one = |bytes: &[u8]| -> IntEdge {
        decode_scalars(bytes, dtype)
            .ok()
            .and_then(|scalars| scalars.first().map(|s| scalar_to_int_edge(*s, dtype)))
    };
    let typed_many = |bytes: &[u8]| -> Edges<IntEdge> {
        match decode_scalars(bytes, dtype) {
            Ok(scalars) if scalars.len() == numel => Edges::Each(
                scalars
                    .into_iter()
                    .map(|s| Some(scalar_to_int_edge(s, dtype)))
                    .collect(),
            ),
            _ => Edges::All(None),
        }
    };
    match &spec.bounds {
        Some(BoxBounds::Uniform(bounds)) => (
            Edges::All(float_low(bounds.low)),
            Edges::All(float_high(bounds.high)),
        ),
        Some(BoxBounds::Elementwise(bounds)) => (
            float_edges(bounds.low.as_slice(), numel, float_low),
            float_edges(bounds.high.as_slice(), numel, float_high),
        ),
        Some(BoxBounds::TypedUniform(bounds)) => (
            Edges::All(typed_one(&bounds.low)),
            Edges::All(typed_one(&bounds.high)),
        ),
        Some(BoxBounds::TypedElementwise(bounds)) => {
            (typed_many(&bounds.low), typed_many(&bounds.high))
        }
        Some(BoxBounds::Unbounded(_)) | None => (Edges::All(None), Edges::All(None)),
    }
}

fn scalar_to_int_edge(scalar: Scalar, dtype: DType) -> i128 {
    match scalar {
        Scalar::Int(_) if dtype == DType::Uint64 => i128::from(scalar.as_u64()),
        Scalar::Float(value) => value as i128,
        other => i128::from(other.as_i64()),
    }
}

fn elementwise_or_default(values: &[f64], len: usize, default: f64) -> Edges<f64> {
    if values.len() == len {
        Edges::Each(values.to_vec())
    } else {
        Edges::All(default)
    }
}

fn sample_box_scalar<R: rand::Rng + ?Sized>(
    rng: &mut R,
    low: f64,
    high: f64,
    dtype: DType,
) -> Result<Scalar, SpaceError> {
    let value = sample_continuous(rng, low, high)?;
    if matches!(dtype, DType::Bool) {
        return Ok(Scalar::Bool(value.round() != 0.0));
    }
    Ok(Scalar::Float(value))
}

fn sample_continuous<R: rand::Rng + ?Sized>(
    rng: &mut R,
    low: f64,
    high: f64,
) -> Result<f64, SpaceError> {
    let value = if low.is_finite() && high.is_finite() {
        if low > high {
            return Err(SpaceError::invalid(
                "$",
                format!("cannot sample Box element with low {low} greater than high {high}"),
            ));
        }
        rng.random_range(low..=high)
    } else if low.is_finite() {
        low + exp_sample(rng)
    } else if high.is_finite() {
        high - exp_sample(rng)
    } else {
        normal_sample(rng)
    };
    Ok(value)
}

fn sample_integer_box_scalar<R: rand::Rng + ?Sized>(
    rng: &mut R,
    low: IntEdge,
    high: IntEdge,
    dtype: DType,
) -> Result<Scalar, SpaceError> {
    let (dtype_lo, dtype_hi) = integer_dtype_bounds(dtype);
    let lo = low.unwrap_or(dtype_lo);
    let hi = high.unwrap_or(dtype_hi);
    if lo > hi {
        return Err(SpaceError::invalid(
            "$",
            format!("cannot sample integer Box element with low {lo} greater than high {hi}"),
        ));
    }
    let (i64_min, i64_max) = (i128::from(i64::MIN), i128::from(i64::MAX));
    if hi > i64_max && lo >= 0 {
        let hi = u64::try_from(hi).unwrap_or(u64::MAX);
        let lo = u64::try_from(lo).unwrap_or(u64::MAX).min(hi);
        return Ok(Scalar::Int(rng.random_range(lo..=hi) as i64));
    }
    let lo = lo.clamp(i64_min, i64_max) as i64;
    let hi = hi.clamp(i64_min, i64_max) as i64;
    Ok(Scalar::Int(rng.random_range(lo..=hi)))
}

fn is_integer_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Uint8
            | DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::Uint16
            | DType::Uint32
            | DType::Uint64
    )
}

fn integer_dtype_bounds(dtype: DType) -> (i128, i128) {
    match dtype {
        DType::Bool => (0, 1),
        DType::Uint8 => (0, i128::from(u8::MAX)),
        DType::Int8 => (i128::from(i8::MIN), i128::from(i8::MAX)),
        DType::Int16 => (i128::from(i16::MIN), i128::from(i16::MAX)),
        DType::Uint16 => (0, i128::from(u16::MAX)),
        DType::Int32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
        DType::Uint32 => (0, i128::from(u32::MAX)),
        DType::Uint64 => (0, 1i128 << 53),
        _ => (-(1i128 << 53), 1i128 << 53),
    }
}

fn exp_sample<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    let u = (1.0 - rng.random::<f64>()).max(f64::MIN_POSITIVE);
    -u.ln()
}

fn normal_sample<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1 = rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn sample_multi_binary<R: rand::Rng + ?Sized>(space: &SpaceSpec, rng: &mut R) -> Vec<bool> {
    let numel = numel(&space.shape);
    (0..numel).map(|_| rng.random::<bool>()).collect()
}

fn sample_multi_discrete<R: rand::Rng + ?Sized>(
    spec: &MultiDiscreteSpec,
    rng: &mut R,
) -> Result<Vec<i64>, SpaceError> {
    spec.nvec
        .iter()
        .enumerate()
        .map(|(i, n)| {
            if *n <= 0 {
                return Err(SpaceError::invalid(
                    "$",
                    "cannot sample MultiDiscrete space with a non-positive dimension",
                ));
            }
            Ok(spec.start(i) + rng.random_range(0..*n))
        })
        .collect()
}

fn sample_text<R: rand::Rng + ?Sized>(spec: &TextSpec, rng: &mut R) -> String {
    let printable_ascii = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?";
    let chars = if spec.charset.is_empty() {
        printable_ascii.chars().collect::<Vec<_>>()
    } else {
        spec.charset.chars().collect::<Vec<_>>()
    };
    let length = if spec.max_length <= spec.min_length {
        spec.min_length
    } else {
        rng.random_range(spec.min_length..=spec.max_length)
    };
    (0..length)
        .map(|_| chars[rng.random_range(0..chars.len())])
        .collect()
}

fn sample_dict<R: rand::Rng + ?Sized>(
    spec: &DictSpec,
    rng: &mut R,
) -> Result<SpaceValue, SpaceError> {
    // Draw in declared key order (the order the value codec walks) so the RNG stream is stable; the BTreeMap's sorted order is irrelevant.
    let mut values = BTreeMap::new();
    for (key, child) in spec.keys.iter().zip(spec.spaces.iter()) {
        values.insert(key.clone(), sample_with(child, rng)?);
    }
    Ok(SpaceValue::Dict(values))
}

fn sample_tuple<R: rand::Rng + ?Sized>(
    spec: &TupleSpec,
    rng: &mut R,
) -> Result<SpaceValue, SpaceError> {
    let values = spec
        .spaces
        .iter()
        .map(|child| sample_with(child, rng))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SpaceValue::Tuple(values))
}

fn numel(shape: &[i64]) -> usize {
    shape.iter().map(|dim| *dim as usize).product()
}

fn normalize_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Unspecified => DType::Float32,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::{
        BoxSpaceBuilder, DictSpaceBuilder, DiscreteBuilder, MultiDiscreteBuilder, TextBuilder,
        TupleSpaceBuilder, contains,
    };
    use crate::types::{BoxSpec, DiscreteSpec};
    use crate::{TypedUniformBounds, encode_i64_scalars};
    use rand::SeedableRng;

    fn typed_uniform_spec(low: i64, high: i64, dtype: DType) -> BoxSpec {
        BoxSpec {
            bounds: Some(BoxBounds::TypedUniform(TypedUniformBounds {
                low: encode_i64_scalars(&[low], dtype).expect("encode low"),
                high: encode_i64_scalars(&[high], dtype).expect("encode high"),
            })),
        }
    }

    #[test]
    fn same_seed_reproduces_identical_samples() {
        let dict = DictSpaceBuilder::new()
            .insert("choice", DiscreteBuilder::new(5).start(-2).build().unwrap())
            .insert(
                "vec",
                BoxSpaceBuilder::scalar(-1.0, 1.0, vec![2, 3])
                    .dtype(DType::Float32)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();
        let specs = [
            BoxSpaceBuilder::unbounded(vec![4])
                .dtype(DType::Float32)
                .build()
                .unwrap(),
            BoxSpaceBuilder::int_scalar(0, 9, vec![3])
                .dtype(DType::Int32)
                .build()
                .unwrap(),
            DiscreteBuilder::new(7).start(3).build().unwrap(),
            MultiDiscreteBuilder::vector(vec![2, 3, 4]).build().unwrap(),
            TextBuilder::new(12).min_length(3).build().unwrap(),
            TupleSpaceBuilder::new()
                .with(DiscreteBuilder::new(3).build().unwrap())
                .with(TextBuilder::new(8).build().unwrap())
                .build()
                .unwrap(),
            dict,
        ];

        for spec in &specs {
            let a = sample_seeded(spec, 0xC0FFEE).unwrap();
            let b = sample_seeded(spec, 0xC0FFEE).unwrap();
            assert_eq!(a, b, "same seed must reproduce the same sample for {spec}");

            assert!(
                contains(spec, &a).is_ok(),
                "sample must be a member of {spec}"
            );
        }
    }

    #[test]
    fn golden_samples_are_frozen() {
        const SEED: u64 = 0xC0FFEE;

        let unbounded = BoxSpaceBuilder::unbounded(vec![3])
            .dtype(DType::Float64)
            .build()
            .unwrap();
        let SpaceValue::Box(tensor) = sample_seeded(&unbounded, SEED).unwrap() else {
            panic!("expected a Box value");
        };
        assert_eq!(
            tensor.to_contiguous_bytes().to_vec(),
            vec![
                88, 45, 25, 14, 242, 232, 239, 63, 168, 199, 73, 102, 128, 93, 1, 192, 192, 106,
                194, 145, 59, 178, 222, 191
            ]
        );

        let int_box = BoxSpaceBuilder::int_scalar(0, 100, vec![4])
            .dtype(DType::Int32)
            .build()
            .unwrap();
        let SpaceValue::Box(tensor) = sample_seeded(&int_box, SEED).unwrap() else {
            panic!("expected a Box value");
        };
        assert_eq!(
            tensor.to_contiguous_bytes().to_vec(),
            vec![60, 0, 0, 0, 98, 0, 0, 0, 9, 0, 0, 0, 50, 0, 0, 0]
        );

        let discrete = DiscreteBuilder::new(1_000).start(-50).build().unwrap();
        assert_eq!(
            sample_seeded(&discrete, SEED).unwrap(),
            SpaceValue::Discrete(551)
        );

        let multi = MultiDiscreteBuilder::vector(vec![5, 7, 9]).build().unwrap();
        assert_eq!(
            sample_seeded(&multi, SEED).unwrap(),
            SpaceValue::MultiDiscrete(vec![3, 6, 0])
        );

        let text = TextBuilder::new(10).min_length(5).build().unwrap();
        assert_eq!(
            sample_seeded(&text, SEED).unwrap(),
            SpaceValue::Text("I!XgKHTz".to_string())
        );
    }

    #[test]
    fn text_with_zero_bounds_samples_within_bounds() {
        let empty_only = TextBuilder::new(0).min_length(0).build().unwrap();
        assert_eq!(
            sample_seeded(&empty_only, 7).unwrap(),
            SpaceValue::Text(String::new())
        );

        let space = TextBuilder::new(3).min_length(0).build().unwrap();
        for seed in 0..32 {
            let value = sample_seeded(&space, seed).unwrap();
            assert!(contains(&space, &value).is_ok(), "seed {seed}: {value:?}");
        }
    }

    #[test]
    fn rng_stream_advances_across_calls() {
        let spec = BoxSpaceBuilder::unbounded(vec![4])
            .dtype(DType::Float64)
            .build()
            .unwrap();
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let first = sample_with(&spec, &mut rng).unwrap();
        let second = sample_with(&spec, &mut rng).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn degenerate_specs_are_rng_independent() {
        assert_eq!(
            sample_seeded(&DiscreteBuilder::new(1).build().unwrap(), 1).unwrap(),
            SpaceValue::Discrete(0)
        );
        assert_eq!(
            sample_seeded(
                &MultiDiscreteBuilder::vector(vec![1, 1, 1]).build().unwrap(),
                1
            )
            .unwrap(),
            SpaceValue::MultiDiscrete(vec![0, 0, 0])
        );
        let degenerate_box = sample_seeded(
            &BoxSpaceBuilder::scalar(5.0, 5.0, vec![1])
                .dtype(DType::Float32)
                .build()
                .unwrap(),
            1,
        )
        .unwrap();
        let SpaceValue::Box(tensor) = degenerate_box else {
            panic!("expected a Box value");
        };
        assert_eq!(
            tensor.to_contiguous_bytes().to_vec(),
            5.0f32.to_le_bytes().to_vec()
        );
    }

    #[test]
    fn box_sample_storage_is_aligned() {
        let spec = BoxSpaceBuilder::scalar(-1.0, 1.0, vec![8])
            .dtype(DType::Float64)
            .build()
            .unwrap();
        let SpaceValue::Box(tensor) = sample_seeded(&spec, 7).unwrap() else {
            panic!("expected a Box value");
        };
        assert_eq!(
            tensor.storage().as_slice().as_ptr() as usize % 64,
            0,
            "sampled Box tensor must use 64-byte-aligned storage"
        );
    }

    #[test]
    fn uint64_max_bound_decodes_to_positive_for_sampling() {
        let spec = typed_uniform_spec(0, -1, DType::Uint64);
        let (low, high) = integer_box_bounds(&spec, 1, DType::Uint64);
        assert_eq!(low.at(0), Some(0));
        assert_eq!(
            high.at(0),
            Some(i128::from(u64::MAX)),
            "u64::MAX must decode positive and exact"
        );

        let mut rng = ChaCha12Rng::seed_from_u64(7);
        let sampled = sample_integer_box_scalar(&mut rng, low.at(0), high.at(0), DType::Uint64);
        assert!(
            sampled.is_ok(),
            "sampling a [0, u64::MAX] Uint64 box must not error: {sampled:?}"
        );
    }

    fn box_u64_samples(spec: &SpaceSpec, seeds: std::ops::Range<u64>) -> Vec<u64> {
        seeds
            .flat_map(|seed| {
                let SpaceValue::Box(tensor) = sample_seeded(spec, seed).unwrap() else {
                    panic!("expected a Box value");
                };
                tensor
                    .to_contiguous_bytes()
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|chunk| u64::from_le_bytes(*chunk))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn uint64_bounds_above_i64_max_sample_exactly() {
        let low = (1u64 << 63) + 10;
        let high = (1u64 << 63) + 13;
        let spec = BoxSpaceBuilder::uint_scalar(low, high, vec![16])
            .dtype(DType::Uint64)
            .build()
            .unwrap();
        let samples = box_u64_samples(&spec, 0..8);
        assert!(
            samples.iter().all(|v| (low..=high).contains(v)),
            "{samples:?}"
        );
        for v in low..=high {
            assert!(samples.contains(&v), "{v} never sampled: {samples:?}");
        }
    }

    #[test]
    fn int64_bounds_beyond_2_53_sample_exactly() {
        let low = (1i64 << 60) + 1;
        let high = (1i64 << 60) + 3;
        let spec = BoxSpaceBuilder::int_scalar(low, high, vec![16])
            .dtype(DType::Int64)
            .build()
            .unwrap();
        let samples = box_u64_samples(&spec, 0..8);
        for v in &samples {
            let v = *v as i64;
            assert!((low..=high).contains(&v), "{v} outside [{low}, {high}]");
        }
        assert!(
            samples.contains(&((low + 1) as u64)),
            "inner value never sampled"
        );
    }

    #[test]
    fn scalar_to_f64_matches_dtype() {
        assert_eq!(Scalar::Int(-1).to_f64(DType::Uint64), u64::MAX as f64);
        assert_eq!(Scalar::Int(-1).to_f64(DType::Int64), -1.0);
    }

    #[test]
    fn empty_discrete_is_an_error() {
        let spec = SpaceSpec {
            shape: vec![],
            dtype: DType::Int64,
            spec: Some(SpaceKind::Discrete(DiscreteSpec { n: 0, start: 0 })),
        };
        assert!(sample_seeded(&spec, 1).is_err());
    }
}

#[cfg(test)]
mod boundary_regressions {
    use super::*;
    use crate::spaces::DiscreteBuilder;

    #[test]
    fn discrete_samples_the_entire_signed_boundary() {
        for start in [i64::MIN, i64::MAX] {
            let spec = DiscreteBuilder::new(1).start(start).build().unwrap();
            assert_eq!(
                sample_seeded(&spec, 0).unwrap(),
                SpaceValue::Discrete(start)
            );
        }
        let spec = DiscreteBuilder::new(2).start(i64::MAX - 1).build().unwrap();
        for seed in 0..20 {
            let SpaceValue::Discrete(value) = sample_seeded(&spec, seed).unwrap() else {
                panic!()
            };
            assert!((i64::MAX - 1..=i64::MAX).contains(&value));
        }
    }
}
