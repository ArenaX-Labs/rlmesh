//! Deterministic, seedable sampling of values from a space spec.
//!
//! This is the single cross-language sampler. The Python bindings drive it
//! through the same code path, so a given seed produces byte-identical samples
//! in Rust and Python. The RNG is pinned to [`ChaCha12Rng`] (re-exported from
//! the `chacha20` crate) rather than `rand`'s `StdRng`: the two are the same
//! algorithm today, but `rand` reserves the right to change `StdRng` between
//! releases, so pinning the concrete cipher freezes the seed->sample contract
//! across `rand` upgrades.
//!
//! Sampling semantics mirror Gymnasium: integer Box elements are uniform over
//! `[ceil(low), floor(high)]`, unbounded/half-bounded continuous elements fall
//! back to exponential/normal tails, Text draws a length then characters from
//! the charset, and composites recurse in declared order.

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

/// The pinned sampler RNG. Re-exported so callers (e.g. the Python bindings)
/// hold and seed the same concrete type without depending on `chacha20`.
pub use chacha20::ChaCha12Rng;

/// Sample a value from `spec`, drawing from `rng`.
///
/// `rng` is advanced in place, so repeated calls on one RNG yield a stream of
/// distinct samples (Gymnasium's seed-once-then-sample semantics). Use
/// [`sample_seeded`] for a one-shot reproducible draw from a fresh stream.
///
/// Returns a [`SpaceError`] for specs that cannot be sampled (e.g. an empty
/// `Discrete`, a non-positive `MultiDiscrete` dimension, or inverted bounds).
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
            Ok(SpaceValue::Discrete(rng.random_range(
                discrete.start..(discrete.start + discrete.n),
            )))
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
/// Every call starts a new RNG stream, so the same `(spec, seed)` always yields
/// the same value. For a stream of distinct samples, build one [`ChaCha12Rng`]
/// and call [`sample_with`] repeatedly.
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
    let (low, high) = box_bounds(spec, numel.max(1), space.dtype);
    let scalars = low
        .iter()
        .zip(high.iter())
        .take(numel)
        .map(|(low, high)| sample_box_scalar(rng, *low, *high, space.dtype))
        .collect::<Result<Vec<_>, _>>()?;

    // The sampler treats `Unspecified` specs as float32, matching the codecs.
    let dtype = normalize_dtype(space.dtype);
    let bytes =
        encode_scalars(&scalars, dtype).map_err(|err| SpaceError::invalid("$", err.to_string()))?;
    // `from_slice` (not `from_vec`) so the buffer lands in 64-byte-aligned
    // storage: the value codec hands a Box tensor over zero-copy, and the
    // DLPack/numpy consumers downstream rely on that alignment.
    Tensor::from_slice(&bytes, &space.shape, dtype)
        .map_err(|err| SpaceError::invalid("$", err.to_string()))
}

fn box_bounds(spec: &BoxSpec, numel: usize, dtype: DType) -> (Vec<f64>, Vec<f64>) {
    match &spec.bounds {
        Some(BoxBounds::Uniform(bounds)) => (vec![bounds.low; numel], vec![bounds.high; numel]),
        Some(BoxBounds::Elementwise(bounds)) => (
            elementwise_or_default(bounds.low.as_slice(), numel, f64::NEG_INFINITY),
            elementwise_or_default(bounds.high.as_slice(), numel, f64::INFINITY),
        ),
        // Typed byte bounds are decoded with the space dtype into f64 for
        // sampling. The sampler only needs an approximate range, so the f64
        // view is acceptable here even for int64/uint64.
        Some(BoxBounds::TypedUniform(bounds)) => {
            let low = decode_typed_one(&bounds.low, dtype).unwrap_or(f64::NEG_INFINITY);
            let high = decode_typed_one(&bounds.high, dtype).unwrap_or(f64::INFINITY);
            (vec![low; numel], vec![high; numel])
        }
        Some(BoxBounds::TypedElementwise(bounds)) => (
            decode_typed_many(&bounds.low, dtype, numel, f64::NEG_INFINITY),
            decode_typed_many(&bounds.high, dtype, numel, f64::INFINITY),
        ),
        Some(BoxBounds::Unbounded(_)) | None => {
            (vec![f64::NEG_INFINITY; numel], vec![f64::INFINITY; numel])
        }
    }
}

fn decode_typed_one(bytes: &[u8], dtype: DType) -> Option<f64> {
    decode_scalars(bytes, dtype)
        .ok()
        .and_then(|scalars| scalars.first().copied().map(|s| s.to_f64(dtype)))
}

fn decode_typed_many(bytes: &[u8], dtype: DType, numel: usize, default: f64) -> Vec<f64> {
    match decode_scalars(bytes, dtype) {
        Ok(scalars) if scalars.len() == numel => {
            scalars.into_iter().map(|s| s.to_f64(dtype)).collect()
        }
        _ => vec![default; numel],
    }
}

/// Validated elementwise bounds always carry one value per element; anything
/// else falls back to the unbounded default for sampling purposes.
fn elementwise_or_default(values: &[f64], len: usize, default: f64) -> Vec<f64> {
    if values.len() == len {
        values.to_vec()
    } else {
        vec![default; len]
    }
}

fn sample_box_scalar<R: rand::Rng + ?Sized>(
    rng: &mut R,
    low: f64,
    high: f64,
    dtype: DType,
) -> Result<Scalar, SpaceError> {
    if is_integer_dtype(dtype) {
        return sample_integer_box_scalar(rng, low, high, dtype);
    }

    let value = sample_continuous(rng, low, high)?;
    if matches!(dtype, DType::Bool) {
        return Ok(Scalar::Bool(value.round() != 0.0));
    }
    Ok(Scalar::Float(value))
}

/// Sample a continuous value within `[low, high]`, falling back to
/// exponential/normal sampling for half- and un-bounded ranges.
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

/// Sample an integer-dtype Box element with a uniform integer distribution
/// over `[ceil(low), floor(high)]`, matching Gymnasium instead of rounding a
/// continuous uniform (which biases the endpoints).
fn sample_integer_box_scalar<R: rand::Rng + ?Sized>(
    rng: &mut R,
    low: f64,
    high: f64,
    dtype: DType,
) -> Result<Scalar, SpaceError> {
    // Clamp the open ends to the dtype's representable range so an unbounded
    // integer Box still samples a finite value.
    let (dtype_lo, dtype_hi) = integer_dtype_bounds(dtype);
    let lo = if low.is_finite() {
        low.ceil()
    } else {
        dtype_lo
    };
    let hi = if high.is_finite() {
        high.floor()
    } else {
        dtype_hi
    };
    if lo > hi {
        return Err(SpaceError::invalid(
            "$",
            format!("cannot sample integer Box element with low {low} greater than high {high}"),
        ));
    }
    let lo = lo as i64;
    let hi = hi as i64;
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

/// Conservative finite sampling bounds for an unbounded integer Box element.
/// These stay well within `i64` so `random_range` cannot overflow.
fn integer_dtype_bounds(dtype: DType) -> (f64, f64) {
    match dtype {
        DType::Bool => (0.0, 1.0),
        DType::Uint8 => (0.0, u8::MAX as f64),
        DType::Int8 => (i8::MIN as f64, i8::MAX as f64),
        DType::Int16 => (i16::MIN as f64, i16::MAX as f64),
        DType::Uint16 => (0.0, u16::MAX as f64),
        DType::Int32 => (i32::MIN as f64, i32::MAX as f64),
        DType::Uint32 => (0.0, u32::MAX as f64),
        // Keep i64/u64 within a range that round-trips through f64 exactly.
        DType::Int64 => (-(1i64 << 53) as f64, (1i64 << 53) as f64),
        DType::Uint64 => (0.0, (1u64 << 53) as f64),
        _ => (-(1i64 << 53) as f64, (1i64 << 53) as f64),
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
        .map(|n| {
            if *n <= 0 {
                return Err(SpaceError::invalid(
                    "$",
                    "cannot sample MultiDiscrete space with a non-positive dimension",
                ));
            }
            Ok(rng.random_range(0..*n))
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
    // Draw in declared key order (the same order the value codec walks) so the
    // RNG stream is stable; the BTreeMap stores them sorted, which does not
    // affect the draw order.
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

/// The sampler treats `Unspecified` specs as float32, matching the codecs.
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

    // --- Determinism: same seed -> identical value, across every space kind. ---

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

            // Every sample is a full member of its space.
            assert!(
                contains(spec, &a).is_ok(),
                "sample must be a member of {spec}"
            );
        }
    }

    #[test]
    fn golden_samples_are_frozen() {
        // Freeze-forward golden: a fixed seed must produce these exact bytes for
        // all time. The chacha20 pin guarantees the RNG stream is stable across
        // `rand`/`chacha20` upgrades; this locks the *sampling math* on top of it
        // (integer/continuous Box draws, normal/exp tails, Text length+charset).
        // Captured under rand 0.10 / chacha20 0.10. If this fails, the sampling
        // algorithm changed — and the Python side, which runs this same code, has
        // changed with it. That is a deliberate, breaking decision, not a fixup.
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
    fn rng_stream_advances_across_calls() {
        // Seed once, sample twice off the *same* RNG: the second draw must come
        // from the advanced stream, not repeat the first (gym semantics). A
        // continuous unbounded Box has effectively zero collision probability, so
        // this also confirms the seed actually feeds the draw.
        let spec = BoxSpaceBuilder::unbounded(vec![4])
            .dtype(DType::Float64)
            .build()
            .unwrap();
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let first = sample_with(&spec, &mut rng).unwrap();
        let second = sample_with(&spec, &mut rng).unwrap();
        assert_ne!(first, second);
    }

    // --- Degenerate specs sample a single forced value, RNG-independent. ---

    #[test]
    fn degenerate_specs_are_rng_independent() {
        // Discrete(1): random_range(0..1) is always 0.
        assert_eq!(
            sample_seeded(&DiscreteBuilder::new(1).build().unwrap(), 1).unwrap(),
            SpaceValue::Discrete(0)
        );
        // MultiDiscrete with all-1 categories: every element is forced to 0.
        assert_eq!(
            sample_seeded(
                &MultiDiscreteBuilder::vector(vec![1, 1, 1]).build().unwrap(),
                1
            )
            .unwrap(),
            SpaceValue::MultiDiscrete(vec![0, 0, 0])
        );
        // A zero-width Box [5.0, 5.0] always samples 5.0.
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

    // --- Box leaves must land in 64-byte-aligned storage (DLPack zero-copy). ---

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

    // --- Bound decoding for sampling (moved from the PyO3 sampler). ---

    #[test]
    fn uint64_max_bound_decodes_to_positive_for_sampling() {
        let spec = typed_uniform_spec(0, -1, DType::Uint64);
        let (low, high) = box_bounds(&spec, 1, DType::Uint64);
        assert_eq!(low, vec![0.0]);
        assert_eq!(high, vec![u64::MAX as f64], "u64::MAX must decode positive");

        let mut rng = ChaCha12Rng::seed_from_u64(7);
        let sampled = sample_integer_box_scalar(&mut rng, low[0], high[0], DType::Uint64);
        assert!(
            sampled.is_ok(),
            "sampling a [0, u64::MAX] Uint64 box must not error: {sampled:?}"
        );
    }

    #[test]
    fn scalar_to_f64_matches_dtype() {
        // Uint64 -1 bits read as the large positive magnitude; Int64 stays -1.
        assert_eq!(Scalar::Int(-1).to_f64(DType::Uint64), u64::MAX as f64);
        assert_eq!(Scalar::Int(-1).to_f64(DType::Int64), -1.0);
    }

    // --- Invalid specs surface a SpaceError instead of panicking. ---

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
