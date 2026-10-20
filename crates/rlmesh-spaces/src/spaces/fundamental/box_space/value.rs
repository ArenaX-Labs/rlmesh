use half::f16;

use crate::BoxBounds;
use crate::dtype::{DType, dtype_size};
use crate::errors::{SpaceError, err_space};
use crate::scalar::{Scalar, ScalarError, decode_scalar};
use crate::spaces::{Conformance, SpaceKind, SpaceSpec, SpaceValue};

/// A low/high bound pair. `None` means "unbounded on this side" so integer
/// bounds never have to invent a sentinel that collides with a real value
/// (`i64::MIN`/`u64::MAX` are legitimate bounds).
type Bound = (Option<Scalar>, Option<Scalar>);

/// Elements per block of the vectorized scans: small enough to stay in L1,
/// large enough that the per-block branch is noise.
const BLOCK: usize = 4096;

pub(crate) fn conform_box(space: &SpaceSpec, value: &SpaceValue, path: &str) -> Conformance {
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
    if dtype == DType::Unspecified {
        // Tensor constructors reject Unspecified, so this cannot occur.
        return Conformance::Structural(SpaceError::invalid(
            path,
            "Box value dtype is unspecified",
        ));
    }

    let numel = tensor.numel();
    let bounds = match box_bounds(space, numel, dtype, path) {
        Ok(bounds) => bounds,
        Err(err) => return Conformance::Structural(err),
    };

    // Borrowed straight from storage for a contiguous tensor: the scans below
    // read the little-endian slab in place and allocate nothing.
    let data = tensor.to_contiguous_bytes();
    let elem = dtype_size(dtype);
    if !data.len().is_multiple_of(elem) {
        let err = ScalarError::ByteLengthMismatch {
            len: data.len(),
            elem,
            dtype,
        };
        return Conformance::Structural(SpaceError::invalid(
            path,
            format!("cannot decode Box value: {err}"),
        ));
    }

    // One scan. A NaN is never a member of any space, so it is a structural
    // rejection that outranks an out-of-bounds element found earlier in the
    // tensor; an out-of-bounds value is only a range deviation the serving side
    // may tolerate. Every element is still checked for NaN after a range
    // deviation is recorded, so a later NaN takes priority.
    let finding = match bounds {
        Bounds::Uniform(bound) => scan_uniform(&data, dtype, bound),
        Bounds::Elementwise { low, high } => scan_elementwise(&data, dtype, low, high),
        Bounds::TypedElementwise { low, high } => scan_typed_elementwise(&data, dtype, low, high),
    };

    match finding {
        None => Conformance::Ok,
        Some(Finding::Nan(index)) => Conformance::Structural(SpaceError::invalid(
            path,
            format!("Box value at element {index} is NaN"),
        )),
        Some(Finding::Range(index, side)) => Conformance::Range(SpaceError::invalid(
            path,
            format!(
                "Box value at element {index} out of bounds: {}",
                match side {
                    Side::Below => "below low bound",
                    Side::Above => "above high bound",
                }
            ),
        )),
    }
}

/// The first deviation a scan found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Finding {
    /// The first NaN element (structural).
    Nan(usize),
    /// The first out-of-bounds element, when there is no NaN (range).
    Range(usize, Side),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Below,
    Above,
}

/// A Box space's bounds, resolved and length-checked but never expanded to
/// one entry per element: per-element storage is read in place.
enum Bounds<'a> {
    /// The same bound for every element (uniform or absent bounds).
    Uniform(Bound),
    /// One `double` bound per element.
    Elementwise { low: &'a [f64], high: &'a [f64] },
    /// One dtype-typed little-endian bound per element.
    TypedElementwise { low: &'a [u8], high: &'a [u8] },
}

/// Resolve the space's bounds for every dtype, comparing in the dtype's native
/// domain. Float bounds carried as `double` (`Uniform`/`Elementwise`) are
/// `Scalar::Float`s; dtype-typed byte bounds are decoded with the space's
/// dtype so integer ranges stay exact.
fn box_bounds<'a>(
    space: &'a SpaceSpec,
    numel: usize,
    dtype: DType,
    path: &str,
) -> Result<Bounds<'a>, SpaceError> {
    let spec = match &space.spec {
        Some(SpaceKind::Box(spec)) => spec,
        _ => return err_space!(path, "space is not Box"),
    };

    Ok(match &spec.bounds {
        Some(BoxBounds::Uniform(bounds)) => {
            Bounds::Uniform((finite_float(bounds.low), finite_float(bounds.high)))
        }
        // Elementwise bounds carry one value per element; the validator
        // guarantees low.len() == high.len() == numel, so a length mismatch
        // here means an unvalidated/corrupt spec reached containment. Error
        // rather than silently cycling or truncating the bounds.
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
            Bounds::Elementwise {
                low: &bounds.low,
                high: &bounds.high,
            }
        }
        Some(BoxBounds::TypedUniform(bounds)) => {
            check_typed_bounds(&bounds.low, 1, dtype, path)?;
            check_typed_bounds(&bounds.high, 1, dtype, path)?;
            Bounds::Uniform((
                Some(decode_scalar(&bounds.low, dtype)),
                Some(decode_scalar(&bounds.high, dtype)),
            ))
        }
        Some(BoxBounds::TypedElementwise(bounds)) => {
            check_typed_bounds(&bounds.low, numel, dtype, path)?;
            check_typed_bounds(&bounds.high, numel, dtype, path)?;
            Bounds::TypedElementwise {
                low: &bounds.low,
                high: &bounds.high,
            }
        }
        Some(BoxBounds::Unbounded(_)) | None => Bounds::Uniform((None, None)),
    })
}

/// An infinite `double` bound means "unbounded on this side"; a finite one is
/// a real `Scalar::Float` comparison value.
fn finite_float(value: f64) -> Option<Scalar> {
    value.is_finite().then_some(Scalar::Float(value))
}

/// Typed bounds must hold exactly `count` whole elements of `dtype`.
fn check_typed_bounds(
    bytes: &[u8],
    count: usize,
    dtype: DType,
    path: &str,
) -> Result<(), SpaceError> {
    let elem = dtype_size(dtype);
    if !bytes.len().is_multiple_of(elem) {
        let err = ScalarError::ByteLengthMismatch {
            len: bytes.len(),
            elem,
            dtype,
        };
        return err_space!(path, format!("cannot decode typed Box bounds: {err}"));
    }
    let got = bytes.len() / elem;
    if got != count {
        return err_space!(
            path,
            format!("Box typed bounds length mismatch: expected {count} elements, got {got}")
        );
    }
    Ok(())
}

/// A bound shared by every element: the slab is read in its native type,
/// block by block, with branch-free comparisons the compiler can vectorize.
fn scan_uniform(data: &[u8], dtype: DType, (low, high): Bound) -> Option<Finding> {
    if dtype.is_float() {
        let (low, high) = (float_low(low, dtype), float_high(high, dtype));
        return match dtype {
            DType::Float16 => scan_float::<2>(data, read_f16, low, high),
            DType::Float32 => scan_float::<4>(data, read_f32, low, high),
            _ => scan_float::<8>(data, f64::from_le_bytes, low, high),
        };
    }
    let (low, high) = (int_low(low, dtype), int_high(high, dtype));
    match dtype {
        DType::Bool => scan_int::<u8, 1>(data, |[b]| u8::from(b != 0), (low, high), (0, 1)),
        DType::Uint8 => scan_int::<u8, 1>(data, |[b]| b, (low, high), range::<u8>()),
        DType::Int8 => scan_int::<i8, 1>(data, i8::from_le_bytes, (low, high), range::<i8>()),
        DType::Uint16 => scan_int::<u16, 2>(data, u16::from_le_bytes, (low, high), range::<u16>()),
        DType::Int16 => scan_int::<i16, 2>(data, i16::from_le_bytes, (low, high), range::<i16>()),
        DType::Uint32 => scan_int::<u32, 4>(data, u32::from_le_bytes, (low, high), range::<u32>()),
        DType::Int32 => scan_int::<i32, 4>(data, i32::from_le_bytes, (low, high), range::<i32>()),
        DType::Uint64 => scan_int::<u64, 8>(data, u64::from_le_bytes, (low, high), range::<u64>()),
        DType::Int64 => scan_int::<i64, 8>(data, i64::from_le_bytes, (low, high), range::<i64>()),
        _ => unreachable!("conform_box rejects an unspecified dtype"),
    }
}

/// An integer type's full range, in `i128`.
fn range<T: Bounded + Into<i128>>() -> (i128, i128) {
    (T::MIN.into(), T::MAX.into())
}

trait Bounded {
    const MIN: Self;
    const MAX: Self;
}

macro_rules! bounded {
    ($($t:ty),*) => {$(impl Bounded for $t {
        const MIN: Self = <$t>::MIN;
        const MAX: Self = <$t>::MAX;
    })*};
}
bounded!(u8, i8, u16, i16, u32, i32, u64, i64);

/// A float dtype's low bound as an `f64` threshold: an element is below it iff
/// `low > v`. Absent is `-inf` (never greater); a NaN bound stays NaN, which,
/// like `partial_cmp`, never compares greater.
fn float_low(low: Option<Scalar>, dtype: DType) -> f64 {
    low.map_or(f64::NEG_INFINITY, |low| low.to_f64(dtype))
}

/// A float dtype's high bound: an element is above it iff `v > high`.
fn float_high(high: Option<Scalar>, dtype: DType) -> f64 {
    high.map_or(f64::INFINITY, |high| high.to_f64(dtype))
}

/// An integer dtype's low bound in `i128`: an element `i` is below it iff
/// `i < low`. A float bound `f` compares exactly, so `i < f` iff
/// `i < ceil(f)` (saturating for out-of-range `f`); absent or NaN imposes no
/// bound.
fn int_low(low: Option<Scalar>, dtype: DType) -> i128 {
    match low {
        None => i128::MIN,
        Some(Scalar::Float(f)) if f.is_nan() => i128::MIN,
        Some(Scalar::Float(f)) => f.ceil() as i128,
        Some(int) => int_scalar(int, dtype),
    }
}

/// An integer dtype's high bound: `i` is above it iff `i > high`, and
/// `i > f` iff `i > floor(f)`.
fn int_high(high: Option<Scalar>, dtype: DType) -> i128 {
    match high {
        None => i128::MAX,
        Some(Scalar::Float(f)) if f.is_nan() => i128::MAX,
        Some(Scalar::Float(f)) => f.floor() as i128,
        Some(int) => int_scalar(int, dtype),
    }
}

/// An integral scalar's value in its dtype's domain (`Uint64` unsigned).
fn int_scalar(value: Scalar, dtype: DType) -> i128 {
    if dtype == DType::Uint64 {
        i128::from(value.as_u64())
    } else {
        i128::from(value.as_i64())
    }
}

/// Per-element `double` bounds (`Elementwise`), converted to the dtype's
/// comparison domain element by element exactly as a uniform bound is (see
/// [`float_low`] and [`int_low`]): an infinite bound is no bound.
fn scan_elementwise(data: &[u8], dtype: DType, low: &[f64], high: &[f64]) -> Option<Finding> {
    let float = |index: usize| {
        let (low, high) = (low[index], high[index]);
        (
            if low.is_finite() {
                low
            } else {
                f64::NEG_INFINITY
            },
            if high.is_finite() {
                high
            } else {
                f64::INFINITY
            },
        )
    };
    let int = |index: usize| {
        let (low, high) = (low[index], high[index]);
        (
            if low.is_finite() {
                low.ceil() as i128
            } else {
                i128::MIN
            },
            if high.is_finite() {
                high.floor() as i128
            } else {
                i128::MAX
            },
        )
    };
    match dtype {
        DType::Float16 => scan_at::<_, 2>(data, read_f16, float),
        DType::Float32 => scan_at::<_, 4>(data, read_f32, float),
        DType::Float64 => scan_at::<_, 8>(data, f64::from_le_bytes, float),
        DType::Bool => scan_at::<_, 1>(data, |[b]| i128::from(b != 0), int),
        DType::Uint8 => scan_at::<_, 1>(data, |[b]| i128::from(b), int),
        DType::Int8 => scan_at::<_, 1>(data, |b| i128::from(i8::from_le_bytes(b)), int),
        DType::Uint16 => scan_at::<_, 2>(data, |b| i128::from(u16::from_le_bytes(b)), int),
        DType::Int16 => scan_at::<_, 2>(data, |b| i128::from(i16::from_le_bytes(b)), int),
        DType::Uint32 => scan_at::<_, 4>(data, |b| i128::from(u32::from_le_bytes(b)), int),
        DType::Int32 => scan_at::<_, 4>(data, |b| i128::from(i32::from_le_bytes(b)), int),
        DType::Uint64 => scan_at::<_, 8>(data, |b| i128::from(u64::from_le_bytes(b)), int),
        DType::Int64 => scan_at::<_, 8>(data, |b| i128::from(i64::from_le_bytes(b)), int),
        DType::Unspecified => unreachable!("conform_box rejects an unspecified dtype"),
    }
}

/// Per-element typed bounds (`TypedElementwise`): bounds share the value's
/// dtype, so every comparison is native (`Uint64` unsigned, `Bool` as 0/1,
/// floats with IEEE semantics, where a NaN bound never compares greater).
fn scan_typed_elementwise(data: &[u8], dtype: DType, low: &[u8], high: &[u8]) -> Option<Finding> {
    fn typed<V: PartialOrd + Copy, const N: usize>(
        data: &[u8],
        low: &[u8],
        high: &[u8],
        read: impl Fn([u8; N]) -> V + Copy,
    ) -> Option<Finding> {
        let (low, _) = low.as_chunks::<N>();
        let (high, _) = high.as_chunks::<N>();
        scan_at(data, read, |index| (read(low[index]), read(high[index])))
    }
    match dtype {
        DType::Float16 => typed(data, low, high, read_f16),
        DType::Float32 => typed(data, low, high, read_f32),
        DType::Float64 => typed(data, low, high, f64::from_le_bytes),
        DType::Bool => typed(data, low, high, |[b]: [u8; 1]| b != 0),
        DType::Uint8 => typed(data, low, high, |[b]: [u8; 1]| b),
        DType::Int8 => typed(data, low, high, i8::from_le_bytes),
        DType::Uint16 => typed(data, low, high, u16::from_le_bytes),
        DType::Int16 => typed(data, low, high, i16::from_le_bytes),
        DType::Uint32 => typed(data, low, high, u32::from_le_bytes),
        DType::Int32 => typed(data, low, high, i32::from_le_bytes),
        DType::Uint64 => typed(data, low, high, u64::from_le_bytes),
        DType::Int64 => typed(data, low, high, i64::from_le_bytes),
        DType::Unspecified => unreachable!("conform_box rejects an unspecified dtype"),
    }
}

fn read_f16(bytes: [u8; 2]) -> f64 {
    f16::from_le_bytes(bytes).to_f64()
}

fn read_f32(bytes: [u8; 4]) -> f64 {
    f64::from(f32::from_le_bytes(bytes))
}

/// Elements against per-element `(low, high)` thresholds: below iff
/// `low > v`, above iff `v > high`, NaN (a value unordered with itself)
/// structural and first-wins over any range deviation.
fn scan_at<V: PartialOrd + Copy, const N: usize>(
    data: &[u8],
    read: impl Fn([u8; N]) -> V,
    bound_at: impl Fn(usize) -> (V, V),
) -> Option<Finding> {
    let (elements, _) = data.as_chunks::<N>();
    let mut range = None;
    for (index, &bytes) in elements.iter().enumerate() {
        let v = read(bytes);
        if v.partial_cmp(&v).is_none() {
            return Some(Finding::Nan(index));
        }
        if range.is_none() {
            let (low, high) = bound_at(index);
            if low > v {
                range = Some(Finding::Range(index, Side::Below));
            } else if v > high {
                range = Some(Finding::Range(index, Side::Above));
            }
        }
    }
    range
}

fn scan_float<const N: usize>(
    data: &[u8],
    read: impl Fn([u8; N]) -> f64 + Copy,
    low: f64,
    high: f64,
) -> Option<Finding> {
    let (elements, _) = data.as_chunks::<N>();
    let below = |v: f64| low > v;
    let above = |v: f64| v > high;
    let mut range = None;
    for (block_index, block) in elements.chunks(BLOCK).enumerate() {
        let base = block_index * BLOCK;
        // Branch-free reduction over the block; locate only on a hit.
        let (nan, out) = block.iter().fold((false, false), |(nan, out), &bytes| {
            let v = read(bytes);
            (nan | v.is_nan(), out | below(v) | above(v))
        });
        if nan {
            let offset = block.iter().position(|&bytes| read(bytes).is_nan());
            return Some(Finding::Nan(base + offset.unwrap_or_default()));
        }
        if out && range.is_none() {
            range = block.iter().enumerate().find_map(|(offset, &bytes)| {
                let v = read(bytes);
                let side = if below(v) {
                    Side::Below
                } else if above(v) {
                    Side::Above
                } else {
                    return None;
                };
                Some(Finding::Range(base + offset, side))
            });
        }
    }
    range
}

/// `low`/`high` are `i128` thresholds (`i < low` is below, `i > high` above);
/// `min`/`max` are the dtype's range. Integers cannot be NaN, so the first
/// deviation is the answer.
fn scan_int<T, const N: usize>(
    data: &[u8],
    read: impl Fn([u8; N]) -> T + Copy,
    (low, high): (i128, i128),
    (min, max): (i128, i128),
) -> Option<Finding>
where
    T: Copy + Ord + Into<i128> + TryFrom<i128>,
{
    let (elements, _) = data.as_chunks::<N>();
    let first = elements.first()?;
    // Bounds that cover the whole dtype: no element can deviate (e.g. a uint8
    // image in 0..=255), so the shape and length checks already decided it.
    if low <= min && high >= max {
        return None;
    }
    // A bound past the dtype's far end: every element deviates, so the first
    // one is the answer.
    if low > max || high < min {
        let side = if read(*first).into() < low {
            Side::Below
        } else {
            Side::Above
        };
        return Some(Finding::Range(0, side));
    }
    // Both thresholds now clamp into the dtype exactly: for `min <= v <= max`,
    // `v < low` iff `v < max(low, min)`, and likewise for `high`.
    let (Ok(low), Ok(high)) = (T::try_from(low.max(min)), T::try_from(high.min(max))) else {
        unreachable!("clamped bounds lie in the dtype's range");
    };
    for (block_index, block) in elements.chunks(BLOCK).enumerate() {
        let out = block.iter().fold(false, |out, &bytes| {
            let v = read(bytes);
            out | (v < low) | (v > high)
        });
        if out {
            let base = block_index * BLOCK;
            return block.iter().enumerate().find_map(|(offset, &bytes)| {
                let v = read(bytes);
                let side = if v < low {
                    Side::Below
                } else if v > high {
                    Side::Above
                } else {
                    return None;
                };
                Some(Finding::Range(base + offset, side))
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::fundamental::BoxSpaceBuilder;
    use crate::spaces::{Conformance, conform, contains};
    use crate::tensor::{Storage, Tensor};

    fn box_space(low: f64, high: f64, shape: Vec<i64>, dtype: DType) -> SpaceSpec {
        BoxSpaceBuilder::scalar(low, high, shape)
            .dtype(dtype)
            .build()
            .expect("valid space")
    }

    #[test]
    fn test_box_contains() {
        let space = box_space(0.0, 1.0, vec![2, 3], DType::Float32);

        let valid = SpaceValue::Box(
            Tensor::from_vec(vec![0u8; 24], vec![2, 3], DType::Float32).expect("valid tensor"),
        );
        assert!(contains(&space, &valid).is_ok());

        let wrong_shape = SpaceValue::Box(
            Tensor::from_vec(vec![0u8; 24], vec![3, 2], DType::Float32).expect("valid tensor"),
        );
        assert!(contains(&space, &wrong_shape).is_err());

        let wrong_dtype = SpaceValue::Box(
            Tensor::from_vec(vec![0u8; 48], vec![2, 3], DType::Float64).expect("valid tensor"),
        );
        assert!(contains(&space, &wrong_dtype).is_err());
    }

    #[test]
    fn test_box_contains_rejects_out_of_bounds_values() {
        let space = box_space(0.0, 1.0, vec![2], DType::Float32);

        let data: Vec<u8> = [0.5f32, 2.5f32]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let invalid =
            SpaceValue::Box(Tensor::from_vec(data, vec![2], DType::Float32).expect("valid tensor"));

        assert!(contains(&space, &invalid).is_err());
    }

    #[test]
    fn test_box_conform_classifies_deviations() {
        let space = box_space(0.0, 1.0, vec![2], DType::Float32);

        // Out of bounds is a Range deviation the serving side may tolerate.
        let oob: Vec<u8> = [0.5f32, 2.5].iter().flat_map(|v| v.to_le_bytes()).collect();
        let oob = SpaceValue::Box(Tensor::from_vec(oob, vec![2], DType::Float32).expect("tensor"));
        assert!(matches!(conform(&space, &oob), Conformance::Range(_)));

        // NaN is Structural even when another element is also out of bounds, so a
        // NaN is never masked by a range deviation found earlier in the scan.
        let nan_and_oob: Vec<u8> = [2.5f32, f32::NAN]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let nan_and_oob = SpaceValue::Box(
            Tensor::from_vec(nan_and_oob, vec![2], DType::Float32).expect("tensor"),
        );
        assert!(matches!(
            conform(&space, &nan_and_oob),
            Conformance::Structural(_)
        ));
    }

    #[test]
    fn test_box_contains_rejects_nan_even_when_unbounded() {
        // An unbounded Box accepts any finite value but never NaN: NaN is not a
        // member of any space, so it is rejected regardless of bounds.
        let space = box_space(f64::NEG_INFINITY, f64::INFINITY, vec![2], DType::Float32);
        let data: Vec<u8> = [f32::NAN, 0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let value =
            SpaceValue::Box(Tensor::from_vec(data, vec![2], DType::Float32).expect("tensor"));
        assert!(contains(&space, &value).is_err());
        assert!(matches!(
            conform(&space, &value),
            Conformance::Structural(_)
        ));
    }

    #[test]
    fn test_box_contains_rejects_nan_for_bounded_float_values() {
        let space = box_space(0.0, 1.0, vec![2], DType::Float32);

        let data: Vec<u8> = [f32::NAN, 0.5]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let invalid =
            SpaceValue::Box(Tensor::from_vec(data, vec![2], DType::Float32).expect("valid tensor"));

        assert!(contains(&space, &invalid).is_err());
    }

    #[test]
    fn test_box_contains_new_int_dtypes() {
        let space = box_space(-100.0, 100.0, vec![2], DType::Int16);
        let in_bounds: Vec<u8> = [(-50i16), 99]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let valid = SpaceValue::Box(
            Tensor::from_vec(in_bounds, vec![2], DType::Int16).expect("valid tensor"),
        );
        assert!(contains(&space, &valid).is_ok());

        let out_of_bounds: Vec<u8> = [(-50i16), 101]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let invalid = SpaceValue::Box(
            Tensor::from_vec(out_of_bounds, vec![2], DType::Int16).expect("valid tensor"),
        );
        assert!(contains(&space, &invalid).is_err());

        let space = box_space(0.0, 70000.0, vec![1], DType::Uint32);
        let value = SpaceValue::Box(
            Tensor::from_vec(65536u32.to_le_bytes().to_vec(), vec![1], DType::Uint32)
                .expect("valid tensor"),
        );
        assert!(contains(&space, &value).is_ok());
    }

    #[test]
    fn test_box_contains_rejects_mismatched_elementwise_bounds() {
        use crate::{BoxSpec, ElementwiseBounds};

        // A hand-built (unvalidated) spec whose elementwise bounds carry fewer
        // values than the tensor has elements. Containment must reject this
        // rather than silently cycling the bounds to fit.
        let space = SpaceSpec {
            shape: vec![3],
            dtype: DType::Float32,
            spec: Some(SpaceKind::Box(BoxSpec {
                bounds: Some(BoxBounds::Elementwise(ElementwiseBounds {
                    low: vec![0.0],
                    high: vec![1.0],
                })),
            })),
        };

        let value = SpaceValue::Box(
            Tensor::from_vec(vec![0u8; 12], vec![3], DType::Float32).expect("valid tensor"),
        );
        assert!(contains(&space, &value).is_err());
    }

    #[test]
    fn test_strided_view_passes_contains() {
        // Storage holds [0.5, 9.0, 0.5, 9.0]; the stride-2 view sees only
        // [0.5, 0.5], so the out-of-bounds 9.0s must not be inspected.
        let data: Vec<u8> = [0.5f32, 9.0, 0.5, 9.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let storage = Storage::from_slice(&data);
        let view = Tensor::from_storage(storage, DType::Float32, vec![2], Some(vec![2]), 0)
            .expect("valid tensor");
        assert!(!view.is_contiguous());

        let space = box_space(0.0, 1.0, vec![2], DType::Float32);
        assert!(contains(&space, &SpaceValue::Box(view)).is_ok());
    }

    fn i64_box(low: i64, high: i64, shape: Vec<i64>, dtype: DType) -> SpaceSpec {
        BoxSpaceBuilder::int_scalar(low, high, shape)
            .dtype(dtype)
            .build()
            .expect("valid space")
    }

    #[test]
    fn test_box_contains_int64_extreme_bounds_are_exact() {
        // Bounds at the very top of the i64 range: 2^63 - 2 .. 2^63 - 1.
        // An f64 round-trip would round both to 2^63 and accept 2^63 - 1 even
        // when it is the only in-bounds value, or reject it entirely. The
        // native-domain comparison keeps the one-ULP distinction exact.
        let space = i64_box(i64::MAX - 1, i64::MAX, vec![3], DType::Int64);

        let inside: Vec<u8> = [i64::MAX - 1, i64::MAX, i64::MAX - 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert!(
            contains(
                &space,
                &SpaceValue::Box(Tensor::from_vec(inside, vec![3], DType::Int64).expect("tensor")),
            )
            .is_ok()
        );

        // i64::MAX - 2 is just below the low bound and must be rejected even
        // though it is f64-indistinguishable from the bound.
        let below: Vec<u8> = [i64::MAX - 2, i64::MAX, i64::MAX]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert!(
            contains(
                &space,
                &SpaceValue::Box(Tensor::from_vec(below, vec![3], DType::Int64).expect("tensor")),
            )
            .is_err()
        );
    }

    #[test]
    fn test_box_contains_int64_min_bound() {
        let space = i64_box(i64::MIN, i64::MIN + 1, vec![2], DType::Int64);

        let inside: Vec<u8> = [i64::MIN, i64::MIN + 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert!(
            contains(
                &space,
                &SpaceValue::Box(Tensor::from_vec(inside, vec![2], DType::Int64).expect("tensor")),
            )
            .is_ok()
        );

        // i64::MIN + 2 is above the high bound.
        let above: Vec<u8> = [i64::MIN, i64::MIN + 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert!(
            contains(
                &space,
                &SpaceValue::Box(Tensor::from_vec(above, vec![2], DType::Int64).expect("tensor")),
            )
            .is_err()
        );
    }

    #[test]
    fn test_box_contains_uint64_max_bound() {
        // u64::MAX as a bound, compared in the unsigned domain. The largest
        // value (u64::MAX) is in bounds; u64::MAX as i64 is -1, so an i64
        // comparison would wrongly reject it.
        let space = BoxSpaceBuilder::uint_scalar(0, u64::MAX, vec![1])
            .build()
            .expect("valid space");

        let top: Vec<u8> = u64::MAX.to_le_bytes().to_vec();
        assert!(
            contains(
                &space,
                &SpaceValue::Box(Tensor::from_vec(top, vec![1], DType::Uint64).expect("tensor")),
            )
            .is_ok()
        );

        // A mid-range value is also in bounds.
        let mid: Vec<u8> = (1u64 << 63).to_le_bytes().to_vec();
        assert!(
            contains(
                &space,
                &SpaceValue::Box(Tensor::from_vec(mid, vec![1], DType::Uint64).expect("tensor")),
            )
            .is_ok()
        );
    }

    #[test]
    fn test_box_contains_typed_elementwise_bounds() {
        let space = BoxSpaceBuilder::int_tensor(vec![0, 100], vec![10, i64::MAX], vec![2])
            .dtype(DType::Int64)
            .build()
            .expect("valid space");

        let inside: Vec<u8> = [5i64, i64::MAX]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert!(
            contains(
                &space,
                &SpaceValue::Box(Tensor::from_vec(inside, vec![2], DType::Int64).expect("tensor")),
            )
            .is_ok()
        );

        // Second element below its low bound (100).
        let below: Vec<u8> = [5i64, 99].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(
            contains(
                &space,
                &SpaceValue::Box(Tensor::from_vec(below, vec![2], DType::Int64).expect("tensor")),
            )
            .is_err()
        );
    }

    #[test]
    fn test_box_contains_float_bound_uint64_dtype_no_truncation() {
        let space = box_space(-1.0, 100.0, vec![1], DType::Uint64);
        for v in [0u64, 1, 50, 100] {
            let value = SpaceValue::Box(
                Tensor::from_vec(v.to_le_bytes().to_vec(), vec![1], DType::Uint64).expect("tensor"),
            );
            assert!(
                contains(&space, &value).is_ok(),
                "u64 {v} must be in [-1.0, 100.0]"
            );
        }
        // 101 is above the high bound.
        let above = SpaceValue::Box(
            Tensor::from_vec(101u64.to_le_bytes().to_vec(), vec![1], DType::Uint64)
                .expect("tensor"),
        );
        assert!(contains(&space, &above).is_err());
    }

    #[test]
    fn test_box_rejects_fractional_float_bound_on_int_dtype() {
        // A fractional float-form bound on an integer dtype cannot be packed
        // onto the bytes-only wire without truncation (0.5 -> 0), which would
        // silently widen the space's admitted set. Reject it at construction.
        let err = BoxSpaceBuilder::scalar(0.5, 10.0, vec![1])
            .dtype(DType::Int64)
            .build()
            .expect_err("fractional int-dtype bound must be rejected");
        assert!(
            err.to_string().contains("finite whole number"),
            "unexpected error: {err}"
        );
        // An infinite bound on an integer dtype is rejected for the same reason
        // (it would saturate to the dtype min/max).
        assert!(
            BoxSpaceBuilder::scalar(f64::NEG_INFINITY, 10.0, vec![1])
                .dtype(DType::Int64)
                .build()
                .is_err()
        );

        // Integral float bounds round-trip exactly and remain valid.
        let space = box_space(0.0, 10.0, vec![1], DType::Int64);
        let one = SpaceValue::Box(
            Tensor::from_vec(1i64.to_le_bytes().to_vec(), vec![1], DType::Int64).expect("tensor"),
        );
        assert!(contains(&space, &one).is_ok());
        let neg = SpaceValue::Box(
            Tensor::from_vec((-1i64).to_le_bytes().to_vec(), vec![1], DType::Int64)
                .expect("tensor"),
        );
        assert!(contains(&space, &neg).is_err(), "-1 < 0 must be rejected");
        let eleven = SpaceValue::Box(
            Tensor::from_vec(11i64.to_le_bytes().to_vec(), vec![1], DType::Int64).expect("tensor"),
        );
        assert!(
            contains(&space, &eleven).is_err(),
            "11 > 10 must be rejected"
        );
    }

    #[test]
    fn test_validate_rejects_typed_bounds_byte_length_mismatch() {
        use crate::{BoxSpec, TypedUniformBounds};

        // A Uint64 typed-uniform bound whose `low` is only 4 bytes (half a
        // scalar) must be rejected by validation.
        let space = SpaceSpec {
            shape: vec![1],
            dtype: DType::Uint64,
            spec: Some(SpaceKind::Box(BoxSpec {
                bounds: Some(BoxBounds::TypedUniform(TypedUniformBounds {
                    low: vec![0u8; 4],
                    high: vec![0u8; 8],
                })),
            })),
        };
        assert!(crate::spaces::validate_space(&space).is_err());
    }
}
