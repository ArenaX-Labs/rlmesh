use crate::dtype::dtype_size;
use crate::errors::{SpaceError, err_space};
use crate::scalar::{
    Scalar, check_int_in_dtype_range, check_uint_in_dtype_range, decode_scalars,
    encode_i64_scalars, encode_scalars,
};
use crate::spaces::{SpaceKind, SpaceSpec};
use crate::{
    BoxBounds, BoxSpec, DType, ElementwiseBounds, TypedElementwiseBounds, TypedUniformBounds,
    UniformBounds,
};

/// Bounds requested at the builder, before the dtype is known. Integer
/// entry points defer byte encoding until [`BoxSpaceBuilder::build`], when the
/// dtype has been selected.
enum PendingBounds {
    Ready(BoxBounds),
    IntUniform { low: i64, high: i64 },
    IntTensor { low: Vec<i64>, high: Vec<i64> },
    UintUniform { low: u64, high: u64 },
}

#[must_use = "a space builder does nothing until .build() is called"]
pub struct BoxSpaceBuilder {
    shape: Vec<i64>,
    dtype: DType,
    bounds: PendingBounds,
}

impl BoxSpaceBuilder {
    pub fn unbounded(shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::Float32,
            bounds: PendingBounds::Ready(BoxBounds::Unbounded(true)),
        }
    }

    pub fn scalar(low: f64, high: f64, shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::Float32,
            bounds: PendingBounds::Ready(BoxBounds::Uniform(UniformBounds { low, high })),
        }
    }

    pub fn tensor(low: Vec<f64>, high: Vec<f64>, shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::Float32,
            bounds: PendingBounds::Ready(BoxBounds::Elementwise(ElementwiseBounds { low, high })),
        }
    }

    /// A uniform integer bound pair, carried as dtype-typed bytes so values
    /// such as `i64::MAX`/`u64::MAX` round-trip exactly. The dtype defaults to
    /// `Float32`; set an integer dtype with [`BoxSpaceBuilder::dtype`] before
    /// building.
    pub fn int_scalar(low: i64, high: i64, shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::Int64,
            bounds: PendingBounds::IntUniform { low, high },
        }
    }

    /// Per-element integer bounds (row-major), carried as dtype-typed bytes.
    pub fn int_tensor(low: Vec<i64>, high: Vec<i64>, shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::Int64,
            bounds: PendingBounds::IntTensor { low, high },
        }
    }

    /// A uniform unsigned-integer bound pair, carried as dtype-typed bytes so
    /// values up to `u64::MAX` round-trip exactly. Defaults to `Uint64`.
    pub fn uint_scalar(low: u64, high: u64, shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::Uint64,
            bounds: PendingBounds::UintUniform { low, high },
        }
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn build(self) -> Result<SpaceSpec, SpaceError> {
        let dtype = self.dtype;
        let bounds = match self.bounds {
            PendingBounds::Ready(bounds) => bounds,
            PendingBounds::IntUniform { low, high } => {
                BoxBounds::TypedUniform(TypedUniformBounds {
                    low: encode_int_bound(&[low], dtype)?,
                    high: encode_int_bound(&[high], dtype)?,
                })
            }
            PendingBounds::IntTensor { low, high } => {
                BoxBounds::TypedElementwise(TypedElementwiseBounds {
                    low: encode_int_bound(&low, dtype)?,
                    high: encode_int_bound(&high, dtype)?,
                })
            }
            PendingBounds::UintUniform { low, high } => {
                BoxBounds::TypedUniform(TypedUniformBounds {
                    low: encode_uint_bound(&[low], dtype)?,
                    high: encode_uint_bound(&[high], dtype)?,
                })
            }
        };
        let spec = SpaceSpec {
            shape: self.shape,
            dtype,
            spec: Some(SpaceKind::Box(BoxSpec {
                bounds: Some(bounds),
            })),
        };
        crate::spaces::validate_space(&spec)?;
        Ok(spec)
    }
}

/// Encode signed-integer bounds, failing fast if any value falls outside the
/// dtype's exact integer range rather than silently wrapping (which would build
/// a "valid" space whose bounds differ from what the caller requested).
fn encode_int_bound(values: &[i64], dtype: DType) -> Result<Vec<u8>, SpaceError> {
    for &value in values {
        check_int_in_dtype_range(value, dtype).map_err(bound_encode_error)?;
    }
    encode_i64_scalars(values, dtype).map_err(bound_encode_error)
}

/// Encode unsigned-integer bounds, failing fast on out-of-range values. Integer
/// dtypes preserve the bit pattern through the `i64` codec: for `Uint64`,
/// `u64::MAX` re-encodes to all-ones bytes; smaller unsigned dtypes fit `i64`
/// after the range check. Float dtypes encode the numeric `u64` value directly.
fn encode_uint_bound(values: &[u64], dtype: DType) -> Result<Vec<u8>, SpaceError> {
    if matches!(dtype, DType::Float16 | DType::Float32 | DType::Float64) {
        let mut scalars = Vec::with_capacity(values.len());
        for &value in values {
            check_uint_in_dtype_range(value, dtype).map_err(bound_encode_error)?;
            scalars.push(Scalar::Float(value as f64));
        }
        return encode_scalars(&scalars, dtype).map_err(bound_encode_error);
    }

    let mut signed = Vec::with_capacity(values.len());
    for &value in values {
        check_uint_in_dtype_range(value, dtype).map_err(bound_encode_error)?;
        signed.push(value as i64);
    }
    encode_i64_scalars(&signed, dtype).map_err(bound_encode_error)
}

fn bound_encode_error(err: crate::scalar::ScalarError) -> SpaceError {
    SpaceError::Invalid {
        path: "Box".to_string(),
        msg: format!("cannot encode integer Box bounds: {err}"),
    }
}

/// Which side of a Box bound pair is being validated. The saturating wire cast
/// is asymmetric, so the range check depends on the side (see
/// [`reject_non_integral_int_bound`]).
#[derive(Clone, Copy)]
enum BoundSide {
    Low,
    High,
}

impl BoundSide {
    fn label(self) -> &'static str {
        match self {
            BoundSide::Low => "low",
            BoundSide::High => "high",
        }
    }
}

/// A float-form Box bound on an integer dtype is packed onto the wire by a
/// saturating `f64 as iN` cast. That cast silently corrupts a bound in three
/// ways, each of which this guard rejects at construction so the frozen wire
/// form only ever carries a faithfully representable value:
///
/// - a fractional bound (`0.5` -> `0`) or a non-finite one (`inf` clamps to the
///   dtype min/max) — rejected for either side;
/// - a `low` *above* the dtype max, which saturates DOWN to the max and turns an
///   empty admitted set into `{max}` (`scalar(300.0, 400.0).dtype(Uint8)` would
///   wire as `[255, 255]`, advertising `{255}` while the local space admits
///   nothing) — rejected;
/// - a `high` *below* the dtype min, which saturates UP to the min and turns an
///   empty set into `{min}` — rejected.
///
/// A *vacuous* out-of-range bound saturates harmlessly and is allowed: a `low`
/// below the dtype min (e.g. `low = -1.0` on an unsigned dtype, the established
/// "no lower bound" idiom) or a `high` above the dtype max both clamp to a value
/// that leaves the admitted set unchanged. An integral, finite, in-range float
/// (e.g. `0.0`/`1.0` on `Uint8`) round-trips exactly and is allowed.
fn reject_non_integral_int_bound(
    dtype: DType,
    value: f64,
    side: BoundSide,
    path: &str,
) -> Result<(), SpaceError> {
    if !dtype.is_integer() {
        return Ok(());
    }
    let name = side.label();
    if !value.is_finite() || value.fract() != 0.0 {
        return err_space!(
            path,
            "Box",
            format!("integer-dtype bound {name}={value} must be a finite whole number")
        );
    }
    // Only saturation that CHANGES the admitted set is a silent contract change.
    // The range is `[min, max_exclusive)` where `max_exclusive == true_max + 1`.
    // The low side tests `value >= max_exclusive` rather than `value > true_max`
    // because `i64::MAX`/`u64::MAX` are NOT representable in f64 and round UP to
    // 2^63/2^64: a `low` sitting at that rounded max would slip a `> max` test
    // and then saturate to `{max}` on the wire. `max_exclusive` is a power of two
    // (`2^w`/`2^(w-1)`), exact in f64, so `>=` is precise at the 64-bit edge.
    // (Exact 64-bit bounds belong on the typed `int_scalar`/`uint_scalar`
    // builders, which carry exact dtype bytes instead of an f64.)
    let (min, max_exclusive) = int_dtype_f64_bounds(dtype);
    let changes_admitted_set = match side {
        BoundSide::Low => value >= max_exclusive,
        BoundSide::High => value < min,
    };
    if changes_admitted_set {
        return err_space!(
            path,
            "Box",
            format!(
                "integer-dtype bound {name}={value} is not representable in {dtype} \
                 and would saturate to a different admitted set"
            )
        );
    }
    Ok(())
}

/// Range of an integer dtype as `(inclusive_min, exclusive_upper)` in `f64`,
/// where `exclusive_upper == true_max + 1`. Both endpoints are powers of two
/// (`0`/`-2^(w-1)` and `2^w`/`2^(w-1)`), hence exact in `f64` — unlike
/// `i64::MAX`/`u64::MAX`, which round up. Only integer dtypes reach here (callers
/// gate on [`DType::is_integer`]); other dtypes carry no exact-integer
/// constraint.
fn int_dtype_f64_bounds(dtype: DType) -> (f64, f64) {
    match dtype {
        DType::Uint8 => (0.0, 2.0_f64.powi(8)),
        DType::Uint16 => (0.0, 2.0_f64.powi(16)),
        DType::Uint32 => (0.0, 2.0_f64.powi(32)),
        DType::Uint64 => (0.0, 2.0_f64.powi(64)),
        DType::Int8 => (-(2.0_f64.powi(7)), 2.0_f64.powi(7)),
        DType::Int16 => (-(2.0_f64.powi(15)), 2.0_f64.powi(15)),
        DType::Int32 => (-(2.0_f64.powi(31)), 2.0_f64.powi(31)),
        DType::Int64 => (-(2.0_f64.powi(63)), 2.0_f64.powi(63)),
        _ => (f64::NEG_INFINITY, f64::INFINITY),
    }
}

pub(crate) fn validate_box_at(space: &SpaceSpec, path: &str) -> Result<(), SpaceError> {
    if space.shape.is_empty() {
        return err_space!(path, "Box", "shape must be set (rank >= 1)");
    }

    if space.dtype == DType::Unspecified {
        return err_space!(path, "Box", "dtype must be set");
    }

    for (i, &d) in space.shape.iter().enumerate() {
        if d <= 0 {
            return err_space!(path, "Box", format!("shape[{i}] must be > 0"));
        }
    }

    let b = match &space.spec {
        Some(SpaceKind::Box(b)) => b,
        _ => return err_space!(path, "Box", "spec.box must be set"),
    };

    let numel: usize = space
        .shape
        .iter()
        .try_fold(1usize, |acc, &d| (d as usize).checked_mul(acc))
        .ok_or_else(|| SpaceError::Invalid {
            path: path.to_string(),
            msg: "Box.shape product overflowed".to_string(),
        })?;

    match &b.bounds {
        Some(BoxBounds::Unbounded(_)) => Ok(()),

        Some(BoxBounds::Uniform(s)) => {
            // NaN slips past `low > high` (every NaN comparison is false), then
            // reads as "unbounded" at conformance -- reject it as a corrupt spec.
            if s.low.is_nan() || s.high.is_nan() {
                return err_space!(path, "Box", "scalar bounds invalid: NaN bound");
            }
            if s.low > s.high {
                return err_space!(path, "Box", "scalar bounds invalid: low > high");
            }
            reject_non_integral_int_bound(space.dtype, s.low, BoundSide::Low, path)?;
            reject_non_integral_int_bound(space.dtype, s.high, BoundSide::High, path)?;
            Ok(())
        }

        // elementwise / tensor: len == numel
        Some(BoxBounds::Elementwise(t)) => {
            if t.low.len() != t.high.len() {
                return err_space!(
                    path,
                    "Box",
                    "tensor bounds invalid: low/high length mismatch"
                );
            }
            if t.low.len() != numel {
                return err_space!(
                    path,
                    "Box",
                    format!("tensor bounds invalid: expected length {numel}")
                );
            }
            for i in 0..numel {
                if t.low[i].is_nan() || t.high[i].is_nan() {
                    return err_space!(
                        path,
                        "Box",
                        format!("tensor bounds invalid: NaN bound at element {i}")
                    );
                }
                if t.low[i] > t.high[i] {
                    return err_space!(
                        path,
                        "Box",
                        format!("tensor bounds invalid: low>high at element {i}")
                    );
                }
                reject_non_integral_int_bound(space.dtype, t.low[i], BoundSide::Low, path)?;
                reject_non_integral_int_bound(space.dtype, t.high[i], BoundSide::High, path)?;
            }
            Ok(())
        }

        // dtype-typed uniform: one scalar each, dtype-sized.
        Some(BoxBounds::TypedUniform(t)) => {
            validate_typed_bounds(&t.low, &t.high, 1, space.dtype, path)
        }

        // dtype-typed elementwise: numel scalars each, dtype-sized.
        Some(BoxBounds::TypedElementwise(t)) => {
            validate_typed_bounds(&t.low, &t.high, numel, space.dtype, path)
        }

        None => err_space!(path, "Box", "bounds must be set"),
    }
}

/// Validate dtype-typed Box bounds: byte length must equal
/// `count * dtype_size(dtype)`, the dtype must be representable, and each
/// `low <= high` comparison runs in the dtype's native domain (integers
/// compare as integers, floats as floats) so no precision is lost.
fn validate_typed_bounds(
    low: &[u8],
    high: &[u8],
    count: usize,
    dtype: DType,
    path: &str,
) -> Result<(), SpaceError> {
    if dtype == DType::Unspecified {
        return err_space!(path, "Box", "typed bounds require a concrete dtype");
    }
    let elem = dtype_size(dtype);
    let expected = count.checked_mul(elem).ok_or_else(|| SpaceError::Invalid {
        path: path.to_string(),
        msg: "Box typed bounds length overflowed".to_string(),
    })?;
    if low.len() != expected || high.len() != expected {
        return err_space!(
            path,
            "Box",
            format!(
                "typed bounds invalid: expected {expected} bytes each \
                 ({count} x {elem}-byte {dtype}), got low={}, high={}",
                low.len(),
                high.len()
            )
        );
    }

    let low_scalars = decode_typed(low, dtype, path)?;
    let high_scalars = decode_typed(high, dtype, path)?;
    for (index, (lo, hi)) in low_scalars.iter().zip(high_scalars.iter()).enumerate() {
        if matches!(*lo, Scalar::Float(v) if v.is_nan())
            || matches!(*hi, Scalar::Float(v) if v.is_nan())
        {
            return err_space!(
                path,
                "Box",
                format!("typed bounds invalid: NaN bound at element {index}")
            );
        }
        if scalar_gt(*lo, *hi, dtype) {
            return err_space!(
                path,
                "Box",
                format!("typed bounds invalid: low>high at element {index}")
            );
        }
    }
    Ok(())
}

fn decode_typed(bytes: &[u8], dtype: DType, path: &str) -> Result<Vec<Scalar>, SpaceError> {
    decode_scalars(bytes, dtype).map_err(|err| SpaceError::Invalid {
        path: path.to_string(),
        msg: format!("cannot decode typed Box bounds: {err}"),
    })
}

/// `low > high` in the dtype's native domain, via the centralized
/// [`Scalar::cmp_typed`]. Integers compare as integers (`Uint64` unsigned),
/// floats as floats, and a mixed float-bound/int-value pair is compared exactly
/// (no truncation of either side). A `NaN` operand is unordered; for the
/// `low > high` validation question we treat that as "not greater" (`false`) so
/// NaN is handled by the dedicated NaN check at containment time rather than
/// here.
pub(crate) fn scalar_gt(low: Scalar, high: Scalar, dtype: DType) -> bool {
    low.cmp_typed(high, dtype) == Some(std::cmp::Ordering::Greater)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_builder_encodes_typed_uniform_bounds() {
        let spec = BoxSpaceBuilder::int_scalar(i64::MIN, i64::MAX, vec![2])
            .dtype(DType::Int64)
            .build()
            .expect("valid space");
        let Some(SpaceKind::Box(b)) = spec.spec else {
            panic!("expected Box");
        };
        let Some(BoxBounds::TypedUniform(t)) = b.bounds else {
            panic!("expected typed-uniform bounds");
        };
        assert_eq!(t.low, i64::MIN.to_le_bytes());
        assert_eq!(t.high, i64::MAX.to_le_bytes());
    }

    #[test]
    fn test_uint_builder_encodes_u64_max_exactly() {
        let spec = BoxSpaceBuilder::uint_scalar(0, u64::MAX, vec![1])
            .build()
            .expect("valid space");
        let Some(SpaceKind::Box(b)) = spec.spec else {
            panic!("expected Box");
        };
        let Some(BoxBounds::TypedUniform(t)) = b.bounds else {
            panic!("expected typed-uniform bounds");
        };
        assert_eq!(t.high, u64::MAX.to_le_bytes());
    }

    #[test]
    fn test_uint_builder_encodes_large_float_bounds_without_signed_wrap() {
        let low = u64::MAX - 1;
        let high = u64::MAX;
        let spec = BoxSpaceBuilder::uint_scalar(low, high, vec![1])
            .dtype(DType::Float64)
            .build()
            .expect("valid space");
        let Some(SpaceKind::Box(b)) = spec.spec else {
            panic!("expected Box");
        };
        let Some(BoxBounds::TypedUniform(t)) = b.bounds else {
            panic!("expected typed-uniform bounds");
        };
        let decoded_low = decode_typed(&t.low, DType::Float64, "$").expect("decode");
        let [Scalar::Float(decoded_low)] = decoded_low.as_slice() else {
            panic!("expected float low bound");
        };
        let decoded_high = decode_typed(&t.high, DType::Float64, "$").expect("decode");
        let [Scalar::Float(decoded_high)] = decoded_high.as_slice() else {
            panic!("expected float high bound");
        };
        assert_eq!(*decoded_low, low as f64);
        assert_eq!(*decoded_high, high as f64);
        assert!(*decoded_low > 0.0);
        assert!(*decoded_high > 0.0);
    }

    #[test]
    fn test_validate_rejects_typed_uniform_low_above_high() {
        // 10 > 5 in the native i64 domain.
        let spec = SpaceSpec {
            shape: vec![1],
            dtype: DType::Int64,
            spec: Some(SpaceKind::Box(BoxSpec {
                bounds: Some(BoxBounds::TypedUniform(TypedUniformBounds {
                    low: 10i64.to_le_bytes().to_vec(),
                    high: 5i64.to_le_bytes().to_vec(),
                })),
            })),
        };
        assert!(crate::spaces::validate_space(&spec).is_err());
    }

    #[test]
    fn test_validate_rejects_nan_bounds() {
        // NaN slips past `low > high` (all NaN comparisons are false) and would
        // read as "unbounded" at conformance, so validation must reject it --
        // both for f64 Uniform bounds and dtype-typed float bounds.
        let uniform = SpaceSpec {
            shape: vec![1],
            dtype: DType::Float64,
            spec: Some(SpaceKind::Box(BoxSpec {
                bounds: Some(BoxBounds::Uniform(UniformBounds {
                    low: f64::NAN,
                    high: 1.0,
                })),
            })),
        };
        assert!(crate::spaces::validate_space(&uniform).is_err());

        let typed = SpaceSpec {
            shape: vec![1],
            dtype: DType::Float64,
            spec: Some(SpaceKind::Box(BoxSpec {
                bounds: Some(BoxBounds::TypedUniform(TypedUniformBounds {
                    low: f64::NAN.to_le_bytes().to_vec(),
                    high: 1.0f64.to_le_bytes().to_vec(),
                })),
            })),
        };
        assert!(crate::spaces::validate_space(&typed).is_err());
    }

    #[test]
    fn test_int_builder_rejects_out_of_dtype_range_bounds() {
        let err = BoxSpaceBuilder::int_scalar(0, 300, vec![1])
            .dtype(DType::Int8)
            .build();
        assert!(
            err.is_err(),
            "300 is out of Int8 range and must be rejected"
        );

        // uint_scalar(0, 1<<33).dtype(Uint32) must fail, not wrap high -> 0.
        let err = BoxSpaceBuilder::uint_scalar(0, 1 << 33, vec![1])
            .dtype(DType::Uint32)
            .build();
        assert!(
            err.is_err(),
            "1<<33 is out of Uint32 range and must be rejected"
        );

        // A value that *does* fit still builds.
        assert!(
            BoxSpaceBuilder::int_scalar(0, 100, vec![1])
                .dtype(DType::Int8)
                .build()
                .is_ok()
        );
        // uint_scalar still round-trips u64::MAX exactly under Uint64.
        assert!(
            BoxSpaceBuilder::uint_scalar(0, u64::MAX, vec![1])
                .build()
                .is_ok()
        );
    }

    #[test]
    fn test_float_form_int_bound_rejects_set_changing_saturation() {
        // DANGEROUS: a `low` above the dtype max saturates DOWN to the max,
        // turning an empty admitted set into {max}. scalar(300.0, 400.0) on
        // Uint8 would wire as [255, 255] -- advertising {255} while the local
        // space admits no uint8 value at all, a silent contract change.
        assert!(
            BoxSpaceBuilder::scalar(300.0, 400.0, vec![1])
                .dtype(DType::Uint8)
                .build()
                .is_err(),
            "low above the dtype max must be rejected (saturation changes the set)"
        );

        // DANGEROUS elementwise: a per-element low above the dtype max.
        assert!(
            BoxSpaceBuilder::tensor(vec![0.0, 300.0], vec![10.0, 400.0], vec![2])
                .dtype(DType::Uint8)
                .build()
                .is_err(),
            "elementwise low above the dtype max must be rejected"
        );

        // DANGEROUS: a `high` below the dtype min saturates UP to the min.
        assert!(
            BoxSpaceBuilder::scalar(-400.0, -300.0, vec![1])
                .dtype(DType::Uint8)
                .build()
                .is_err(),
            "high below the dtype min must be rejected (saturation changes the set)"
        );

        // BENIGN (allowed): a `low` below the dtype min is a vacuous lower bound
        // -- the established "low = -1 means unbounded below" idiom for unsigned
        // dtypes, where saturating -1 -> 0 leaves the admitted set unchanged.
        assert!(
            BoxSpaceBuilder::scalar(-1.0, 100.0, vec![1])
                .dtype(DType::Uint64)
                .build()
                .is_ok(),
            "a low below the dtype min must stay allowed (vacuous, saturates harmlessly)"
        );

        // BENIGN (allowed): a `high` above the dtype max is a vacuous upper bound.
        assert!(
            BoxSpaceBuilder::scalar(0.0, 999.0, vec![1])
                .dtype(DType::Uint8)
                .build()
                .is_ok(),
            "a high above the dtype max must stay allowed (vacuous, saturates harmlessly)"
        );

        // A fractional bound on an integer dtype is still rejected (unchanged).
        assert!(
            BoxSpaceBuilder::scalar(0.0, 1.5, vec![1])
                .dtype(DType::Int32)
                .build()
                .is_err()
        );

        // An exactly-representable in-range bound builds.
        assert!(
            BoxSpaceBuilder::scalar(0.0, 255.0, vec![1])
                .dtype(DType::Uint8)
                .build()
                .is_ok()
        );
    }

    #[test]
    fn test_float_form_int_bound_rejects_rounded_64bit_max() {
        // i64::MAX / u64::MAX are NOT representable in f64: they round UP to
        // 2^63 / 2^64, one above the true max. A `low` at that rounded value
        // admits nothing locally (no int is >= 2^63 for Int64) but the wire cast
        // saturates DOWN to the dtype max, so the peer admits {max} -- the
        // Uint8(300..400) drift at the 64-bit precision edge. Both must reject.
        assert!(
            BoxSpaceBuilder::scalar(i64::MAX as f64, i64::MAX as f64, vec![1])
                .dtype(DType::Int64)
                .build()
                .is_err(),
            "a low bound at the f64-rounded i64::MAX must be rejected"
        );
        assert!(
            BoxSpaceBuilder::scalar(u64::MAX as f64, u64::MAX as f64, vec![1])
                .dtype(DType::Uint64)
                .build()
                .is_err(),
            "a low bound at the f64-rounded u64::MAX must be rejected"
        );

        // A genuinely in-range large 64-bit bound (2^62) still builds: it is
        // below the exclusive upper and round-trips through the saturating cast.
        assert!(
            BoxSpaceBuilder::scalar(2.0_f64.powi(62), 2.0_f64.powi(62), vec![1])
                .dtype(DType::Int64)
                .build()
                .is_ok(),
            "an in-range 64-bit bound (2^62) must still build"
        );
    }

    #[test]
    fn test_validate_uint64_high_below_low_in_unsigned_domain() {
        // low = u64::MAX (bytes), high = 0. As i64 these are -1 and 0, so an
        // i64 comparison would wrongly accept low <= high; the unsigned
        // comparison correctly rejects it.
        let spec = SpaceSpec {
            shape: vec![1],
            dtype: DType::Uint64,
            spec: Some(SpaceKind::Box(BoxSpec {
                bounds: Some(BoxBounds::TypedUniform(TypedUniformBounds {
                    low: u64::MAX.to_le_bytes().to_vec(),
                    high: 0u64.to_le_bytes().to_vec(),
                })),
            })),
        };
        assert!(crate::spaces::validate_space(&spec).is_err());
    }
}
