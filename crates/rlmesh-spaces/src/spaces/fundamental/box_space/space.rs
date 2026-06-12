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
    UintTensor { low: Vec<u64>, high: Vec<u64> },
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

    /// Per-element unsigned-integer bounds (row-major), carried as dtype-typed
    /// bytes. Defaults to `Uint64`.
    pub fn uint_tensor(low: Vec<u64>, high: Vec<u64>, shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::Uint64,
            bounds: PendingBounds::UintTensor { low, high },
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
            PendingBounds::UintTensor { low, high } => {
                BoxBounds::TypedElementwise(TypedElementwiseBounds {
                    low: encode_uint_bound(&low, dtype)?,
                    high: encode_uint_bound(&high, dtype)?,
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
    if matches!(
        dtype,
        DType::Float16 | DType::Bfloat16 | DType::Float32 | DType::Float64
    ) {
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
            if s.low > s.high {
                return err_space!(path, "Box", "scalar bounds invalid: low > high");
            }
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
                if t.low[i] > t.high[i] {
                    return err_space!(
                        path,
                        "Box",
                        format!("tensor bounds invalid: low>high at element {i}")
                    );
                }
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
    fn test_int_builder_rejects_out_of_dtype_range_bounds() {
        // Bug 3: int_scalar(0, 300).dtype(Int8) must fail, not wrap 300 -> 44.
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
