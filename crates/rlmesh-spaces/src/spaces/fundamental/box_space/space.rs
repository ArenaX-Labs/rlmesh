use crate::dtype::dtype_size;
use crate::errors::{SpaceError, err_space};
use crate::scalar::{Scalar, decode_scalars, encode_i64_scalars};
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
                    low: encode_int_bound(&[low as i64], dtype)?,
                    high: encode_int_bound(&[high as i64], dtype)?,
                })
            }
            PendingBounds::UintTensor { low, high } => {
                let low: Vec<i64> = low.into_iter().map(|v| v as i64).collect();
                let high: Vec<i64> = high.into_iter().map(|v| v as i64).collect();
                BoxBounds::TypedElementwise(TypedElementwiseBounds {
                    low: encode_int_bound(&low, dtype)?,
                    high: encode_int_bound(&high, dtype)?,
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

fn encode_int_bound(values: &[i64], dtype: DType) -> Result<Vec<u8>, SpaceError> {
    encode_i64_scalars(values, dtype).map_err(|err| SpaceError::Invalid {
        path: "Box".to_string(),
        msg: format!("cannot encode integer Box bounds: {err}"),
    })
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

/// Compare two decoded scalars in their native domain. `Uint64` is the one
/// dtype whose values do not fit `i64`; it is compared as `u64`.
pub(crate) fn scalar_gt(low: Scalar, high: Scalar, dtype: DType) -> bool {
    match (low, high) {
        (Scalar::Float(lo), Scalar::Float(hi)) => lo > hi,
        (Scalar::Bool(lo), Scalar::Bool(hi)) => lo & !hi,
        (lo, hi) if dtype == DType::Uint64 => (lo.as_i64() as u64) > (hi.as_i64() as u64),
        (lo, hi) => lo.as_i64() > hi.as_i64(),
    }
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
