use crate::BoxBounds;
use crate::dtype::DType;
use crate::errors::{SpaceError, err_space};
use crate::scalar::{Scalar, decode_scalars};
use crate::spaces::{SpaceKind, SpaceSpec, SpaceValue};

/// A per-element low/high bound. `None` means "unbounded on this side" so
/// integer bounds never have to invent a sentinel that collides with a real
/// value (`i64::MIN`/`u64::MAX` are legitimate bounds).
type Bound = (Option<Scalar>, Option<Scalar>);

pub(crate) fn contains_box(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let tensor = match value {
        SpaceValue::Box(tensor) => tensor,
        _ => return err_space!(path, "expected Box value"),
    };

    if tensor.shape() != space.shape.as_slice() {
        return err_space!(
            path,
            format!(
                "shape mismatch: expected {:?}, got {:?}",
                space.shape,
                tensor.shape()
            )
        );
    }

    if tensor.dtype() != space.dtype {
        return err_space!(
            path,
            format!(
                "dtype mismatch: expected {:?}, got {:?}",
                space.dtype,
                tensor.dtype()
            )
        );
    }

    let dtype = tensor.dtype();
    if dtype == DType::Unspecified {
        // Tensor constructors reject Unspecified, so this cannot occur.
        return err_space!(path, "Box value dtype is unspecified");
    }

    let numel = tensor.numel();
    let bounds = box_bounds(space, numel, dtype, path)?;

    let data = tensor.to_contiguous_bytes();
    let values = decode_scalars(&data, dtype).map_err(|err| SpaceError::Invalid {
        path: path.to_string(),
        msg: format!("cannot decode Box value: {err}"),
    })?;

    for (index, value) in values.iter().enumerate() {
        let (low, high) = &bounds[index];
        validate_box_scalar(*value, low.as_ref(), high.as_ref(), dtype, path, index)?;
    }
    Ok(())
}

/// Resolve per-element bounds for every dtype, comparing in the dtype's native
/// domain. Float bounds carried as `double` (`Uniform`/`Elementwise`) are
/// decoded as `Scalar::Float`; dtype-typed byte bounds are decoded with the
/// space's dtype so integer ranges stay exact.
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

/// An infinite `double` bound means "unbounded on this side"; a finite one is
/// a real `Scalar::Float` comparison value.
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

fn validate_box_scalar(
    value: Scalar,
    low: Option<&Scalar>,
    high: Option<&Scalar>,
    dtype: DType,
    path: &str,
    index: usize,
) -> Result<(), SpaceError> {
    if let Scalar::Float(v) = value
        && v.is_nan()
    {
        // A NaN value is out of bounds unless both sides are unbounded.
        if low.is_some() || high.is_some() {
            return err_space!(path, format!("Box value at element {index} is NaN"));
        }
        return Ok(());
    }

    if let Some(low) = low
        && super::space::scalar_gt(*low, value, dtype)
    {
        return err_space!(
            path,
            format!("Box value at element {index} out of bounds: below low bound")
        );
    }
    if let Some(high) = high
        && super::space::scalar_gt(value, *high, dtype)
    {
        return err_space!(
            path,
            format!("Box value at element {index} out of bounds: above high bound")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::contains;
    use crate::spaces::fundamental::BoxSpaceBuilder;
    use crate::tensor::{Storage, Tensor};
    use half::bf16;

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
    fn test_box_contains_bfloat16_bounds() {
        let space = box_space(0.0, 1.0, vec![2], DType::Bfloat16);

        let in_bounds: Vec<u8> = [bf16::from_f32(0.25), bf16::from_f32(1.0)]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let valid = SpaceValue::Box(
            Tensor::from_vec(in_bounds, vec![2], DType::Bfloat16).expect("valid tensor"),
        );
        assert!(contains(&space, &valid).is_ok());

        let out_of_bounds: Vec<u8> = [bf16::from_f32(0.25), bf16::from_f32(2.5)]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let invalid = SpaceValue::Box(
            Tensor::from_vec(out_of_bounds, vec![2], DType::Bfloat16).expect("valid tensor"),
        );
        assert!(contains(&space, &invalid).is_err());
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
    fn test_box_contains_fractional_float_bound_int_dtype_is_exact() {
        let space = box_space(0.5, 10.0, vec![1], DType::Int64);
        let zero = SpaceValue::Box(
            Tensor::from_vec(0i64.to_le_bytes().to_vec(), vec![1], DType::Int64).expect("tensor"),
        );
        assert!(contains(&space, &zero).is_err(), "0 < 0.5 must be rejected");

        // 1 is in [0.5, 10.0].
        let one = SpaceValue::Box(
            Tensor::from_vec(1i64.to_le_bytes().to_vec(), vec![1], DType::Int64).expect("tensor"),
        );
        assert!(contains(&space, &one).is_ok());

        // 10 is in bounds (== high); 11 is above.
        let ten = SpaceValue::Box(
            Tensor::from_vec(10i64.to_le_bytes().to_vec(), vec![1], DType::Int64).expect("tensor"),
        );
        assert!(contains(&space, &ten).is_ok());
        let eleven = SpaceValue::Box(
            Tensor::from_vec(11i64.to_le_bytes().to_vec(), vec![1], DType::Int64).expect("tensor"),
        );
        assert!(contains(&space, &eleven).is_err());
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
