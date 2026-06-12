use crate::box_spec;
use crate::dtype::DType;
use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceSpec, SpaceValue, space_spec};
use half::{bf16, f16};

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

    let (low, high) = box_bounds(space, tensor.numel(), path)?;
    let data = tensor.to_contiguous_bytes();

    // i64/u64 values beyond 2^53 lose precision in the f64 comparison.
    match tensor.dtype() {
        DType::Bool => check_bounds::<1>(&data, &low, &high, path, |bytes| {
            if bytes[0] == 0 { 0.0 } else { 1.0 }
        }),
        DType::Uint8 => check_bounds::<1>(&data, &low, &high, path, |bytes| bytes[0] as f64),
        DType::Int8 => check_bounds::<1>(&data, &low, &high, path, |bytes| bytes[0] as i8 as f64),
        DType::Int16 => check_bounds::<2>(&data, &low, &high, path, |bytes| {
            i16::from_le_bytes(bytes) as f64
        }),
        DType::Uint16 => check_bounds::<2>(&data, &low, &high, path, |bytes| {
            u16::from_le_bytes(bytes) as f64
        }),
        DType::Int32 => check_bounds::<4>(&data, &low, &high, path, |bytes| {
            i32::from_le_bytes(bytes) as f64
        }),
        DType::Uint32 => check_bounds::<4>(&data, &low, &high, path, |bytes| {
            u32::from_le_bytes(bytes) as f64
        }),
        DType::Int64 => check_bounds::<8>(&data, &low, &high, path, |bytes| {
            i64::from_le_bytes(bytes) as f64
        }),
        DType::Uint64 => check_bounds::<8>(&data, &low, &high, path, |bytes| {
            u64::from_le_bytes(bytes) as f64
        }),
        DType::Float16 => check_bounds::<2>(&data, &low, &high, path, |bytes| {
            f16::from_le_bytes(bytes).to_f64()
        }),
        DType::Bfloat16 => check_bounds::<2>(&data, &low, &high, path, |bytes| {
            bf16::from_le_bytes(bytes).to_f64()
        }),
        DType::Float32 => check_bounds::<4>(&data, &low, &high, path, |bytes| {
            f32::from_le_bytes(bytes) as f64
        }),
        DType::Float64 => check_bounds::<8>(&data, &low, &high, path, f64::from_le_bytes),
        // Tensor constructors reject Unspecified, so this cannot occur.
        DType::Unspecified => err_space!(path, "Box value dtype is unspecified"),
    }
}

fn check_bounds<const N: usize>(
    data: &[u8],
    low: &[f64],
    high: &[f64],
    path: &str,
    decode: impl Fn([u8; N]) -> f64,
) -> Result<(), SpaceError> {
    for (index, chunk) in data.chunks_exact(N).enumerate() {
        let bytes: [u8; N] = chunk.try_into().expect("chunks_exact yields N-byte chunks");
        validate_box_scalar(decode(bytes), low[index], high[index], path, index)?;
    }
    Ok(())
}

fn box_bounds(
    space: &SpaceSpec,
    numel: usize,
    path: &str,
) -> Result<(Vec<f64>, Vec<f64>), SpaceError> {
    let spec = match &space.spec {
        Some(space_spec::Spec::Box(spec)) => spec,
        _ => return err_space!(path, "space is not Box"),
    };

    Ok(match &spec.bounds {
        Some(box_spec::Bounds::Uniform(bounds)) => {
            (vec![bounds.low; numel], vec![bounds.high; numel])
        }
        Some(box_spec::Bounds::Axiswise(bounds)) => (
            repeat_or_truncate(bounds.low.as_slice(), numel, f64::NEG_INFINITY),
            repeat_or_truncate(bounds.high.as_slice(), numel, f64::INFINITY),
        ),
        Some(box_spec::Bounds::Elementwise(bounds)) => (
            repeat_or_truncate(bounds.low.as_slice(), numel, f64::NEG_INFINITY),
            repeat_or_truncate(bounds.high.as_slice(), numel, f64::INFINITY),
        ),
        Some(box_spec::Bounds::Unbounded(_)) | None => {
            (vec![f64::NEG_INFINITY; numel], vec![f64::INFINITY; numel])
        }
    })
}

fn repeat_or_truncate(values: &[f64], len: usize, default: f64) -> Vec<f64> {
    match values.len() {
        0 => vec![default; len],
        1 => vec![values[0]; len],
        current if current >= len => values[..len].to_vec(),
        current => values
            .iter()
            .copied()
            .cycle()
            .take(len.max(current))
            .take(len)
            .collect(),
    }
}

fn validate_box_scalar(
    value: f64,
    low: f64,
    high: f64,
    path: &str,
    index: usize,
) -> Result<(), SpaceError> {
    if value < low || value > high {
        return err_space!(
            path,
            format!(
                "Box value at element {index} out of bounds: got {value}, expected [{low}, {high}]"
            )
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
}
