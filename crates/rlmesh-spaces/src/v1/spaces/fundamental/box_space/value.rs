use crate::errors::{SpaceError, err_space};
use crate::v1::spaces::{SpaceSpec, SpaceValue, space_spec};
use crate::v1::{DType, box_spec};
use half::f16;

/// A continuous tensor value for Box spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxValue {
    /// Raw byte data in C-contiguous order
    pub data: Vec<u8>,
    /// Shape of the tensor
    pub shape: Vec<i64>,
    /// Data type
    pub dtype: DType,
}

impl BoxValue {
    /// Create a new BoxValue from raw bytes.
    pub fn new(data: Vec<u8>, shape: Vec<i64>, dtype: DType) -> Self {
        Self { data, shape, dtype }
    }

    /// Get the number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Get the expected byte size of the data.
    pub fn expected_byte_size(&self) -> usize {
        self.numel() * dtype_size(self.dtype)
    }

    /// Check if the data size matches the expected size.
    pub fn is_valid_size(&self) -> bool {
        self.data.len() == self.expected_byte_size()
    }
}

/// Get the byte size of a dtype.
pub fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::Unspecified => 0,
        DType::Bool => 1,
        DType::Uint8 => 1,
        DType::Int32 => 4,
        DType::Int64 => 8,
        DType::Float16 => 2,
        DType::Float32 => 4,
        DType::Float64 => 8,
    }
}

pub(crate) fn contains_box(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let box_val = match value {
        SpaceValue::Box(v) => v,
        _ => return err_space!(path, "expected Box value"),
    };

    // Check shape matches
    if box_val.shape != space.shape {
        return err_space!(
            path,
            format!(
                "shape mismatch: expected {:?}, got {:?}",
                space.shape, box_val.shape
            )
        );
    }

    // Check dtype matches
    let expected_dtype = space.dtype;
    if box_val.dtype != expected_dtype {
        return err_space!(
            path,
            format!(
                "dtype mismatch: expected {:?}, got {:?}",
                expected_dtype, box_val.dtype
            )
        );
    }

    // Check data size
    if !box_val.is_valid_size() {
        return err_space!(
            path,
            format!(
                "data size mismatch: expected {} bytes, got {}",
                box_val.expected_byte_size(),
                box_val.data.len()
            )
        );
    }

    let numel = box_val.numel();
    let (low_bounds, high_bounds) = box_bounds(space, numel, path)?;

    match box_val.dtype {
        DType::Bool => {
            for (index, byte) in box_val.data.iter().enumerate() {
                validate_box_scalar(
                    if *byte == 0 { 0.0 } else { 1.0 },
                    low_bounds[index],
                    high_bounds[index],
                    path,
                    index,
                )?;
            }
        }
        DType::Uint8 => {
            for (index, byte) in box_val.data.iter().enumerate() {
                validate_box_scalar(
                    *byte as f64,
                    low_bounds[index],
                    high_bounds[index],
                    path,
                    index,
                )?;
            }
        }
        DType::Int32 => {
            for (index, chunk) in box_val.data.chunks_exact(4).enumerate() {
                validate_box_scalar(
                    i32::from_le_bytes(chunk.try_into().expect("chunk")) as f64,
                    low_bounds[index],
                    high_bounds[index],
                    path,
                    index,
                )?;
            }
        }
        DType::Int64 => {
            for (index, chunk) in box_val.data.chunks_exact(8).enumerate() {
                validate_box_scalar(
                    i64::from_le_bytes(chunk.try_into().expect("chunk")) as f64,
                    low_bounds[index],
                    high_bounds[index],
                    path,
                    index,
                )?;
            }
        }
        DType::Float32 | DType::Unspecified => {
            for (index, chunk) in box_val.data.chunks_exact(4).enumerate() {
                validate_box_scalar(
                    f32::from_le_bytes(chunk.try_into().expect("chunk")) as f64,
                    low_bounds[index],
                    high_bounds[index],
                    path,
                    index,
                )?;
            }
        }
        DType::Float64 => {
            for (index, chunk) in box_val.data.chunks_exact(8).enumerate() {
                validate_box_scalar(
                    f64::from_le_bytes(chunk.try_into().expect("chunk")),
                    low_bounds[index],
                    high_bounds[index],
                    path,
                    index,
                )?;
            }
        }
        DType::Float16 => {
            for (index, chunk) in box_val.data.chunks_exact(2).enumerate() {
                validate_box_scalar(
                    f16::from_le_bytes(chunk.try_into().expect("chunk")).to_f64(),
                    low_bounds[index],
                    high_bounds[index],
                    path,
                    index,
                )?;
            }
        }
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
    use crate::v1::spaces::contains;
    use crate::v1::spaces::fundamental::BoxSpaceBuilder;

    #[test]
    fn test_box_contains() {
        let space = BoxSpaceBuilder::scalar(0.0, 1.0, vec![2, 3])
            .dtype(DType::Float32)
            .build()
            .unwrap();

        // Valid value: 2*3*4 = 24 bytes
        let valid = SpaceValue::Box(BoxValue::new(vec![0u8; 24], vec![2, 3], DType::Float32));
        assert!(contains(&space, &valid).is_ok());

        // Wrong shape
        let wrong_shape = SpaceValue::Box(BoxValue::new(vec![0u8; 12], vec![3, 2], DType::Float32));
        assert!(contains(&space, &wrong_shape).is_err());

        // Wrong dtype
        let wrong_dtype = SpaceValue::Box(BoxValue::new(
            vec![0u8; 48], // 2*3*8 for float64
            vec![2, 3],
            DType::Float64,
        ));
        assert!(contains(&space, &wrong_dtype).is_err());
    }

    #[test]
    fn test_box_contains_rejects_out_of_bounds_values() {
        let space = BoxSpaceBuilder::scalar(0.0, 1.0, vec![2])
            .dtype(DType::Float32)
            .build()
            .unwrap();

        let invalid = SpaceValue::Box(BoxValue::new(
            vec![
                0, 0, 0, 63, // 0.5f32
                0, 0, 32, 64, // 2.5f32
            ],
            vec![2],
            DType::Float32,
        ));

        assert!(contains(&space, &invalid).is_err());
    }
}
