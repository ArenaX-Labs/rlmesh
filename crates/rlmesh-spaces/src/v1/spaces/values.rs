//! Space value types and validation.
//!
//! Runtime value representations for RLMesh spaces
//! and functions to validate that values belong to their spaces.

use std::collections::BTreeMap;

use crate::errors::{SpaceError, err_space};
use crate::v1::spaces::{SpaceSpec, SpaceType, space_spec};
use crate::v1::{DType, box_spec, multi_binary_spec, multi_discrete_spec};
use half::f16;

/// A runtime value that can belong to a space.
///
/// This is the Rust representation of values that flow through the
/// environment interface (observations, actions, etc.).
#[derive(Debug, Clone, PartialEq)]
pub enum SpaceValue {
    /// Box space value - continuous tensor
    Box(BoxValue),

    /// Discrete space value - single integer
    Discrete(i64),

    /// MultiBinary space value - boolean array
    MultiBinary(Vec<bool>),

    /// MultiDiscrete space value - integer array
    MultiDiscrete(Vec<i64>),

    /// Text space value - string
    Text(String),

    /// Dict space value - named sub-values
    Dict(BTreeMap<String, SpaceValue>),

    /// Tuple space value - ordered sub-values
    Tuple(Vec<SpaceValue>),
}

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

/// Check if a value belongs to a space.
///
/// Returns Ok(()) if the value is valid for the space, or an error describing
/// why the value doesn't fit.
pub fn contains(space: &SpaceSpec, value: &SpaceValue) -> Result<(), SpaceError> {
    contains_at(space, value, "$")
}

fn contains_at(space: &SpaceSpec, value: &SpaceValue, path: &str) -> Result<(), SpaceError> {
    match space.space_type() {
        SpaceType::Box => contains_box(space, value, path),
        SpaceType::Discrete => contains_discrete(space, value, path),
        SpaceType::MultiBinary => contains_multibinary(space, value, path),
        SpaceType::MultiDiscrete => contains_multidiscrete(space, value, path),
        SpaceType::Text => contains_text(space, value, path),
        SpaceType::Dict => contains_dict(space, value, path),
        SpaceType::Tuple => contains_tuple(space, value, path),
        SpaceType::Unspecified => err_space!(path, "space type not specified"),
    }
}

fn contains_box(space: &SpaceSpec, value: &SpaceValue, path: &str) -> Result<(), SpaceError> {
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

fn contains_discrete(space: &SpaceSpec, value: &SpaceValue, path: &str) -> Result<(), SpaceError> {
    let val = match value {
        SpaceValue::Discrete(v) => *v,
        _ => return err_space!(path, "expected Discrete value"),
    };

    let d = match &space.spec {
        Some(space_spec::Spec::Discrete(d)) => d,
        _ => return err_space!(path, "space is not Discrete"),
    };

    // Check value is in range [start, start + n)
    if val < d.start || val >= d.start + d.n {
        return err_space!(
            path,
            format!(
                "value {} not in range [{}, {})",
                val,
                d.start,
                d.start + d.n
            )
        );
    }

    Ok(())
}

fn contains_multibinary(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let vals = match value {
        SpaceValue::MultiBinary(v) => v,
        _ => return err_space!(path, "expected MultiBinary value"),
    };

    let mb = match &space.spec {
        Some(space_spec::Spec::MultiBinary(mb)) => mb,
        _ => return err_space!(path, "space is not MultiBinary"),
    };

    // Get expected size from the space
    let expected_size = match &mb.n {
        Some(multi_binary_spec::N::Size(n)) => *n as usize,
        Some(multi_binary_spec::N::Dims(dims)) => dims.data.iter().map(|&d| d as usize).product(),
        None => return err_space!(path, "MultiBinary.n not set"),
    };

    if vals.len() != expected_size {
        return err_space!(
            path,
            format!(
                "MultiBinary size mismatch: expected {}, got {}",
                expected_size,
                vals.len()
            )
        );
    }

    // Values are bools, always valid
    Ok(())
}

fn contains_multidiscrete(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let vals = match value {
        SpaceValue::MultiDiscrete(v) => v,
        _ => return err_space!(path, "expected MultiDiscrete value"),
    };

    let md = match &space.spec {
        Some(space_spec::Spec::MultiDiscrete(md)) => md,
        _ => return err_space!(path, "space is not MultiDiscrete"),
    };

    // Get nvec from the space
    let nvec: Vec<i64> = match &md.nvec {
        Some(multi_discrete_spec::Nvec::Flat(v)) => v.data.clone(),
        Some(multi_discrete_spec::Nvec::Shaped(m)) => {
            m.data.iter().flat_map(|row| row.data.clone()).collect()
        }
        None => return err_space!(path, "MultiDiscrete.nvec not set"),
    };

    if vals.len() != nvec.len() {
        return err_space!(
            path,
            format!(
                "MultiDiscrete size mismatch: expected {}, got {}",
                nvec.len(),
                vals.len()
            )
        );
    }

    // Check each value is in range [0, nvec[i])
    for (i, (&val, &n)) in vals.iter().zip(nvec.iter()).enumerate() {
        if val < 0 || val >= n {
            return err_space!(
                path,
                format!("value[{}] = {} not in range [0, {})", i, val, n)
            );
        }
    }

    Ok(())
}

fn contains_text(space: &SpaceSpec, value: &SpaceValue, path: &str) -> Result<(), SpaceError> {
    let text = match value {
        SpaceValue::Text(s) => s,
        _ => return err_space!(path, "expected Text value"),
    };

    let t = match &space.spec {
        Some(space_spec::Spec::Text(t)) => t,
        _ => return err_space!(path, "space is not Text"),
    };

    // Check length
    let len = text.len() as i64;
    if len < t.min_length {
        return err_space!(
            path,
            format!("text length {} below minimum {}", len, t.min_length)
        );
    }
    if len > t.max_length {
        return err_space!(
            path,
            format!("text length {} exceeds maximum {}", len, t.max_length)
        );
    }

    // Check charset if specified
    if !t.charset.is_empty() {
        for c in text.chars() {
            if !t.charset.contains(c) {
                return err_space!(path, format!("character '{}' not in charset", c));
            }
        }
    }

    Ok(())
}

fn contains_dict(space: &SpaceSpec, value: &SpaceValue, path: &str) -> Result<(), SpaceError> {
    let dict_val = match value {
        SpaceValue::Dict(d) => d,
        _ => return err_space!(path, "expected Dict value"),
    };

    let d = match &space.spec {
        Some(space_spec::Spec::Dict(d)) => d,
        _ => return err_space!(path, "space is not Dict"),
    };

    // Check all required keys are present
    for (i, key) in d.keys.iter().enumerate() {
        match dict_val.get(key) {
            Some(sub_val) => {
                contains_at(&d.spaces[i], sub_val, &format!("{path}.{key}"))?;
            }
            None => {
                return err_space!(path, format!("missing key '{}'", key));
            }
        }
    }

    // Check no extra keys
    for key in dict_val.keys() {
        if !d.keys.contains(key) {
            return err_space!(path, format!("unexpected key '{}'", key));
        }
    }

    Ok(())
}

fn contains_tuple(space: &SpaceSpec, value: &SpaceValue, path: &str) -> Result<(), SpaceError> {
    let tuple_val = match value {
        SpaceValue::Tuple(t) => t,
        _ => return err_space!(path, "expected Tuple value"),
    };

    let t = match &space.spec {
        Some(space_spec::Spec::Tuple(t)) => t,
        _ => return err_space!(path, "space is not Tuple"),
    };

    if tuple_val.len() != t.spaces.len() {
        return err_space!(
            path,
            format!(
                "tuple length mismatch: expected {}, got {}",
                t.spaces.len(),
                tuple_val.len()
            )
        );
    }

    for (i, (sub_space, sub_val)) in t.spaces.iter().zip(tuple_val.iter()).enumerate() {
        contains_at(sub_space, sub_val, &format!("{path}[{i}]"))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v1::spaces::composite::{DictSpaceBuilder, TupleSpaceBuilder};
    use crate::v1::spaces::fundamental::{BoxSpaceBuilder, DiscreteBuilder};

    #[test]
    fn test_discrete_contains() {
        let space = DiscreteBuilder::new(4).build().unwrap();

        assert!(contains(&space, &SpaceValue::Discrete(0)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(3)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(4)).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(-1)).is_err());
    }

    #[test]
    fn test_discrete_with_start() {
        let space = DiscreteBuilder::new(4).start(10).build().unwrap();

        assert!(contains(&space, &SpaceValue::Discrete(10)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(13)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(9)).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(14)).is_err());
    }

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

    #[test]
    fn test_text_contains() {
        use crate::v1::spaces::fundamental::TextBuilder;

        let unrestricted = TextBuilder::new(32).build().unwrap();
        assert!(
            contains(
                &unrestricted,
                &SpaceValue::Text("pick up the object!".to_string())
            )
            .is_ok()
        );

        let space = TextBuilder::new(5)
            .min_length(2)
            .charset("abc".to_string())
            .build()
            .unwrap();

        assert!(contains(&space, &SpaceValue::Text("abc".to_string())).is_ok());
        assert!(contains(&space, &SpaceValue::Text("ab".to_string())).is_ok());
        assert!(contains(&space, &SpaceValue::Text("a".to_string())).is_err()); // too short
        assert!(contains(&space, &SpaceValue::Text("abcdef".to_string())).is_err()); // too long
        assert!(contains(&space, &SpaceValue::Text("abd".to_string())).is_err()); // 'd' not in charset
    }

    #[test]
    fn test_dict_contains() {
        let box_space = BoxSpaceBuilder::scalar(0.0, 1.0, vec![3]).build().unwrap();
        let discrete = DiscreteBuilder::new(4).build().unwrap();

        let space = DictSpaceBuilder::new()
            .insert("obs", box_space)
            .insert("action", discrete)
            .build()
            .unwrap();

        let valid = SpaceValue::Dict(BTreeMap::from([
            (
                "obs".to_string(),
                SpaceValue::Box(BoxValue::new(vec![0u8; 12], vec![3], DType::Float32)),
            ),
            ("action".to_string(), SpaceValue::Discrete(2)),
        ]));
        assert!(contains(&space, &valid).is_ok());

        // Missing key
        let missing = SpaceValue::Dict(BTreeMap::from([(
            "obs".to_string(),
            SpaceValue::Box(BoxValue::new(vec![0u8; 12], vec![3], DType::Float32)),
        )]));
        assert!(contains(&space, &missing).is_err());

        // Extra key
        let extra = SpaceValue::Dict(BTreeMap::from([
            (
                "obs".to_string(),
                SpaceValue::Box(BoxValue::new(vec![0u8; 12], vec![3], DType::Float32)),
            ),
            ("action".to_string(), SpaceValue::Discrete(2)),
            ("extra".to_string(), SpaceValue::Discrete(0)),
        ]));
        assert!(contains(&space, &extra).is_err());
    }

    #[test]
    fn test_tuple_contains() {
        let box_space = BoxSpaceBuilder::scalar(0.0, 1.0, vec![3]).build().unwrap();
        let discrete = DiscreteBuilder::new(4).build().unwrap();

        let space = TupleSpaceBuilder::new()
            .with(box_space)
            .with(discrete)
            .build()
            .unwrap();

        let valid = SpaceValue::Tuple(vec![
            SpaceValue::Box(BoxValue::new(vec![0u8; 12], vec![3], DType::Float32)),
            SpaceValue::Discrete(2),
        ]);
        assert!(contains(&space, &valid).is_ok());

        // Wrong length
        let wrong_len = SpaceValue::Tuple(vec![SpaceValue::Discrete(2)]);
        assert!(contains(&space, &wrong_len).is_err());
    }
}
