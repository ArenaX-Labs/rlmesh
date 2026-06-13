//! The value model the apply engine operates on.

use std::collections::BTreeMap;

use super::error::ApplyError;

/// Element type of an [`Array`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    U8,
    I32,
    I64,
    F32,
    F64,
}

impl Dtype {
    /// Parse a NumPy-style dtype name.
    pub fn parse(name: &str) -> Result<Self, ApplyError> {
        match name {
            "uint8" => Ok(Self::U8),
            "int32" => Ok(Self::I32),
            "int64" => Ok(Self::I64),
            "float32" => Ok(Self::F32),
            "float64" => Ok(Self::F64),
            other => Err(ApplyError::new(format!(
                "unsupported dtype {other:?} (supported: uint8, int32, int64, \
                 float32, float64)"
            ))),
        }
    }

    /// NumPy-style dtype name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }
}

/// Typed flat storage backing an [`Array`].
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayData {
    U8(Vec<u8>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// A dense n-dimensional array in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: ArrayData,
}

impl Array {
    /// Build a float32 array, validating the element count.
    pub fn from_f32(shape: Vec<usize>, data: Vec<f32>) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Self {
            dtype: Dtype::F32,
            shape,
            data: ArrayData::F32(data),
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match &self.data {
            ArrayData::U8(data) => data.len(),
            ArrayData::I32(data) => data.len(),
            ArrayData::I64(data) => data.len(),
            ArrayData::F32(data) => data.len(),
            ArrayData::F64(data) => data.len(),
        }
    }

    /// Whether the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Flat float32 copy of the elements (NumPy `astype(float32).reshape(-1)`).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match &self.data {
            ArrayData::U8(data) => data.iter().map(|&x| f32::from(x)).collect(),
            ArrayData::I32(data) => data.iter().map(|&x| x as f32).collect(),
            ArrayData::I64(data) => data.iter().map(|&x| x as f32).collect(),
            ArrayData::F32(data) => data.clone(),
            ArrayData::F64(data) => data.iter().map(|&x| x as f32).collect(),
        }
    }

    /// Cast to another dtype (NumPy `astype` semantics for in-range values).
    pub fn cast(&self, dtype: Dtype) -> Self {
        if dtype == self.dtype {
            return self.clone();
        }
        let f64s: Vec<f64> = match &self.data {
            ArrayData::U8(data) => data.iter().map(|&x| f64::from(x)).collect(),
            ArrayData::I32(data) => data.iter().map(|&x| f64::from(x)).collect(),
            ArrayData::I64(data) => data.iter().map(|&x| x as f64).collect(),
            ArrayData::F32(data) => data.iter().map(|&x| f64::from(x)).collect(),
            ArrayData::F64(data) => data.clone(),
        };
        let data = match dtype {
            Dtype::U8 => ArrayData::U8(f64s.iter().map(|&x| x as u8).collect()),
            Dtype::I32 => ArrayData::I32(f64s.iter().map(|&x| x as i32).collect()),
            Dtype::I64 => ArrayData::I64(f64s.iter().map(|&x| x as i64).collect()),
            Dtype::F32 => ArrayData::F32(f64s.iter().map(|&x| x as f32).collect()),
            Dtype::F64 => ArrayData::F64(f64s),
        };
        Self {
            dtype,
            shape: self.shape.clone(),
            data,
        }
    }
}

/// A value flowing through the apply engine.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Array(Array),
    Text(String),
    Number(f64),
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}
