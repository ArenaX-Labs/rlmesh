/// Element data type for tensor values exchanged across the wire.
///
/// Discriminants match the `rlmesh.spaces.v1.DType` proto enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum DType {
    #[default]
    Unspecified = 0,
    Bool = 1,
    Uint8 = 2,
    Int32 = 3,
    Int64 = 4,
    Float16 = 5,
    Float32 = 6,
    Float64 = 7,
    Int8 = 8,
    Int16 = 9,
    Uint16 = 10,
    Uint32 = 11,
    Uint64 = 12,
    Bfloat16 = 13,
}

impl TryFrom<i32> for DType {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Unspecified),
            1 => Ok(Self::Bool),
            2 => Ok(Self::Uint8),
            3 => Ok(Self::Int32),
            4 => Ok(Self::Int64),
            5 => Ok(Self::Float16),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Float64),
            8 => Ok(Self::Int8),
            9 => Ok(Self::Int16),
            10 => Ok(Self::Uint16),
            11 => Ok(Self::Uint32),
            12 => Ok(Self::Uint64),
            13 => Ok(Self::Bfloat16),
            _ => Err("invalid dtype"),
        }
    }
}

impl From<DType> for i32 {
    fn from(value: DType) -> Self {
        value as i32
    }
}

/// Get the byte size of a dtype. `Unspecified` has no size and returns 0.
pub const fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::Unspecified => 0,
        DType::Bool | DType::Uint8 | DType::Int8 => 1,
        DType::Float16 | DType::Bfloat16 | DType::Int16 | DType::Uint16 => 2,
        DType::Int32 | DType::Uint32 | DType::Float32 => 4,
        DType::Int64 | DType::Uint64 | DType::Float64 => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [DType; 14] = [
        DType::Unspecified,
        DType::Bool,
        DType::Uint8,
        DType::Int32,
        DType::Int64,
        DType::Float16,
        DType::Float32,
        DType::Float64,
        DType::Int8,
        DType::Int16,
        DType::Uint16,
        DType::Uint32,
        DType::Uint64,
        DType::Bfloat16,
    ];

    #[test]
    fn test_dtype_i32_roundtrip() {
        for dtype in ALL {
            let raw = i32::from(dtype);
            assert_eq!(DType::try_from(raw), Ok(dtype));
        }
    }

    #[test]
    fn test_dtype_rejects_unknown_values() {
        assert!(DType::try_from(-1).is_err());
        assert!(DType::try_from(14).is_err());
    }

    #[test]
    fn test_dtype_size_table() {
        let expected = [
            (DType::Unspecified, 0),
            (DType::Bool, 1),
            (DType::Uint8, 1),
            (DType::Int8, 1),
            (DType::Float16, 2),
            (DType::Bfloat16, 2),
            (DType::Int16, 2),
            (DType::Uint16, 2),
            (DType::Int32, 4),
            (DType::Uint32, 4),
            (DType::Float32, 4),
            (DType::Int64, 8),
            (DType::Uint64, 8),
            (DType::Float64, 8),
        ];
        for (dtype, size) in expected {
            assert_eq!(dtype_size(dtype), size, "size mismatch for {dtype:?}");
        }
    }
}
