/// Element data type for tensor values exchanged across the wire.
///
/// Discriminants are kept byte-identical to the `rlmesh.spaces.v1.DType` proto
/// enum. The equality is enforced at compile time by a cross-check in
/// `rlmesh-grpc` (`wire::spaces`); `rlmesh-spaces` stays free of a
/// `rlmesh-proto` dependency, so the assert cannot live here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum DType {
    #[default]
    Unspecified = 0,
    Bool = 1,
    Uint8 = 2,
    Uint16 = 3,
    Uint32 = 4,
    Uint64 = 5,
    Int8 = 6,
    Int16 = 7,
    Int32 = 8,
    Int64 = 9,
    Float16 = 10,
    Float32 = 11,
    Float64 = 12,
}

impl TryFrom<i32> for DType {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Unspecified),
            1 => Ok(Self::Bool),
            2 => Ok(Self::Uint8),
            3 => Ok(Self::Uint16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Uint64),
            6 => Ok(Self::Int8),
            7 => Ok(Self::Int16),
            8 => Ok(Self::Int32),
            9 => Ok(Self::Int64),
            10 => Ok(Self::Float16),
            11 => Ok(Self::Float32),
            12 => Ok(Self::Float64),
            _ => Err("invalid dtype"),
        }
    }
}

impl From<DType> for i32 {
    fn from(value: DType) -> Self {
        value as i32
    }
}

impl DType {
    /// Every dtype, including `Unspecified`, in discriminant order.
    pub const ALL: [DType; 13] = [
        DType::Unspecified,
        DType::Bool,
        DType::Uint8,
        DType::Uint16,
        DType::Uint32,
        DType::Uint64,
        DType::Int8,
        DType::Int16,
        DType::Int32,
        DType::Int64,
        DType::Float16,
        DType::Float32,
        DType::Float64,
    ];

    /// The canonical lowercase dtype name (for example `"float32"`).
    pub const fn name(self) -> &'static str {
        match self {
            DType::Unspecified => "unspecified",
            DType::Bool => "bool",
            DType::Uint8 => "uint8",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Uint16 => "uint16",
            DType::Uint32 => "uint32",
            DType::Uint64 => "uint64",
        }
    }

    /// Parse a canonical dtype name. Only the 12 concrete dtypes are
    /// recognized; `"unspecified"` and unknown names return `None`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "bool" => Some(DType::Bool),
            "uint8" => Some(DType::Uint8),
            "int32" => Some(DType::Int32),
            "int64" => Some(DType::Int64),
            "float16" => Some(DType::Float16),
            "float32" => Some(DType::Float32),
            "float64" => Some(DType::Float64),
            "int8" => Some(DType::Int8),
            "int16" => Some(DType::Int16),
            "uint16" => Some(DType::Uint16),
            "uint32" => Some(DType::Uint32),
            "uint64" => Some(DType::Uint64),
            _ => None,
        }
    }

    /// Whether this is an integer dtype (signed or unsigned). `Bool` and the
    /// float dtypes are not integers.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            DType::Uint8
                | DType::Int8
                | DType::Int16
                | DType::Uint16
                | DType::Int32
                | DType::Uint32
                | DType::Int64
                | DType::Uint64
        )
    }

    /// Whether this is a floating-point dtype.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, DType::Float16 | DType::Float32 | DType::Float64)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Get the byte size of a dtype. `Unspecified` has no size and returns 0.
pub const fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::Unspecified => 0,
        DType::Bool | DType::Uint8 | DType::Int8 => 1,
        DType::Float16 | DType::Int16 | DType::Uint16 => 2,
        DType::Int32 | DType::Uint32 | DType::Float32 => 4,
        DType::Int64 | DType::Uint64 | DType::Float64 => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_i32_roundtrip() {
        for dtype in DType::ALL {
            let raw = i32::from(dtype);
            assert_eq!(DType::try_from(raw), Ok(dtype));
        }
    }

    #[test]
    fn test_dtype_rejects_unknown_values() {
        assert!(DType::try_from(-1).is_err());
        for unused in [13, 20, 30, 99] {
            assert!(
                DType::try_from(unused).is_err(),
                "{unused} should be invalid"
            );
        }
    }

    #[test]
    fn test_dtype_name_roundtrip() {
        for dtype in DType::ALL {
            if dtype == DType::Unspecified {
                continue;
            }
            assert_eq!(DType::from_name(dtype.name()), Some(dtype));
            assert_eq!(dtype.to_string(), dtype.name());
        }
        assert_eq!(DType::Unspecified.name(), "unspecified");
        assert_eq!(DType::from_name("unspecified"), None);
        assert_eq!(DType::from_name("complex64"), None);
        assert_eq!(DType::from_name("Float32"), None);
    }

    #[test]
    fn test_dtype_size_table() {
        let expected = [
            (DType::Unspecified, 0),
            (DType::Bool, 1),
            (DType::Uint8, 1),
            (DType::Int8, 1),
            (DType::Float16, 2),
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
