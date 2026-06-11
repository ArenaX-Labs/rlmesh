use crate::v1::dtype::DType;

/// DLPack data type codes (`DLDataTypeCode`).
mod code {
    pub const INT: u8 = 0;
    pub const UINT: u8 = 1;
    pub const FLOAT: u8 = 2;
    pub const BFLOAT: u8 = 4;
    pub const BOOL: u8 = 6;
}

/// A DLPack `DLDataType` triple describing a tensor element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DLPackType {
    /// `DLDataTypeCode` (0 = int, 1 = uint, 2 = float, 4 = bfloat, 6 = bool).
    pub code: u8,
    /// Element width in bits.
    pub bits: u8,
    /// Vector lanes; always 1 for RLMesh tensors.
    pub lanes: u16,
}

/// Map a dtype to its DLPack data type. `Unspecified` has no DLPack form.
pub fn dlpack_type(dtype: DType) -> Option<DLPackType> {
    let (code, bits) = match dtype {
        DType::Unspecified => return None,
        DType::Bool => (code::BOOL, 8),
        DType::Uint8 => (code::UINT, 8),
        DType::Uint16 => (code::UINT, 16),
        DType::Uint32 => (code::UINT, 32),
        DType::Uint64 => (code::UINT, 64),
        DType::Int8 => (code::INT, 8),
        DType::Int16 => (code::INT, 16),
        DType::Int32 => (code::INT, 32),
        DType::Int64 => (code::INT, 64),
        DType::Float16 => (code::FLOAT, 16),
        DType::Float32 => (code::FLOAT, 32),
        DType::Float64 => (code::FLOAT, 64),
        DType::Bfloat16 => (code::BFLOAT, 16),
    };
    Some(DLPackType {
        code,
        bits,
        lanes: 1,
    })
}

/// Map a DLPack data type back to a dtype. Returns `None` for unsupported
/// codes or widths and for vectorized types (`lanes != 1`).
pub fn dtype_from_dlpack(ty: DLPackType) -> Option<DType> {
    if ty.lanes != 1 {
        return None;
    }
    match (ty.code, ty.bits) {
        (code::BOOL, 8) => Some(DType::Bool),
        (code::UINT, 8) => Some(DType::Uint8),
        (code::UINT, 16) => Some(DType::Uint16),
        (code::UINT, 32) => Some(DType::Uint32),
        (code::UINT, 64) => Some(DType::Uint64),
        (code::INT, 8) => Some(DType::Int8),
        (code::INT, 16) => Some(DType::Int16),
        (code::INT, 32) => Some(DType::Int32),
        (code::INT, 64) => Some(DType::Int64),
        (code::FLOAT, 16) => Some(DType::Float16),
        (code::FLOAT, 32) => Some(DType::Float32),
        (code::FLOAT, 64) => Some(DType::Float64),
        (code::BFLOAT, 16) => Some(DType::Bfloat16),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v1::dtype::dtype_size;

    const SUPPORTED: [DType; 13] = [
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
        DType::Bfloat16,
    ];

    #[test]
    fn test_dlpack_type_table() {
        let expected = [
            (DType::Bool, 6, 8),
            (DType::Uint8, 1, 8),
            (DType::Uint16, 1, 16),
            (DType::Uint32, 1, 32),
            (DType::Uint64, 1, 64),
            (DType::Int8, 0, 8),
            (DType::Int16, 0, 16),
            (DType::Int32, 0, 32),
            (DType::Int64, 0, 64),
            (DType::Float16, 2, 16),
            (DType::Float32, 2, 32),
            (DType::Float64, 2, 64),
            (DType::Bfloat16, 4, 16),
        ];
        for (dtype, code, bits) in expected {
            let ty = dlpack_type(dtype).expect("supported dtype");
            assert_eq!((ty.code, ty.bits, ty.lanes), (code, bits, 1), "{dtype:?}");
        }
        assert_eq!(dlpack_type(DType::Unspecified), None);
    }

    #[test]
    fn test_dlpack_type_bits_match_dtype_size() {
        for dtype in SUPPORTED {
            let ty = dlpack_type(dtype).expect("supported dtype");
            assert_eq!(ty.bits as usize, dtype_size(dtype) * 8, "{dtype:?}");
        }
    }

    #[test]
    fn test_dlpack_type_roundtrip() {
        for dtype in SUPPORTED {
            let ty = dlpack_type(dtype).expect("supported dtype");
            assert_eq!(dtype_from_dlpack(ty), Some(dtype));
        }
    }

    #[test]
    fn test_dtype_from_dlpack_rejects_unsupported() {
        // Vectorized types.
        assert_eq!(
            dtype_from_dlpack(DLPackType {
                code: 2,
                bits: 32,
                lanes: 4
            }),
            None
        );
        // Unknown width.
        assert_eq!(
            dtype_from_dlpack(DLPackType {
                code: 0,
                bits: 128,
                lanes: 1
            }),
            None
        );
        // Unknown code (e.g. complex = 5).
        assert_eq!(
            dtype_from_dlpack(DLPackType {
                code: 5,
                bits: 64,
                lanes: 1
            }),
            None
        );
    }
}
