//! `RlmeshDType` — the DLPack `(code, bits, lanes)` triple, aligned to core's
//! `dlpack_type`/`dtype_from_dlpack` so DLPack export is a field copy.
#![allow(unsafe_code)] // FFI: no_mangle export + repr(C) struct.

use rlmesh_spaces::DType;
use rlmesh_spaces::dtype_size;
use rlmesh_spaces::tensor::{DLPackType, dlpack_type, dtype_from_dlpack};

/// Element type as a DLPack `(code, bits, lanes)` triple. `code` equals DLPack's
/// `DLDataTypeCode` (int=0, uint=1, float=2, bfloat=4, bool=6); `lanes` is 1.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RlmeshDType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

impl From<DLPackType> for RlmeshDType {
    fn from(t: DLPackType) -> Self {
        Self {
            code: t.code,
            bits: t.bits,
            lanes: t.lanes,
        }
    }
}

impl From<RlmeshDType> for DLPackType {
    fn from(t: RlmeshDType) -> Self {
        Self {
            code: t.code,
            bits: t.bits,
            lanes: t.lanes,
        }
    }
}

impl RlmeshDType {
    pub(crate) fn from_core(dtype: DType) -> Option<Self> {
        dlpack_type(dtype).map(Into::into)
    }
    pub(crate) fn to_core(self) -> Option<DType> {
        dtype_from_dlpack(self.into())
    }
}

/// Byte size of one element, or 0 if the dtype is unsupported (or `lanes != 1`).
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_dtype_size(dtype: RlmeshDType) -> usize {
    dtype.to_core().map_or(0, dtype_size)
}
