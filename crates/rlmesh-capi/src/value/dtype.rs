//! `RLMeshDLPackDType` — the DLPack `(code, bits, lanes)` triple, aligned to core's
//! `dlpack_type`/`dtype_from_dlpack` so DLPack export is a field copy.
#![allow(unsafe_code)] // FFI: no_mangle export + repr(C) struct.

use rlmesh_spaces::DType;
use rlmesh_spaces::dtype_size;
use rlmesh_spaces::tensor::{DLPackType, dlpack_type, dtype_from_dlpack};

/// Element type as a DLPack `(code, bits, lanes)` triple. `code` equals DLPack's
/// `DLDataTypeCode` (int=0, uint=1, float=2, bfloat=4, bool=6); `lanes` is 1.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RLMeshDLPackDType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

impl From<DLPackType> for RLMeshDLPackDType {
    fn from(t: DLPackType) -> Self {
        Self {
            code: t.code,
            bits: t.bits,
            lanes: t.lanes,
        }
    }
}

impl From<RLMeshDLPackDType> for DLPackType {
    fn from(t: RLMeshDLPackDType) -> Self {
        Self {
            code: t.code,
            bits: t.bits,
            lanes: t.lanes,
        }
    }
}

impl RLMeshDLPackDType {
    pub(crate) fn from_core(dtype: DType) -> Option<Self> {
        dlpack_type(dtype).map(Into::into)
    }
    pub(crate) fn to_core(self) -> Option<DType> {
        dtype_from_dlpack(self.into())
    }
}

/// Byte size of one element, or 0 if the dtype is unsupported (or `lanes != 1`).
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_dtype_size(dtype: RLMeshDLPackDType) -> usize {
    dtype.to_core().map_or(0, dtype_size)
}

/// `RLMESH_DTYPE_OTHER`: Adapters' tolerant dtype fields only; Spaces refuses it.
pub(crate) const DTYPE_OTHER: i32 = 1000;

/// The `RLMeshDType` enum value (the wire number, 1-12) of a facade dtype.
pub(crate) fn dtype_to_c(dtype: rlmesh::DType) -> i32 {
    dtype as i32
}

/// The facade dtype of an `RLMeshDType` enum value; `INVALID`, `OTHER` and
/// unknown values are `None`.
pub(crate) fn dtype_from_c(value: i32) -> Option<rlmesh::DType> {
    let core = DType::try_from(value).ok()?;
    if core == DType::Unspecified {
        return None;
    }
    rlmesh::DType::parse(core.name())
}

/// Python's spelling of `dtype` (static storage), or NULL for `INVALID`,
/// `RLMESH_DTYPE_OTHER` or an unknown value.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_dtype_str(dtype: i32) -> *const std::ffi::c_char {
    dtype_from_c(dtype).map_or(std::ptr::null(), |dtype| static_str(dtype.as_str()))
}

/// Byte size of one element of `dtype`, or 0 for `INVALID`, `OTHER` or an
/// unknown value.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_ext_dtype_size(dtype: i32) -> usize {
    dtype_from_c(dtype).map_or(0, rlmesh::ext::dtype_size)
}

/// A NUL-terminated copy of a `'static` vocabulary spelling, interned so the
/// pointer stays valid for the life of the process.
pub(crate) fn static_str(text: &'static str) -> *const std::ffi::c_char {
    use std::collections::HashMap;
    use std::ffi::CString;
    use std::sync::{Mutex, OnceLock};
    static TABLE: OnceLock<Mutex<HashMap<&'static str, &'static std::ffi::CStr>>> = OnceLock::new();
    let table = TABLE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut table = table.lock().unwrap_or_else(|err| err.into_inner());
    let entry = table.entry(text).or_insert_with(|| {
        let owned = CString::new(text).unwrap_or_default();
        &*std::boxed::Box::leak(owned.into_boxed_c_str())
    });
    entry.as_ptr()
}
