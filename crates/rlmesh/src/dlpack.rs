//! Shared DLPack ABI and ownership. Only this module dereferences producers'
//! pointers; the Python layer handles capsule negotiation and naming.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::ptr::NonNull;

use crate::{DType, Error, Result, Tensor, ToDlpackOptions};
use rlmesh_spaces::Storage;

pub use rlmesh_spaces::DLPackType;

/// Map a facade dtype to its DLPack triple.
pub fn dlpack_type(dtype: DType) -> Option<DLPackType> {
    rlmesh_spaces::dlpack_type(dtype.to_core())
}

/// Parse a supported scalar DLPack triple.
pub fn dtype_from_dlpack(dtype: DLPackType) -> Option<DType> {
    rlmesh_spaces::dtype_from_dlpack(dtype).map(DType::from_core)
}

/// DLPack device descriptor (`kDLCPU = 1`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDevice {
    /// DLPack device type code (1 for CPU).
    pub device_type: i32,
    /// Device index (0 for CPU).
    pub device_id: i32,
}

/// DLPack element type descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDataType {
    /// DLPack scalar kind code.
    pub code: u8,
    /// Width of one scalar in bits.
    pub bits: u8,
    /// Number of vector lanes (1 for scalar dtypes).
    pub lanes: u16,
}

/// A borrowed DLPack tensor view. Strides are in **elements**, unlike
/// [`Tensor::strides`], which returns bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLTensor {
    /// Base allocation pointer; add byte_offset to find the first element.
    pub data: *mut c_void,
    /// Device holding the data.
    pub device: DLDevice,
    /// Number of dimensions.
    pub ndim: i32,
    /// Element type descriptor.
    pub dtype: DLDataType,
    /// Array of ndim dimension lengths.
    pub shape: *mut i64,
    /// Array of ndim element strides; NULL means C-contiguous.
    pub strides: *mut i64,
    /// Offset from data to the first element, in bytes.
    pub byte_offset: u64,
}

/// Legacy capsule layout, used only by the Python binding.
#[repr(C)]
pub struct DLManagedTensor {
    /// Borrowed layout and element data, valid until the deleter runs.
    pub dl_tensor: DLTensor,
    /// Opaque producer context; consumers must leave it untouched.
    pub manager_ctx: *mut c_void,
    /// Optional producer callback, invoked once by the owner.
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// DLPack protocol version.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLPackVersion {
    /// Major protocol version.
    pub major: u32,
    /// Minor protocol version.
    pub minor: u32,
}

/// DLPack 1.0 managed tensor layout. Ownership passes through `deleter`.
#[repr(C)]
pub struct DLManagedTensorVersioned {
    /// Protocol version describing this managed tensor.
    pub version: DLPackVersion,
    /// Opaque producer context; consumers must leave it untouched.
    pub manager_ctx: *mut c_void,
    /// Optional producer callback, invoked once by the owner.
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    /// DLPack flag bits: READ_ONLY = 1 and IS_COPIED = 2.
    pub flags: u64,
    /// Borrowed layout and element data, valid until the deleter runs.
    pub dl_tensor: DLTensor,
}

/// Owns a versioned DLPack export and invokes its deleter exactly once on
/// drop. Use [`Self::into_raw`] to transfer ownership to a consumer.
pub struct DlpackTensor(NonNull<DLManagedTensorVersioned>);

impl DlpackTensor {
    /// Give ownership to a consumer without calling the deleter.
    pub fn into_raw(self) -> NonNull<DLManagedTensorVersioned> {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }

    /// Take back ownership, for example after a failed import.
    ///
    /// # Safety
    /// `ptr` must be a live, exclusively owned managed tensor whose deleter
    /// may be called once. No other owner may invoke it after this call.
    pub unsafe fn from_raw(ptr: NonNull<DLManagedTensorVersioned>) -> Self {
        Self(ptr)
    }
}

impl Drop for DlpackTensor {
    fn drop(&mut self) {
        // SAFETY: construction requires exclusive ownership of this live
        // managed tensor, and into_raw suppresses Drop on ownership transfer.
        unsafe {
            if let Some(deleter) = self.0.as_ref().deleter {
                deleter(self.0.as_ptr());
            }
        }
    }
}

impl Tensor {
    /// Copy a versioned CPU DLPack tensor into fresh contiguous storage.
    /// On success, calls the producer's deleter; on error, ownership stays
    /// with the caller, who may recover it with [`DlpackTensor::from_raw`].
    ///
    /// # Safety
    /// `obj` must point to a live, exclusively owned managed tensor. Its
    /// shape/stride arrays must be readable for `ndim` entries and all
    /// described element addresses must be readable, initialized CPU bytes
    /// for the duration of the call. The producer must not mutate them
    /// concurrently. The deleter must be valid to call once on success.
    pub unsafe fn from_dlpack(obj: NonNull<DLManagedTensorVersioned>) -> Result<Tensor> {
        // SAFETY: the caller guarantees a live descriptor and its metadata.
        unsafe {
            let managed = obj.as_ref();
            if managed.version.major > 1 {
                return Err(Error::invalid_argument(format!(
                    "unsupported DLPack major version {}",
                    managed.version.major
                )));
            }
            let tensor = copy_dl_tensor(&managed.dl_tensor)?;
            if let Some(deleter) = managed.deleter {
                deleter(obj.as_ptr());
            }
            Ok(tensor)
        }
    }

    /// Export a DLPack 1.0 tensor, sharing read-only storage by default.
    /// A copying export owns fresh writable storage. The returned owner
    /// keeps storage and layout alive even after this tensor is destroyed.
    pub fn to_dlpack(&self, options: ToDlpackOptions) -> Result<DlpackTensor> {
        if let Some(device) = options.dl_device
            && device != self.dlpack_device()
        {
            return Err(Error::invalid_argument(format!(
                "cannot export to device {device:?}; only CPU (1, 0) is supported"
            )));
        }
        let tensor = self;
        let ndim = i32::try_from(tensor.ndim())
            .map_err(|_| Error::invalid_argument("DLPack tensor rank overflows i32"))?;
        let ty = dlpack_type(tensor.dtype()).expect("all facade dtypes support DLPack");
        let strides = match tensor.core().strides().filter(|_| !options.copy) {
            Some(strides) => strides.to_vec(),
            None => {
                let mut strides = vec![0; tensor.ndim()];
                let mut stride = 1i64;
                for (out, &dim) in strides.iter_mut().zip(tensor.core().shape()).rev() {
                    *out = stride;
                    stride = stride
                        .checked_mul(dim)
                        .ok_or_else(|| Error::invalid_argument("DLPack tensor strides overflow"))?;
                }
                strides
            }
        };
        let (data, storage, copy) = if options.copy {
            // Own the writable allocation directly. Derive its pointer from
            // as_mut_ptr, never from a shared Storage slice. Keep the same
            // 64-byte alignment as Tensor::copy without lending Rust slices
            // into this allocation while the consumer may write it.
            let bytes = tensor.tobytes();
            let capacity = bytes
                .len()
                .checked_add(63)
                .ok_or_else(|| Error::invalid_argument("DLPack copy size overflows usize"))?;
            let mut copy = Vec::with_capacity(capacity);
            let offset = (64 - copy.as_ptr() as usize % 64) % 64;
            copy.resize(offset, 0);
            copy.extend_from_slice(&bytes);
            let data = copy.as_mut_ptr().wrapping_add(offset).cast::<c_void>();
            (data, None, Some(copy))
        } else {
            let storage = tensor.core().storage().clone();
            let data = storage.as_slice()[tensor.core().byte_offset()..].as_ptr() as *mut c_void;
            (data, Some(storage), None)
        };
        let mut holder = Box::new(ExportHolder {
            managed: DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 0 },
                manager_ctx: std::ptr::null_mut(),
                deleter: Some(versioned_deleter),
                // DLPACK_FLAG_BITMASK_READ_ONLY (1), IS_COPIED (2).
                flags: if options.copy { 2 } else { 1 },
                dl_tensor: DLTensor {
                    data,
                    device: DLDevice {
                        device_type: 1,
                        device_id: 0,
                    },
                    ndim,
                    dtype: DLDataType {
                        code: ty.code,
                        bits: ty.bits,
                        lanes: ty.lanes,
                    },
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
            },
            _storage: storage,
            _copy: copy,
            shape: tensor.core().shape().to_vec(),
            strides,
        });
        holder.managed.dl_tensor.shape = holder.shape.as_mut_ptr();
        holder.managed.dl_tensor.strides = holder.strides.as_mut_ptr();
        holder.managed.manager_ctx = (&raw mut *holder).cast();
        // repr(C) puts managed at offset zero. The Box and its Vec buffers
        // stay put until the managed tensor's deleter is called.
        let ptr = NonNull::new(Box::into_raw(holder).cast()).expect("Box is non-null");
        Ok(DlpackTensor(ptr))
    }

    /// The DLPack device `(kDLCPU, 0)`.
    pub fn dlpack_device(&self) -> (i32, i32) {
        (1, 0)
    }
}

#[repr(C)]
struct ExportHolder {
    managed: DLManagedTensorVersioned,
    _storage: Option<Storage>,
    _copy: Option<Vec<u8>>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn versioned_deleter(managed: *mut DLManagedTensorVersioned) {
    if !managed.is_null() {
        // SAFETY: to_dlpack allocated this repr(C) holder with managed at
        // offset zero; the consumer owns the sole right to call its deleter.
        unsafe { drop(Box::from_raw(managed.cast::<ExportHolder>())) };
    }
}

#[repr(C)]
struct LegacyHolder {
    managed: DLManagedTensor,
    _versioned: DlpackTensor,
}

/// Adapt an owned versioned export for Python's legacy capsule protocol.
pub fn export_legacy(export: DlpackTensor) -> NonNull<DLManagedTensor> {
    // SAFETY: export owns a live descriptor, kept alive by the new holder.
    let dl_tensor = unsafe { export.0.as_ref().dl_tensor };
    let holder = Box::new(LegacyHolder {
        managed: DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(legacy_deleter),
        },
        _versioned: export,
    });
    NonNull::new(Box::into_raw(holder).cast()).expect("Box is non-null")
}

unsafe extern "C" fn legacy_deleter(managed: *mut DLManagedTensor) {
    if !managed.is_null() {
        // SAFETY: export_legacy allocates this holder with managed at offset
        // zero. Dropping its versioned owner releases the storage once.
        unsafe { drop(Box::from_raw(managed.cast::<LegacyHolder>())) };
    }
}

/// Copy and consume a legacy Python capsule descriptor on success only.
///
/// # Safety
/// The pointer and ownership requirements are those of [`Tensor::from_dlpack`].
pub unsafe fn import_legacy(obj: NonNull<DLManagedTensor>) -> Result<Tensor> {
    // SAFETY: caller guarantees live, exclusively owned descriptor and data.
    unsafe {
        let managed = obj.as_ref();
        let tensor = copy_dl_tensor(&managed.dl_tensor)?;
        if let Some(deleter) = managed.deleter {
            deleter(obj.as_ptr());
        }
        Ok(tensor)
    }
}

/// Copy arbitrary (including negative) element strides without adopting
/// producer memory. All arithmetic is checked before reading any elements.
unsafe fn copy_dl_tensor(dl: &DLTensor) -> Result<Tensor> {
    let invalid = Error::invalid_argument;
    if dl.device.device_type != 1 {
        return Err(invalid(format!(
            "only CPU DLPack tensors are supported, got device type {}",
            dl.device.device_type
        )));
    }
    let dtype = dtype_from_dlpack(DLPackType {
        code: dl.dtype.code,
        bits: dl.dtype.bits,
        lanes: dl.dtype.lanes,
    })
    .ok_or_else(|| {
        invalid(format!(
            "unsupported DLPack dtype (code={}, bits={}, lanes={})",
            dl.dtype.code, dl.dtype.bits, dl.dtype.lanes
        ))
    })?;
    if dl.ndim < 0 {
        return Err(invalid("DLPack tensor has negative ndim".to_owned()));
    }
    let ndim = dl.ndim as usize;
    if ndim > isize::MAX as usize / std::mem::size_of::<i64>() {
        return Err(invalid("DLPack metadata size overflows isize".to_owned()));
    }
    if ndim > 0 && dl.shape.is_null() {
        return Err(invalid("DLPack tensor has a null shape".to_owned()));
    }
    // SAFETY: the caller guarantees readable metadata arrays; don't form a
    // slice from NULL even for rank zero.
    let shape = if ndim == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(dl.shape, ndim) }
    };
    let shape: Vec<usize> = shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim)
                .map_err(|_| invalid("DLPack tensor has a negative dimension".to_owned()))
        })
        .collect::<Result<_>>()?;
    let overflow = || invalid("DLPack tensor size or strides overflow".to_owned());
    let numel = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(overflow)?
    };
    let itemsize = crate::ext::dtype_size(dtype);
    let nbytes = numel
        .checked_mul(itemsize)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(overflow)?;
    if numel == 0 {
        return Tensor::new(&[], &shape, dtype);
    }
    if dl.data.is_null() {
        return Err(invalid("DLPack tensor has null data".to_owned()));
    }
    let mut strides = vec![0isize; ndim];
    if dl.strides.is_null() {
        let mut stride = itemsize;
        for (out, &dim) in strides.iter_mut().zip(&shape).rev() {
            *out = isize::try_from(stride).map_err(|_| overflow())?;
            stride = stride.checked_mul(dim).ok_or_else(overflow)?;
        }
    } else {
        // SAFETY: caller guarantees ndim readable entries when non-null.
        let elements = unsafe { std::slice::from_raw_parts(dl.strides, ndim) };
        for (out, &stride) in strides.iter_mut().zip(elements) {
            *out = isize::try_from(stride)
                .ok()
                .and_then(|s| s.checked_mul(itemsize as isize))
                .ok_or_else(overflow)?;
        }
    }
    let mut min = 0isize;
    let mut max = 0isize;
    for (&dim, &stride) in shape.iter().zip(&strides) {
        let extent = isize::try_from(dim - 1)
            .ok()
            .and_then(|d| d.checked_mul(stride))
            .ok_or_else(overflow)?;
        min = min.checked_add(extent.min(0)).ok_or_else(overflow)?;
        max = max.checked_add(extent.max(0)).ok_or_else(overflow)?;
    }
    max.checked_sub(min)
        .and_then(|n| n.checked_add(itemsize as isize))
        .ok_or_else(overflow)?;
    let byte_offset = usize::try_from(dl.byte_offset)
        .ok()
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(overflow)?;
    // SAFETY: caller guarantees every described element is readable. The
    // checked extents ensure no stride calculation or slice size overflows.
    let base = unsafe { dl.data.cast::<u8>().add(byte_offset) };
    let mut expected = itemsize;
    let contiguous = shape.iter().zip(&strides).rev().all(|(&dim, &stride)| {
        let matches = dim == 1 || stride == expected as isize;
        expected *= dim; // bounded by the already checked nbytes
        matches
    });
    if contiguous {
        // SAFETY: a contiguous descriptor covers nbytes initialized bytes
        // from base, and nbytes was checked against isize::MAX above.
        return Tensor::new(
            unsafe { std::slice::from_raw_parts(base, nbytes) },
            &shape,
            dtype,
        );
    }
    let mut data = Vec::with_capacity(nbytes);
    let mut index = vec![0usize; ndim];
    for _ in 0..numel {
        let offset: isize = index
            .iter()
            .zip(&strides)
            .map(|(&i, &s)| i as isize * s)
            .sum();
        // SAFETY: validated extents bound this offset, and the caller owns
        // the initialized itemsize bytes at this element address.
        unsafe {
            data.extend_from_slice(std::slice::from_raw_parts(base.offset(offset), itemsize));
        }
        for axis in (0..ndim).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    // Adopt the freshly gathered buffer; imports never retain the producer.
    crate::ext::TensorExt::from_vec(data, &shape, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ext::TensorExt;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[repr(C)]
    struct Tracked {
        managed: DLManagedTensorVersioned,
        calls: Arc<AtomicUsize>,
        _export: DlpackTensor,
    }

    unsafe extern "C" fn tracked_deleter(ptr: *mut DLManagedTensorVersioned) {
        // SAFETY: tracked() allocated this repr(C) box, with managed first.
        let holder = unsafe { Box::from_raw(ptr.cast::<Tracked>()) };
        holder.calls.fetch_add(1, Ordering::SeqCst);
    }

    fn tracked(tensor: &Tensor) -> (NonNull<DLManagedTensorVersioned>, Arc<AtomicUsize>) {
        let export = tensor.to_dlpack(ToDlpackOptions::default()).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let holder = Box::new(Tracked {
            managed: DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 0 },
                manager_ctx: std::ptr::null_mut(),
                deleter: Some(tracked_deleter),
                flags: 1,
                // SAFETY: export owns the descriptor and stays in the holder.
                dl_tensor: unsafe { export.0.as_ref().dl_tensor },
            },
            calls: calls.clone(),
            _export: export,
        });
        (NonNull::new(Box::into_raw(holder).cast()).unwrap(), calls)
    }

    #[test]
    fn all_dtypes_copy_in_and_release_producer_once() {
        for dtype in DType::ALL {
            let bytes = vec![1; crate::ext::dtype_size(dtype) * 6];
            let tensor = Tensor::new(&bytes, &[2, 3], dtype).unwrap();
            let (ptr, calls) = tracked(&tensor);
            // SAFETY: tracked produces a valid exclusively owned descriptor.
            let imported = unsafe { Tensor::from_dlpack(ptr) }.unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(imported, tensor);
            assert_ne!(imported.buffer().as_ptr(), tensor.buffer().as_ptr());
            drop(tensor);
            assert_eq!(imported.tobytes(), bytes);
            drop(imported);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(dtype_from_dlpack(dlpack_type(dtype).unwrap()), Some(dtype));
        }
    }

    #[test]
    fn exports_keep_source_storage_and_layout_alive() {
        let storage = Storage::from_vec((0..24).collect());
        let tensor =
            Tensor::from_storage(storage, DType::Int32, &[2, 2], Some(&[3, 2]), 0).unwrap();
        assert_eq!(tensor.strides(), vec![12, 8]);
        let bytes = tensor.tobytes();
        let original_ptr = tensor.buffer().as_ptr();
        let shared = tensor.to_dlpack(ToDlpackOptions::default()).unwrap();
        let copied = tensor
            .to_dlpack(ToDlpackOptions {
                copy: true,
                dl_device: Some((1, 0)),
            })
            .unwrap();
        drop(tensor);
        // SAFETY: both exports own live descriptors and metadata arrays.
        unsafe {
            let dl = shared.0.as_ref();
            assert_eq!(dl.flags, 1);
            assert_eq!(dl.dl_tensor.data.cast_const().cast::<u8>(), original_ptr);
            assert_eq!(std::slice::from_raw_parts(dl.dl_tensor.strides, 2), &[3, 2]);
            assert_eq!(copied.0.as_ref().flags, 2);
            assert_ne!(
                copied.0.as_ref().dl_tensor.data.cast_const().cast::<u8>(),
                original_ptr
            );
            let from_shared = Tensor::from_dlpack(shared.into_raw()).unwrap();
            let from_copy = Tensor::from_dlpack(copied.into_raw()).unwrap();
            assert_eq!(from_shared.tobytes(), bytes);
            assert_eq!(from_copy.tobytes(), bytes);
            assert_eq!(from_shared.strides(), vec![8, 4]);
        }
    }

    #[test]
    fn copying_export_is_writable_and_isolated() {
        let tensor = Tensor::new(&[1, 2, 3, 4], &[4], DType::Uint8).unwrap();
        let copy = tensor
            .to_dlpack(ToDlpackOptions {
                copy: true,
                ..Default::default()
            })
            .unwrap();
        // SAFETY: a copying export is exclusively owned and writable. No
        // Rust slice into its fresh storage exists during this write.
        unsafe {
            *copy.0.as_ref().dl_tensor.data.cast::<u8>() = 42;
        }
        assert_eq!(tensor.tobytes(), vec![1, 2, 3, 4]);
        // SAFETY: copy transfers a live uniquely owned descriptor.
        let imported = unsafe { Tensor::from_dlpack(copy.into_raw()) }.unwrap();
        assert_eq!(imported.tobytes(), vec![42, 2, 3, 4]);
    }

    #[test]
    fn failed_import_leaves_ownership_with_caller() {
        let tensor = Tensor::new(&[0; 24], &[2, 3], DType::Float32).unwrap();
        for case in 0..10 {
            let (mut ptr, calls) = tracked(&tensor);
            // SAFETY: we own this descriptor. Each malformed field is either
            // rejected before data access, or describes an empty tensor.
            unsafe {
                let dl = &mut ptr.as_mut().dl_tensor;
                let mut dims = [i64::MAX, 3];
                let mut strides = [i64::MAX, 1];
                match case {
                    0 => ptr.as_mut().version.major = 2,
                    1 => dl.device.device_type = 2,
                    2 => dl.dtype.lanes = 2,
                    3 => dl.ndim = -1,
                    4 => dl.shape = std::ptr::null_mut(),
                    5 => dl.data = std::ptr::null_mut(),
                    6 => dl.shape = dims.as_mut_ptr(),
                    7 => dl.strides = strides.as_mut_ptr(),
                    8 => dl.byte_offset = u64::MAX,
                    _ => {
                        dims[0] = -1;
                        dl.shape = dims.as_mut_ptr();
                    }
                }
                let err = Tensor::from_dlpack(ptr).unwrap_err();
                assert_eq!(err.class(), crate::ErrorClass::ValueError);
                assert_eq!(calls.load(Ordering::SeqCst), 0, "case {case}");
                drop(DlpackTensor::from_raw(ptr));
                assert_eq!(calls.load(Ordering::SeqCst), 1, "case {case}");
            }
        }
    }

    #[test]
    fn negative_zero_and_null_strides_and_byte_offsets() {
        let source = Tensor::new(&[0, 1, 2, 3, 4, 5], &[2, 3], DType::Uint8).unwrap();
        for (strides, offset, expected) in [
            (Some([3, -1]), 2, vec![2, 1, 0, 5, 4, 3]),
            (Some([0, 1]), 1, vec![1, 2, 3, 1, 2, 3]),
            (None, 0, vec![0, 1, 2, 3, 4, 5]),
        ] {
            let (mut ptr, calls) = tracked(&source);
            let mut strides = strides;
            // SAFETY: each modified layout only addresses bytes in source;
            // strides stays alive during import, which consumes the holder.
            unsafe {
                ptr.as_mut().dl_tensor.strides = strides
                    .as_mut()
                    .map_or(std::ptr::null_mut(), |s| s.as_mut_ptr());
                ptr.as_mut().dl_tensor.byte_offset = offset;
                let imported = Tensor::from_dlpack(ptr).unwrap();
                assert_eq!(imported.tobytes(), expected);
                assert_eq!(imported.strides(), vec![3, 1]);
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn scalar_and_empty_null_data_imports() {
        for shape in [&[][..], &[0, 3][..]] {
            let source = Tensor::zeros(shape, DType::Int64).unwrap();
            let (mut ptr, calls) = tracked(&source);
            // SAFETY: NULL metadata for scalars and NULL data for empty
            // tensors are valid; neither is dereferenced during import.
            unsafe {
                if shape.is_empty() {
                    ptr.as_mut().dl_tensor.shape = std::ptr::null_mut();
                    ptr.as_mut().dl_tensor.strides = std::ptr::null_mut();
                } else {
                    ptr.as_mut().dl_tensor.data = std::ptr::null_mut();
                }
                assert_eq!(Tensor::from_dlpack(ptr).unwrap(), source);
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn unsupported_export_device_and_defaults() {
        let options = ToDlpackOptions::default();
        assert_eq!(options.dl_device, None);
        assert!(!options.copy);
        let source = Tensor::new(&[], &[0], DType::Bool).unwrap();
        assert_eq!(source.dlpack_device(), (1, 0));
        for device in [(2, 0), (1, 1)] {
            let result = source.to_dlpack(ToDlpackOptions {
                dl_device: Some(device),
                ..Default::default()
            });
            assert_eq!(result.err().unwrap().class(), crate::ErrorClass::ValueError);
        }
    }

    #[test]
    fn exported_owner_drop_releases_shared_storage() {
        let tensor = Tensor::new(&[1, 2], &[2], DType::Uint8).unwrap();
        let (ptr, calls) = tracked(&tensor);
        // SAFETY: tracked gives exclusive ownership; Drop calls its deleter.
        let owned = unsafe { DlpackTensor::from_raw(ptr) };
        drop(tensor);
        drop(owned);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
