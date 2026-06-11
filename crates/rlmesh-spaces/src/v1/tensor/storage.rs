use std::sync::Arc;

/// Alignment guaranteed for buffers allocated by [`Storage`] itself.
const ALIGN: usize = 64;

/// Reference-counted, immutable byte storage shared by one or more
/// [`Tensor`](super::Tensor) views.
///
/// Buffers allocated by `Storage` ([`from_slice`](Storage::from_slice),
/// [`zeroed`](Storage::zeroed)) are 64-byte aligned. Buffers adopted from a
/// caller ([`from_vec`](Storage::from_vec)) keep their original allocation and
/// carry no alignment guarantee.
#[derive(Debug, Clone)]
pub struct Storage(Inner);

#[derive(Debug, Clone)]
enum Inner {
    /// Storage-allocated buffer, 64-byte aligned.
    Aligned(Arc<AlignedBytes>),
    /// Caller-owned buffer adopted without copying.
    Adopted(Arc<Vec<u8>>),
}

impl Storage {
    /// Adopt an existing buffer without copying. No alignment guarantee.
    pub fn from_vec(data: Vec<u8>) -> Self {
        Storage(Inner::Adopted(Arc::new(data)))
    }

    /// Copy `data` into a fresh 64-byte-aligned buffer.
    pub fn from_slice(data: &[u8]) -> Self {
        Self::aligned_with(data.len(), |buf| buf.extend_from_slice(data))
    }

    /// Allocate a zero-filled 64-byte-aligned buffer of `len` bytes.
    pub fn zeroed(len: usize) -> Self {
        Self::aligned_with(len, |buf| buf.resize(buf.len() + len, 0))
    }

    /// Build a 64-byte-aligned buffer by letting `fill` append exactly `len`
    /// payload bytes. `fill` must only append; growing past the reserved
    /// capacity would reallocate and lose the alignment.
    pub(crate) fn aligned_with(len: usize, fill: impl FnOnce(&mut Vec<u8>)) -> Self {
        let mut bytes = AlignedBytes::with_aligned_offset(len);
        fill(&mut bytes.buf);
        debug_assert_eq!(bytes.buf.len(), bytes.offset + bytes.len);
        Storage(Inner::Aligned(Arc::new(bytes)))
    }

    /// The full backing buffer.
    pub fn as_slice(&self) -> &[u8] {
        match &self.0 {
            Inner::Aligned(bytes) => bytes.as_slice(),
            Inner::Adopted(bytes) => bytes.as_slice(),
        }
    }

    /// Length of the backing buffer in bytes.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Whether the backing buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// Whether two storages share the same underlying allocation.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Inner::Aligned(a), Inner::Aligned(b)) => Arc::ptr_eq(a, b),
            (Inner::Adopted(a), Inner::Adopted(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// Owned bytes positioned at a 64-byte-aligned offset inside an
/// over-allocated `Vec`.
#[derive(Debug)]
struct AlignedBytes {
    buf: Vec<u8>,
    offset: usize,
    len: usize,
}

impl AlignedBytes {
    /// Reserve capacity for `len` payload bytes plus alignment padding and
    /// fill the padding. The reserved capacity guarantees later appends of up
    /// to `len` bytes never reallocate, so the offset computed from the heap
    /// pointer stays aligned.
    fn with_aligned_offset(len: usize) -> Self {
        let mut buf = Vec::with_capacity(len + ALIGN - 1);
        let addr = buf.as_ptr() as usize;
        let offset = (ALIGN - addr % ALIGN) % ALIGN;
        buf.resize(offset, 0);
        Self { buf, offset, len }
    }

    fn as_slice(&self) -> &[u8] {
        &self.buf[self.offset..self.offset + self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_storage_is_64_byte_aligned() {
        for len in [1usize, 7, 64, 100, 4096] {
            let storage = Storage::zeroed(len);
            assert_eq!(storage.as_slice().as_ptr() as usize % ALIGN, 0);
            assert_eq!(storage.len(), len);

            let storage = Storage::from_slice(&vec![7u8; len]);
            assert_eq!(storage.as_slice().as_ptr() as usize % ALIGN, 0);
            assert_eq!(storage.as_slice(), vec![7u8; len].as_slice());
        }
    }

    #[test]
    fn test_from_vec_adopts_without_copying() {
        let data = vec![1u8, 2, 3, 4];
        let heap_ptr = data.as_ptr();
        let storage = Storage::from_vec(data);
        assert_eq!(storage.as_slice().as_ptr(), heap_ptr);
        assert_eq!(storage.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_ptr_eq_tracks_shared_allocations() {
        let aligned = Storage::from_slice(&[1, 2, 3]);
        let adopted = Storage::from_vec(vec![1, 2, 3]);

        assert!(aligned.ptr_eq(&aligned.clone()));
        assert!(adopted.ptr_eq(&adopted.clone()));
        assert!(!aligned.ptr_eq(&adopted));
        assert!(!aligned.ptr_eq(&Storage::from_slice(&[1, 2, 3])));
        assert!(!adopted.ptr_eq(&Storage::from_vec(vec![1, 2, 3])));
    }

    #[test]
    fn test_zero_length_storage() {
        assert!(Storage::zeroed(0).is_empty());
        assert!(Storage::from_slice(&[]).is_empty());
        assert!(Storage::from_vec(Vec::new()).is_empty());
    }
}
