use std::borrow::Cow;

use prost::bytes::Bytes;
use rlmesh_spaces as native;

/// The tensor's element bytes as a wire-ready [`Bytes`].
///
/// A contiguous tensor shares its refcounted [`Storage`](native::Storage)
/// with the message. No element bytes are copied until the message is
/// serialized. Non-contiguous layouts gather into a fresh buffer, which
/// `Bytes` then adopts without a further copy.
pub(super) fn tensor_wire_bytes(tensor: &native::Tensor) -> Bytes {
    match tensor.to_contiguous_bytes() {
        Cow::Borrowed(_) => {
            let start = tensor.byte_offset();
            Bytes::from_owner(SharedStorage(tensor.storage().clone()))
                .slice(start..start + tensor.nbytes())
        }
        Cow::Owned(gathered) => Bytes::from(gathered),
    }
}

/// Adapter giving [`Bytes::from_owner`] a view of a tensor's refcounted
/// storage, keeping the allocation alive for the message's lifetime.
struct SharedStorage(native::Storage);

impl AsRef<[u8]> for SharedStorage {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}
