//! Core Value handles, opaque facade tensors and the dtype vocabulary.

pub(crate) mod dtype;
pub(crate) mod handle;
pub(crate) mod tensor;

#[cfg(test)]
mod tests;
