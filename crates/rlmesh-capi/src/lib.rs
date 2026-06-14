//! Experimental C ABI for RLMesh — a thin `extern "C"` projection of the core
//! model/environment runtime.
//!
//! Each module that defines FFI entry points scopes `#![allow(unsafe_code)]` with a
//! justifying note; the crate is otherwise held to the workspace lints. Errors travel
//! by value to the caller's return thread (the thread-local last-error slot is the
//! final hop only).

mod abi;
mod codec;
mod model;
mod spaces;
mod value;

#[cfg(test)]
mod tests;
