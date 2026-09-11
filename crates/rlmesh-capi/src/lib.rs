//! Experimental C ABI for RLMesh — a thin `extern "C"` projection of the core
//! model/environment runtime. Errors travel by value to the caller's return
//! thread (the thread-local last-error slot is the final hop only).

mod abi;
mod adapters;
mod codec;
mod model;
mod spaces;
mod value;

#[cfg(test)]
mod tests;
