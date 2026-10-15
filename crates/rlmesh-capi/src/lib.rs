//! Experimental C ABI for RLMesh — a thin `extern "C"` projection of the core
//! model/environment runtime. Errors travel by value to the caller's return
//! thread (the thread-local last-error slot is the final hop only).

mod abi;
mod adapters;
mod build;
mod cancel;
mod codec;
mod contract;
mod describe;
mod env;
mod env_api;
mod env_factory;
mod env_server;
mod ffi;
mod model;
mod space_handle;
mod spaces;
mod value;
mod variant;

#[cfg(test)]
mod tests;
