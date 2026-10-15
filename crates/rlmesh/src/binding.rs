//! Plumbing for the in-repo bindings (the Python extension and the C ABI).
//!
//! Hidden from the docs and exempt from semver: nothing here is part of the
//! shared surface, and it changes with the bindings.

use crate::ServeOptions;

/// The transport's options for `options` (the env server the Python binding
/// hosts itself).
pub fn serve_options_to_grpc(options: ServeOptions) -> rlmesh_grpc::ServeOptions {
    options.to_grpc()
}
