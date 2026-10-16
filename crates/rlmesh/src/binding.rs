//! Plumbing for the in-repo bindings (the Python extension and the C ABI).
//!
//! Hidden from the docs and exempt from semver: nothing here is part of the
//! shared surface, and it changes with the bindings.

pub mod spaces;

use crate::{Error, Result, ServeOptions, Tensor};

/// The transport's options for `options` (the env server the Python binding
/// hosts itself).
pub fn serve_options_to_grpc(options: ServeOptions) -> rlmesh_grpc::ServeOptions {
    options.to_grpc()
}

/// The shared [`crate::spaces::SpaceSpec`] for a wire spec.
pub fn space_spec_from_wire(spec: rlmesh_spaces::SpaceSpec) -> crate::spaces::SpaceSpec {
    crate::spaces::SpaceSpec::from_wire(spec)
}

/// The wire spec behind a shared [`crate::spaces::SpaceSpec`].
pub fn space_spec_wire(spec: &crate::spaces::SpaceSpec) -> &rlmesh_spaces::SpaceSpec {
    spec.wire()
}

/// The shared [`Tensor`] for a core tensor; refuses an unspecified dtype.
pub fn tensor_from_core(tensor: rlmesh_spaces::Tensor) -> Result<Tensor> {
    if tensor.dtype() == rlmesh_spaces::DType::Unspecified {
        return Err(Error::invalid_argument("tensor dtype must be specified"));
    }
    Ok(Tensor::wrap(tensor))
}

/// The core tensor behind a shared [`Tensor`].
pub fn tensor_core(tensor: &Tensor) -> &rlmesh_spaces::Tensor {
    tensor.core()
}

/// A per-element Box (the C and C++ `ext` builders; SPC-22): `low` and `high`
/// hold one bound per element of `shape`.
pub fn box_elementwise(
    low: &[crate::spaces::Bound],
    high: &[crate::spaces::Bound],
    shape: Vec<i64>,
    options: crate::spaces::BoxOptions,
) -> Result<crate::spaces::Box> {
    let spec = crate::spaces::box_space::box_elementwise_spec(low, high, shape, options.dtype)?;
    crate::spaces::Box::from_spec(spec)
}
