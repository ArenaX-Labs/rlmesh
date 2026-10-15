//! The wire space system ([`rlmesh_spaces`]): specs as public-field structs,
//! payloads, builders, values and the core tensor.
//!
//! Plumbing for the bindings and for the [`crate::Env`] / model traits until
//! they take the shared types; user code reaches spaces through
//! [`crate::spaces`]. The single-env request family stays namespaced under
//! `request`.

// Namespaced submodules. The single-env request family is reached via
// `rlmesh::binding::spaces::request::{ResetRequest, StepRequest, ...}`.
pub use rlmesh_spaces::{dtype, errors, meta, render, request, scalar, spaces, tensor, types};

pub use rlmesh_spaces::errors::{EnvRuntimeError, SpaceError};

// Curated flat re-exports. Note we intentionally do NOT flatten the request
// family here — use `request::*` for those — but `CloseRequest`,
// `RenderRequest`, and `RenderFrame`/`RenderResult` are shared and re-exported.
pub use rlmesh_spaces::dtype::{DType, dtype_size};
pub use rlmesh_spaces::meta::{MetaMap, MetaValue};
pub use rlmesh_spaces::render::{BinaryPayload, RenderFrame, RenderRequest, RenderResult};
pub use rlmesh_spaces::request::CloseRequest;
pub use rlmesh_spaces::scalar::{
    Scalar, ScalarError, decode_scalars, encode_i64_scalars, encode_scalars,
};
pub use rlmesh_spaces::spaces::{Conformance, SpaceValue, conform, contains, validate_space};
pub use rlmesh_spaces::tensor::{
    DLPackType, Device, Storage, Tensor, TensorError, contiguous_strides, dlpack_type,
    dtype_from_dlpack,
};
pub use rlmesh_spaces::types::{
    BoxBounds, BoxSpec, DictSpec, DiscreteSpec, ElementwiseBounds, EnvContract, MultiBinarySpec,
    MultiDiscreteSpec, SpaceKind, SpaceSpec, SpaceType, TextSpec, TupleSpec,
    TypedElementwiseBounds, TypedUniformBounds, UniformBounds,
};
