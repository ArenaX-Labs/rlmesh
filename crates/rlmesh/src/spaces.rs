//! Curated re-exports of the [`rlmesh_spaces`] space-system types.
//!
//! This module deliberately avoids a blanket `pub use rlmesh_spaces::*` so the
//! facade surface stays curated and the two request families do not collide:
//!
//! - The crate-root [`crate::ResetRequest`] / [`crate::StepRequest`] (and their
//!   results) are the **vectorized env-layer** requests (`seeds: Vec`,
//!   `actions: Vec`, batched observations) used by [`crate::RemoteEnv`].
//! - The **single-env** request family lives under [`request`]
//!   (`rlmesh::spaces::request::ResetRequest`, ...) and is what the
//!   [`crate::SingleEnv`] trait consumes.
//!
//! Keeping the single-env family namespaced under `request` removes the
//! same-named `rlmesh::ResetRequest` vs `rlmesh::spaces::ResetRequest` clash.

// Namespaced submodules. The single-env request family is reached via
// `rlmesh::spaces::request::{ResetRequest, StepRequest, ResetResult, ...}`.
pub use rlmesh_spaces::{dtype, errors, meta, render, request, scalar, spaces, tensor, types};

pub use rlmesh_spaces::errors::{EnvRuntimeError, SpaceError};

// Curated flat re-exports. Note we intentionally do NOT flatten the single-env
// request family (ResetRequest/StepRequest/ResetResult/StepResult/CloseResult)
// here — use `request::*` for those — but `CloseRequest`, `RenderRequest`, and
// `RenderFrame`/`RenderResult` are shared with the env layer and re-exported.
pub use rlmesh_spaces::dtype::{DType, dtype_size};
pub use rlmesh_spaces::meta::{MetaMap, MetaValue};
pub use rlmesh_spaces::render::{BinaryPayload, RenderFrame, RenderRequest, RenderResult};
pub use rlmesh_spaces::request::CloseRequest;
pub use rlmesh_spaces::scalar::{
    Scalar, ScalarError, decode_scalars, encode_i64_scalars, encode_scalars,
};
pub use rlmesh_spaces::spaces::{SpaceValue, contains, validate_space};
pub use rlmesh_spaces::tensor::{
    DLPackType, Device, Storage, Tensor, TensorError, contiguous_strides, dlpack_type,
    dtype_from_dlpack,
};
pub use rlmesh_spaces::types::{
    AxiswiseBounds, BoxBounds, BoxSpec, DictSpec, DiscreteSpec, ElementwiseBounds, EnvContract,
    MultiBinaryDims, MultiBinarySpec, MultiDiscreteNvec, MultiDiscreteSpec, SpaceKind, SpaceSpec,
    SpaceType, TextSpec, TupleSpec, UniformBounds,
};
