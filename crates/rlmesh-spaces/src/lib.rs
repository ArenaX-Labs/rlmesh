//! Space specifications, values, and tensors for RLMesh.
//!
//! A [`SpaceSpec`] is the pure-data description of an observation or action
//! space: the fundamentals Box, Discrete, MultiBinary, MultiDiscrete, and Text,
//! plus the Dict and Tuple composites, mirroring Gymnasium's taxonomy. A
//! [`SpaceValue`] is a concrete value drawn from one. [`contains`] and
//! [`conform`] check membership, [`sample_with`] draws seeded values, and
//! [`flatten_leaves`]/[`assemble_value`] move between a composite value and its
//! ordered fundamental leaves.
//!
//! Tensor payloads use [`Tensor`], an immutable n-dimensional view over shared
//! [`Storage`] with DLPack-compatible dtype, shape, and strides. [`dlpack_type`]
//! maps the [`DType`] table to and from DLPack data-type codes for zero-copy
//! exchange with numpy and the framework bridges.
//!
//! This crate carries no `rlmesh-proto` dependency: the [`DType`] and
//! [`SpaceType`] discriminants are kept byte-identical to the proto enums and
//! cross-checked in `rlmesh-grpc`. Sampling and the scalar byte codec are the
//! single cross-language implementations, so a given seed or value encodes to
//! identical bytes in Rust and the Python bindings.

pub mod errors;

mod display;
pub mod dtype;
pub mod meta;
pub mod render;
pub mod request;
pub mod sample;
pub mod scalar;
pub mod spaces;
pub mod tensor;
pub mod types;

pub use dtype::{DType, dtype_size};
pub use meta::{MetaMap, MetaValue};
pub use render::{BinaryPayload, RenderFrame, RenderRequest, RenderResult};
pub use request::{CloseRequest, CloseResult, ResetRequest, ResetResult, StepRequest, StepResult};
pub use sample::{ChaCha12Rng, sample_seeded, sample_with};
pub use scalar::{
    Scalar, ScalarError, check_int_in_dtype_range, decode_scalars, encode_i64_scalars,
    encode_scalars, f64_to_f16_bits,
};
pub use spaces::{
    Conformance, PolicyOutcome, SpaceValue, ValidationPolicy, assemble_value, conform, contains,
    flatten_leaves, leaf_specs, validate_space,
};
pub use tensor::{
    DLPackType, Device, Storage, Tensor, TensorError, contiguous_strides, dlpack_type,
    dtype_from_dlpack,
};
pub use types::{
    AutoresetMode, BoxBounds, BoxSpec, DictSpec, DiscreteSpec, ElementwiseBounds, EnvContract,
    MultiBinarySpec, MultiDiscreteSpec, SpaceKind, SpaceSpec, SpaceType, TextSpec, TupleSpec,
    TypedElementwiseBounds, TypedUniformBounds, UniformBounds, UnknownAutoresetMode,
};
