//! Space specifications, values, and tensors for RLMesh.
//!
//! A [`SpaceSpec`] describes an observation or action space: Box, Discrete, MultiBinary,
//! MultiDiscrete and Text, plus the Dict and Tuple composites, as in Gymnasium. A
//! [`SpaceValue`] is a value drawn from one. [`contains`] and [`conform`] check membership,
//! [`sample_with`] draws seeded values, and [`flatten_leaves`]/[`assemble_value`] move between
//! a composite value and its ordered leaves. [`leaves`] encodes each leaf as raw
//! little-endian bytes, one value or a lane slab at a time.
//!
//! [`Tensor`] is an immutable n-dimensional view over shared [`Storage`] with DLPack-compatible
//! dtype, shape and strides; [`dlpack_type`] maps [`DType`] to DLPack type codes.
//!
//! Sampling and the scalar byte codec are shared with the Python bindings, so a given seed or
//! value encodes to identical bytes in both.

pub mod errors;

pub mod advisory;
mod display;
pub mod dtype;
pub mod leaves;
pub mod meta;
pub mod render;
pub mod request;
pub mod sample;
pub mod scalar;
pub mod spaces;
pub mod tensor;
pub mod types;

pub use advisory::{Advisory, AdvisorySeverity};
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
