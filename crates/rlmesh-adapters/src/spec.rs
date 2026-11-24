//! v1 spec types.
//!
//! The serde codec is a tolerant reader: every growable leaf carries a flattened capture map for
//! unrecognized fields, and an unrecognized leaf `type` parses into an `Unknown` arm, so any
//! structurally valid spec round-trips without loss. Strictness is a separate post-parse pass,
//! [`reject_unknowns_env`] and [`reject_unknowns_model`] (see [`strict`]): publishing runs it,
//! reading does not, so an unsupported feature surfaces only at resolve when a model input
//! references it. The envelope wire structs and fixed containers (`EnvTags`, `ModelSpec`,
//! `Action`) still deny unknown fields, and every value guard (`dim >= 1`, a non-reversed `range`)
//! is a hard parse error.
//!
//! Both specs are recursive trees (`ObsNode`, `InputNode`) whose container type is the runtime
//! container type. The node discriminant is structural (an array is a Tuple, an object with a leaf
//! `"type"` is a Leaf, one with an unknown string `"type"` an `Unknown` leaf, else a Dict), so
//! `"type"` is a reserved Dict key.

mod accept_set;
mod action;
mod conventions;
mod custom_encoding;
mod env;
mod env_tags;
pub(crate) mod labels;
mod layouts;
mod leaf_codec;
mod model;
mod num;
mod rotation_literal;
mod rotations;
pub(crate) mod strict;

pub use accept_set::AcceptSet;
pub use action::{Action, Actuator};
pub use conventions::Polarity;
pub use custom_encoding::{ActionEncoding, CustomEncoding, StateEncoding};
pub use env::{EnvFeature, EnvFeatures, EnvImage, EnvState, EnvText, UnknownFeature};
pub use env_tags::{EnvTags, Field, ImageTag, ObsLeaf, ObsNode, SplitLayout, StateTag, TextTag};
pub use layouts::{FitMode, ImageLayout};
pub use model::{
    CHANNEL_ORDERS, CROP_MODES, ConcatPart, Custom, Image, InputNode, ModelLeaf, ModelSpec,
    Normalize, PartSource, StackPad, State, StateContainer, Text, TextContainer, Timing,
    render_requests,
};
pub use rlmesh_session_core::spec::role::frames::{
    Attr, FRAMES, FrameLaw, FrameRef, PROVENANCES, Provenance, REFERENCES, ReferenceLaw,
};
pub use rotation_literal::RotationLiteral;
pub use rotations::RotationEncoding;
pub use strict::{
    FramePolicy, RolePolicy, reject_bare_fields_env, reject_bare_fields_model,
    reject_unframed_roles_env, reject_unframed_roles_model, reject_unknowns_env,
    reject_unknowns_model, reject_unsanctioned_roles_env, reject_unsanctioned_roles_model,
};
