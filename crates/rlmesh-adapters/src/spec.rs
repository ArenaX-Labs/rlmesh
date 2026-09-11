//! v1 spec types.
//!
//! **Tolerant reader.** The serde codec is unconditionally tolerant: every
//! growable leaf (ImageTag/StateTag/TextTag/Actuator; Image/State/Text/Custom)
//! carries a `#[serde(flatten)]` capture map for unrecognized fields, and an
//! unrecognized leaf `type` parses into an `Unknown` arm. Any structurally-valid
//! spec round-trips without loss, so a newer peer's additive field or new
//! modality survives an older core. Strictness is decoupled into a separate
//! post-parse pass, [`reject_unknowns_env`]/[`reject_unknowns_model`] (see
//! [`strict`]): the PUBLISH doors run it so a typo dies at the trust boundary,
//! the READ door does not, surfacing an unsupported feature only at resolve and
//! only when a model input references it.
//!
//! The growable *inner* leaves (`Field` inside a split layout, `ConcatPart`
//! inside a state input) follow the same rule: a reader tolerates an
//! unrecognized field and carries it through, and the publish gate rejects it.
//!
//! Still strict at the serde layer (cross-field `TryFrom` validators): the
//! envelope wire structs (SplitLayoutWire, ActionWire) and the fixed containers
//! (EnvTags, ModelSpec, Action) keep `deny_unknown_fields`, and every value
//! guard (`dim >= 1`, a non-reversed `range`) stays a hard parse error.
//!
//! The two specs are **recursive trees** (`ObsNode`, `InputNode`) whose
//! container type = the runtime container type; the tree node discriminant is
//! structural (a JSON array → Tuple, an object with a leaf `"type"` → Leaf, an
//! object with an unknown string `"type"` → `Unknown` leaf, else a Dict), so
//! `"type"` is a reserved Dict key.

mod accept_set;
mod action;
mod custom_encoding;
mod env;
mod env_tags;
mod frames;
mod layouts;
mod leaf_codec;
mod model;
mod num;
mod rotation_literal;
mod rotations;
mod strict;

pub use accept_set::AcceptSet;
pub use action::{Action, Actuator};
pub use custom_encoding::{ActionEncoding, CustomEncoding, StateEncoding};
pub use env::{EnvFeature, EnvFeatures, EnvImage, EnvState, EnvText, UnknownFeature};
pub use env_tags::{EnvTags, Field, ImageTag, ObsLeaf, ObsNode, SplitLayout, StateTag, TextTag};
pub use frames::{Attr, FRAMES, FrameLaw, FrameRef, REFERENCES, ReferenceLaw};
pub use layouts::{FitMode, ImageLayout};
pub use model::{
    CHANNEL_ORDERS, CROP_MODES, ConcatPart, Custom, Image, InputNode, ModelLeaf, ModelSpec,
    Normalize, State, StateContainer, Text, TextContainer,
};
pub use rotation_literal::RotationLiteral;
pub use rotations::RotationEncoding;
pub use strict::{
    FramePolicy, RolePolicy, reject_bare_fields_env, reject_bare_fields_model,
    reject_unframed_roles_env, reject_unframed_roles_model, reject_unknowns_env,
    reject_unknowns_model, reject_unsanctioned_roles_env, reject_unsanctioned_roles_model,
};
