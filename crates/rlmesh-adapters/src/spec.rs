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
//! Still strict at the serde layer (cross-field `TryFrom` validators):
//! the wire structs (FieldWire, SplitLayoutWire, ConcatPartWire, ActionWire) and
//! the fixed containers (EnvTags, ModelSpec, Action) keep `deny_unknown_fields`.
//!
//! The two specs are **recursive trees** (`ObsNode`, `InputNode`) whose
//! container type = the runtime container type; the tree node discriminant is
//! structural (a JSON array → Tuple, an object with a leaf `"type"` → Leaf, an
//! object with an unknown string `"type"` → `Unknown` leaf, else a Dict), so
//! `"type"` is a reserved Dict key.

mod accept_set;
mod action;
mod env;
mod env_tags;
mod layouts;
mod model;
mod num;
mod rotations;
mod strict;

pub use accept_set::AcceptSet;
pub use action::{Action, Actuator};
pub use env::{EnvFeature, EnvFeatures, EnvImage, EnvState, EnvText, UnknownFeature};
pub use env_tags::{EnvTags, Field, ImageTag, ObsLeaf, ObsNode, SplitLayout, StateTag, TextTag};
pub use layouts::{FitMode, ImageLayout};
pub use model::{
    ConcatPart, Custom, Image, InputNode, ModelLeaf, ModelSpec, State, StateContainer, Text,
    TextContainer,
};
pub use rotations::RotationEncoding;
pub use strict::{
    reject_bare_fields_env, reject_bare_fields_model, reject_unknowns_env, reject_unknowns_model,
};
