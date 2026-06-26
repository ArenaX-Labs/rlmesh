//! v1 spec types.
//!
//! `#[serde(deny_unknown_fields)]` is applied so a typo'd field is rejected, not
//! silently dropped: on the plain structs (EnvTags, ModelSpec, Action,
//! Actuator, ConcatPart, Field), on the wire structs that back the try_from
//! types (SplitLayout, State), and -- since serde 1.0.228 honors it on an
//! internally-tagged variant (the `type` tag is stripped before the variant
//! deserializes) -- on the env-side ObsLeaf leaf tags (ImageTag/StateTag/TextTag).
//!
//! Still lenient: the ModelLeaf variant payloads (Image/State/Text/Custom).
//! Migrating them is now possible (the serde limitation is gone) but not yet
//! done; the Python from_dict mirror stays lenient on those payloads to match.
//!
//! The two specs are **recursive trees** (`ObsNode`, `InputNode`) whose
//! container type = the runtime container type; the tree node discriminant is
//! structural (a JSON array → Tuple, an object with a leaf `"type"` → Leaf,
//! else a Dict), so `"type"` is a reserved Dict key.

mod accept_set;
mod action;
mod env;
mod env_tags;
mod layouts;
mod model;
mod num;
mod rotations;

pub use accept_set::AcceptSet;
pub use action::{Action, Actuator};
pub use env::{EnvFeature, EnvFeatures, EnvImage, EnvState, EnvText};
pub use env_tags::{EnvTags, Field, ImageTag, ObsLeaf, ObsNode, SplitLayout, StateTag, TextTag};
pub use layouts::{FitMode, ImageLayout};
pub use model::{
    ConcatPart, Custom, Image, InputNode, ModelLeaf, ModelSpec, State, StateContainer, Text,
    TextContainer,
};
pub use rotations::RotationEncoding;
