//! v1 spec types.
//!
//! `#[serde(deny_unknown_fields)]` is applied so a typo'd field is rejected, not
//! silently dropped: on the plain structs (EnvTags, ModelSpec, ActionLayout,
//! ActionComponent, StateComponent, StateField), on the wire structs that back
//! the try_from types (StateLayout), and -- since serde 1.0.228 honors it on an
//! internally-tagged variant (the `type` tag is stripped before the variant
//! deserializes) -- on the env-side ObsTag leaf tags (ImageTag/StateTag/TextTag).
//!
//! Still lenient: the ModelInput variant payloads (ImageInput/StateInput/
//! TextInput/CustomInput). Migrating them is now possible (the serde limitation
//! is gone) but not yet done; the Python from_dict mirror stays lenient on those
//! payloads to match.

mod accept_set;
mod action;
mod env;
mod env_tags;
mod layouts;
mod model;
mod num;
mod rotations;

pub use accept_set::AcceptSet;
pub use action::{ActionComponent, ActionLayout};
pub use env::{EnvFeature, EnvFeatures, EnvImage, EnvState, EnvText};
pub use env_tags::{EnvTags, ImageTag, ObsTag, StateField, StateLayout, StateTag, TextTag};
pub use layouts::{FitMode, ImageLayout};
pub use model::{
    CustomInput, ImageInput, ModelInput, ModelSpec, StateComponent, StateContainer, StateInput,
    TextContainer, TextInput,
};
pub use rotations::RotationEncoding;
