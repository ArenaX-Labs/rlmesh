//! v1 spec types.
//!
//! `#[serde(deny_unknown_fields)]` is applied to the plain structs (EnvTags,
//! ModelSpec, ActionLayout, ActionComponent, StateComponent, StateField) so a
//! typo'd field is rejected, not silently dropped. It is deliberately NOT on
//! the internally-tagged-enum variant payloads (ImageInput/StateInput/
//! TextInput/CustomInput under ModelInput; ImageTag/StateTag/TextTag/StateLayout
//! under ObsTag) -- serde treats the `type` tag as an unknown field there and
//! would break deserialization. Strict unknown-field rejection for those lands
//! with the Rust normalize door (a manual key check); until then the Python
//! from_dict mirror also stays lenient on those payloads.

mod action;
mod env;
mod env_tags;
mod layouts;
mod model;
mod num;
mod rotations;

pub use action::{ActionComponent, ActionLayout};
pub use env::{EnvFeature, EnvFeatures, EnvImage, EnvState, EnvText};
pub use env_tags::{EnvTags, ImageTag, ObsTag, StateField, StateLayout, StateTag, TextTag};
pub use layouts::ImageLayout;
pub use model::{
    CustomInput, ImageInput, ModelInput, ModelSpec, StateComponent, StateContainer, StateInput,
    TextContainer, TextInput,
};
pub use rotations::RotationEncoding;
