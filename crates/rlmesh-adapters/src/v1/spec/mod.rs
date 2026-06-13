//! v1 spec types.

mod action;
mod env;
mod env_annotations;
mod layouts;
mod model;
mod rotations;

pub use action::{ActionComponent, ActionLayout};
pub use env::{EnvFeature, EnvFeatures, EnvImage, EnvState, EnvText};
pub use env_annotations::{
    EnvAnnotations, ImageAnnotation, ObsAnnotation, StateAnnotation, TextAnnotation,
};
pub use layouts::ImageLayout;
pub use model::{
    CustomInput, ImageInput, ModelInput, ModelSpec, StateComponent, StateContainer, StateInput,
    TextContainer, TextInput,
};
pub use rotations::RotationEncoding;
