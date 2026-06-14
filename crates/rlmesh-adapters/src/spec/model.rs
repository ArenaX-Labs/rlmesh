//! The model-side spec: expected input payload plus the action output.

mod custom;
mod image;
mod state;
mod text;

use serde::{Deserialize, Serialize};

use super::action::ActionLayout;

pub use custom::CustomInput;
pub use image::ImageInput;
pub use state::{StateComponent, StateContainer, StateInput};
pub use text::{TextContainer, TextInput};

/// One input feature expected by a model, declared by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ModelInput {
    Image(ImageInput),
    State(StateInput),
    Text(TextInput),
    Custom(CustomInput),
}

/// Declarative description of a model's input payload and action output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub inputs: Vec<ModelInput>,
    pub action: ActionLayout,
}
