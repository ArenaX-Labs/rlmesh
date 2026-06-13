//! The environment-side spec: observation features plus the action layout.

mod image;
mod state;
mod text;

use serde::{Deserialize, Serialize};

use super::action::ActionLayout;

pub use image::EnvImage;
pub use state::EnvState;
pub use text::EnvText;

/// One entry in an environment observation, declared by the env.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum EnvFeature {
    Image(EnvImage),
    State(EnvState),
    Text(EnvText),
}

/// Declarative description of an environment's observation and action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvIoSpec {
    pub observation: Vec<EnvFeature>,
    pub action: ActionLayout,
}
