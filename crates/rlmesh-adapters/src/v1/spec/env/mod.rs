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

/// An environment's resolved observation features plus its action layout.
///
/// This is the internal, fully-keyed form the resolver consumes: every
/// feature carries its observation key and derived width/range. It is
/// produced by [`join`](super::super::join::join) from the sparse
/// [`EnvTags`](super::env_tags::EnvTags) layered over a
/// gymnasium space — it is not authored or serialized directly by users.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvFeatures {
    pub observation: Vec<EnvFeature>,
    pub action: ActionLayout,
}
