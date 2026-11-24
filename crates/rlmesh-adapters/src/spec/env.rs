mod image;
mod state;
mod text;

use crate::path::NodePath;

use super::action::Action;

pub use image::EnvImage;
pub use state::EnvState;
pub use text::EnvText;

/// One entry in an environment observation, declared by the env.
#[derive(Debug, Clone, PartialEq)]
pub enum EnvFeature {
    Image(EnvImage),
    State(EnvState),
    Text(EnvText),
}

/// An observation leaf the env declared with a kind this core does not define.
///
/// It produces no [`EnvFeature`]. A model input that references its `role` fails resolution
/// with `UnsupportedKind`; an unreferenced one is ignored with an advisory.
#[derive(Debug, Clone, PartialEq)]
pub struct UnknownFeature {
    pub source: NodePath,
    pub role: Option<String>,
    pub kind: String,
}

/// An environment's resolved observation features plus its action layout.
///
/// The fully keyed form the resolver consumes, produced by [`join`](crate::join::join) from
/// sparse [`EnvTags`](super::env_tags::EnvTags) over a gymnasium space; not authored or
/// serialized directly.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvFeatures {
    pub observation: Vec<EnvFeature>,
    pub action: Action,
    /// Observation leaves of an unrecognized kind (see [`UnknownFeature`]).
    pub unknown: Vec<UnknownFeature>,
    /// Non-fatal hints derived from the env declaration alone (e.g. an image
    /// layout that looks mis-declared given its shape).
    pub advisories: Vec<crate::advisory::Advisory>,
}
