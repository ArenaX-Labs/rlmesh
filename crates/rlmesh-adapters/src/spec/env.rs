//! The environment-side spec: observation features plus the action layout.

mod image;
mod state;
mod text;

use crate::path::NodePath;

use super::action::Action;

pub use image::EnvImage;
pub use state::EnvState;
pub use text::EnvText;

// EnvFeature/EnvFeatures (and the EnvImage/EnvState/EnvText leaves) are the
// internal post-`join` form the resolver consumes; they are never serialized
// (the authored, wire-serialized form is `EnvTags`). So they deliberately do
// NOT derive Serialize/Deserialize — that would read as a wire type.

/// One entry in an environment observation, declared by the env.
#[derive(Debug, Clone, PartialEq)]
pub enum EnvFeature {
    Image(EnvImage),
    State(EnvState),
    Text(EnvText),
}

/// An observation leaf the env declared with a kind this core does not define.
///
/// It produces no `EnvFeature` (an old core has no apply path for it), but is
/// recorded so the resolver can tell *referenced* from *unreferenced*: an unknown
/// kind whose `role` a model input asks for is a localized
/// [`UnsupportedKind`](crate::error::ErrorCode) error; an unreferenced one is
/// ignored with an advisory.
#[derive(Debug, Clone, PartialEq)]
pub struct UnknownFeature {
    pub source: NodePath,
    pub role: Option<String>,
    pub kind: String,
}

/// An environment's resolved observation features plus its action layout.
///
/// This is the internal, fully-keyed form the resolver consumes: every
/// feature carries its observation key and derived width/range. It is
/// produced by [`join`](crate::join::join) from the sparse
/// [`EnvTags`](super::env_tags::EnvTags) layered over a
/// gymnasium space — it is not authored or serialized directly by users.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvFeatures {
    pub observation: Vec<EnvFeature>,
    pub action: Action,
    /// Observation leaves of an unrecognized kind (see [`UnknownFeature`]).
    /// Empty on any spec this core fully understands.
    pub unknown: Vec<UnknownFeature>,
    /// Non-fatal hints derived from the env declaration alone (e.g. an image
    /// layout that looks mis-declared given its shape). Surfaced — not raised —
    /// at both join seams: authoring (`adapt.tag`) and serve-time resolve. Empty
    /// when nothing looks off.
    pub advisories: Vec<String>,
}
