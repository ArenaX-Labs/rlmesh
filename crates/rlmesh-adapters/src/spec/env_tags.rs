//! The sparse env-side tags users author over a gymnasium space.
//!
//! Tags carry only *semantics* — the role each observation entry plays
//! and how to interpret it (image layout, rotation encoding, value range).
//! All *structure* (keys' widths, dtypes, bounds) lives in the gymnasium
//! space and is derived by [`join`](crate::join::join), which validates
//! the tags against the space and produces the internal
//! [`EnvFeatures`](super::env::EnvFeatures) the resolver consumes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::action::ActionLayout;
use super::layouts::ImageLayout;
use super::rotations::RotationEncoding;

/// A camera image entry's semantics. Width/height/channels are derived from
/// the space, so only the layout (genuinely underdetermined by shape) and the
/// upside-down flag are carried.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageTag {
    pub role: String,
    #[serde(default)]
    pub layout: ImageLayout,
    #[serde(default)]
    pub upside_down: bool,
}

/// A numeric proprioception entry's semantics. The width is derived from the
/// space; an `encoding` declares a rotation representation (and its width is
/// then checked against the space) and `range` overrides infinite space
/// bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateTag {
    pub role: String,
    #[serde(default)]
    pub encoding: Option<RotationEncoding>,
    #[serde(default)]
    pub range: Option<(f64, f64)>,
}

/// A text entry's semantics (typically the task instruction).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextTag {
    pub role: String,
}

/// One contiguous field of a flat numeric observation leaf.
///
/// The observation-side mirror of [`ActionComponent`](super::action::ActionComponent):
/// a slice of `dim` elements carrying a `role`, with offsets implied by order
/// within a [`StateLayout`]. A field with no `role` is a *skip* — it advances
/// the offset and contributes to the layout's width but produces no feature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateField {
    #[serde(default)]
    pub role: Option<String>,
    pub dim: u32,
    #[serde(default)]
    pub encoding: Option<RotationEncoding>,
    #[serde(default)]
    pub range: Option<(f64, f64)>,
}

/// An ordered split of one flat numeric observation leaf into role fields.
///
/// The observation-side mirror of [`ActionLayout`](super::action::ActionLayout):
/// fields are laid out in order, offsets accumulate, and `join` requires the
/// field widths to sum to the leaf width. Use it when an env returns a flat
/// `Box` whose fixed index ranges carry distinct semantics (e.g. Metaworld).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateLayout {
    pub fields: Vec<StateField>,
}

/// One observation tag, tagged by the kind of space leaf it describes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ObsTag {
    Image(ImageTag),
    State(StateTag),
    Layout(StateLayout),
    Text(TextTag),
}

/// The env-side tags: a sparse map from observation key-path to its
/// semantics, plus the action layout.
///
/// Observation keys are space key-paths: a dotted path traverses nested
/// `Dict` spaces (`"robot.eef_pos"`), and the reserved key `"."` denotes a
/// flat/root observation (valid only when it is the sole entry). Untagged
/// space keys are allowed; they simply carry no semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvTags {
    pub observation: BTreeMap<String, ObsTag>,
    pub action: ActionLayout,
}
