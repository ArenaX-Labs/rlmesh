//! The sparse env-side annotations users author over a gymnasium space.
//!
//! Annotations carry only *semantics* — the role each observation entry plays
//! and how to interpret it (image layout, rotation encoding, value range).
//! All *structure* (keys' widths, dtypes, bounds) lives in the gymnasium
//! space and is derived by [`join`](super::super::join::join), which validates
//! the annotations against the space and produces the internal
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
pub struct ImageAnnotation {
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
pub struct StateAnnotation {
    pub role: String,
    #[serde(default)]
    pub encoding: Option<RotationEncoding>,
    #[serde(default)]
    pub range: Option<(f64, f64)>,
}

/// A text entry's semantics (typically the task instruction).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextAnnotation {
    pub role: String,
}

/// One observation annotation, tagged by the kind of space leaf it describes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ObsAnnotation {
    Image(ImageAnnotation),
    State(StateAnnotation),
    Text(TextAnnotation),
}

impl ObsAnnotation {
    /// The semantic role this annotation assigns.
    pub fn role(&self) -> &str {
        match self {
            ObsAnnotation::Image(image) => &image.role,
            ObsAnnotation::State(state) => &state.role,
            ObsAnnotation::Text(text) => &text.role,
        }
    }
}

/// The env-side annotations: a sparse map from observation key-path to its
/// semantics, plus the action layout.
///
/// Observation keys are space key-paths: a dotted path traverses nested
/// `Dict` spaces (`"robot.eef_pos"`), and the reserved key `"."` denotes a
/// flat/root observation (valid only when it is the sole entry). Unannotated
/// space keys are allowed; they simply carry no semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvAnnotations {
    pub observation: BTreeMap<String, ObsAnnotation>,
    pub action: ActionLayout,
}
