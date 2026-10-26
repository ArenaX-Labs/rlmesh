//! What the Gym door binds: a model's leaves and actuators, and an env's
//! tagged features, lowered from their v1 declarations (`ModelSpec`,
//! `EnvTags` joined against the Gym spaces) by `rlmesh-adapters`.
//!
//! Lowering resolves defaults and vocabularies (names become [`Rotation`],
//! [`Resample`], [`Fit`], [`DType`]) and validates what a declaration alone
//! can get wrong; everything that depends on the other side is the binder's.
//! A declaration's conversions stay consent-by-declaration: a declared
//! target size is the consent to resize, a declared encoding the consent to
//! re-encode. The only conversions that need explicit consent are the named
//! [`Projection`]s in [`Spec::allow`].

use serde::{Deserialize, Serialize};

use crate::spec::plan::Fit;
use crate::spec::policy::{ImageLayout, Resample};
use crate::spec::projection::Projection;
use crate::spec::role::Advisory;
use crate::spec::role::frames::{FrameRef, Provenance};
use crate::spec::role::path::NodePath;
use crate::spec::rotation::{Matrix, Rotation};
use crate::wire::part::DType;

/// A tolerant accept set: every name as written, in preference order, and
/// the ones this core recognizes, in the same order. A peer's newer value
/// survives lowering and is refused only where a binding needs it.
#[derive(Debug, Clone, PartialEq)]
pub struct Accept<T> {
    /// Every name, as declared.
    pub names: Vec<String>,
    /// The recognized entries.
    pub known: Vec<T>,
}

impl<T: Copy + PartialEq> Accept<T> {
    /// The first recognized entry.
    pub fn first_known(&self) -> Option<T> {
        self.known.first().copied()
    }

    /// Whether `value` is accepted.
    pub fn accepts(&self, value: T) -> bool {
        self.known.contains(&value)
    }
}

/// Which way a gripper moves as its value rises: `open` (a higher command
/// opens it, a higher reading is more open: openness) or `close` (closedness).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Polarity {
    /// A higher value is more open.
    Open,
    /// A higher value is more closed.
    Close,
}

impl Polarity {
    /// Its wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Close => "close",
        }
    }

    /// The other polarity.
    pub fn flipped(self) -> Self {
        match self {
            Self::Open => Self::Close,
            Self::Close => Self::Open,
        }
    }

    /// The polarity after a side's own sign corrections: `invert` and a
    /// negative `scale` each reverse it.
    pub fn through(self, invert: bool, scale: Option<f64>) -> Self {
        if invert ^ scale.is_some_and(|scale| scale < 0.0) {
            self.flipped()
        } else {
            self
        }
    }
}

/// The host arms of a custom rotation encoding: a host-side repacking that
/// binds as its base encoding (the arms themselves run in the host).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CustomArms {
    /// The base → custom arm, which an observation needs.
    pub from_base: bool,
    /// The custom → base arm, which an action needs.
    pub to_base: bool,
}

/// What fills a frame window before an episode has produced enough frames.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StackPad {
    /// Replicate the first observed frame.
    #[default]
    First,
    /// A black (raw 8-bit `0`) frame pushed through the plan.
    Black,
}

/// A model's training-time center crop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crop {
    /// Side fraction kept on each axis.
    pub fraction: f64,
    /// The area fraction, when the spec declared it that way (describe only).
    pub area: Option<f64>,
    /// An integer cut before the resize (`slice`), rather than the box the
    /// resize samples through (`zoom`).
    pub slice: bool,
}

/// A model image input.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    /// The camera role it reads.
    pub role: String,
    /// The body part it reads.
    pub part: Option<String>,
    /// Target height (the env's when only the width is declared).
    pub height: Option<u32>,
    /// Target width.
    pub width: Option<u32>,
    /// Delivered layout.
    pub layout: ImageLayout,
    /// Expected channels.
    pub channels: Option<u32>,
    /// Delivered dtype.
    pub dtype: DType,
    /// `[0, 255]` normalised into this range before the cast.
    pub normalize: Option<(f64, f64)>,
    /// Unit axes prepended.
    pub lead_dims: u32,
    /// Trained on 180°-rotated frames.
    pub upside_down: bool,
    /// The resize kernel.
    pub resample: Resample,
    /// May interpolate past the camera's resolution.
    pub allow_upscale: bool,
    /// Accepted aspect fits, in preference order (`None`: undeclared).
    pub fit: Option<Accept<Fit>>,
    /// Zero-filled when the env has no such camera.
    pub optional: bool,
    /// The 8-bit level of that fill.
    pub fill: Option<u8>,
    /// Frames stacked on a leading axis (1: none).
    pub stack: u32,
    /// Which frames, as non-positive offsets ending at 0.
    pub offsets: Option<Vec<i32>>,
    /// The window's pad.
    pub stack_pad: StackPad,
    /// The training-time crop.
    pub crop: Option<Crop>,
    /// The JPEG quality the training frames were stored at.
    pub jpeg_quality: Option<u8>,
    /// Swap red and blue.
    pub bgr: bool,
    /// The camera size the model asserts it was trained on.
    pub render: Option<(u32, u32)>,
}

/// One part of a state concat.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    /// The env state role it reads (`None`: a constant).
    pub role: Option<String>,
    /// Reads the model's own previous action for the role, not the env.
    pub previous: bool,
    /// The accepted rotation encodings.
    pub encoding: Option<Accept<Rotation>>,
    /// A custom encoding's host arms (it binds as `encoding`'s one entry).
    pub custom: Option<CustomArms>,
    /// Elements kept.
    pub dim: Option<u32>,
    /// The one element kept.
    pub index: Option<u32>,
    /// Target range the env range maps into.
    pub range: Option<(f64, f64)>,
    /// Filled when the env lacks the role.
    pub optional: bool,
    /// A constant's value, an absent role's fill, or a previous action's
    /// before the first.
    pub fill: f64,
    /// A rotation right-multiplied before re-encoding.
    pub post_rotate: Option<Matrix>,
    /// Model-side scale.
    pub scale: Option<f64>,
    /// Model-side offset.
    pub offset: Option<f64>,
    /// Model-side scale per axis.
    pub axis_scale: Option<Vec<f64>>,
    /// Model-side offset per axis.
    pub axis_offset: Option<Vec<f64>>,
    /// The frame it was trained in.
    pub frame: Option<FrameRef>,
    /// The provenances it accepts.
    pub provenance: Option<Accept<Provenance>>,
    /// The body part it reads.
    pub part: Option<String>,
    /// Axis names, in the model's order.
    pub labels: Option<Vec<String>>,
    /// The gripper polarity it reads.
    pub polarity: Option<Polarity>,
}

/// A model state input: pieces concatenated, then clamped, padded and cast.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    /// The pieces, in order.
    pub pieces: Vec<Piece>,
    /// Zero-padded to this width.
    pub pad_to: Option<u32>,
    /// Delivered dtype.
    pub dtype: DType,
    /// Delivered shape (one `-1` inferred).
    pub reshape: Option<Vec<i64>>,
    /// The assembled vector clamped to this range, before the pad.
    pub clip: Option<(f64, f64)>,
    /// Delivered as a list of numbers rather than an array.
    pub list: bool,
}

/// A model text input.
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    /// The env text role it reads.
    pub role: String,
    /// Delivered when the env lacks the role.
    pub fill: Option<String>,
    /// Delivered as a one-item list rather than a string.
    pub list: bool,
}

/// A model input leaf.
#[derive(Debug, Clone, PartialEq)]
pub enum LeafKind {
    /// A camera.
    Image(Image),
    /// A state vector.
    State(State),
    /// Text.
    Text(Text),
    /// A host-language hole the host fills from the raw observation.
    Custom {
        /// The host reference (`module:callable` or `host:<key>`).
        transform: String,
    },
}

/// A model input leaf and where its value lands in the payload tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Leaf {
    /// Its placement.
    pub at: NodePath,
    /// What it is.
    pub kind: LeafKind,
}

/// An action component, on either side: one model output, or one env
/// actuator of the action vector, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Actuator {
    /// Its role (`None`: opaque dims, matched by nothing).
    pub role: Option<String>,
    /// Its width.
    pub dim: u32,
    /// Its rotation encoding.
    pub encoding: Option<Rotation>,
    /// A custom encoding's host arms (it binds as `encoding`).
    pub custom: Option<CustomArms>,
    /// Its range.
    pub range: Option<(f64, f64)>,
    /// Snapped to ±1.
    pub binary: bool,
    /// Scale.
    pub scale: Option<f64>,
    /// Offset.
    pub offset: Option<f64>,
    /// Scale per axis.
    pub axis_scale: Option<Vec<f64>>,
    /// Offset per axis.
    pub axis_offset: Option<Vec<f64>>,
    /// Negated.
    pub invert: bool,
    /// Binary decision boundary.
    pub threshold: Option<f64>,
    /// Clamped to `range` (env side).
    pub clip: bool,
    /// The constant an opaque or absent optional component takes (env side).
    pub fill: f64,
    /// Filled when the model does not output the role (env side).
    pub optional: bool,
    /// The frame of an absolute pose.
    pub frame: Option<FrameRef>,
    /// The pose a delta is integrated against.
    pub reference: Option<FrameRef>,
    /// The body part.
    pub part: Option<String>,
    /// Axis names.
    pub labels: Option<Vec<String>>,
    /// The gripper polarity.
    pub polarity: Option<Polarity>,
    /// The SI size of one delta unit.
    pub unit: Option<f64>,
}

/// A model's declaration, lowered.
#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    /// Its input leaves, in tree order.
    pub inputs: Vec<Leaf>,
    /// Its action layout, in order.
    pub output: Vec<Actuator>,
    /// The named projections it allows.
    pub allow: Vec<Projection>,
}

/// An env camera.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvImage {
    /// Where it is read from.
    pub source: NodePath,
    /// Its role.
    pub role: String,
    /// Its body part.
    pub part: Option<String>,
    /// Its layout.
    pub layout: ImageLayout,
    /// Rendered rotated 180°.
    pub upside_down: bool,
    /// Rows (0: unknown).
    pub height: u32,
    /// Columns (0: unknown).
    pub width: u32,
    /// Channels (0: unknown).
    pub channels: u32,
    /// A float image's value range.
    pub value_range: Option<(f64, f64)>,
}

/// An env state feature: a whole leaf, or one field of a split leaf.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvState {
    /// Where it is read from.
    pub source: NodePath,
    /// Its role.
    pub role: String,
    /// Its body part.
    pub part: Option<String>,
    /// Its offset in a split leaf.
    pub slice_offset: Option<u32>,
    /// Its width, when known.
    pub dim: Option<u32>,
    /// Its rotation encodings (the first recognized one is native).
    pub encoding: Option<Accept<Rotation>>,
    /// Its range.
    pub range: Option<(f64, f64)>,
    /// Its frame.
    pub frame: Option<FrameRef>,
    /// Where its numbers come from.
    pub provenance: Option<FrameRef>,
    /// Axis names.
    pub labels: Option<Vec<String>>,
    /// The gripper polarity.
    pub polarity: Option<Polarity>,
}

/// An env text feature.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvText {
    /// Where it is read from.
    pub source: NodePath,
    /// Its role.
    pub role: String,
}

/// An env observation leaf of a kind this core does not define.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvUnknown {
    /// Where it is.
    pub source: NodePath,
    /// Its role.
    pub role: Option<String>,
    /// Its kind.
    pub kind: String,
}

/// An env's tagged features and action layout, lowered from its joined tags.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Env {
    /// Cameras.
    pub images: Vec<EnvImage>,
    /// State features.
    pub states: Vec<EnvState>,
    /// Text features.
    pub texts: Vec<EnvText>,
    /// Unrecognized kinds.
    pub unknown: Vec<EnvUnknown>,
    /// The action layout, in order.
    pub action: Vec<Actuator>,
    /// The clamp on the assembled action.
    pub clip: Option<(f64, f64)>,
    /// What joining the tags noted.
    pub advisories: Vec<Advisory>,
}

#[cfg(test)]
mod tests {
    use super::Polarity;

    #[test]
    fn polarity_is_open_or_close_and_reverses_through_a_sign_correction() {
        let open: Polarity = serde_json::from_str(r#""open""#).unwrap();
        assert_eq!(open, Polarity::Open);
        assert!(serde_json::from_str::<Polarity>(r#""opened""#).is_err());
        assert_eq!(open.through(true, None), Polarity::Close);
        assert_eq!(open.through(true, Some(-2.0)), Polarity::Open);
        assert_eq!(open.through(false, Some(0.5)), Polarity::Open);
    }
}
