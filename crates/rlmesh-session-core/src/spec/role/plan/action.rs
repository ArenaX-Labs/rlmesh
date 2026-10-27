//! Resolved instructions for the action vector.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::spec::role::frames::FrameRef;
use crate::spec::rotation::Rotation;

/// Resolved mapping for one env action component.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionSegment {
    /// `None` for an opaque (role-less) segment -- see `fill`.
    pub role: Option<String>,
    /// The model action slice it reads, `[start, stop)`.
    pub start: u32,
    /// The slice's end.
    pub stop: u32,
    /// The model's rotation encoding.
    pub src_encoding: Option<Rotation>,
    /// The env's rotation encoding.
    pub dst_encoding: Option<Rotation>,
    /// The model's range, mapped into `dst_range` (unset when they agree).
    pub src_range: Option<(f64, f64)>,
    /// The env's range.
    pub dst_range: Option<(f64, f64)>,
    /// Model-side scalar corrections (the model's own output convention), applied
    /// after the format bridge and *before* the env-side corrections, in the order
    /// scale, invert, threshold. Unset for a model that emits the env's convention.
    pub model_scale: Option<f64>,
    /// The model's offset.
    pub model_offset: Option<f64>,
    /// The model's scale per axis.
    pub model_axis_scale: Option<Vec<f64>>,
    /// The model's offset per axis.
    pub model_axis_offset: Option<Vec<f64>>,
    /// The model negates its output.
    pub model_invert: bool,
    /// The model's binary decision boundary.
    pub model_threshold: Option<f64>,
    /// The `convert_units` factor, applied after the model-side corrections:
    /// the model's unit over the env's, net of the gain between them.
    pub unit_factor: Option<f64>,
    /// The `flip_gripper` reflection, applied after `unit_factor`: a value `v`
    /// becomes `flip - v` (the env range's `low + high`, else 0).
    pub flip: Option<f64>,
    /// Env-side scalar corrections, applied after the model-side ones in the order
    /// scale, offset, invert, threshold, `binarize`, then `clip`.
    pub scale: Option<f64>,
    /// The env's offset.
    pub offset: Option<f64>,
    /// The env's scale per axis.
    pub axis_scale: Option<Vec<f64>>,
    /// The env's offset per axis.
    pub axis_offset: Option<Vec<f64>>,
    /// The env negates the command.
    pub invert: bool,
    /// The env's binary decision boundary.
    pub threshold: Option<f64>,
    /// Per env axis, the model-slice index that drives it: a permutation.
    /// `None` when the model's order is the env's.
    pub scatter: Option<Vec<u32>>,
    /// The env actuator's labels and the model output's, for `describe`.
    pub labels: Option<Vec<String>>,
    /// The model output's labels.
    pub model_labels: Option<Vec<String>>,
    /// The command snaps to ±1.
    pub binarize: bool,
    /// Env-side per-actuator clamp bounds (the actuator's `range`), applied last.
    /// `None` means no per-component clamp (the global `ActionPlan.clip` still runs).
    pub clip: Option<(f64, f64)>,
    /// The agreed coordinate frame of an absolute pose command, and the agreed
    /// reference pose a delta is integrated against, when either side declared
    /// one. Rendered by `describe`; a disagreement is already a resolve error.
    pub frame: Option<FrameRef>,
    /// The agreed delta reference.
    pub reference: Option<FrameRef>,
    /// The body part this actuator was bound under, when a side declared one
    /// (the env actuator's own `part`, else the model output's). Rendered by
    /// `describe` as `#part`.
    pub part: Option<String>,
    /// An opaque segment: `Some((width, value))` emits `width` copies of `value`
    /// and reads nothing from the model (the env requires these dims but no model
    /// produces them). `None` for a normal model-mapped segment.
    pub fill: Option<(u32, f64)>,
}

/// Resolved instructions for the whole action vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionPlan {
    /// One segment per env actuator, in env order.
    pub segments: Vec<ActionSegment>,
    /// The env's clamp on the assembled vector, applied last.
    pub clip: Option<(f64, f64)>,
    /// The model action's width.
    pub in_dim: u32,
}
