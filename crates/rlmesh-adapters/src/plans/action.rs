//! Resolved instructions for the action vector.

use crate::spec::RotationEncoding;

/// Resolved mapping for one env action component.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionSegment {
    /// `None` for an opaque (role-less) segment -- see `fill`.
    pub role: Option<String>,
    pub start: u32,
    pub stop: u32,
    pub src_encoding: Option<RotationEncoding>,
    pub dst_encoding: Option<RotationEncoding>,
    pub src_range: Option<(f64, f64)>,
    pub dst_range: Option<(f64, f64)>,
    /// Model-side scalar corrections (the model's own output convention), applied
    /// after the format bridge and *before* the env-side corrections, in the order
    /// scale, invert, threshold. Unset for a model that emits the env's convention.
    pub model_scale: Option<f64>,
    pub model_invert: bool,
    pub model_threshold: Option<f64>,
    /// Env-side scalar corrections, applied after the model-side ones in the order
    /// scale, invert, threshold, `binarize`, then `clip`.
    pub scale: Option<f64>,
    pub invert: bool,
    pub threshold: Option<f64>,
    pub binarize: bool,
    /// Env-side per-actuator clamp bounds (the actuator's `range`), applied last.
    /// `None` means no per-component clamp (the global `ActionPlan.clip` still runs).
    pub clip: Option<(f64, f64)>,
    /// An opaque segment: `Some((width, value))` emits `width` copies of `value`
    /// and reads nothing from the model (the env requires these dims but no model
    /// produces them). `None` for a normal model-mapped segment.
    pub fill: Option<(u32, f64)>,
}

/// Resolved instructions for the whole action vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionPlan {
    pub segments: Vec<ActionSegment>,
    pub clip: Option<(f64, f64)>,
    pub in_dim: u32,
}
