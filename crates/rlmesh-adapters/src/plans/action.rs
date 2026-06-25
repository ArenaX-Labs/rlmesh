//! Resolved instructions for the action vector.

use crate::spec::RotationEncoding;

/// Resolved mapping for one env action component.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionSegment {
    pub role: String,
    pub start: u32,
    pub stop: u32,
    pub src_encoding: Option<RotationEncoding>,
    pub dst_encoding: Option<RotationEncoding>,
    pub src_range: Option<(f64, f64)>,
    pub dst_range: Option<(f64, f64)>,
    /// Env-side scalar corrections, applied after rotation/range bridging in the
    /// order scale, invert, threshold, then `binarize`.
    pub scale: Option<f64>,
    pub invert: bool,
    pub threshold: Option<f64>,
    pub binarize: bool,
    pub out_dim: u32,
}

/// Resolved instructions for the whole action vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionPlan {
    pub segments: Vec<ActionSegment>,
    pub clip: Option<(f64, f64)>,
    pub in_dim: u32,
    /// Number of model actions to replay per predicted chunk before re-planning;
    /// `1` = predict every step. Resolved from the *model's* `ActionLayout`; the
    /// per-episode replay queue lives in [`crate::stateful::ChunkBuffers`].
    pub execute_horizon: u32,
}
