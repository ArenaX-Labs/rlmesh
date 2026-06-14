//! Resolved instructions for one model image input.

use super::super::spec::ImageLayout;

/// Resolved instructions for one model image input.
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePlan {
    pub model_key: String,
    pub env_key: String,
    pub src_layout: ImageLayout,
    pub dst_layout: ImageLayout,
    pub flip: bool,
    pub size: Option<(u32, u32)>,
    pub resample: String,
    pub dtype: String,
    pub normalize: bool,
    pub lead_dims: u32,
    /// Source pixel value range `(low, high)` from the env image's space, used
    /// to map a float image into 8-bit. `None` when the space is unbounded
    /// (the image is then assumed normalized `[0, 1]`).
    pub src_range: Option<(f64, f64)>,
}
