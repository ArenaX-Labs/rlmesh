//! Resolved instructions for one model image input.

use crate::path::NodePath;
use crate::spec::{FitMode, ImageLayout};

/// Resolved instructions for one model image input.
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePlan {
    /// Where this image lands in the assembled payload tree.
    pub placement: NodePath,
    /// Where this image is read from in the raw observation tree (empty when
    /// `zero_fill` is set — a synthesized frame has no env source).
    pub source: NodePath,
    pub src_layout: ImageLayout,
    pub dst_layout: ImageLayout,
    pub flip: bool,
    pub size: Option<(u32, u32)>,
    /// How a target with a different aspect ratio than the source is reconciled.
    pub fit: FitMode,
    pub resample: String,
    pub dtype: String,
    /// Target value range to normalize 8-bit pixels into before the dtype cast,
    /// or `None` to skip normalization. `Some((0.0, 1.0))` is the conventional
    /// `/255`; the model may declare another range (e.g. `(-1.0, 1.0)`).
    pub normalize: Option<(f64, f64)>,
    pub lead_dims: u32,
    /// Source pixel value range `(low, high)` from the env image's space, used
    /// to map a float image into 8-bit. `None` when the space is unbounded
    /// (the image is then assumed normalized `[0, 1]`).
    pub src_range: Option<(f64, f64)>,
    /// Frame-stack depth: the model stacks this many consecutive frames on a new
    /// leading axis (frame history); `1` = no stacking. Buffered per-episode and
    /// stacked natively in the core (see [`crate::v1::FrameBuffers`]); only the keys with
    /// `stack > 1` carry a per-episode window.
    pub stack: u32,
    /// When `Some((height, width, channels))` this input has no env source: the
    /// adapter synthesizes a black HWC frame of that shape (an optional camera
    /// the env did not provide), then applies the normalize/dtype/layout/lead
    /// steps like a real frame. `None` for a normal image. `source` is the
    /// empty (root) path when this is set.
    pub zero_fill: Option<(u32, u32, u32)>,
    /// Raw 8-bit level the zero-filled frame is filled with (`0` = black, the
    /// default). Only meaningful when `zero_fill` is `Some`.
    pub absent_fill: u8,
}
