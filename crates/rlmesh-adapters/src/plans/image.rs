//! Resolved instructions for one model image input.

use crate::path::NodePath;
use crate::spec::{FitMode, ImageLayout};

/// The center box a `crop` / `crop_area` keeps, and how it is taken.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CropPlan {
    /// Side fraction of the source kept on each axis (the square root of a
    /// declared `crop_area`).
    pub fraction: f64,
    /// The *area* fraction the spec declared, when it declared one; carried so
    /// describe can print it next to the side fraction it works out to.
    pub area: Option<f64>,
    /// `true`: an integer center cut taken *before* the resize (`crop_mode =
    /// "slice"`). `false`: the fractional box the resize itself samples
    /// through, straight to the target (`crop_mode = "zoom"`).
    pub slice: bool,
    /// The integer box a `slice` takes at the env camera's declared resolution
    /// — describe text only (apply cuts from the frame it is actually handed).
    /// `None` when zooming, or when the env's resolution is unknown.
    pub cut: Option<(u32, u32)>,
}

/// Resolved instructions for one model image input.
///
/// `#[non_exhaustive]`: the image pipeline keeps gaining declared steps, and a
/// downstream literal would break on each one.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
    pub fill: u8,
    /// JPEG quality to round-trip the upright frame through before the crop
    /// and resize, reproducing the codec artifacts the model was trained on;
    /// `None` for an untouched frame.
    pub jpeg_quality: Option<u8>,
    /// The center box to keep before (or as part of) the resize; `None` for
    /// the whole frame.
    pub crop: Option<CropPlan>,
    /// Swap the red and blue channels after the spatial ops and before the
    /// dtype cast (`channel_order = "bgr"`); needs a 3-channel image.
    pub swap_rb: bool,
    /// The `[height, width]` the model asserted the bound camera renders at,
    /// when it asserted one and the camera's resolution was derivable (so the
    /// assertion was actually checked). Describe text only — the assertion
    /// itself is settled at resolve.
    pub render: Option<(u32, u32)>,
    /// `Some((requested, bound))` when the lone-camera fallback bound this input
    /// to the env's single camera under a different role than the model asked
    /// for. Surfaced as a resolve advisory; `None` for an exact role match.
    pub role_rebound: Option<(String, String)>,
}
