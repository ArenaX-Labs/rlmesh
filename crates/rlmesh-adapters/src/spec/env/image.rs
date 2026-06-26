//! A camera image entry in an environment observation.
//!
//! Internal post-`join` form; never serialized (see `spec::env`), so no serde.

use crate::path::NodePath;
use crate::spec::layouts::ImageLayout;

/// A camera image entry in an environment observation.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvImage {
    /// Structured source path into the raw observation tree this image is read
    /// from (the env-side placement); empty (root) for a bare single-leaf obs.
    pub source: NodePath,
    pub role: String,
    pub layout: ImageLayout,
    pub upside_down: bool,
    /// Pixel height of the image, derived from the observation space by
    /// [`join`](crate::v1::join). Lets a model that resizes only one axis
    /// fill the other from the env's native resolution.
    pub height: u32,
    /// Pixel width of the image, derived from the observation space.
    pub width: u32,
    /// Channel count of the image (the layout's channel axis), derived from the
    /// observation space. Used to reject a model expecting a different channel
    /// count (e.g. RGB vs grayscale) before it silently mis-feeds the model.
    pub channels: u32,
    /// Source pixel value range `(low, high)` projected from the space's
    /// uniform finite bounds, used to map a float image into 8-bit pixels.
    /// `None` when the space is unbounded or non-uniform.
    pub value_range: Option<(f64, f64)>,
}
