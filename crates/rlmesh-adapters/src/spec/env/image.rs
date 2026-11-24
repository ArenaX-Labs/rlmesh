use crate::path::NodePath;
use crate::spec::layouts::ImageLayout;

/// A camera image entry in an environment observation.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvImage {
    /// Structured source path into the raw observation tree this image is read
    /// from (the env-side placement); empty (root) for a bare single-leaf obs.
    pub source: NodePath,
    pub role: String,
    /// The body part the camera sits on, lifted verbatim from the `ImageTag`.
    pub part: Option<String>,
    pub layout: ImageLayout,
    pub upside_down: bool,
    /// Pixel height of the image, derived from the observation space by
    /// [`join`](crate::v1::join).
    pub height: u32,
    /// Pixel width of the image, derived from the observation space.
    pub width: u32,
    /// Channel count of the image (the layout's channel axis), derived from the
    /// observation space. A model expecting a different channel count (RGB vs grayscale) is
    /// rejected rather than mis-fed.
    pub channels: u32,
    /// Source pixel value range `(low, high)` projected from the space's
    /// uniform finite bounds, used to map a float image into 8-bit pixels.
    pub value_range: Option<(f64, f64)>,
}
