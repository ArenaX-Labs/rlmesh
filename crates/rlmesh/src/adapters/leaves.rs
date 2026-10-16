//! Immutable leaf values used to author adapter trees.
use super::*;
use crate::DType;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "Image::empty", deny_unknown_fields)]
/// An image input expected by a model.
pub struct Image {
    /// Semantic role matched against env image features.
    pub role: String,
    /// Target image height, or None to keep the env height.
    pub height: Option<u32>,
    /// Target image width, or None to keep the env width.
    pub width: Option<u32>,
    /// Axis layout the model expects (`"hwc"` the default, or `"chw"`). This is the model author's
    /// declaration: when it differs from the env's layout the adapter transposes silently to match,
    /// so a model that wants `"chw"` must say so -- an omitted `layout` means `"hwc"` and the env's
    /// frame is fed through unchanged. (`channels` guards the channel *count*, not the axis order.)
    pub layout: ImageLayout,
    /// Channel count the model expects (e.g. 3 for RGB, 1 for grayscale). When set, a resolve error
    /// if the env image differs; the adapter does not convert between channel counts.
    pub channels: Option<u32>,
    /// Element dtype name the model expects.
    pub dtype: Tolerant<DType>,
    /// Whether (and into what range) to map 8-bit pixel values before casting: `False` (off, the
    /// default), `True` (the conventional `[0, 1]`), or a `(low, high)` pair for a model trained on
    /// a different range (e.g. `(-1.0, 1.0)`). One field, so an on/off flag can never disagree with
    /// a range; `False` is an authoritative off-switch.
    pub normalize: Normalize,
    /// Number of leading singleton axes to add (batch/time).
    pub lead_dims: u32,
    /// Whether the model was trained on images rotated 180 degrees from the canonical upright
    /// orientation (a true rotation -- rows and columns reversed -- not a vertical flip). Declared
    /// on both ends: the adapter rotates only when the env and model disagree, so an env that
    /// already renders upside-down pairs with a model that also sets `upside_down` and no rotation
    /// happens.
    pub upside_down: bool,
    /// Resize algorithm the model's training pipeline used. An un-suffixed name has OpenCV/torch
    /// semantics, an `_aa` suffix has PIL's (an antialiased filter whose support widens with the
    /// downscale factor): `"bilinear"` (4-tap half-pixel-center bilinear; the default, which most
    /// trained policies match), `"bilinear_aa"` / `"bicubic_aa"` / `"lanczos3_aa"` (PIL's
    /// `BILINEAR` / `BICUBIC` / `LANCZOS`), or `"area"` (OpenCV `INTER_AREA`). Bare
    /// `"bicubic"`/`"lanczos3"` are not accepted -- name the library whose kernel you trained
    /// against.
    pub resample: Tolerant<Resample>,
    /// Permit a target larger than the env's native resolution (interpolating detail that is not
    /// there). Off by default: an upscaling target is a resolve error unless this is set.
    pub allow_upscale: bool,
    #[serde(with = "super::codec::accept")]
    /// How to reconcile a target whose aspect ratio differs from the env image: `"stretch"`
    /// (distort), `"crop"` (cover + center-crop), or `"pad"` (letterbox) -- or a sequence of them
    /// in preference order. The resolver picks, per env, the first that does not need a disallowed
    /// upscale, so one spec can crop a large camera and letterbox a small one. Required only on an
    /// aspect mismatch; absent it, an aspect-changing resize is a resolve error.
    pub fit: Option<Vec<Tolerant<FitMode>>>,
    /// Zero-fill a black frame when the env does not provide this camera, instead of failing
    /// resolution. Needs `height`, `width`, and `channels` so the blank can be sized.
    pub optional: bool,
    /// Fill value (0-255) for the blank frame an `optional` camera synthesizes when the env offers
    /// no matching image; default 0 (black). Requires `optional=True` (without it the fill could
    /// never take effect, so setting it alone is a construction error); named to match
    /// `Actuator.fill`.
    pub fill: Option<u8>,
    /// Number of frames to stack on a new leading axis (frame history). `1` (default) means no
    /// stacking. Stacking is applied by the adapter core from an episode-keyed rolling window,
    /// padded at the start of an episode (see `stack_pad`) and cleared on `reset`. Which frames it
    /// gathers is `offsets` (or `stride`); by default they are the last `stack` consecutive ones.
    /// The env still sends one frame per step either way.
    pub stack: u32,
    /// Side fraction of the frame a center crop keeps, in `(0, 1]` (`2/3` keeps the middle two
    /// thirds of each axis). Pass `crop` or `crop_area`, not both.
    pub crop: Option<f64>,
    /// The same center crop stated as an *area* fraction, in `(0, 1]` -- the form training
    /// pipelines usually quote ("a 90% center crop"). The side fraction is its square root (`0.9`
    /// -> `0.949`).
    pub crop_area: Option<f64>,
    /// How the crop box is taken. `"zoom"` (the default) resamples the fractional box straight to
    /// the target -- PIL's `Image.resize(size, box=...)`, one pass, no intermediate rounding.
    /// `"slice"` cuts an integer center box out first and resizes that, which is what a `numpy`
    /// slice in a training pipeline does.
    pub crop_mode: Tolerant<CropMode>,
    /// Quality of a JPEG round-trip applied to the upright frame *before* the crop and resize, on
    /// the IJG 1-100 scale (`95` is the usual training value). A *declaration*, not a request: a
    /// pipeline that stored its frames as JPEG fed the model the codec's artifacts, so the adapter
    /// reproduces them rather than handing the model a cleaner frame than it was trained on.
    /// Baseline sequential, 4:2:0 box-averaged chroma, standard IJG tables; needs a 3-channel
    /// image.
    pub jpeg_quality: Option<u8>,
    /// Channel order the model was trained on. `"bgr"` swaps red and blue after the spatial ops and
    /// before the dtype cast, and needs a 3-channel image.
    pub channel_order: Tolerant<ChannelOrder>,
    /// The camera resolution this model was trained against, as a square `int` or an `(height,
    /// width)` pair (each axis 1-4096). An *assertion*, not a request: the adapter never resizes a
    /// camera to reach it. The platform binds the environment's camera dial from it before the run;
    /// resolution then fails with `RenderMismatch` if the camera it bound does not actually render
    /// at this size. Declare it instead of pinning the env's camera width/height by hand.
    pub render: Option<(u32, u32)>,
    /// The frame window `stack` gathers, as non-positive offsets from the current step -- oldest
    /// first, ending at `0`. `(-6, -4, -2, 0)` is "every second frame of the last seven". `None`
    /// (the default) is the contiguous window `stack` already describes. `offsets` and `stack` are
    /// never inferred from each other: both are declared and they must agree (`len(offsets) ==
    /// stack`), and the window may span at most 128 consecutive frames.
    pub offsets: Option<StackSpec>,
    /// What fills the window before the episode has produced enough frames: `"first"` (the default)
    /// replicates the first observed frame, `"black"` a raw 8-bit `0` frame pushed through this
    /// input's pipeline -- so under `normalize=(-1.0, 1.0)` a black pad frame is `-1.0`, not `0.0`.
    /// Requires `stack > 1`.
    pub stack_pad: StackPad,
    /// The body part the wanted camera sits on, when the role repeats across a body (`"head"`,
    /// `"left_arm"`, ...): an identity key the resolver matches on. Naming one binds only that
    /// camera; naming none binds the env's only camera of the role under any part (with an
    /// `info`) and fails when there are several. Omitted from the wire when unset.
    pub part: Option<String>,
}
impl Image {
    fn empty() -> Self {
        Self {
            role: String::new(),
            height: None,
            width: None,
            layout: ImageLayout::Hwc,
            channels: None,
            dtype: DType::Uint8.into(),
            normalize: Normalize::Off,
            lead_dims: 0,
            upside_down: false,
            resample: Resample::Bilinear.into(),
            allow_upscale: false,
            fit: None,
            optional: false,
            fill: None,
            stack: 1,
            crop: None,
            crop_area: None,
            crop_mode: CropMode::Zoom.into(),
            jpeg_quality: None,
            channel_order: ChannelOrder::Rgb.into(),
            render: None,
            offsets: None,
            stack_pad: StackPad::First,
            part: None,
        }
    }
    /// Construct an image input with HWC layout, uint8 data and no
    /// normalization.
    pub fn new(role: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            ..Self::empty()
        }
    }
}
impl std::fmt::Display for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "Image",
            &[
                ("role", super::repr::Repr::repr(&self.role)),
                ("height", super::repr::Repr::repr(&self.height)),
                ("width", super::repr::Repr::repr(&self.width)),
                ("layout", super::repr::Repr::repr(&self.layout)),
                ("channels", super::repr::Repr::repr(&self.channels)),
                ("dtype", super::repr::Repr::repr(&self.dtype)),
                ("normalize", super::repr::Repr::repr(&self.normalize)),
                ("lead_dims", super::repr::Repr::repr(&self.lead_dims)),
                ("upside_down", super::repr::Repr::repr(&self.upside_down)),
                ("resample", super::repr::Repr::repr(&self.resample)),
                (
                    "allow_upscale",
                    super::repr::Repr::repr(&self.allow_upscale),
                ),
                ("fit", super::repr::accept(&self.fit)),
                ("optional", super::repr::Repr::repr(&self.optional)),
                ("fill", super::repr::Repr::repr(&self.fill)),
                ("stack", super::repr::Repr::repr(&self.stack)),
                ("crop", super::repr::Repr::repr(&self.crop)),
                ("crop_area", super::repr::Repr::repr(&self.crop_area)),
                ("crop_mode", super::repr::Repr::repr(&self.crop_mode)),
                ("jpeg_quality", super::repr::Repr::repr(&self.jpeg_quality)),
                (
                    "channel_order",
                    super::repr::Repr::repr(&self.channel_order),
                ),
                ("render", super::repr::Repr::repr(&self.render)),
                ("offsets", super::repr::Repr::repr(&self.offsets)),
                ("stack_pad", super::repr::Repr::repr(&self.stack_pad)),
                ("part", super::repr::Repr::repr(&self.part)),
            ],
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "State::empty", deny_unknown_fields)]
/// A single-part numeric state input expected by a model.
pub struct State {
    /// Semantic role matched against env state features.
    pub role: String,
    /// Rotation encoding the model expects for this part. A single encoding, or a sequence of them
    /// in preference order (most-preferred first) — the resolver picks the env's native encoding
    /// when it appears here (no conversion), else converts into the first entry. A `CustomEncoding`
    /// is a single host-side packing (not a set). `"gravity_xyz"` reads any env rotation as the
    /// gravity direction in the body frame (projected gravity, 3 wide); an env that publishes
    /// `gravity_xyz` itself binds only a part that wants it.
    pub encoding: Option<StateEncoding>,
    /// Optional number of leading elements to keep from the source.
    pub dim: Option<u32>,
    /// Optional single element to select after any conversion.
    pub index: Option<u32>,
    /// Zero-fill this part when the env does not declare the role, instead of failing resolution.
    /// The fill width comes from `index` (one), `dim`, or `encoding`; one of them must be set so
    /// the width is known without an env feature.
    pub optional: bool,
    /// Optional `(low, high)` the model expects this part in; when the env declares its own
    /// (derived or tagged) range, the value is affinely mapped from the env range to this one
    /// (symmetric to action ranges). With no env source range there is nothing to map from, so it
    /// is a no-op -- it does not clamp or rescale on its own.
    pub range: Option<(f64, f64)>,
    /// Zero-pad the resulting vector to this length. Padding is the last step: every part is
    /// converted, ranged and scaled, the parts are concatenated in order, and only then is the
    /// result padded.
    pub pad_to: Option<u32>,
    /// Element dtype name of the resulting value.
    pub dtype: Tolerant<DType>,
    /// Optional target shape for the resulting value.
    pub reshape: Option<Vec<i64>>,
    /// Emit a tensor or a list of values.
    pub container: StateContainer,
    /// Value this part contributes when `optional` is set and the env lacks the role (zeros by
    /// default). A non-zero `fill` requires `optional`, matching `Actuator.fill`.
    pub fill: f64,
    /// A fixed `Rotation` right-multiplied onto the env's rotation (`R_out = R_in @
    /// R(post_rotate)`) before it is re-encoded into `encoding`. Needs a rotation `encoding`; not
    /// combinable with a `CustomEncoding`.
    pub post_rotate: Option<Rotation>,
    /// Model-side multiplier applied after the range map: one float for every axis, or a sequence
    /// with one value per axis in this part's own `labels` order (its resolved width otherwise),
    /// serialized as `axis_scale`.
    pub scale: Option<AxisValue>,
    /// Model-side addend applied after `scale` (`value * scale + offset`) -- e.g. a `1 - 2g`
    /// gripper is `scale=-2, offset=1`, a stand pose uses the negated default joint positions as
    /// offsets. A float or a per-axis sequence like `scale`.
    pub offset: Option<AxisValue>,
    /// Coordinate frame the checkpoint was trained to read this part in, when the role is an
    /// absolute pose (`proprio/eef_*`). Omitted from the wire when unset. A frame the env
    /// contradicts fails resolution; a frame the env does not declare draws a caution (the model
    /// states a requirement nothing can confirm).
    pub frame: Option<Tolerant<Frame>>,
    #[serde(with = "super::codec::accept")]
    /// Where the checkpoint expects this part's numbers to come from: `"sensed"`, `"estimated"`
    /// or `"privileged"`, or a sequence of the ones it accepts. Against an env that publishes
    /// the role under several provenances the declaration picks the leaf (an env publishing
    /// several and a model naming none is an error); a value the env contradicts fails
    /// resolution, and one the env does not declare draws a caution. Omitted from the wire when
    /// unset.
    pub provenance: Option<Vec<Tolerant<Provenance>>>,
    /// The body part this part reads, when the role repeats across a body (`LEFT_ARM`,
    /// `RIGHT_ARM`, or any identifier both sides agree on): an identity key the resolver matches
    /// on. Naming one binds only that env leaf (a missing one fills if `optional`, else fails);
    /// naming none binds the env's only leaf of the role under any part (with an `info`) and
    /// fails when there are several, naming them. Omitted from the wire when unset.
    pub part: Option<String>,
    /// The axis names this part reads, in the order the checkpoint was trained on. Fixes the
    /// width. The env leaf must label its axes with the same set; the values are gathered by
    /// name, so a differing order is a permutation, and a label either side lacks (or an env
    /// leaf with no labels) is a resolve error. Omitted from the wire when unset; not combinable
    /// with `index`. On an action-source part the names reorder the actuator's labels.
    pub labels: Option<Vec<String>>,
    /// Where the value comes from: `"observation"` (the default, an env state feature matched by
    /// role) or `"action"` (the model's own output actuator of the same `role` and `part`, read
    /// back as the raw action executed at the previous step, in model order, before the
    /// actuator's scale/offset; `fill` before the episode's first action). `Previous` is the
    /// sugar. An action- source part carries only `part`, `labels`, `dim` and `fill`. Omitted
    /// from the wire when default.
    pub source: StateSource,
    /// Clamp the assembled vector to `(low, high)` after every part's own transforms and before
    /// `pad_to` (legged_gym's `clip_obs`). Omitted from the wire when unset.
    pub clip: Option<(f64, f64)>,
}
impl State {
    fn empty() -> Self {
        Self {
            role: String::new(),
            encoding: None,
            dim: None,
            index: None,
            optional: false,
            range: None,
            pad_to: None,
            dtype: DType::Float32.into(),
            reshape: None,
            container: StateContainer::Array,
            fill: 0.0,
            post_rotate: None,
            scale: None,
            offset: None,
            frame: None,
            provenance: None,
            part: None,
            labels: None,
            source: StateSource::Observation,
            clip: None,
        }
    }
    /// Construct a required observation input producing float32 tensors.
    pub fn new(role: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            ..Self::empty()
        }
    }
}
impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "State",
            &[
                ("role", super::repr::Repr::repr(&self.role)),
                ("encoding", super::repr::Repr::repr(&self.encoding)),
                ("dim", super::repr::Repr::repr(&self.dim)),
                ("index", super::repr::Repr::repr(&self.index)),
                ("optional", super::repr::Repr::repr(&self.optional)),
                ("range", super::repr::Repr::repr(&self.range)),
                ("fill", super::repr::Repr::repr(&self.fill)),
                ("post_rotate", super::repr::Repr::repr(&self.post_rotate)),
                ("scale", super::repr::Repr::repr(&self.scale)),
                ("offset", super::repr::Repr::repr(&self.offset)),
                ("frame", super::repr::Repr::repr(&self.frame)),
                ("provenance", super::repr::accept(&self.provenance)),
                ("part", super::repr::Repr::repr(&self.part)),
                ("labels", super::repr::Repr::repr(&self.labels)),
                ("source", super::repr::Repr::repr(&self.source)),
                ("pad_to", super::repr::Repr::repr(&self.pad_to)),
                ("dtype", super::repr::Repr::repr(&self.dtype)),
                ("reshape", super::repr::Repr::repr(&self.reshape)),
                ("container", super::repr::Repr::repr(&self.container)),
                ("clip", super::repr::Repr::repr(&self.clip)),
            ],
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "Concat::empty", deny_unknown_fields)]
/// A multi-part numeric state input: several roles packed into one tensor.
pub struct Concat {
    /// Roles (or `State` / `Constant` parts) concatenated in order. At least one part must carry a
    /// role -- a state of only constants reads nothing from the env.
    pub parts: Vec<ConcatPart>,
    /// Zero-pad the concatenated vector to this length. Padding is the last step: every part is
    /// converted, ranged and scaled, the parts are concatenated in order, and only then is the
    /// result padded.
    pub pad_to: Option<u32>,
    /// Element dtype name of the resulting value.
    pub dtype: Tolerant<DType>,
    /// Optional target shape for the resulting value.
    pub reshape: Option<Vec<i64>>,
    /// Emit a tensor or a list of values.
    pub container: StateContainer,
    /// Clamp the concatenated vector to `(low, high)` after every part's own transforms and before
    /// `pad_to`.
    pub clip: Option<(f64, f64)>,
}
impl Concat {
    fn empty() -> Self {
        Self {
            parts: Vec::new(),
            pad_to: None,
            dtype: DType::Float32.into(),
            reshape: None,
            container: StateContainer::Array,
            clip: None,
        }
    }
    /// Construct a concatenation producing float32 tensors without padding or
    /// clipping.
    pub fn new<T: Into<ConcatPart>>(parts: impl IntoIterator<Item = T>) -> Self {
        Self {
            parts: parts.into_iter().map(Into::into).collect(),
            ..Self::empty()
        }
    }
}
impl std::fmt::Display for Concat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "Concat",
            &[
                ("parts", super::repr::Repr::repr(&self.parts)),
                ("pad_to", super::repr::Repr::repr(&self.pad_to)),
                ("dtype", super::repr::Repr::repr(&self.dtype)),
                ("reshape", super::repr::Repr::repr(&self.reshape)),
                ("container", super::repr::Repr::repr(&self.container)),
                ("clip", super::repr::Repr::repr(&self.clip)),
            ],
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "Constant::empty", deny_unknown_fields)]
/// A constant part of a `Concat`: `dim` copies of `fill`.
pub struct Constant {
    /// Width of the constant block.
    pub dim: u32,
    /// The value every element carries.
    pub fill: f64,
}
impl Constant {
    fn empty() -> Self {
        Self { dim: 1, fill: 0.0 }
    }
}
impl Default for Constant {
    fn default() -> Self {
        Self::empty()
    }
}
impl std::fmt::Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "Constant",
            &[
                ("dim", super::repr::Repr::repr(&self.dim)),
                ("fill", super::repr::Repr::repr(&self.fill)),
            ],
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash)]
#[serde(default = "Text::empty", deny_unknown_fields)]
/// A text input expected by a model.
pub struct Text {
    /// Semantic role matched against env text features.
    pub role: String,
    /// Emit a plain string or a single-element list.
    pub container: TextContainer,
    /// Fallback value used when the observation omits the feature; when None the input is omitted
    /// from the payload instead. Named to match `Actuator.fill`.
    pub fill: Option<String>,
}
impl Text {
    fn empty() -> Self {
        Self {
            role: String::new(),
            container: TextContainer::Str,
            fill: None,
        }
    }
    /// Construct a string input with no fallback value.
    pub fn new(role: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            ..Self::empty()
        }
    }
}
impl std::fmt::Display for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "Text",
            &[
                ("role", super::repr::Repr::repr(&self.role)),
                ("container", super::repr::Repr::repr(&self.container)),
                ("fill", super::repr::Repr::repr(&self.fill)),
            ],
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "Actuator::empty", deny_unknown_fields)]
/// One contiguous slice of an action vector.
pub struct Actuator {
    /// Semantic role used for matching, e.g. `action/gripper`. `None` makes the actuator
    /// *opaque*: it occupies `dim` dims of the action with the constant `fill`, matched by no
    /// model output -- the action-side mirror of a role-less [`Field`]. Use it for dims the env
    /// requires but no model produces (e.g. a control-mode selector). An opaque actuator carries
    /// only `dim` and `fill`.
    pub role: Option<String>,
    /// Number of action dimensions occupied by this component.
    pub dim: u32,
    /// Rotation encoding when the component is a rotation.
    pub encoding: Option<ActionEncoding>,
    /// Optional `(low, high)` range of the component values.
    pub range: Option<(f64, f64)>,
    /// Whether the component encodes a binary decision (resolved adapters snap the value to a
    /// definite side after range mapping: `>= 0` opens (`+1`), below closes (`-1`); a value exactly
    /// on the boundary opens rather than emitting an undefined `0`).
    pub binary: bool,
    /// Optional multiplier applied to the model value for this role: one float, or a sequence with
    /// one value per axis in this side's own `labels` order (serialized as `axis_scale`).
    pub scale: Option<AxisValue>,
    /// Negate the model value for this role (equivalent to `scale=-1` but explicit; the common
    /// gripper-sign correction).
    pub invert: bool,
    /// Subtract this from the value, recentering the decision boundary -- typically paired with
    /// `binary` so the snap splits at `threshold` instead of zero.
    pub threshold: Option<f64>,
    /// Clamp this actuator's mapped value to its declared `range`. The per-component safety clamp
    /// the global `Action.clip` cannot give a mixed-range action: a global clip applies one bound
    /// to every dim, so it is wrong when dims have different ranges (e.g. delta-pos in `[-1, 1]`
    /// but rotation in `[-pi/2, pi/2]`). `clip=True` requires `range`.
    pub clip: bool,
    /// Constant emitted for each dim of an opaque (role-less) actuator, and the fallback for an
    /// `optional` roled actuator. Defaults to `0.0`; inert (must stay `0.0`) on a roled, non-
    /// optional actuator.
    pub fill: f64,
    /// On a roled actuator, make the role optional -- if no model output declares it, fill the
    /// actuator's `dim` dims with `fill` instead of failing resolution (the action-side mirror of a
    /// model input's `optional` zero-fill). A model that does output the role drives it normally.
    /// Meaningless on a role-less actuator (already always filled).
    pub optional: bool,
    /// Coordinate frame a Cartesian command's axes are expressed in. Required of an absolute
    /// pose (`action/eef_*`) under the require-frames tier; a delta (`action/delta_eef_*`) may
    /// declare it too (a tool-frame delta and a base-frame delta are different commands), and
    /// when both sides do they must agree. Omitted from the wire when unset.
    pub frame: Option<Tolerant<Frame>>,
    /// What a *delta* command is integrated against (`action/delta_eef_*`): an env declares what
    /// its Cartesian controller adds the delta to -- the measured pose (`"current"`) or the last
    /// commanded target (`"target"`) -- and a model declares what it was trained against. A
    /// disagreement is a hard resolve error, which is what stops an absolute-pose head from binding
    /// cleanly to a delta controller. `reference` says what a delta is added to; `frame` says which
    /// axes it is expressed in. Declare both unless the controller fixes the frame.
    pub reference: Option<Tolerant<Reference>>,
    /// The body part this actuator drives, when the role repeats across a body (`LEFT_ARM`,
    /// `RIGHT_ARM`, or any identifier both sides agree on): an identity key the resolver matches
    /// on, never a value it checks. Omitted from the wire when unset; an opaque actuator may not
    /// carry one.
    pub part: Option<String>,
    /// Optional addend applied after `scale` (`value * scale + offset`), a float or a per-axis
    /// sequence like `scale`: the stand pose a joint-position target is expressed around..
    pub offset: Option<AxisValue>,
    /// The axis names this actuator drives, `dim` of them in this side's own order. When a model
    /// names labels the env actuator must name the same set; the model's output is then
    /// reordered onto the env's axes by name. Omitted from the wire when unset; an opaque
    /// actuator may not carry them.
    pub labels: Option<Vec<String>>,
}
impl Actuator {
    fn empty() -> Self {
        Self {
            role: None,
            dim: 0,
            encoding: None,
            range: None,
            binary: false,
            scale: None,
            invert: false,
            threshold: None,
            clip: false,
            fill: 0.0,
            optional: false,
            frame: None,
            reference: None,
            part: None,
            offset: None,
            labels: None,
        }
    }
}
impl Default for Actuator {
    fn default() -> Self {
        Self::empty()
    }
}
impl std::fmt::Display for Actuator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "Actuator",
            &[
                ("role", super::repr::Repr::repr(&self.role)),
                ("dim", super::repr::Repr::repr(&self.dim)),
                ("encoding", super::repr::Repr::repr(&self.encoding)),
                ("range", super::repr::Repr::repr(&self.range)),
                ("binary", super::repr::Repr::repr(&self.binary)),
                ("scale", super::repr::Repr::repr(&self.scale)),
                ("invert", super::repr::Repr::repr(&self.invert)),
                ("threshold", super::repr::Repr::repr(&self.threshold)),
                ("clip", super::repr::Repr::repr(&self.clip)),
                ("fill", super::repr::Repr::repr(&self.fill)),
                ("optional", super::repr::Repr::repr(&self.optional)),
                ("frame", super::repr::Repr::repr(&self.frame)),
                ("reference", super::repr::Repr::repr(&self.reference)),
                ("part", super::repr::Repr::repr(&self.part)),
                ("offset", super::repr::Repr::repr(&self.offset)),
                ("labels", super::repr::Repr::repr(&self.labels)),
            ],
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "Action::empty", deny_unknown_fields)]
/// Ordered action actuators plus optional clipping bounds.
pub struct Action {
    /// Action actuators in vector order.
    pub components: Vec<Actuator>,
    /// Optional `(low, high)` clip applied to the final vector.
    pub clip: Option<(f64, f64)>,
}
impl Action {
    fn empty() -> Self {
        Self {
            components: Vec::new(),
            clip: None,
        }
    }
    /// Construct ordered action actuators without clipping.
    pub fn new(items: impl IntoIterator<Item = Actuator>) -> Self {
        Self {
            components: items.into_iter().collect(),
            ..Self::empty()
        }
    }
}
impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "Action",
            &[
                ("components", super::repr::Repr::repr(&self.components)),
                ("clip", super::repr::Repr::repr(&self.clip)),
            ],
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash)]
#[serde(default = "ImageTag::empty", deny_unknown_fields)]
/// A camera image entry in an environment observation.
pub struct ImageTag {
    /// Semantic role used for matching, e.g. `image/primary`.
    pub role: String,
    /// Axis layout of the stored image.
    pub layout: ImageLayout,
    /// Whether the image is rendered rotated 180 degrees (a true rotation, not a vertical flip)
    /// relative to the canonical upright orientation. Declared on both ends; the adapter flips only
    /// when the env and the model disagree. (If a second orientation is ever needed, this should
    /// become a constrained string like `dtype`, not a wider `bool`.)
    pub upside_down: bool,
    /// The body part this camera sits on, when the role repeats across a body (`"head"`,
    /// `"left_arm"`, ...). An identity key the resolver matches on, never a value it checks.
    /// Omitted from the wire when unset.
    pub part: Option<String>,
}
impl ImageTag {
    fn empty() -> Self {
        Self {
            role: String::new(),
            layout: ImageLayout::Hwc,
            upside_down: false,
            part: None,
        }
    }
    /// Construct a camera tag with HWC layout and upright orientation.
    pub fn new(role: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            ..Self::empty()
        }
    }
}
impl std::fmt::Display for ImageTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "ImageTag",
            &[
                ("role", super::repr::Repr::repr(&self.role)),
                ("layout", super::repr::Repr::repr(&self.layout)),
                ("upside_down", super::repr::Repr::repr(&self.upside_down)),
                ("part", super::repr::Repr::repr(&self.part)),
            ],
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "StateTag::empty", deny_unknown_fields)]
/// A numeric proprioception entry in an environment observation.
pub struct StateTag {
    /// Semantic role used for matching, e.g. `proprio/eef_pos`.
    pub role: String,
    #[serde(with = "super::codec::accept")]
    /// Rotation encoding when the role is a rotation. A single encoding, or a sequence of them (the
    /// env's native first, then alternatives it can emit) for cross-version negotiation.
    pub encoding: Option<Vec<Tolerant<RotationEncoding>>>,
    /// Optional `(low, high)` value range, supplying the bounds where the space leaves this leaf
    /// unbounded. If the space declares finite bounds that disagree with it, resolution errors
    /// rather than silently overriding them.
    pub range: Option<(f64, f64)>,
    /// Coordinate frame this entry's values are expressed in, when the role is an absolute pose
    /// (`proprio/eef_*`). Omitted from the wire when unset. A model that declares a different
    /// one fails resolution; a model that declares one the env does not draws a caution.
    pub frame: Option<Tolerant<Frame>>,
    /// Where this entry's numbers come from: `"sensed"` (a physical sensor or its simulated
    /// equivalent), `"estimated"` (a state estimator) or `"privileged"` (simulator truth with no
    /// hardware counterpart). Part of the leaf's identity, so an env may publish one role twice
    /// under two provenances; a model that declares one the env contradicts fails resolution.
    /// Omitted from the wire when unset.
    pub provenance: Option<Tolerant<Provenance>>,
    /// The body part this entry belongs to, when the role repeats across a body (`LEFT_ARM`,
    /// `RIGHT_ARM`, or any identifier both sides agree on). An identity key the resolver matches
    /// on; see `ImageTag`. Omitted from the wire when unset.
    pub part: Option<String>,
    /// The axis names of this entry in the order the env emits them, one per element of the
    /// space leaf (checked at `join`). A model that names labels must name this same set;
    /// another order is gathered by name. Omitted from the wire when unset.
    pub labels: Option<Vec<String>>,
}
impl StateTag {
    fn empty() -> Self {
        Self {
            role: String::new(),
            encoding: None,
            range: None,
            frame: None,
            provenance: None,
            part: None,
            labels: None,
        }
    }
    /// Construct a state tag with only its role declared.
    pub fn new(role: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            ..Self::empty()
        }
    }
}
impl std::fmt::Display for StateTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "StateTag",
            &[
                ("role", super::repr::Repr::repr(&self.role)),
                ("encoding", super::repr::accept(&self.encoding)),
                ("range", super::repr::Repr::repr(&self.range)),
                ("frame", super::repr::Repr::repr(&self.frame)),
                ("provenance", super::repr::Repr::repr(&self.provenance)),
                ("part", super::repr::Repr::repr(&self.part)),
                ("labels", super::repr::Repr::repr(&self.labels)),
            ],
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash)]
#[serde(default = "TextTag::empty", deny_unknown_fields)]
/// A text entry (typically the task instruction) in an observation.
pub struct TextTag {
    /// Semantic role used for matching (required, e.g. `INSTRUCTION`).
    pub role: String,
}
impl TextTag {
    fn empty() -> Self {
        Self {
            role: String::new(),
        }
    }
    /// Construct a role reference.
    pub fn new(role: impl Into<String>) -> Self {
        Self { role: role.into() }
    }
}
impl std::fmt::Display for TextTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "TextTag",
            &[("role", super::repr::Repr::repr(&self.role))],
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "Field::empty", deny_unknown_fields)]
/// One contiguous field of a flat numeric observation leaf.
pub struct Field {
    /// Semantic role matched against model state parts, or None to skip this slice.
    pub role: Option<String>,
    /// Number of elements this field occupies.
    pub dim: u32,
    #[serde(with = "super::codec::accept")]
    /// Rotation encoding when the field is a rotation. A single encoding, or a sequence of them
    /// (native first) for negotiation.
    pub encoding: Option<Vec<Tolerant<RotationEncoding>>>,
    /// Optional `(low, high)` value range for this field's slice, supplying the bounds where the
    /// space leaves it unbounded. If the space declares finite bounds for the slice that disagree
    /// with it, resolution errors rather than silently overriding them.
    pub range: Option<(f64, f64)>,
    /// Coordinate frame this field's values are expressed in; see :attr:`StateTag.frame`. A role-
    /// less skip may not carry one.
    pub frame: Option<Tolerant<Frame>>,
    /// Where this field's numbers come from; see :attr:`StateTag.provenance`. A role-less skip may
    /// not carry one.
    pub provenance: Option<Tolerant<Provenance>>,
    /// The body part this field belongs to; see :attr:`StateTag.part`. A role-less skip may not
    /// carry one.
    pub part: Option<String>,
    /// The axis names of this field, `dim` of them; see :attr:`StateTag.labels`. A role-less skip
    /// may not carry them.
    pub labels: Option<Vec<String>>,
}
impl Field {
    fn empty() -> Self {
        Self {
            role: None,
            dim: 0,
            encoding: None,
            range: None,
            frame: None,
            provenance: None,
            part: None,
            labels: None,
        }
    }
}
impl Default for Field {
    fn default() -> Self {
        Self::empty()
    }
}
impl std::fmt::Display for Field {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "Field",
            &[
                ("role", super::repr::Repr::repr(&self.role)),
                ("dim", super::repr::Repr::repr(&self.dim)),
                ("encoding", super::repr::accept(&self.encoding)),
                ("range", super::repr::Repr::repr(&self.range)),
                ("frame", super::repr::Repr::repr(&self.frame)),
                ("provenance", super::repr::Repr::repr(&self.provenance)),
                ("part", super::repr::Repr::repr(&self.part)),
                ("labels", super::repr::Repr::repr(&self.labels)),
            ],
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "Split::empty", deny_unknown_fields)]
/// An ordered split of one flat numeric observation leaf into role fields.
pub struct Split {
    /// State fields in vector order.
    pub fields: Vec<Field>,
}
impl Split {
    fn empty() -> Self {
        Self { fields: Vec::new() }
    }
    /// Construct a split with fields in the supplied order.
    pub fn new(items: impl IntoIterator<Item = Field>) -> Self {
        Self {
            fields: items.into_iter().collect(),
        }
    }
}
impl std::fmt::Display for Split {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "Split",
            &[("fields", super::repr::Repr::repr(&self.fields))],
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash)]
#[serde(default = "ObservationRoles::empty", deny_unknown_fields)]
/// The observation roles an environment declares, grouped by kind.
pub struct ObservationRoles {
    /// Roles of `ImageTag` leaves.
    pub images: Vec<String>,
    /// Roles of `StateTag` leaves and of non-skip `Field` slices inside a `Split`.
    pub states: Vec<String>,
    /// Roles of `TextTag` leaves.
    pub texts: Vec<String>,
}
impl ObservationRoles {
    fn empty() -> Self {
        Self {
            images: Vec::new(),
            states: Vec::new(),
            texts: Vec::new(),
        }
    }
}
impl Default for ObservationRoles {
    fn default() -> Self {
        Self::empty()
    }
}
impl std::fmt::Display for ObservationRoles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        super::repr::dataclass(
            f,
            "ObservationRoles",
            &[
                ("images", super::repr::Repr::repr(&self.images)),
                ("states", super::repr::Repr::repr(&self.states)),
                ("texts", super::repr::Repr::repr(&self.texts)),
            ],
        )
    }
}
