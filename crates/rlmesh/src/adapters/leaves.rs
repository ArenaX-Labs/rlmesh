//! Immutable leaf values used to author adapter trees.
use super::*;
use crate::DType;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "Image::empty", deny_unknown_fields)]
pub struct Image {
    pub role: String,
    pub height: Option<u32>,
    pub width: Option<u32>,
    pub layout: ImageLayout,
    pub channels: Option<u32>,
    pub dtype: Tolerant<DType>,
    pub normalize: Normalize,
    pub lead_dims: u32,
    pub upside_down: bool,
    pub resample: Tolerant<Resample>,
    pub allow_upscale: bool,
    #[serde(with = "super::codec::accept")]
    pub fit: Option<Vec<Tolerant<FitMode>>>,
    pub optional: bool,
    pub fill: Option<u8>,
    pub stack: u32,
    pub crop: Option<f64>,
    pub crop_area: Option<f64>,
    pub crop_mode: Tolerant<CropMode>,
    pub jpeg_quality: Option<u8>,
    pub channel_order: Tolerant<ChannelOrder>,
    pub render: Option<(u32, u32)>,
    pub offsets: Option<StackSpec>,
    pub stack_pad: StackPad,
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
pub struct State {
    pub role: String,
    pub encoding: Option<StateEncoding>,
    pub dim: Option<u32>,
    pub index: Option<u32>,
    pub optional: bool,
    pub range: Option<(f64, f64)>,
    pub pad_to: Option<u32>,
    pub dtype: Tolerant<DType>,
    pub reshape: Option<Vec<i64>>,
    pub container: StateContainer,
    pub fill: f64,
    pub post_rotate: Option<Rotation>,
    pub scale: Option<AxisValue>,
    pub offset: Option<AxisValue>,
    pub frame: Option<Tolerant<Frame>>,
    #[serde(with = "super::codec::accept")]
    pub provenance: Option<Vec<Tolerant<Provenance>>>,
    pub part: Option<String>,
    pub labels: Option<Vec<String>>,
    pub source: StateSource,
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
pub struct Concat {
    pub parts: Vec<ConcatPart>,
    pub pad_to: Option<u32>,
    pub dtype: Tolerant<DType>,
    pub reshape: Option<Vec<i64>>,
    pub container: StateContainer,
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
pub struct Constant {
    pub dim: u32,
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
pub struct Text {
    pub role: String,
    pub container: TextContainer,
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
pub struct Actuator {
    pub role: Option<String>,
    pub dim: u32,
    pub encoding: Option<ActionEncoding>,
    pub range: Option<(f64, f64)>,
    pub binary: bool,
    pub scale: Option<AxisValue>,
    pub invert: bool,
    pub threshold: Option<f64>,
    pub clip: bool,
    pub fill: f64,
    pub optional: bool,
    pub frame: Option<Tolerant<Frame>>,
    pub reference: Option<Tolerant<Reference>>,
    pub part: Option<String>,
    pub offset: Option<AxisValue>,
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
pub struct Action {
    pub components: Vec<Actuator>,
    pub clip: Option<(f64, f64)>,
}
impl Action {
    fn empty() -> Self {
        Self {
            components: Vec::new(),
            clip: None,
        }
    }
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
pub struct ImageTag {
    pub role: String,
    pub layout: ImageLayout,
    pub upside_down: bool,
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
pub struct StateTag {
    pub role: String,
    #[serde(with = "super::codec::accept")]
    pub encoding: Option<Vec<Tolerant<RotationEncoding>>>,
    pub range: Option<(f64, f64)>,
    pub frame: Option<Tolerant<Frame>>,
    pub provenance: Option<Tolerant<Provenance>>,
    pub part: Option<String>,
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
pub struct TextTag {
    pub role: String,
}
impl TextTag {
    fn empty() -> Self {
        Self {
            role: String::new(),
        }
    }
    pub fn new(role: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            ..Self::empty()
        }
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
pub struct Field {
    pub role: Option<String>,
    pub dim: u32,
    #[serde(with = "super::codec::accept")]
    pub encoding: Option<Vec<Tolerant<RotationEncoding>>>,
    pub range: Option<(f64, f64)>,
    pub frame: Option<Tolerant<Frame>>,
    pub provenance: Option<Tolerant<Provenance>>,
    pub part: Option<String>,
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
pub struct Split {
    pub fields: Vec<Field>,
}
impl Split {
    fn empty() -> Self {
        Self { fields: Vec::new() }
    }
    pub fn new(items: impl IntoIterator<Item = Field>) -> Self {
        Self {
            fields: items.into_iter().collect(),
            ..Self::empty()
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
pub struct ObservationRoles {
    pub images: Vec<String>,
    pub states: Vec<String>,
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
