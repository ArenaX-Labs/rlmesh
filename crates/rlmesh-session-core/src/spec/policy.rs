//! The policy map (§6.1-§6.3) and run-time options (§6.4), parsed and
//! normalised: shorthands expanded, defaults that do not depend on a
//! candidate filled in, `allow` merged, caps checked (§3.4).
//!
//! ```json
//! {
//!   "hz": 15,
//!   "inputs": {
//!     "image": {"kind": "camera", "at": "/scene/cam", "size": [224, 224]},
//!     "state": {"kind": "joints", "at": "arm", "fields": ["pos"]},
//!     "prompt": "text"
//!   },
//!   "actions": {
//!     "arm": {"kind": "joints", "at": "arm", "command": "position", "horizon": 10}
//!   },
//!   "allow": ["resize", "pad"]
//! }
//! ```
//!
//! `size` is `[height, width]`. Ranges are `[low, high]`; `scale` and
//! `offset` are a number or one number per element.

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::BTreeMap;
use core::fmt;
use core::time::Duration;

use crate::wire::part::{DType, FieldSpec, MAX_CHUNK_ACTIONS};
use crate::wire::path::is_identifier;
use crate::wire::{PartPath, ParticipantId};
use serde::Deserialize;
use serde_json::Value;

use crate::spec::contract::MAX_HISTORY;
use crate::spec::kinds::Kind;
use crate::spec::normalise::{FieldDecl, Ordered};
use crate::spec::rotation::Rotation;
use crate::spec::selector::{RUN_OPTIONS, Selector};

/// The longest entry name.
const MAX_ENTRY_NAME: usize = 128;
/// The highest rate a policy declares.
const MAX_HZ: f64 = 10_000.0;
/// The longest declared duration, in ms.
const MAX_MS: f64 = 60_000.0;
/// The largest target image side.
const MAX_IMAGE_SIDE: u32 = 8192;

/// A conversion a policy may allow (§7.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Allow {
    /// Resample an image to the target size.
    Resize,
    /// Let `resize` enlarge.
    Upscale,
    /// Aspect fit by cover and center crop.
    Crop,
    /// Aspect fit by letterboxing.
    Pad,
    /// Aspect fit by distortion.
    Stretch,
    /// Swap R and B.
    Bgr,
    /// Rotate an image 180°.
    Rotate180,
    /// An inexact cast.
    Dtype,
    /// A non-angular unit conversion.
    Units,
    /// A rotation re-encoding other than quaternion order.
    Rotation,
    /// Take the model's joints out of a larger set.
    Select,
    /// A linear map between declared ranges.
    Rescale,
    /// Deliver a stale sample, flagged.
    Hold,
    /// Pad a window at episode start.
    Warmup,
    /// Bind an absent optional input to a constant.
    Blank,
    /// Discard an output the robot has no part for.
    Drop,
}

impl Allow {
    /// Every conversion.
    pub const ALL: [Self; 16] = [
        Self::Resize,
        Self::Upscale,
        Self::Crop,
        Self::Pad,
        Self::Stretch,
        Self::Bgr,
        Self::Rotate180,
        Self::Dtype,
        Self::Units,
        Self::Rotation,
        Self::Select,
        Self::Rescale,
        Self::Hold,
        Self::Warmup,
        Self::Blank,
        Self::Drop,
    ];

    /// Its name in `allow`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Resize => "resize",
            Self::Upscale => "upscale",
            Self::Crop => "crop",
            Self::Pad => "pad",
            Self::Stretch => "stretch",
            Self::Bgr => "bgr",
            Self::Rotate180 => "rotate180",
            Self::Dtype => "dtype",
            Self::Units => "units",
            Self::Rotation => "rotation",
            Self::Select => "select",
            Self::Rescale => "rescale",
            Self::Hold => "hold",
            Self::Warmup => "warmup",
            Self::Blank => "blank",
            Self::Drop => "drop",
        }
    }

    /// The conversion `name` names.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.name() == name)
    }
}

impl fmt::Display for Allow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of allowed conversions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Allows(u32);

impl Allows {
    /// Whether `allow` is in the set.
    pub fn has(self, allow: Allow) -> bool {
        self.0 & (1 << allow as u32) != 0
    }

    /// The set with `allow` added.
    pub fn with(self, allow: Allow) -> Self {
        Self(self.0 | 1 << allow as u32)
    }

    /// The union.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    fn parse(where_: &str, names: &[String]) -> Result<Self, String> {
        names.iter().try_fold(Self::default(), |set, name| {
            Allow::from_name(name)
                .map(|a| set.with(a))
                .ok_or_else(|| format!("{where_}: allow {name:?} is not a conversion (§7.2)"))
        })
    }

    /// The conversions in the set, in [`Allow::ALL`] order.
    pub fn iter(self) -> impl Iterator<Item = Allow> {
        Allow::ALL.into_iter().filter(move |a| self.has(*a))
    }
}

/// How an entry's joint names are matched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Names {
    /// No names: take the robot's order and count (`count` checks it).
    Robot,
    /// The model's names, in its order.
    List(Vec<String>),
    /// By index: element i is the robot's element i.
    Positional,
}

/// Image layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImageLayout {
    /// Height, width, channels.
    #[default]
    Hwc,
    /// Channels, height, width.
    Chw,
}

impl ImageLayout {
    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Hwc => "hwc",
            Self::Chw => "chw",
        }
    }
}

/// Image pixel format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImageFormat {
    /// RGB.
    #[default]
    Rgb8,
    /// BGR.
    Bgr8,
    /// One channel.
    Mono8,
}

impl ImageFormat {
    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rgb8 => "rgb8",
            Self::Bgr8 => "bgr8",
            Self::Mono8 => "mono8",
        }
    }
}

/// A resize kernel (v1 semantics).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Resample {
    /// The 4-tap half-pixel-center sampler (cv2/torch).
    #[default]
    Bilinear,
    /// PIL's antialiased triangle filter.
    BilinearAa,
    /// PIL's antialiased cubic filter (a = -0.5).
    BicubicAa,
    /// PIL's antialiased Lanczos-3 filter.
    Lanczos3Aa,
    /// cv2's `INTER_AREA`.
    Area,
    /// Nearest neighbour: depth images, always.
    Nearest,
}

impl Resample {
    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bilinear => "bilinear",
            Self::BilinearAa => "bilinear_aa",
            Self::BicubicAa => "bicubic_aa",
            Self::Lanczos3Aa => "lanczos3_aa",
            Self::Area => "area",
            Self::Nearest => "nearest",
        }
    }

    /// The pixel kernel `name` names (`nearest` is depth's, never declared).
    pub fn parse(name: &str) -> Option<Self> {
        [
            Self::Bilinear,
            Self::BilinearAa,
            Self::BicubicAa,
            Self::Lanczos3Aa,
            Self::Area,
        ]
        .into_iter()
        .find(|r| r.name() == name)
    }
}

/// Where an input's samples come from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Source {
    /// The part's own stream.
    #[default]
    Published,
    /// What the edge dispatched to it (`{path}/_applied`, §9.4).
    Applied,
}

/// When a chunk starts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChunkStart {
    /// At the observation instant.
    #[default]
    Snapshot,
    /// When it is sent.
    Send,
}

/// What inputs and actions share.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// Its name in `predict`'s dicts.
    pub name: String,
    /// The wanted kind.
    pub kind: Kind,
    /// The kind as written (a custom kind's full name).
    pub kind_name: String,
    /// Its selector; `None` searches every part of the kind below the scope
    /// root.
    pub at: Option<Selector>,
    /// May stay unbound (with `blank` or `drop` allowed).
    pub optional: bool,
    /// How joint names match.
    pub names: Names,
    /// The expected joint count.
    pub count: Option<u32>,
    /// The model's rotation encoding.
    pub rotation: Option<Rotation>,
    /// Required frame identities.
    pub frame: Option<String>,
    /// Pose: the required child frame.
    pub child: Option<String>,
    /// Target units per field.
    pub units: Vec<(String, String)>,
    /// The policy's `allow` and the entry's own, merged.
    pub allow: Allows,
    /// Declared affine scale (scalar or per element).
    pub scale: Option<Vec<f64>>,
    /// Declared affine offset.
    pub offset: Option<Vec<f64>>,
    /// The model's range: an image's normalisation target, or the range
    /// `rescale` maps.
    pub range: Option<(f64, f64)>,
    /// A custom kind's explicit field schema.
    pub schema: Option<Vec<FieldSpec>>,
}

/// An input entry (§6.2).
#[derive(Clone, Debug, PartialEq)]
pub struct Input {
    /// The common keys.
    pub entry: Entry,
    /// Fields to deliver, in this order (`None`: all published).
    pub fields: Option<Vec<String>>,
    /// Expected per-field shape.
    pub shape: Option<Vec<u32>>,
    /// Target dtype.
    pub dtype: Option<DType>,
    /// Target image size, `(height, width)`.
    pub size: Option<(u32, u32)>,
    /// Image layout.
    pub layout: ImageLayout,
    /// Pixel format.
    pub format: ImageFormat,
    /// Resize kernel.
    pub resample: Resample,
    /// Training-time center crop, side fraction.
    pub crop: Option<f64>,
    /// Trained on 180°-rotated frames.
    pub upside_down: bool,
    /// Wrench: the required reference point.
    pub point: Option<String>,
    /// Window length (1: no time axis).
    pub history: u32,
    /// The rate the window was trained at.
    pub history_hz: Option<f64>,
    /// Freshness bound at the policy tick.
    pub max_age: Option<Duration>,
    /// Published or applied.
    pub source: Source,
}

/// An action entry (§6.3).
#[derive(Clone, Debug, PartialEq)]
pub struct Action {
    /// The common keys.
    pub entry: Entry,
    /// Command mode; must equal the part's.
    pub command: Option<String>,
    /// Pose delta reference; must equal the part's.
    pub reference: Option<String>,
    /// The action grid rate (`None`: the policy rate).
    pub hz: Option<f64>,
    /// Actions per chunk.
    pub horizon: u32,
    /// Gripper: the model value meaning open.
    pub open: Option<f64>,
    /// Gripper: the model value meaning closed.
    pub closed: Option<f64>,
    /// When a chunk starts.
    pub t0: ChunkStart,
    /// A cap on a chunk's life.
    pub ttl: Option<Duration>,
}

/// A parsed policy map.
#[derive(Clone, Debug, PartialEq)]
pub struct Policy {
    /// The replan rate R.
    pub hz: f64,
    /// Inputs, in authored order.
    pub inputs: Vec<Input>,
    /// Actions, in authored order.
    pub actions: Vec<Action>,
}

/// A number or one number per element.
#[derive(Deserialize)]
#[serde(untagged)]
enum Numbers {
    One(f64),
    Many(Vec<f64>),
}

/// An entry's `fields`: names (core kinds) or a schema (custom kinds).
#[derive(Deserialize)]
#[serde(untagged)]
enum EntryFields {
    Names(Vec<String>),
    Schema(Ordered<FieldDecl>),
}

/// An entry's `names`: a list, or `"positional"`.
#[derive(Deserialize)]
#[serde(untagged)]
enum NamesDecl {
    List(Vec<String>),
    Word(String),
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct EntryDecl {
    kind: String,
    at: Option<String>,
    optional: Option<bool>,
    fields: Option<EntryFields>,
    names: Option<NamesDecl>,
    count: Option<u32>,
    shape: Option<Vec<u32>>,
    dtype: Option<String>,
    size: Option<[u32; 2]>,
    height: Option<u32>,
    width: Option<u32>,
    layout: Option<String>,
    format: Option<String>,
    range: Option<[f64; 2]>,
    resample: Option<String>,
    crop: Option<f64>,
    upside_down: Option<bool>,
    rotation: Option<String>,
    frame: Option<String>,
    child: Option<String>,
    point: Option<String>,
    units: Option<Ordered<String>>,
    scale: Option<Numbers>,
    offset: Option<Numbers>,
    history: Option<u32>,
    history_hz: Option<f64>,
    max_age_ms: Option<f64>,
    source: Option<String>,
    allow: Option<Vec<String>>,
    command: Option<String>,
    reference: Option<String>,
    hz: Option<f64>,
    horizon: Option<u32>,
    open: Option<f64>,
    closed: Option<f64>,
    t0: Option<String>,
    ttl_ms: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyDecl {
    hz: f64,
    #[serde(default)]
    inputs: Option<Ordered<Value>>,
    #[serde(default)]
    actions: Option<Ordered<Value>>,
    #[serde(default)]
    allow: Vec<String>,
}

fn rate(where_: &str, key: &str, hz: f64) -> Result<f64, String> {
    if hz.is_finite() && hz > 0.0 && hz <= MAX_HZ {
        Ok(hz)
    } else {
        Err(format!("{where_}: {key} {hz} is not in (0, 10000]"))
    }
}

fn millis(where_: &str, key: &str, ms: f64) -> Result<Duration, String> {
    if ms.is_finite() && ms > 0.0 && ms <= MAX_MS {
        Ok(Duration::from_secs_f64(ms / 1e3))
    } else {
        Err(format!("{where_}: {key} {ms} is out of range"))
    }
}

fn ident(where_: &str, key: &str, value: Option<String>) -> Result<Option<String>, String> {
    value
        .map(|v| {
            if is_identifier(&v, 64) {
                Ok(v)
            } else {
                Err(format!("{where_}: {key} {v:?} must be [a-z][a-z0-9_]*"))
            }
        })
        .transpose()
}

fn numbers(where_: &str, key: &str, value: Option<Numbers>) -> Result<Option<Vec<f64>>, String> {
    let values = match value {
        None => return Ok(None),
        Some(Numbers::One(v)) => vec![v],
        Some(Numbers::Many(v)) => v,
    };
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return Err(format!("{where_}: {key} must be finite numbers"));
    }
    Ok(Some(values))
}

/// Which keys only inputs and only actions take.
const INPUT_ONLY: [&str; 14] = [
    "fields",
    "shape",
    "dtype",
    "size",
    "height",
    "width",
    "layout",
    "format",
    "resample",
    "crop",
    "upside_down",
    "point",
    "history",
    "source",
];
const ACTION_ONLY: [&str; 8] = [
    "command",
    "reference",
    "hz",
    "horizon",
    "open",
    "closed",
    "t0",
    "ttl_ms",
];

impl EntryDecl {
    fn set_keys(&self) -> Vec<&'static str> {
        [
            ("fields", self.fields.is_some()),
            ("shape", self.shape.is_some()),
            ("dtype", self.dtype.is_some()),
            ("size", self.size.is_some()),
            ("height", self.height.is_some()),
            ("width", self.width.is_some()),
            ("layout", self.layout.is_some()),
            ("format", self.format.is_some()),
            ("resample", self.resample.is_some()),
            ("crop", self.crop.is_some()),
            ("upside_down", self.upside_down.is_some()),
            ("point", self.point.is_some()),
            (
                "history",
                self.history.is_some() || self.history_hz.is_some(),
            ),
            ("max_age_ms", self.max_age_ms.is_some()),
            ("source", self.source.is_some()),
            ("command", self.command.is_some()),
            ("reference", self.reference.is_some()),
            ("hz", self.hz.is_some()),
            ("horizon", self.horizon.is_some()),
            ("open", self.open.is_some()),
            ("closed", self.closed.is_some()),
            ("t0", self.t0.is_some()),
            ("ttl_ms", self.ttl_ms.is_some()),
        ]
        .into_iter()
        .filter(|(_, set)| *set)
        .map(|(k, _)| k)
        .collect()
    }
}

fn entry_name(dir: &str, name: &str) -> Result<String, String> {
    if name.is_empty() || name.len() > MAX_ENTRY_NAME {
        Err(format!("{dir} {name:?}: names are 1-128 characters"))
    } else {
        Ok(name.to_string())
    }
}

fn common(
    where_: &str,
    name: String,
    decl: &mut EntryDecl,
    policy_allow: Allows,
) -> Result<Entry, String> {
    let kind = Kind::parse(&decl.kind).map_err(|err| format!("{where_}: {err}"))?;
    let at = decl
        .at
        .take()
        .map(|at| Selector::parse(&at).map_err(|err| format!("{where_}: {err}")))
        .transpose()?;
    let names = match decl.names.take() {
        None => Names::Robot,
        Some(NamesDecl::Word(w)) if w == "positional" => Names::Positional,
        Some(NamesDecl::Word(w)) => {
            return Err(format!(
                "{where_}: names {w:?}: a list of names, or \"positional\""
            ));
        }
        Some(NamesDecl::List(list)) => {
            for (i, n) in list.iter().enumerate() {
                if n.is_empty() || n.len() > 128 || list[..i].contains(n) {
                    return Err(format!(
                        "{where_}: names: {n:?} is empty, longer than 128 or repeated"
                    ));
                }
            }
            if list.is_empty() {
                return Err(format!("{where_}: names is empty"));
            }
            Names::List(list)
        }
    };
    if (names != Names::Robot || decl.count.is_some()) && !kind.is_joint_set() {
        return Err(format!(
            "{where_}: names and count are for joints and hand entries"
        ));
    }
    if names == Names::Positional && decl.count.is_none() {
        return Err(format!("{where_}: positional names need a count"));
    }
    if let (Names::List(list), Some(count)) = (&names, decl.count)
        && list.len() as u64 != u64::from(count)
    {
        return Err(format!(
            "{where_}: count differs from the {} names",
            list.len()
        ));
    }
    if decl.count.is_some_and(|c| c == 0 || c > 1024) {
        return Err(format!("{where_}: count is 1 to 1024"));
    }
    let rotation = decl
        .rotation
        .take()
        .map(|r| Rotation::from_name(&r).ok_or_else(|| format!("{where_}: rotation {r:?}")))
        .transpose()?;
    let units = decl.units.take().map(|u| u.0).unwrap_or_default();
    let allow = Allows::parse(where_, &decl.allow.take().unwrap_or_default())?;
    let range = decl
        .range
        .map(|[low, high]| {
            if low.is_finite() && high.is_finite() && low != high {
                Ok((low, high))
            } else {
                Err(format!("{where_}: range is two finite, different numbers"))
            }
        })
        .transpose()?;
    let schema = match decl.fields.take() {
        Some(EntryFields::Schema(schema)) => {
            if kind != Kind::Custom {
                return Err(format!(
                    "{where_}: fields is a list of names for a {kind} entry"
                ));
            }
            Some(
                schema
                    .0
                    .into_iter()
                    .map(|(field, FieldDecl(dtype, shape))| {
                        DType::from_name(&dtype)
                            .map(|d| FieldSpec::new(field.clone(), d, &shape))
                            .ok_or_else(|| format!("{where_}: field {field:?} dtype {dtype:?}"))
                    })
                    .collect::<Result<_, _>>()?,
            )
        }
        other => {
            decl.fields = other;
            None
        }
    };
    if kind == Kind::Custom && schema.is_none() {
        return Err(format!(
            "{where_}: a custom kind binds only with an explicit fields schema {{name: [dtype, shape]}}"
        ));
    }
    Ok(Entry {
        name,
        kind,
        kind_name: decl.kind.clone(),
        at,
        optional: decl.optional.unwrap_or(false),
        names,
        count: decl.count,
        rotation,
        frame: ident(where_, "frame", decl.frame.take())?,
        child: ident(where_, "child", decl.child.take())?,
        units,
        allow: allow.union(policy_allow),
        scale: numbers(where_, "scale", decl.scale.take())?,
        offset: numbers(where_, "offset", decl.offset.take())?,
        range,
        schema,
    })
}

/// An entry, or its shorthand: a bare kind.
fn parse_entry(where_: &str, value: Value) -> Result<EntryDecl, String> {
    match value {
        Value::String(kind) => Ok(EntryDecl {
            kind,
            ..EntryDecl::default()
        }),
        value => serde_json::from_value(value).map_err(|err| format!("{where_}: {err}")),
    }
}

fn input(name: &str, value: Value, policy_allow: Allows) -> Result<Input, String> {
    let where_ = format!("input {name:?}");
    let mut decl = parse_entry(&where_, value)?;
    if let Some(key) = decl
        .set_keys()
        .into_iter()
        .find(|k| ACTION_ONLY.contains(k))
    {
        return Err(format!("{where_}: {key} is an action key"));
    }
    let entry = common(&where_, entry_name("input", name)?, &mut decl, policy_allow)?;
    let fields = match decl.fields.take() {
        Some(EntryFields::Names(names)) => {
            if names.is_empty() {
                return Err(format!("{where_}: fields is empty"));
            }
            for (i, f) in names.iter().enumerate() {
                if names[..i].contains(f) {
                    return Err(format!("{where_}: field {f:?} is listed twice"));
                }
            }
            Some(names)
        }
        _ => None,
    };
    let size = match (decl.size, decl.height, decl.width) {
        (Some([h, w]), None, None) | (None, Some(h), Some(w)) => Some((h, w)),
        (None, None, None) => None,
        _ => return Err(format!("{where_}: size, or height and width")),
    };
    if size.is_some_and(|(h, w)| {
        !(1..=MAX_IMAGE_SIDE).contains(&h) || !(1..=MAX_IMAGE_SIDE).contains(&w)
    }) {
        return Err(format!("{where_}: image sides are 1 to {MAX_IMAGE_SIDE}"));
    }
    let image_keys = [
        ("size", size.is_some()),
        ("layout", decl.layout.is_some()),
        ("format", decl.format.is_some()),
        ("resample", decl.resample.is_some()),
        ("crop", decl.crop.is_some()),
        ("upside_down", decl.upside_down.is_some()),
    ];
    if !entry.kind.is_image()
        && let Some((key, _)) = image_keys.iter().find(|(_, set)| *set)
    {
        return Err(format!("{where_}: {key} is for camera and depth entries"));
    }
    let layout = match decl.layout.as_deref() {
        None | Some("hwc") => ImageLayout::Hwc,
        Some("chw") => ImageLayout::Chw,
        Some(other) => return Err(format!("{where_}: layout {other:?}: hwc or chw")),
    };
    let format = match decl.format.as_deref() {
        None | Some("rgb8") => ImageFormat::Rgb8,
        Some("bgr8") => ImageFormat::Bgr8,
        Some("mono8") => ImageFormat::Mono8,
        Some(other) => return Err(format!("{where_}: format {other:?}: rgb8, bgr8 or mono8")),
    };
    let resample = match decl.resample.as_deref() {
        None => Resample::Bilinear,
        Some(name) => Resample::parse(name).ok_or_else(|| {
            format!(
                "{where_}: resample {name:?}: bilinear, bilinear_aa, bicubic_aa, lanczos3_aa or area"
            )
        })?,
    };
    if let Some(crop) = decl.crop
        && !(crop.is_finite() && crop > 0.0 && crop <= 1.0)
    {
        return Err(format!("{where_}: crop is a side fraction in (0, 1]"));
    }
    let dtype = decl
        .dtype
        .as_deref()
        .map(|d| {
            DType::from_name(d)
                .filter(|d| *d != DType::Utf8)
                .ok_or_else(|| format!("{where_}: dtype {d:?}"))
        })
        .transpose()?;
    let history = decl.history.unwrap_or(1);
    if !(1..=MAX_HISTORY).contains(&history) {
        return Err(format!("{where_}: history is 1 to {MAX_HISTORY} samples"));
    }
    let history_hz = decl
        .history_hz
        .map(|hz| rate(&where_, "history_hz", hz))
        .transpose()?;
    if history_hz.is_some() && history == 1 {
        return Err(format!("{where_}: history_hz without history"));
    }
    let source = match decl.source.as_deref() {
        None | Some("published") => Source::Published,
        Some("applied") => Source::Applied,
        Some(other) => return Err(format!("{where_}: source {other:?}: published or applied")),
    };
    if entry.kind.is_event() && (history > 1 || source == Source::Applied) {
        return Err(format!(
            "{where_}: events take no history or applied source"
        ));
    }
    Ok(Input {
        fields,
        shape: decl.shape,
        dtype,
        size,
        layout,
        format,
        resample,
        crop: decl.crop,
        upside_down: decl.upside_down.unwrap_or(false),
        point: ident(&where_, "point", decl.point.take())?,
        history,
        history_hz,
        max_age: decl
            .max_age_ms
            .map(|ms| millis(&where_, "max_age_ms", ms))
            .transpose()?,
        source,
        entry,
    })
}

fn action(name: &str, value: Value, policy_allow: Allows) -> Result<Action, String> {
    let where_ = format!("action {name:?}");
    let mut decl = parse_entry(&where_, value)?;
    if let Some(key) = decl
        .set_keys()
        .into_iter()
        .find(|k| INPUT_ONLY.contains(k) || *k == "max_age_ms")
        .filter(|k| !(*k == "fields" && matches!(decl.fields, Some(EntryFields::Schema(_)))))
    {
        return Err(format!("{where_}: {key} is an input key"));
    }
    let entry = common(
        &where_,
        entry_name("action", name)?,
        &mut decl,
        policy_allow,
    )?;
    if entry.kind.command_modes().is_empty() && entry.kind != Kind::Custom {
        return Err(format!("{where_}: a {} part takes no commands", entry.kind));
    }
    let horizon = decl.horizon.unwrap_or(1);
    if horizon == 0 || horizon as usize > MAX_CHUNK_ACTIONS {
        return Err(format!("{where_}: horizon is 1 to {MAX_CHUNK_ACTIONS}"));
    }
    if (decl.open.is_some() || decl.closed.is_some()) && entry.kind != Kind::Gripper {
        return Err(format!("{where_}: open and closed are for gripper actions"));
    }
    for (key, value) in [("open", decl.open), ("closed", decl.closed)] {
        if value.is_some_and(|v| !v.is_finite()) {
            return Err(format!("{where_}: {key} must be finite"));
        }
    }
    if decl.open.unwrap_or(1.0) == decl.closed.unwrap_or(0.0) {
        return Err(format!("{where_}: open and closed must differ"));
    }
    let t0 = match decl.t0.as_deref() {
        None | Some("snapshot") => ChunkStart::Snapshot,
        Some("send") => ChunkStart::Send,
        Some(other) => return Err(format!("{where_}: t0 {other:?}: snapshot or send")),
    };
    Ok(Action {
        command: ident(&where_, "command", decl.command.take())?,
        reference: match decl.reference.take() {
            None => None,
            Some(r) if r == "current" || r == "target" => Some(r),
            Some(r) => return Err(format!("{where_}: reference {r:?}: current or target")),
        },
        hz: decl.hz.map(|hz| rate(&where_, "hz", hz)).transpose()?,
        horizon,
        open: decl.open,
        closed: decl.closed,
        t0,
        ttl: decl
            .ttl_ms
            .map(|ms| millis(&where_, "ttl_ms", ms))
            .transpose()?,
        entry,
    })
}

impl Policy {
    /// Parses and normalises a policy map.
    pub fn parse(json: &str) -> Result<Self, String> {
        let mut de = serde_json::Deserializer::from_str(json);
        let decl: PolicyDecl = serde_path_to_error::deserialize(&mut de)
            .map_err(|err| format!("policy map: {err}"))?;
        de.end().map_err(|err| format!("policy map: {err}"))?;
        let hz = rate("policy", "hz", decl.hz)?;
        let allow = Allows::parse("policy", &decl.allow)?;
        let inputs = decl
            .inputs
            .map(|i| i.0)
            .unwrap_or_default()
            .into_iter()
            .map(|(name, value)| input(&name, value, allow))
            .collect::<Result<Vec<_>, _>>()?;
        let actions = decl
            .actions
            .map(|a| a.0)
            .unwrap_or_default()
            .into_iter()
            .map(|(name, value)| action(&name, value, allow))
            .collect::<Result<Vec<_>, _>>()?;
        if inputs.is_empty() && actions.is_empty() {
            return Err("policy map: no inputs and no actions".into());
        }
        Ok(Self {
            hz,
            inputs,
            actions,
        })
    }
}

/// How a policy is run (§6.4): the overrides `rlmesh.run(...)` takes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunOptions {
    /// The policy's name, for the table and refusals.
    pub name: String,
    /// The robot the actions target (needed when several are declared).
    pub robot: Option<ParticipantId>,
    /// The scope root.
    pub under: Option<PartPath>,
    /// Literal paths from the robot root, per entry.
    pub at: BTreeMap<String, PartPath>,
    /// Entries bound positionally.
    pub positional: Vec<String>,
    /// Extra conversions, for every entry.
    pub allow: Vec<Allow>,
    /// Capture values.
    pub captures: BTreeMap<String, String>,
}

impl RunOptions {
    /// Options for a policy named `name`.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Scoped under `path`.
    pub fn under(mut self, path: &str) -> Result<Self, String> {
        self.under = Some(PartPath::new(path).map_err(|e| e.to_string())?);
        Ok(self)
    }

    /// With a capture value.
    pub fn capture(mut self, name: &str, value: &str) -> Result<Self, String> {
        if !is_identifier(name, 32) || RUN_OPTIONS.contains(&name) {
            return Err(format!("capture name {name:?}"));
        }
        PartPath::new(value).map_err(|e| format!("capture {name}={value:?}: {e}"))?;
        if value.contains('/') {
            return Err(format!("capture {name}={value:?} is one segment"));
        }
        self.captures.insert(name.to_string(), value.to_string());
        Ok(self)
    }

    /// With an `at=` override.
    pub fn at(mut self, entry: &str, path: &str) -> Result<Self, String> {
        self.at.insert(
            entry.to_string(),
            PartPath::new(path).map_err(|e| e.to_string())?,
        );
        Ok(self)
    }

    /// With `names={entry: "positional"}`.
    pub fn positional(mut self, entry: &str) -> Self {
        self.positional.push(entry.to_string());
        self
    }

    /// With extra conversions.
    pub fn allow(mut self, allow: Allow) -> Self {
        self.allow.push(allow);
        self
    }

    /// For the robot `robot`.
    pub fn robot(mut self, robot: &str) -> Result<Self, String> {
        self.robot = Some(ParticipantId::new(robot).map_err(|e| e.to_string())?);
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shorthands_defaults_and_allow_merge() {
        let policy = Policy::parse(
            r#"{"hz": 15,
                "inputs": {"prompt": "text",
                           "image": {"kind": "camera", "at": "/scene/cam", "size": [224, 224],
                                     "allow": ["crop"]}},
                "actions": {"arm": {"kind": "joints", "at": "arm", "count": 6, "horizon": 10}},
                "allow": ["resize"]}"#,
        )
        .expect("parse");
        let prompt = &policy.inputs[0];
        assert_eq!((prompt.entry.kind, prompt.history), (Kind::Text, 1));
        assert!(prompt.entry.at.is_none());
        let image = &policy.inputs[1].entry;
        assert!(image.allow.has(Allow::Crop) && image.allow.has(Allow::Resize));
        assert_eq!(policy.inputs[1].size, Some((224, 224)));
        let arm = &policy.actions[0];
        assert_eq!(
            (arm.horizon, arm.hz, arm.t0),
            (10, None, ChunkStart::Snapshot)
        );
        assert!(arm.entry.allow.has(Allow::Resize) && !arm.entry.allow.has(Allow::Crop));
    }

    #[test]
    fn malformed_policies_are_refused_naming_the_entry() {
        for (json, needle) in [
            (r#"{"inputs": {"a": "text"}}"#, "hz"),
            (r#"{"hz": 0, "inputs": {"a": "text"}}"#, "hz 0"),
            (r#"{"hz": 1}"#, "no inputs"),
            (r#"{"hz": 1, "inputs": {"a": "points"}}"#, "core kind"),
            (
                r#"{"hz": 1, "inputs": {"a": {"kind": "camera", "colour": 1}}}"#,
                "colour",
            ),
            (
                r#"{"hz": 1, "inputs": {"a": {"kind": "camera", "horizon": 3}}}"#,
                "action key",
            ),
            (
                r#"{"hz": 1, "inputs": {"a": {"kind": "joints", "size": [1, 1]}}}"#,
                "camera and depth",
            ),
            (
                r#"{"hz": 1, "inputs": {"a": {"kind": "joints", "names": "positional"}}}"#,
                "count",
            ),
            (
                r#"{"hz": 1, "inputs": {"a": {"kind": "joints", "names": ["x", "x"]}}}"#,
                "repeated",
            ),
            (
                r#"{"hz": 1, "inputs": {"a": {"kind": "camera", "names": ["x"]}}}"#,
                "joints and hand",
            ),
            (
                r#"{"hz": 1, "inputs": {"a": {"kind": "joints", "history": 2000}}}"#,
                "1 to 1024",
            ),
            (
                r#"{"hz": 1, "inputs": {"a": {"kind": "joints", "at": "{at}/x"}}}"#,
                "run-time option",
            ),
            (
                r#"{"hz": 1, "inputs": {"a": {"kind": "x/foo"}}}"#,
                "explicit fields schema",
            ),
            (
                r#"{"hz": 1, "inputs": {"a": "text"}, "allow": ["exact"]}"#,
                "not a conversion",
            ),
            (
                r#"{"hz": 1, "actions": {"a": {"kind": "camera"}}}"#,
                "no commands",
            ),
            (
                r#"{"hz": 1, "actions": {"a": {"kind": "joints", "history": 3}}}"#,
                "input key",
            ),
            (
                r#"{"hz": 1, "actions": {"a": {"kind": "joints", "horizon": 0}}}"#,
                "horizon",
            ),
            (
                r#"{"hz": 1, "actions": {"a": {"kind": "gripper", "open": 1, "closed": 1}}}"#,
                "differ",
            ),
            (
                r#"{"hz": 1, "actions": {"a": {"kind": "joints", "open": 1}}}"#,
                "gripper",
            ),
        ] {
            let err = Policy::parse(json).expect_err(json);
            assert!(err.contains(needle), "{json}: {err}");
        }
    }
}
