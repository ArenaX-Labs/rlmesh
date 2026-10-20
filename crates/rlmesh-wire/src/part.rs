//! Parts and streams (§3, §10.5, §10.6): what the host makes of a robot's
//! declaration, and the payload layout every sample and action follows.
//!
//! An embodiment sends its authored robot map; the host normalises each part
//! to a [`PartSpec`] with host-assigned, never-reused channel ids. The
//! generic core (transport, edge queue, history store) only sees
//! [`StreamSpec`]s: a channel id, an owner, a key path and a typed
//! [`Layout`]. It never interprets kinds.
//!
//! # Packing (normative, §10.6)
//!
//! A sample's fields go in [`FieldSpec`] order, each C-contiguous and
//! little-endian, each starting at a multiple of 8 bytes from the start of
//! the sample (zero padding). The **stride** is the packed size rounded up
//! to a multiple of 8, so every sample of a batch, and every action of a
//! chunk, is 8-aligned relative to the payload start. `bool` is one byte, 0
//! or 1. A `utf8` field is allowed only as the single field of a part: its
//! payload is the raw UTF-8 and cannot be batched. An encoded (JPEG) image
//! is the sole payload of its sample, at most raw size + 16 KiB.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use crate::ids::{ChannelId, ParticipantId};
use crate::path::{APPLIED, PartPath};

/// The most fields a part has.
pub const MAX_FIELDS: usize = 16;
/// The most dimensions a field has.
pub const MAX_DIMS: usize = 4;
/// The most elements a field has.
pub const MAX_FIELD_ELEMENTS: u64 = 1 << 26;
/// The largest raw sample.
pub const MAX_SAMPLE_BYTES: usize = 32 << 20;
/// How much larger than raw an encoded image may be.
pub const MAX_ENCODED_OVERHEAD: usize = 16 << 10;
/// The longest text sample.
pub const MAX_TEXT_BYTES: usize = 4096;
/// The most parts a robot declares.
pub const MAX_PARTS: usize = 256;
/// The most samples one batch carries.
pub const MAX_BATCH: usize = 256;
/// The most actions one chunk carries.
pub const MAX_CHUNK_ACTIONS: usize = 1024;
/// The largest chunk payload.
pub const MAX_CHUNK_BYTES: usize = 4 << 20;
/// Small state samples ride the realtime class and are sent every sample by
/// default; larger ones ride bulk and are coalesced (§4.1, custom kinds).
pub const SMALL_SAMPLE_BYTES: usize = 4 << 10;

/// An element type. Wire codes are v1's `DType` (`rlmesh-spaces`), with
/// `utf8 = 32`; there is no `bfloat16` in v0.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// One byte, 0 or 1.
    Bool,
    /// `u8`.
    U8,
    /// `u16`.
    U16,
    /// `u32`.
    U32,
    /// `u64`.
    U64,
    /// `i8`.
    I8,
    /// `i16`.
    I16,
    /// `i32`.
    I32,
    /// `i64`.
    I64,
    /// IEEE binary16.
    F16,
    /// IEEE binary32.
    F32,
    /// IEEE binary64.
    F64,
    /// UTF-8 text: the single field of its part.
    Utf8,
}

impl DType {
    /// Every dtype.
    pub const ALL: [Self; 13] = [
        Self::Bool,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F16,
        Self::F32,
        Self::F64,
        Self::Utf8,
    ];

    /// Bytes per element (0 for `utf8`, which has no fixed size).
    pub fn size(self) -> u64 {
        match self {
            Self::Utf8 => 0,
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// The wire code.
    pub fn code(self) -> u32 {
        match self {
            Self::Bool => 1,
            Self::U8 => 2,
            Self::U16 => 3,
            Self::U32 => 4,
            Self::U64 => 5,
            Self::I8 => 6,
            Self::I16 => 7,
            Self::I32 => 8,
            Self::I64 => 9,
            Self::F16 => 10,
            Self::F32 => 11,
            Self::F64 => 12,
            Self::Utf8 => 32,
        }
    }

    /// The dtype a wire code names.
    pub fn from_code(code: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|d| d.code() == code)
    }

    /// The name a declaration uses (`float32`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::U64 => "uint64",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::F16 => "float16",
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::Utf8 => "utf8",
        }
    }

    /// The dtype a declaration names.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|d| d.name() == name)
    }

    /// Whether it is a float type.
    pub fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One field of a sample or an action: a dense little-endian row-major
/// tensor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FieldSpec {
    /// Its name, unique in the part.
    pub name: String,
    /// The element type.
    pub dtype: DType,
    /// The shape; empty is a scalar.
    pub shape: Vec<u32>,
    /// Its unit (§5.1); empty when unitless.
    pub unit: String,
    /// The rotation encoding (§5.3), for rotation fields; empty otherwise.
    pub rotation: String,
}

impl FieldSpec {
    /// A field with no unit or rotation.
    pub fn new(name: impl Into<String>, dtype: DType, shape: &[u32]) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            unit: String::new(),
            rotation: String::new(),
        }
    }

    /// Elements, `None` on overflow.
    pub fn elements(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(1_u64, |n, d| n.checked_mul(u64::from(*d)))
    }

    /// Raw bytes, `None` on overflow (0 for `utf8`).
    pub fn byte_len(&self) -> Option<u64> {
        self.elements()?.checked_mul(self.dtype.size())
    }

    /// `(height, width)` when this is a `u8` `[H, W, 3]` image.
    pub fn rgb8_dims(&self) -> Option<(u32, u32)> {
        match (self.dtype, self.shape.as_slice()) {
            (DType::U8, [h, w, 3]) if *h > 0 && *w > 0 => Some((*h, *w)),
            _ => None,
        }
    }
}

/// Where each field of a packed value sits.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Fixed-size fields at 8-aligned offsets.
    Packed {
        /// Byte offset of each field.
        offsets: Vec<usize>,
        /// Byte length of each field.
        lens: Vec<usize>,
        /// The value's size, rounded up to a multiple of 8.
        stride: usize,
    },
    /// One `utf8` field: the payload is the text, of any length up to
    /// [`MAX_TEXT_BYTES`].
    Text,
}

fn round8(n: usize) -> Option<usize> {
    n.checked_add(7).map(|n| n & !7)
}

impl Layout {
    /// The layout of `fields`, checking the §3.4 caps.
    pub fn of(fields: &[FieldSpec]) -> Result<Self, String> {
        if fields.is_empty() {
            return Err("no fields".into());
        }
        if fields.len() > MAX_FIELDS {
            return Err(format!("{} fields (at most {MAX_FIELDS})", fields.len()));
        }
        let mut seen = Vec::with_capacity(fields.len());
        for field in fields {
            if seen.contains(&field.name.as_str()) {
                return Err(format!("field {:?} is declared twice", field.name));
            }
            seen.push(field.name.as_str());
            if field.shape.len() > MAX_DIMS {
                return Err(format!(
                    "field {:?} has {} dims (at most {MAX_DIMS})",
                    field.name,
                    field.shape.len()
                ));
            }
            if field.shape.contains(&0) {
                return Err(format!("field {:?} has a zero dimension", field.name));
            }
            if field.elements().is_none_or(|n| n > MAX_FIELD_ELEMENTS) {
                return Err(format!(
                    "field {:?} has more than 2^26 elements",
                    field.name
                ));
            }
        }
        if fields.iter().any(|f| f.dtype == DType::Utf8) {
            return match fields {
                [field] if field.shape.is_empty() => Ok(Self::Text),
                [_] => Err("a utf8 field is a scalar".into()),
                _ => Err("utf8 is only allowed as the single field of a part".into()),
            };
        }
        let mut offsets = Vec::with_capacity(fields.len());
        let mut lens = Vec::with_capacity(fields.len());
        let mut end = 0_usize;
        for field in fields {
            let len = field
                .byte_len()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| format!("field {:?} is too large", field.name))?;
            let offset = round8(end).ok_or("too large")?;
            offsets.push(offset);
            lens.push(len);
            end = offset.checked_add(len).ok_or("too large")?;
        }
        let stride = round8(end).ok_or("too large")?;
        if stride > MAX_SAMPLE_BYTES {
            return Err(format!(
                "a value of {stride} bytes is over the {MAX_SAMPLE_BYTES}-byte cap"
            ));
        }
        Ok(Self::Packed {
            offsets,
            lens,
            stride,
        })
    }

    /// The stride of a packed layout (`None` for text).
    pub fn stride(&self) -> Option<usize> {
        match self {
            Self::Packed { stride, .. } => Some(*stride),
            Self::Text => None,
        }
    }

    /// The byte range of field `i` in one packed value.
    pub fn field(&self, i: usize) -> Option<std::ops::Range<usize>> {
        match self {
            Self::Packed { offsets, lens, .. } => {
                let (offset, len) = (*offsets.get(i)?, *lens.get(i)?);
                Some(offset..offset + len)
            }
            Self::Text => None,
        }
    }

    /// Packs one value from its fields' bytes, in order (zero padding).
    pub fn pack(&self, fields: &[&[u8]]) -> Result<Vec<u8>, String> {
        match self {
            Self::Text => match fields {
                [text] => Ok(text.to_vec()),
                _ => Err(format!("{} fields for a text value", fields.len())),
            },
            Self::Packed {
                offsets,
                lens,
                stride,
            } => {
                if fields.len() != offsets.len() {
                    return Err(format!(
                        "{} fields, declared {}",
                        fields.len(),
                        offsets.len()
                    ));
                }
                let mut out = vec![0_u8; *stride];
                for (i, bytes) in fields.iter().enumerate() {
                    if bytes.len() != lens[i] {
                        return Err(format!(
                            "field {i} is {} bytes, declared {}",
                            bytes.len(),
                            lens[i]
                        ));
                    }
                    out[offsets[i]..offsets[i] + lens[i]].copy_from_slice(bytes);
                }
                Ok(out)
            }
        }
    }
}

/// How a part's samples travel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// Packed raw fields.
    Raw,
    /// One baseline JPEG of a `u8` `[H, W, 3]` image at `quality` (1-100).
    Jpeg {
        /// JPEG quality, 1-100.
        quality: u8,
    },
}

impl Encoding {
    /// The name on the wire.
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Jpeg { .. } => "jpeg",
        }
    }
}

/// The traffic class a stream rides (§10.2). Higher classes are never queued
/// behind lower ones: the session maps them to separate transport sessions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChannelPriority {
    /// Session control, and chunks of `priority: control` parts.
    Control,
    /// Small state samples, chunks, `_applied`.
    Realtime,
    /// Tactile, small custom kinds, events, history replies.
    Data,
    /// Cameras, depth, large custom kinds.
    Bulk,
}

impl ChannelPriority {
    /// The name a declaration uses.
    pub fn name(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Realtime => "realtime",
            Self::Data => "data",
            Self::Bulk => "bulk",
        }
    }

    /// The class a declaration names.
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Control, Self::Realtime, Self::Data, Self::Bulk]
            .into_iter()
            .find(|p| p.name() == name)
    }
}

/// How the edge merges a new chunk into its queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MergePolicy {
    /// Keep committed actions before `replace_from`, replace from there on.
    Replace,
    /// Later (§19): refused at declaration.
    Blend,
    /// Later (§19): refused at declaration.
    RtcPrefix,
}

/// Bounds on one command field (§3.2, §9.6).
#[derive(Clone, Debug, PartialEq)]
pub struct FieldLimit {
    /// The command field.
    pub field: String,
    /// Lower bounds: one (broadcast) or one per element; empty with
    /// `max_norm`.
    pub low: Vec<f64>,
    /// Upper bounds, as `low`.
    pub high: Vec<f64>,
    /// A norm bound on a delta translation or rotation vector.
    pub max_norm: Option<f64>,
}

/// What a commandable part accepts (§3.2).
#[derive(Clone, Debug, PartialEq)]
pub struct CommandSpec {
    /// The command mode (`position`, `delta`, ...).
    pub mode: String,
    /// One action's fields.
    pub fields: Vec<FieldSpec>,
    /// Bounds on command fields; the edge clamps.
    pub limits: Vec<FieldLimit>,
    /// The robot's own safe behaviour, by id (`stop`, `brake`).
    pub safe_behavior: String,
    /// The robot's hold allowance once valid coverage ends (§9.5).
    pub hold_max: Duration,
    /// The edge's cap on a chunk's life from receipt (§9.1).
    pub max_action_age: Duration,
    /// How chunks merge: `Replace` only.
    pub merge: MergePolicy,
    /// `current` or `target` (delta modes).
    pub reference: String,
    /// The command group within the robot (§9.7).
    pub group: Option<String>,
    /// Commands snap to 0 or 1 at 0.5.
    pub binary: bool,
    /// During Hold, repeat the last dispatched target (position-type modes);
    /// otherwise run the safe behaviour.
    pub hold_last: bool,
}

impl CommandSpec {
    /// The default hold allowance.
    pub const DEFAULT_HOLD_MAX: Duration = Duration::from_millis(100);
    /// The default open-loop budget.
    pub const DEFAULT_MAX_ACTION_AGE: Duration = Duration::from_millis(200);

    /// Whether this is a delta mode (§9.3): each action is dispatched at
    /// most once, and Hold dispatches nothing.
    pub fn is_delta(&self) -> bool {
        self.mode == "delta"
    }

    /// One action's layout.
    pub fn layout(&self) -> Result<Layout, String> {
        Layout::of(&self.fields)
    }

    /// The limits of command field `name`.
    pub fn limit(&self, name: &str) -> Option<&FieldLimit> {
        self.limits.iter().find(|l| l.field == name)
    }
}

/// A publication ceiling (§8.3), parsed from `PartSpec::send`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SendPolicy {
    /// At most this many samples per second (newest wins); `None`: every
    /// sample.
    pub hz: Option<f64>,
    /// A bitrate budget in Mbit/s.
    pub max_mbps: Option<f64>,
}

impl SendPolicy {
    /// Parses `"every"`, `"latest(hz=10)"`, `"latest(max_mbps=8)"` or
    /// `"latest(hz=10, max_mbps=8)"`.
    pub fn parse(send: &str) -> Result<Self, String> {
        let send = send.trim();
        if send == "every" {
            return Ok(Self::default());
        }
        let args = send
            .strip_prefix("latest(")
            .and_then(|rest| rest.strip_suffix(')'))
            .ok_or_else(|| format!("send {send:?}: expected every or latest(...)"))?;
        let mut policy = Self::default();
        for arg in args.split(',').map(str::trim).filter(|a| !a.is_empty()) {
            let (key, value) = arg
                .split_once('=')
                .ok_or_else(|| format!("send {send:?}: {arg:?} is not key=value"))?;
            let value: f64 = value
                .trim()
                .parse()
                .ok()
                .filter(|v: &f64| v.is_finite() && *v > 0.0)
                .ok_or_else(|| format!("send {send:?}: {key} must be finite and positive"))?;
            match key.trim() {
                "hz" if policy.hz.is_none() => policy.hz = Some(value),
                "max_mbps" if policy.max_mbps.is_none() => policy.max_mbps = Some(value),
                other => return Err(format!("send {send:?}: unexpected {other:?}")),
            }
        }
        if policy == Self::default() {
            return Err(format!("send {send:?}: latest() needs hz or max_mbps"));
        }
        Ok(policy)
    }
}

/// A part, as the host normalised it.
#[derive(Clone, Debug, PartialEq)]
pub struct PartSpec {
    /// Its path within the robot.
    pub path: PartPath,
    /// A core kind or `ns/name`; opaque below the binder.
    pub kind: String,
    /// Its stream's id, never reused within the session.
    pub channel: ChannelId,
    /// Its applied stream's id (commandable parts only).
    pub applied_channel: Option<ChannelId>,
    /// The sample layout, in payload order.
    pub fields: Vec<FieldSpec>,
    /// The source rate; 0 for events.
    pub hz: f64,
    /// The publication ceiling (§8.3).
    pub send: String,
    /// The frame values are expressed in.
    pub frame: String,
    /// Element names (joint names).
    pub names: Vec<String>,
    /// How samples travel.
    pub encoding: Encoding,
    /// The robot-side freshness bound, if declared.
    pub max_age: Option<Duration>,
    /// What it accepts, when commandable.
    pub command: Option<CommandSpec>,
    /// Everything else the declaration carried (`doc`, `range`, `x_*`).
    pub attrs: BTreeMap<String, String>,
    /// Its traffic class.
    pub priority: ChannelPriority,
    /// The robot: the embodiment that declared it.
    pub owner: ParticipantId,
    /// An event part (text, reward, success): reliable, never decimated.
    pub event: bool,
    /// Features a consumer must implement to use it.
    pub requires: Vec<String>,
    /// The edge control rate (commandable parts).
    pub control_hz: f64,
}

impl PartSpec {
    /// Its sample stream.
    pub fn stream(&self) -> Result<StreamSpec, String> {
        Ok(StreamSpec {
            channel: self.channel,
            owner: self.owner.clone(),
            path: self.path.to_string(),
            layout: Layout::of(&self.fields)?,
            fields: self.fields.clone(),
            encoding: self.encoding,
            priority: self.priority,
            event: self.event,
            max_age: self.max_age,
            applied: false,
        })
    }

    /// Its applied stream (§9.4): the command fields, on
    /// `{path}/_applied`, realtime class.
    pub fn applied_stream(&self) -> Option<Result<StreamSpec, String>> {
        let (command, channel) = (self.command.as_ref()?, self.applied_channel?);
        Some(command.layout().map(|layout| StreamSpec {
            channel,
            owner: self.owner.clone(),
            path: self.path.applied(),
            layout,
            fields: command.fields.clone(),
            encoding: Encoding::Raw,
            priority: ChannelPriority::Realtime,
            event: false,
            max_age: None,
            applied: true,
        }))
    }

    /// The edge control period (from `control_hz`), at least 1 ns.
    pub fn control_period(&self) -> Duration {
        if self.control_hz.is_finite() && self.control_hz > 0.0 {
            Duration::from_secs_f64(1.0 / self.control_hz).max(Duration::from_nanos(1))
        } else {
            Duration::from_millis(1)
        }
    }

    /// The parsed publication ceiling (`every` when it does not parse).
    pub fn send_policy(&self) -> SendPolicy {
        SendPolicy::parse(&self.send).unwrap_or_default()
    }
}

impl fmt::Display for PartSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}/{} ({}) @{}Hz {}",
            self.channel,
            self.owner,
            self.path,
            self.kind,
            self.hz,
            self.priority.name()
        )
    }
}

/// One stream: what a publisher, a subscriber or the history store needs.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamSpec {
    /// Its id.
    pub channel: ChannelId,
    /// The robot that publishes it.
    pub owner: ParticipantId,
    /// Its key path under `d/{owner}/` (`arm`, `arm/_applied`).
    pub path: String,
    /// Its fields.
    pub fields: Vec<FieldSpec>,
    /// Their packing.
    pub layout: Layout,
    /// How samples travel.
    pub encoding: Encoding,
    /// Its traffic class.
    pub priority: ChannelPriority,
    /// An event stream.
    pub event: bool,
    /// The freshness bound, if declared.
    pub max_age: Option<Duration>,
    /// A derived `_applied` stream.
    pub applied: bool,
}

impl StreamSpec {
    /// Whether several samples can share a batch (raw packed values).
    pub fn batchable(&self) -> bool {
        self.encoding == Encoding::Raw && matches!(self.layout, Layout::Packed { .. })
    }

    /// The check every sample payload gets before any view of it (§10.6):
    /// the length, valid booleans and valid UTF-8. An applied sample with
    /// nothing dispatched has no payload.
    pub fn check_payload(&self, payload: &[u8]) -> Result<(), String> {
        match (&self.layout, self.encoding) {
            (Layout::Text, _) => {
                if payload.len() > MAX_TEXT_BYTES {
                    return Err(format!(
                        "{} bytes of text (at most {MAX_TEXT_BYTES})",
                        payload.len()
                    ));
                }
                std::str::from_utf8(payload).map_err(|err| format!("not UTF-8: {err}"))?;
                Ok(())
            }
            (Layout::Packed { stride, .. }, Encoding::Jpeg { .. }) => {
                if payload.is_empty() || payload.len() > stride + MAX_ENCODED_OVERHEAD {
                    return Err(format!(
                        "an encoded image of {} bytes (raw {stride})",
                        payload.len()
                    ));
                }
                Ok(())
            }
            (layout @ Layout::Packed { stride, .. }, Encoding::Raw) => {
                if payload.len() != *stride {
                    return Err(format!("{} bytes, stride {stride}", payload.len()));
                }
                check_bools(&self.fields, layout, payload)
            }
        }
    }
}

/// Every `bool` element of one packed value is 0 or 1.
pub fn check_bools(fields: &[FieldSpec], layout: &Layout, value: &[u8]) -> Result<(), String> {
    for (i, field) in fields.iter().enumerate() {
        if field.dtype != DType::Bool {
            continue;
        }
        let range = layout.field(i).ok_or("layout mismatch")?;
        if value
            .get(range)
            .is_none_or(|bytes| bytes.iter().any(|b| *b > 1))
        {
            return Err(format!(
                "field {:?} has a bool other than 0 or 1",
                field.name
            ));
        }
    }
    Ok(())
}

/// Whether `path` is an applied stream's key path.
pub fn is_applied_path(path: &str) -> bool {
    path.rsplit_once('/')
        .is_some_and(|(_, last)| last == APPLIED)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn pid(id: &str) -> ParticipantId {
        ParticipantId::new(id).expect("id")
    }

    /// A UR-style arm: joint positions in, positions and velocities out.
    pub(crate) fn arm() -> PartSpec {
        PartSpec {
            path: PartPath::new("arm").expect("path"),
            kind: "x/joints".into(),
            channel: ChannelId(1),
            applied_channel: Some(ChannelId(2)),
            fields: vec![
                FieldSpec::new("pos", DType::F32, &[6]),
                FieldSpec::new("vel", DType::F32, &[6]),
            ],
            hz: 500.0,
            send: "every".into(),
            frame: String::new(),
            names: (0..6).map(|i| format!("j{i}")).collect(),
            encoding: Encoding::Raw,
            max_age: Some(Duration::from_millis(20)),
            command: Some(CommandSpec {
                mode: "position".into(),
                fields: vec![FieldSpec::new("pos", DType::F32, &[6])],
                limits: vec![FieldLimit {
                    field: "pos".into(),
                    low: vec![-3.0],
                    high: vec![3.0],
                    max_norm: None,
                }],
                safe_behavior: "stop".into(),
                hold_max: CommandSpec::DEFAULT_HOLD_MAX,
                max_action_age: CommandSpec::DEFAULT_MAX_ACTION_AGE,
                merge: MergePolicy::Replace,
                reference: String::new(),
                group: Some("arm".into()),
                binary: false,
                hold_last: true,
            }),
            attrs: BTreeMap::from([("doc".to_string(), "the arm".to_string())]),
            priority: ChannelPriority::Realtime,
            owner: pid("ur10e"),
            event: false,
            requires: Vec::new(),
            control_hz: 500.0,
        }
    }

    /// A JPEG wrist camera.
    pub(crate) fn camera() -> PartSpec {
        PartSpec {
            path: PartPath::new("wrist/cam").expect("path"),
            kind: "camera".into(),
            channel: ChannelId(3),
            applied_channel: None,
            fields: vec![FieldSpec::new("image", DType::U8, &[480, 640, 3])],
            hz: 30.0,
            send: "latest(hz=30, max_mbps=8)".into(),
            frame: "wrist_optical".into(),
            names: Vec::new(),
            encoding: Encoding::Jpeg { quality: 80 },
            max_age: None,
            command: None,
            attrs: BTreeMap::new(),
            priority: ChannelPriority::Bulk,
            owner: pid("ur10e"),
            event: false,
            requires: vec!["x.depth".into()],
            control_hz: 0.0,
        }
    }

    #[test]
    fn fields_pack_at_8_aligned_offsets_with_an_8_aligned_stride() {
        let layout = Layout::of(&[
            FieldSpec::new("a", DType::F32, &[3]),
            FieldSpec::new("flag", DType::Bool, &[]),
            FieldSpec::new("b", DType::F64, &[2]),
        ])
        .expect("layout");
        assert_eq!(layout.field(0), Some(0..12));
        assert_eq!(layout.field(1), Some(16..17));
        assert_eq!(layout.field(2), Some(24..40));
        assert_eq!(layout.stride(), Some(40));
        // An odd-sized float32 field still strides by 8.
        let one = Layout::of(&[FieldSpec::new("x", DType::F32, &[])]).expect("layout");
        assert_eq!(one.stride(), Some(8));
        let packed = layout.pack(&[&[1; 12], &[1], &[2; 16]]).expect("pack");
        assert_eq!(packed.len(), 40);
        assert_eq!(&packed[12..16], &[0; 4], "zero padding");
        assert!(layout.pack(&[&[1; 12], &[1]]).is_err());
    }

    #[test]
    fn layouts_enforce_the_caps_and_the_utf8_rule() {
        let field = |n: &str, d, s: &[u32]| FieldSpec::new(n, d, s);
        assert_eq!(
            Layout::of(&[field("text", DType::Utf8, &[])]),
            Ok(Layout::Text)
        );
        for bad in [
            vec![],
            vec![field("t", DType::Utf8, &[]), field("x", DType::F32, &[])],
            vec![field("t", DType::Utf8, &[2])],
            vec![field("x", DType::F32, &[1, 1, 1, 1, 1])],
            vec![field("x", DType::F32, &[0])],
            vec![field("x", DType::F32, &[1 << 14, 1 << 13])],
            vec![field("x", DType::F32, &[2]), field("x", DType::F32, &[2])],
            (0..17)
                .map(|i| field(&format!("f{i}"), DType::U8, &[]))
                .collect(),
        ] {
            assert!(Layout::of(&bad).is_err(), "{bad:?} accepted");
        }
        assert!(Layout::of(&[field("hd", DType::U8, &[1080, 1920, 3])]).is_ok());
        assert!(
            Layout::of(&[field("big", DType::U8, &[4096, 4096, 3])]).is_err(),
            "over 32 MiB"
        );
    }

    #[test]
    fn payloads_are_checked_against_the_stream() {
        let arm = arm();
        let stream = arm.stream().expect("stream");
        assert!(stream.check_payload(&[0; 48]).is_ok());
        assert!(stream.check_payload(&[0; 44]).is_err());
        let applied = arm.applied_stream().expect("commandable").expect("layout");
        assert_eq!(
            (
                applied.channel,
                applied.path.as_str(),
                applied.layout.stride()
            ),
            (ChannelId(2), "arm/_applied", Some(24))
        );
        assert!(applied.applied && is_applied_path(&applied.path));
        let camera = camera().stream().expect("stream");
        assert!(!camera.batchable());
        assert!(camera.check_payload(&[0; 60_000]).is_ok());
        assert!(camera.check_payload(&[]).is_err());
        assert!(
            camera
                .check_payload(&vec![0; 921_600 + MAX_ENCODED_OVERHEAD + 1])
                .is_err()
        );
        let flags = StreamSpec {
            fields: vec![FieldSpec::new("on", DType::Bool, &[2])],
            layout: Layout::of(&[FieldSpec::new("on", DType::Bool, &[2])]).expect("layout"),
            ..stream.clone()
        };
        assert!(flags.check_payload(&[1, 0, 0, 0, 0, 0, 0, 0]).is_ok());
        assert!(flags.check_payload(&[2, 0, 0, 0, 0, 0, 0, 0]).is_err());
        let text = StreamSpec {
            fields: vec![FieldSpec::new("text", DType::Utf8, &[])],
            layout: Layout::Text,
            ..stream
        };
        assert!(text.check_payload("put the cup down".as_bytes()).is_ok());
        assert!(text.check_payload(&[0xff, 0xfe]).is_err());
        assert!(text.check_payload(&[b'a'; MAX_TEXT_BYTES + 1]).is_err());
    }

    #[test]
    fn send_policies_parse() {
        assert_eq!(SendPolicy::parse("every"), Ok(SendPolicy::default()));
        assert_eq!(
            SendPolicy::parse("latest(hz=10, max_mbps=8)"),
            Ok(SendPolicy {
                hz: Some(10.0),
                max_mbps: Some(8.0)
            })
        );
        for bad in [
            "",
            "latest()",
            "latest(hz=0)",
            "latest(hz=10, hz=2)",
            "latest(fps=3)",
            "batch(10)",
        ] {
            assert!(SendPolicy::parse(bad).is_err(), "{bad:?} accepted");
        }
        assert_eq!(camera().send_policy().max_mbps, Some(8.0));
    }

    #[test]
    fn dtypes_use_the_v1_codes_and_utf8_is_32() {
        for dtype in DType::ALL {
            assert_eq!(DType::from_code(dtype.code()), Some(dtype));
            assert_eq!(DType::from_name(dtype.name()), Some(dtype));
        }
        assert_eq!(DType::F32.code(), 11);
        assert_eq!(DType::Utf8.code(), 32);
        assert_eq!(DType::from_code(13), None, "no bfloat16");
        assert_eq!(arm().control_period(), Duration::from_millis(2));
    }
}
