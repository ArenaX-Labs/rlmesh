//! A policy map built in Rust: [`PolicyBuilder`], and what the
//! [`inputs`](crate::spec::inputs) and [`actions`](crate::spec::actions)
//! builders share.
//!
//! Every builder lowers to the JSON entry it stands for, and the policy is
//! parsed by [`Policy::parse`], so the normaliser's rules and refusals apply
//! as they are. Only what JSON cannot carry (a NaN or an infinity) is
//! refused before it, in the normaliser's words.

#[allow(unused_imports)]
use crate::prelude::*;

use serde::Serialize;
use serde::ser::{Error as _, SerializeMap, SerializeSeq, Serializer};
use serde_json::Value;

use super::{Allow, Numbers, Policy, crop, finite, millis, numbers, range, rate};
use crate::wire::part::DType;

/// One number for every element, or one per element: an entry's `scale`
/// and `offset`.
///
/// `0.5` and `[1.0, 1.0, 0.5]` convert into it.
#[derive(Clone, Debug, PartialEq)]
pub enum PerElement {
    /// The same number for every element.
    All(f64),
    /// One number per element, in order.
    Each(Vec<f64>),
}

impl From<f64> for PerElement {
    fn from(value: f64) -> Self {
        Self::All(value)
    }
}

impl From<Vec<f64>> for PerElement {
    fn from(values: Vec<f64>) -> Self {
        Self::Each(values)
    }
}

impl From<&[f64]> for PerElement {
    fn from(values: &[f64]) -> Self {
        Self::Each(values.to_vec())
    }
}

impl<const N: usize> From<[f64; N]> for PerElement {
    fn from(values: [f64; N]) -> Self {
        Self::Each(values.to_vec())
    }
}

/// A custom kind's field: its element type and shape, written
/// `["float32", [17, 3]]` in the map.
///
/// ```
/// use rlmesh_session_core::spec::inputs::Array;
/// use rlmesh_session_core::wire::DType;
///
/// let keypoints = Array::new(DType::F32, [17, 3]);
/// assert_eq!(keypoints.shape(), &[17, 3]);
/// let score = Array::new(DType::F32, []); // a scalar
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Array {
    dtype: DType,
    shape: Vec<u32>,
}

impl Array {
    /// `dtype` elements in `shape`, `[]` for a scalar.
    pub fn new(dtype: DType, shape: impl IntoIterator<Item = u32>) -> Self {
        Self {
            dtype,
            shape: shape.into_iter().collect(),
        }
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The shape.
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }
}

impl Serialize for Array {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(2))?;
        seq.serialize_element(self.dtype.name())?;
        seq.serialize_element(&self.shape)?;
        seq.end()
    }
}

/// A value an entry's key holds, as its JSON is written.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Val {
    Str(String),
    Bool(bool),
    Num(f64),
    Int(u32),
    Pair(f64, f64),
    Names(Vec<String>),
    Dims(Vec<u32>),
    Numbers(PerElement),
    Map(Vec<(String, String)>),
    Schema(Vec<(String, Array)>),
    Json(Value),
}

/// A number as JSON: a whole number as an integer, as a hand-written map
/// has it.
struct Num(f64);

impl Serialize for Num {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // 2^53: every whole number up to it is exact in both.
        const EXACT: f64 = 9_007_199_254_740_992.0;
        let v = self.0;
        if (-EXACT..=EXACT).contains(&v) && v == (v as i64) as f64 {
            s.serialize_i64(v as i64)
        } else {
            s.serialize_f64(v)
        }
    }
}

impl Serialize for Val {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Str(v) => s.serialize_str(v),
            Self::Bool(v) => s.serialize_bool(*v),
            Self::Num(v) => Num(*v).serialize(s),
            Self::Int(v) => s.serialize_u32(*v),
            Self::Pair(low, high) => [Num(*low), Num(*high)].serialize(s),
            Self::Names(v) => v.serialize(s),
            Self::Dims(v) => v.serialize(s),
            Self::Numbers(PerElement::All(v)) => Num(*v).serialize(s),
            Self::Numbers(PerElement::Each(v)) => {
                let mut seq = s.serialize_seq(Some(v.len()))?;
                for n in v {
                    seq.serialize_element(&Num(*n))?;
                }
                seq.end()
            }
            Self::Map(pairs) => {
                let mut map = s.serialize_map(Some(pairs.len()))?;
                for (k, v) in pairs {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
            Self::Schema(fields) => {
                let mut map = s.serialize_map(Some(fields.len()))?;
                for (k, v) in fields {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
            Self::Json(v) => v.serialize(s),
        }
    }
}

/// An entry being built: its kind and the keys set, in the order set.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Draft {
    kind: String,
    keys: Vec<(&'static str, Val)>,
    /// The keys the kind takes, when it is known only at run time
    /// (`inputs::of`).
    takes: Option<&'static [&'static str]>,
    /// Why the entry cannot be built, found before the policy is.
    problem: Option<String>,
}

impl Draft {
    pub(crate) fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            keys: Vec::new(),
            takes: None,
            problem: None,
        }
    }

    /// Sets `key`, replacing an earlier value.
    pub(crate) fn set(&mut self, key: &'static str, value: Val) {
        match self.keys.iter_mut().find(|(k, _)| *k == key) {
            Some((_, slot)) => *slot = value,
            None => self.keys.push((key, value)),
        }
    }

    /// Adds names to the list `key`.
    pub(crate) fn extend(&mut self, key: &'static str, names: impl IntoIterator<Item = String>) {
        if let Some((_, Val::Names(list))) = self.keys.iter_mut().find(|(k, _)| *k == key) {
            list.extend(names);
        } else {
            self.set(key, Val::Names(names.into_iter().collect()));
        }
    }

    /// Sets `name` in the map `key` (`units`).
    pub(crate) fn put(&mut self, key: &'static str, name: String, value: String) {
        if let Some((_, Val::Map(pairs))) = self.keys.iter_mut().find(|(k, _)| *k == key) {
            match pairs.iter_mut().find(|(n, _)| *n == name) {
                Some((_, slot)) => *slot = value,
                None => pairs.push((name, value)),
            }
        } else {
            self.set(key, Val::Map(vec![(name, value)]));
        }
    }

    /// Sets the custom kind's field `name`.
    pub(crate) fn field(&mut self, name: String, array: Array) {
        if let Some((_, Val::Schema(fields))) = self.keys.iter_mut().find(|(k, _)| *k == "fields") {
            match fields.iter_mut().find(|(n, _)| *n == name) {
                Some((_, slot)) => *slot = array,
                None => fields.push((name, array)),
            }
        } else {
            self.set("fields", Val::Schema(vec![(name, array)]));
        }
    }

    /// Takes only `keys`, refusing any other when the policy is built.
    pub(crate) fn taking(mut self, keys: &'static [&'static str]) -> Self {
        self.takes = Some(keys);
        self
    }

    /// Refuses to build, for `problem`.
    pub(crate) fn refused(mut self, problem: String) -> Self {
        self.problem = Some(problem);
        self
    }

    /// What the normaliser would refuse but JSON cannot carry: a NaN or an
    /// infinity, refused in the normaliser's words; and what the builder
    /// found wrong.
    pub(crate) fn check(&self, where_: &str) -> Result<(), String> {
        if let Some(problem) = &self.problem {
            return Err(format!("{where_}: {problem}"));
        }
        if let Some(takes) = self.takes
            && let Some((key, _)) = self.keys.iter().find(|(k, _)| !takes.contains(k))
        {
            return Err(format!("{where_}: a {} input takes no {key}", self.kind));
        }
        for (key, value) in &self.keys {
            match value {
                Val::Num(v) if !v.is_finite() => match *key {
                    "hz" | "history_hz" => rate(where_, key, *v).map(drop)?,
                    "crop" => crop(where_, *v).map(drop)?,
                    "max_age_ms" | "ttl_ms" => millis(where_, key, *v).map(drop)?,
                    _ => finite(where_, key, *v).map(drop)?,
                },
                Val::Pair(low, high) if !(low.is_finite() && high.is_finite()) => {
                    range(where_, [*low, *high]).map(drop)?;
                }
                Val::Numbers(PerElement::Each(v)) if v.iter().any(|n| !n.is_finite()) => {
                    numbers(where_, key, Some(Numbers::Many(v.clone()))).map(drop)?;
                }
                Val::Numbers(PerElement::All(v)) if !v.is_finite() => {
                    numbers(where_, key, Some(Numbers::One(*v))).map(drop)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

impl Serialize for Draft {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.check(&self.kind).map_err(S::Error::custom)?;
        let mut map = s.serialize_map(Some(1 + self.keys.len()))?;
        map.serialize_entry("kind", &self.kind)?;
        for (key, value) in &self.keys {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

/// An entry as given: built, or written as JSON.
#[derive(Clone, Debug, PartialEq)]
enum Written {
    Built(Draft),
    Json(Value),
}

impl Written {
    fn to_json(&self, where_: &str) -> Result<String, String> {
        match self {
            Self::Built(draft) => draft.check(where_)?,
            Self::Json(_) => {}
        }
        let text = match self {
            Self::Built(draft) => serde_json::to_string(draft),
            Self::Json(value) => serde_json::to_string(value),
        };
        text.map_err(|err| format!("{where_}: {err}"))
    }
}

/// An input entry for [`PolicyBuilder::input`]: one of the
/// [`inputs`](crate::spec::inputs) builders, or the entry as JSON (a
/// [`serde_json::Value`], such as `json!({"kind": "text"})` or the
/// shorthand `json!("text")`).
#[derive(Clone, Debug, PartialEq)]
pub struct InputEntry(Written);

impl InputEntry {
    pub(crate) fn built(draft: Draft) -> Self {
        Self(Written::Built(draft))
    }
}

impl From<Value> for InputEntry {
    fn from(value: Value) -> Self {
        Self(Written::Json(value))
    }
}

/// An action entry for [`PolicyBuilder::action`]: one of the
/// [`actions`](crate::spec::actions) builders, or the entry as JSON.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionEntry(Written);

impl ActionEntry {
    pub(crate) fn built(draft: Draft) -> Self {
        Self(Written::Built(draft))
    }
}

impl From<Value> for ActionEntry {
    fn from(value: Value) -> Self {
        Self(Written::Json(value))
    }
}

/// A policy map, entry by entry: the replan rate, the named inputs and
/// actions, and the conversions every entry may use. [`build`] parses it
/// into the same [`Policy`] that [`Policy::parse`] gives for its JSON.
///
/// Entries come from the [`inputs`](crate::spec::inputs) and
/// [`actions`](crate::spec::actions) builders or as JSON, mixed freely; a
/// name given twice is refused, as it is in JSON. Keys are checked when the
/// policy is built, with the normaliser's refusals, each naming the entry.
///
/// ```
/// use rlmesh_session_core::spec::actions::{self, JointCommand};
/// use rlmesh_session_core::spec::inputs;
/// use rlmesh_session_core::spec::policy::Allow;
/// use rlmesh_session_core::spec::Policy;
///
/// let policy = Policy::builder()
///     .hz(30)
///     .input("cam", inputs::camera().at("hallo-cam:cam").size(224))
///     .input("task", serde_json::json!("text"))
///     .action(
///         "left",
///         actions::joints()
///             .at("gimbal-left:servos")
///             .command(JointCommand::Position)
///             .hz(50)
///             .chunk(4),
///     )
///     .allow([Allow::Resize])
///     .build()?;
/// assert_eq!(policy.actions[0].chunk, 4);
/// # Ok::<(), String>(())
/// ```
///
/// [`build`]: PolicyBuilder::build
#[derive(Clone, Debug, Default, PartialEq)]
#[must_use = "a policy builder does nothing until it is built"]
pub struct PolicyBuilder {
    hz: Option<f64>,
    inputs: Vec<(String, InputEntry)>,
    actions: Vec<(String, ActionEntry)>,
    allow: Vec<Allow>,
}

impl Policy {
    /// A policy map built entry by entry; see [`PolicyBuilder`].
    pub fn builder() -> PolicyBuilder {
        PolicyBuilder::default()
    }
}

impl PolicyBuilder {
    /// The replan rate: how often the policy starts a prediction, in Hz.
    pub fn hz(mut self, hz: impl Into<f64>) -> Self {
        self.hz = Some(hz.into());
        self
    }

    /// Adds the input `name`, its key in `predict`'s observation.
    pub fn input(mut self, name: impl Into<String>, entry: impl Into<InputEntry>) -> Self {
        self.inputs.push((name.into(), entry.into()));
        self
    }

    /// Adds the action `name`, its key in `predict`'s actions.
    pub fn action(mut self, name: impl Into<String>, entry: impl Into<ActionEntry>) -> Self {
        self.actions.push((name.into(), entry.into()));
        self
    }

    /// Conversions every entry may use, beside each entry's own.
    pub fn allow(mut self, allow: impl IntoIterator<Item = Allow>) -> Self {
        self.allow.extend(allow);
        self
    }

    /// The policy map as JSON, entries in the order added.
    pub fn to_json(&self) -> Result<String, String> {
        let mut out = String::from("{");
        if let Some(hz) = self.hz {
            if !hz.is_finite() {
                rate("policy", "hz", hz)?;
            }
            out.push_str("\"hz\":");
            out.push_str(&serde_json::to_string(&Num(hz)).map_err(|e| e.to_string())?);
        }
        let inputs = self.inputs.iter().map(|(n, e)| (n, &e.0));
        let actions = self.actions.iter().map(|(n, e)| (n, &e.0));
        for (key, noun, entries) in [
            ("inputs", "input", inputs.collect::<Vec<_>>()),
            ("actions", "action", actions.collect()),
        ] {
            if entries.is_empty() {
                continue;
            }
            if out.len() > 1 {
                out.push(',');
            }
            out.push_str(&format!("\"{key}\":{{"));
            for (i, (name, entry)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let quoted = serde_json::to_string(name).map_err(|e| e.to_string())?;
                out.push_str(&quoted);
                out.push(':');
                out.push_str(&entry.to_json(&format!("{noun} {name:?}"))?);
            }
            out.push('}');
        }
        if !self.allow.is_empty() {
            if out.len() > 1 {
                out.push(',');
            }
            let names: Vec<&str> = self.allow.iter().map(|a| a.name()).collect();
            out.push_str("\"allow\":");
            out.push_str(&serde_json::to_string(&names).map_err(|e| e.to_string())?);
        }
        out.push('}');
        Ok(out)
    }

    /// Parses the map with [`Policy::parse`].
    pub fn build(&self) -> Result<Policy, String> {
        Policy::parse(&self.to_json()?)
    }
}

/// The JSON key a setter writes.
macro_rules! key {
    (max_age) => {
        "max_age_ms"
    };
    (ttl) => {
        "ttl_ms"
    };
    (unit) => {
        "units"
    };
    (field) => {
        "fields"
    };
    (positional) => {
        "names"
    };
    (custom_command) => {
        "command"
    };
    ($key:ident) => {
        stringify!($key)
    };
}
pub(crate) use key;

/// The setter for one key, as every builder that takes it has it.
macro_rules! setter {
    (at) => {
        /// The part's address, the entry's `at`: `robot:path`, a path below
        /// the scope (`wrist/cam`), or `/path` from the robot's root; a
        /// segment may be `*` or a `{capture}`. Without one, the entry binds
        /// any part of its kind.
        pub fn at(mut self, at: impl Into<String>) -> Self {
            self.0.set("at", $crate::spec::policy::Val::Str(at.into()));
            self
        }
    };
    (optional) => {
        /// Whether the entry may stay unbound: an input with `blank`
        /// allowed, an action with `drop` allowed.
        pub fn optional(mut self, optional: bool) -> Self {
            self.0
                .set("optional", $crate::spec::policy::Val::Bool(optional));
            self
        }
    };
    (max_age) => {
        /// The oldest sample the policy accepts at a start (`max_age_ms`,
        /// at most 60 s).
        pub fn max_age(mut self, max_age: core::time::Duration) -> Self {
            let ms = max_age.as_nanos() as f64 / 1e6;
            self.0.set("max_age_ms", $crate::spec::policy::Val::Num(ms));
            self
        }
    };
    (allow) => {
        /// Conversions this entry may use, beside the policy's own.
        pub fn allow(
            mut self,
            allow: impl IntoIterator<Item = $crate::spec::policy::Allow>,
        ) -> Self {
            self.0
                .extend("allow", allow.into_iter().map(|a| a.name().into()));
            self
        }
    };
    (history) => {
        /// A window of `history` samples (1 to 1024; 1, the default, has no
        /// time axis).
        pub fn history(mut self, history: u32) -> Self {
            self.0
                .set("history", $crate::spec::policy::Val::Int(history));
            self
        }
    };
    (history_hz) => {
        /// The rate the window was trained at, in Hz; needs a `history` of
        /// 2 or more.
        pub fn history_hz(mut self, hz: impl Into<f64>) -> Self {
            self.0
                .set("history_hz", $crate::spec::policy::Val::Num(hz.into()));
            self
        }
    };
    (source) => {
        /// [`Source::Applied`](crate::spec::policy::Source::Applied) reads
        /// what the part's edge dispatched instead of what the robot
        /// published.
        pub fn source(mut self, source: $crate::spec::policy::Source) -> Self {
            self.0.set(
                "source",
                $crate::spec::policy::Val::Str(source.name().into()),
            );
            self
        }
    };
    (size) => {
        /// The image size the model takes: one side for a square (`224`) or
        /// `(height, width)`. Resizing needs [`Allow::Resize`], and another
        /// aspect also `Crop`, `Pad` or `Stretch`.
        ///
        /// [`Allow::Resize`]: crate::spec::policy::Allow::Resize
        pub fn size(mut self, size: impl Into<$crate::spec::inputs::ImageSize>) -> Self {
            let size = size.into();
            self.0.set(
                "size",
                $crate::spec::policy::Val::Dims(alloc::vec![size.height, size.width]),
            );
            self
        }
    };
    (layout) => {
        /// The image layout (default `Hwc`).
        pub fn layout(mut self, layout: $crate::spec::policy::ImageLayout) -> Self {
            self.0.set(
                "layout",
                $crate::spec::policy::Val::Str(layout.name().into()),
            );
            self
        }
    };
    (format) => {
        /// The pixel format (default `Rgb8`).
        pub fn format(mut self, format: $crate::spec::policy::ImageFormat) -> Self {
            self.0.set(
                "format",
                $crate::spec::policy::Val::Str(format.name().into()),
            );
            self
        }
    };
    (resample) => {
        /// The resize kernel (default `Bilinear`).
        pub fn resample(mut self, resample: $crate::spec::policy::Resample) -> Self {
            self.0.set(
                "resample",
                $crate::spec::policy::Val::Str(resample.name().into()),
            );
            self
        }
    };
    (crop) => {
        /// The side fraction, in (0, 1], of the center crop the model was
        /// trained with.
        pub fn crop(mut self, crop: f64) -> Self {
            self.0.set("crop", $crate::spec::policy::Val::Num(crop));
            self
        }
    };
    (upside_down) => {
        /// Whether the model was trained on frames rotated 180°.
        pub fn upside_down(mut self, upside_down: bool) -> Self {
            self.0
                .set("upside_down", $crate::spec::policy::Val::Bool(upside_down));
            self
        }
    };
    (dtype) => {
        /// The element type the model takes; a lossy cast needs
        /// [`Allow::Dtype`](crate::spec::policy::Allow::Dtype).
        pub fn dtype(mut self, dtype: $crate::wire::part::DType) -> Self {
            self.0
                .set("dtype", $crate::spec::policy::Val::Str(dtype.name().into()));
            self
        }
    };
    (range) => {
        /// The model's range, `low` to `high`: the range
        /// [`Allow::Rescale`](crate::spec::policy::Allow::Rescale) maps, or
        /// an image's normalisation target.
        pub fn range(mut self, low: impl Into<f64>, high: impl Into<f64>) -> Self {
            self.0.set(
                "range",
                $crate::spec::policy::Val::Pair(low.into(), high.into()),
            );
            self
        }
    };
    (frame) => {
        /// The frame the model needs; a part declared in another refuses
        /// to bind.
        pub fn frame(mut self, frame: impl Into<String>) -> Self {
            self.0
                .set("frame", $crate::spec::policy::Val::Str(frame.into()));
            self
        }
    };
    (child) => {
        /// The child frame whose pose this is.
        pub fn child(mut self, child: impl Into<String>) -> Self {
            self.0
                .set("child", $crate::spec::policy::Val::Str(child.into()));
            self
        }
    };
    (point) => {
        /// The reference point the model needs the wrench at.
        pub fn point(mut self, point: impl Into<String>) -> Self {
            self.0
                .set("point", $crate::spec::policy::Val::Str(point.into()));
            self
        }
    };
    (names) => {
        /// The model's joints, in its order; a subset of the robot's needs
        /// [`Allow::Select`](crate::spec::policy::Allow::Select).
        pub fn names<N: Into<String>>(mut self, names: impl IntoIterator<Item = N>) -> Self {
            let names = names.into_iter().map(Into::into).collect();
            self.0.set("names", $crate::spec::policy::Val::Names(names));
            self
        }
    };
    (positional) => {
        /// The robot's first `count` joints, by index (`names:
        /// "positional"`).
        pub fn positional(mut self, count: u32) -> Self {
            self.0
                .set("names", $crate::spec::policy::Val::Str("positional".into()));
            self.0.set("count", $crate::spec::policy::Val::Int(count));
            self
        }
    };
    (count) => {
        /// The number of joints the model expects.
        pub fn count(mut self, count: u32) -> Self {
            self.0.set("count", $crate::spec::policy::Val::Int(count));
            self
        }
    };
    (fields $field:ty) => {
        /// The fields to deliver, in this order (default: all the part
        /// publishes).
        pub fn fields(mut self, fields: impl IntoIterator<Item = $field>) -> Self {
            let fields = fields.into_iter().map(|f| f.name().into()).collect();
            self.0
                .set("fields", $crate::spec::policy::Val::Names(fields));
            self
        }
    };
    (field) => {
        /// Adds the field `name` of the custom kind's schema, as the part
        /// declares it; fields keep the order added.
        pub fn field(
            mut self,
            name: impl Into<String>,
            array: $crate::spec::policy::Array,
        ) -> Self {
            self.0.field(name.into(), array);
            self
        }
    };
    (unit) => {
        /// The unit the model takes `field` in (`units`), such as `("pos",
        /// "deg")`.
        pub fn unit(mut self, field: impl Into<String>, unit: impl Into<String>) -> Self {
            self.0.put("units", field.into(), unit.into());
            self
        }
    };
    (scale) => {
        /// An affine scale the model applies: one number, or one per
        /// element.
        pub fn scale(mut self, scale: impl Into<$crate::spec::policy::PerElement>) -> Self {
            self.0
                .set("scale", $crate::spec::policy::Val::Numbers(scale.into()));
            self
        }
    };
    (offset) => {
        /// An affine offset the model applies: one number, or one per
        /// element.
        pub fn offset(mut self, offset: impl Into<$crate::spec::policy::PerElement>) -> Self {
            self.0
                .set("offset", $crate::spec::policy::Val::Numbers(offset.into()));
            self
        }
    };
    (rotation) => {
        /// The rotation encoding the model uses; another than the part's
        /// needs [`Allow::Rotation`](crate::spec::policy::Allow::Rotation)
        /// unless it only reorders a quaternion.
        pub fn rotation(mut self, rotation: $crate::spec::Rotation) -> Self {
            self.0.set(
                "rotation",
                $crate::spec::policy::Val::Str(rotation.name().into()),
            );
            self
        }
    };
    (shape) => {
        /// The shape the model expects.
        pub fn shape(mut self, shape: impl IntoIterator<Item = u32>) -> Self {
            let dims = shape.into_iter().collect();
            self.0.set("shape", $crate::spec::policy::Val::Dims(dims));
            self
        }
    };
    (command $mode:ty) => {
        /// The part's command mode; the binding refuses a part in another.
        pub fn command(mut self, command: $mode) -> Self {
            self.0.set(
                "command",
                $crate::spec::policy::Val::Str(command.name().into()),
            );
            self
        }
    };
    (custom_command) => {
        /// The part's command mode, `[a-z][a-z0-9_]*`; the binding refuses
        /// a part in another.
        pub fn command(mut self, command: impl Into<String>) -> Self {
            self.0
                .set("command", $crate::spec::policy::Val::Str(command.into()));
            self
        }
    };
    (reference) => {
        /// What a delta is relative to; must be the part's.
        pub fn reference(mut self, reference: $crate::spec::actions::Reference) -> Self {
            self.0.set(
                "reference",
                $crate::spec::policy::Val::Str(reference.name().into()),
            );
            self
        }
    };
    (hz) => {
        /// The rate of the actions in a chunk, in Hz (default: the
        /// policy's).
        pub fn hz(mut self, hz: impl Into<f64>) -> Self {
            self.0.set("hz", $crate::spec::policy::Val::Num(hz.into()));
            self
        }
    };
    (chunk) => {
        /// Actions per chunk (default 1).
        pub fn chunk(mut self, chunk: u32) -> Self {
            self.0.set("chunk", $crate::spec::policy::Val::Int(chunk));
            self
        }
    };
    (t0) => {
        /// When a chunk's first action is due: at the observation it was
        /// computed from (the default) or when it is sent.
        pub fn t0(mut self, t0: $crate::spec::policy::ChunkStart) -> Self {
            self.0
                .set("t0", $crate::spec::policy::Val::Str(t0.name().into()));
            self
        }
    };
    (ttl) => {
        /// A cap on a chunk's life (`ttl_ms`, at most 60 s).
        pub fn ttl(mut self, ttl: core::time::Duration) -> Self {
            let ms = ttl.as_nanos() as f64 / 1e6;
            self.0.set("ttl_ms", $crate::spec::policy::Val::Num(ms));
            self
        }
    };
    (open) => {
        /// The model's value for fully open (default 1).
        pub fn open(mut self, open: impl Into<f64>) -> Self {
            self.0
                .set("open", $crate::spec::policy::Val::Num(open.into()));
            self
        }
    };
    (closed) => {
        /// The model's value for fully closed (default 0).
        pub fn closed(mut self, closed: impl Into<f64>) -> Self {
            self.0
                .set("closed", $crate::spec::policy::Val::Num(closed.into()));
            self
        }
    };
}
pub(crate) use setter;

/// One builder type: its setters, its keys, and its conversions into a
/// policy entry and to JSON.
macro_rules! builder {
    (
        $(#[$doc:meta])*
        $ty:ident => $entry:ident [$($key:ident $(($arg:ty))?),* $(,)?]
    ) => {
        $(#[$doc])*
        #[derive(Clone, Debug, PartialEq)]
        #[must_use = "a builder does nothing until it is added to a policy"]
        pub struct $ty($crate::spec::policy::Draft);

        impl $ty {
            /// The JSON keys its entries take.
            #[allow(dead_code)]
            pub(crate) const KEYS: &'static [&'static str] =
                &["at", $($crate::spec::policy::key!($key)),*];

            $crate::spec::policy::setter!(at);
            $($crate::spec::policy::setter!($key $($arg)?);)*
        }

        impl From<$ty> for $crate::spec::policy::$entry {
            fn from(builder: $ty) -> Self {
                Self::built(builder.0)
            }
        }

        /// The entry as the policy map has it.
        impl serde::Serialize for $ty {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                serde::Serialize::serialize(&self.0, s)
            }
        }
    };
}
pub(crate) use builder;

/// A closed set of names: an enum with its names in the policy map.
macro_rules! named {
    (
        $(#[$doc:meta])*
        $ty:ident { $($(#[$vdoc:meta])* $variant:ident = $name:literal),* $(,)? }
    ) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $ty {
            $($(#[$vdoc])* $variant),*
        }

        impl $ty {
            /// Every value.
            pub const ALL: &'static [Self] = &[$(Self::$variant),*];

            /// Its name in the policy map.
            pub fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name),*
                }
            }
        }

        impl core::fmt::Display for $ty {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(self.name())
            }
        }
    };
}
pub(crate) use named;
