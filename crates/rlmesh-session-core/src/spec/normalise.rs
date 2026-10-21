//! The shared normaliser (§3): an embodiment's authored robot map (JSON) to
//! [`PartSpec`]s, with the §3.4 validation, the kind table's defaults
//! expanded, and the declaration fingerprint.
//!
//! The participant sends only the authored map; the host normalises it, so
//! every peer sees one canonical declaration. Channel ids are left at zero
//! here; the host's authority assigns them.
//!
//! ```json
//! {
//!   "arm":       {"kind": "joints", "names": ["shoulder_pan_joint", "..."],
//!                 "fields": ["pos", "vel", "effort"], "hz": 500,
//!                 "command": "position", "safe": "stop",
//!                 "limits": {"pos": [-6.283, 6.283]}},
//!   "wrist/cam": {"kind": "camera", "width": 640, "height": 480, "hz": 30,
//!                 "encoding": "jpeg"},
//!   "feet/force": {"kind": "unitree/foot_force", "hz": 500,
//!                  "fields": {"force": ["int16", [4]]}, "units": {"force": "raw"}},
//!   "task":      {"kind": "text"}
//! }
//! ```
//!
//! A core kind's fields come from the kind table ([`crate::spec::kinds`]); its
//! `fields` key, where it has one (joints, hand, imu), lists which optional
//! fields it publishes, and they are packed in the table's order. A custom
//! kind declares its fields as an object of `name: [dtype, shape]`, in
//! payload order (an order-keeping parse; duplicate keys are refused).
//!
//! # Canonical form and fingerprint (§3.3)
//!
//! [`canonical()`] is the normalised map (defaults expanded, host-assigned
//! fields omitted, units spelled per §5.1) as JSON: an array of parts in
//! authored order, each part's fields an array in payload order. RFC 8785
//! sorts object members, so nothing whose order matters is an object.
//! [`fingerprint`] is BLAKE3 over its RFC 8785 bytes, first 8 bytes.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use crate::wire::part::{
    ChannelPriority, CommandSpec, DType, Encoding, FieldLimit, FieldSpec, Layout, MAX_PARTS,
    MergePolicy, PartSpec, SMALL_SAMPLE_BYTES, SendPolicy,
};
use crate::wire::path::is_identifier;
use crate::wire::version::SPEC_VERSION;
use crate::wire::{ChannelId, CtlError, CtlStatus, Fingerprint, PartPath, ParticipantId};
use serde::Deserialize;
use serde::de::value::MapAccessDeserializer;
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde_json::{Value, json};

use crate::spec::canonical;
use crate::spec::kinds::{Kind, joint_quantity, quantity};
use crate::spec::rotation::Rotation;
use crate::spec::units::{NORMALIZED, Quantity};

/// The longest frame, mode, behaviour or group id.
const MAX_NAME_LEN: usize = 64;
/// The longest joint name.
const MAX_ELEMENT_NAME_LEN: usize = 128;
/// The most joints in one part.
const MAX_JOINTS: u32 = 1024;
/// The highest declared rate.
const MAX_HZ: f64 = 10_000.0;
/// The longest declared duration, in ms.
const MAX_MS: f64 = 60_000.0;
/// The longest `doc` or `layout` text.
const MAX_DOC_LEN: usize = 4096;

/// Attribute keys the normaliser writes into [`PartSpec::attrs`]; the binder
/// reads them.
pub mod attrs {
    /// Pose: the child frame.
    pub const CHILD: &str = "child";
    /// Wrench: the frame the torque is taken about, when not `frame`.
    pub const POINT: &str = "point";
    /// Gripper: `[low, high]` physical span in metres (JSON).
    pub const WIDTH_M: &str = "width_m";
    /// Camera: `upside_down` when mounted rotated 180°.
    pub const MOUNT: &str = "mount";
    /// Per-field measurement ranges: `{"field": [low, high]}` (JSON).
    pub const RANGE: &str = "range";
    /// Camera, depth: the field of view in degrees.
    pub const FOV_DEG: &str = "fov_deg";
    /// Camera, depth: `[fx, fy, cx, cy]` (JSON).
    pub const INTRINSICS: &str = "intrinsics";
    /// Joints, hand: controller gains, informational (JSON).
    pub const GAINS: &str = "gains";
    /// Tactile: what the taxels are.
    pub const LAYOUT: &str = "layout";
    /// Free text for the cockpit.
    pub const DOC: &str = "doc";
    /// Gym compatibility (§12.4): the original Gym space of the part's
    /// leaves (JSON).
    pub const GYM_SPACE: &str = "gym_space";
    /// Gym compatibility: the v1 EnvTags tag of the part's leaves (JSON).
    pub const ENV_TAGS: &str = "env_tags";
    /// Gym compatibility: the original Dict/Tuple keys of each leaf (JSON).
    pub const GYM_KEY: &str = "gym_key";
    /// Gym compatibility: the whole v1 env contract and handshake (JSON), on
    /// the robot's first part.
    pub const GYM_ENV: &str = "gym_env";
    /// The pass-through keys any part may carry (beyond `x_*`), recorded
    /// verbatim as canonical JSON.
    pub const PASS_THROUGH: [&str; 4] = [GYM_SPACE, ENV_TAGS, GYM_KEY, GYM_ENV];
}

/// A JSON object kept in authored order, refusing duplicate keys.
pub(crate) struct Ordered<T>(pub(crate) Vec<(String, T)>);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Ordered<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct OrderedVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for OrderedVisitor<T> {
            type Value = Ordered<T>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries: Vec<(String, T)> = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, T>()? {
                    if entries.iter().any(|(k, _)| *k == key) {
                        return Err(serde::de::Error::custom(format!("duplicate key {key:?}")));
                    }
                    entries.push((key, value));
                }
                Ok(Ordered(entries))
            }
        }

        deserializer.deserialize_map(OrderedVisitor(std::marker::PhantomData))
    }
}

/// `[dtype, shape]`: one field of a custom kind.
#[derive(Deserialize)]
pub(crate) struct FieldDecl(pub(crate) String, pub(crate) Vec<u32>);

/// A part's `fields`: a list of optional field names (joints, hand, imu),
/// or a custom kind's schema `{name: [dtype, shape]}` in payload order.
pub(crate) enum FieldsDecl {
    /// Core kinds: which optional fields are published.
    Names(Vec<String>),
    /// Custom kinds: the schema.
    Schema(Ordered<FieldDecl>),
}

impl<'de> Deserialize<'de> for FieldsDecl {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FieldsVisitor;

        impl<'de> Visitor<'de> for FieldsVisitor {
            type Value = FieldsDecl;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a list of field names, or {name: [dtype, shape]}")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut names = Vec::new();
                while let Some(name) = seq.next_element::<String>()? {
                    names.push(name);
                }
                Ok(FieldsDecl::Names(names))
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                Ordered::deserialize(MapAccessDeserializer::new(map)).map(FieldsDecl::Schema)
            }
        }

        deserializer.deserialize_any(FieldsVisitor)
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum LimitDecl {
    Pair([f64; 2]),
    Pairs(Vec<[f64; 2]>),
    Norm { max_norm: f64 },
}

#[derive(Deserialize)]
struct PartDecl {
    kind: String,
    hz: Option<f64>,
    send: Option<String>,
    frame: Option<String>,
    dtype: Option<String>,
    units: Option<Ordered<String>>,
    range: Option<Ordered<[f64; 2]>>,
    max_age_ms: Option<f64>,
    fields: Option<FieldsDecl>,
    priority: Option<String>,
    doc: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    encoding: Option<String>,
    quality: Option<u32>,
    format: Option<String>,
    fov_deg: Option<f64>,
    intrinsics: Option<[f64; 4]>,
    mount: Option<String>,
    names: Option<Vec<String>>,
    count: Option<u32>,
    prismatic: Option<bool>,
    gains: Option<BTreeMap<String, f64>>,
    point: Option<String>,
    rotation: Option<String>,
    child: Option<String>,
    width_m: Option<[f64; 2]>,
    binary: Option<bool>,
    dof: Option<String>,
    shape: Option<Vec<u32>>,
    layout: Option<String>,
    command: Option<String>,
    command_fields: Option<FieldsDecl>,
    control_hz: Option<f64>,
    safe: Option<String>,
    hold_max_ms: Option<f64>,
    max_action_age_ms: Option<f64>,
    limits: Option<BTreeMap<String, LimitDecl>>,
    merge: Option<String>,
    group: Option<String>,
    reference: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

/// Keys every streaming kind takes.
const STREAMING: &[&str] = &["kind", "hz", "send", "priority", "doc", "max_age_ms"];
/// Keys of a commandable part (§3.2), less the kind-specific ones.
const COMMAND: &[&str] = &[
    "command",
    "control_hz",
    "safe",
    "hold_max_ms",
    "max_action_age_ms",
    "limits",
    "merge",
    "group",
];

impl PartDecl {
    /// The keys this declaration sets.
    fn present(&self) -> Vec<&'static str> {
        let keys: [(&'static str, bool); 41] = [
            ("kind", true),
            ("hz", self.hz.is_some()),
            ("send", self.send.is_some()),
            ("frame", self.frame.is_some()),
            ("dtype", self.dtype.is_some()),
            ("units", self.units.is_some()),
            ("range", self.range.is_some()),
            ("max_age_ms", self.max_age_ms.is_some()),
            ("fields", self.fields.is_some()),
            ("priority", self.priority.is_some()),
            ("doc", self.doc.is_some()),
            ("width", self.width.is_some()),
            ("height", self.height.is_some()),
            ("encoding", self.encoding.is_some()),
            ("quality", self.quality.is_some()),
            ("format", self.format.is_some()),
            ("fov_deg", self.fov_deg.is_some()),
            ("intrinsics", self.intrinsics.is_some()),
            ("mount", self.mount.is_some()),
            ("names", self.names.is_some()),
            ("count", self.count.is_some()),
            ("prismatic", self.prismatic.is_some()),
            ("gains", self.gains.is_some()),
            ("point", self.point.is_some()),
            ("rotation", self.rotation.is_some()),
            ("child", self.child.is_some()),
            ("width_m", self.width_m.is_some()),
            ("binary", self.binary.is_some()),
            ("dof", self.dof.is_some()),
            ("shape", self.shape.is_some()),
            ("layout", self.layout.is_some()),
            ("command", self.command.is_some()),
            ("command_fields", self.command_fields.is_some()),
            ("control_hz", self.control_hz.is_some()),
            ("safe", self.safe.is_some()),
            ("hold_max_ms", self.hold_max_ms.is_some()),
            ("max_action_age_ms", self.max_action_age_ms.is_some()),
            ("limits", self.limits.is_some()),
            ("merge", self.merge.is_some()),
            ("group", self.group.is_some()),
            ("reference", self.reference.is_some()),
        ];
        keys.into_iter()
            .filter(|(_, set)| *set)
            .map(|(k, _)| k)
            .collect()
    }
}

/// The keys `kind` takes (beyond `x_*`).
fn allowed(kind: Kind) -> Vec<&'static str> {
    let mut keys: Vec<&'static str> = match kind {
        Kind::Text | Kind::Reward | Kind::Success => return vec!["kind", "priority", "doc"],
        _ => STREAMING.to_vec(),
    };
    let commandable = !kind.command_modes().is_empty() || kind == Kind::Custom;
    if commandable {
        keys.extend(COMMAND);
    }
    keys.extend(match kind {
        Kind::Camera => &[
            "frame",
            "width",
            "height",
            "encoding",
            "quality",
            "format",
            "fov_deg",
            "intrinsics",
            "mount",
        ][..],
        Kind::Depth => &[
            "frame",
            "width",
            "height",
            "dtype",
            "range",
            "fov_deg",
            "intrinsics",
        ],
        Kind::Joints | Kind::Hand => &[
            "dtype",
            "units",
            "range",
            "fields",
            "names",
            "count",
            "prismatic",
            "gains",
        ],
        Kind::Wrench => &["frame", "point", "dtype", "units", "range"],
        Kind::Imu => &["frame", "dtype", "units", "range", "fields", "rotation"],
        Kind::Pose => &[
            "frame",
            "child",
            "rotation",
            "dtype",
            "units",
            "range",
            "reference",
        ],
        Kind::Gripper => &["dtype", "units", "range", "width_m", "binary"],
        Kind::Velocity => &["frame", "dof", "dtype", "units", "range"],
        Kind::Tactile => &["shape", "layout", "dtype", "units", "range"],
        Kind::Custom => &["frame", "units", "range", "fields", "command_fields"],
        Kind::Text | Kind::Reward | Kind::Success => &[],
    });
    keys
}

fn invalid(path: &str, message: impl fmt::Display) -> CtlError {
    CtlError::new(CtlStatus::InvalidRequest, format!("{path}: {message}"))
}

/// Validates a rate in Hz.
fn rate(path: &str, key: &str, hz: f64) -> Result<f64, CtlError> {
    if hz.is_finite() && hz > 0.0 && hz <= MAX_HZ {
        Ok(hz)
    } else {
        Err(invalid(path, format!("{key} {hz} is not in (0, 10000]")))
    }
}

/// Validates a duration in ms; `zero_ok` allows 0.
fn millis(path: &str, key: &str, ms: f64, zero_ok: bool) -> Result<Duration, CtlError> {
    let ok = ms.is_finite() && ms <= MAX_MS && (ms > 0.0 || (zero_ok && ms == 0.0));
    if ok {
        Ok(Duration::from_secs_f64(ms / 1e3))
    } else {
        Err(invalid(path, format!("{key} {ms} is out of range")))
    }
}

fn name(path: &str, key: &str, value: &str) -> Result<String, CtlError> {
    if is_identifier(value, MAX_NAME_LEN) {
        Ok(value.to_string())
    } else {
        Err(invalid(
            path,
            format!("{key} {value:?} must be [a-z][a-z0-9_]*"),
        ))
    }
}

fn pair(path: &str, key: &str, [low, high]: [f64; 2]) -> Result<(), CtlError> {
    if low.is_finite() && high.is_finite() && low < high {
        Ok(())
    } else {
        Err(invalid(path, format!("{key}: finite, low < high")))
    }
}

/// A custom kind's field schema, in payload order.
fn schema(path: &str, decl: Ordered<FieldDecl>) -> Result<Vec<FieldSpec>, CtlError> {
    decl.0
        .into_iter()
        .map(|(field, FieldDecl(dtype, shape))| {
            let field = name(path, "field", &field)?;
            let dtype = DType::from_name(&dtype)
                .ok_or_else(|| invalid(path, format!("field {field:?} dtype {dtype:?}")))?;
            Ok(FieldSpec::new(field, dtype, &shape))
        })
        .collect()
}

/// The float dtype a part declares (`float32` by default).
fn float_dtype(path: &str, dtype: Option<&str>) -> Result<DType, CtlError> {
    match dtype {
        None | Some("float32") => Ok(DType::F32),
        Some("float64") => Ok(DType::F64),
        Some(other) => Err(invalid(
            path,
            format!("dtype {other:?}: float32 or float64"),
        )),
    }
}

/// Which optional fields a joint set or IMU publishes, in table order.
fn optional_fields(
    path: &str,
    kind: Kind,
    decl: Option<FieldsDecl>,
    default: &[&'static str],
) -> Result<Vec<&'static str>, CtlError> {
    let table = kind.optional_fields();
    let listed = match decl {
        None => return Ok(default.to_vec()),
        Some(FieldsDecl::Names(names)) => names,
        Some(FieldsDecl::Schema(_)) => {
            return Err(invalid(
                path,
                format!("{kind} fields is a list of {}", table.join(", ")),
            ));
        }
    };
    if listed.is_empty() {
        return Err(invalid(path, "fields is empty"));
    }
    for (i, field) in listed.iter().enumerate() {
        if !table.contains(&field.as_str()) || listed[..i].contains(field) {
            return Err(invalid(
                path,
                format!(
                    "fields: {field:?} is not one of {} or is repeated",
                    table.join(", ")
                ),
            ));
        }
    }
    Ok(table
        .iter()
        .copied()
        .filter(|f| listed.iter().any(|l| l == f))
        .collect())
}

/// A field with its SI (or declared) unit.
fn field(name: &str, dtype: DType, shape: &[u32], unit: &str) -> FieldSpec {
    let mut spec = FieldSpec::new(name, dtype, shape);
    spec.unit = unit.to_string();
    spec
}

fn rotation_field(name: &str, dtype: DType, rotation: Rotation) -> FieldSpec {
    let mut spec = FieldSpec::new(name, dtype, &[rotation.width()]);
    spec.rotation = rotation.name().to_string();
    spec
}

/// Applies a part's declared `units` to its fields (published and command),
/// checking each against the field's quantity.
fn apply_units(
    path: &str,
    kind: Kind,
    units: &[(String, String)],
    fields: &mut [&mut FieldSpec],
    quantities: &dyn Fn(&str) -> Quantity,
    width_m: bool,
) -> Result<(), CtlError> {
    for (key, unit) in units {
        let targets: Vec<&mut &mut FieldSpec> =
            fields.iter_mut().filter(|f| &f.name == key).collect();
        if targets.is_empty() {
            return Err(invalid(path, format!("units on unknown field {key:?}")));
        }
        let quantity = quantities(key);
        if !quantity.accepts(unit) {
            return Err(invalid(
                path,
                format!(
                    "units {unit:?} on {key:?} ({quantity}): one of {}",
                    match quantity.alternatives() {
                        alts if alts.is_empty() => "no unit".to_string(),
                        alts => alts.join(", "),
                    }
                ),
            ));
        }
        if kind == Kind::Gripper && unit == "m" && !width_m {
            return Err(invalid(
                path,
                "a gripper publishes metres only with width_m",
            ));
        }
        for target in targets {
            target.unit.clone_from(unit);
        }
    }
    Ok(())
}

fn limits(
    path: &str,
    delta: bool,
    fields: &[FieldSpec],
    decl: BTreeMap<String, LimitDecl>,
) -> Result<Vec<FieldLimit>, CtlError> {
    decl.into_iter()
        .map(|(field, limit)| {
            let spec = fields.iter().find(|f| f.name == field).ok_or_else(|| {
                invalid(path, format!("limits on {field:?}, not a command field"))
            })?;
            if !spec.dtype.is_float() || spec.dtype == DType::F16 {
                return Err(invalid(
                    path,
                    format!("limits on {field:?}: only on float32 and float64 fields"),
                ));
            }
            let elements = spec.elements().unwrap_or(0);
            let pairs = match limit {
                LimitDecl::Norm { max_norm } => {
                    if !delta || spec.shape.len() != 1 {
                        return Err(invalid(
                            path,
                            format!("max_norm on {field:?}: only on a delta vector field"),
                        ));
                    }
                    if !(max_norm.is_finite() && max_norm > 0.0) {
                        return Err(invalid(path, format!("max_norm {max_norm} on {field:?}")));
                    }
                    return Ok(FieldLimit {
                        field,
                        low: Vec::new(),
                        high: Vec::new(),
                        max_norm: Some(max_norm),
                    });
                }
                LimitDecl::Pair(pair) => vec![pair],
                LimitDecl::Pairs(pairs) => {
                    if pairs.len() as u64 != elements {
                        return Err(invalid(
                            path,
                            format!(
                                "limits on {field:?}: {} pairs for {elements} elements",
                                pairs.len()
                            ),
                        ));
                    }
                    pairs
                }
            };
            if pairs
                .iter()
                .any(|[low, high]| !(low.is_finite() && high.is_finite() && low <= high))
            {
                return Err(invalid(
                    path,
                    format!("limits on {field:?}: finite, low <= high"),
                ));
            }
            Ok(FieldLimit {
                field,
                low: pairs.iter().map(|p| p[0]).collect(),
                high: pairs.iter().map(|p| p[1]).collect(),
                max_norm: None,
            })
        })
        .collect()
}

/// The fields a part publishes and, for a commandable part, the fields one
/// action carries.
struct Lowered {
    fields: Vec<FieldSpec>,
    command_fields: Option<Vec<FieldSpec>>,
    names: Vec<String>,
    encoding: Encoding,
}

/// Expands a part's fields from the kind table.
fn lower(path: &str, kind: Kind, decl: &mut PartDecl) -> Result<Lowered, CtlError> {
    let mode = decl.command.clone();
    let mut names = Vec::new();
    let mut encoding = Encoding::Raw;
    let (fields, command_fields) = match kind {
        Kind::Camera => {
            let (Some(width), Some(height)) = (decl.width, decl.height) else {
                return Err(invalid(path, "a camera declares width and height"));
            };
            let channels = match decl.format.as_deref() {
                None | Some("rgb8") => 3,
                Some("mono8") => 1,
                Some(other) => {
                    return Err(invalid(path, format!("format {other:?}: rgb8 or mono8")));
                }
            };
            encoding = match (decl.encoding.as_deref(), decl.quality) {
                (None | Some("raw"), None) => Encoding::Raw,
                (Some("jpeg"), quality) => {
                    if channels != 3 {
                        return Err(invalid(path, "jpeg is only for rgb8 cameras"));
                    }
                    let quality = quality.unwrap_or(80);
                    Encoding::Jpeg {
                        quality: u8::try_from(quality)
                            .ok()
                            .filter(|q| (1..=100).contains(q))
                            .ok_or_else(|| invalid(path, format!("quality {quality}")))?,
                    }
                }
                (None | Some("raw"), Some(_)) => return Err(invalid(path, "quality without jpeg")),
                (Some(other), _) => return Err(invalid(path, format!("encoding {other:?}"))),
            };
            (
                vec![field("image", DType::U8, &[height, width, channels], "")],
                None,
            )
        }
        Kind::Depth => {
            let (Some(width), Some(height)) = (decl.width, decl.height) else {
                return Err(invalid(path, "a depth camera declares width and height"));
            };
            let (dtype, unit) = match decl.dtype.as_deref() {
                None | Some("float32") => (DType::F32, "m"),
                Some("float64") => (DType::F64, "m"),
                Some("uint16") => (DType::U16, "mm"),
                Some(other) => {
                    return Err(invalid(
                        path,
                        format!("dtype {other:?}: float32, float64 or uint16 (mm)"),
                    ));
                }
            };
            (vec![field("depth", dtype, &[height, width], unit)], None)
        }
        Kind::Joints | Kind::Hand => {
            let n = match (decl.names.take(), decl.count) {
                (Some(list), count) => {
                    for (i, n) in list.iter().enumerate() {
                        if n.is_empty() || n.len() > MAX_ELEMENT_NAME_LEN || list[..i].contains(n) {
                            return Err(invalid(
                                path,
                                format!("names: {n:?} is empty, longer than 128 or repeated"),
                            ));
                        }
                    }
                    let n = u32::try_from(list.len()).unwrap_or(u32::MAX);
                    if count.is_some_and(|c| c != n) {
                        return Err(invalid(path, format!("count differs from the {n} names")));
                    }
                    names = list;
                    n
                }
                (None, Some(_)) if kind == Kind::Hand => {
                    return Err(invalid(path, "a hand declares its joint names"));
                }
                (None, Some(count)) => count,
                (None, None) => return Err(invalid(path, "declare names (or count)")),
            };
            if !(1..=MAX_JOINTS).contains(&n) {
                return Err(invalid(path, format!("{n} joints (1 to {MAX_JOINTS})")));
            }
            let dtype = float_dtype(path, decl.dtype.as_deref())?;
            let prismatic = decl.prismatic.unwrap_or(false);
            let joint = |f: &str| field(f, dtype, &[n], joint_quantity(f, prismatic).si());
            let published = optional_fields(path, kind, decl.fields.take(), &["pos", "vel"])?;
            let command = match mode.as_deref() {
                None => None,
                Some("position") => Some(vec![joint("pos")]),
                Some("velocity") => Some(vec![joint("vel")]),
                Some("torque") => Some(vec![joint("effort")]),
                Some(_) => None,
            };
            (published.into_iter().map(joint).collect(), command)
        }
        Kind::Wrench => {
            let dtype = float_dtype(path, decl.dtype.as_deref())?;
            let fields = vec![
                field("force", dtype, &[3], "N"),
                field("torque", dtype, &[3], "N*m"),
            ];
            let command = (mode.as_deref() == Some("wrench")).then(|| fields.clone());
            (fields, command)
        }
        Kind::Imu => {
            let dtype = float_dtype(path, decl.dtype.as_deref())?;
            if decl.fields.is_none() {
                return Err(invalid(
                    path,
                    "an imu lists its fields (many IMUs have no orientation estimate)",
                ));
            }
            let published = optional_fields(path, kind, decl.fields.take(), &[])?;
            let rotation = source_rotation(path, decl.rotation.as_deref())?;
            if decl.rotation.is_some() && !published.contains(&"orientation") {
                return Err(invalid(path, "rotation without the orientation field"));
            }
            let fields = published
                .into_iter()
                .map(|f| match f {
                    "orientation" => rotation_field(f, dtype, rotation),
                    "ang_vel" => field(f, dtype, &[3], "rad/s"),
                    "lin_acc" => field(f, dtype, &[3], "m/s^2"),
                    _ => field(f, dtype, &[3], ""),
                })
                .collect();
            (fields, None)
        }
        Kind::Pose => {
            let dtype = float_dtype(path, decl.dtype.as_deref())?;
            let rotation = source_rotation(path, decl.rotation.as_deref())?;
            let fields = vec![
                field("pos", dtype, &[3], "m"),
                rotation_field("rot", dtype, rotation),
            ];
            let command = match mode.as_deref() {
                Some("absolute") => Some(fields.clone()),
                Some("delta") => Some(vec![
                    field("pos", dtype, &[3], "m"),
                    rotation_field("rot", dtype, Rotation::AxisAngle),
                ]),
                _ => None,
            };
            (fields, command)
        }
        Kind::Gripper => {
            let dtype = float_dtype(path, decl.dtype.as_deref())?;
            let fields = vec![field("pos", dtype, &[1], NORMALIZED)];
            let command = (mode.as_deref() == Some("position")).then(|| fields.clone());
            (fields, command)
        }
        Kind::Velocity => {
            let dtype = float_dtype(path, decl.dtype.as_deref())?;
            let (linear, angular) = match decl.dof.as_deref() {
                None | Some("planar") => (2, 1),
                Some("spatial") => (3, 3),
                Some(other) => {
                    return Err(invalid(path, format!("dof {other:?}: planar or spatial")));
                }
            };
            let fields = vec![
                field("linear", dtype, &[linear], "m/s"),
                field("angular", dtype, &[angular], "rad/s"),
            ];
            let command = (mode.as_deref() == Some("velocity")).then(|| fields.clone());
            (fields, command)
        }
        Kind::Tactile => {
            let shape = decl
                .shape
                .take()
                .ok_or_else(|| invalid(path, "a tactile part declares its shape"))?;
            if shape.is_empty() {
                return Err(invalid(path, "shape has no dimensions"));
            }
            let dtype = float_dtype(path, decl.dtype.as_deref())?;
            (vec![field("values", dtype, &shape, NORMALIZED)], None)
        }
        Kind::Text => (vec![field("text", DType::Utf8, &[], "")], None),
        Kind::Reward => (vec![field("reward", DType::F32, &[], "")], None),
        Kind::Success => (vec![field("success", DType::Bool, &[], "")], None),
        Kind::Custom => {
            let fields = match decl.fields.take() {
                Some(FieldsDecl::Schema(s)) => schema(path, s)?,
                Some(FieldsDecl::Names(_)) => {
                    return Err(invalid(
                        path,
                        "a custom kind declares fields as {name: [dtype, shape]}",
                    ));
                }
                None => return Err(invalid(path, "a custom kind declares its fields")),
            };
            let command = match (mode.as_deref(), decl.command_fields.take()) {
                (None, _) => None,
                (Some(_), None) => Some(fields.clone()),
                (Some(_), Some(FieldsDecl::Schema(s))) => Some(schema(path, s)?),
                (Some(_), Some(FieldsDecl::Names(_))) => {
                    return Err(invalid(path, "command_fields is {name: [dtype, shape]}"));
                }
            };
            (fields, command)
        }
    };
    Ok(Lowered {
        fields,
        command_fields,
        names,
        encoding,
    })
}

/// A source's rotation encoding (`quat_xyzw` by default; never
/// `gravity_xyz`, which is a target only).
fn source_rotation(path: &str, rotation: Option<&str>) -> Result<Rotation, CtlError> {
    match rotation {
        None => Ok(Rotation::QuatXyzw),
        Some(name) => Rotation::from_name(name)
            .filter(|r| r.is_source())
            .ok_or_else(|| invalid(path, format!("rotation {name:?} is not a source encoding"))),
    }
}

fn part(robot: &ParticipantId, path: &str, mut decl: PartDecl) -> Result<PartSpec, CtlError> {
    let part_path = PartPath::new(path).map_err(|err| invalid(path, err.reason))?;
    let kind = Kind::parse(&decl.kind).map_err(|err| invalid(path, err))?;
    let mut attrs = BTreeMap::new();
    for (key, value) in &decl.extra {
        if !key.starts_with("x_") && !attrs::PASS_THROUGH.contains(&key.as_str()) {
            return Err(invalid(path, format!("unknown key {key:?}")));
        }
        attrs.insert(key.clone(), canonical::to_string(value));
    }
    let allowed = allowed(kind);
    if let Some(key) = decl.present().into_iter().find(|k| !allowed.contains(k)) {
        return Err(invalid(
            path,
            format!(
                "{key} does not apply to a {} part",
                if kind == Kind::Custom {
                    decl.kind.as_str()
                } else {
                    kind.name()
                }
            ),
        ));
    }
    if decl.command.is_none()
        && let Some(key) = COMMAND
            .iter()
            .chain(&["reference", "binary", "command_fields"])
            .find(|k| decl.present().contains(k))
    {
        return Err(invalid(path, format!("{key} without command")));
    }
    if let Some(mode) = &decl.command {
        let modes = kind.command_modes();
        if kind != Kind::Custom && !modes.contains(&mode.as_str()) {
            return Err(invalid(
                path,
                format!(
                    "command {mode:?}: a {} part takes {}",
                    kind.name(),
                    modes.join(", ")
                ),
            ));
        }
        name(path, "command", mode)?;
    }
    for (required, set) in [
        ("frame", decl.frame.is_some()),
        ("child", decl.child.is_some()),
    ] {
        let needs = match required {
            "frame" => matches!(kind, Kind::Wrench | Kind::Imu | Kind::Pose | Kind::Velocity),
            _ => kind == Kind::Pose,
        };
        if needs && !set {
            return Err(invalid(
                path,
                format!("a {} part declares its {required}", kind.name()),
            ));
        }
    }
    let event = kind.is_event();
    let hz = match (decl.hz, event) {
        (Some(hz), false) => rate(path, "hz", hz)?,
        (_, true) => 0.0,
        (None, false) => return Err(invalid(path, "hz is required")),
    };

    let Lowered {
        mut fields,
        mut command_fields,
        names,
        encoding,
    } = lower(path, kind, &mut decl)?;

    // Units, per field, published and command fields alike.
    let width_m = decl.width_m.is_some();
    if let Some(units) = decl.units.take() {
        let kind_of = kind;
        let quantity_of = |f: &str| {
            if kind_of.is_joint_set() {
                joint_quantity(f, decl.prismatic.unwrap_or(false))
            } else {
                quantity(kind_of, f, "")
            }
        };
        let mut targets: Vec<&mut FieldSpec> = fields.iter_mut().collect();
        if let Some(command) = command_fields.as_mut() {
            targets.extend(command.iter_mut());
        }
        apply_units(path, kind, &units.0, &mut targets, &quantity_of, width_m)?;
    }
    let layout = Layout::of(&fields).map_err(|err| invalid(path, err))?;

    if let Some(range) = decl.range.take() {
        let mut map = serde_json::Map::new();
        for (key, bounds) in range.0 {
            if !fields.iter().any(|f| f.name == key) {
                return Err(invalid(path, format!("range on unknown field {key:?}")));
            }
            pair(path, &format!("range on {key:?}"), bounds)?;
            map.insert(key, json!(bounds));
        }
        attrs.insert(
            attrs::RANGE.into(),
            canonical::to_string(&Value::Object(map)),
        );
    }
    if let Some(bounds) = decl.width_m {
        pair(path, "width_m", bounds)?;
        attrs.insert(attrs::WIDTH_M.into(), canonical::to_string(&json!(bounds)));
    }
    if let Some(doc) = decl.doc {
        if doc.len() > MAX_DOC_LEN {
            return Err(invalid(path, "doc is longer than 4096 bytes"));
        }
        attrs.insert(attrs::DOC.into(), doc);
    }
    if let Some(layout) = decl.layout {
        if layout.len() > MAX_DOC_LEN {
            return Err(invalid(path, "layout is longer than 4096 bytes"));
        }
        attrs.insert(attrs::LAYOUT.into(), layout);
    }
    if let Some(child) = decl.child {
        attrs.insert(attrs::CHILD.into(), name(path, "child", &child)?);
    }
    if let Some(point) = decl.point {
        attrs.insert(attrs::POINT.into(), name(path, "point", &point)?);
    }
    match decl.mount.as_deref() {
        None | Some("upright") => {}
        Some("upside_down") => {
            attrs.insert(attrs::MOUNT.into(), "upside_down".into());
        }
        Some(other) => {
            return Err(invalid(
                path,
                format!("mount {other:?}: upright or upside_down"),
            ));
        }
    }
    if let Some(fov) = decl.fov_deg {
        if !(fov.is_finite() && fov > 0.0 && fov < 360.0) {
            return Err(invalid(path, format!("fov_deg {fov}")));
        }
        attrs.insert(attrs::FOV_DEG.into(), canonical::number(fov));
    }
    if let Some(intrinsics) = decl.intrinsics {
        if intrinsics.iter().any(|v| !v.is_finite()) {
            return Err(invalid(path, "intrinsics are finite [fx, fy, cx, cy]"));
        }
        attrs.insert(
            attrs::INTRINSICS.into(),
            canonical::to_string(&json!(intrinsics)),
        );
    }
    if let Some(gains) = decl.gains {
        if gains.values().any(|v| !v.is_finite()) {
            return Err(invalid(path, "gains are finite numbers"));
        }
        attrs.insert(attrs::GAINS.into(), canonical::to_string(&json!(gains)));
    }

    let max_age = decl
        .max_age_ms
        .map(|ms| millis(path, "max_age_ms", ms, false))
        .transpose()?;
    let frame = decl
        .frame
        .map(|f| name(path, "frame", &f))
        .transpose()?
        .unwrap_or_default();
    let small = layout
        .stride()
        .is_none_or(|stride| stride <= SMALL_SAMPLE_BYTES);
    let priority = match decl.priority {
        Some(p) => ChannelPriority::from_name(&p)
            .ok_or_else(|| invalid(path, format!("priority {p:?}")))?,
        None => match kind {
            _ if event => ChannelPriority::Data,
            Kind::Camera | Kind::Depth => ChannelPriority::Bulk,
            Kind::Tactile => ChannelPriority::Data,
            Kind::Custom if small => ChannelPriority::Data,
            Kind::Custom => ChannelPriority::Bulk,
            _ => ChannelPriority::Realtime,
        },
    };
    let send = match decl.send {
        Some(send) => {
            SendPolicy::parse(&send).map_err(|err| invalid(path, err))?;
            send
        }
        None => match kind {
            Kind::Camera | Kind::Depth => format!("latest(hz={})", canonical::number(hz)),
            Kind::Custom if !small => format!("latest(hz={})", canonical::number(hz)),
            _ => "every".into(),
        },
    };

    let (command, control_hz) = match (decl.command, command_fields) {
        (Some(mode), Some(command_fields)) => {
            if encoding != Encoding::Raw {
                return Err(invalid(path, "encoded parts take no commands"));
            }
            if matches!(Layout::of(&command_fields), Ok(Layout::Text)) {
                return Err(invalid(path, "text cannot be commanded"));
            }
            Layout::of(&command_fields).map_err(|err| invalid(path, err))?;
            let merge = match decl.merge.as_deref() {
                None | Some("replace") => MergePolicy::Replace,
                Some("blend" | "rtc_prefix") => {
                    return Err(CtlError::new(
                        CtlStatus::Unsupported,
                        format!("{path}: only the replace merge policy is implemented"),
                    ));
                }
                Some(other) => return Err(invalid(path, format!("merge {other:?}"))),
            };
            let delta = mode == "delta";
            let reference = match (decl.reference.as_deref(), delta) {
                (None, true) => "current".to_string(),
                (None, false) => String::new(),
                (Some(r @ ("current" | "target")), true) => r.to_string(),
                (Some(r), _) => {
                    return Err(invalid(
                        path,
                        format!("reference {r:?}: current or target, delta only"),
                    ));
                }
            };
            let limits = limits(
                path,
                delta,
                &command_fields,
                decl.limits.unwrap_or_default(),
            )?;
            let control_hz = rate(path, "control_hz", decl.control_hz.unwrap_or(hz))?;
            (
                Some(CommandSpec {
                    hold_last: kind.holds_target(&mode),
                    mode,
                    fields: command_fields,
                    limits,
                    safe_behavior: name(path, "safe", decl.safe.as_deref().unwrap_or("stop"))?,
                    hold_max: decl
                        .hold_max_ms
                        .map(|ms| millis(path, "hold_max_ms", ms, true))
                        .transpose()?
                        .unwrap_or(CommandSpec::DEFAULT_HOLD_MAX),
                    max_action_age: decl
                        .max_action_age_ms
                        .map(|ms| millis(path, "max_action_age_ms", ms, false))
                        .transpose()?
                        .unwrap_or(CommandSpec::DEFAULT_MAX_ACTION_AGE),
                    merge,
                    reference,
                    group: decl.group.map(|g| name(path, "group", &g)).transpose()?,
                    binary: decl.binary.unwrap_or(false),
                }),
                control_hz,
            )
        }
        _ => (None, 0.0),
    };
    Ok(PartSpec {
        path: part_path,
        kind: decl.kind,
        channel: ChannelId(0),
        applied_channel: None,
        fields,
        hz,
        send,
        frame,
        names,
        encoding,
        max_age,
        command,
        attrs,
        priority,
        owner: robot.clone(),
        event,
        requires: Vec::new(),
        control_hz,
    })
}

/// Normalises `robot`'s authored map. Every part comes back with channel id
/// zero, in authored order.
pub fn normalise(
    robot: &ParticipantId,
    spec_version: u32,
    robot_json: &str,
) -> Result<Vec<PartSpec>, CtlError> {
    if spec_version != SPEC_VERSION {
        return Err(CtlError::new(
            CtlStatus::Incompatible,
            format!("kind table version {spec_version}; this host speaks {SPEC_VERSION}"),
        ));
    }
    let mut de = serde_json::Deserializer::from_str(robot_json);
    let map: Ordered<PartDecl> = serde_path_to_error::deserialize(&mut de)
        .map_err(|err| CtlError::new(CtlStatus::InvalidRequest, format!("robot map: {err}")))?;
    de.end()
        .map_err(|err| CtlError::new(CtlStatus::InvalidRequest, format!("robot map: {err}")))?;
    if map.0.len() > MAX_PARTS {
        return Err(CtlError::new(
            CtlStatus::InvalidRequest,
            format!("{} parts (at most {MAX_PARTS})", map.0.len()),
        ));
    }
    let parts = map
        .0
        .into_iter()
        .map(|(path, decl)| part(robot, &path, decl))
        .collect::<Result<Vec<_>, _>>()?;
    let mut groups: BTreeMap<&str, &PartSpec> = BTreeMap::new();
    for spec in &parts {
        let Some(command) = &spec.command else {
            continue;
        };
        let Some(group) = command.group.as_deref() else {
            continue;
        };
        match groups.get(group) {
            Some(first) => {
                let first_command = first.command.as_ref();
                if first_command.map(|c| (&c.safe_behavior, c.hold_max))
                    != Some((&command.safe_behavior, command.hold_max))
                {
                    return Err(invalid(
                        spec.path.as_str(),
                        format!(
                            "group {group:?} members must declare the same safe and hold_max_ms as {}",
                            first.path
                        ),
                    ));
                }
            }
            None => {
                groups.insert(group, spec);
            }
        }
    }
    Ok(parts)
}

fn field_json(field: &FieldSpec) -> Value {
    json!({
        "name": field.name,
        "dtype": field.dtype.name(),
        "shape": field.shape,
        "unit": field.unit,
        "rotation": field.rotation,
    })
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

/// One normalised part as canonical-form JSON: everything but the
/// host-assigned channel ids and owner.
pub fn part_json(part: &PartSpec) -> Value {
    let command = part.command.as_ref().map(|c| {
        json!({
            "mode": c.mode,
            "fields": c.fields.iter().map(field_json).collect::<Vec<_>>(),
            "limits": c.limits.iter().map(|l| json!({
                "field": l.field,
                "low": l.low,
                "high": l.high,
                "max_norm": l.max_norm,
            })).collect::<Vec<_>>(),
            "safe": c.safe_behavior,
            "hold_max_ms": ms(c.hold_max),
            "max_action_age_ms": ms(c.max_action_age),
            "merge": "replace",
            "reference": c.reference,
            "group": c.group,
            "binary": c.binary,
            "hold_last": c.hold_last,
        })
    });
    let (encoding, quality) = match part.encoding {
        Encoding::Raw => ("raw", None),
        Encoding::Jpeg { quality } => ("jpeg", Some(quality)),
    };
    json!({
        "path": part.path.as_str(),
        "kind": part.kind,
        "fields": part.fields.iter().map(field_json).collect::<Vec<_>>(),
        "hz": part.hz,
        "send": part.send,
        "frame": part.frame,
        "names": part.names,
        "encoding": encoding,
        "quality": quality,
        "max_age_ms": part.max_age.map(ms),
        "command": command,
        "control_hz": part.control_hz,
        "attrs": part.attrs,
        "priority": part.priority.name(),
        "event": part.event,
        "requires": part.requires,
    })
}

/// A robot's normalised map as canonical-form JSON: its parts in authored
/// order.
pub fn canonical(parts: &[PartSpec]) -> Value {
    Value::Array(parts.iter().map(part_json).collect())
}

/// The declaration fingerprint (§3.3): BLAKE3 over the RFC 8785 bytes of
/// [`canonical()`], first 8 bytes.
pub fn fingerprint(parts: &[PartSpec]) -> Fingerprint {
    canonical::fingerprint(&canonical(parts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn robot() -> ParticipantId {
        ParticipantId::new("ur10e").expect("id")
    }

    fn parts(json: &str) -> Vec<PartSpec> {
        normalise(&robot(), SPEC_VERSION, json).expect("normalise")
    }

    fn refused(json: &str) -> CtlError {
        normalise(&robot(), SPEC_VERSION, json).expect_err("refused")
    }

    fn shapes(part: &PartSpec) -> Vec<(String, DType, Vec<u32>, String)> {
        part.fields
            .iter()
            .map(|f| (f.name.clone(), f.dtype, f.shape.clone(), f.unit.clone()))
            .collect()
    }

    #[test]
    fn core_kinds_expand_to_their_fields() {
        let ur = parts(
            r#"{
            "arm": {"kind": "joints", "names": ["a", "b", "c"], "fields": ["effort", "pos"],
                    "hz": 500, "command": "position", "limits": {"pos": [-6, 6]}, "group": "arm",
                    "units": {"pos": "deg"}},
            "arm/tcp": {"kind": "pose", "frame": "base", "child": "tool0", "hz": 500,
                        "command": "delta", "group": "arm",
                        "limits": {"pos": {"max_norm": 0.01}, "rot": {"max_norm": 0.05}}},
            "wrist/cam": {"kind": "camera", "width": 640, "height": 480, "hz": 30,
                          "encoding": "jpeg", "quality": 75, "mount": "upside_down"},
            "head/depth": {"kind": "depth", "width": 64, "height": 48, "hz": 30, "dtype": "uint16"},
            "gripper": {"kind": "gripper", "hz": 100, "command": "position", "width_m": [0, 0.085],
                        "units": {"pos": "m"}, "binary": true},
            "base/imu": {"kind": "imu", "frame": "base", "hz": 500, "rotation": "quat_wxyz",
                         "fields": ["lin_acc", "orientation"]},
            "base/twist": {"kind": "velocity", "frame": "base", "hz": 50, "command": "velocity"},
            "hand/tactile": {"kind": "tactile", "shape": [9, 12], "hz": 100},
            "task": {"kind": "text"},
            "reward": {"kind": "reward"},
            "success": {"kind": "success"}
        }"#,
        );
        let arm = &ur[0];
        assert_eq!(
            shapes(arm),
            [
                ("pos".into(), DType::F32, vec![3], "deg".into()),
                ("effort".into(), DType::F32, vec![3], "N*m".into())
            ],
            "table order, declared units"
        );
        let command = arm.command.as_ref().expect("commandable");
        assert_eq!(
            command.fields[0].unit, "deg",
            "units apply to command fields"
        );
        assert!(command.hold_last && arm.control_hz == 500.0);
        let tcp = ur[1].command.as_ref().expect("delta");
        assert_eq!(
            (
                ur[1].fields[1].rotation.as_str(),
                tcp.fields[1].rotation.as_str()
            ),
            ("quat_xyzw", "axis_angle")
        );
        assert_eq!(tcp.reference, "current");
        assert!(!tcp.hold_last && ur[1].attrs[attrs::CHILD] == "tool0");
        let cam = &ur[2];
        assert_eq!(shapes(cam)[0].2, [480, 640, 3]);
        assert_eq!(cam.encoding, Encoding::Jpeg { quality: 75 });
        assert_eq!(
            (
                cam.priority,
                cam.send.as_str(),
                cam.attrs[attrs::MOUNT].as_str()
            ),
            (ChannelPriority::Bulk, "latest(hz=30)", "upside_down")
        );
        assert_eq!(
            shapes(&ur[3])[0],
            ("depth".into(), DType::U16, vec![48, 64], "mm".into())
        );
        let gripper = &ur[4];
        assert_eq!(gripper.fields[0].unit, "m");
        assert_eq!(gripper.attrs[attrs::WIDTH_M], "[0,0.085]");
        assert!(gripper.command.as_ref().is_some_and(|c| c.binary));
        let imu = &ur[5];
        assert_eq!(
            imu.fields
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            ["orientation", "lin_acc"]
        );
        assert_eq!(imu.fields[0].rotation, "quat_wxyz");
        assert_eq!(shapes(&ur[6])[0].2, [2], "planar by default");
        assert_eq!(
            (ur[7].priority, ur[7].fields[0].unit.as_str()),
            (ChannelPriority::Data, NORMALIZED)
        );
        for event in &ur[8..] {
            assert!(event.event && event.hz == 0.0 && event.priority == ChannelPriority::Data);
        }
        assert_eq!(ur[8].fields[0].dtype, DType::Utf8);
    }

    #[test]
    fn custom_kinds_keep_their_declared_field_order() {
        let go2 = parts(
            r#"{"feet/force": {"kind": "unitree/foot_force", "hz": 500,
                "fields": {"force": ["int16", [4]], "contact": ["bool", [4]]}, "units": {"force": "raw"}},
                "x": {"kind": "x/blob", "hz": 10, "fields": {"b": ["uint8", [8192]]}}}"#,
        );
        assert_eq!(
            shapes(&go2[0]),
            [
                ("force".into(), DType::I16, vec![4], "raw".into()),
                ("contact".into(), DType::Bool, vec![4], String::new())
            ]
        );
        assert_eq!(go2[0].priority, ChannelPriority::Data);
        assert_eq!(
            (go2[1].priority, go2[1].send.as_str()),
            (ChannelPriority::Bulk, "latest(hz=10)")
        );
        // The canonical form keeps the order in an array, so swapping the
        // two fields changes the fingerprint.
        let swapped = parts(
            r#"{"feet/force": {"kind": "unitree/foot_force", "hz": 500,
                "fields": {"contact": ["bool", [4]], "force": ["int16", [4]]}, "units": {"force": "raw"}},
                "x": {"kind": "x/blob", "hz": 10, "fields": {"b": ["uint8", [8192]]}}}"#,
        );
        assert_ne!(fingerprint(&go2), fingerprint(&swapped));
    }

    #[test]
    fn the_fingerprint_is_of_the_normalised_form() {
        let a = parts(r#"{"arm": {"kind": "joints", "count": 2, "hz": 100}}"#);
        let b = parts(
            r#"{"arm": {"hz": 100.0, "fields": ["vel", "pos"], "kind": "joints", "count": 2,
                "send": "every", "dtype": "float32"}}"#,
        );
        assert_eq!(fingerprint(&a), fingerprint(&b), "defaults expanded");
        let c = parts(r#"{"arm": {"kind": "joints", "count": 2, "hz": 101}}"#);
        assert_ne!(fingerprint(&a), fingerprint(&c));
        let text = canonical::to_string(&canonical(&a));
        assert!(text.starts_with(r#"[{"attrs":{},"command":null,"control_hz":0,"encoding":"raw","event":false,"fields":[{"dtype":"float32","name":"pos""#), "{text}");
    }

    #[test]
    fn declarations_failing_the_rules_are_refused_naming_the_path_and_key() {
        let cases = [
            (r#"{"Arm": {"kind": "joints", "count": 1, "hz": 1}}"#, "Arm"),
            (
                r#"{"_session/x": {"kind": "joints", "count": 1, "hz": 1}}"#,
                "reserved",
            ),
            (
                r#"{"arm": {"kind": "rlmesh/a", "hz": 1, "fields": {"v": ["float32", [1]]}}}"#,
                "reserved",
            ),
            (r#"{"arm": {"kind": "points", "hz": 1}}"#, "core kind"),
            (
                r#"{"arm": {"kind": "joints", "count": 1}}"#,
                "hz is required",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 1, "hz": 20000}}"#,
                "hz",
            ),
            (r#"{"arm": {"kind": "joints", "hz": 1}}"#, "names"),
            (
                r#"{"arm": {"kind": "joints", "names": ["a", "a"], "hz": 1}}"#,
                "repeated",
            ),
            (
                r#"{"arm": {"kind": "joints", "names": ["a"], "count": 2, "hz": 1}}"#,
                "count",
            ),
            (r#"{"arm": {"kind": "hand", "count": 2, "hz": 1}}"#, "names"),
            (
                r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "fields": ["pos", "acc"]}}"#,
                "acc",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "width": 3}}"#,
                "width does not apply",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "color": "red"}}"#,
                "color",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "safe": "stop"}}"#,
                "without command",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "command": "delta"}}"#,
                "position, velocity, torque",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "units": {"pos": "mm"}}}"#,
                "angle",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "prismatic": true, "units": {"pos": "deg"}}}"#,
                "length",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "limits": {"pos": [[0, 1]]}}}"#,
                "pairs",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "limits": {"pos": [1, 0]}}}"#,
                "low <= high",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "limits": {"pos": {"max_norm": 1}}}}"#,
                "max_norm",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "limits": {"vel": [0, 1]}}}"#,
                "not a command field",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "hold_max_ms": 70000}}"#,
                "hold_max_ms",
            ),
            (
                r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "range": {"pos": [1, 1]}}}"#,
                "low < high",
            ),
            (
                r#"{"cam": {"kind": "camera", "width": 4, "hz": 1}}"#,
                "width and height",
            ),
            (
                r#"{"cam": {"kind": "camera", "width": 4, "height": 4, "hz": 1, "format": "mono8", "encoding": "jpeg"}}"#,
                "rgb8",
            ),
            (
                r#"{"cam": {"kind": "camera", "width": 4, "height": 4, "hz": 1, "quality": 50}}"#,
                "quality without jpeg",
            ),
            (
                r#"{"cam": {"kind": "camera", "width": 4096, "height": 4096, "hz": 1}}"#,
                "cap",
            ),
            (r#"{"ft": {"kind": "wrench", "hz": 1}}"#, "frame"),
            (
                r#"{"imu": {"kind": "imu", "frame": "base", "hz": 1}}"#,
                "lists its fields",
            ),
            (
                r#"{"imu": {"kind": "imu", "frame": "base", "hz": 1, "fields": ["ang_vel"], "rotation": "quat_wxyz"}}"#,
                "orientation",
            ),
            (
                r#"{"tcp": {"kind": "pose", "frame": "base", "hz": 1}}"#,
                "child",
            ),
            (
                r#"{"tcp": {"kind": "pose", "frame": "base", "child": "t", "hz": 1, "rotation": "gravity_xyz"}}"#,
                "source",
            ),
            (
                r#"{"tcp": {"kind": "pose", "frame": "base", "child": "t", "hz": 1, "command": "absolute", "reference": "target"}}"#,
                "delta only",
            ),
            (
                r#"{"g": {"kind": "gripper", "hz": 1, "units": {"pos": "m"}}}"#,
                "width_m",
            ),
            (r#"{"t": {"kind": "text", "hz": 10}}"#, "hz does not apply"),
            (r#"{"x": {"kind": "x/a", "hz": 1}}"#, "declares its fields"),
            (
                r#"{"x": {"kind": "x/a", "hz": 1, "fields": {"t": ["utf8", []], "v": ["float32", [1]]}}}"#,
                "utf8",
            ),
            (
                r#"{"x": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [1]], "v": ["float32", [1]]}}}"#,
                "duplicate",
            ),
            (
                r#"{"a": {"kind": "joints", "count": 2, "hz": 1, "command": "velocity", "group": "g"},
                 "b": {"kind": "joints", "count": 2, "hz": 1, "command": "velocity", "group": "g", "safe": "brake"}}"#,
                "same safe",
            ),
            ("[]", "robot map"),
            ("{", "robot map"),
            (r#"{} {}"#, "robot map"),
        ];
        for (json, needle) in cases {
            let err = refused(json);
            assert_eq!(err.status, CtlStatus::InvalidRequest, "{json}: {err}");
            assert!(err.message.contains(needle), "{json}: {}", err.message);
        }
        assert_eq!(
            refused(r#"{"a": {"kind": "joints", "count": 1, "hz": 1, "command": "position", "merge": "blend"}}"#).status,
            CtlStatus::Unsupported
        );
        assert_eq!(
            normalise(&robot(), 2, "{}").expect_err("version").status,
            CtlStatus::Incompatible
        );
        let many: Vec<String> = (0..=MAX_PARTS)
            .map(|i| format!(r#""p{i}": {{"kind": "text"}}"#))
            .collect();
        assert!(
            refused(&format!("{{{}}}", many.join(",")))
                .message
                .contains("at most 256")
        );
    }
}
