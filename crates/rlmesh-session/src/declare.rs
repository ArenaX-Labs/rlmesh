//! The host's normaliser: an embodiment's authored robot map (JSON, §3) to
//! [`PartSpec`]s, with the §3.4 validation the host needs to accept it.
//!
//! The participant sends only the authored map; the host normalises it, so
//! every peer sees one canonical declaration. Channel ids are left at zero
//! here; the [`Authority`](crate::Authority) assigns them.
//!
//! # The explicit-field form (until the kind table)
//!
//! The kind and unit tables, and with them the default fields of the core
//! kinds (`camera`, `joints`, ...), belong to `rlmesh-spec` (wave 2). Until
//! it lands, the host does not interpret kinds: every part spells its fields
//! the way §4.2 has custom kinds do, and `kind` is an opaque, validated
//! string.
//!
//! ```json
//! {
//!   "arm":       {"kind": "joints", "hz": 500,
//!                 "fields": {"pos": ["float32", [6]], "vel": ["float32", [6]]},
//!                 "command": "position", "command_fields": {"pos": ["float32", [6]]},
//!                 "limits": {"pos": [-6.0, 6.0]}, "safe": "stop"},
//!   "wrist/cam": {"kind": "camera", "hz": 30, "encoding": "jpeg",
//!                 "fields": {"image": ["uint8", [480, 640, 3]]}},
//!   "task":      {"kind": "text", "fields": {"text": ["utf8", []]}}
//! }
//! ```
//!
//! Field order is the payload order, so `fields` keeps the authored order.
//! A core kind without `fields` is refused `UNSUPPORTED` until the kind
//! table lands. The only kind-dependent rules applied here are the ones the
//! wire carries: `text`, `reward` and `success` are events (reliable, never
//! decimated), and a `position` or `absolute` command on a core kind may
//! repeat its last target during Hold (§9.5).
//!
//! Beyond §3.1 and §3.2, the explicit form accepts `rotation` as a map from
//! field to rotation encoding (§5.3), so the edge can validate rotation
//! fields (§9.6).

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use rlmesh_wire::part::{
    ChannelPriority, CommandSpec, DType, Encoding, FieldLimit, FieldSpec, Layout, MAX_PARTS,
    MergePolicy, PartSpec, SMALL_SAMPLE_BYTES, SendPolicy,
};
use rlmesh_wire::path::is_identifier;
use rlmesh_wire::version::SPEC_VERSION;
use rlmesh_wire::{ChannelId, CtlError, CtlStatus, PartPath, ParticipantId};
use serde::Deserialize;
use serde::de::{MapAccess, Visitor};

/// The longest frame, mode or behaviour id.
const MAX_NAME_LEN: usize = 64;
/// The longest joint name.
const MAX_ELEMENT_NAME_LEN: usize = 128;
/// The highest declared rate.
const MAX_HZ: f64 = 10_000.0;
/// The longest declared duration, in ms.
const MAX_MS: f64 = 60_000.0;

/// Kinds that are events: reliable, never decimated (§4.2, §8.5).
const EVENT_KINDS: [&str; 3] = ["text", "reward", "success"];
/// Command modes that may repeat their last target during Hold (§9.5), on
/// core kinds.
const HOLDABLE_MODES: [&str; 2] = ["position", "absolute"];
/// Rotation encodings (§5.3) and their widths.
const ROTATIONS: [(&str, u32); 8] = [
    ("quat_xyzw", 4),
    ("quat_wxyz", 4),
    ("axis_angle", 3),
    ("rot6d", 6),
    ("rot6d_rowmajor", 6),
    ("matrix", 9),
    ("euler_xyz", 3),
    ("gravity_xyz", 3),
];

/// A JSON object kept in authored order, refusing duplicate keys.
struct Ordered<T>(Vec<(String, T)>);

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

/// `[dtype, shape]`.
#[derive(Deserialize)]
struct FieldDecl(String, Vec<u32>);

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
    fields: Option<Ordered<FieldDecl>>,
    hz: Option<f64>,
    send: Option<String>,
    frame: Option<String>,
    names: Option<Vec<String>>,
    units: Option<BTreeMap<String, String>>,
    rotation: Option<BTreeMap<String, String>>,
    range: Option<BTreeMap<String, [f64; 2]>>,
    max_age_ms: Option<f64>,
    priority: Option<String>,
    doc: Option<String>,
    encoding: Option<String>,
    quality: Option<u32>,
    command: Option<String>,
    command_fields: Option<Ordered<FieldDecl>>,
    control_hz: Option<f64>,
    safe: Option<String>,
    hold_max_ms: Option<f64>,
    max_action_age_ms: Option<f64>,
    limits: Option<BTreeMap<String, LimitDecl>>,
    merge: Option<String>,
    group: Option<String>,
    reference: Option<String>,
    binary: Option<bool>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
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

/// A core kind, or `ns/name` with `ns` other than `rlmesh`.
fn kind(path: &str, kind: &str) -> Result<(), CtlError> {
    let ok = match kind.split_once('/') {
        None => is_identifier(kind, MAX_NAME_LEN),
        Some((ns, name)) => {
            ns != "rlmesh" && is_identifier(ns, MAX_NAME_LEN) && is_identifier(name, MAX_NAME_LEN)
        }
    };
    if ok {
        Ok(())
    } else {
        Err(invalid(
            path,
            format!("kind {kind:?} must be a core kind or ns/name (ns not rlmesh)"),
        ))
    }
}

fn lower_fields(
    path: &str,
    decl: Ordered<FieldDecl>,
    units: &BTreeMap<String, String>,
    rotation: &BTreeMap<String, String>,
) -> Result<(Vec<FieldSpec>, Layout), CtlError> {
    let fields = decl
        .0
        .into_iter()
        .map(|(field, FieldDecl(dtype, shape))| {
            let field = name(path, "field", &field)?;
            let dtype = DType::from_name(&dtype)
                .ok_or_else(|| invalid(path, format!("field {field:?} dtype {dtype:?}")))?;
            let mut spec = FieldSpec::new(field, dtype, &shape);
            if let Some(unit) = units.get(&spec.name) {
                spec.unit.clone_from(unit);
            }
            if let Some(encoding) = rotation.get(&spec.name) {
                let width = ROTATIONS
                    .iter()
                    .find(|(n, _)| n == encoding)
                    .map(|(_, w)| *w)
                    .ok_or_else(|| invalid(path, format!("rotation {encoding:?}")))?;
                let elements = spec.elements().unwrap_or(0);
                let fits = matches!(spec.dtype, DType::F32 | DType::F64)
                    && (spec.shape.last() == Some(&width)
                        || (width == 9 && spec.shape.ends_with(&[3, 3])));
                if !fits || elements == 0 {
                    return Err(invalid(
                        path,
                        format!(
                            "field {:?} is not a float32/float64 field of {encoding} ({width} wide)",
                            spec.name
                        ),
                    ));
                }
                spec.rotation.clone_from(encoding);
            }
            Ok(spec)
        })
        .collect::<Result<Vec<_>, CtlError>>()?;
    let layout = Layout::of(&fields).map_err(|err| invalid(path, err))?;
    Ok((fields, layout))
}

fn limits(
    path: &str,
    command: &str,
    fields: &[FieldSpec],
    decl: BTreeMap<String, LimitDecl>,
) -> Result<Vec<FieldLimit>, CtlError> {
    decl.into_iter()
        .map(|(field, limit)| {
            let spec = fields.iter().find(|f| f.name == field).ok_or_else(|| {
                invalid(path, format!("limits on {field:?}, not a command field"))
            })?;
            if !matches!(spec.dtype, DType::F32 | DType::F64) {
                return Err(invalid(
                    path,
                    format!("limits on {field:?}: only on float32 and float64 fields"),
                ));
            }
            let elements = spec.elements().unwrap_or(0);
            let pairs = match limit {
                LimitDecl::Norm { max_norm } => {
                    if command != "delta" || spec.shape.len() != 1 {
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

fn part(robot: &ParticipantId, path: &str, decl: PartDecl) -> Result<PartSpec, CtlError> {
    let part_path = PartPath::new(path).map_err(|err| invalid(path, err.reason))?;
    kind(path, &decl.kind)?;
    let mut attrs = BTreeMap::new();
    for (key, value) in &decl.extra {
        if !key.starts_with("x_") {
            return Err(invalid(path, format!("unknown key {key:?}")));
        }
        attrs.insert(key.clone(), value.to_string());
    }
    let Some(field_decl) = decl.fields else {
        return Err(if decl.kind.contains('/') {
            invalid(path, "a custom kind declares its fields")
        } else {
            CtlError::new(
                CtlStatus::Unsupported,
                format!(
                    "{path}: core kind {:?} needs the kind table (rlmesh-spec); until then, \
                     declare its fields explicitly",
                    decl.kind
                ),
            )
        });
    };
    let units = decl.units.unwrap_or_default();
    let rotation = decl.rotation.unwrap_or_default();
    for key in units.keys().chain(rotation.keys()) {
        let known = field_decl.0.iter().any(|(f, _)| f == key)
            || decl
                .command_fields
                .as_ref()
                .is_some_and(|c| c.0.iter().any(|(f, _)| f == key));
        if !known {
            return Err(invalid(
                path,
                format!("units or rotation on unknown field {key:?}"),
            ));
        }
    }
    let (fields, layout) = lower_fields(path, field_decl, &units, &rotation)?;
    let event = EVENT_KINDS.contains(&decl.kind.as_str());
    let hz = match (decl.hz, event) {
        (Some(hz), _) => rate(path, "hz", hz)?,
        (None, true) => 0.0,
        (None, false) => return Err(invalid(path, "hz is required")),
    };
    let encoding = match (decl.encoding.as_deref(), decl.quality) {
        (None | Some("raw"), None) => Encoding::Raw,
        (Some("jpeg"), quality) => {
            if !matches!(fields.as_slice(), [f] if f.rgb8_dims().is_some()) {
                return Err(invalid(path, "jpeg is only for one uint8 [H, W, 3] image"));
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
    let names = decl.names.unwrap_or_default();
    for (i, n) in names.iter().enumerate() {
        if n.is_empty() || n.len() > MAX_ELEMENT_NAME_LEN || names[..i].contains(n) {
            return Err(invalid(
                path,
                format!("names: {n:?} is empty, too long or repeated"),
            ));
        }
    }
    if let Some(range) = decl.range {
        for (field, [low, high]) in &range {
            if !fields.iter().any(|f| &f.name == field) {
                return Err(invalid(path, format!("range on unknown field {field:?}")));
            }
            if !(low.is_finite() && high.is_finite() && low < high) {
                return Err(invalid(
                    path,
                    format!("range on {field:?}: finite, low < high"),
                ));
            }
        }
        attrs.insert(
            "range".into(),
            serde_json::to_string(&range).unwrap_or_default(),
        );
    }
    if let Some(doc) = decl.doc {
        attrs.insert("doc".into(), doc);
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
        None if event => ChannelPriority::Data,
        None if small => ChannelPriority::Realtime,
        None => ChannelPriority::Bulk,
    };
    let send = match decl.send {
        Some(send) => {
            SendPolicy::parse(&send).map_err(|err| invalid(path, err))?;
            send
        }
        None if event || small => "every".into(),
        None => format!("latest(hz={hz})"),
    };

    let command_only = [
        ("command_fields", decl.command_fields.is_some()),
        ("control_hz", decl.control_hz.is_some()),
        ("safe", decl.safe.is_some()),
        ("hold_max_ms", decl.hold_max_ms.is_some()),
        ("max_action_age_ms", decl.max_action_age_ms.is_some()),
        ("limits", decl.limits.is_some()),
        ("merge", decl.merge.is_some()),
        ("group", decl.group.is_some()),
        ("reference", decl.reference.is_some()),
        ("binary", decl.binary.is_some()),
    ];
    let (command, control_hz) = match decl.command {
        None => {
            if let Some((key, _)) = command_only.iter().find(|(_, set)| *set) {
                return Err(invalid(path, format!("{key} without command")));
            }
            (None, 0.0)
        }
        Some(mode) => {
            if event || encoding != Encoding::Raw {
                return Err(invalid(path, "events and encoded parts take no commands"));
            }
            let mode = name(path, "command", &mode)?;
            let command_fields = match decl.command_fields {
                Some(decl) => lower_fields(path, decl, &units, &rotation)?.0,
                None => fields.clone(),
            };
            if matches!(Layout::of(&command_fields), Ok(Layout::Text)) {
                return Err(invalid(path, "text cannot be commanded"));
            }
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
                &mode,
                &command_fields,
                decl.limits.unwrap_or_default(),
            )?;
            let control_hz = rate(path, "control_hz", decl.control_hz.unwrap_or(hz))?;
            let hold_last = !decl.kind.contains('/') && HOLDABLE_MODES.contains(&mode.as_str());
            (
                Some(CommandSpec {
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
                    hold_last,
                }),
                control_hz,
            )
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn robot() -> ParticipantId {
        ParticipantId::new("ur10e").expect("id")
    }

    fn status(json: &str) -> CtlStatus {
        normalise(&robot(), SPEC_VERSION, json)
            .expect_err("refused")
            .status
    }

    const UR: &str = r#"{
        "arm": {"kind": "joints", "hz": 500,
                "fields": {"vel": ["float32", [6]], "pos": ["float32", [6]]},
                "units": {"pos": "rad"},
                "names": ["a", "b", "c", "d", "e", "f"],
                "command": "position", "command_fields": {"pos": ["float32", [6]]},
                "limits": {"pos": [-6.0, 6.0]}, "control_hz": 1000, "group": "arm",
                "x_vendor": {"model": "ur10e"}},
        "arm/tcp": {"kind": "pose", "hz": 500, "frame": "base",
                    "fields": {"pos": ["float32", [3]], "rot": ["float32", [4]]},
                    "rotation": {"rot": "quat_xyzw", "drot": "axis_angle"},
                    "command": "delta",
                    "command_fields": {"dpos": ["float32", [3]], "drot": ["float32", [3]]},
                    "limits": {"dpos": {"max_norm": 0.05}}, "group": "arm"},
        "wrist/cam": {"kind": "camera", "hz": 30, "encoding": "jpeg", "quality": 75,
                      "fields": {"image": ["uint8", [480, 640, 3]]}, "send": "latest(max_mbps=8)"},
        "task": {"kind": "text", "fields": {"text": ["utf8", []]}, "doc": "the instruction"}
    }"#;

    #[test]
    fn the_explicit_form_normalises_in_authored_order() {
        let parts = normalise(&robot(), SPEC_VERSION, UR).expect("normalise");
        let paths: Vec<&str> = parts.iter().map(|p| p.path.as_str()).collect();
        assert_eq!(paths, ["arm", "arm/tcp", "wrist/cam", "task"]);
        let arm = &parts[0];
        assert_eq!(
            arm.fields
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            ["vel", "pos"],
            "field order is payload order"
        );
        assert_eq!(arm.fields[1].unit, "rad");
        assert_eq!(arm.attrs["x_vendor"], r#"{"model":"ur10e"}"#);
        assert_eq!(
            (arm.priority, arm.send.as_str()),
            (ChannelPriority::Realtime, "every")
        );
        let command = arm.command.as_ref().expect("commandable");
        assert!(command.hold_last, "a position command holds its target");
        assert_eq!(arm.control_hz, 1000.0);
        assert_eq!(command.limits[0].low, vec![-6.0]);
        assert_eq!(command.hold_max, CommandSpec::DEFAULT_HOLD_MAX);
        let tcp = parts[1].command.as_ref().expect("delta");
        assert!(tcp.is_delta() && !tcp.hold_last);
        assert_eq!(
            (tcp.reference.as_str(), tcp.fields[1].rotation.as_str()),
            ("current", "axis_angle")
        );
        assert_eq!(parts[1].fields[1].rotation, "quat_xyzw");
        let cam = &parts[2];
        assert_eq!(cam.encoding, Encoding::Jpeg { quality: 75 });
        assert_eq!(cam.priority, ChannelPriority::Bulk);
        let task = &parts[3];
        assert!(task.event && task.hz == 0.0);
        assert_eq!(
            (task.priority, task.attrs["doc"].as_str()),
            (ChannelPriority::Data, "the instruction")
        );
        assert!(
            parts
                .iter()
                .all(|p| p.channel == ChannelId(0) && p.owner == robot())
        );
        let ft = normalise(
            &robot(),
            SPEC_VERSION,
            r#"{"wrist/ft": {"kind": "x/ft", "hz": 500, "fields": {"wrench": ["float64", [6]]},
                "command": "wrench"}}"#,
        )
        .expect("custom");
        assert!(
            !ft[0].command.as_ref().expect("cmd").hold_last,
            "custom kinds run safe"
        );
    }

    #[test]
    fn declarations_failing_the_caps_are_refused_naming_the_path() {
        let refused = [
            r#"{"Arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [1]]}}}"#,
            r#"{"_session/x": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [1]]}}}"#,
            r#"{"arm": {"kind": "rlmesh/a", "hz": 1, "fields": {"v": ["float32", [1]]}}}"#,
            r#"{"arm": {"kind": "x/a", "fields": {"v": ["float32", [1]]}}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 20000, "fields": {"v": ["float32", [1]]}}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["bfloat16", [1]]}}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [0]]}}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [1]], "v": ["float32", [1]]}}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [1]]}, "color": "red"}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [1]]}, "safe": "stop"}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [3]]}, "encoding": "jpeg"}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"t": ["utf8", []], "v": ["float32", [1]]}}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [2]]},
                "command": "position", "limits": {"v": [[0, 1]]}}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [2]]},
                "command": "position", "limits": {"v": [1, 0]}}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [2]]},
                "command": "position", "limits": {"v": {"max_norm": 1}}}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [2]]},
                "command": "position", "hold_max_ms": 70000}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [2]]},
                "command": "position", "reference": "target"}}"#,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"r": ["float32", [3]]},
                "rotation": {"r": "quat_xyzw"}}}"#,
            r#"{"a": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [2]]},
                     "command": "velocity", "group": "g"},
                "b": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [2]]},
                     "command": "velocity", "group": "g", "safe": "brake"}}"#,
            "[]",
            "{",
        ];
        for json in refused {
            assert_eq!(status(json), CtlStatus::InvalidRequest, "{json}");
        }
        let err = normalise(
            &robot(),
            SPEC_VERSION,
            r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [1]]}, "color": "red"}}"#,
        )
        .expect_err("refused");
        assert!(
            err.message.contains("arm") && err.message.contains("color"),
            "{}",
            err.message
        );
        assert_eq!(
            status(r#"{"arm": {"kind": "joints", "names": ["a"], "hz": 500}}"#),
            CtlStatus::Unsupported,
            "core kinds wait for the kind table"
        );
        assert_eq!(
            status(
                r#"{"arm": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [2]]},
                    "command": "position", "merge": "blend"}}"#
            ),
            CtlStatus::Unsupported
        );
        assert_eq!(
            normalise(&robot(), 2, "{}").expect_err("version").status,
            CtlStatus::Incompatible
        );
        let many: Vec<String> = (0..=MAX_PARTS)
            .map(|i| {
                format!(r#""p{i}": {{"kind": "x/a", "hz": 1, "fields": {{"v": ["uint8", []]}}}}"#)
            })
            .collect();
        assert_eq!(
            status(&format!("{{{}}}", many.join(","))),
            CtlStatus::InvalidRequest
        );
    }
}
