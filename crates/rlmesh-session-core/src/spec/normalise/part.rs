//! One part: its declaration, lowered to the kind's fields, then the
//! normalised [`PartSpec`].

use super::attrs;
use super::check::{
    apply_units, field, float_dtype, invalid, limits, millis, name, optional_fields, pair, rate,
    rotation_field, schema,
};
use super::decl::{COMMAND, FieldsDecl, LimitDecl, Ordered, RangeDecl, allowed};
use super::{MAX_DOC_LEN, MAX_ELEMENT_NAME_LEN, MAX_JOINTS};
use crate::prelude::*;

use alloc::collections::BTreeMap;

use crate::wire::part::{
    ChannelPriority, CommandSpec, DType, Encoding, FieldSpec, Layout, MergePolicy, PartSpec,
    SMALL_SAMPLE_BYTES, SendPolicy,
};
use crate::wire::{ChannelId, CtlError, CtlStatus, PartPath, ParticipantId};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::spec::canonical;
use crate::spec::kinds::{Kind, joint_quantity, quantity};
use crate::spec::rotation::Rotation;
use crate::spec::units::{NORMALIZED, Quantity};

#[derive(Deserialize)]
pub(super) struct PartDecl {
    kind: String,
    hz: Option<f64>,
    send: Option<String>,
    frame: Option<String>,
    dtype: Option<String>,
    units: Option<Ordered<String>>,
    unit: Option<String>,
    range: Option<RangeDecl>,
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
    latch: Option<bool>,
    limits: Option<BTreeMap<String, LimitDecl>>,
    merge: Option<String>,
    group: Option<String>,
    reference: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl PartDecl {
    /// The keys this declaration sets.
    fn present(&self) -> Vec<&'static str> {
        let keys: [(&'static str, bool); 43] = [
            ("kind", true),
            ("hz", self.hz.is_some()),
            ("send", self.send.is_some()),
            ("frame", self.frame.is_some()),
            ("dtype", self.dtype.is_some()),
            ("units", self.units.is_some()),
            ("unit", self.unit.is_some()),
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
            ("latch", self.latch.is_some()),
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
        Kind::Switch => {
            let fields = vec![field("on", DType::Bool, &[], "")];
            let command = (mode.as_deref() == Some("set")).then(|| fields.clone());
            (fields, command)
        }
        Kind::Value => {
            let unit = decl.unit.as_deref();
            if let Some(unit) = unit.filter(|u| !Quantity::Custom.accepts(u)) {
                return Err(invalid(path, format!("unit {unit:?}: 1 to 32 bytes")));
            }
            let fields = vec![field("value", DType::F32, &[], unit.unwrap_or(""))];
            let command = (mode.as_deref() == Some("set")).then(|| fields.clone());
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

pub(super) fn part(
    robot: &ParticipantId,
    path: &str,
    mut decl: PartDecl,
) -> Result<PartSpec, CtlError> {
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
        let ranges = match (kind, range) {
            (Kind::Value, RangeDecl::Bounds(bounds)) => vec![("value".to_string(), bounds)],
            (Kind::Value, RangeDecl::Fields(_)) => {
                return Err(invalid(path, "range is [low, high]"));
            }
            (_, RangeDecl::Bounds(_)) => {
                return Err(invalid(path, "range is {field: [low, high]}"));
            }
            (_, RangeDecl::Fields(ranges)) => ranges.0,
        };
        let mut map = serde_json::Map::new();
        for (key, bounds) in ranges {
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
            let Some(safe) = decl.safe.as_deref() else {
                return Err(invalid(
                    path,
                    format!(
                        "command {mode:?} needs a safe behaviour: \"safe\": \"stop\", \"hold\", \
                         or the name of one the robot's firmware implements"
                    ),
                ));
            };
            if encoding != Encoding::Raw {
                return Err(invalid(path, "encoded parts take no commands"));
            }
            // A text command (one utf8 field) is sent whole, as its slab.
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
            let hold_last = kind.holds_target(&mode);
            if decl.latch == Some(true) && !hold_last {
                return Err(invalid(
                    path,
                    format!(
                        "latch: command {mode:?} does not hold its target, \
                         and a latched velocity or torque would run away"
                    ),
                ));
            }
            (
                Some(CommandSpec {
                    hold_last,
                    latch: decl
                        .latch
                        .unwrap_or(matches!(kind, Kind::Switch | Kind::Value)),
                    mode,
                    fields: command_fields,
                    limits,
                    safe_behavior: name(path, "safe", safe)?,
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
