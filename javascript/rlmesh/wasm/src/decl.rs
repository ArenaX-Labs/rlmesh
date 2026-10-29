//! What a session declares, as the browser names it, and the value codec.
//!
//! A stream or control path is `{robot}/{part}` (the robot's participant id,
//! then the part path), and a part's applied stream is
//! `{robot}/{part}/_applied`. Streams ride the `state` socket, except bulk
//! ones (cameras) which ride `bulk`. Numeric values cross into JS as `f64`,
//! field by field in declaration order; JPEG frames, raw `u8 [H, W, 3]`
//! images and text cross as bytes.

use rlmesh_session_core::wire::part::{DType, FieldSpec, Layout};
use rlmesh_session_core::wire::path::APPLIED;
use rlmesh_session_core::wire::{
    AuthoritySnapshot, ChannelId, ChannelPriority, CommandSpec, EdgeMode, Encoding, Mode, PartSpec,
    ParticipantId, PriorityClass, ReasonCode, StreamSpec,
};

/// How a stream's values reach JS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Packed numbers, delivered as `f64`.
    F64,
    /// One JPEG per sample.
    Jpeg,
    /// One raw `u8 [H, W, 3]` image per sample.
    Rgb8,
    /// UTF-8 text.
    Text,
}

impl Kind {
    /// Its name at the JS boundary.
    pub fn name(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::Jpeg => "jpeg",
            Self::Rgb8 => "rgb8",
            Self::Text => "text",
        }
    }

    /// Whether samples are delivered as frames (bytes) rather than numbers.
    pub fn is_frame(self) -> bool {
        matches!(self, Self::Jpeg | Self::Rgb8)
    }
}

/// One field, for JS.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldDecl {
    /// Its name.
    pub name: String,
    /// Its dtype's name (`f32`, `u8`, ...).
    pub dtype: &'static str,
    /// Its shape.
    pub shape: Vec<u32>,
    /// Its unit.
    pub unit: String,
    /// Its first element's index in the `f64` values.
    pub offset: usize,
    /// Its element count.
    pub len: usize,
}

/// A stream the browser can subscribe to.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamDecl {
    /// `{robot}/{part}` or `{robot}/{part}/_applied`.
    pub path: String,
    /// The publishing robot.
    pub robot: String,
    /// The part path.
    pub part: String,
    /// An applied stream.
    pub applied: bool,
    /// The stream id.
    pub channel: ChannelId,
    /// `state` or `bulk`.
    pub class: &'static str,
    /// How values reach JS.
    pub kind: Kind,
    /// The declared rate (0 for events).
    pub hz: f64,
    /// The declared freshness bound, ms (`NaN`: none).
    pub max_age_ms: f64,
    /// The fields.
    pub fields: Vec<FieldDecl>,
    /// Image width (frames).
    pub width: u32,
    /// Image height (frames).
    pub height: u32,
}

/// A commandable part.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlDecl {
    /// `{robot}/{part}`.
    pub path: String,
    /// The robot.
    pub robot: String,
    /// The part path.
    pub part: String,
    /// Its id.
    pub channel: ChannelId,
    /// Its kind (`joints`, `gripper`, ...).
    pub kind: String,
    /// Its command mode (`position`, `velocity`, ...).
    pub mode: String,
    /// Its command group, if any (one member is leased at a time).
    pub group: Option<String>,
    /// The command fields, in the order `send`/`drive` values follow.
    pub fields: Vec<FieldDecl>,
    /// Its control rate.
    pub hz: f64,
    /// How long the edge holds before going safe, ms.
    pub hold_max_ms: f64,
    /// Whether this participant's binding covers it (`lease()` accepts it).
    pub bound: bool,
}

/// Splits `{robot}/{part}[/_applied]`.
pub fn split_path(path: &str) -> Option<(ParticipantId, &str, bool)> {
    let (robot, part) = path.split_once('/')?;
    let robot = ParticipantId::new(robot).ok()?;
    match part.strip_suffix(APPLIED).and_then(|p| p.strip_suffix('/')) {
        Some(part) if !part.is_empty() => Some((robot, part, true)),
        _ if part.is_empty() => None,
        _ => Some((robot, part, false)),
    }
}

/// The socket a stream rides.
pub fn class_of(priority: ChannelPriority) -> &'static str {
    match priority {
        ChannelPriority::Bulk => "bulk",
        _ => "state",
    }
}

/// How a stream's values reach JS.
pub fn kind_of(spec: &StreamSpec) -> Kind {
    match (&spec.layout, spec.encoding) {
        (Layout::Text, _) => Kind::Text,
        (_, Encoding::Jpeg { .. }) => Kind::Jpeg,
        (_, Encoding::Raw) => match spec.fields.as_slice() {
            [field] if field.rgb8_dims().is_some() => Kind::Rgb8,
            _ => Kind::F64,
        },
    }
}

fn fields(specs: &[FieldSpec]) -> Vec<FieldDecl> {
    let mut offset = 0;
    specs
        .iter()
        .map(|f| {
            let len = f
                .elements()
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(0);
            let decl = FieldDecl {
                name: f.name.clone(),
                dtype: f.dtype.name(),
                shape: f.shape.clone(),
                unit: f.unit.clone(),
                offset,
                len,
            };
            offset += len;
            decl
        })
        .collect()
}

fn stream_decl(part: &PartSpec, spec: &StreamSpec) -> StreamDecl {
    let (height, width) = match spec.fields.as_slice() {
        [field] => field.rgb8_dims().unwrap_or((0, 0)),
        _ => (0, 0),
    };
    let path = format!("{}/{}", part.owner, spec.path);
    StreamDecl {
        path,
        robot: part.owner.to_string(),
        part: part.path.to_string(),
        applied: spec.applied,
        channel: spec.channel,
        class: class_of(spec.priority),
        kind: kind_of(spec),
        hz: if spec.applied {
            part.control_hz
        } else {
            part.hz
        },
        max_age_ms: spec.max_age.map_or(f64::NAN, |d| d.as_secs_f64() * 1e3),
        fields: fields(&spec.fields),
        width,
        height,
    }
}

/// Every stream of every part: its samples, and the applied stream of each
/// commandable part.
pub fn streams(snapshot: &AuthoritySnapshot) -> Vec<StreamDecl> {
    let mut out = Vec::new();
    for part in &snapshot.parts {
        if let Ok(spec) = part.stream() {
            out.push(stream_decl(part, &spec));
        }
        if let Some(Ok(spec)) = part.applied_stream() {
            out.push(stream_decl(part, &spec));
        }
    }
    out
}

/// The stream at `path`, with its declaration.
pub fn stream(snapshot: &AuthoritySnapshot, path: &str) -> Option<(StreamSpec, StreamDecl)> {
    let (robot, part_path, applied) = split_path(path)?;
    let part = snapshot.part_at(&robot, part_path)?;
    let spec = if applied {
        part.applied_stream()?.ok()?
    } else {
        part.stream().ok()?
    };
    let decl = stream_decl(part, &spec);
    Some((spec, decl))
}

/// Every commandable part; `bound` when `me`'s binding lists it.
pub fn controls(snapshot: &AuthoritySnapshot, me: Option<&ParticipantId>) -> Vec<ControlDecl> {
    let bound = me
        .and_then(|me| snapshot.binding(me))
        .map(|b| b.channels.clone())
        .unwrap_or_default();
    snapshot
        .parts
        .iter()
        .filter_map(|part| {
            let command = part.command.as_ref()?;
            Some(ControlDecl {
                path: format!("{}/{}", part.owner, part.path),
                robot: part.owner.to_string(),
                part: part.path.to_string(),
                channel: part.channel,
                kind: part.kind.clone(),
                mode: command.mode.clone(),
                group: command.group.clone(),
                fields: fields(&command.fields),
                hz: part.control_hz,
                hold_max_ms: command.hold_max.as_secs_f64() * 1e3,
                bound: bound.contains(&part.channel),
            })
        })
        .collect()
}

/// The commandable part at `path`.
pub fn control<'a>(snapshot: &'a AuthoritySnapshot, path: &str) -> Option<&'a PartSpec> {
    let (robot, part_path, applied) = split_path(path)?;
    if applied {
        return None;
    }
    snapshot
        .part_at(&robot, part_path)
        .filter(|p| p.command.is_some())
}

/// The session mode, lowercase.
pub fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Disarmed => "disarmed",
        Mode::Armed => "armed",
        Mode::Active => "active",
        Mode::Safe => "safe",
        Mode::Estop => "estop",
    }
}

/// A reason code, lowercase (`robot_lost`).
pub fn reason_name(code: ReasonCode) -> String {
    code.name().to_ascii_lowercase()
}

/// The edge mode, lowercase.
pub fn edge_mode_name(mode: EdgeMode) -> &'static str {
    match mode {
        EdgeMode::Idle => "idle",
        EdgeMode::Driving => "driving",
        EdgeMode::Hold => "hold",
        EdgeMode::Safe => "safe",
        EdgeMode::Estop => "estop",
    }
}

/// A lease class, lowercase.
pub fn priority_name(priority: PriorityClass) -> &'static str {
    match priority {
        PriorityClass::Scripted => "scripted",
        PriorityClass::Model => "model",
        PriorityClass::Operator => "operator",
        PriorityClass::Safety => "safety",
    }
}

/// An IEEE 754 half as `f32`.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let frac = f32::from(bits & 0x3ff);
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        0x1f if frac == 0.0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// An `f32` as an IEEE 754 half (nearest; overflow to infinity).
fn f32_to_f16(value: f32) -> u16 {
    let sign: u16 = if value.is_sign_negative() { 0x8000 } else { 0 };
    let v = value.abs();
    if v.is_nan() {
        return 0x7e00;
    }
    if v >= 65520.0 {
        return sign | 0x7c00;
    }
    if v < 6.103_515_6e-5 {
        // Subnormal: units of 2^-24.
        return sign | (v * 2f32.powi(24)).round() as u16;
    }
    let exp = v.log2().floor() as i32;
    let mut mant = ((v / 2f32.powi(exp) - 1.0) * 1024.0).round() as u32;
    let mut e = exp + 15;
    if mant == 1024 {
        mant = 0;
        e += 1;
    }
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    sign | ((e as u16) << 10) | mant as u16
}

fn read_f64(dtype: DType, b: &[u8]) -> f64 {
    let a = |n: usize| -> [u8; 8] {
        let mut out = [0; 8];
        out[..n].copy_from_slice(&b[..n]);
        out
    };
    match dtype {
        DType::Bool | DType::U8 => f64::from(b[0]),
        DType::I8 => f64::from(b[0] as i8),
        DType::U16 => f64::from(u16::from_le_bytes([b[0], b[1]])),
        DType::I16 => f64::from(i16::from_le_bytes([b[0], b[1]])),
        DType::U32 => f64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        DType::I32 => f64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        DType::U64 => u64::from_le_bytes(a(8)) as f64,
        DType::I64 => i64::from_le_bytes(a(8)) as f64,
        DType::F16 => f64::from(f16_to_f32(u16::from_le_bytes([b[0], b[1]]))),
        DType::F32 => f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        DType::F64 => f64::from_le_bytes(a(8)),
        DType::Utf8 => f64::NAN,
    }
}

/// Every element of one packed value as `f64`, field by field, into `out`
/// (cleared first).
pub fn unpack_f64(fields: &[FieldSpec], layout: &Layout, value: &[u8], out: &mut Vec<f64>) {
    out.clear();
    for (i, field) in fields.iter().enumerate() {
        let Some(range) = layout.field(i) else {
            continue;
        };
        let Some(bytes) = value.get(range) else {
            continue;
        };
        let size = field.dtype.size() as usize;
        if size == 0 {
            continue;
        }
        out.extend(bytes.chunks_exact(size).map(|b| read_f64(field.dtype, b)));
    }
}

fn write(dtype: DType, value: f64, out: &mut Vec<u8>) -> Result<(), String> {
    let int = |lo: f64, hi: f64| -> Result<f64, String> {
        let v = value.round();
        if value.is_finite() && (lo..=hi).contains(&v) {
            Ok(v)
        } else {
            // No `{value}`: float formatting would add ~16 KB to the .wasm.
            Err(format!("a value does not fit {}", dtype.name()))
        }
    };
    match dtype {
        DType::Bool => out.push(u8::from(value != 0.0)),
        DType::U8 => out.push(int(0.0, 255.0)? as u8),
        DType::I8 => out.push(int(-128.0, 127.0)? as i8 as u8),
        DType::U16 => out.extend((int(0.0, 65_535.0)? as u16).to_le_bytes()),
        DType::I16 => out.extend((int(-32_768.0, 32_767.0)? as i16).to_le_bytes()),
        DType::U32 => out.extend((int(0.0, 4_294_967_295.0)? as u32).to_le_bytes()),
        DType::I32 => out.extend((int(-2_147_483_648.0, 2_147_483_647.0)? as i32).to_le_bytes()),
        DType::U64 => out.extend((int(0.0, 9_007_199_254_740_991.0)? as u64).to_le_bytes()),
        DType::I64 => out
            .extend((int(-9_007_199_254_740_991.0, 9_007_199_254_740_991.0)? as i64).to_le_bytes()),
        DType::F16 => out.extend(f32_to_f16(value as f32).to_le_bytes()),
        DType::F32 => out.extend((value as f32).to_le_bytes()),
        DType::F64 => out.extend(value.to_le_bytes()),
        DType::Utf8 => return Err("a text command takes no numbers".into()),
    }
    Ok(())
}

/// One command value packed at the command layout from `values`, the
/// command fields' elements in order.
pub fn pack_command(command: &CommandSpec, values: &[f64]) -> Result<Vec<u8>, String> {
    let layout = command.layout()?;
    let mut fields: Vec<Vec<u8>> = Vec::with_capacity(command.fields.len());
    let mut rest = values;
    for field in &command.fields {
        let n = field
            .elements()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("field too large")?;
        if rest.len() < n {
            return Err(format!(
                "{} values for {} elements ({})",
                values.len(),
                command
                    .fields
                    .iter()
                    .filter_map(FieldSpec::elements)
                    .sum::<u64>(),
                command
                    .fields
                    .iter()
                    .map(|f| format!("{} {}{:?}", f.name, f.dtype.name(), f.shape))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let (mine, tail) = rest.split_at(n);
        let mut bytes = Vec::with_capacity(n * field.dtype.size() as usize);
        for v in mine {
            write(field.dtype, *v, &mut bytes).map_err(|e| format!("{}: {e}", field.name))?;
        }
        fields.push(bytes);
        rest = tail;
    }
    if !rest.is_empty() {
        return Err(format!("{} values too many", rest.len()));
    }
    let refs: Vec<&[u8]> = fields.iter().map(Vec::as_slice).collect();
    layout.pack(&refs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_split_into_robot_part_and_applied() {
        let (robot, part, applied) = split_path("arm7/arm/_applied").expect("path");
        assert_eq!((robot.as_str(), part, applied), ("arm7", "arm", true));
        let (_, part, applied) = split_path("arm7/scene/cam").expect("path");
        assert_eq!((part, applied), ("scene/cam", false));
        assert!(split_path("arm7").is_none());
        assert!(split_path("arm7/").is_none());
        assert!(split_path("a b/arm").is_none());
    }

    #[test]
    fn halves_round_trip_and_values_pack_at_the_layout() {
        for v in [0.0_f32, 1.0, -2.5, 65504.0, 6.1e-5, 1e-7] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= v.abs() * 1e-3 + 6e-8, "{v} -> {back}");
        }
        let command = CommandSpec {
            mode: "position".into(),
            fields: vec![
                FieldSpec::new("pos", DType::F32, &[3]),
                FieldSpec::new("on", DType::Bool, &[]),
            ],
            limits: Vec::new(),
            safe_behavior: "stop".into(),
            hold_max: CommandSpec::DEFAULT_HOLD_MAX,
            max_action_age: CommandSpec::DEFAULT_MAX_ACTION_AGE,
            merge: rlmesh_session_core::wire::MergePolicy::Replace,
            reference: String::new(),
            group: None,
            binary: false,
            hold_last: true,
        };
        let packed = pack_command(&command, &[1.0, 2.0, 3.0, 1.0]).expect("packed");
        assert_eq!(packed.len(), 24);
        let mut out = Vec::new();
        let layout = command.layout().expect("layout");
        unpack_f64(&command.fields, &layout, &packed, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 1.0]);
        assert!(pack_command(&command, &[1.0]).is_err());
        assert!(pack_command(&command, &[1.0, 2.0, 3.0, 1.0, 5.0]).is_err());
    }
}
