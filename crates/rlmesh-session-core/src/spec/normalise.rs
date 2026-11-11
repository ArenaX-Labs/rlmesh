//! The shared normaliser: an embodiment's authored robot map (JSON) to
//! [`PartSpec`]s, with validation, the kind table's defaults
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
//! # Canonical form and fingerprint
//!
//! [`canonical()`] is the normalised map (defaults expanded, host-assigned
//! fields omitted, units normalised) as JSON: an array of parts in
//! authored order, each part's fields an array in payload order. RFC 8785
//! sorts object members, so nothing whose order matters is an object.
//! [`fingerprint`] is BLAKE3 over its RFC 8785 bytes, first 8 bytes.

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::BTreeMap;
use core::time::Duration;

use crate::wire::part::{Encoding, FieldSpec, MAX_PARTS, PartSpec};
use crate::wire::version::SPEC_VERSION;
use crate::wire::{CtlError, CtlStatus, Fingerprint, ParticipantId};
use serde_json::{Value, json};

use crate::spec::canonical;

mod check;
mod decl;
mod part;

use check::invalid;
pub(crate) use decl::{FieldDecl, Ordered};
use part::{PartDecl, part};

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
    /// Gym compatibility: the original Gym space of the part's
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

/// The authored map's top-level paths, in authored order. Duplicate keys
/// are refused, as [`normalise`] refuses them. Keys are not validated as
/// part paths, nor values as parts: [`normalise`] does that.
pub fn authored_paths(robot_json: &str) -> Result<Vec<String>, CtlError> {
    let mut de = serde_json::Deserializer::from_str(robot_json);
    let map: Ordered<serde::de::IgnoredAny> = serde_path_to_error::deserialize(&mut de)
        .map_err(|err| CtlError::new(CtlStatus::InvalidRequest, format!("robot map: {err}")))?;
    de.end()
        .map_err(|err| CtlError::new(CtlStatus::InvalidRequest, format!("robot map: {err}")))?;
    Ok(map.0.into_iter().map(|(path, _)| path).collect())
}

/// A host accepts a robot on any kind table from 1 up to its own.
fn check_spec_version(declared: u32, own: u32) -> Result<(), CtlError> {
    if declared == 0 {
        return Err(CtlError::new(
            CtlStatus::Incompatible,
            "kind table version 0: versions start at 1",
        ));
    }
    if declared > own {
        return Err(CtlError::new(
            CtlStatus::Incompatible,
            format!(
                "kind table version {declared} is newer than this host's {own}: upgrade the host"
            ),
        ));
    }
    Ok(())
}

/// Normalises `robot`'s authored map. Every part comes back with channel id
/// zero, in authored order.
pub fn normalise(
    robot: &ParticipantId,
    spec_version: u32,
    robot_json: &str,
) -> Result<Vec<PartSpec>, CtlError> {
    check_spec_version(spec_version, SPEC_VERSION)?;
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
            "latch": c.latch,
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

/// The declaration fingerprint: BLAKE3 over the RFC 8785 bytes of
/// [`canonical()`], first 8 bytes.
pub fn fingerprint(parts: &[PartSpec]) -> Fingerprint {
    canonical::fingerprint(&canonical(parts))
}

#[cfg(test)]
mod tests;
