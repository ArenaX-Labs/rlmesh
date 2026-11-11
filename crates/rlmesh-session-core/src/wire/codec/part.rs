//! Part declarations: fields, commands and channel priorities.

use super::convert::{bad_enum, collect};
use super::pb;
use crate::prelude::*;

use core::time::Duration;

use crate::wire::error::WireError;
use crate::wire::ids::{ChannelId, ParticipantId};
use crate::wire::part::{
    ChannelPriority, CommandSpec, DType, Encoding, FieldLimit, FieldSpec, MergePolicy, PartSpec,
};
use crate::wire::path::PartPath;

pub(super) fn duration_ns(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

fn field_to_pb(f: &FieldSpec) -> pb::FieldSpec {
    pb::FieldSpec {
        name: f.name.clone(),
        dtype: f.dtype.code(),
        shape: f.shape.clone(),
        unit: f.unit.clone(),
        rotation: f.rotation.clone(),
    }
}

fn field_from_pb(f: pb::FieldSpec) -> Result<FieldSpec, WireError> {
    Ok(FieldSpec {
        dtype: DType::from_code(f.dtype).ok_or_else(|| {
            WireError::malformed(
                "field spec",
                format!("field {:?} dtype {}", f.name, f.dtype),
            )
        })?,
        name: f.name,
        shape: f.shape,
        unit: f.unit,
        rotation: f.rotation,
    })
}

fn command_to_pb(c: &CommandSpec) -> pb::Command {
    pb::Command {
        mode: c.mode.clone(),
        fields: c.fields.iter().map(field_to_pb).collect(),
        limits: c
            .limits
            .iter()
            .map(|l| pb::FieldLimit {
                field: l.field.clone(),
                low: l.low.clone(),
                high: l.high.clone(),
                max_norm: l.max_norm.unwrap_or(0.0),
            })
            .collect(),
        safe_behavior: c.safe_behavior.clone(),
        hold_max_ns: duration_ns(c.hold_max),
        max_action_age_ns: duration_ns(c.max_action_age),
        merge: match c.merge {
            MergePolicy::Replace => pb::MergePolicy::Replace,
        } as i32,
        reference: c.reference.clone(),
        group: c.group.clone().unwrap_or_default(),
        binary: c.binary,
        hold_last: c.hold_last,
        latch: c.latch,
    }
}

fn command_from_pb(c: pb::Command) -> Result<CommandSpec, WireError> {
    Ok(CommandSpec {
        mode: c.mode,
        fields: collect(c.fields, field_from_pb)?,
        limits: c
            .limits
            .into_iter()
            .map(|l| FieldLimit {
                field: l.field,
                low: l.low,
                high: l.high,
                max_norm: (l.max_norm > 0.0).then_some(l.max_norm),
            })
            .collect(),
        safe_behavior: c.safe_behavior,
        hold_max: Duration::from_nanos(c.hold_max_ns),
        max_action_age: Duration::from_nanos(c.max_action_age_ns),
        merge: match pb::MergePolicy::try_from(c.merge) {
            Ok(pb::MergePolicy::Replace | pb::MergePolicy::Unspecified) => MergePolicy::Replace,
            Err(_) => return Err(bad_enum("merge policy", c.merge)),
        },
        reference: c.reference,
        group: (!c.group.is_empty()).then_some(c.group),
        binary: c.binary,
        hold_last: c.hold_last,
        latch: c.latch,
    })
}

fn channel_priority_to_pb(p: ChannelPriority) -> pb::ChannelPriority {
    match p {
        ChannelPriority::Control => pb::ChannelPriority::Control,
        ChannelPriority::Realtime => pb::ChannelPriority::Realtime,
        ChannelPriority::Data => pb::ChannelPriority::Data,
        ChannelPriority::Bulk => pb::ChannelPriority::Bulk,
    }
}

pub(super) fn channel_priority_from_pb(raw: i32) -> Result<ChannelPriority, WireError> {
    match pb::ChannelPriority::try_from(raw) {
        Ok(pb::ChannelPriority::Control) => Ok(ChannelPriority::Control),
        Ok(pb::ChannelPriority::Realtime) => Ok(ChannelPriority::Realtime),
        Ok(pb::ChannelPriority::Data) => Ok(ChannelPriority::Data),
        Ok(pb::ChannelPriority::Bulk) => Ok(ChannelPriority::Bulk),
        _ => Err(bad_enum("channel priority", raw)),
    }
}

pub(super) fn part_to_pb(p: &PartSpec) -> pb::PartSpec {
    pb::PartSpec {
        path: p.path.to_string(),
        kind: p.kind.clone(),
        channel: p.channel.0,
        applied_channel: p.applied_channel.map_or(0, |c| c.0),
        fields: p.fields.iter().map(field_to_pb).collect(),
        hz: p.hz,
        send: p.send.clone(),
        frame: p.frame.clone(),
        names: p.names.clone(),
        encoding: p.encoding.wire_name().into(),
        quality: match p.encoding {
            Encoding::Raw => 0,
            Encoding::Jpeg { quality } => u32::from(quality),
        },
        max_age_ns: p.max_age.map_or(0, duration_ns),
        command: p.command.as_ref().map(command_to_pb),
        attrs: p
            .attrs
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        priority: channel_priority_to_pb(p.priority) as i32,
        owner: p.owner.to_string(),
        event: p.event,
        requires: p.requires.clone(),
        control_hz: p.control_hz,
    }
}

pub(super) fn part_from_pb(p: pb::PartSpec) -> Result<PartSpec, WireError> {
    let encoding = match p.encoding.as_str() {
        "" | "raw" => Encoding::Raw,
        "jpeg" => Encoding::Jpeg {
            quality: u8::try_from(p.quality)
                .ok()
                .filter(|q| (1..=100).contains(q))
                .ok_or_else(|| {
                    WireError::malformed("part spec", format!("jpeg quality {}", p.quality))
                })?,
        },
        other => {
            return Err(WireError::malformed(
                "part spec",
                format!("{} encoding {other:?}", p.path),
            ));
        }
    };
    Ok(PartSpec {
        path: PartPath::new(p.path)?,
        kind: p.kind,
        channel: ChannelId(p.channel),
        applied_channel: (p.applied_channel > 0).then_some(ChannelId(p.applied_channel)),
        fields: collect(p.fields, field_from_pb)?,
        hz: p.hz,
        send: p.send,
        frame: p.frame,
        names: p.names,
        encoding,
        max_age: (p.max_age_ns > 0).then(|| Duration::from_nanos(p.max_age_ns)),
        command: p.command.map(command_from_pb).transpose()?,
        attrs: p.attrs,
        priority: channel_priority_from_pb(p.priority)?,
        owner: ParticipantId::new(p.owner)?,
        event: p.event,
        requires: p.requires,
        control_hz: p.control_hz,
    })
}
