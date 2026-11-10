//! A device's view of the session.

use super::convert::{
    clock_mode_from_pb, clock_mode_to_pb, collect, fingerprint_from_pb, mode_from_pb, mode_to_pb,
};
use super::part::{duration_ns, part_from_pb, part_to_pb};
use super::pb;
use super::state::{
    clock_from_pb, clock_to_pb, demand_from_pb, demand_to_pb, last_stop_from_pb, last_stop_to_pb,
    lease_from_pb, lease_to_pb, reason_code_from_pb, reason_code_to_pb, reason_from_pb,
    reason_to_pb,
};
use crate::prelude::*;

use core::time::Duration;

use bytes::Bytes;
use prost::Message as _;

use crate::wire::error::WireError;
use crate::wire::ids::{ChannelId, Epoch, ParticipantId};
use crate::wire::view::{DeviceView, Member, ViewGrid};

/// Encodes one robot's device view.
pub fn encode_view(view: &DeviceView) -> Vec<u8> {
    view_to_pb(view).encode_to_vec()
}

pub(super) fn view_to_pb(view: &DeviceView) -> pb::DeviceView {
    pb::DeviceView {
        robot: view.robot.to_string(),
        epoch: view.epoch.0,
        revision: view.revision,
        mode: mode_to_pb(view.mode) as i32,
        mode_reason: reason_code_to_pb(view.mode_reason),
        last_stop: Some(last_stop_to_pb(view.last_stop)),
        host_ns: view.host_ns,
        clock_mode: clock_mode_to_pb(view.clock_mode) as i32,
        tick_ns: view.tick_ns,
        generation: view.generation,
        parts: view.parts.iter().map(part_to_pb).collect(),
        leases: view.leases.iter().map(lease_to_pb).collect(),
        clocks: view.clocks.iter().map(clock_to_pb).collect(),
        grids: view
            .grids
            .iter()
            .map(|g| pb::ViewGrid {
                binding: Bytes::copy_from_slice(g.binding.as_bytes()),
                channel: g.channel.0,
                dt_ns: duration_ns(g.dt),
            })
            .collect(),
        demands: view.demands.iter().map(demand_to_pb).collect(),
        members: view
            .members
            .iter()
            .map(|m| pb::Member {
                id: m.id.to_string(),
                generation: m.generation,
            })
            .collect(),
        closing: view.closing.as_ref().map(reason_to_pb),
    }
}

/// Decodes one robot's device view, without the authority state's
/// decoder (a board links only this).
pub fn decode_view(buf: &[u8]) -> Result<DeviceView, WireError> {
    view_from_pb(pb::DeviceView::decode(buf)?)
}

pub(super) fn view_from_pb(v: pb::DeviceView) -> Result<DeviceView, WireError> {
    Ok(DeviceView {
        robot: ParticipantId::new(v.robot)?,
        epoch: Epoch(v.epoch),
        revision: v.revision,
        mode: mode_from_pb(v.mode)?,
        mode_reason: reason_code_from_pb(v.mode_reason),
        last_stop: last_stop_from_pb(v.last_stop),
        host_ns: v.host_ns,
        clock_mode: clock_mode_from_pb(v.clock_mode)?,
        tick_ns: v.tick_ns,
        generation: v.generation,
        parts: collect(v.parts, part_from_pb)?,
        leases: collect(v.leases, lease_from_pb)?,
        clocks: collect(v.clocks, clock_from_pb)?,
        grids: collect(v.grids, |g| {
            Ok::<_, WireError>(ViewGrid {
                binding: fingerprint_from_pb("view grid", &g.binding)?,
                channel: ChannelId(g.channel),
                dt: Duration::from_nanos(g.dt_ns),
            })
        })?,
        demands: collect(v.demands, demand_from_pb)?,
        members: collect(v.members, |m| {
            Ok::<_, WireError>(Member {
                id: ParticipantId::new(m.id)?,
                generation: m.generation,
            })
        })?,
        closing: v.closing.map(reason_from_pb),
    })
}
