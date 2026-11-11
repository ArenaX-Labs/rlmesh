//! A device's view of the session.

use super::bound;
use super::convert::{clock_mode_to_pb, mode_to_pb};
use super::part::{duration_ns, part_from_pb, part_to_pb};
use super::pb;
use super::reader::{Fields, Value};
use super::state::{
    clock_to_pb, demand_to_pb, last_stop_to_pb, lease_to_pb, reason_code_from_pb,
    reason_code_to_pb, reason_to_pb,
};
use crate::prelude::*;

use core::time::Duration;

use bytes::Bytes;
use prost::Message as _;

use crate::wire::authority::{
    ClockMapping, ClockMode, LastStop, Lease, Mode, PriorityClass, Reason, ReasonCode,
};
use crate::wire::contract::{Demand, Plan};
use crate::wire::datagram::{DatagramRoute, DatagramRoutes};
use crate::wire::error::{Unread, WireError};
use crate::wire::ids::{ChannelId, Epoch, Fingerprint, LeaseToken, ParticipantId};
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
        datagram: view.datagram.as_ref().map(|d| pb::DatagramRoutes {
            probe: u32::from(d.probe),
            routes: d
                .routes
                .iter()
                .map(|r| pb::DatagramRoute {
                    route: u32::from(r.route),
                    channel: r.channel.0,
                    up: r.up,
                })
                .collect(),
        }),
    }
}

/// The most of each a board takes in one device view: more is refused
/// unread ([`Unread::TooLarge`]). Its own parts, a lease, grid and two
/// streams' plans per part, and a clock per lease holder.
pub const MAX_VIEW_PARTS: usize = 32;
/// The most leases in one view (one per part).
pub const MAX_VIEW_LEASES: usize = MAX_VIEW_PARTS;
/// The most clock mappings in one view (its own and each lease holder's).
pub const MAX_VIEW_CLOCKS: usize = MAX_VIEW_LEASES + 1;
/// The most action grids in one view.
pub const MAX_VIEW_GRIDS: usize = MAX_VIEW_LEASES;
/// The most delivery plans in one view (a stream and an applied stream
/// per part).
pub const MAX_VIEW_DEMANDS: usize = 2 * MAX_VIEW_PARTS;
/// The most members one view lists: a session larger than this is more
/// than a board keeps e-stop keys for.
pub const MAX_VIEW_MEMBERS: usize = 64;
/// The longest text in a view (the closing reason's).
pub const MAX_VIEW_TEXT: usize = 1024;
/// The most datagram routes in one view: a part's chunks, its samples and
/// its `_applied` entries.
pub const MAX_VIEW_ROUTES: usize = 3 * MAX_VIEW_PARTS;

const VIEW: &str = "device view";

/// Decodes one robot's device view, without the authority state's
/// decoder (a board links only this).
///
/// # Errors
/// [`WireError`]: what [`decode_view_into`] refuses.
pub fn decode_view(buf: &[u8]) -> Result<DeviceView, WireError> {
    let mut view = DeviceView::empty(ParticipantId::unset());
    decode_view_into(buf, &mut view, true)?;
    Ok(view)
}

/// Decodes one robot's device view into `into`, reusing what it holds: its
/// vectors' room, and each id that is the same as before. A board that
/// decodes each view into the one it held two views ago allocates nothing
/// while its session stays the same. Its parts are decoded only when
/// `parts` is set (a board takes them once, from its declaration's reply);
/// otherwise they are skipped unread and `into` holds none.
///
/// Every count and length is checked against this module's caps
/// (`MAX_VIEW_*`, and an id's 128 bytes) before anything is allocated for
/// it, every allocation is fallible, and no error
/// allocates: a view a board cannot take is refused, never a panic.
///
/// # Errors
/// [`WireError::Unread`]: malformed, over a cap, or more than the heap can
/// give now; a part spec's own errors when `parts` is set. `into` is then
/// unspecified: decode into a spare, and keep the view held.
pub fn decode_view_into(buf: &[u8], into: &mut DeviceView, parts: bool) -> Result<(), WireError> {
    // Each repeated field's count first, so each vector is sized once.
    let mut counts = [0usize; 6];
    let mut fields = Fields::new(buf, VIEW);
    while let Some((tag, _)) = fields.next_field()? {
        if let Some(n) = tag
            .checked_sub(11)
            .and_then(|i| counts.get_mut(usize::try_from(i).ok()?))
        {
            *n += 1;
        }
    }
    let [n_parts, n_leases, n_clocks, n_grids, n_demands, n_members] = counts;
    into.parts.clear();
    room(
        &mut into.parts,
        if parts { n_parts } else { 0 },
        MAX_VIEW_PARTS,
        "parts",
    )?;
    room(&mut into.leases, n_leases, MAX_VIEW_LEASES, "leases")?;
    room(&mut into.clocks, n_clocks, MAX_VIEW_CLOCKS, "clocks")?;
    room(&mut into.grids, n_grids, MAX_VIEW_GRIDS, "grids")?;
    room(&mut into.demands, n_demands, MAX_VIEW_DEMANDS, "demands")?;
    room(&mut into.members, n_members, MAX_VIEW_MEMBERS, "members")?;

    let (mut robot, mut mode, mut clock_mode, mut closing) = (false, 0, 0, false);
    let mut datagram = false;
    into.epoch = Epoch(0);
    into.revision = 0;
    into.mode_reason = reason_code_from_pb(0);
    into.last_stop = LastStop::default();
    into.host_ns = 0;
    into.tick_ns = 0;
    into.generation = 0;
    let mut at = [0usize; 6];
    let mut fields = Fields::new(buf, VIEW);
    while let Some((tag, value)) = fields.next_field()? {
        match tag {
            1 => {
                into.robot.assign(value.bytes(VIEW)?).map_err(unread)?;
                robot = true;
            }
            2 => into.epoch = Epoch(value.u64(VIEW)?),
            3 => into.revision = value.u64(VIEW)?,
            4 => mode = value.i32(VIEW)?,
            5 => into.mode_reason = reason_code_from_pb(value.i32(VIEW)?),
            6 => last_stop_into(value.bytes(VIEW)?, &mut into.last_stop)?,
            7 => into.host_ns = value.i64(VIEW)?,
            8 => clock_mode = value.i32(VIEW)?,
            9 => into.tick_ns = value.u64(VIEW)?,
            10 => into.generation = value.u32(VIEW)?,
            11 if parts => {
                let part = value.bytes(VIEW)?;
                bound::check(part, &bound::PART)?;
                into.parts.push(part_from_pb(pb::PartSpec::decode(part)?)?);
            }
            12 => lease_into(
                value.bytes(VIEW)?,
                next(&mut into.leases, &mut at[1], || Lease {
                    channel: ChannelId(0),
                    holder: ParticipantId::unset(),
                    priority: PriorityClass::Scripted,
                    token: LeaseToken(0),
                    ttl_ns: 0,
                    granted_ns: 0,
                    expires_ns: 0,
                    binding: Fingerprint::default(),
                    key_id: 0,
                })?,
            )?,
            13 => clock_into(
                value.bytes(VIEW)?,
                next(&mut into.clocks, &mut at[2], || ClockMapping {
                    participant: ParticipantId::unset(),
                    offset_ns: 0,
                    error_ns: 0,
                    rtt_ns: 0,
                    sample_ns: 0,
                    drift_bound_ppm: 0.0,
                })?,
            )?,
            14 => grid_into(
                value.bytes(VIEW)?,
                next(&mut into.grids, &mut at[3], || ViewGrid {
                    binding: Fingerprint::default(),
                    channel: ChannelId(0),
                    dt: Duration::ZERO,
                })?,
            )?,
            15 => demand_into(
                value.bytes(VIEW)?,
                next(&mut into.demands, &mut at[4], || Demand {
                    channel: ChannelId(0),
                    revision: 0,
                    plan: Plan::Every,
                    retain: Duration::ZERO,
                    acked: 0,
                })?,
            )?,
            16 => member_into(
                value.bytes(VIEW)?,
                next(&mut into.members, &mut at[5], || Member {
                    id: ParticipantId::unset(),
                    generation: 0,
                })?,
            )?,
            17 => {
                let reason = into.closing.get_or_insert_with(Reason::default);
                if !closing {
                    // Cleared in place: its text buffers stay.
                    reason.code = ReasonCode::default();
                    reason.by.clear();
                    reason.detail.clear();
                    reason.at_ns = 0;
                    reason.revision = 0;
                    reason.verified = false;
                    closing = true;
                }
                reason_into(value.bytes(VIEW)?, reason)?;
            }
            18 => {
                let routes = into.datagram.get_or_insert_with(DatagramRoutes::default);
                if !datagram {
                    routes.probe = 0;
                    routes.routes.clear();
                    datagram = true;
                }
                routes_into(value.bytes(VIEW)?, routes)?;
            }
            _ => {}
        }
    }
    if !robot {
        return Err(unread(Unread::Malformed("robot")));
    }
    if !closing {
        into.closing = None;
    }
    if !datagram {
        into.datagram = None;
    }
    into.mode = mode_of(mode)?;
    into.clock_mode = match pb::ClockMode::try_from(clock_mode) {
        Ok(pb::ClockMode::Wall | pb::ClockMode::Unspecified) => ClockMode::Wall,
        Ok(pb::ClockMode::Virtual) => ClockMode::Virtual,
        Err(_) => return Err(unread(Unread::Malformed("clock mode"))),
    };
    into.leases.truncate(n_leases);
    into.clocks.truncate(n_clocks);
    into.grids.truncate(n_grids);
    into.demands.truncate(n_demands);
    into.members.truncate(n_members);
    Ok(())
}

fn unread(why: Unread) -> WireError {
    WireError::unread(VIEW, why)
}

/// Room in `v` for `n`, at most `max`, without allocating past it.
fn room<T>(v: &mut Vec<T>, n: usize, max: usize, what: &'static str) -> Result<(), WireError> {
    if n > max {
        return Err(unread(Unread::TooLarge(what)));
    }
    if let Some(more) = n.checked_sub(v.len()).filter(|_| v.capacity() < n) {
        v.try_reserve_exact(more)
            .map_err(|_| unread(Unread::NoMemory))?;
    }
    Ok(())
}

/// The next element of `v` to decode into: the one held there, or a blank
/// one in the room [`room`] made.
fn next<'v, T>(
    v: &'v mut Vec<T>,
    at: &mut usize,
    blank: impl FnOnce() -> T,
) -> Result<&'v mut T, WireError> {
    let i = *at;
    *at += 1;
    if i >= v.len() {
        if v.len() == v.capacity() {
            return Err(unread(Unread::NoMemory));
        }
        v.push(blank());
    }
    v.get_mut(i).ok_or_else(|| unread(Unread::NoMemory))
}

fn mode_of(raw: i32) -> Result<Mode, WireError> {
    match pb::Mode::try_from(raw) {
        Ok(pb::Mode::Disarmed) => Ok(Mode::Disarmed),
        Ok(pb::Mode::Armed) => Ok(Mode::Armed),
        Ok(pb::Mode::Active) => Ok(Mode::Active),
        Ok(pb::Mode::Safe) => Ok(Mode::Safe),
        Ok(pb::Mode::Estop) => Ok(Mode::Estop),
        _ => Err(unread(Unread::Malformed("mode"))),
    }
}

fn fingerprint(raw: Option<&[u8]>, what: &'static str) -> Result<Fingerprint, WireError> {
    Fingerprint::from_slice(raw.unwrap_or_default()).ok_or(unread(Unread::Malformed(what)))
}

fn last_stop_into(buf: &[u8], into: &mut LastStop) -> Result<(), WireError> {
    const M: &str = "last stop";
    let mut fields = Fields::new(buf, M);
    while let Some((tag, value)) = fields.next_field()? {
        match tag {
            1 => into.epoch = Epoch(value.u64(M)?),
            2 => into.revision = value.u64(M)?,
            _ => {}
        }
    }
    Ok(())
}

fn lease_into(buf: &[u8], into: &mut Lease) -> Result<(), WireError> {
    const M: &str = "lease";
    let (mut holder, mut priority, mut binding) = (false, 0, None);
    into.channel = ChannelId(0);
    into.token = LeaseToken(0);
    into.ttl_ns = 0;
    into.granted_ns = 0;
    into.expires_ns = 0;
    into.key_id = 0;
    let mut fields = Fields::new(buf, M);
    while let Some((tag, value)) = fields.next_field()? {
        match tag {
            1 => into.channel = ChannelId(value.u32(M)?),
            2 => {
                into.holder.assign(value.bytes(M)?).map_err(unread)?;
                holder = true;
            }
            3 => priority = value.i32(M)?,
            4 => into.token = LeaseToken(value.u64(M)?),
            5 => into.ttl_ns = value.u64(M)?,
            6 => into.granted_ns = value.i64(M)?,
            7 => into.expires_ns = value.i64(M)?,
            8 => binding = Some(value.bytes(M)?),
            9 => into.key_id = value.u32(M)?,
            _ => {}
        }
    }
    if !holder {
        return Err(unread(Unread::Malformed("lease holder")));
    }
    into.priority = match pb::PriorityClass::try_from(priority) {
        Ok(pb::PriorityClass::Scripted) => PriorityClass::Scripted,
        Ok(pb::PriorityClass::Model) => PriorityClass::Model,
        Ok(pb::PriorityClass::Operator) => PriorityClass::Operator,
        Ok(pb::PriorityClass::Safety) => PriorityClass::Safety,
        _ => return Err(unread(Unread::Malformed("lease priority"))),
    };
    into.binding = fingerprint(binding, "lease binding")?;
    Ok(())
}

fn clock_into(buf: &[u8], into: &mut ClockMapping) -> Result<(), WireError> {
    const M: &str = "clock mapping";
    let mut participant = false;
    into.offset_ns = 0;
    into.error_ns = 0;
    into.drift_bound_ppm = 0.0;
    into.rtt_ns = 0;
    into.sample_ns = 0;
    let mut fields = Fields::new(buf, M);
    while let Some((tag, value)) = fields.next_field()? {
        match tag {
            1 => {
                into.participant.assign(value.bytes(M)?).map_err(unread)?;
                participant = true;
            }
            2 => into.offset_ns = value.i64(M)?,
            3 => into.error_ns = value.u64(M)?,
            4 => into.drift_bound_ppm = value.f64(M)?,
            5 => into.rtt_ns = value.u64(M)?,
            6 => into.sample_ns = value.i64(M)?,
            _ => {}
        }
    }
    if !participant {
        return Err(unread(Unread::Malformed("clock participant")));
    }
    Ok(())
}

fn grid_into(buf: &[u8], into: &mut ViewGrid) -> Result<(), WireError> {
    const M: &str = "view grid";
    let mut binding = None;
    into.channel = ChannelId(0);
    into.dt = Duration::ZERO;
    let mut fields = Fields::new(buf, M);
    while let Some((tag, value)) = fields.next_field()? {
        match tag {
            1 => binding = Some(value.bytes(M)?),
            2 => into.channel = ChannelId(value.u32(M)?),
            3 => into.dt = Duration::from_nanos(value.u64(M)?),
            _ => {}
        }
    }
    into.binding = fingerprint(binding, "grid binding")?;
    Ok(())
}

fn demand_into(buf: &[u8], into: &mut Demand) -> Result<(), WireError> {
    const M: &str = "demand";
    let (mut plan, mut hz) = (0, 0.0);
    into.channel = ChannelId(0);
    into.revision = 0;
    into.retain = Duration::ZERO;
    into.acked = 0;
    let mut fields = Fields::new(buf, M);
    while let Some((tag, value)) = fields.next_field()? {
        match tag {
            1 => into.channel = ChannelId(value.u32(M)?),
            2 => into.revision = value.u64(M)?,
            3 => plan = value.i32(M)?,
            7 => hz = value.f64(M)?,
            8 => into.retain = Duration::from_nanos(value.u64(M)?),
            9 => into.acked = value.u64(M)?,
            _ => {}
        }
    }
    into.plan = match pb::PlanKind::try_from(plan) {
        Ok(pb::PlanKind::Every) => Plan::Every,
        Ok(pb::PlanKind::Latest) => Plan::Latest { hz },
        _ => return Err(unread(Unread::Malformed("demand plan"))),
    };
    Ok(())
}

fn member_into(buf: &[u8], into: &mut Member) -> Result<(), WireError> {
    const M: &str = "member";
    let mut id = false;
    into.generation = 0;
    let mut fields = Fields::new(buf, M);
    while let Some((tag, value)) = fields.next_field()? {
        match tag {
            1 => {
                into.id.assign(value.bytes(M)?).map_err(unread)?;
                id = true;
            }
            2 => into.generation = value.u32(M)?,
            _ => {}
        }
    }
    if !id {
        return Err(unread(Unread::Malformed("member id")));
    }
    Ok(())
}

fn routes_into(buf: &[u8], into: &mut DatagramRoutes) -> Result<(), WireError> {
    const M: &str = "datagram routes";
    let mut n = 0;
    let mut fields = Fields::new(buf, M);
    while let Some((tag, _)) = fields.next_field()? {
        n += usize::from(tag == 2);
    }
    let total = into.routes.len().saturating_add(n);
    room(&mut into.routes, total, MAX_VIEW_ROUTES, "datagram routes")?;
    let mut fields = Fields::new(buf, M);
    while let Some((tag, value)) = fields.next_field()? {
        match tag {
            1 => into.probe = route_id(value.u32(M)?)?,
            2 => {
                let route = route_from(value.bytes(M)?)?;
                if into.routes.len() == into.routes.capacity() {
                    return Err(unread(Unread::NoMemory));
                }
                into.routes.push(route);
            }
            _ => {}
        }
    }
    Ok(())
}

fn route_from(buf: &[u8]) -> Result<DatagramRoute, WireError> {
    const M: &str = "datagram route";
    let mut route = DatagramRoute {
        route: 0,
        channel: ChannelId(0),
        up: false,
    };
    let mut fields = Fields::new(buf, M);
    while let Some((tag, value)) = fields.next_field()? {
        match tag {
            1 => route.route = route_id(value.u32(M)?)?,
            2 => route.channel = ChannelId(value.u32(M)?),
            3 => route.up = value.bool(M)?,
            _ => {}
        }
    }
    Ok(route)
}

fn route_id(raw: u32) -> Result<u16, WireError> {
    u16::try_from(raw).map_err(|_| unread(Unread::Malformed("datagram route id")))
}

fn reason_into(buf: &[u8], into: &mut Reason) -> Result<(), WireError> {
    const M: &str = "reason";
    let text = |s: &mut String, value: Value<'_>| -> Result<(), WireError> {
        let new = value.str(M)?;
        if new.len() > MAX_VIEW_TEXT {
            return Err(unread(Unread::TooLarge("reason")));
        }
        s.clear();
        s.try_reserve_exact(new.len())
            .map_err(|_| unread(Unread::NoMemory))?;
        s.push_str(new);
        Ok(())
    };
    let mut fields = Fields::new(buf, M);
    while let Some((tag, value)) = fields.next_field()? {
        match tag {
            1 => into.code = reason_code_from_pb(value.i32(M)?),
            2 => text(&mut into.by, value)?,
            3 => text(&mut into.detail, value)?,
            4 => into.at_ns = value.i64(M)?,
            5 => into.revision = value.u64(M)?,
            6 => into.verified = value.bool(M)?,
            _ => {}
        }
    }
    Ok(())
}
