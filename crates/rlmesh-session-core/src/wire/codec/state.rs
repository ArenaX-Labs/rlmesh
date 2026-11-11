//! The authority snapshot and its parts: participants, leases, reasons,
//! clocks, bindings, contracts and demands.

use super::convert::{
    clock_mode_from_pb, clock_mode_to_pb, collect, fingerprint_from_pb, mode_from_pb, mode_to_pb,
    priority_from_pb, priority_to_pb, role_from_pb, role_to_pb,
};
use super::part::{duration_ns, part_from_pb, part_to_pb};
use super::pb;
use super::scene::{place_from_pb, place_to_pb, scene_info_from_pb, scene_info_to_pb};
use crate::prelude::*;

use core::time::Duration;

use bytes::Bytes;
use prost::Message as _;

use crate::wire::authority::{
    AuthoritySnapshot, BindingRecord, ClockMapping, HeldRobot, LastStop, Lease, LeaseEnd,
    ParticipantInfo, Reason, ReasonCode,
};
use crate::wire::contract::{ActionGrid, ContractMode, Demand, Plan, StreamContract};
use crate::wire::error::WireError;
use crate::wire::ids::{ChannelId, Epoch, LeaseToken, ParticipantId};

/// Encodes an authority snapshot.
pub fn encode_state(state: &AuthoritySnapshot) -> Vec<u8> {
    state_to_pb(state).encode_to_vec()
}

/// Decodes an authority snapshot.
pub fn decode_state(buf: &[u8]) -> Result<AuthoritySnapshot, WireError> {
    state_from_pb(pb::AuthorityState::decode(buf)?)
}

/// Encodes one controller's binding record.
pub fn encode_binding(binding: &BindingRecord) -> Vec<u8> {
    binding_to_pb(binding).encode_to_vec()
}

/// Decodes one controller's binding record.
pub fn decode_binding(buf: &[u8]) -> Result<BindingRecord, WireError> {
    binding_from_pb(pb::BindingRecord::decode(buf)?)
}

/// Every code with its wire value, so both directions read one table.
const REASON_CODES: [(ReasonCode, pb::ReasonCode); 30] = [
    (ReasonCode::Unspecified, pb::ReasonCode::Unspecified),
    (ReasonCode::Armed, pb::ReasonCode::Armed),
    (ReasonCode::Disarmed, pb::ReasonCode::Disarmed),
    (ReasonCode::FirstLease, pb::ReasonCode::FirstLease),
    (ReasonCode::LastRelease, pb::ReasonCode::LastRelease),
    (ReasonCode::EstopRemote, pb::ReasonCode::EstopRemote),
    (ReasonCode::EstopLocal, pb::ReasonCode::EstopLocal),
    (ReasonCode::EstopSignal, pb::ReasonCode::EstopSignal),
    (ReasonCode::EstopCleared, pb::ReasonCode::EstopCleared),
    (ReasonCode::HolderLost, pb::ReasonCode::HolderLost),
    (ReasonCode::HeartbeatLapse, pb::ReasonCode::HeartbeatLapse),
    (ReasonCode::QueueDry, pb::ReasonCode::QueueDry),
    (ReasonCode::EdgeLeaseLapsed, pb::ReasonCode::EdgeLeaseLapsed),
    (ReasonCode::HandoverTimeout, pb::ReasonCode::HandoverTimeout),
    (ReasonCode::DeviceWatchdog, pb::ReasonCode::DeviceWatchdog),
    (ReasonCode::RobotLost, pb::ReasonCode::RobotLost),
    (ReasonCode::HostLost, pb::ReasonCode::HostLost),
    (ReasonCode::Released, pb::ReasonCode::Released),
    (ReasonCode::Preempted, pb::ReasonCode::Preempted),
    (ReasonCode::Rebound, pb::ReasonCode::Rebound),
    (ReasonCode::VoidedBySafe, pb::ReasonCode::VoidedBySafe),
    (ReasonCode::VoidedByEstop, pb::ReasonCode::VoidedByEstop),
    (ReasonCode::VoidedByDisarm, pb::ReasonCode::VoidedByDisarm),
    (ReasonCode::VoidedByEpoch, pb::ReasonCode::VoidedByEpoch),
    (ReasonCode::PartRemoved, pb::ReasonCode::PartRemoved),
    (ReasonCode::HolderDetached, pb::ReasonCode::HolderDetached),
    (ReasonCode::Revoked, pb::ReasonCode::Revoked),
    (ReasonCode::AccessChanged, pb::ReasonCode::AccessChanged),
    (ReasonCode::SceneEnded, pb::ReasonCode::SceneEnded),
    (ReasonCode::SceneFailed, pb::ReasonCode::SceneFailed),
];

pub(super) fn reason_code_to_pb(code: ReasonCode) -> i32 {
    REASON_CODES
        .iter()
        .find(|(c, _)| *c == code)
        .map_or(0, |(_, pb)| *pb as i32)
}

/// A code this peer does not know yet (a newer minor's) reads as
/// unspecified: reasons explain, they never gate.
pub(super) fn reason_code_from_pb(raw: i32) -> ReasonCode {
    REASON_CODES
        .iter()
        .find(|(_, pb)| *pb as i32 == raw)
        .map_or(ReasonCode::Unspecified, |(c, _)| *c)
}

pub(super) fn reason_to_pb(r: &Reason) -> pb::Reason {
    pb::Reason {
        code: reason_code_to_pb(r.code),
        by: r.by.clone(),
        detail: r.detail.clone(),
        at_ns: r.at_ns,
        revision: r.revision,
        verified: r.verified,
    }
}

pub(super) fn reason_from_pb(r: pb::Reason) -> Reason {
    Reason {
        code: reason_code_from_pb(r.code),
        by: r.by,
        detail: r.detail,
        at_ns: r.at_ns,
        revision: r.revision,
        verified: r.verified,
    }
}

fn lease_end_to_pb(e: &LeaseEnd) -> pb::LeaseEnd {
    pb::LeaseEnd {
        lease: Some(lease_to_pb(&e.lease)),
        reason: Some(reason_to_pb(&e.reason)),
        by: e.by.as_ref().map(lease_to_pb),
    }
}

fn lease_end_from_pb(e: pb::LeaseEnd) -> Result<LeaseEnd, WireError> {
    Ok(LeaseEnd {
        lease: lease_from_pb(
            e.lease
                .ok_or_else(|| WireError::malformed("lease end", "no lease"))?,
        )?,
        reason: e.reason.map(reason_from_pb).unwrap_or_default(),
        by: e.by.map(lease_from_pb).transpose()?,
    })
}

fn participant_to_pb(p: &ParticipantInfo) -> pb::ParticipantInfo {
    pb::ParticipantInfo {
        id: p.id.to_string(),
        name: p.name.clone(),
        roles: p.roles.iter().map(|r| role_to_pb(*r) as i32).collect(),
        priority: priority_to_pb(p.priority) as i32,
        attached_ns: p.attached_ns,
        protocol_minor: p.protocol_minor,
        supports: p.supports.clone(),
        requires: p.requires.clone(),
        workflow_edition: p.workflow_edition.clone(),
        workflow_editions: p.workflow_editions.clone(),
        present: p.present,
        left: p.left.as_ref().map(reason_to_pb),
        device: p.device.clone(),
        generation: p.generation,
    }
}

fn participant_from_pb(p: pb::ParticipantInfo) -> Result<ParticipantInfo, WireError> {
    Ok(ParticipantInfo {
        id: ParticipantId::new(p.id)?,
        name: p.name,
        roles: collect(p.roles, role_from_pb)?,
        priority: priority_from_pb(p.priority)?,
        attached_ns: p.attached_ns,
        protocol_minor: p.protocol_minor,
        supports: p.supports,
        requires: p.requires,
        workflow_edition: p.workflow_edition,
        workflow_editions: p.workflow_editions,
        present: p.present,
        left: p.left.map(reason_from_pb),
        device: p.device,
        generation: p.generation,
    })
}

pub(super) fn lease_to_pb(l: &Lease) -> pb::Lease {
    pb::Lease {
        channel: l.channel.0,
        holder: l.holder.to_string(),
        priority: priority_to_pb(l.priority) as i32,
        token: l.token.0,
        ttl_ns: l.ttl_ns,
        granted_ns: l.granted_ns,
        expires_ns: l.expires_ns,
        binding: Bytes::copy_from_slice(l.binding.as_bytes()),
        key_id: l.key_id,
    }
}

pub(super) fn lease_from_pb(l: pb::Lease) -> Result<Lease, WireError> {
    Ok(Lease {
        channel: ChannelId(l.channel),
        holder: ParticipantId::new(l.holder)?,
        priority: priority_from_pb(l.priority)?,
        token: LeaseToken(l.token),
        ttl_ns: l.ttl_ns,
        granted_ns: l.granted_ns,
        expires_ns: l.expires_ns,
        binding: fingerprint_from_pb("lease", &l.binding)?,
        key_id: l.key_id,
    })
}

pub(super) fn clock_to_pb(c: &ClockMapping) -> pb::ClockMapping {
    pb::ClockMapping {
        participant: c.participant.to_string(),
        offset_ns: c.offset_ns,
        error_ns: c.error_ns,
        drift_bound_ppm: c.drift_bound_ppm,
        rtt_ns: c.rtt_ns,
        sample_ns: c.sample_ns,
    }
}

pub(super) fn clock_from_pb(c: pb::ClockMapping) -> Result<ClockMapping, WireError> {
    Ok(ClockMapping {
        participant: ParticipantId::new(c.participant)?,
        offset_ns: c.offset_ns,
        error_ns: c.error_ns,
        rtt_ns: c.rtt_ns,
        sample_ns: c.sample_ns,
        drift_bound_ppm: c.drift_bound_ppm,
    })
}

pub(super) fn contract_to_pb(c: &StreamContract) -> pb::StreamContract {
    pb::StreamContract {
        channel: c.channel.0,
        mode: match c.mode {
            ContractMode::Latest => pb::ContractMode::Latest,
            ContractMode::History => pb::ContractMode::History,
            ContractMode::Every => pb::ContractMode::Every,
        } as i32,
        delivery_hz: c.delivery_hz,
        history: c.history,
        max_age_ns: c.max_age.map_or(0, duration_ns),
        max_age_explicit: c.max_age_explicit,
    }
}

pub(super) fn contract_from_pb(c: pb::StreamContract) -> Result<StreamContract, WireError> {
    Ok(StreamContract {
        channel: ChannelId(c.channel),
        mode: match pb::ContractMode::try_from(c.mode) {
            Ok(pb::ContractMode::Latest) => ContractMode::Latest,
            Ok(pb::ContractMode::History) => ContractMode::History,
            Ok(pb::ContractMode::Every) => ContractMode::Every,
            _ => {
                return Err(WireError::malformed(
                    "stream contract",
                    format!("mode {}", c.mode),
                ));
            }
        },
        delivery_hz: c.delivery_hz,
        history: c.history,
        max_age: (c.max_age_ns > 0).then(|| Duration::from_nanos(c.max_age_ns)),
        max_age_explicit: c.max_age_explicit,
    })
}

pub(super) fn grid_to_pb(g: &ActionGrid) -> pb::ActionGrid {
    pb::ActionGrid {
        channel: g.channel.0,
        dt_ns: duration_ns(g.dt),
    }
}

pub(super) fn grid_from_pb(g: pb::ActionGrid) -> ActionGrid {
    ActionGrid {
        channel: ChannelId(g.channel),
        dt: Duration::from_nanos(g.dt_ns),
    }
}

pub(super) fn demand_to_pb(d: &Demand) -> pb::Demand {
    let mut out = pb::Demand {
        channel: d.channel.0,
        revision: d.revision,
        retain_ns: duration_ns(d.retain),
        acked: d.acked,
        ..pb::Demand::default()
    };
    match d.plan {
        Plan::Every => out.plan = pb::PlanKind::Every as i32,
        Plan::Latest { hz } => {
            out.plan = pb::PlanKind::Latest as i32;
            out.latest_hz = hz;
        }
    }
    out
}

pub(super) fn demand_from_pb(d: pb::Demand) -> Result<Demand, WireError> {
    let plan = match pb::PlanKind::try_from(d.plan) {
        Ok(pb::PlanKind::Every) => Plan::Every,
        Ok(pb::PlanKind::Latest) => Plan::Latest { hz: d.latest_hz },
        _ => return Err(WireError::malformed("demand", format!("plan {}", d.plan))),
    };
    Ok(Demand {
        channel: ChannelId(d.channel),
        revision: d.revision,
        plan,
        retain: Duration::from_nanos(d.retain_ns),
        acked: d.acked,
    })
}

fn binding_to_pb(b: &BindingRecord) -> pb::BindingRecord {
    pb::BindingRecord {
        participant: b.participant.to_string(),
        fingerprint: Bytes::copy_from_slice(b.fingerprint.as_bytes()),
        revision: b.revision,
        table_json: b.table_json.clone(),
        channels: b.channels.iter().map(|c| c.0).collect(),
        replan_hz: b.replan_hz,
        contracts: b.contracts.iter().map(contract_to_pb).collect(),
        grids: b.grids.iter().map(grid_to_pb).collect(),
    }
}

fn binding_from_pb(b: pb::BindingRecord) -> Result<BindingRecord, WireError> {
    Ok(BindingRecord {
        participant: ParticipantId::new(b.participant)?,
        fingerprint: fingerprint_from_pb("binding record", &b.fingerprint)?,
        revision: b.revision,
        table_json: b.table_json,
        channels: b.channels.into_iter().map(ChannelId).collect(),
        replan_hz: b.replan_hz,
        contracts: collect(b.contracts, contract_from_pb)?,
        grids: b.grids.into_iter().map(grid_from_pb).collect(),
    })
}

fn held_from_pb(h: pb::HeldRobot) -> Result<HeldRobot, WireError> {
    Ok(HeldRobot {
        robot: ParticipantId::new(h.robot)?,
        reason: h.reason.map(reason_from_pb).unwrap_or_default(),
    })
}

pub(super) fn state_to_pb(s: &AuthoritySnapshot) -> pb::AuthorityState {
    pb::AuthorityState {
        session: s.session.to_string(),
        epoch: s.epoch.0,
        revision: s.revision,
        mode: mode_to_pb(s.mode) as i32,
        estop_latched: s.estop_latched,
        roster: s.roster.iter().map(participant_to_pb).collect(),
        leases: s.leases.iter().map(lease_to_pb).collect(),
        clocks: s.clocks.iter().map(clock_to_pb).collect(),
        host_ns: s.host_ns,
        parts: s.parts.iter().map(part_to_pb).collect(),
        protocol_minor: s.protocol_minor,
        supports: s.supports.clone(),
        clock_mode: clock_mode_to_pb(s.clock_mode) as i32,
        tick_ns: s.tick_ns,
        bindings: s.bindings.iter().map(binding_to_pb).collect(),
        contract: s
            .contract
            .iter()
            .map(|(robot, fp)| (robot.to_string(), Bytes::copy_from_slice(fp.as_bytes())))
            .collect(),
        demands: s.demands.iter().map(demand_to_pb).collect(),
        mode_reason: Some(reason_to_pb(&s.mode_reason)),
        ended: s.ended.iter().map(lease_end_to_pb).collect(),
        last_stop: Some(last_stop_to_pb(s.last_stop)),
        places: s.places.iter().map(place_to_pb).collect(),
        scene: s.scene.as_deref().map(scene_info_to_pb),
        held: s
            .held
            .iter()
            .map(|h| pb::HeldRobot {
                robot: h.robot.to_string(),
                reason: Some(reason_to_pb(&h.reason)),
            })
            .collect(),
    }
}

pub(super) fn state_from_pb(s: pb::AuthorityState) -> Result<AuthoritySnapshot, WireError> {
    Ok(AuthoritySnapshot {
        session: s.session.parse()?,
        epoch: Epoch(s.epoch),
        revision: s.revision,
        mode: mode_from_pb(s.mode)?,
        estop_latched: s.estop_latched,
        roster: collect(s.roster, participant_from_pb)?,
        leases: collect(s.leases, lease_from_pb)?,
        clocks: collect(s.clocks, clock_from_pb)?,
        host_ns: s.host_ns,
        parts: collect(s.parts, part_from_pb)?,
        protocol_minor: s.protocol_minor,
        supports: s.supports,
        clock_mode: clock_mode_from_pb(s.clock_mode)?,
        tick_ns: s.tick_ns,
        bindings: collect(s.bindings, binding_from_pb)?,
        contract: s
            .contract
            .into_iter()
            .map(|(robot, fp)| {
                Ok((
                    ParticipantId::new(robot)?,
                    fingerprint_from_pb("declaration fingerprint", &fp)?,
                ))
            })
            .collect::<Result<_, WireError>>()?,
        demands: collect(s.demands, demand_from_pb)?,
        mode_reason: s.mode_reason.map(reason_from_pb).unwrap_or_default(),
        ended: collect(s.ended, lease_end_from_pb)?,
        last_stop: last_stop_from_pb(s.last_stop),
        places: collect(s.places, place_from_pb)?,
        scene: s
            .scene
            .map(|s| scene_info_from_pb(s).map(Box::new))
            .transpose()?,
        held: collect(s.held, held_from_pb)?,
    })
}

pub(super) fn last_stop_to_pb(l: LastStop) -> pb::LastStop {
    pb::LastStop {
        epoch: l.epoch.0,
        revision: l.revision,
    }
}

pub(super) fn last_stop_from_pb(l: Option<pb::LastStop>) -> LastStop {
    l.map_or_else(LastStop::default, |l| LastStop {
        epoch: Epoch(l.epoch),
        revision: l.revision,
    })
}
