//! The lobby and scenes: assignments, host starts, run states, places and
//! scene info.

use super::convert::{
    bad_enum, bytes32, collect, fingerprint_from_pb, priority_from_pb, priority_to_pb,
};
use super::ctl::{error_from_pb, error_to_pb};
use super::pb;
use super::state::{reason_from_pb, reason_to_pb};
use crate::prelude::*;

use bytes::Bytes;

use crate::wire::error::WireError;
use crate::wire::ids::ParticipantId;
use crate::wire::scene::{
    Assignment, DeviceRecord, HostOrder, HostStarted, Join, MemberStage, MemberState, Place, Pose,
    RobotPin, RunStage, RunState, SceneInfo, StopOrder,
};

pub(super) fn board_id_to_pb(id: Option<&[u8; 8]>) -> Bytes {
    id.map_or_else(Bytes::new, |id| Bytes::copy_from_slice(id))
}

pub(super) fn board_id_from_pb(
    what: &'static str,
    raw: &[u8],
) -> Result<Option<[u8; 8]>, WireError> {
    if raw.is_empty() {
        return Ok(None);
    }
    raw.try_into()
        .map(Some)
        .map_err(|_| WireError::malformed(what, format!("a {}-byte board_id", raw.len())))
}

/// An optional 32-byte key: empty is none.
fn key32_from_pb(what: &'static str, raw: &[u8]) -> Result<Option<[u8; 32]>, WireError> {
    (!raw.is_empty())
        .then(|| bytes32(what, "host_key", raw))
        .transpose()
}

pub(super) fn assignment_to_pb(a: &Assignment) -> pb::Assignment {
    use pb::assignment::Order;
    let order = match a {
        Assignment::Join(j) => Order::Join(pb::Join {
            run: j.run.clone(),
            session: j.session.to_string(),
            robot: j.robot.to_string(),
            addresses: j.addresses.clone(),
            host_key: Bytes::copy_from_slice(&j.host_key),
            host_id: j.host_id.clone(),
            tenant: j.tenant.clone(),
        }),
        Assignment::Host(h) => Order::Host(pb::HostOrder {
            run: h.run.clone(),
            session: h.session.to_string(),
            scene_toml: h.scene_toml.clone(),
            devices: h
                .devices
                .iter()
                .map(|d| pb::DeviceRecord {
                    id: d.id.clone(),
                    public_key: Bytes::copy_from_slice(&d.public_key),
                    max_priority: d.max_priority.map_or(0, |p| priority_to_pb(p) as i32),
                    parts: d.parts.clone(),
                    verbs: d.verbs.clone(),
                })
                .collect(),
            pins: h
                .pins
                .iter()
                .map(|p| pb::RobotPin {
                    robot: p.robot.to_string(),
                    device: p.device.clone(),
                    board_id: board_id_to_pb(p.board_id.as_ref()),
                })
                .collect(),
        }),
        Assignment::Stop(s) => Order::Stop(pb::StopOrder {
            run: s.run.clone(),
            reason: Some(reason_to_pb(&s.reason)),
        }),
    };
    pb::Assignment { order: Some(order) }
}

pub(super) fn assignment_from_pb(a: pb::Assignment) -> Result<Assignment, WireError> {
    use pb::assignment::Order;
    Ok(
        match a
            .order
            .ok_or_else(|| WireError::malformed("assignment", "no order"))?
        {
            Order::Join(j) => Assignment::Join(Join {
                run: j.run,
                session: j.session.parse()?,
                robot: ParticipantId::new(j.robot)?,
                addresses: j.addresses,
                host_key: bytes32("join", "host_key", &j.host_key)?,
                host_id: j.host_id,
                tenant: j.tenant,
            }),
            Order::Host(h) => Assignment::Host(HostOrder {
                run: h.run,
                session: h.session.parse()?,
                scene_toml: h.scene_toml,
                devices: collect(h.devices, |d| {
                    Ok::<_, WireError>(DeviceRecord {
                        id: d.id,
                        public_key: bytes32("device record", "public_key", &d.public_key)?,
                        max_priority: match d.max_priority {
                            0 => None,
                            raw => Some(priority_from_pb(raw)?),
                        },
                        parts: d.parts,
                        verbs: d.verbs,
                    })
                })?,
                pins: collect(h.pins, |p| {
                    Ok::<_, WireError>(RobotPin {
                        robot: ParticipantId::new(p.robot)?,
                        device: p.device,
                        board_id: board_id_from_pb("robot pin", &p.board_id)?,
                    })
                })?,
            }),
            Order::Stop(s) => Assignment::Stop(StopOrder {
                run: s.run,
                reason: s.reason.map(reason_from_pb).unwrap_or_default(),
            }),
        },
    )
}

pub(super) fn host_started_to_pb(h: &HostStarted) -> pb::HostStarted {
    let (host_key, addresses, error) = match &h.result {
        Ok((key, addresses)) => (Bytes::copy_from_slice(key), addresses.clone(), None),
        Err(err) => (Bytes::new(), Vec::new(), Some(error_to_pb(err))),
    };
    pb::HostStarted {
        run: h.run.clone(),
        host_key,
        addresses,
        error,
    }
}

pub(super) fn host_started_from_pb(h: pb::HostStarted) -> Result<HostStarted, WireError> {
    let result = match h.error {
        Some(err) => Err(error_from_pb(err)),
        None => Ok((
            bytes32("host started", "host_key", &h.host_key)?,
            h.addresses,
        )),
    };
    Ok(HostStarted { run: h.run, result })
}

const RUN_STAGES: [(RunStage, pb::RunStage); 11] = [
    (RunStage::Checking, pb::RunStage::Checking),
    (RunStage::Hosting, pb::RunStage::Hosting),
    (RunStage::Assigning, pb::RunStage::Assigning),
    (RunStage::Declaring, pb::RunStage::Declaring),
    (RunStage::Binding, pb::RunStage::Binding),
    (RunStage::Arming, pb::RunStage::Arming),
    (RunStage::Running, pb::RunStage::Running),
    (RunStage::Ending, pb::RunStage::Ending),
    (RunStage::Ended, pb::RunStage::Ended),
    (RunStage::Failed, pb::RunStage::Failed),
    (RunStage::Rejoining, pb::RunStage::Rejoining),
];

const MEMBER_STAGES: [(MemberStage, pb::MemberStage); 7] = [
    (MemberStage::Waiting, pb::MemberStage::Waiting),
    (MemberStage::Assigned, pb::MemberStage::Assigned),
    (MemberStage::Joined, pb::MemberStage::Joined),
    (MemberStage::Declared, pb::MemberStage::Declared),
    (MemberStage::Bound, pb::MemberStage::Bound),
    (MemberStage::Lost, pb::MemberStage::Lost),
    (MemberStage::Returned, pb::MemberStage::Returned),
];

/// The wire value of `value` in `table` (every value is listed).
fn enum_to_pb<T: PartialEq, P: Copy + Into<i32>>(table: &[(T, P)], value: &T) -> i32 {
    table
        .iter()
        .find(|(v, _)| v == value)
        .map_or(0, |(_, pb)| (*pb).into())
}

fn enum_from_pb<T: Copy, P: Copy + Into<i32>>(
    table: &[(T, P)],
    what: &'static str,
    raw: i32,
) -> Result<T, WireError> {
    table
        .iter()
        .find(|(_, pb)| (*pb).into() == raw)
        .map(|(v, _)| *v)
        .ok_or_else(|| bad_enum(what, raw))
}

pub(super) fn run_state_to_pb(r: &RunState) -> pb::RunState {
    pb::RunState {
        run: r.run.clone(),
        version: r.version,
        stage: enum_to_pb(&RUN_STAGES, &r.stage),
        members: r
            .members
            .iter()
            .map(|m| pb::MemberState {
                name: m.name.to_string(),
                stage: enum_to_pb(&MEMBER_STAGES, &m.stage),
                error: m.error.as_ref().map(error_to_pb),
            })
            .collect(),
        error: r.error.as_ref().map(error_to_pb),
        session: r.session.map(|s| s.to_string()).unwrap_or_default(),
        addresses: r.addresses.clone(),
        host_key: r
            .host_key
            .map_or_else(Bytes::new, |k| Bytes::copy_from_slice(&k)),
        scene_fingerprint: r
            .scene_fingerprint
            .map_or_else(Bytes::new, |f| Bytes::copy_from_slice(f.as_bytes())),
    }
}

pub(super) fn run_state_from_pb(r: pb::RunState) -> Result<RunState, WireError> {
    Ok(RunState {
        run: r.run,
        version: r.version,
        stage: enum_from_pb(&RUN_STAGES, "run stage", r.stage)?,
        members: collect(r.members, |m| {
            Ok::<_, WireError>(MemberState {
                name: ParticipantId::new(m.name)?,
                stage: enum_from_pb(&MEMBER_STAGES, "member stage", m.stage)?,
                error: m.error.map(error_from_pb),
            })
        })?,
        error: r.error.map(error_from_pb),
        session: (!r.session.is_empty())
            .then(|| r.session.parse())
            .transpose()?,
        addresses: r.addresses,
        host_key: key32_from_pb("run state", &r.host_key)?,
        scene_fingerprint: (!r.scene_fingerprint.is_empty())
            .then(|| fingerprint_from_pb("run state", &r.scene_fingerprint))
            .transpose()?,
    })
}

pub(super) fn place_to_pb(p: &Place) -> pb::Place {
    pb::Place {
        robot: p.robot.to_string(),
        at: p.at.clone(),
        from: p.from.clone(),
        frame: p.frame.clone(),
        pose: Some(pb::Pose {
            xyz: p.pose.xyz.to_vec(),
            quat_xyzw: p.pose.quat_xyzw.to_vec(),
        }),
    }
}

pub(super) fn place_from_pb(p: pb::Place) -> Result<Place, WireError> {
    let pose = p
        .pose
        .ok_or_else(|| WireError::malformed("place", "no pose"))?;
    Ok(Place {
        robot: ParticipantId::new(p.robot)?,
        at: p.at,
        from: p.from,
        frame: p.frame,
        pose: Pose {
            xyz: pose.xyz[..]
                .try_into()
                .map_err(|_| WireError::malformed("pose", "xyz is not 3 values"))?,
            quat_xyzw: pose.quat_xyzw[..]
                .try_into()
                .map_err(|_| WireError::malformed("pose", "quat_xyzw is not 4 values"))?,
        },
    })
}

pub(super) fn scene_info_to_pb(s: &SceneInfo) -> pb::SceneInfo {
    pb::SceneInfo {
        run: s.run.clone(),
        name: s.name.clone(),
        fingerprint: Bytes::copy_from_slice(s.fingerprint.as_bytes()),
        closing: s.closing.as_ref().map(reason_to_pb),
        rejoin_until_ns: s.rejoin_until_ns.unwrap_or(0),
    }
}

pub(super) fn scene_info_from_pb(s: pb::SceneInfo) -> Result<SceneInfo, WireError> {
    Ok(SceneInfo {
        run: s.run,
        name: s.name,
        fingerprint: fingerprint_from_pb("scene info", &s.fingerprint)?,
        closing: s.closing.map(reason_from_pb),
        rejoin_until_ns: (s.rejoin_until_ns != 0).then_some(s.rejoin_until_ns),
    })
}
