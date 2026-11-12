//! Control messages: verbs, the host's replies (with or without the
//! state or a device view) and e-stop signal bodies.

use super::bound;
use super::convert::{
    channels, collect, fingerprint_from_pb, priority_from_pb, priority_to_pb, role_from_pb,
    role_to_pb,
};
use super::keys::{MAX_KEYS, MIN_KEY, decode_key, key_from_pb, key_to_pb};
use super::pb;
use super::reader::Fields;
use super::scene::{
    assignment_from_pb, assignment_to_pb, board_id_from_pb, board_id_to_pb, host_started_from_pb,
    host_started_to_pb, run_state_from_pb, run_state_to_pb,
};
use super::state::{
    contract_from_pb, contract_to_pb, grid_from_pb, grid_to_pb, lease_from_pb, lease_to_pb,
    reason_code_from_pb, reason_code_to_pb, reason_from_pb, reason_to_pb, state_from_pb,
    state_to_pb,
};
use super::tick::{chunk_ref_from_pb, chunk_ref_to_pb, watermark_from_pb, watermark_to_pb};
use super::view::{decode_view_into, view_to_pb};
use crate::prelude::*;

use bytes::Bytes;
use prost::Message as _;

use crate::wire::authority::{Command, CtlError, CtlReply, CtlStatus, EstopSignal, Request};
use crate::wire::error::{Unread, WireError};
use crate::wire::ids::{ChannelId, Epoch, LeaseToken, ParticipantId};
use crate::wire::key::KeysAck;
use crate::wire::scene::LobbyWait;
use crate::wire::view::{DeviceView, ViewReply};

/// Encodes a control verb.
pub fn encode_request(request: &Request) -> Vec<u8> {
    request_to_pb(request).encode_to_vec()
}

/// Decodes a control verb.
pub fn decode_request(buf: &[u8]) -> Result<Request, WireError> {
    request_from_pb(pb::CtlRequest::decode(buf)?)
}

/// Encodes the host's answer.
pub fn encode_reply(reply: &CtlReply) -> Vec<u8> {
    let (status, message) = match &reply.result {
        Ok(()) => (pb::CtlStatus::Ok, String::new()),
        Err(err) => (status_to_pb(err.status), err.message.clone()),
    };
    pb::CtlReply {
        status: status as i32,
        message,
        state: Some(state_to_pb(&reply.state)),
        leases: reply.leases.iter().map(lease_to_pb).collect(),
        request_id: reply.request_id,
        keys: reply.keys.iter().map(key_to_pb).collect(),
        assignment: reply.assignment.as_ref().map(assignment_to_pb),
        run: reply.run.as_ref().map(run_state_to_pb),
        view: None,
    }
    .encode_to_vec()
}

/// Encodes the host's answer to a robot that takes its device view: its
/// view in `view`, and no `state`.
pub fn encode_view_reply(reply: &ViewReply) -> Vec<u8> {
    let (status, message) = match &reply.result {
        Ok(()) => (pb::CtlStatus::Ok, String::new()),
        Err(err) => (status_to_pb(err.status), err.message.clone()),
    };
    pb::CtlReply {
        status: status as i32,
        message,
        state: None,
        leases: reply.leases.iter().map(lease_to_pb).collect(),
        request_id: reply.request_id,
        keys: reply.keys.iter().map(key_to_pb).collect(),
        assignment: reply.assignment.as_ref().map(assignment_to_pb),
        run: reply.run.as_ref().map(run_state_to_pb),
        view: reply.view.as_ref().map(view_to_pb),
    }
    .encode_to_vec()
}

/// Decodes any host answer as a robot that takes its device view does:
/// `state`, if present, is skipped unread.
///
/// # Errors
/// [`WireError`]: what [`decode_view_reply_into`] refuses.
pub fn decode_view_reply(buf: &[u8]) -> Result<ViewReply, WireError> {
    let mut view = DeviceView::empty(ParticipantId::unset());
    let (mut reply, carried) = decode_view_reply_into(buf, &mut view, true)?;
    reply.view = carried.then_some(view);
    Ok(reply)
}

/// The most leases one reply grants (one per part).
const MAX_GRANTED: usize = 256;
/// The fewest bytes a lease encodes to: a holder and a binding, tagged.
const MIN_LEASE: usize = 13;
/// The longest refusal message a robot keeps.
const MAX_MESSAGE: usize = 1024;

/// [`decode_view_reply`] with the reply's view decoded into `view`
/// ([`decode_view_into`], its parts only when `parts`): the reply without
/// it, and whether it carried one. Read field by field: the `state` is
/// skipped unread, so a board never links its decoder; leases and keys are
/// decoded one at a time, each capped, into vectors allocated once and
/// fallibly; an assignment or a run is capped before it is decoded.
///
/// # Errors
/// [`WireError`]: malformed, or over a cap; `view` is then unspecified.
pub fn decode_view_reply_into(
    buf: &[u8],
    view: &mut DeviceView,
    parts: bool,
) -> Result<(ViewReply, bool), WireError> {
    const M: &str = "ctl reply";
    let (mut status, mut message, mut request_id) = (0, "", 0);
    let (mut carried, mut assignment, mut run) = (None, None, None);
    let mut fields = Fields::new(buf, M);
    while let Some((tag, value)) = fields.next_field()? {
        match tag {
            1 => status = value.i32(M)?,
            2 => message = value.str(M)?,
            5 => request_id = value.u64(M)?,
            7 => assignment = Some(value.bytes(M)?),
            8 => run = Some(value.bytes(M)?),
            9 if carried.is_some() => {
                return Err(WireError::unread(M, Unread::Malformed("two views")));
            }
            9 => carried = Some(value.bytes(M)?),
            _ => {}
        }
    }
    let leases = bound::each(buf, M, (4, MAX_GRANTED, MIN_LEASE), |l| {
        lease_from_pb(pb::Lease::decode(l)?)
    })?;
    let keys = bound::each(buf, M, (6, MAX_KEYS, MIN_KEY), decode_key)?;
    let assignment = match assignment {
        Some(a) => {
            bound::check(a, &bound::ASSIGNMENT)?;
            Some(assignment_from_pb(pb::Assignment::decode(a)?)?)
        }
        None => None,
    };
    let run = match run {
        Some(r) => {
            bound::check(r, &bound::RUN_STATE)?;
            Some(run_state_from_pb(pb::RunState::decode(r)?)?)
        }
        None => None,
    };
    if message.len() > MAX_MESSAGE {
        return Err(WireError::unread(M, Unread::TooLarge("message")));
    }
    let mut text = String::new();
    text.try_reserve_exact(message.len())
        .map_err(|_| WireError::unread(M, Unread::NoMemory))?;
    text.push_str(message);
    if let Some(carried) = carried {
        decode_view_into(carried, view, parts)?;
    }
    let reply = ViewReply {
        result: status_from_pb(status, text),
        view: None,
        leases,
        request_id,
        keys,
        assignment,
        run,
    };
    Ok((reply, carried.is_some()))
}

/// The unsealed `UNAUTHENTICATED` refusal `buf` is, if it is one: only its
/// status and message are read (at most 256 bytes of it), whatever else it
/// holds, and nothing is allocated for anything else. Anyone can answer
/// `ctl`, so this reads what may be a stranger's bytes.
pub fn decode_unauthenticated(buf: &[u8]) -> Option<CtlError> {
    const M: &str = "ctl reply";
    const MAX_MESSAGE: usize = 256;
    let (mut status, mut message) = (0, "");
    let mut fields = Fields::new(buf, M);
    while let Some((tag, value)) = fields.next_field().ok()? {
        match tag {
            1 => status = value.i32(M).ok()?,
            2 => message = value.str(M).ok()?,
            _ => {}
        }
    }
    if pb::CtlStatus::try_from(status) != Ok(pb::CtlStatus::Unauthenticated) {
        return None;
    }
    let mut end = message.len().min(MAX_MESSAGE);
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    let mut text = String::new();
    text.try_reserve_exact(end).ok()?;
    text.push_str(&message[..end]);
    Some(CtlError {
        status: CtlStatus::Unauthenticated,
        message: text,
    })
}

/// `CtlReply` without its `view` (field 9), which only robots that take
/// their device view get: what [`decode_reply`] decodes, so the browser's
/// binding never links the view's decoder.
#[derive(Clone, PartialEq, prost::Message)]
struct ReplyWithoutView {
    #[prost(enumeration = "pb::CtlStatus", tag = "1")]
    status: i32,
    #[prost(string, tag = "2")]
    message: String,
    #[prost(message, optional, tag = "3")]
    state: Option<pb::AuthorityState>,
    #[prost(message, repeated, tag = "4")]
    leases: Vec<pb::Lease>,
    #[prost(uint64, tag = "5")]
    request_id: u64,
    #[prost(message, repeated, tag = "6")]
    keys: Vec<pb::Key>,
    #[prost(message, optional, tag = "7")]
    assignment: Option<pb::Assignment>,
    #[prost(message, optional, tag = "8")]
    run: Option<pb::RunState>,
}

/// Decodes the host's answer (a `view`, if present, is skipped unread).
pub fn decode_reply(buf: &[u8]) -> Result<CtlReply, WireError> {
    let reply = ReplyWithoutView::decode(buf)?;
    let state = reply
        .state
        .ok_or_else(|| WireError::malformed("ctl reply", "no state"))?;
    Ok(CtlReply {
        result: status_from_pb(reply.status, reply.message),
        state: state_from_pb(state)?,
        leases: collect(reply.leases, lease_from_pb)?,
        request_id: reply.request_id,
        keys: collect(reply.keys, key_from_pb)?,
        assignment: reply.assignment.map(assignment_from_pb).transpose()?,
        run: reply.run.map(run_state_from_pb).transpose()?,
    })
}

/// Encodes an e-stop signal's body.
pub fn encode_estop_signal(signal: &EstopSignal) -> Vec<u8> {
    pb::EstopSignal {
        reason: signal.reason.clone(),
        sent_ns: signal.sent_ns,
    }
    .encode_to_vec()
}

/// Decodes an e-stop signal's body.
pub fn decode_estop_signal(buf: &[u8]) -> Result<EstopSignal, WireError> {
    let s = pb::EstopSignal::decode(buf)?;
    Ok(EstopSignal {
        reason: s.reason,
        sent_ns: s.sent_ns,
    })
}

pub(super) fn error_to_pb(err: &CtlError) -> pb::CtlError {
    pb::CtlError {
        status: status_to_pb(err.status) as i32,
        message: err.message.clone(),
    }
}

pub(super) fn error_from_pb(err: pb::CtlError) -> CtlError {
    status_from_pb(err.status, err.message)
        .err()
        .unwrap_or_else(|| CtlError::new(CtlStatus::InvalidRequest, "a refusal with OK"))
}

pub(super) fn status_to_pb(status: CtlStatus) -> pb::CtlStatus {
    match status {
        CtlStatus::InvalidRequest => pb::CtlStatus::InvalidRequest,
        CtlStatus::NotAttached => pb::CtlStatus::NotAttached,
        CtlStatus::Forbidden => pb::CtlStatus::Forbidden,
        CtlStatus::InvalidTransition => pb::CtlStatus::InvalidTransition,
        CtlStatus::EstopLatched => pb::CtlStatus::EstopLatched,
        CtlStatus::StaleEpoch => pb::CtlStatus::StaleEpoch,
        CtlStatus::StaleToken => pb::CtlStatus::StaleToken,
        CtlStatus::LeaseBusy => pb::CtlStatus::LeaseBusy,
        CtlStatus::UnknownPath => pb::CtlStatus::UnknownPath,
        CtlStatus::Unsupported => pb::CtlStatus::Unsupported,
        CtlStatus::Incompatible => pb::CtlStatus::Incompatible,
        CtlStatus::GroupBusy => pb::CtlStatus::GroupBusy,
        CtlStatus::NotBound => pb::CtlStatus::NotBound,
        CtlStatus::NotReady => pb::CtlStatus::NotReady,
        CtlStatus::Unauthenticated => pb::CtlStatus::Unauthenticated,
        CtlStatus::NotAuthorized => pb::CtlStatus::NotAuthorized,
        CtlStatus::Expired => pb::CtlStatus::Expired,
        CtlStatus::RobotBusy => pb::CtlStatus::RobotBusy,
        CtlStatus::NotPresent => pb::CtlStatus::NotPresent,
    }
}

/// `Ok(())` for OK; anything else (an unknown, newer status included) is a
/// refusal.
pub(super) fn status_from_pb(raw: i32, message: String) -> Result<(), CtlError> {
    let status = match pb::CtlStatus::try_from(raw) {
        Ok(pb::CtlStatus::Ok) => return Ok(()),
        Ok(pb::CtlStatus::NotAttached) => CtlStatus::NotAttached,
        Ok(pb::CtlStatus::Forbidden) => CtlStatus::Forbidden,
        Ok(pb::CtlStatus::InvalidTransition) => CtlStatus::InvalidTransition,
        Ok(pb::CtlStatus::EstopLatched) => CtlStatus::EstopLatched,
        Ok(pb::CtlStatus::StaleEpoch) => CtlStatus::StaleEpoch,
        Ok(pb::CtlStatus::StaleToken) => CtlStatus::StaleToken,
        Ok(pb::CtlStatus::LeaseBusy) => CtlStatus::LeaseBusy,
        Ok(pb::CtlStatus::UnknownPath) => CtlStatus::UnknownPath,
        Ok(pb::CtlStatus::Unsupported) => CtlStatus::Unsupported,
        Ok(pb::CtlStatus::Incompatible) => CtlStatus::Incompatible,
        Ok(pb::CtlStatus::GroupBusy) => CtlStatus::GroupBusy,
        Ok(pb::CtlStatus::NotBound) => CtlStatus::NotBound,
        Ok(pb::CtlStatus::NotReady) => CtlStatus::NotReady,
        Ok(pb::CtlStatus::Unauthenticated) => CtlStatus::Unauthenticated,
        Ok(pb::CtlStatus::NotAuthorized) => CtlStatus::NotAuthorized,
        Ok(pb::CtlStatus::Expired) => CtlStatus::Expired,
        Ok(pb::CtlStatus::RobotBusy) => CtlStatus::RobotBusy,
        Ok(pb::CtlStatus::NotPresent) => CtlStatus::NotPresent,
        Ok(pb::CtlStatus::InvalidRequest | pb::CtlStatus::Unspecified) | Err(_) => {
            CtlStatus::InvalidRequest
        }
    };
    Err(CtlError { status, message })
}

fn request_to_pb(r: &Request) -> pb::CtlRequest {
    use pb::ctl_request::Verb;
    let verb = match &r.command {
        Command::Attach {
            name,
            roles,
            priority,
            protocol_minor,
            supports,
            requires,
            workflow_edition,
            workflow_editions,
            estop,
        } => Verb::Attach(pb::Attach {
            estop: estop.as_ref().map(|reason| pb::LocalEstop {
                latched: true,
                reason: Some(reason_to_pb(reason)),
            }),
            name: name.clone(),
            roles: roles.iter().map(|r| role_to_pb(*r) as i32).collect(),
            priority: priority_to_pb(*priority) as i32,
            protocol_minor: *protocol_minor,
            supports: supports.clone(),
            requires: requires.clone(),
            workflow_edition: workflow_edition.clone(),
            workflow_editions: workflow_editions.clone(),
        }),
        Command::Detach => Verb::Detach(pb::Detach {}),
        Command::Arm => Verb::Arm(pb::Arm {
            robot: String::new(),
        }),
        Command::Rearm { robot } => Verb::Arm(pb::Arm {
            robot: robot.to_string(),
        }),
        Command::Disarm => Verb::Disarm(pb::Disarm {}),
        Command::DeclareParts {
            spec_version,
            robot_json,
            seal,
            board_id,
        } => Verb::DeclareParts(pb::DeclareParts {
            spec_version: *spec_version,
            robot_json: robot_json.clone(),
            seal: seal.clone(),
            board_id: board_id_to_pb(board_id.as_ref()),
        }),
        Command::DeclareBinding {
            fingerprint,
            revision,
            table_json,
            channels: bound,
            replan_hz,
            contracts,
            grids,
        } => Verb::DeclareBinding(pb::DeclareBinding {
            fingerprint: Bytes::copy_from_slice(fingerprint.as_bytes()),
            revision: *revision,
            table_json: table_json.clone(),
            channels: channels(bound),
            replan_hz: *replan_hz,
            contracts: contracts.iter().map(contract_to_pb).collect(),
            grids: grids.iter().map(grid_to_pb).collect(),
        }),
        Command::DemandAck { channel, revision } => Verb::DemandAck(pb::DemandAck {
            channel: channel.0,
            revision: *revision,
        }),
        Command::GrantLease {
            channels: grant,
            release,
            binding,
            ttl_ns,
        } => Verb::GrantLease(pb::GrantLease {
            channels: channels(grant),
            release: channels(release),
            binding: Bytes::copy_from_slice(binding.as_bytes()),
            ttl_ns: ttl_ns.unwrap_or(0),
        }),
        Command::ReleaseLease {
            channels: release,
            token,
            to,
        } => Verb::ReleaseLease(pb::ReleaseLease {
            channels: channels(release),
            token: token.0,
            to: to.as_ref().map(ToString::to_string).unwrap_or_default(),
        }),
        Command::Estop {
            reason,
            code,
            origin,
            at_ns,
        } => Verb::Estop(pb::Estop {
            reason: reason.clone(),
            code: reason_code_to_pb(*code),
            origin: origin.as_ref().map(ToString::to_string).unwrap_or_default(),
            at_ns: *at_ns,
        }),
        Command::Clear => Verb::Clear(pb::Clear {}),
        Command::EnterSafe { reason, code } => Verb::EnterSafe(pb::EnterSafe {
            reason: reason.clone(),
            code: reason_code_to_pb(*code),
        }),
        Command::Ready { tick, watermarks } => Verb::Ready(pb::Ready {
            tick: *tick,
            watermarks: watermarks.iter().map(watermark_to_pb).collect(),
        }),
        Command::Decided { tick, chunks } => Verb::Decided(pb::Decided {
            tick: *tick,
            chunks: chunks.iter().map(chunk_ref_to_pb).collect(),
        }),
        Command::KeysAck(ack) => Verb::KeysAck(pb::KeysAck {
            installed: ack.installed.clone(),
            sending: ack.sending.clone(),
            retired: ack.retired.clone(),
        }),
        Command::GetKeys => Verb::GetKeys(pb::GetKeys {}),
        Command::LobbyWait(wait) => Verb::LobbyWait(pb::LobbyWait {
            run: wait.run.clone(),
            last: wait.last.as_ref().map(error_to_pb),
        }),
        Command::RunScene {
            scene_toml,
            manual_arm,
        } => Verb::RunScene(pb::RunScene {
            scene_toml: scene_toml.clone(),
            manual_arm: *manual_arm,
        }),
        Command::SceneStatus { run, after } => Verb::SceneStatus(pb::SceneStatus {
            run: run.clone(),
            after: *after,
        }),
        Command::StopScene { run } => Verb::StopScene(pb::StopScene { run: run.clone() }),
        Command::HostStarted(started) => Verb::HostStarted(host_started_to_pb(started)),
        Command::ArmScene { run } => Verb::ArmScene(pb::ArmScene { run: run.clone() }),
        Command::Keep { label } => Verb::Keep(pb::Keep {
            label: label.clone(),
        }),
    };
    pb::CtlRequest {
        participant: r.from.to_string(),
        epoch: r.epoch.map_or(0, |e| e.0),
        request_id: r.request_id,
        verb: Some(verb),
    }
}

fn request_from_pb(r: pb::CtlRequest) -> Result<Request, WireError> {
    use pb::ctl_request::Verb;
    let verb = r.verb.ok_or_else(|| {
        WireError::malformed("ctl request", "no verb (or one this peer does not know)")
    })?;
    let ids = |ids: Vec<u32>| ids.into_iter().map(ChannelId).collect::<Vec<_>>();
    let command = match verb {
        Verb::Attach(a) => Command::Attach {
            name: a.name,
            roles: collect(a.roles, role_from_pb)?,
            priority: priority_from_pb(a.priority)?,
            protocol_minor: a.protocol_minor,
            supports: a.supports,
            requires: a.requires,
            workflow_edition: a.workflow_edition,
            workflow_editions: a.workflow_editions,
            estop: a
                .estop
                .filter(|e| e.latched)
                .map(|e| e.reason.map(reason_from_pb).unwrap_or_default()),
        },
        Verb::Detach(_) => Command::Detach,
        Verb::Arm(a) if a.robot.is_empty() => Command::Arm,
        Verb::Arm(a) => Command::Rearm {
            robot: ParticipantId::new(a.robot)?,
        },
        Verb::Disarm(_) => Command::Disarm,
        Verb::DeclareParts(d) => Command::DeclareParts {
            spec_version: d.spec_version,
            robot_json: d.robot_json,
            seal: d.seal,
            board_id: board_id_from_pb("declare parts", &d.board_id)?,
        },
        Verb::DeclareBinding(b) => Command::DeclareBinding {
            fingerprint: fingerprint_from_pb("declare binding", &b.fingerprint)?,
            revision: b.revision,
            table_json: b.table_json,
            channels: ids(b.channels),
            replan_hz: b.replan_hz,
            contracts: collect(b.contracts, contract_from_pb)?,
            grids: b.grids.into_iter().map(grid_from_pb).collect(),
        },
        Verb::DemandAck(a) => Command::DemandAck {
            channel: ChannelId(a.channel),
            revision: a.revision,
        },
        Verb::GrantLease(g) => Command::GrantLease {
            channels: ids(g.channels),
            release: ids(g.release),
            binding: fingerprint_from_pb("grant lease", &g.binding)?,
            ttl_ns: (g.ttl_ns > 0).then_some(g.ttl_ns),
        },
        Verb::ReleaseLease(rl) => Command::ReleaseLease {
            channels: ids(rl.channels),
            token: LeaseToken(rl.token),
            to: (!rl.to.is_empty())
                .then(|| ParticipantId::new(rl.to))
                .transpose()?,
        },
        Verb::Estop(e) => Command::Estop {
            reason: e.reason,
            code: reason_code_from_pb(e.code),
            origin: (!e.origin.is_empty())
                .then(|| ParticipantId::new(e.origin))
                .transpose()?,
            at_ns: e.at_ns,
        },
        Verb::Clear(_) => Command::Clear,
        Verb::EnterSafe(e) => Command::EnterSafe {
            reason: e.reason,
            code: reason_code_from_pb(e.code),
        },
        Verb::Ready(ready) => Command::Ready {
            tick: ready.tick,
            watermarks: ready
                .watermarks
                .into_iter()
                .map(watermark_from_pb)
                .collect(),
        },
        Verb::Decided(decided) => Command::Decided {
            tick: decided.tick,
            chunks: decided.chunks.into_iter().map(chunk_ref_from_pb).collect(),
        },
        Verb::KeysAck(ack) => Command::KeysAck(KeysAck {
            installed: ack.installed,
            sending: ack.sending,
            retired: ack.retired,
        }),
        Verb::GetKeys(_) => Command::GetKeys,
        Verb::LobbyWait(w) => Command::LobbyWait(LobbyWait {
            run: w.run,
            last: w.last.map(error_from_pb),
        }),
        Verb::RunScene(r) => Command::RunScene {
            scene_toml: r.scene_toml,
            manual_arm: r.manual_arm,
        },
        Verb::SceneStatus(s) => Command::SceneStatus {
            run: s.run,
            after: s.after,
        },
        Verb::StopScene(s) => Command::StopScene { run: s.run },
        Verb::HostStarted(h) => Command::HostStarted(host_started_from_pb(h)?),
        Verb::ArmScene(a) => Command::ArmScene { run: a.run },
        Verb::Keep(k) => Command::Keep { label: k.label },
    };
    Ok(Request {
        from: ParticipantId::new(r.participant)?,
        epoch: (r.epoch > 0).then_some(Epoch(r.epoch)),
        request_id: r.request_id,
        command,
    })
}
