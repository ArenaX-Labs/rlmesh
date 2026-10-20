//! The byte encoding: protobuf, with tensors as raw `bytes` leaves.
//!
//! This is the only module that knows how messages look on the wire (see the
//! crate docs for why). Everything else in the crate, and every crate above
//! it, speaks the plain types this module encodes and decodes.
//!
//! Every message is one protobuf message of `rlmesh.session.v2` (see
//! `proto/rlmesh/session/v2/session.proto`) per payload; the key says which.
//!
//! - Control messages ([`Request`], [`CtlReply`], [`AuthoritySnapshot`],
//!   [`Heartbeat`], [`HeartbeatEcho`]) encode to one buffer and decode from
//!   any byte slice.
//! - [`Sample`]s and [`ActionChunk`]s carry their tensors as `bytes` leaves
//!   in the channel's declared layout. [`encode_sample`] and
//!   [`encode_chunk`] emit the message as [`Encoded`] segments: small
//!   buffers for the fields and each leaf's tag and length, and every leaf as
//!   its own [`Bytes`] segment, so a transport that gathers segments (Zenoh's
//!   `ZBytes`) sends the leaves without a copy. The concatenation is ordinary
//!   protobuf that any implementation decodes. [`decode_sample`] and
//!   [`decode_chunk`] take the received buffer as [`Bytes`], and the leaves
//!   slice it.
//!
//! Leaf contents are not validated here; the layout check against the
//! channel declaration is [`ChannelSpec::check_leaf_lens`].

use bytes::Bytes;
use prost::Message as _;
use prost::encoding::{encode_varint, encoded_len_varint};

use crate::action::{ActionChunk, ObservationRef};
use crate::authority::{
    AuthoritySnapshot, ClockMapping, Command, CtlError, CtlReply, CtlStatus, Lease, Mode,
    ParticipantInfo, PriorityClass, Request, Role,
};
use crate::channel::{
    Actuation, ChannelPriority, ChannelQos, ChannelSpec, Congestion, DType, Delivery, Direction,
    Encoding, History, MergePolicy, Rate, TensorLayout,
};
use crate::error::WireError;
use crate::heartbeat::{Health, Heartbeat, HeartbeatEcho, TimingSample};
use crate::ids::{ChannelId, Epoch, LeaseToken, ParticipantId};
use crate::sample::Sample;

/// Generated `rlmesh.session.v2` messages: private, so the schema can move
/// without touching any public type.
#[allow(clippy::all, clippy::pedantic)]
mod pb {
    include!(concat!(env!("OUT_DIR"), "/rlmesh.session.v2.rs"));
}

// --- control messages --------------------------------------------------------

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
        lease: reply.lease.as_ref().map(lease_to_pb),
    }
    .encode_to_vec()
}

/// Decodes the host's answer.
pub fn decode_reply(buf: &[u8]) -> Result<CtlReply, WireError> {
    let reply = pb::CtlReply::decode(buf)?;
    let state = reply
        .state
        .ok_or_else(|| WireError::malformed("ctl reply", "no state"))?;
    Ok(CtlReply {
        result: status_from_pb(reply.status, reply.message),
        state: state_from_pb(state)?,
        lease: reply.lease.map(lease_from_pb).transpose()?,
    })
}

/// Encodes an authority snapshot.
pub fn encode_state(state: &AuthoritySnapshot) -> Vec<u8> {
    state_to_pb(state).encode_to_vec()
}

/// Decodes an authority snapshot.
pub fn decode_state(buf: &[u8]) -> Result<AuthoritySnapshot, WireError> {
    state_from_pb(pb::AuthorityState::decode(buf)?)
}

/// Encodes a heartbeat.
pub fn encode_heartbeat(hb: &Heartbeat) -> Vec<u8> {
    pb::Heartbeat {
        participant: hb.participant.to_string(),
        epoch: hb.epoch.0,
        mode: mode_to_pb(hb.mode) as i32,
        seq: hb.seq,
        t1_ns: hb.t1_ns,
        previous: hb.previous.map(timing_to_pb),
        last_seq_seen: hb
            .last_seq_seen
            .iter()
            .map(|(channel, seq)| pb::ChannelSeq {
                channel: channel.0,
                seq: *seq,
            })
            .collect(),
        last_chunk_accepted: hb.last_chunk_accepted,
        last_chunk_applied: hb.last_chunk_applied,
        queue_depth: hb.queue_depth,
        health: match hb.health {
            Health::Unknown => pb::Health::Unspecified,
            Health::Ok => pb::Health::Ok,
            Health::Degraded => pb::Health::Degraded,
            Health::Unhealthy => pb::Health::Unhealthy,
        } as i32,
    }
    .encode_to_vec()
}

/// Decodes a heartbeat.
pub fn decode_heartbeat(buf: &[u8]) -> Result<Heartbeat, WireError> {
    let hb = pb::Heartbeat::decode(buf)?;
    Ok(Heartbeat {
        participant: ParticipantId::new(hb.participant)?,
        epoch: Epoch(hb.epoch),
        mode: mode_from_pb(hb.mode)?,
        seq: hb.seq,
        t1_ns: hb.t1_ns,
        previous: hb.previous.map(timing_from_pb),
        last_seq_seen: hb
            .last_seq_seen
            .into_iter()
            .map(|c| (ChannelId(c.channel), c.seq))
            .collect(),
        last_chunk_accepted: hb.last_chunk_accepted,
        last_chunk_applied: hb.last_chunk_applied,
        queue_depth: hb.queue_depth,
        // An unknown (newer) health value reads as unknown, not an error.
        health: match pb::Health::try_from(hb.health) {
            Ok(pb::Health::Ok) => Health::Ok,
            Ok(pb::Health::Degraded) => Health::Degraded,
            Ok(pb::Health::Unhealthy) => Health::Unhealthy,
            _ => Health::Unknown,
        },
    })
}

/// Encodes a heartbeat echo.
pub fn encode_echo(echo: &HeartbeatEcho) -> Vec<u8> {
    pb::HeartbeatEcho {
        seq: echo.seq,
        t1_ns: echo.t1_ns,
        t2_ns: echo.t2_ns,
        t3_ns: echo.t3_ns,
        epoch: echo.epoch.0,
    }
    .encode_to_vec()
}

/// Decodes a heartbeat echo.
pub fn decode_echo(buf: &[u8]) -> Result<HeartbeatEcho, WireError> {
    let echo = pb::HeartbeatEcho::decode(buf)?;
    Ok(HeartbeatEcho {
        seq: echo.seq,
        t1_ns: echo.t1_ns,
        t2_ns: echo.t2_ns,
        t3_ns: echo.t3_ns,
        epoch: Epoch(echo.epoch),
    })
}

// --- samples and chunks ------------------------------------------------------

/// The most actions one chunk may carry.
pub const MAX_CHUNK_ACTIONS: usize = 1 << 16;

/// An encoded message as buffer segments, to be sent in order. Gathering
/// transports send the leaf segments without copying them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Encoded {
    /// The segments, in order.
    pub segments: Vec<Bytes>,
}

impl Encoded {
    /// Total length.
    pub fn len(&self) -> usize {
        self.segments.iter().map(Bytes::len).sum()
    }

    /// Whether the message is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The message as one contiguous buffer. Copies the leaves: for
    /// transports that cannot gather segments, and for tests.
    pub fn to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.len());
        for segment in &self.segments {
            out.extend_from_slice(segment);
        }
        out
    }
}

/// A `bytes` field's tag (field number, wire type 2).
const fn len_delimited_tag(field: u32) -> u64 {
    ((field as u64) << 3) | 2
}

const SAMPLE_LEAVES: u32 = 6;
const CHUNK_ACTIONS: u32 = 11;
const ACTION_LEAVES: u32 = 1;

/// Appends `tag | len` for one length-delimited field to `prefix`.
fn put_field_header(prefix: &mut Vec<u8>, field: u32, len: usize) {
    encode_varint(len_delimited_tag(field), prefix);
    encode_varint(len as u64, prefix);
}

/// Bytes `leaves` take as repeated `bytes` field `field`.
fn leaves_len(field: u32, leaves: &[Bytes]) -> usize {
    leaves
        .iter()
        .map(|l| {
            encoded_len_varint(len_delimited_tag(field))
                + encoded_len_varint(l.len() as u64)
                + l.len()
        })
        .sum()
}

/// Appends `leaves` as repeated `bytes` field `field`: each leaf's header
/// joins `pending`, and each leaf becomes its own segment.
fn push_leaves(segments: &mut Vec<Bytes>, pending: &mut Vec<u8>, field: u32, leaves: &[Bytes]) {
    for leaf in leaves {
        put_field_header(pending, field, leaf.len());
        segments.push(Bytes::from(std::mem::take(pending)));
        if !leaf.is_empty() {
            segments.push(leaf.clone());
        }
    }
}

/// Encodes a sample. Its leaves become segments (refcount clones, no copy).
pub fn encode_sample(sample: &Sample) -> Encoded {
    let fields = pb::Sample {
        channel: sample.channel.0,
        seq: sample.seq,
        epoch: sample.epoch.0,
        capture_ns: sample.capture_ns,
        publish_ns: sample.publish_ns,
        leaves: Vec::new(),
        tick: sample.tick,
    };
    let mut segments = Vec::with_capacity(1 + 2 * sample.leaves.len());
    let mut pending = fields.encode_to_vec();
    push_leaves(&mut segments, &mut pending, SAMPLE_LEAVES, &sample.leaves);
    if !pending.is_empty() {
        segments.push(Bytes::from(pending));
    }
    Encoded { segments }
}

/// Decodes a sample. The leaves slice `buf` (no copy).
pub fn decode_sample(buf: Bytes) -> Result<Sample, WireError> {
    let s = pb::Sample::decode(buf)?;
    Ok(Sample {
        channel: ChannelId(s.channel),
        seq: s.seq,
        epoch: Epoch(s.epoch),
        capture_ns: s.capture_ns,
        publish_ns: s.publish_ns,
        tick: s.tick,
        leaves: s.leaves,
    })
}

/// Encodes an action chunk. Its leaves become segments (no copy).
pub fn encode_chunk(chunk: &ActionChunk) -> Result<Encoded, WireError> {
    if chunk.actions.len() > MAX_CHUNK_ACTIONS {
        return Err(WireError::malformed(
            "action chunk",
            format!(
                "{} actions (at most {MAX_CHUNK_ACTIONS})",
                chunk.actions.len()
            ),
        ));
    }
    let fields = pb::ActionChunk {
        channel: chunk.channel.0,
        source: chunk.source.to_string(),
        lease_token: chunk.lease_token.0,
        epoch: chunk.epoch.0,
        seq: chunk.seq,
        based_on: chunk
            .based_on
            .iter()
            .map(|o| pb::ObservationRef {
                channel: o.channel.0,
                seq: o.seq,
            })
            .collect(),
        t0_ns: chunk.t0_ns,
        dt_ns: chunk.dt_ns,
        replace_from_ns: chunk.replace_from_ns,
        valid_until_ns: chunk.valid_until_ns,
        actions: Vec::new(),
        publish_ns: chunk.publish_ns,
        tick: chunk.tick,
    };
    let leaves: usize = chunk.actions.iter().map(Vec::len).sum();
    let mut segments = Vec::with_capacity(1 + chunk.actions.len() + 2 * leaves);
    let mut pending = fields.encode_to_vec();
    for action in &chunk.actions {
        put_field_header(
            &mut pending,
            CHUNK_ACTIONS,
            leaves_len(ACTION_LEAVES, action),
        );
        push_leaves(&mut segments, &mut pending, ACTION_LEAVES, action);
    }
    if !pending.is_empty() {
        segments.push(Bytes::from(pending));
    }
    Ok(Encoded { segments })
}

/// Decodes an action chunk. The actions' leaves slice `buf` (no copy).
pub fn decode_chunk(buf: Bytes) -> Result<ActionChunk, WireError> {
    let c = pb::ActionChunk::decode(buf)?;
    if c.actions.len() > MAX_CHUNK_ACTIONS {
        return Err(WireError::malformed(
            "action chunk",
            format!("{} actions (at most {MAX_CHUNK_ACTIONS})", c.actions.len()),
        ));
    }
    Ok(ActionChunk {
        channel: ChannelId(c.channel),
        source: ParticipantId::new(c.source)?,
        lease_token: LeaseToken(c.lease_token),
        epoch: Epoch(c.epoch),
        seq: c.seq,
        publish_ns: c.publish_ns,
        tick: c.tick,
        based_on: c
            .based_on
            .into_iter()
            .map(|o| ObservationRef {
                channel: ChannelId(o.channel),
                seq: o.seq,
            })
            .collect(),
        t0_ns: c.t0_ns,
        dt_ns: c.dt_ns,
        replace_from_ns: c.replace_from_ns,
        valid_until_ns: c.valid_until_ns,
        actions: c.actions.into_iter().map(|a| a.leaves).collect(),
    })
}

// --- conversions -------------------------------------------------------------

fn bad_enum(what: &'static str, raw: i32) -> WireError {
    WireError::malformed(what, format!("unknown value {raw}"))
}

fn role_to_pb(role: Role) -> pb::Role {
    match role {
        Role::Embodiment => pb::Role::Embodiment,
        Role::Controller => pb::Role::Controller,
        Role::Observer => pb::Role::Observer,
    }
}

fn role_from_pb(raw: i32) -> Result<Role, WireError> {
    match pb::Role::try_from(raw) {
        Ok(pb::Role::Embodiment) => Ok(Role::Embodiment),
        Ok(pb::Role::Controller) => Ok(Role::Controller),
        Ok(pb::Role::Observer) => Ok(Role::Observer),
        _ => Err(bad_enum("role", raw)),
    }
}

fn mode_to_pb(mode: Mode) -> pb::Mode {
    match mode {
        Mode::Disarmed => pb::Mode::Disarmed,
        Mode::Armed => pb::Mode::Armed,
        Mode::Active => pb::Mode::Active,
        Mode::Safe => pb::Mode::Safe,
        Mode::Estop => pb::Mode::Estop,
    }
}

fn mode_from_pb(raw: i32) -> Result<Mode, WireError> {
    match pb::Mode::try_from(raw) {
        Ok(pb::Mode::Disarmed) => Ok(Mode::Disarmed),
        Ok(pb::Mode::Armed) => Ok(Mode::Armed),
        Ok(pb::Mode::Active) => Ok(Mode::Active),
        Ok(pb::Mode::Safe) => Ok(Mode::Safe),
        Ok(pb::Mode::Estop) => Ok(Mode::Estop),
        _ => Err(bad_enum("mode", raw)),
    }
}

fn priority_to_pb(priority: PriorityClass) -> pb::PriorityClass {
    match priority {
        PriorityClass::Scripted => pb::PriorityClass::Scripted,
        PriorityClass::Model => pb::PriorityClass::Model,
        PriorityClass::Operator => pb::PriorityClass::Operator,
        PriorityClass::Safety => pb::PriorityClass::Safety,
    }
}

fn priority_from_pb(raw: i32) -> Result<PriorityClass, WireError> {
    match pb::PriorityClass::try_from(raw) {
        Ok(pb::PriorityClass::Scripted) => Ok(PriorityClass::Scripted),
        Ok(pb::PriorityClass::Model) => Ok(PriorityClass::Model),
        Ok(pb::PriorityClass::Operator) => Ok(PriorityClass::Operator),
        Ok(pb::PriorityClass::Safety) => Ok(PriorityClass::Safety),
        _ => Err(bad_enum("priority class", raw)),
    }
}

fn status_to_pb(status: CtlStatus) -> pb::CtlStatus {
    match status {
        CtlStatus::InvalidRequest => pb::CtlStatus::InvalidRequest,
        CtlStatus::NotAttached => pb::CtlStatus::NotAttached,
        CtlStatus::Forbidden => pb::CtlStatus::Forbidden,
        CtlStatus::InvalidTransition => pb::CtlStatus::InvalidTransition,
        CtlStatus::EstopLatched => pb::CtlStatus::EstopLatched,
        CtlStatus::StaleEpoch => pb::CtlStatus::StaleEpoch,
        CtlStatus::StaleToken => pb::CtlStatus::StaleToken,
        CtlStatus::LeaseBusy => pb::CtlStatus::LeaseBusy,
        CtlStatus::ChannelConflict => pb::CtlStatus::ChannelConflict,
        CtlStatus::UnknownChannel => pb::CtlStatus::UnknownChannel,
        CtlStatus::Unsupported => pb::CtlStatus::Unsupported,
        CtlStatus::Incompatible => pb::CtlStatus::Incompatible,
    }
}

/// `Ok(())` for OK; anything else (an unknown, newer status included) is a
/// refusal.
fn status_from_pb(raw: i32, message: String) -> Result<(), CtlError> {
    let status = match pb::CtlStatus::try_from(raw) {
        Ok(pb::CtlStatus::Ok) => return Ok(()),
        Ok(pb::CtlStatus::NotAttached) => CtlStatus::NotAttached,
        Ok(pb::CtlStatus::Forbidden) => CtlStatus::Forbidden,
        Ok(pb::CtlStatus::InvalidTransition) => CtlStatus::InvalidTransition,
        Ok(pb::CtlStatus::EstopLatched) => CtlStatus::EstopLatched,
        Ok(pb::CtlStatus::StaleEpoch) => CtlStatus::StaleEpoch,
        Ok(pb::CtlStatus::StaleToken) => CtlStatus::StaleToken,
        Ok(pb::CtlStatus::LeaseBusy) => CtlStatus::LeaseBusy,
        Ok(pb::CtlStatus::ChannelConflict) => CtlStatus::ChannelConflict,
        Ok(pb::CtlStatus::UnknownChannel) => CtlStatus::UnknownChannel,
        Ok(pb::CtlStatus::Unsupported) => CtlStatus::Unsupported,
        Ok(pb::CtlStatus::Incompatible) => CtlStatus::Incompatible,
        Ok(pb::CtlStatus::InvalidRequest | pb::CtlStatus::Unspecified) | Err(_) => {
            CtlStatus::InvalidRequest
        }
    };
    Err(CtlError { status, message })
}

fn participant_to_pb(p: &ParticipantInfo) -> pb::ParticipantInfo {
    pb::ParticipantInfo {
        id: p.id.to_string(),
        name: p.name.clone(),
        roles: p.roles.iter().map(|r| role_to_pb(*r) as i32).collect(),
        priority: priority_to_pb(p.priority) as i32,
        attached_ns: p.attached_ns,
        protocol_minor: p.protocol_minor,
        capabilities: p.capabilities.clone(),
    }
}

fn participant_from_pb(p: pb::ParticipantInfo) -> Result<ParticipantInfo, WireError> {
    Ok(ParticipantInfo {
        id: ParticipantId::new(p.id)?,
        name: p.name,
        roles: p
            .roles
            .into_iter()
            .map(role_from_pb)
            .collect::<Result<_, _>>()?,
        priority: priority_from_pb(p.priority)?,
        attached_ns: p.attached_ns,
        protocol_minor: p.protocol_minor,
        capabilities: p.capabilities,
    })
}

fn lease_to_pb(l: &Lease) -> pb::Lease {
    pb::Lease {
        channel: l.channel.0,
        holder: l.holder.to_string(),
        priority: priority_to_pb(l.priority) as i32,
        token: l.token.0,
        ttl_ns: l.ttl_ns,
        granted_ns: l.granted_ns,
        expires_ns: l.expires_ns,
    }
}

fn lease_from_pb(l: pb::Lease) -> Result<Lease, WireError> {
    Ok(Lease {
        channel: ChannelId(l.channel),
        holder: ParticipantId::new(l.holder)?,
        priority: priority_from_pb(l.priority)?,
        token: LeaseToken(l.token),
        ttl_ns: l.ttl_ns,
        granted_ns: l.granted_ns,
        expires_ns: l.expires_ns,
    })
}

fn clock_to_pb(c: &ClockMapping) -> pb::ClockMapping {
    pb::ClockMapping {
        participant: c.participant.to_string(),
        offset_ns: c.offset_ns,
        error_ns: c.error_ns,
        drift_bound_ppm: c.drift_bound_ppm,
        rtt_ns: c.rtt_ns,
        sample_ns: c.sample_ns,
    }
}

fn clock_from_pb(c: pb::ClockMapping) -> Result<ClockMapping, WireError> {
    Ok(ClockMapping {
        participant: ParticipantId::new(c.participant)?,
        offset_ns: c.offset_ns,
        error_ns: c.error_ns,
        rtt_ns: c.rtt_ns,
        sample_ns: c.sample_ns,
        drift_bound_ppm: c.drift_bound_ppm,
    })
}

fn state_to_pb(s: &AuthoritySnapshot) -> pb::AuthorityState {
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
        channels: s.channels.iter().map(channel_to_pb).collect(),
        protocol_minor: s.protocol_minor,
        capabilities: s.capabilities.clone(),
    }
}

fn state_from_pb(s: pb::AuthorityState) -> Result<AuthoritySnapshot, WireError> {
    Ok(AuthoritySnapshot {
        session: s.session.parse()?,
        epoch: Epoch(s.epoch),
        revision: s.revision,
        mode: mode_from_pb(s.mode)?,
        estop_latched: s.estop_latched,
        roster: s
            .roster
            .into_iter()
            .map(participant_from_pb)
            .collect::<Result<_, _>>()?,
        leases: s
            .leases
            .into_iter()
            .map(lease_from_pb)
            .collect::<Result<_, _>>()?,
        clocks: s
            .clocks
            .into_iter()
            .map(clock_from_pb)
            .collect::<Result<_, _>>()?,
        host_ns: s.host_ns,
        channels: s
            .channels
            .into_iter()
            .map(channel_from_pb)
            .collect::<Result<_, _>>()?,
        protocol_minor: s.protocol_minor,
        capabilities: s.capabilities,
    })
}

fn timing_to_pb(s: TimingSample) -> pb::TimingSample {
    pb::TimingSample {
        seq: s.seq,
        t1_ns: s.t1_ns,
        t2_ns: s.t2_ns,
        t3_ns: s.t3_ns,
        t4_ns: s.t4_ns,
    }
}

fn timing_from_pb(s: pb::TimingSample) -> TimingSample {
    TimingSample {
        seq: s.seq,
        t1_ns: s.t1_ns,
        t2_ns: s.t2_ns,
        t3_ns: s.t3_ns,
        t4_ns: s.t4_ns,
    }
}

fn request_to_pb(r: &Request) -> pb::CtlRequest {
    use pb::ctl_request::Verb;
    let verb = match &r.command {
        Command::Attach {
            name,
            roles,
            priority,
            protocol_minor,
            capabilities,
        } => Verb::Attach(pb::Attach {
            name: name.clone(),
            roles: roles.iter().map(|r| role_to_pb(*r) as i32).collect(),
            priority: priority_to_pb(*priority) as i32,
            protocol_minor: *protocol_minor,
            capabilities: capabilities.clone(),
        }),
        Command::Detach => Verb::Detach(pb::Detach {}),
        Command::Arm => Verb::Arm(pb::Arm {}),
        Command::Disarm => Verb::Disarm(pb::Disarm {}),
        Command::GrantLease { channel, ttl_ns } => Verb::GrantLease(pb::GrantLease {
            channel: channel.0,
            ttl_ns: ttl_ns.unwrap_or(0),
        }),
        Command::ReleaseLease { channel, token } => Verb::ReleaseLease(pb::ReleaseLease {
            channel: channel.0,
            token: token.0,
        }),
        Command::Estop { reason } => Verb::Estop(pb::Estop {
            reason: reason.clone(),
        }),
        Command::Clear => Verb::Clear(pb::Clear {}),
        Command::DeclareChannels(specs) => Verb::DeclareChannels(pb::DeclareChannels {
            channels: specs.iter().map(channel_to_pb).collect(),
        }),
        Command::EnterSafe { reason } => Verb::EnterSafe(pb::EnterSafe {
            reason: reason.clone(),
        }),
    };
    pb::CtlRequest {
        participant: r.from.to_string(),
        epoch: r.epoch.map_or(0, |e| e.0),
        verb: Some(verb),
    }
}

fn request_from_pb(r: pb::CtlRequest) -> Result<Request, WireError> {
    use pb::ctl_request::Verb;
    let verb = r.verb.ok_or_else(|| {
        WireError::malformed("ctl request", "no verb (or one this peer does not know)")
    })?;
    let command = match verb {
        Verb::Attach(a) => Command::Attach {
            name: a.name,
            roles: a
                .roles
                .into_iter()
                .map(role_from_pb)
                .collect::<Result<_, _>>()?,
            priority: priority_from_pb(a.priority)?,
            protocol_minor: a.protocol_minor,
            capabilities: a.capabilities,
        },
        Verb::Detach(_) => Command::Detach,
        Verb::Arm(_) => Command::Arm,
        Verb::Disarm(_) => Command::Disarm,
        Verb::GrantLease(g) => Command::GrantLease {
            channel: ChannelId(g.channel),
            ttl_ns: (g.ttl_ns > 0).then_some(g.ttl_ns),
        },
        Verb::ReleaseLease(rl) => Command::ReleaseLease {
            channel: ChannelId(rl.channel),
            token: LeaseToken(rl.token),
        },
        Verb::Estop(e) => Command::Estop { reason: e.reason },
        Verb::Clear(_) => Command::Clear,
        Verb::DeclareChannels(d) => Command::DeclareChannels(
            d.channels
                .into_iter()
                .map(channel_from_pb)
                .collect::<Result<_, _>>()?,
        ),
        Verb::EnterSafe(e) => Command::EnterSafe { reason: e.reason },
    };
    Ok(Request {
        from: ParticipantId::new(r.participant)?,
        epoch: (r.epoch > 0).then_some(Epoch(r.epoch)),
        command,
    })
}

fn duration_ns(d: Option<std::time::Duration>) -> u64 {
    d.map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

fn ns_duration(ns: u64) -> Option<std::time::Duration> {
    (ns > 0).then(|| std::time::Duration::from_nanos(ns))
}

fn dtype_to_pb(dtype: DType) -> pb::DType {
    match dtype {
        DType::Bool => pb::DType::Bool,
        DType::U8 => pb::DType::U8,
        DType::I8 => pb::DType::I8,
        DType::U16 => pb::DType::U16,
        DType::I16 => pb::DType::I16,
        DType::U32 => pb::DType::U32,
        DType::I32 => pb::DType::I32,
        DType::U64 => pb::DType::U64,
        DType::I64 => pb::DType::I64,
        DType::F16 => pb::DType::F16,
        DType::Bf16 => pb::DType::Bf16,
        DType::F32 => pb::DType::F32,
        DType::F64 => pb::DType::F64,
    }
}

fn dtype_from_pb(raw: i32) -> Result<DType, WireError> {
    Ok(match pb::DType::try_from(raw) {
        Ok(pb::DType::Bool) => DType::Bool,
        Ok(pb::DType::U8) => DType::U8,
        Ok(pb::DType::I8) => DType::I8,
        Ok(pb::DType::U16) => DType::U16,
        Ok(pb::DType::I16) => DType::I16,
        Ok(pb::DType::U32) => DType::U32,
        Ok(pb::DType::I32) => DType::I32,
        Ok(pb::DType::U64) => DType::U64,
        Ok(pb::DType::I64) => DType::I64,
        Ok(pb::DType::F16) => DType::F16,
        Ok(pb::DType::Bf16) => DType::Bf16,
        Ok(pb::DType::F32) => DType::F32,
        Ok(pb::DType::F64) => DType::F64,
        _ => return Err(bad_enum("dtype", raw)),
    })
}

fn channel_to_pb(c: &ChannelSpec) -> pb::ChannelSpec {
    pb::ChannelSpec {
        id: c.id.0,
        name: c.name.clone(),
        direction: match c.direction {
            Direction::Sense => pb::Direction::Sense,
            Direction::Actuate => pb::Direction::Actuate,
            Direction::Aux => pb::Direction::Aux,
        } as i32,
        role: c.role.clone(),
        layout: c
            .layout
            .iter()
            .map(|t| pb::TensorLayout {
                name: t.name.clone(),
                dtype: dtype_to_pb(t.dtype) as i32,
                shape: t.shape.clone(),
            })
            .collect(),
        units: c.units.clone(),
        frame: c.frame.clone().unwrap_or_default(),
        encoding: c.encoding.wire_name().into(),
        encoding_quality: match c.encoding {
            Encoding::Raw => 0,
            Encoding::Jpeg { quality } => u32::from(quality),
        },
        rate: Some(pb::Rate {
            nominal_hz: c.rate.nominal_hz,
            min_hz: c.rate.min_hz.unwrap_or(0.0),
        }),
        qos: Some(pb::ChannelQos {
            delivery: match c.qos.delivery {
                Delivery::BestEffort => pb::Delivery::BestEffort,
                Delivery::Reliable => pb::Delivery::Reliable,
            } as i32,
            history_depth: u32::try_from(c.qos.history.depth()).unwrap_or(u32::MAX),
            max_age_ns: duration_ns(c.qos.max_age),
            deadline_ns: duration_ns(c.qos.deadline),
            priority: match c.qos.priority {
                ChannelPriority::Control => pb::ChannelPriority::Control,
                ChannelPriority::Realtime => pb::ChannelPriority::Realtime,
                ChannelPriority::Data => pb::ChannelPriority::Data,
                ChannelPriority::Bulk => pb::ChannelPriority::Bulk,
            } as i32,
            congestion: match c.qos.congestion {
                Congestion::Drop => pb::Congestion::Drop,
                Congestion::Block => pb::Congestion::Block,
            } as i32,
            max_bitrate_bps: c.qos.max_bitrate_bps.unwrap_or(0),
        }),
        required: c.required,
        episodic: c.episodic,
        owner: c
            .owner
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default(),
        actuation: c.actuation.as_ref().map(|a| pb::Actuation {
            merge: match a.merge {
                MergePolicy::Replace => pb::MergePolicy::Replace,
                MergePolicy::Blend => pb::MergePolicy::Blend,
                MergePolicy::RtcPrefix => pb::MergePolicy::RtcPrefix,
            } as i32,
            hold_max_ns: duration_ns(Some(a.hold_max)),
            safe_behavior: a.safe_behavior.clone(),
            hold_last: a.hold_last,
        }),
    }
}

fn channel_from_pb(c: pb::ChannelSpec) -> Result<ChannelSpec, WireError> {
    let id = c.id;
    let rate = c.rate.unwrap_or_default();
    let qos = c
        .qos
        .ok_or_else(|| WireError::malformed("channel spec", format!("channel {id} has no qos")))?;
    Ok(ChannelSpec {
        id: ChannelId(id),
        name: c.name,
        direction: match pb::Direction::try_from(c.direction) {
            Ok(pb::Direction::Sense) => Direction::Sense,
            Ok(pb::Direction::Actuate) => Direction::Actuate,
            Ok(pb::Direction::Aux) => Direction::Aux,
            _ => return Err(bad_enum("direction", c.direction)),
        },
        role: c.role,
        layout: c
            .layout
            .into_iter()
            .map(|t| {
                Ok(TensorLayout {
                    name: t.name,
                    dtype: dtype_from_pb(t.dtype)?,
                    shape: t.shape,
                })
            })
            .collect::<Result<_, WireError>>()?,
        units: c.units,
        frame: (!c.frame.is_empty()).then_some(c.frame),
        encoding: match c.encoding.as_str() {
            "" | "raw" => Encoding::Raw,
            "jpeg" => Encoding::Jpeg {
                quality: u8::try_from(c.encoding_quality).map_err(|_| {
                    WireError::malformed(
                        "channel spec",
                        format!("jpeg quality {}", c.encoding_quality),
                    )
                })?,
            },
            other => {
                return Err(WireError::malformed(
                    "channel spec",
                    format!("channel {id} encoding {other:?}"),
                ));
            }
        },
        rate: Rate {
            nominal_hz: rate.nominal_hz,
            min_hz: (rate.min_hz > 0.0).then_some(rate.min_hz),
        },
        qos: ChannelQos {
            delivery: match pb::Delivery::try_from(qos.delivery) {
                Ok(pb::Delivery::BestEffort) => Delivery::BestEffort,
                Ok(pb::Delivery::Reliable) => Delivery::Reliable,
                _ => return Err(bad_enum("delivery", qos.delivery)),
            },
            history: History::from_depth(qos.history_depth),
            max_age: ns_duration(qos.max_age_ns),
            deadline: ns_duration(qos.deadline_ns),
            priority: match pb::ChannelPriority::try_from(qos.priority) {
                Ok(pb::ChannelPriority::Control) => ChannelPriority::Control,
                Ok(pb::ChannelPriority::Realtime) => ChannelPriority::Realtime,
                Ok(pb::ChannelPriority::Data) => ChannelPriority::Data,
                Ok(pb::ChannelPriority::Bulk) => ChannelPriority::Bulk,
                _ => return Err(bad_enum("channel priority", qos.priority)),
            },
            congestion: match pb::Congestion::try_from(qos.congestion) {
                Ok(pb::Congestion::Drop) => Congestion::Drop,
                Ok(pb::Congestion::Block) => Congestion::Block,
                _ => return Err(bad_enum("congestion", qos.congestion)),
            },
            max_bitrate_bps: (qos.max_bitrate_bps > 0).then_some(qos.max_bitrate_bps),
        },
        required: c.required,
        episodic: c.episodic,
        owner: if c.owner.is_empty() {
            None
        } else {
            Some(ParticipantId::new(c.owner)?)
        },
        actuation: c
            .actuation
            .map(|a| -> Result<Actuation, WireError> {
                Ok(Actuation {
                    merge: match pb::MergePolicy::try_from(a.merge) {
                        Ok(pb::MergePolicy::Replace | pb::MergePolicy::Unspecified) => {
                            MergePolicy::Replace
                        }
                        Ok(pb::MergePolicy::Blend) => MergePolicy::Blend,
                        Ok(pb::MergePolicy::RtcPrefix) => MergePolicy::RtcPrefix,
                        Err(_) => return Err(bad_enum("merge policy", a.merge)),
                    },
                    hold_max: ns_duration(a.hold_max_ns)
                        .unwrap_or_else(|| Actuation::default().hold_max),
                    safe_behavior: a.safe_behavior,
                    hold_last: a.hold_last,
                })
            })
            .transpose()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::tests::{camera, joint_cmd};
    use crate::ids::SessionId;
    use crate::version::PROTOCOL_MINOR;

    fn pid(id: &str) -> ParticipantId {
        ParticipantId::new(id).expect("id")
    }

    fn snapshot() -> AuthoritySnapshot {
        AuthoritySnapshot {
            session: SessionId::new(),
            epoch: Epoch(3),
            revision: 9,
            mode: Mode::Active,
            estop_latched: false,
            roster: vec![ParticipantInfo {
                id: pid("op"),
                name: "Operator".into(),
                roles: vec![Role::Controller, Role::Observer],
                priority: PriorityClass::Operator,
                attached_ns: 12,
                protocol_minor: PROTOCOL_MINOR,
                capabilities: vec!["frame.batch".into()],
            }],
            leases: vec![Lease {
                channel: ChannelId(1),
                holder: pid("op"),
                priority: PriorityClass::Operator,
                token: LeaseToken(4),
                ttl_ns: 5,
                granted_ns: 6,
                expires_ns: 11,
            }],
            clocks: vec![ClockMapping {
                participant: pid("op"),
                offset_ns: -42,
                error_ns: 7,
                rtt_ns: 12,
                sample_ns: 3,
                drift_bound_ppm: 100.0,
            }],
            host_ns: 99,
            channels: vec![camera(), joint_cmd(MergePolicy::Replace)],
            protocol_minor: PROTOCOL_MINOR,
            capabilities: Vec::new(),
        }
    }

    #[test]
    fn control_messages_round_trip() {
        let requests = [
            Command::Attach {
                name: "Operator".into(),
                roles: vec![Role::Controller, Role::Observer, Role::Embodiment],
                priority: PriorityClass::Operator,
                protocol_minor: 3,
                capabilities: vec!["merge.blend".into()],
            },
            Command::DeclareChannels(vec![camera(), joint_cmd(MergePolicy::Replace)]),
            Command::Arm,
            Command::Disarm,
            Command::Detach,
            Command::Clear,
            Command::GrantLease {
                channel: ChannelId(4),
                ttl_ns: Some(5),
            },
            Command::GrantLease {
                channel: ChannelId(4),
                ttl_ns: None,
            },
            Command::ReleaseLease {
                channel: ChannelId(4),
                token: LeaseToken(2),
            },
            Command::Estop {
                reason: "red button".into(),
            },
            Command::EnterSafe {
                reason: "queue dry".into(),
            },
        ];
        for command in requests {
            let request = Request {
                from: pid("op"),
                epoch: Some(Epoch(2)),
                command,
            };
            assert_eq!(
                decode_request(&encode_request(&request)).expect("decode"),
                request
            );
        }

        let state = snapshot();
        assert_eq!(decode_state(&encode_state(&state)).expect("decode"), state);
        for result in [
            Ok(()),
            Err(CtlError::new(CtlStatus::LeaseBusy, "busy")),
            Err(CtlError::new(CtlStatus::Incompatible, "needs merge.blend")),
        ] {
            let reply = CtlReply {
                result,
                lease: state.leases.first().cloned(),
                state: state.clone(),
            };
            assert_eq!(decode_reply(&encode_reply(&reply)).expect("decode"), reply);
        }

        let hb = Heartbeat {
            participant: pid("op"),
            epoch: Epoch(3),
            mode: Mode::Armed,
            seq: 8,
            t1_ns: -5,
            previous: Some(TimingSample {
                seq: 7,
                t1_ns: 1,
                t2_ns: 2,
                t3_ns: 3,
                t4_ns: 4,
            }),
            last_seq_seen: vec![(ChannelId(2), 77)],
            last_chunk_accepted: 3,
            last_chunk_applied: 2,
            queue_depth: 40,
            health: Health::Degraded,
        };
        assert_eq!(
            decode_heartbeat(&encode_heartbeat(&hb)).expect("decode"),
            hb
        );
        let echo = HeartbeatEcho {
            seq: 8,
            t1_ns: -5,
            t2_ns: 10,
            t3_ns: 11,
            epoch: Epoch(3),
        };
        assert_eq!(decode_echo(&encode_echo(&echo)).expect("decode"), echo);
        assert!(decode_state(b"\xff\xff\xff").is_err());
    }

    fn ft_sample() -> Sample {
        Sample {
            channel: ChannelId(2),
            seq: 41,
            epoch: Epoch(3),
            capture_ns: 100,
            publish_ns: 120,
            tick: 7,
            leaves: vec![
                Bytes::from(vec![7_u8; 72]),
                Bytes::new(),
                Bytes::from(vec![9_u8; 300]),
            ],
        }
    }

    #[test]
    fn samples_are_plain_protobuf_with_leaves_neither_copied_on_send_nor_on_receipt() {
        let sample = ft_sample();
        let encoded = encode_sample(&sample);
        assert!(
            encoded
                .segments
                .iter()
                .any(|s| s.as_ptr() == sample.leaves[2].as_ptr()),
            "the leaf is a segment of its own"
        );
        let buf = Bytes::from(encoded.to_vec());
        // Any protobuf implementation reads it.
        let plain = pb::Sample::decode(&buf[..]).expect("plain protobuf");
        assert_eq!(plain.leaves.len(), 3);
        assert_eq!((plain.seq, plain.tick), (41, 7));
        let decoded = decode_sample(buf.clone()).expect("decode");
        assert_eq!(decoded, sample);
        let range = buf.as_ptr_range();
        assert!(
            range.contains(&decoded.leaves[2].as_ptr()),
            "the leaf slices the received buffer"
        );
        assert_eq!(decoded.payload_len(), 372);
        assert!(decode_sample(Bytes::from_static(b"\xff\xff")).is_err());
    }

    fn chunk() -> ActionChunk {
        ActionChunk {
            channel: ChannelId(1),
            source: pid("policy"),
            lease_token: LeaseToken(3),
            epoch: Epoch(2),
            seq: 9,
            publish_ns: 78,
            tick: 0,
            based_on: vec![
                ObservationRef {
                    channel: ChannelId(2),
                    seq: 41,
                },
                ObservationRef {
                    channel: ChannelId(5),
                    seq: 7,
                },
            ],
            t0_ns: 1_000,
            dt_ns: 1_000_000,
            replace_from_ns: 900,
            valid_until_ns: 500_000_000,
            actions: (0..5_u8)
                .map(|i| vec![Bytes::from(vec![i; 24]), Bytes::from(vec![i; 200])])
                .collect(),
        }
    }

    #[test]
    fn chunks_round_trip_through_plain_protobuf() {
        let chunk = chunk();
        let encoded = encode_chunk(&chunk).expect("encode");
        let buf = Bytes::from(encoded.to_vec());
        let plain = pb::ActionChunk::decode(&buf[..]).expect("plain protobuf");
        assert_eq!(plain.actions.len(), 5);
        assert_eq!(plain.actions[3].leaves[1], vec![3_u8; 200]);
        let decoded = decode_chunk(buf.clone()).expect("decode");
        assert_eq!(decoded, chunk);
        assert!(buf.as_ptr_range().contains(&decoded.actions[4][1].as_ptr()));

        let mut empty = chunk.clone();
        empty.actions.clear();
        let decoded =
            decode_chunk(Bytes::from(encode_chunk(&empty).expect("encode").to_vec())).expect("ok");
        assert!(decoded.actions.is_empty());
        let mut huge = chunk;
        huge.actions = vec![Vec::new(); MAX_CHUNK_ACTIONS + 1];
        assert!(encode_chunk(&huge).is_err());
    }

    #[test]
    fn unknown_enum_values_are_refused_but_unknown_statuses_are_refusals() {
        assert!(mode_from_pb(0).is_err());
        assert!(mode_from_pb(99).is_err());
        assert_eq!(
            status_from_pb(pb::CtlStatus::LeaseBusy as i32, "busy".into()).map_err(|e| e.status),
            Err(CtlStatus::LeaseBusy)
        );
        assert_eq!(
            status_from_pb(999, String::new()).map_err(|e| e.status),
            Err(CtlStatus::InvalidRequest)
        );
        assert!(status_from_pb(pb::CtlStatus::Ok as i32, String::new()).is_ok());
    }
}
