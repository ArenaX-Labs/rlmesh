//! The byte encoding: protobuf envelopes with raw payloads.
//!
//! This is the only module that knows how messages look on the wire (see the
//! crate docs for why). Everything else in the crate, and every crate above
//! it, speaks the plain types this module encodes and decodes.
//!
//! Every message is one protobuf message of `rlmesh.session.v2` (see
//! `proto/rlmesh/session/v2/session.proto`) per payload; the key says which.
//!
//! - Control messages ([`Request`], [`CtlReply`], [`AuthoritySnapshot`],
//!   [`Heartbeat`], [`HeartbeatEcho`], [`ParticipantReport`], [`VirtualTick`],
//!   [`HistoryQuery`])
//!   encode to one buffer and decode from any byte slice.
//! - [`Sample`]s, [`SampleBatch`]es and [`ActionChunk`]s carry their values
//!   as one packed `bytes` payload. [`encode_sample`], [`encode_batch`] and
//!   [`encode_chunk`] emit the message as [`Encoded`] segments: small
//!   buffers for the fields and the payload's tag and length, and the
//!   payload as its own [`Bytes`] segment, so a transport that gathers
//!   segments (Zenoh's `ZBytes`) sends it without a copy. The concatenation
//!   is ordinary protobuf. The decoders take the received buffer as
//!   [`Bytes`], and the payload slices it.
//! - The lockstep and served-model messages encode to one buffer (one copy
//!   of the slabs).
//!
//! Payload contents are not validated here; the layout check against the
//! declaration is [`StreamSpec::check_payload`] (samples) and the edge's
//! intake (chunks). Lengths are bounded here: a chunk's action count and
//! payload size, a batch's sample count.
//!
//! [`StreamSpec::check_payload`]: crate::wire::part::StreamSpec::check_payload

use std::collections::BTreeMap;
use std::time::Duration;

use bytes::Bytes;
use prost::Message as _;
use prost::encoding::encode_varint;

use crate::wire::action::{ActionChunk, ChunkRef, InputRange};
use crate::wire::authority::{
    AuthoritySnapshot, BindingRecord, ClockMapping, ClockMode, Command, CtlError, CtlReply,
    CtlStatus, Lease, Mode, ParticipantInfo, PriorityClass, Request, Role, Watermark,
};
use crate::wire::contract::{ActionGrid, ContractMode, Demand, Plan, StreamContract};
use crate::wire::error::WireError;
use crate::wire::heartbeat::{Health, Heartbeat, HeartbeatEcho, TimingSample};
use crate::wire::ids::{ChannelId, Epoch, Fingerprint, LeaseToken, ParticipantId};
use crate::wire::lockstep::{LockstepReply, LockstepReset, LockstepStep, PartValues};
use crate::wire::model::{
    ModelEpisode, ModelFault, ModelGroup, ModelHistoryRow, ModelReply, ModelRequest, ModelResolve,
    ModelResolved,
};
use crate::wire::part::{
    ChannelPriority, CommandSpec, DType, Encoding, FieldLimit, FieldSpec, MAX_BATCH,
    MAX_CHUNK_ACTIONS, MAX_CHUNK_BYTES, MergePolicy, PartSpec,
};
use crate::wire::path::PartPath;
use crate::wire::report::{ParticipantReport, PublisherCounters, RunCounters};
use crate::wire::sample::{
    Applied, EdgeMode, HistoryQuery, Sample, SampleBatch, SampleMeta, SeqRange,
};
use crate::wire::tick::{Step, TickState, VirtualTick};
use crate::wire::timing::EndpointTiming;

/// Generated `rlmesh.session.v2` messages: private, so the schema can move
/// without touching any public type.
#[allow(clippy::all, clippy::pedantic)]
mod pb {
    include!(concat!(env!("OUT_DIR"), "/rlmesh.session.v2.rs"));
}

/// The `rlmesh.session.v2` protobuf descriptors (a serialized
/// `google.protobuf.FileDescriptorSet`): the schema a recording names for
/// each envelope it stores (§13).
pub const FILE_DESCRIPTOR_SET: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/session_descriptor.bin"));

/// The full protobuf names of the envelopes a session carries, by role.
pub mod schema {
    /// A data sample (`d/...` keys, one per MCAP message).
    pub const SAMPLE: &str = "rlmesh.session.v2.Sample";
    /// An action chunk (`c/...` keys).
    pub const ACTION_CHUNK: &str = "rlmesh.session.v2.ActionChunk";
    /// The authority snapshot (`host/state`).
    pub const AUTHORITY_STATE: &str = "rlmesh.session.v2.AuthorityState";
    /// One controller's binding.
    pub const BINDING_RECORD: &str = "rlmesh.session.v2.BindingRecord";
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
        leases: reply.leases.iter().map(lease_to_pb).collect(),
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
        leases: collect(reply.leases, lease_from_pb)?,
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

/// Encodes one controller's binding record.
pub fn encode_binding(binding: &BindingRecord) -> Vec<u8> {
    binding_to_pb(binding).encode_to_vec()
}

/// Decodes one controller's binding record.
pub fn decode_binding(buf: &[u8]) -> Result<BindingRecord, WireError> {
    binding_from_pb(pb::BindingRecord::decode(buf)?)
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

/// Encodes a participant report.
pub fn encode_report(report: &ParticipantReport) -> Vec<u8> {
    pb::ParticipantReport {
        participant: report.participant.to_string(),
        epoch: report.epoch.0,
        sent_ns: report.sent_ns,
        streams: report
            .streams
            .iter()
            .map(|s| pb::PublisherCounters {
                channel: s.channel.0,
                published: s.published,
                coalesced: s.coalesced,
            })
            .collect(),
        run: report.run.map(|r| pb::RunCounters {
            starts: r.starts,
            completions: r.completions,
            busy_skips: r.busy_skips,
            waiting_for: r.waiting_for,
            stale_input: r.stale_input,
            window_gap: r.window_gap,
            decode: r.decode,
            no_lease: r.no_lease,
            chunks_sent: r.chunks_sent,
            pending_ns: r.pending_ns,
            latency_p50_ns: r.latency_p50_ns,
            latency_p99_ns: r.latency_p99_ns,
            drifting: r.drifting,
        }),
    }
    .encode_to_vec()
}

/// Decodes a participant report.
pub fn decode_report(buf: &[u8]) -> Result<ParticipantReport, WireError> {
    let report = pb::ParticipantReport::decode(buf)?;
    Ok(ParticipantReport {
        participant: ParticipantId::new(report.participant)?,
        epoch: Epoch(report.epoch),
        sent_ns: report.sent_ns,
        streams: report
            .streams
            .into_iter()
            .map(|s| PublisherCounters {
                channel: ChannelId(s.channel),
                published: s.published,
                coalesced: s.coalesced,
            })
            .collect(),
        run: report.run.map(|r| RunCounters {
            starts: r.starts,
            completions: r.completions,
            busy_skips: r.busy_skips,
            waiting_for: r.waiting_for,
            stale_input: r.stale_input,
            window_gap: r.window_gap,
            decode: r.decode,
            no_lease: r.no_lease,
            chunks_sent: r.chunks_sent,
            pending_ns: r.pending_ns,
            latency_p50_ns: r.latency_p50_ns,
            latency_p99_ns: r.latency_p99_ns,
            drifting: r.drifting,
        }),
    })
}

/// Encodes a virtual-mode tick.
pub fn encode_tick(tick: &VirtualTick) -> Vec<u8> {
    use pb::virtual_tick::Phase;
    let phase = match tick {
        VirtualTick::Observe(state) => Phase::Observe(pb::TickState {
            epoch: state.epoch.0,
            tick: state.tick,
            watermarks: state.watermarks.iter().map(watermark_to_pb).collect(),
            due: state.due.iter().map(ToString::to_string).collect(),
        }),
        VirtualTick::Step(step) => Phase::Step(pb::Step {
            epoch: step.epoch.0,
            tick: step.tick,
            chunks: step.chunks.iter().map(chunk_ref_to_pb).collect(),
        }),
    };
    pb::VirtualTick { phase: Some(phase) }.encode_to_vec()
}

/// Decodes a virtual-mode tick.
pub fn decode_tick(buf: &[u8]) -> Result<VirtualTick, WireError> {
    use pb::virtual_tick::Phase;
    match pb::VirtualTick::decode(buf)?.phase {
        Some(Phase::Observe(state)) => Ok(VirtualTick::Observe(TickState {
            epoch: Epoch(state.epoch),
            tick: state.tick,
            watermarks: state
                .watermarks
                .into_iter()
                .map(watermark_from_pb)
                .collect(),
            due: collect(state.due, ParticipantId::new)?,
        })),
        Some(Phase::Step(step)) => Ok(VirtualTick::Step(Step {
            epoch: Epoch(step.epoch),
            tick: step.tick,
            chunks: step.chunks.into_iter().map(chunk_ref_from_pb).collect(),
        })),
        None => Err(WireError::malformed("virtual tick", "no phase")),
    }
}

/// Encodes a history query.
pub fn encode_history_query(query: &HistoryQuery) -> Vec<u8> {
    pb::HistoryQuery {
        epoch: query.epoch.0,
        after_seq: query.after_seq,
        until_seq: query.until_seq.unwrap_or(0),
        limit: query.limit.unwrap_or(0),
    }
    .encode_to_vec()
}

/// Decodes a history query.
pub fn decode_history_query(buf: &[u8]) -> Result<HistoryQuery, WireError> {
    let q = pb::HistoryQuery::decode(buf)?;
    Ok(HistoryQuery {
        epoch: Epoch(q.epoch),
        after_seq: q.after_seq,
        until_seq: (q.until_seq > 0).then_some(q.until_seq),
        limit: (q.limit > 0).then_some(q.limit),
    })
}

// --- samples, batches and chunks ---------------------------------------------

/// An encoded message as buffer segments, to be sent in order. Gathering
/// transports send the payload segment without copying it.
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

    /// The message as one contiguous buffer. Copies the payload: for
    /// transports that cannot gather segments, and for tests.
    pub fn to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.len());
        for segment in &self.segments {
            out.extend_from_slice(segment);
        }
        out
    }

    /// `head`, then `payload` as `bytes` field `field` (its own segment),
    /// then `tail`.
    fn around(mut head: Vec<u8>, field: u32, payload: &Bytes, tail: Vec<u8>) -> Self {
        let mut segments = Vec::with_capacity(3);
        if !payload.is_empty() {
            encode_varint((u64::from(field) << 3) | 2, &mut head);
            encode_varint(payload.len() as u64, &mut head);
            segments.push(Bytes::from(head));
            segments.push(payload.clone());
        } else {
            segments.push(Bytes::from(head));
        }
        if !tail.is_empty() {
            segments.push(Bytes::from(tail));
        }
        Self { segments }
    }
}

const SAMPLE_PAYLOAD: u32 = 7;
const BATCH_PAYLOAD: u32 = 5;
const CHUNK_ACTIONS: u32 = 12;

/// Encodes a sample. Its payload becomes a segment (a refcount clone, no
/// copy).
pub fn encode_sample(sample: &Sample) -> Encoded {
    let head = pb::Sample {
        channel: sample.channel.0,
        epoch: sample.epoch.0,
        seq: sample.seq,
        capture_ns: sample.capture_ns,
        publish_ns: sample.publish_ns,
        tick: sample.tick,
        payload: Bytes::new(),
        applied: None,
    }
    .encode_to_vec();
    let tail = pb::Sample {
        applied: sample.applied.map(applied_to_pb),
        ..pb::Sample::default()
    }
    .encode_to_vec();
    Encoded::around(head, SAMPLE_PAYLOAD, &sample.payload, tail)
}

/// Decodes a sample. The payload slices `buf` (no copy).
pub fn decode_sample(buf: Bytes) -> Result<Sample, WireError> {
    let s = pb::Sample::decode(buf)?;
    Ok(Sample {
        channel: ChannelId(s.channel),
        epoch: Epoch(s.epoch),
        seq: s.seq,
        capture_ns: s.capture_ns,
        publish_ns: s.publish_ns,
        tick: s.tick,
        payload: s.payload,
        applied: s.applied.map(applied_from_pb).transpose()?,
    })
}

/// Encodes a batch. Its payload becomes a segment (no copy).
pub fn encode_batch(batch: &SampleBatch) -> Result<Encoded, WireError> {
    if batch.samples.len() > MAX_BATCH {
        return Err(WireError::malformed(
            "sample batch",
            format!("{} samples (at most {MAX_BATCH})", batch.samples.len()),
        ));
    }
    let head = pb::SampleBatch {
        channel: batch.channel.0,
        epoch: batch.epoch.0,
        publish_ns: batch.publish_ns,
        samples: batch
            .samples
            .iter()
            .map(|m| pb::SampleMeta {
                seq: m.seq,
                capture_ns: m.capture_ns,
                tick: m.tick,
                applied: m.applied.map(applied_to_pb),
            })
            .collect(),
        payload: Bytes::new(),
        truncated: batch.truncated,
        gaps: batch
            .gaps
            .iter()
            .map(|g| pb::SeqRange {
                first: g.first,
                last: g.last,
            })
            .collect(),
    }
    .encode_to_vec();
    Ok(Encoded::around(
        head,
        BATCH_PAYLOAD,
        &batch.payload,
        Vec::new(),
    ))
}

/// Decodes a batch. The payload slices `buf` (no copy).
pub fn decode_batch(buf: Bytes) -> Result<SampleBatch, WireError> {
    let b = pb::SampleBatch::decode(buf)?;
    if b.samples.len() > MAX_BATCH {
        return Err(WireError::malformed(
            "sample batch",
            format!("{} samples (at most {MAX_BATCH})", b.samples.len()),
        ));
    }
    Ok(SampleBatch {
        channel: ChannelId(b.channel),
        epoch: Epoch(b.epoch),
        publish_ns: b.publish_ns,
        samples: collect(b.samples, |m| {
            Ok::<_, WireError>(SampleMeta {
                seq: m.seq,
                capture_ns: m.capture_ns,
                tick: m.tick,
                applied: m.applied.map(applied_from_pb).transpose()?,
            })
        })?,
        payload: b.payload,
        truncated: b.truncated,
        gaps: b
            .gaps
            .into_iter()
            .map(|g| SeqRange {
                first: g.first,
                last: g.last,
            })
            .collect(),
    })
}

fn check_chunk_size(count: u32, bytes: usize) -> Result<(), WireError> {
    if count as usize > MAX_CHUNK_ACTIONS || bytes > MAX_CHUNK_BYTES {
        return Err(WireError::malformed(
            "action chunk",
            format!(
                "{count} actions in {bytes} bytes (at most {MAX_CHUNK_ACTIONS} in {MAX_CHUNK_BYTES})"
            ),
        ));
    }
    Ok(())
}

/// Encodes an action chunk. Its actions become a segment (no copy).
pub fn encode_chunk(chunk: &ActionChunk) -> Result<Encoded, WireError> {
    check_chunk_size(chunk.count, chunk.actions.len())?;
    let head = pb::ActionChunk {
        channel: chunk.channel.0,
        epoch: chunk.epoch.0,
        lease_token: chunk.lease_token.0,
        seq: chunk.seq,
        binding: Bytes::copy_from_slice(chunk.binding.as_bytes()),
        based_on: chunk
            .based_on
            .iter()
            .map(|r| pb::InputRange {
                channel: r.channel.0,
                seq_first: r.seq_first,
                seq_last: r.seq_last,
            })
            .collect(),
        t0_ns: chunk.t0_ns,
        dt_ns: chunk.dt_ns,
        replace_from_ns: chunk.replace_from_ns,
        valid_until_ns: chunk.valid_until_ns,
        count: chunk.count,
        actions: Bytes::new(),
        publish_ns: chunk.publish_ns,
        tick: chunk.tick,
        source: chunk.source.to_string(),
    }
    .encode_to_vec();
    Ok(Encoded::around(
        head,
        CHUNK_ACTIONS,
        &chunk.actions,
        Vec::new(),
    ))
}

/// Decodes an action chunk. The actions slice `buf` (no copy).
pub fn decode_chunk(buf: Bytes) -> Result<ActionChunk, WireError> {
    let c = pb::ActionChunk::decode(buf)?;
    check_chunk_size(c.count, c.actions.len())?;
    Ok(ActionChunk {
        channel: ChannelId(c.channel),
        epoch: Epoch(c.epoch),
        lease_token: LeaseToken(c.lease_token),
        seq: c.seq,
        source: ParticipantId::new(c.source)?,
        binding: fingerprint_from_pb("action chunk", &c.binding)?,
        based_on: c
            .based_on
            .into_iter()
            .map(|r| InputRange {
                channel: ChannelId(r.channel),
                seq_first: r.seq_first,
                seq_last: r.seq_last,
            })
            .collect(),
        t0_ns: c.t0_ns,
        dt_ns: c.dt_ns,
        replace_from_ns: c.replace_from_ns,
        valid_until_ns: c.valid_until_ns,
        count: c.count,
        actions: c.actions,
        publish_ns: c.publish_ns,
        tick: c.tick,
    })
}

// --- lockstep ----------------------------------------------------------------

fn endpoint_timing_to_pb(t: &EndpointTiming) -> pb::EndpointTiming {
    pb::EndpointTiming {
        total_ns: t.total_ns,
        decode_ns: t.decode_ns,
        user_ns: t.user_ns,
        encode_ns: t.encode_ns,
        queue_ns: t.queue_ns,
        in_flight: t.in_flight,
        adapter_ns: t.adapter_ns,
        held_episodes: t.held_episodes,
        held_state_bytes: t.held_state_bytes,
        lane_skew_ns: t.lane_skew_ns,
    }
}

fn endpoint_timing_from_pb(t: pb::EndpointTiming) -> EndpointTiming {
    EndpointTiming {
        total_ns: t.total_ns,
        decode_ns: t.decode_ns,
        user_ns: t.user_ns,
        encode_ns: t.encode_ns,
        queue_ns: t.queue_ns,
        in_flight: t.in_flight,
        adapter_ns: t.adapter_ns,
        held_episodes: t.held_episodes,
        held_state_bytes: t.held_state_bytes,
        lane_skew_ns: t.lane_skew_ns,
    }
}

fn values_to_pb(values: &[PartValues]) -> Vec<pb::PartValues> {
    values
        .iter()
        .map(|v| pb::PartValues {
            channel: v.channel.0,
            payload: v.payload.clone(),
        })
        .collect()
}

fn values_from_pb(values: Vec<pb::PartValues>) -> Vec<PartValues> {
    values
        .into_iter()
        .map(|v| PartValues {
            channel: ChannelId(v.channel),
            payload: v.payload,
        })
        .collect()
}

/// Encodes a lockstep reset.
pub fn encode_lockstep_reset(reset: &LockstepReset) -> Vec<u8> {
    pb::LockstepReset {
        epoch: reset.epoch.0,
        lanes: reset.lanes.clone(),
        seeds: reset.seeds.clone(),
        options_json: reset.options_json.clone(),
        episode_ids: reset.episode_ids.clone(),
    }
    .encode_to_vec()
}

/// Decodes a lockstep reset.
pub fn decode_lockstep_reset(buf: &[u8]) -> Result<LockstepReset, WireError> {
    let r = pb::LockstepReset::decode(buf)?;
    if !r.seeds.is_empty() && !r.lanes.is_empty() && r.seeds.len() != r.lanes.len() {
        return Err(WireError::malformed(
            "lockstep reset",
            format!("{} seeds for {} lanes", r.seeds.len(), r.lanes.len()),
        ));
    }
    Ok(LockstepReset {
        epoch: Epoch(r.epoch),
        lanes: r.lanes,
        seeds: r.seeds,
        options_json: r.options_json,
        episode_ids: r.episode_ids,
    })
}

/// Encodes a lockstep step.
pub fn encode_lockstep_step(step: &LockstepStep) -> Vec<u8> {
    pb::LockstepStep {
        epoch: step.epoch.0,
        step: step.step,
        lease_token: step.lease_token.0,
        lanes: step.lanes.clone(),
        commands: values_to_pb(&step.commands),
    }
    .encode_to_vec()
}

/// Decodes a lockstep step. The slabs slice `buf` (no copy).
pub fn decode_lockstep_step(buf: Bytes) -> Result<LockstepStep, WireError> {
    let s = pb::LockstepStep::decode(buf)?;
    Ok(LockstepStep {
        epoch: Epoch(s.epoch),
        step: s.step,
        lease_token: LeaseToken(s.lease_token),
        lanes: s.lanes,
        commands: values_from_pb(s.commands),
    })
}

/// Encodes a lockstep reply.
pub fn encode_lockstep_reply(reply: &LockstepReply) -> Vec<u8> {
    pb::LockstepReply {
        epoch: reply.epoch.0,
        step: reply.step,
        observations: values_to_pb(&reply.observations),
        reward: reply.reward.clone(),
        terminated: reply.terminated.clone(),
        truncated: reply.truncated.clone(),
        episode_ids: reply.episode_ids.clone(),
        info_json: reply.info_json.clone(),
        timing: reply.timing.as_ref().map(endpoint_timing_to_pb),
    }
    .encode_to_vec()
}

/// Decodes a lockstep reply. The slabs slice `buf` (no copy).
pub fn decode_lockstep_reply(buf: Bytes) -> Result<LockstepReply, WireError> {
    let r = pb::LockstepReply::decode(buf)?;
    Ok(LockstepReply {
        epoch: Epoch(r.epoch),
        step: r.step,
        observations: values_from_pb(r.observations),
        reward: r.reward,
        terminated: r.terminated,
        truncated: r.truncated,
        episode_ids: r.episode_ids,
        info_json: r.info_json,
        timing: r.timing.map(endpoint_timing_from_pb),
    })
}

// --- served models -----------------------------------------------------------

fn leaves_to_pb(leaves: &[Bytes]) -> pb::Leaves {
    pb::Leaves {
        leaves: leaves.to_vec(),
    }
}

fn leaves_from_pb(leaves: Option<pb::Leaves>) -> Vec<Bytes> {
    leaves.map(|l| l.leaves).unwrap_or_default()
}

fn episodes_to_pb(episodes: &[ModelEpisode]) -> Vec<pb::ModelEpisode> {
    episodes
        .iter()
        .map(|e| pb::ModelEpisode {
            id: e.id.clone(),
            seed: e.seed,
        })
        .collect()
}

fn episodes_from_pb(episodes: Vec<pb::ModelEpisode>) -> Vec<ModelEpisode> {
    episodes
        .into_iter()
        .map(|e| ModelEpisode {
            id: e.id,
            seed: e.seed,
        })
        .collect()
}

fn fault_to_pb(fault: &ModelFault) -> pb::ModelFault {
    pb::ModelFault {
        code: fault.code.clone(),
        message: fault.message.clone(),
        recoverable: fault.recoverable,
    }
}

fn fault_from_pb(fault: pb::ModelFault) -> ModelFault {
    ModelFault {
        code: fault.code,
        message: fault.message,
        recoverable: fault.recoverable,
    }
}

/// Encodes a request to a served model.
pub fn encode_model_request(request: &ModelRequest) -> Vec<u8> {
    use pb::model_request::Kind;
    let kind = match request {
        ModelRequest::Resolve(r) => Kind::Resolve(pb::ModelResolve {
            route: r.route.clone(),
            contract_json: r.contract_json.clone(),
            execution_horizon: r.execution_horizon,
            delivers_history: r.delivers_history,
            workflow_edition: r.workflow_edition.clone(),
        }),
        ModelRequest::Predict(groups) => Kind::Predict(pb::ModelPredict {
            groups: groups
                .iter()
                .map(|g| pb::ModelGroup {
                    route: g.route.clone(),
                    episodes: episodes_to_pb(&g.episodes),
                    observation: Some(leaves_to_pb(&g.observation)),
                    history: g
                        .history
                        .iter()
                        .map(|h| pb::ModelHistoryRow {
                            observation: Some(leaves_to_pb(&h.observation)),
                            episodes: episodes_to_pb(&h.episodes),
                            step: h.step,
                        })
                        .collect(),
                    step: g.step,
                })
                .collect(),
        }),
        ModelRequest::Reset { route, episode_ids } => Kind::Reset(pb::ModelReset {
            route: route.clone(),
            episode_ids: episode_ids.clone(),
        }),
        ModelRequest::Release { route } => Kind::Release(pb::ModelRelease {
            route: route.clone(),
        }),
        ModelRequest::Describe => Kind::Describe(pb::ModelDescribe {}),
    };
    pb::ModelRequest { kind: Some(kind) }.encode_to_vec()
}

/// Decodes a request to a served model. The slabs slice `buf` (no copy).
pub fn decode_model_request(buf: Bytes) -> Result<ModelRequest, WireError> {
    use pb::model_request::Kind;
    let kind = pb::ModelRequest::decode(buf)?
        .kind
        .ok_or_else(|| WireError::malformed("model request", "no kind (a newer request?)"))?;
    Ok(match kind {
        Kind::Resolve(r) => ModelRequest::Resolve(ModelResolve {
            route: r.route,
            contract_json: r.contract_json,
            execution_horizon: r.execution_horizon,
            delivers_history: r.delivers_history,
            workflow_edition: r.workflow_edition,
        }),
        Kind::Predict(p) => ModelRequest::Predict(
            p.groups
                .into_iter()
                .map(|g| ModelGroup {
                    route: g.route,
                    episodes: episodes_from_pb(g.episodes),
                    observation: leaves_from_pb(g.observation),
                    history: g
                        .history
                        .into_iter()
                        .map(|h| ModelHistoryRow {
                            observation: leaves_from_pb(h.observation),
                            episodes: episodes_from_pb(h.episodes),
                            step: h.step,
                        })
                        .collect(),
                    step: g.step,
                })
                .collect(),
        ),
        Kind::Reset(r) => ModelRequest::Reset {
            route: r.route,
            episode_ids: r.episode_ids,
        },
        Kind::Release(r) => ModelRequest::Release { route: r.route },
        Kind::Describe(_) => ModelRequest::Describe,
    })
}

/// Encodes a served model's reply and what the model measured answering.
pub fn encode_model_reply(reply: &ModelReply, timing: Option<&EndpointTiming>) -> Vec<u8> {
    use pb::model_reply::Kind;
    let kind = match reply {
        ModelReply::Resolved(r) => Kind::Resolved(pb::ModelResolved {
            native_chunk: r.native_chunk,
            history: r
                .history
                .as_ref()
                .map(|(keys, prunable)| pb::ModelHistoryNeeds {
                    keys: keys.clone(),
                    prunable: *prunable,
                }),
        }),
        ModelReply::Predicted(outcomes) => Kind::Predicted(pb::ModelPredicted {
            outcomes: outcomes
                .iter()
                .map(|o| pb::ModelOutcome {
                    outcome: Some(match o {
                        Ok(frames) => pb::model_outcome::Outcome::Frames(pb::ModelFrames {
                            frames: frames.iter().map(|f| leaves_to_pb(f)).collect(),
                        }),
                        Err(fault) => pb::model_outcome::Outcome::Fault(fault_to_pb(fault)),
                    }),
                })
                .collect(),
        }),
        ModelReply::Done => Kind::Done(pb::ModelDone {}),
        ModelReply::Described(json) => Kind::DescribeJson(json.clone()),
        ModelReply::Fault(fault) => Kind::Fault(fault_to_pb(fault)),
    };
    pb::ModelReply {
        kind: Some(kind),
        timing: timing.map(endpoint_timing_to_pb),
    }
    .encode_to_vec()
}

/// Decodes a served model's reply and its timing. The slabs slice `buf`
/// (no copy).
pub fn decode_model_reply(buf: Bytes) -> Result<(ModelReply, Option<EndpointTiming>), WireError> {
    use pb::model_reply::Kind;
    let reply = pb::ModelReply::decode(buf)?;
    let timing = reply.timing.map(endpoint_timing_from_pb);
    let kind = reply
        .kind
        .ok_or_else(|| WireError::malformed("model reply", "no kind (a newer reply?)"))?;
    let reply = match kind {
        Kind::Resolved(r) => ModelReply::Resolved(ModelResolved {
            native_chunk: r.native_chunk,
            history: r.history.map(|h| (h.keys, h.prunable)),
        }),
        Kind::Predicted(p) => ModelReply::Predicted(
            p.outcomes
                .into_iter()
                .map(|o| match o.outcome {
                    Some(pb::model_outcome::Outcome::Frames(f)) => {
                        Ok(f.frames.into_iter().map(|frame| frame.leaves).collect())
                    }
                    Some(pb::model_outcome::Outcome::Fault(fault)) => Err(fault_from_pb(fault)),
                    None => Err(ModelFault {
                        code: "INVALID_REQUEST".into(),
                        message: "an outcome with neither frames nor a fault".into(),
                        recoverable: false,
                    }),
                })
                .collect(),
        ),
        Kind::Done(_) => ModelReply::Done,
        Kind::DescribeJson(json) => ModelReply::Described(json),
        Kind::Fault(fault) => ModelReply::Fault(fault_from_pb(fault)),
    };
    Ok((reply, timing))
}

// --- conversions -------------------------------------------------------------

fn collect<T, U, E>(items: Vec<T>, f: impl FnMut(T) -> Result<U, E>) -> Result<Vec<U>, WireError>
where
    WireError: From<E>,
{
    Ok(items.into_iter().map(f).collect::<Result<Vec<U>, E>>()?)
}

fn bad_enum(what: &'static str, raw: i32) -> WireError {
    WireError::malformed(what, format!("unknown value {raw}"))
}

fn fingerprint_from_pb(what: &'static str, bytes: &[u8]) -> Result<Fingerprint, WireError> {
    Fingerprint::from_slice(bytes)
        .ok_or_else(|| WireError::malformed(what, format!("a {}-byte fingerprint", bytes.len())))
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

fn clock_mode_to_pb(mode: ClockMode) -> pb::ClockMode {
    match mode {
        ClockMode::Wall => pb::ClockMode::Wall,
        ClockMode::Virtual => pb::ClockMode::Virtual,
        ClockMode::Lockstep => pb::ClockMode::Lockstep,
    }
}

fn clock_mode_from_pb(raw: i32) -> Result<ClockMode, WireError> {
    match pb::ClockMode::try_from(raw) {
        Ok(pb::ClockMode::Wall | pb::ClockMode::Unspecified) => Ok(ClockMode::Wall),
        Ok(pb::ClockMode::Virtual) => Ok(ClockMode::Virtual),
        Ok(pb::ClockMode::Lockstep) => Ok(ClockMode::Lockstep),
        Err(_) => Err(bad_enum("clock mode", raw)),
    }
}

fn edge_mode_to_pb(mode: EdgeMode) -> pb::EdgeMode {
    match mode {
        EdgeMode::Idle => pb::EdgeMode::Idle,
        EdgeMode::Driving => pb::EdgeMode::Driving,
        EdgeMode::Hold => pb::EdgeMode::Hold,
        EdgeMode::Safe => pb::EdgeMode::Safe,
        EdgeMode::Estop => pb::EdgeMode::Estop,
    }
}

fn edge_mode_from_pb(raw: i32) -> Result<EdgeMode, WireError> {
    match pb::EdgeMode::try_from(raw) {
        Ok(pb::EdgeMode::Idle) => Ok(EdgeMode::Idle),
        Ok(pb::EdgeMode::Driving) => Ok(EdgeMode::Driving),
        Ok(pb::EdgeMode::Hold) => Ok(EdgeMode::Hold),
        Ok(pb::EdgeMode::Safe) => Ok(EdgeMode::Safe),
        Ok(pb::EdgeMode::Estop) => Ok(EdgeMode::Estop),
        _ => Err(bad_enum("edge mode", raw)),
    }
}

fn applied_to_pb(a: Applied) -> pb::Applied {
    pb::Applied {
        mode: edge_mode_to_pb(a.mode) as i32,
        present: a.present,
        lease_token: a.lease_token.0,
        chunk_seq: a.chunk_seq,
        action_index: a.action_index,
    }
}

fn applied_from_pb(a: pb::Applied) -> Result<Applied, WireError> {
    Ok(Applied {
        mode: edge_mode_from_pb(a.mode)?,
        present: a.present,
        lease_token: LeaseToken(a.lease_token),
        chunk_seq: a.chunk_seq,
        action_index: a.action_index,
    })
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
        CtlStatus::UnknownPath => pb::CtlStatus::UnknownPath,
        CtlStatus::Unsupported => pb::CtlStatus::Unsupported,
        CtlStatus::Incompatible => pb::CtlStatus::Incompatible,
        CtlStatus::GroupBusy => pb::CtlStatus::GroupBusy,
        CtlStatus::NotBound => pb::CtlStatus::NotBound,
        CtlStatus::NotReady => pb::CtlStatus::NotReady,
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
        Ok(pb::CtlStatus::UnknownPath) => CtlStatus::UnknownPath,
        Ok(pb::CtlStatus::Unsupported) => CtlStatus::Unsupported,
        Ok(pb::CtlStatus::Incompatible) => CtlStatus::Incompatible,
        Ok(pb::CtlStatus::GroupBusy) => CtlStatus::GroupBusy,
        Ok(pb::CtlStatus::NotBound) => CtlStatus::NotBound,
        Ok(pb::CtlStatus::NotReady) => CtlStatus::NotReady,
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
        supports: p.supports.clone(),
        requires: p.requires.clone(),
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
        binding: Bytes::copy_from_slice(l.binding.as_bytes()),
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
        binding: fingerprint_from_pb("lease", &l.binding)?,
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

fn contract_to_pb(c: &StreamContract) -> pb::StreamContract {
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

fn contract_from_pb(c: pb::StreamContract) -> Result<StreamContract, WireError> {
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

fn grid_to_pb(g: &ActionGrid) -> pb::ActionGrid {
    pb::ActionGrid {
        channel: g.channel.0,
        dt_ns: duration_ns(g.dt),
    }
}

fn grid_from_pb(g: pb::ActionGrid) -> ActionGrid {
    ActionGrid {
        channel: ChannelId(g.channel),
        dt: Duration::from_nanos(g.dt_ns),
    }
}

fn demand_to_pb(d: &Demand) -> pb::Demand {
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

fn demand_from_pb(d: pb::Demand) -> Result<Demand, WireError> {
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
    }
}

fn state_from_pb(s: pb::AuthorityState) -> Result<AuthoritySnapshot, WireError> {
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

fn watermark_to_pb(w: &Watermark) -> pb::Watermark {
    pb::Watermark {
        channel: w.channel.0,
        seq: w.seq,
    }
}

fn watermark_from_pb(w: pb::Watermark) -> Watermark {
    Watermark {
        channel: ChannelId(w.channel),
        seq: w.seq,
    }
}

fn chunk_ref_to_pb(c: &ChunkRef) -> pb::ChunkRef {
    pb::ChunkRef {
        channel: c.channel.0,
        lease_token: c.lease_token.0,
        seq: c.seq,
    }
}

fn chunk_ref_from_pb(c: pb::ChunkRef) -> ChunkRef {
    ChunkRef {
        channel: ChannelId(c.channel),
        lease_token: LeaseToken(c.lease_token),
        seq: c.seq,
    }
}

fn channels(ids: &[ChannelId]) -> Vec<u32> {
    ids.iter().map(|c| c.0).collect()
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
        } => Verb::Attach(pb::Attach {
            name: name.clone(),
            roles: roles.iter().map(|r| role_to_pb(*r) as i32).collect(),
            priority: priority_to_pb(*priority) as i32,
            protocol_minor: *protocol_minor,
            supports: supports.clone(),
            requires: requires.clone(),
        }),
        Command::Detach => Verb::Detach(pb::Detach {}),
        Command::Arm => Verb::Arm(pb::Arm {}),
        Command::Disarm => Verb::Disarm(pb::Disarm {}),
        Command::DeclareParts {
            spec_version,
            robot_json,
        } => Verb::DeclareParts(pb::DeclareParts {
            spec_version: *spec_version,
            robot_json: robot_json.clone(),
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
        } => Verb::ReleaseLease(pb::ReleaseLease {
            channels: channels(release),
            token: token.0,
        }),
        Command::Estop { reason } => Verb::Estop(pb::Estop {
            reason: reason.clone(),
        }),
        Command::Clear => Verb::Clear(pb::Clear {}),
        Command::EnterSafe { reason } => Verb::EnterSafe(pb::EnterSafe {
            reason: reason.clone(),
        }),
        Command::Ready { tick, watermarks } => Verb::Ready(pb::Ready {
            tick: *tick,
            watermarks: watermarks.iter().map(watermark_to_pb).collect(),
        }),
        Command::Decided { tick, chunks } => Verb::Decided(pb::Decided {
            tick: *tick,
            chunks: chunks.iter().map(chunk_ref_to_pb).collect(),
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
    let ids = |ids: Vec<u32>| ids.into_iter().map(ChannelId).collect::<Vec<_>>();
    let command = match verb {
        Verb::Attach(a) => Command::Attach {
            name: a.name,
            roles: collect(a.roles, role_from_pb)?,
            priority: priority_from_pb(a.priority)?,
            protocol_minor: a.protocol_minor,
            supports: a.supports,
            requires: a.requires,
        },
        Verb::Detach(_) => Command::Detach,
        Verb::Arm(_) => Command::Arm,
        Verb::Disarm(_) => Command::Disarm,
        Verb::DeclareParts(d) => Command::DeclareParts {
            spec_version: d.spec_version,
            robot_json: d.robot_json,
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
        },
        Verb::Estop(e) => Command::Estop { reason: e.reason },
        Verb::Clear(_) => Command::Clear,
        Verb::EnterSafe(e) => Command::EnterSafe { reason: e.reason },
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
    };
    Ok(Request {
        from: ParticipantId::new(r.participant)?,
        epoch: (r.epoch > 0).then_some(Epoch(r.epoch)),
        command,
    })
}

fn duration_ns(d: Duration) -> u64 {
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
            MergePolicy::Blend => pb::MergePolicy::Blend,
            MergePolicy::RtcPrefix => pb::MergePolicy::RtcPrefix,
        } as i32,
        reference: c.reference.clone(),
        group: c.group.clone().unwrap_or_default(),
        binary: c.binary,
        hold_last: c.hold_last,
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
            Ok(pb::MergePolicy::Blend) => MergePolicy::Blend,
            Ok(pb::MergePolicy::RtcPrefix) => MergePolicy::RtcPrefix,
            Err(_) => return Err(bad_enum("merge policy", c.merge)),
        },
        reference: c.reference,
        group: (!c.group.is_empty()).then_some(c.group),
        binary: c.binary,
        hold_last: c.hold_last,
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

fn channel_priority_from_pb(raw: i32) -> Result<ChannelPriority, WireError> {
    match pb::ChannelPriority::try_from(raw) {
        Ok(pb::ChannelPriority::Control) => Ok(ChannelPriority::Control),
        Ok(pb::ChannelPriority::Realtime) => Ok(ChannelPriority::Realtime),
        Ok(pb::ChannelPriority::Data) => Ok(ChannelPriority::Data),
        Ok(pb::ChannelPriority::Bulk) => Ok(ChannelPriority::Bulk),
        _ => Err(bad_enum("channel priority", raw)),
    }
}

fn part_to_pb(p: &PartSpec) -> pb::PartSpec {
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

fn part_from_pb(p: pb::PartSpec) -> Result<PartSpec, WireError> {
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
        attrs: p.attrs.into_iter().collect::<BTreeMap<_, _>>(),
        priority: channel_priority_from_pb(p.priority)?,
        owner: ParticipantId::new(p.owner)?,
        event: p.event,
        requires: p.requires,
        control_hz: p.control_hz,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::ids::SessionId;
    use crate::wire::part::tests::{arm, camera, pid};
    use crate::wire::version::PROTOCOL_MINOR;

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
                supports: vec!["frame.fixed".into()],
                requires: vec!["merge.blend".into()],
            }],
            leases: vec![Lease {
                channel: ChannelId(1),
                holder: pid("op"),
                priority: PriorityClass::Operator,
                token: LeaseToken(4),
                ttl_ns: 5,
                granted_ns: 6,
                expires_ns: 11,
                binding: Fingerprint::from(77),
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
            parts: vec![camera(), arm()],
            protocol_minor: PROTOCOL_MINOR,
            supports: Vec::new(),
            clock_mode: ClockMode::Virtual,
            tick_ns: 1_000_000,
            bindings: vec![BindingRecord {
                participant: pid("op"),
                fingerprint: Fingerprint::from(77),
                revision: 2,
                table_json: "{}".into(),
                channels: vec![ChannelId(1)],
                replan_hz: 20.0,
                contracts: vec![StreamContract {
                    channel: ChannelId(3),
                    mode: ContractMode::History,
                    delivery_hz: 20.0,
                    history: 50,
                    max_age: Some(Duration::from_millis(150)),
                    max_age_explicit: false,
                }],
                grids: vec![ActionGrid {
                    channel: ChannelId(1),
                    dt: Duration::from_millis(10),
                }],
            }],
            contract: [(pid("ur10e"), Fingerprint::from(0xabcd))].into(),
            demands: vec![
                Demand {
                    channel: ChannelId(3),
                    revision: 2,
                    plan: Plan::Every,
                    retain: Duration::from_secs(2),
                    acked: 1,
                },
                Demand {
                    channel: ChannelId(4),
                    revision: 1,
                    plan: Plan::Latest { hz: 15.0 },
                    retain: Duration::ZERO,
                    acked: 0,
                },
                Demand {
                    channel: ChannelId(5),
                    revision: 1,
                    plan: Plan::Every,
                    retain: Duration::ZERO,
                    acked: 1,
                },
            ],
        }
    }

    #[test]
    fn binding_records_round_trip() {
        let state = snapshot();
        let binding = &state.bindings[0];
        let decoded = decode_binding(&encode_binding(binding)).expect("decodes");
        assert_eq!(&decoded, binding);
    }

    #[test]
    fn descriptors_name_every_recorded_envelope() {
        let names = String::from_utf8_lossy(FILE_DESCRIPTOR_SET);
        assert!(names.contains("rlmesh/session/v2/session.proto"));
        for full in [
            schema::SAMPLE,
            schema::ACTION_CHUNK,
            schema::AUTHORITY_STATE,
            schema::BINDING_RECORD,
        ] {
            let message = full.rsplit('.').next().expect("a message name");
            assert!(names.contains(message), "{message} is not described");
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
                supports: vec!["frame.fixed".into()],
                requires: vec!["merge.blend".into()],
            },
            Command::DeclareParts {
                spec_version: 1,
                robot_json: r#"{"arm": {"kind": "x/arm"}}"#.into(),
            },
            Command::DeclareBinding {
                fingerprint: Fingerprint::from(9),
                revision: 4,
                table_json: "{}".into(),
                channels: vec![ChannelId(1), ChannelId(5)],
                replan_hz: 20.0,
                contracts: vec![StreamContract {
                    channel: ChannelId(2),
                    mode: ContractMode::Latest,
                    delivery_hz: 20.0,
                    history: 1,
                    max_age: None,
                    max_age_explicit: false,
                }],
                grids: vec![ActionGrid {
                    channel: ChannelId(5),
                    dt: Duration::from_nanos(66_666_667),
                }],
            },
            Command::DemandAck {
                channel: ChannelId(3),
                revision: 2,
            },
            Command::Arm,
            Command::Disarm,
            Command::Detach,
            Command::Clear,
            Command::GrantLease {
                channels: vec![ChannelId(4), ChannelId(6)],
                release: vec![ChannelId(5)],
                binding: Fingerprint::from(9),
                ttl_ns: Some(5),
            },
            Command::GrantLease {
                channels: vec![ChannelId(4)],
                release: Vec::new(),
                binding: Fingerprint::from(9),
                ttl_ns: None,
            },
            Command::ReleaseLease {
                channels: vec![ChannelId(4)],
                token: LeaseToken(2),
            },
            Command::Estop {
                reason: "red button".into(),
            },
            Command::EnterSafe {
                reason: "queue dry".into(),
            },
            Command::Ready {
                tick: 7,
                watermarks: vec![Watermark {
                    channel: ChannelId(2),
                    seq: 8,
                }],
            },
            Command::Decided {
                tick: 7,
                chunks: vec![ChunkRef {
                    channel: ChannelId(1),
                    lease_token: LeaseToken(3),
                    seq: 9,
                }],
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
            Err(CtlError::new(CtlStatus::GroupBusy, "busy")),
            Err(CtlError::new(CtlStatus::NotBound, "not in the binding")),
            Err(CtlError::new(CtlStatus::UnknownPath, "no such part")),
            Err(CtlError::new(
                CtlStatus::NotReady,
                "wrist/cam has not installed its plan",
            )),
        ] {
            let reply = CtlReply {
                result,
                leases: state.leases.clone(),
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
        for run in [
            None,
            Some(RunCounters {
                starts: 120,
                completions: 118,
                busy_skips: 4,
                waiting_for: 2,
                stale_input: 1,
                window_gap: 1,
                decode: 0,
                no_lease: 9,
                chunks_sent: 118,
                pending_ns: 2_000_000_000,
                latency_p50_ns: 8_000_000,
                latency_p99_ns: 31_000_000,
                drifting: true,
            }),
        ] {
            let report = ParticipantReport {
                participant: pid("policy"),
                epoch: Epoch(3),
                sent_ns: -9,
                streams: vec![PublisherCounters {
                    channel: ChannelId(4),
                    published: 600,
                    coalesced: 300,
                }],
                run,
            };
            assert_eq!(
                decode_report(&encode_report(&report)).expect("decode"),
                report
            );
        }
        assert!(
            decode_report(b"\x0a\x01!").is_err(),
            "an invalid participant id"
        );
        assert!(decode_state(b"\xff\xff\xff").is_err());

        for tick in [
            VirtualTick::Observe(TickState {
                epoch: Epoch(2),
                tick: 40,
                watermarks: vec![Watermark {
                    channel: ChannelId(1),
                    seq: 41,
                }],
                due: vec![pid("policy")],
            }),
            VirtualTick::Step(Step {
                epoch: Epoch(2),
                tick: 40,
                chunks: vec![ChunkRef {
                    channel: ChannelId(1),
                    lease_token: LeaseToken(3),
                    seq: 2,
                }],
            }),
        ] {
            assert_eq!(decode_tick(&encode_tick(&tick)).expect("decode"), tick);
        }
        let query = HistoryQuery {
            epoch: Epoch(2),
            after_seq: 10,
            until_seq: Some(20),
            limit: None,
        };
        assert_eq!(
            decode_history_query(&encode_history_query(&query)).expect("decode"),
            query
        );
    }

    #[test]
    fn a_bad_fingerprint_or_part_is_malformed() {
        let mut state = state_to_pb(&snapshot());
        state.leases[0].binding = Bytes::from_static(&[1, 2, 3]);
        assert!(state_from_pb(state).is_err());
        let mut state = state_to_pb(&snapshot());
        state.parts[0].path = "Wrist/Cam".into();
        assert!(state_from_pb(state).is_err(), "an invalid path");
        let mut state = state_to_pb(&snapshot());
        state.parts[1].fields[0].dtype = 13;
        assert!(
            state_from_pb(state).is_err(),
            "bfloat16 is not a v0.2 dtype"
        );
        let mut state = state_to_pb(&snapshot());
        state.parts[0].quality = 0;
        assert!(state_from_pb(state).is_err());
    }

    fn applied_sample() -> Sample {
        Sample {
            channel: ChannelId(2),
            epoch: Epoch(3),
            seq: 41,
            capture_ns: 100,
            publish_ns: 120,
            tick: 7,
            payload: Bytes::from(vec![7_u8; 24]),
            applied: Some(Applied {
                mode: EdgeMode::Hold,
                present: true,
                lease_token: LeaseToken(4),
                chunk_seq: 9,
                action_index: 3,
            }),
        }
    }

    #[test]
    fn samples_are_plain_protobuf_with_the_payload_neither_copied_on_send_nor_on_receipt() {
        let sample = applied_sample();
        let encoded = encode_sample(&sample);
        assert!(
            encoded
                .segments
                .iter()
                .any(|s| s.as_ptr() == sample.payload.as_ptr()),
            "the payload is a segment of its own"
        );
        let buf = Bytes::from(encoded.to_vec());
        // Any protobuf implementation reads it.
        let plain = pb::Sample::decode(&buf[..]).expect("plain protobuf");
        assert_eq!((plain.seq, plain.tick, plain.payload.len()), (41, 7, 24));
        assert_eq!(plain.applied.map(|a| a.chunk_seq), Some(9));
        let decoded = decode_sample(buf.clone()).expect("decode");
        assert_eq!(decoded, sample);
        assert!(
            buf.as_ptr_range().contains(&decoded.payload.as_ptr()),
            "the payload slices the received buffer"
        );
        // An applied tick with nothing dispatched has no payload.
        let absent = Sample {
            payload: Bytes::new(),
            applied: Some(Applied::absent(EdgeMode::Safe)),
            ..sample
        };
        assert_eq!(
            decode_sample(Bytes::from(encode_sample(&absent).to_vec())).expect("decode"),
            absent
        );
        let mut bad = pb::Sample::decode(&buf[..]).expect("plain");
        bad.applied = Some(pb::Applied {
            mode: 99,
            ..pb::Applied::default()
        });
        assert!(decode_sample(Bytes::from(bad.encode_to_vec())).is_err());
        assert!(decode_sample(Bytes::from_static(b"\xff\xff")).is_err());
    }

    #[test]
    fn batches_round_trip_with_their_gaps() {
        let batch = SampleBatch {
            channel: ChannelId(2),
            epoch: Epoch(3),
            publish_ns: 5,
            samples: vec![
                SampleMeta {
                    seq: 4,
                    capture_ns: 1,
                    tick: 0,
                    applied: None,
                },
                SampleMeta {
                    seq: 6,
                    capture_ns: 2,
                    tick: 0,
                    applied: applied_sample().applied,
                },
            ],
            payload: Bytes::from(vec![1_u8; 48]),
            truncated: true,
            gaps: vec![SeqRange { first: 5, last: 5 }],
        };
        let buf = Bytes::from(encode_batch(&batch).expect("encode").to_vec());
        let decoded = decode_batch(buf.clone()).expect("decode");
        assert_eq!(decoded, batch);
        assert!(buf.as_ptr_range().contains(&decoded.payload.as_ptr()));
        let mut huge = batch;
        huge.samples = vec![huge.samples[0]; MAX_BATCH + 1];
        assert!(encode_batch(&huge).is_err());
    }

    fn chunk() -> ActionChunk {
        ActionChunk {
            channel: ChannelId(1),
            epoch: Epoch(2),
            lease_token: LeaseToken(3),
            seq: 9,
            source: pid("policy"),
            binding: Fingerprint::from(0xfeed),
            based_on: vec![
                InputRange::one(ChannelId(2), 41),
                InputRange {
                    channel: ChannelId(5),
                    seq_first: 7,
                    seq_last: 10,
                },
            ],
            t0_ns: 1_000,
            dt_ns: 1_000_000,
            replace_from_ns: 900,
            valid_until_ns: 500_000_000,
            count: 5,
            actions: Bytes::from((0..120_u8).collect::<Vec<_>>()),
            publish_ns: 78,
            tick: 0,
        }
    }

    #[test]
    fn chunks_round_trip_through_plain_protobuf_without_copying_the_actions() {
        let chunk = chunk();
        let encoded = encode_chunk(&chunk).expect("encode");
        let buf = Bytes::from(encoded.to_vec());
        let plain = pb::ActionChunk::decode(&buf[..]).expect("plain protobuf");
        assert_eq!((plain.count, plain.actions.len()), (5, 120));
        let decoded = decode_chunk(buf.clone()).expect("decode");
        assert_eq!(decoded, chunk);
        assert!(buf.as_ptr_range().contains(&decoded.actions.as_ptr()));
        assert_eq!(
            decoded.action(4, 24).map(|a| a[0]),
            Some(96),
            "action 4 at a 24-byte stride"
        );
        assert_eq!(decoded.action(5, 24), None);
        assert_eq!(decoded.reference().seq, 9);

        let mut huge = chunk.clone();
        huge.count = u32::try_from(MAX_CHUNK_ACTIONS + 1).expect("count");
        assert!(encode_chunk(&huge).is_err());
        let mut no_binding = pb::ActionChunk::decode(&buf[..]).expect("plain");
        no_binding.binding = Bytes::new();
        assert!(decode_chunk(Bytes::from(no_binding.encode_to_vec())).is_err());
    }

    #[test]
    fn lockstep_messages_round_trip() {
        let reset = LockstepReset {
            epoch: Epoch(2),
            lanes: vec![0, 3],
            seeds: vec![11, 12],
            options_json: "{}".into(),
            episode_ids: vec![100, 101],
        };
        assert_eq!(
            decode_lockstep_reset(&encode_lockstep_reset(&reset)).expect("decode"),
            reset
        );
        let mut bad = reset.clone();
        bad.seeds.push(13);
        assert!(decode_lockstep_reset(&encode_lockstep_reset(&bad)).is_err());
        let step = LockstepStep {
            epoch: Epoch(2),
            step: 41,
            lease_token: LeaseToken(3),
            lanes: Vec::new(),
            commands: vec![PartValues {
                channel: ChannelId(1),
                payload: Bytes::from(vec![1_u8; 32]),
            }],
        };
        let buf = Bytes::from(encode_lockstep_step(&step));
        let decoded = decode_lockstep_step(buf.clone()).expect("decode");
        assert_eq!(decoded, step);
        assert!(
            buf.as_ptr_range()
                .contains(&decoded.commands[0].payload.as_ptr())
        );
        let reply = LockstepReply {
            epoch: Epoch(2),
            step: 41,
            observations: vec![PartValues {
                channel: ChannelId(4),
                payload: Bytes::from(vec![2_u8; 16]),
            }],
            reward: Bytes::from(1.5_f32.to_le_bytes().to_vec()),
            terminated: Bytes::from_static(&[0]),
            truncated: Bytes::from_static(&[1]),
            episode_ids: vec![7],
            info_json: String::new(),
            timing: Some(EndpointTiming {
                total_ns: Some(9),
                lane_skew_ns: Some(1),
                ..EndpointTiming::default()
            }),
        };
        assert_eq!(
            decode_lockstep_reply(Bytes::from(encode_lockstep_reply(&reply))).expect("decode"),
            reply
        );
    }

    #[test]
    fn model_messages_round_trip() {
        let slab = |b: u8| vec![Bytes::from(vec![b; 8]), Bytes::from(vec![b; 3])];
        let episode = |id: &str| ModelEpisode {
            id: id.into(),
            seed: Some(-4),
        };
        for request in [
            ModelRequest::Resolve(ModelResolve {
                route: "cartpole".into(),
                contract_json: "{}".into(),
                execution_horizon: 4,
                delivers_history: true,
                workflow_edition: "2026.06".into(),
            }),
            ModelRequest::Predict(vec![ModelGroup {
                route: "cartpole".into(),
                episodes: vec![episode("a"), episode("b")],
                observation: slab(1),
                history: vec![ModelHistoryRow {
                    observation: slab(2),
                    episodes: vec![episode("a")],
                    step: 3,
                }],
                step: Some(4),
            }]),
            ModelRequest::Reset {
                route: "cartpole".into(),
                episode_ids: vec!["a".into()],
            },
            ModelRequest::Release {
                route: "cartpole".into(),
            },
            ModelRequest::Describe,
        ] {
            let decoded = decode_model_request(Bytes::from(encode_model_request(&request)));
            assert_eq!(decoded.expect("decode"), request);
        }
        let fault = ModelFault {
            code: "INTERNAL".into(),
            message: "no".into(),
            recoverable: true,
        };
        for reply in [
            ModelReply::Resolved(ModelResolved {
                native_chunk: Some(8),
                history: Some((vec!["image".into()], false)),
            }),
            ModelReply::Predicted(vec![Ok(vec![slab(5), slab(6)]), Err(fault.clone())]),
            ModelReply::Done,
            ModelReply::Described("{\"kind\":\"model\"}".into()),
            ModelReply::Fault(fault),
        ] {
            let timing = EndpointTiming {
                user_ns: Some(7),
                held_episodes: Some(2),
                ..EndpointTiming::default()
            };
            let decoded =
                decode_model_reply(Bytes::from(encode_model_reply(&reply, Some(&timing))));
            assert_eq!(decoded.expect("decode"), (reply, Some(timing)));
        }
        assert!(decode_model_request(Bytes::new()).is_err());
    }

    #[test]
    fn unknown_enum_values_are_refused_but_unknown_statuses_are_refusals() {
        assert!(mode_from_pb(0).is_err());
        assert!(mode_from_pb(99).is_err());
        assert!(edge_mode_from_pb(0).is_err());
        assert!(channel_priority_from_pb(0).is_err());
        assert_eq!(clock_mode_from_pb(0).ok(), Some(ClockMode::Wall));
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
