//! Samples, batches and action chunks: encoded as segments so the payload
//! is never copied.

use super::bound;
use super::convert::{bad_enum, collect, fingerprint_from_pb};
use super::pb;
use super::state::{reason_code_from_pb, reason_code_to_pb};
use crate::prelude::*;

use bytes::Bytes;
use prost::Message as _;
use prost::encoding::encode_varint;

use crate::wire::action::{ActionChunk, InputRange};
use crate::wire::error::WireError;
use crate::wire::ids::{ChannelId, Epoch, LeaseToken, ParticipantId};
use crate::wire::part::{MAX_BATCH, MAX_CHUNK_ACTIONS, MAX_CHUNK_BYTES};
use crate::wire::sample::{Applied, EdgeMode, Sample, SampleBatch, SampleMeta, SeqRange};

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
        request_id: batch.request_id,
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
        request_id: b.request_id,
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
    bound::check(&buf, &bound::CHUNK)?;
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

fn edge_mode_to_pb(mode: EdgeMode) -> pb::EdgeMode {
    match mode {
        EdgeMode::Idle => pb::EdgeMode::Idle,
        EdgeMode::Driving => pb::EdgeMode::Driving,
        EdgeMode::Hold => pb::EdgeMode::Hold,
        EdgeMode::Safe => pb::EdgeMode::Safe,
        EdgeMode::Estop => pb::EdgeMode::Estop,
    }
}

pub(super) fn edge_mode_from_pb(raw: i32) -> Result<EdgeMode, WireError> {
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
        reason: reason_code_to_pb(a.reason),
    }
}

fn applied_from_pb(a: pb::Applied) -> Result<Applied, WireError> {
    Ok(Applied {
        mode: edge_mode_from_pb(a.mode)?,
        present: a.present,
        lease_token: LeaseToken(a.lease_token),
        chunk_seq: a.chunk_seq,
        action_index: a.action_index,
        reason: reason_code_from_pb(a.reason),
    })
}
