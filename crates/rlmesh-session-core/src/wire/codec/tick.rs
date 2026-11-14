//! Virtual ticks, history queries and the tick verbs' watermarks and chunk
//! references.

use super::convert::collect;
use super::pb;
use crate::prelude::*;

use prost::Message as _;

use crate::wire::action::ChunkRef;
use crate::wire::authority::Watermark;
use crate::wire::error::WireError;
use crate::wire::ids::{ChannelId, Epoch, LeaseToken, ParticipantId};
use crate::wire::sample::HistoryQuery;
use crate::wire::tick::{Step, TickState, VirtualTick};

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
        request_id: query.request_id,
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
        request_id: q.request_id,
    })
}

pub(super) fn watermark_to_pb(w: &Watermark) -> pb::Watermark {
    pb::Watermark {
        channel: w.channel.0,
        seq: w.seq,
    }
}

pub(super) fn watermark_from_pb(w: pb::Watermark) -> Watermark {
    Watermark {
        channel: ChannelId(w.channel),
        seq: w.seq,
    }
}

pub(super) fn chunk_ref_to_pb(c: &ChunkRef) -> pb::ChunkRef {
    pb::ChunkRef {
        channel: c.channel.0,
        lease_token: c.lease_token.0,
        seq: c.seq,
    }
}

pub(super) fn chunk_ref_from_pb(c: pb::ChunkRef) -> ChunkRef {
    ChunkRef {
        channel: ChannelId(c.channel),
        lease_token: LeaseToken(c.lease_token),
        seq: c.seq,
    }
}
