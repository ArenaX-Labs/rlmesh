//! Lockstep steps, requests and replies.

use super::pb;
use crate::prelude::*;

use bytes::Bytes;
use prost::Message as _;

use crate::wire::error::WireError;
use crate::wire::ids::{ChannelId, Epoch, LeaseToken};
use crate::wire::lockstep::{
    LockstepOp, LockstepReply, LockstepRequest, LockstepReset, LockstepResponse, LockstepResult,
    LockstepStep, PartValues,
};
use crate::wire::timing::EndpointTiming;

pub(super) fn endpoint_timing_to_pb(t: &EndpointTiming) -> pb::EndpointTiming {
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

pub(super) fn endpoint_timing_from_pb(t: pb::EndpointTiming) -> EndpointTiming {
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
    lockstep_reply_to_pb(reply).encode_to_vec()
}

fn lockstep_reply_to_pb(reply: &LockstepReply) -> pb::LockstepReply {
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
}

/// Decodes a lockstep reply. The slabs slice `buf` (no copy).
pub fn decode_lockstep_reply(buf: Bytes) -> Result<LockstepReply, WireError> {
    Ok(lockstep_reply_from_pb(pb::LockstepReply::decode(buf)?))
}

fn lockstep_reply_from_pb(r: pb::LockstepReply) -> LockstepReply {
    LockstepReply {
        epoch: Epoch(r.epoch),
        step: r.step,
        observations: values_from_pb(r.observations),
        reward: r.reward,
        terminated: r.terminated,
        truncated: r.truncated,
        episode_ids: r.episode_ids,
        info_json: r.info_json,
        timing: r.timing.map(endpoint_timing_from_pb),
    }
}

/// Encodes a lockstep request.
pub fn encode_lockstep_request(request: &LockstepRequest) -> Vec<u8> {
    use pb::lockstep_request::Op;
    let op = match &request.op {
        LockstepOp::Step(step) => Op::Step(pb::LockstepStep {
            epoch: step.epoch.0,
            step: step.step,
            lease_token: step.lease_token.0,
            lanes: step.lanes.clone(),
            commands: values_to_pb(&step.commands),
        }),
        LockstepOp::Reset(reset) => Op::Reset(pb::LockstepReset {
            epoch: reset.epoch.0,
            lanes: reset.lanes.clone(),
            seeds: reset.seeds.clone(),
            options_json: reset.options_json.clone(),
            episode_ids: reset.episode_ids.clone(),
            lease_token: reset.lease_token.0,
        }),
        LockstepOp::Render(lane) => Op::Render(*lane),
        LockstepOp::Probe => Op::Probe(pb::LockstepProbe {}),
    };
    pb::LockstepRequest {
        request_id: request.request_id,
        op: Some(op),
    }
    .encode_to_vec()
}

/// Decodes a lockstep request. A step's slabs slice `buf` (no copy).
pub fn decode_lockstep_request(buf: Bytes) -> Result<LockstepRequest, WireError> {
    use pb::lockstep_request::Op;
    let r = pb::LockstepRequest::decode(buf)?;
    let op = match r.op {
        Some(Op::Step(s)) => LockstepOp::Step(LockstepStep {
            epoch: Epoch(s.epoch),
            step: s.step,
            lease_token: LeaseToken(s.lease_token),
            lanes: s.lanes,
            commands: values_from_pb(s.commands),
        }),
        Some(Op::Reset(r)) => {
            if !r.seeds.is_empty() && !r.lanes.is_empty() && r.seeds.len() != r.lanes.len() {
                return Err(WireError::malformed(
                    "lockstep reset",
                    format!("{} seeds for {} lanes", r.seeds.len(), r.lanes.len()),
                ));
            }
            LockstepOp::Reset(LockstepReset {
                epoch: Epoch(r.epoch),
                lanes: r.lanes,
                seeds: r.seeds,
                options_json: r.options_json,
                episode_ids: r.episode_ids,
                lease_token: LeaseToken(r.lease_token),
            })
        }
        Some(Op::Render(lane)) => LockstepOp::Render(lane),
        Some(Op::Probe(_)) => LockstepOp::Probe,
        None => return Err(WireError::malformed("lockstep request", "no operation")),
    };
    Ok(LockstepRequest {
        request_id: r.request_id,
        op,
    })
}

/// Encodes a lockstep response.
pub fn encode_lockstep_response(response: &LockstepResponse) -> Vec<u8> {
    use pb::lockstep_response::Result as R;
    let result = match &response.result {
        LockstepResult::Obs(reply) => R::Obs(lockstep_reply_to_pb(reply)),
        LockstepResult::Frame(frame) => R::Frame(frame.clone()),
        LockstepResult::Probe => R::Probe(pb::LockstepProbe {}),
        LockstepResult::Failure(failure) => R::Failure(failure.clone()),
    };
    pb::LockstepResponse {
        request_id: response.request_id,
        result: Some(result),
    }
    .encode_to_vec()
}

/// Decodes a lockstep response. Slabs and frames slice `buf` (no copy).
pub fn decode_lockstep_response(buf: Bytes) -> Result<LockstepResponse, WireError> {
    use pb::lockstep_response::Result as R;
    let r = pb::LockstepResponse::decode(buf)?;
    let result = match r.result {
        Some(R::Obs(reply)) => LockstepResult::Obs(Box::new(lockstep_reply_from_pb(reply))),
        Some(R::Frame(frame)) => LockstepResult::Frame(frame),
        Some(R::Probe(_)) => LockstepResult::Probe,
        Some(R::Failure(failure)) => LockstepResult::Failure(failure),
        None => return Err(WireError::malformed("lockstep response", "no result")),
    };
    Ok(LockstepResponse {
        request_id: r.request_id,
        result,
    })
}
