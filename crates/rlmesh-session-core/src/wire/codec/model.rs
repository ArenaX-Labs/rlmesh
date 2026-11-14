//! Served-model requests and replies.

use super::lockstep::{endpoint_timing_from_pb, endpoint_timing_to_pb};
use super::pb;
use crate::prelude::*;

use bytes::Bytes;
use prost::Message as _;

use crate::wire::error::WireError;
use crate::wire::model::{
    ModelEpisode, ModelFault, ModelGroup, ModelHistoryRow, ModelReply, ModelRequest, ModelResolve,
    ModelResolved,
};
use crate::wire::timing::EndpointTiming;

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
            reads_advisories: r.reads_advisories,
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
        ModelRequest::ResetOptions {
            route,
            episode,
            trial_index,
        } => Kind::ResetOptions(pb::ModelResetOptions {
            route: route.clone(),
            episode: episodes_to_pb(core::slice::from_ref(episode)).pop(),
            trial_index: *trial_index,
        }),
        ModelRequest::Join { session } => Kind::Join(pb::ModelJoin {
            session: session.clone(),
        }),
        ModelRequest::Leave { session } => Kind::Leave(pb::ModelLeave {
            session: session.clone(),
        }),
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
            reads_advisories: r.reads_advisories,
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
        Kind::ResetOptions(r) => ModelRequest::ResetOptions {
            route: r.route,
            episode: episodes_from_pb(r.episode.into_iter().collect())
                .pop()
                .unwrap_or_default(),
            trial_index: r.trial_index,
        },
        Kind::Join(j) => ModelRequest::Join { session: j.session },
        Kind::Leave(l) => ModelRequest::Leave { session: l.session },
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
            reset_options: r.reset_options,
            advisories: r
                .advisories
                .iter()
                .map(|(severity, message)| pb::ModelAdvisory {
                    severity: severity.clone(),
                    message: message.clone(),
                })
                .collect(),
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
        ModelReply::Options(json) => Kind::OptionsJson(json.clone()),
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
            reset_options: r.reset_options,
            advisories: r
                .advisories
                .into_iter()
                .map(|a| (a.severity, a.message))
                .collect(),
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
        Kind::OptionsJson(json) => ModelReply::Options(json),
        Kind::Fault(fault) => ModelReply::Fault(fault_from_pb(fault)),
    };
    Ok((reply, timing))
}
