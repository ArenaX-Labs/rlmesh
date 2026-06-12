use rlmesh_proto::env::v1::{
    RenderRequest, RenderResponse, ResetRequest, ResetResponse, StepRequest, StepResponse,
};
use rlmesh_spaces as native;

use crate::error::ProtocolError;
use crate::wire::spaces::{meta_map_from_struct, meta_map_to_struct};

use super::payload::{decode_value, encode_value};

pub fn reset_request_to_proto(
    request: &native::ResetRequest,
) -> Result<ResetRequest, ProtocolError> {
    Ok(ResetRequest {
        seeds: request.seed.into_iter().collect(),
        options: request.options.as_ref().map(meta_map_to_struct),
        timeout_ms: request.timeout_ms,
    })
}

pub fn reset_result_from_proto(
    response: ResetResponse,
    observation_space: &native::SpaceSpec,
) -> Result<native::ResetResult, ProtocolError> {
    Ok(native::ResetResult {
        observation: decode_value(response.observation.as_ref(), observation_space)?,
        info: response.infos.map(meta_map_from_struct),
        episode_id: response.episode_ids.into_iter().next(),
    })
}

pub fn step_request_to_proto(
    request: &native::StepRequest,
    action_space: &native::SpaceSpec,
) -> Result<StepRequest, ProtocolError> {
    Ok(StepRequest {
        action: request
            .action
            .as_ref()
            .map(|value| encode_value(value, action_space))
            .transpose()?,
        timeout_ms: request.timeout_ms,
    })
}

pub fn step_result_from_proto(
    response: StepResponse,
    observation_space: &native::SpaceSpec,
) -> Result<native::StepResult, ProtocolError> {
    Ok(native::StepResult {
        observation: decode_value(response.observation.as_ref(), observation_space)?,
        reward: response.rewards.first().copied().unwrap_or_default(),
        terminated: response
            .terminated_mask
            .first()
            .copied()
            .unwrap_or_default()
            != 0,
        truncated: response.truncated_mask.first().copied().unwrap_or_default() != 0,
        info: response.infos.map(meta_map_from_struct),
    })
}

pub fn render_request_to_proto(request: &native::RenderRequest) -> RenderRequest {
    RenderRequest {
        mask: request.env_index.map(render_mask).unwrap_or_default(),
        timeout_ms: request.timeout_ms,
    }
}

pub fn render_result_from_proto(
    response: RenderResponse,
) -> Result<native::RenderResult, ProtocolError> {
    Ok(native::RenderResult {
        frame: response
            .png_frame
            .map(|png_frame| native::RenderFrame { png_frame }),
    })
}

fn render_mask(env_index: usize) -> Vec<u8> {
    let mut mask = vec![0_u8; env_index + 1];
    if let Some(byte) = mask.get_mut(env_index) {
        *byte = 1;
    }
    mask
}

pub fn reset_result_to_proto(
    result: &native::ResetResult,
    observation_space: &native::SpaceSpec,
) -> Result<ResetResponse, ProtocolError> {
    Ok(ResetResponse {
        observation: result
            .observation
            .as_ref()
            .map(|value| encode_value(value, observation_space))
            .transpose()?,
        infos: result.info.as_ref().map(meta_map_to_struct),
        episode_ids: result.episode_id.iter().cloned().collect(),
    })
}

pub fn step_result_to_proto(
    result: &native::StepResult,
    observation_space: &native::SpaceSpec,
) -> Result<StepResponse, ProtocolError> {
    Ok(StepResponse {
        observation: result
            .observation
            .as_ref()
            .map(|value| encode_value(value, observation_space))
            .transpose()?,
        rewards: vec![result.reward],
        terminated_mask: vec![u8::from(result.terminated)],
        truncated_mask: vec![u8::from(result.truncated)],
        infos: result.info.as_ref().map(meta_map_to_struct),
        completed_episodes: vec![],
        episode_ids: vec![],
    })
}

pub fn render_result_to_proto(result: &native::RenderResult) -> RenderResponse {
    RenderResponse {
        png_frame: result.frame.as_ref().map(|frame| frame.png_frame.clone()),
    }
}
