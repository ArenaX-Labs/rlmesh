use rlmesh_proto::env::v1::{
    RenderFormat, RenderRequest, RenderResponse, ResetRequest, ResetResponse, StepRequest,
    StepResponse,
};
use rlmesh_spaces as native;

use crate::data::ProtocolError;
use crate::data::spaces::{meta_map_from_proto, meta_map_to_proto};

use super::payload::{decode_value, encode_value};

pub fn reset_request_to_proto(
    request: &native::ResetRequest,
) -> Result<ResetRequest, ProtocolError> {
    Ok(ResetRequest {
        seeds: request.seed.into_iter().collect(),
        options: request.options.as_ref().map(meta_map_to_proto),
        timeout_ms: request.timeout_ms.max(0) as u64,
        env_indices: Vec::new(),
        episode_ids: Vec::new(),
    })
}

pub fn reset_result_from_proto(
    response: ResetResponse,
    observation_space: &native::SpaceSpec,
) -> Result<native::ResetResult, ProtocolError> {
    Ok(native::ResetResult {
        observation: decode_value(response.observation.as_ref(), observation_space)?,
        info: response.infos.map(meta_map_from_proto),
        episode_id: None,
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
        timeout_ms: request.timeout_ms.max(0) as u64,
        env_indices: Vec::new(),
        episode_ids: Vec::new(),
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
        info: response.infos.map(meta_map_from_proto),
    })
}

pub fn render_request_to_proto(request: &native::RenderRequest) -> RenderRequest {
    RenderRequest {
        env_indices: request
            .env_index
            .map(|i| vec![i as u32])
            .unwrap_or_default(),
        timeout_ms: request.timeout_ms.max(0) as u64,
    }
}

pub fn render_result_from_proto(
    response: RenderResponse,
) -> Result<native::RenderResult, ProtocolError> {
    // A frame whose format this build does not recognize is skipped, never surfaced as the default PNG.
    let understood = matches!(
        RenderFormat::try_from(response.format),
        Ok(RenderFormat::Unspecified | RenderFormat::Png)
    );
    Ok(native::RenderResult {
        frame: response
            .frame
            .filter(|_| understood)
            .map(|frame| native::RenderFrame { frame }),
    })
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
        infos: result.info.as_ref().map(meta_map_to_proto),
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
        infos: result.info.as_ref().map(meta_map_to_proto),
        completed_episodes: vec![],
        env_indices: vec![],
    })
}

pub fn render_result_to_proto(result: &native::RenderResult) -> RenderResponse {
    let format = if result.frame.is_some() {
        RenderFormat::Png
    } else {
        RenderFormat::Unspecified
    };
    RenderResponse {
        frame: result.frame.as_ref().map(|frame| frame.frame.clone()),
        format: i32::from(format),
    }
}
