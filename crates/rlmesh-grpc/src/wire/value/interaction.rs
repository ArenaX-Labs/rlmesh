use rlmesh_proto::env::v1::{
    RenderFormat, RenderRequest, RenderResponse, ResetRequest, ResetResponse, StepRequest,
    StepResponse,
};
use rlmesh_spaces as native;

use crate::error::ProtocolError;
use crate::wire::spaces::{meta_map_from_proto, meta_map_to_proto};

use super::payload::{decode_value, encode_value};

pub fn reset_request_to_proto(
    request: &native::ResetRequest,
) -> Result<ResetRequest, ProtocolError> {
    Ok(ResetRequest {
        seeds: request.seed.into_iter().collect(),
        options: request.options.as_ref().map(meta_map_to_proto),
        // Native timeout_ms is i64 (>=0 by construction); proto field is uint64.
        timeout_ms: request.timeout_ms.max(0) as u64,
        // Whole-vector reset; partial reset threads through reset_subset (A3).
        env_indices: Vec::new(),
        // Runtime-minted ids are pushed by the orchestrating runtime, not this
        // bare single-env conversion; left empty here.
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
        // The env no longer returns ids (R1); the runtime is authoritative and
        // pushes ids down. This bare conversion has no runtime, so no id.
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
        // Native timeout_ms is i64 (>=0 by construction); proto field is uint64.
        timeout_ms: request.timeout_ms.max(0) as u64,
        // Full-width step; subset-stepping is reserved-but-deferred.
        env_indices: Vec::new(),
        // See reset_request_to_proto: ids are runtime-pushed, not minted here.
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
        mask: request.env_index.map(render_mask).unwrap_or_default(),
        // Native timeout_ms is i64 (>=0 by construction); proto field is uint64.
        timeout_ms: request.timeout_ms.max(0) as u64,
    }
}

pub fn render_result_from_proto(
    response: RenderResponse,
) -> Result<native::RenderResult, ProtocolError> {
    // Per the RenderFormat contract, a frame whose format this build does not
    // recognize must be SKIPPED (treated as no frame), never surfaced as if it
    // were the default PNG -- otherwise a caller (e.g. the Python client) would
    // misinterpret a future codec's bytes as PNG. Only UNSPECIFIED (the historical
    // PNG default) and PNG are understood today; anything else drops the frame.
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
        // Full-width response; partial-width is reserved-but-deferred.
        env_indices: vec![],
    })
}

pub fn render_result_to_proto(result: &native::RenderResult) -> RenderResponse {
    // Frames are PNG today; stamp the format explicitly so the discriminator is
    // load-bearing from the freeze. A frame-less response leaves it UNSPECIFIED.
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
