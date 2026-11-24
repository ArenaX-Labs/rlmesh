//! The wire-level env traits: the async request/result shape the transport
//! adapters serve. The Python binding implements it; authors implement the

use async_trait::async_trait;

use super::types::{
    CloseResult as VectorCloseResult, ResetRequest as VectorResetRequest,
    ResetResult as VectorResetResult, StepRequest as VectorStepRequest,
    StepResult as VectorStepResult,
};
use crate::binding::spaces::{
    self, CloseRequest, RenderRequest, RenderResult,
    request::{CloseResult, ResetRequest, ResetResult, StepRequest, StepResult},
};
use rlmesh_proto::EndpointPhases;

/// A single environment: one reset, one action and one transition per endpoint. Use
/// [`VectorEnv`] only for a deliberately batched local environment.
#[async_trait]
pub trait Env: Send {
    /// The space observations belong to.
    fn observation_space(&self) -> &spaces::SpaceSpec;
    /// The space actions belong to.
    fn action_space(&self) -> &spaces::SpaceSpec;
    /// The environment contract (spaces, id, render mode, metadata).
    fn env_contract(&self) -> &spaces::EnvContract;

    /// Reset the environment and return its initial observation.
    async fn reset(
        &mut self,
        req: ResetRequest,
    ) -> std::result::Result<ResetResult, spaces::EnvRuntimeError>;

    /// Apply one action and return the transition.
    async fn step(
        &mut self,
        req: StepRequest,
    ) -> std::result::Result<StepResult, spaces::EnvRuntimeError>;

    /// Produce a render frame for the current state.
    async fn render(
        &mut self,
        req: RenderRequest,
    ) -> std::result::Result<RenderResult, spaces::EnvRuntimeError>;

    /// Release resources and return metadata for any final episodes.
    async fn close(
        &mut self,
        req: CloseRequest,
    ) -> std::result::Result<CloseResult, spaces::EnvRuntimeError>;

    /// The phase split of the op just completed, cleared by the read.
    ///
    /// Override it to report where an op's time went (decoding the request, your own work,
    /// encoding the result); the server stamps the split on the response beside
    /// `endpoint_total_ns`. The default measures nothing, so the whole call reports as `user_ns`.
    fn take_last_phases(&mut self) -> EndpointPhases {
        EndpointPhases::default()
    }
}

/// A vectorized environment: one implementation steps `num_envs`
/// sub-environments in lockstep.
///
/// Each `reset`/`step` carries one seed or action per sub-environment and returns one
/// observation, reward and terminated/truncated flag per sub-environment. Host it with the
/// vector wire adapter.
#[async_trait]
pub trait VectorEnv: Send + Sync {
    /// The space each observation in a batch belongs to.
    fn observation_space(&self) -> &spaces::SpaceSpec;
    /// The space each action in a batch belongs to.
    fn action_space(&self) -> &spaces::SpaceSpec;
    /// The number of sub-environments stepped together (the batch size).
    fn num_envs(&self) -> usize;
    /// The environment contract (spaces, id, render mode, metadata).
    fn env_contract(&self) -> &spaces::EnvContract;

    /// Reset all sub-environments and return their initial observations.
    async fn reset(
        &mut self,
        req: VectorResetRequest,
    ) -> std::result::Result<VectorResetResult, spaces::EnvRuntimeError>;

    /// Reset only the sub-environments named in `req.env_indices` (a partial /
    /// per-lane reset), leaving the others running. An empty `env_indices`
    async fn reset_subset(
        &mut self,
        req: VectorResetRequest,
    ) -> std::result::Result<VectorResetResult, spaces::EnvRuntimeError> {
        if req.env_indices.is_empty() {
            self.reset(req).await
        } else {
            Err(spaces::EnvRuntimeError::Runtime(format!(
                "partial reset of sub-envs {:?} is not supported by this environment. \
                 Per-lane reset is only available for an env that overrides \
                 `VectorEnv::reset_subset`. Use NEXT_STEP autoreset (the env resets done lanes \
                 itself), run with num_envs == 1, or ensure all lanes terminate on the same \
                 step so the whole vector resets together.",
                req.env_indices
            )))
        }
    }

    /// Apply one action per sub-environment and return the batched transition.
    async fn step(
        &mut self,
        req: VectorStepRequest,
    ) -> std::result::Result<VectorStepResult, spaces::EnvRuntimeError>;

    /// Produce a render frame for the current state.
    async fn render(
        &mut self,
        req: RenderRequest,
    ) -> std::result::Result<RenderResult, spaces::EnvRuntimeError>;

    /// Release resources and return metadata for any final episodes.
    async fn close(
        &mut self,
        req: CloseRequest,
    ) -> std::result::Result<VectorCloseResult, spaces::EnvRuntimeError>;

    /// The phase split of the op just completed, cleared by the read.
    ///
    /// Override it to report where an op's time went (decoding the request, your own work,
    /// encoding the result); the server stamps the split on the response beside
    /// `endpoint_total_ns`. The default measures nothing, so the whole call reports as `user_ns`.
    fn take_last_phases(&mut self) -> EndpointPhases {
        EndpointPhases::default()
    }
}
