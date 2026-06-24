//! Environment traits, requests, the [`EnvServer`], and remote clients.

mod client;
mod server;
mod types;
mod wire;

use async_trait::async_trait;

use crate::spaces;

pub use client::{RemoteEnv, RemoteVectorEnv};
pub use server::{BoundEnvServer, EnvServer, VectorEnvServer};
pub use spaces::request::{CloseResult, ResetRequest, ResetResult, StepRequest, StepResult};
pub use spaces::{CloseRequest, RenderRequest, RenderResult};
pub use types::{
    CloseResult as VectorCloseResult, EpisodeMetadata, ResetRequest as VectorResetRequest,
    ResetResult as VectorResetResult, StepRequest as VectorStepRequest,
    StepResult as VectorStepResult,
};
#[doc(hidden)]
pub use wire::{ScalarEnvAdapter, WireEnvAdapter};

/// A single environment.
///
/// This is the default RLMesh environment shape: one reset, one action, and one
/// transition per endpoint. Use [`VectorEnv`] only when the implementation is
/// deliberately a local batched/vectorized environment.
#[async_trait]
pub trait Env: Send + Sync {
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
    ///
    /// Called once by the server when the *server* stops — not on each client
    /// session close (see [`RemoteEnv::close`]).
    async fn close(
        &mut self,
        req: CloseRequest,
    ) -> std::result::Result<CloseResult, spaces::EnvRuntimeError>;
}

/// A vectorized environment: one implementation steps `num_envs`
/// sub-environments in lockstep.
///
/// This is an explicit local batching optimization. Each `reset`/`step` carries
/// batched inputs (one seed/action per sub-environment) and returns batched
/// outputs (one observation, reward, terminated/truncated flag per
/// sub-environment). Host it with [`VectorEnvServer`].
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
    /// delegates to [`reset`](Self::reset).
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
}
