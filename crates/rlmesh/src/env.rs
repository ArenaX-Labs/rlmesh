//! Environment traits, requests, the [`EnvServer`], and the [`RemoteEnv`] client.

mod client;
mod server;
mod types;
mod wire;

use async_trait::async_trait;

use crate::spaces;

pub use client::RemoteEnv;
pub use server::{BoundEnvServer, EnvServer};
pub use types::{
    CloseRequest, CloseResult, EpisodeMetadata, RenderRequest, RenderResult, ResetRequest,
    ResetResult, StepRequest, StepResult,
};
#[doc(hidden)]
pub use wire::WireEnvAdapter;

/// A vectorized environment: one implementation steps `num_envs`
/// sub-environments in lockstep.
///
/// Each `reset`/`step` carries batched inputs (one seed/action per
/// sub-environment) and returns batched outputs (one observation, reward,
/// terminated/truncated flag per sub-environment). For a non-vectorized
/// environment, implement [`SingleEnv`](crate::SingleEnv) and wrap it in
/// [`SingleEnvAdapter`](crate::SingleEnvAdapter) instead. Host any `Env` with
/// [`EnvServer`].
#[async_trait]
pub trait Env: Send + Sync {
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
        req: ResetRequest,
    ) -> std::result::Result<ResetResult, spaces::EnvRuntimeError>;

    /// Reset only the sub-environments named in `req.env_indices` (a partial /
    /// per-lane reset), leaving the others running. An empty `env_indices`
    /// delegates to [`reset`](Self::reset).
    ///
    /// The default **rejects** a non-empty request. Per-lane reset is not
    /// something a vectorized env can do unless it explicitly supports resetting
    /// individual sub-environments — stock gymnasium vector envs cannot, so they
    /// fall through to this default and fail loud rather than silently resetting
    /// the whole vector. A future in-house vector engine that can reset
    /// individual lanes overrides this to enable the reproducible / staggered
    /// `DISABLED`-autoreset path. The runtime only calls this for a strict
    /// subset of done lanes under `DISABLED` autoreset; whole-vector resets and
    /// `NEXT_STEP` envs never reach it.
    async fn reset_subset(
        &mut self,
        req: ResetRequest,
    ) -> std::result::Result<ResetResult, spaces::EnvRuntimeError> {
        if req.env_indices.is_empty() {
            self.reset(req).await
        } else {
            Err(spaces::EnvRuntimeError::Runtime(format!(
                "partial reset of sub-envs {:?} is not supported by this environment. \
                 Per-lane reset is only available for an env that overrides \
                 `Env::reset_subset`. Use NEXT_STEP autoreset (the env resets done lanes \
                 itself), run with num_envs == 1, or ensure all lanes terminate on the same \
                 step so the whole vector resets together.",
                req.env_indices
            )))
        }
    }

    /// Apply one action per sub-environment and return the batched transition.
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
