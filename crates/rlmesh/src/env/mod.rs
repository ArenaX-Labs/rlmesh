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
