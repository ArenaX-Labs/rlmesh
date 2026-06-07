//! Environment trait for environments served over RLMesh gRPC.

use async_trait::async_trait;
pub use rlmesh_proto::env::v1::{
    CloseResponse, RenderRequest, RenderResponse, ResetRequest, ResetResponse, StepRequest,
    StepResponse,
};
use rlmesh_spaces::v1::{EnvContract, spaces::SpaceSpec};

use crate::error::EnvError;

/// An environment that can be served over RLMesh gRPC.
///
/// This is the transport-facing contract. Public native Gym-style
/// environments should normally be adapted by the `rlmesh` facade instead of
/// implementing this trait directly.
#[async_trait]
pub trait Environment: Send + Sync {
    /// Get the observation space specification.
    fn observation_space(&self) -> &SpaceSpec;

    /// Get the action space specification.
    fn action_space(&self) -> &SpaceSpec;

    /// Get the number of parallel environments (1 for single env).
    fn num_envs(&self) -> usize;

    /// Get the full gymnasium spec.
    fn env_contract(&self) -> &EnvContract;

    /// Reset the environment(s).
    async fn reset(&mut self, req: ResetRequest) -> Result<ResetResponse, EnvError>;

    /// Take a step in the environment(s).
    async fn step(&mut self, req: StepRequest) -> Result<StepResponse, EnvError>;

    /// Render the environment(s).
    async fn render(&mut self, req: RenderRequest) -> Result<RenderResponse, EnvError>;

    /// Close the environment(s).
    async fn close(&mut self) -> Result<CloseResponse, EnvError>;
}
