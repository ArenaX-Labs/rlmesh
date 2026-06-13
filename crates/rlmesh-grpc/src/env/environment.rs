//! Environment trait for environments served over RLMesh gRPC.

use async_trait::async_trait;
pub use rlmesh_proto::env::v1::{
    CloseResponse, RenderRequest, RenderResponse, ResetRequest, ResetResponse, StepRequest,
    StepResponse,
};
use rlmesh_spaces::{EnvContract, spaces::SpaceSpec};

use crate::error::EnvError;

/// Transport-facing environment contract.
///
/// Most users adapt environments through the higher-level `rlmesh` facade
/// instead of implementing this trait directly.
#[async_trait]
pub trait Environment: Send + Sync {
    /// Observation space.
    fn observation_space(&self) -> &SpaceSpec;

    /// Action space.
    fn action_space(&self) -> &SpaceSpec;

    /// Number of parallel environments, or `1` for a single environment.
    fn num_envs(&self) -> usize;

    /// Full environment contract.
    fn env_contract(&self) -> &EnvContract;

    /// Reset the environment or vector.
    async fn reset(&mut self, req: ResetRequest) -> Result<ResetResponse, EnvError>;

    /// Step the environment or vector.
    async fn step(&mut self, req: StepRequest) -> Result<StepResponse, EnvError>;

    /// Render the environment or vector.
    async fn render(&mut self, req: RenderRequest) -> Result<RenderResponse, EnvError>;

    /// Close the environment or vector.
    async fn close(&mut self) -> Result<CloseResponse, EnvError>;
}
