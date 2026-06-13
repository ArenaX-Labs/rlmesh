//! Environment trait for environments served over RLMesh gRPC.

use async_trait::async_trait;
pub use rlmesh_proto::env::v1::{
    CloseResponse, RenderRequest, RenderResponse, ResetRequest, ResetResponse, StepRequest,
    StepResponse,
};
use rlmesh_spaces::{EnvContract, spaces::SpaceSpec};

use crate::error::{EnvError, EnvErrorCode};

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

    /// Reset only the lanes named in `req.env_indices` (explicit/manual partial
    /// reset — e.g. controlled-seed eval). An empty `env_indices` is a
    /// whole-vector reset and delegates to [`reset`](Self::reset).
    ///
    /// The default rejects a non-empty request: environments that support
    /// partial reset (e.g. gym vector envs via `options["reset_mask"]`) override
    /// this. The server routes a non-empty `env_indices` here so unsupported
    /// envs fail loud instead of silently resetting the whole vector.
    async fn reset_subset(&mut self, req: ResetRequest) -> Result<ResetResponse, EnvError> {
        if req.env_indices.is_empty() {
            self.reset(req).await
        } else {
            Err(EnvError::new(
                EnvErrorCode::Internal,
                "partial reset (env_indices) is not supported by this environment",
            ))
        }
    }

    /// Step the environment or vector.
    async fn step(&mut self, req: StepRequest) -> Result<StepResponse, EnvError>;

    /// Render the environment or vector.
    async fn render(&mut self, req: RenderRequest) -> Result<RenderResponse, EnvError>;

    /// Close the environment or vector.
    async fn close(&mut self) -> Result<CloseResponse, EnvError>;
}
