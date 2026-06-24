//! Environment trait for environments served over RLMesh gRPC.

use async_trait::async_trait;
pub use rlmesh_proto::env::v1::{
    CloseEnvsResponse, RenderRequest, RenderResponse, ResetRequest, ResetResponse, StepRequest,
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

    /// Reset only the lanes named in `req.env_indices` (a partial / per-lane
    /// reset — e.g. controlled-seed eval). An empty `env_indices` is a
    /// whole-vector reset and delegates to [`reset`](Self::reset).
    ///
    /// The default **rejects** a non-empty request. Per-lane reset requires an
    /// env that can reset individual sub-environments; stock gymnasium vector
    /// envs cannot, so they fall through to this default and fail loud rather
    /// than silently resetting the whole vector. An env that supports it (a
    /// future in-house vector engine) overrides this. The server routes a
    /// non-empty `env_indices` here, which the runtime only sends for a strict
    /// subset of done lanes under `DISABLED` autoreset.
    async fn reset_subset(&mut self, req: ResetRequest) -> Result<ResetResponse, EnvError> {
        if req.env_indices.is_empty() {
            self.reset(req).await
        } else {
            Err(EnvError::new(
                EnvErrorCode::Internal,
                format!(
                    "partial reset of sub-envs {:?} is not supported by this environment. \
                     Per-lane reset is only available for an env that implements reset_subset. \
                     Use NEXT_STEP autoreset (the env resets done lanes itself), run with \
                     num_envs == 1, or ensure all lanes terminate on the same step so the whole \
                     vector resets together.",
                    req.env_indices
                ),
            ))
        }
    }

    /// Step the environment or vector.
    async fn step(&mut self, req: StepRequest) -> Result<StepResponse, EnvError>;

    /// Render the environment or vector.
    async fn render(&mut self, req: RenderRequest) -> Result<RenderResponse, EnvError>;

    /// Close the environment or vector.
    async fn close(&mut self) -> Result<CloseEnvsResponse, EnvError>;
}
