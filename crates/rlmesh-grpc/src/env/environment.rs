//! Environment trait for environments served over RLMesh gRPC.

use async_trait::async_trait;
use rlmesh_proto::Edition;
pub use rlmesh_proto::env::v1::{
    CloseEnvsResponse, RenderRequest, RenderResponse, ResetRequest, ResetResponse, StepRequest,
    StepResponse,
};
pub use rlmesh_proto::{EndpointPhases, lane_skew_ns};
use rlmesh_spaces::{EnvContract, spaces::SpaceSpec};

use crate::error::EnvError;

/// Transport-facing environment contract.
///
/// Every op takes `&self`: the implementation owns its own synchronization (a
/// serial env behind a lock, or one actor per lane), so the server never
/// locks and can keep several lane-scoped requests in flight at once. Each op
/// returns its own phase split beside the reply.
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

    /// Whether lanes can be reset and stepped individually and concurrently.
    /// Advertised at handshake as the `subset_step` capability: a runtime that
    /// sees it drives every lane as its own episode loop with one request per
    /// lane in flight. A lockstep env (a gym vector env) leaves this `false`
    /// and rejects a step that names lanes.
    fn supports_lanes(&self) -> bool {
        false
    }

    /// The workflow edition a Join session runs at, recorded once the session
    /// settles it: the runtime's `ConfigureEnv` pin (its first Join message), or
    /// this build's current edition when the session opens with `Reset` instead
    /// (a runtime built before the pin existed sends none). Branch every
    /// edition-governed env-side default on it. Default: nothing to branch.
    fn pin_workflow_edition(&self, _edition: Edition) {}

    /// Reset the whole vector (empty `env_indices`) or just the named lanes.
    /// A reset naming lanes replies only those lanes, positionally.
    async fn reset(&self, req: ResetRequest) -> Result<(ResetResponse, EndpointPhases), EnvError>;

    /// Step the whole vector (empty `env_indices`) or just the named lanes; a
    /// step naming lanes replies only those lanes, positionally. An env that
    /// does not [`supports_lanes`](Self::supports_lanes) rejects the latter.
    async fn step(&self, req: StepRequest) -> Result<(StepResponse, EndpointPhases), EnvError>;

    /// Render one lane (`env_indices` names it; empty is lane 0).
    async fn render(
        &self,
        req: RenderRequest,
    ) -> Result<(RenderResponse, EndpointPhases), EnvError>;

    /// Close the environment or vector.
    async fn close(&self) -> Result<CloseEnvsResponse, EnvError>;
}
