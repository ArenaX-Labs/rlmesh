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

#[async_trait]
pub trait Env: Send + Sync {
    fn observation_space(&self) -> &spaces::SpaceSpec;
    fn action_space(&self) -> &spaces::SpaceSpec;
    fn num_envs(&self) -> usize;
    fn env_contract(&self) -> &spaces::EnvContract;

    async fn reset(
        &mut self,
        req: ResetRequest,
    ) -> std::result::Result<ResetResult, spaces::EnvRuntimeError>;

    async fn step(
        &mut self,
        req: StepRequest,
    ) -> std::result::Result<StepResult, spaces::EnvRuntimeError>;

    async fn render(
        &mut self,
        req: RenderRequest,
    ) -> std::result::Result<RenderResult, spaces::EnvRuntimeError>;

    async fn close(
        &mut self,
        req: CloseRequest,
    ) -> std::result::Result<CloseResult, spaces::EnvRuntimeError>;
}
