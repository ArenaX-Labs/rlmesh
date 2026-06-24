//! Rust SDK for RLMesh model-environment evaluation workflows.
//!
//! RLMesh connects a model to an environment over gRPC. This crate is the Rust
//! facade over the transport, wire, and runtime crates. Python users should
//! start with the `rlmesh` Python package; use this crate when serving or
//! driving environments and models from Rust.
//!
//! # The two roles
//!
//! Most deployments have one environment server and one model worker.
//!
//! - **Serve an environment.** Implement [`Env`] for vectorized environments,
//!   or [`SingleEnv`] for a single environment, then host it with [`EnvServer`].
//!
//! - **Drive or serve a model.** Implement [`ModelHandler`], then run it against
//!   a remote environment with [`ModelWorker::run_local`] or serve it as an
//!   endpoint with [`ModelWorker::serve`].
//!
//! Use [`RemoteEnv`] when you want to step an environment server directly.
//!
//! # Bind-first servers
//!
//! [`EnvServer::bind`] and [`ModelWorker::bind_async`] reserve the socket before
//! serving and return the resolved address, including OS-assigned port 0. This
//! avoids bind-drop-rebind races and poll-connect loops. Use the one-shot
//! `serve`/`serve_async` methods when you do not need that address first.
//!
//! # Errors
//!
//! Fallible operations return [`Result`] (alias for `Result<T, `[`Error`]`>`).
//! [`Error`] separates transport/server faults from two domain failures:
//! [`Error::Environment`] (carrying an [`ErrorCode`]) and [`Error::Model`] (a
//! failure your [`ModelHandler`] raised). Both carry an `is_recoverable` flag
//! surfaced by [`Error::is_recoverable`].
//!
//! # Example: serve an environment
//!
//! ```no_run
//! use rlmesh::prelude::*;
//!
//! struct MyEnv {
//!     observation_space: SpaceSpec,
//!     action_space: SpaceSpec,
//!     contract: EnvContract,
//! }
//!
//! #[async_trait::async_trait]
//! impl Env for MyEnv {
//!     fn observation_space(&self) -> &SpaceSpec { &self.observation_space }
//!     fn action_space(&self) -> &SpaceSpec { &self.action_space }
//!     fn num_envs(&self) -> usize { 1 }
//!     fn env_contract(&self) -> &EnvContract { &self.contract }
//!
//!     // Env methods use the two-arg std::result::Result form.
//!     async fn reset(&mut self, _req: ResetRequest)
//!         -> Result<ResetResult, EnvRuntimeError>
//!     {
//!         Ok(ResetResult::default())
//!     }
//!     async fn step(&mut self, _req: StepRequest)
//!         -> Result<StepResult, EnvRuntimeError>
//!     {
//!         Ok(StepResult::default())
//!     }
//!     async fn render(&mut self, _req: RenderRequest)
//!         -> Result<RenderResult, EnvRuntimeError>
//!     {
//!         Ok(RenderResult::default())
//!     }
//!     async fn close(&mut self, _req: CloseRequest)
//!         -> Result<CloseResult, EnvRuntimeError>
//!     {
//!         Ok(CloseResult::default())
//!     }
//! }
//!
//! # async fn run(env: MyEnv) -> rlmesh::Result<()> {
//! // Bind first when the caller needs the resolved address.
//! let bound = EnvServer::new(env).bind(BindAddress::parse("tcp://127.0.0.1:0")?).await?;
//! println!("listening on {}", bound.local_addr());
//! bound.serve().await
//! # }
//! ```
//!
//! # Example: drive a model against that environment
//!
//! ```no_run
//! use rlmesh::prelude::*;
//!
//! struct MyModel;
//!
//! #[async_trait::async_trait]
//! impl ModelHandler for MyModel {
//!     async fn predict(&mut self, _obs: ModelObservation)
//!         -> rlmesh::Result<Vec<SpaceValue>>
//!     {
//!         // Read `_obs.decoded_lanes()`, run your policy, return one action per lane.
//!         Ok(vec![SpaceValue::Discrete(0)])
//!     }
//! }
//!
//! # async fn run() -> rlmesh::Result<()> {
//! // Drive a running env server for 100 episodes.
//! let report = ModelWorker::new(MyModel)
//!     .run_local_async(RunLocalOptions::parse("tcp://127.0.0.1:50051")?.for_episodes(100))
//!     .await?;
//! // `report.telemetry_summary` carries the session's final telemetry summary.
//! println!("ran {} steps", report.total_steps);
//! Ok(())
//! # }
//! ```

#![warn(missing_docs)]

mod address;
mod bound;
pub mod env;
mod error;
pub mod model;
pub mod prelude;
pub mod serve_options;
mod single;
pub mod spaces;

pub use address::{BindAddress, ConnectAddress};
pub use env::{
    BoundEnvServer, CloseRequest, CloseResult, Env, EnvServer, EpisodeMetadata, RemoteEnv,
    RenderRequest, RenderResult, ResetRequest, ResetResult, StepRequest, StepResult,
};
pub use error::{EnvironmentError, Error, ErrorCode, ModelError, Result};
pub use model::{
    BoundModelServer, EnvClientRuntimeEnv, ModelEpisodeEnd, ModelHandler, ModelHandlerRuntimeModel,
    ModelLaneReset, ModelObservation, ModelRouteContext, ModelRouteSetup, ModelRouteSlot,
    ModelWorker, RemoteModel, RunLocalOptions, ServeModelOptions, telemetry_summary_to_proto,
};
#[doc(no_inline)]
pub use rlmesh_runtime::RuntimeReport;
pub use serve_options::ServeOptions;
pub use single::{SingleEnv, SingleEnvAdapter};
pub use spaces::{EnvContract, EnvRuntimeError, RenderFrame, SpaceSpec, SpaceValue};

#[cfg(test)]
mod tests;
