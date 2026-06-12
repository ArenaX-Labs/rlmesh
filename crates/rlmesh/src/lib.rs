//! Rust SDK for rlmesh model-environment evaluation workflows.
//!
//! rlmesh connects a *model* (your policy) to an *environment* (the task it
//! acts in) over gRPC. This crate is the user-facing facade: it hides the
//! transport, wire, and runtime crates behind a small set of traits and
//! servers. Most Python users should install the `rlmesh` Python package
//! instead; reach for this crate to serve or drive environments and models
//! directly from Rust.
//!
//! # The two roles
//!
//! Everything here serves one of two roles, and a typical deployment runs one
//! of each, in separate processes:
//!
//! - **Serve an environment.** Implement [`Env`] (vectorized: one server may
//!   step `num_envs` sub-environments at once) or [`SingleEnv`] (one
//!   sub-environment; wrap it in [`SingleEnvAdapter`] to get an [`Env`]), then
//!   host it with [`EnvServer`]. Clients connect over gRPC.
//!
//! - **Drive or serve a model.** Implement [`ModelHandler`] (your `predict`
//!   policy plus episode-lifecycle hooks), then either
//!   [`ModelWorker::run_local`] it against a remote environment in-process, or
//!   [`ModelWorker::serve`] it as a standalone model endpoint that an
//!   orchestrator joins.
//!
//! To act as a *client* of an already-running environment server — stepping it
//! by hand rather than handing control to a [`ModelHandler`] — connect with
//! [`RemoteEnv`].
//!
//! # Bind-first servers
//!
//! Both servers are *bind-first*: [`EnvServer::bind`] /
//! [`ModelWorker::bind_async`] reserve the socket and return a bound handle
//! ([`BoundEnvServer`] / [`BoundModelServer`]) whose `local_addr()` reports the
//! resolved address — crucially the OS-assigned port when you bind to port 0 —
//! *before* you call `serve()` to run until shutdown. This removes the
//! bind-drop-rebind races and poll-connect loops callers otherwise reimplement.
//! The one-shot `serve`/`serve_async` methods bind and serve in a single call
//! when you do not need the address up front.
//!
//! # Errors
//!
//! All fallible operations return [`Result`] (alias for `Result<T, `[`Error`]`>`).
//! [`Error`] separates transport/server faults from two domain failures:
//! [`Error::Environment`] (carrying an [`ErrorCode`]) and [`Error::Model`] (a
//! failure your [`ModelHandler`] *raised*, distinct from an rlmesh bug). Both
//! carry an `is_recoverable` flag surfaced by [`Error::is_recoverable`].
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
//!     async fn reset(&mut self, _req: ResetRequest)
//!         -> std::result::Result<ResetResult, EnvRuntimeError>
//!     {
//!         Ok(ResetResult::default())
//!     }
//!     async fn step(&mut self, _req: StepRequest)
//!         -> std::result::Result<StepResult, EnvRuntimeError>
//!     {
//!         Ok(StepResult::default())
//!     }
//!     async fn render(&mut self, _req: RenderRequest)
//!         -> std::result::Result<RenderResult, EnvRuntimeError>
//!     {
//!         Ok(RenderResult::default())
//!     }
//!     async fn close(&mut self, _req: CloseRequest)
//!         -> std::result::Result<CloseResult, EnvRuntimeError>
//!     {
//!         Ok(CloseResult::default())
//!     }
//! }
//!
//! # async fn run(env: MyEnv) -> rlmesh::Result<()> {
//! // Bind first to learn the resolved address, then serve until shutdown.
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
//!         -> rlmesh::Result<rlmesh::spaces::BinaryPayload>
//!     {
//!         // Decode `_obs`, run your policy, return the encoded action bytes.
//!         Ok(rlmesh::spaces::BinaryPayload { data: Vec::new() })
//!     }
//! }
//!
//! # async fn run() -> rlmesh::Result<()> {
//! // Connect in-process to a running env server and run 100 episodes.
//! use rlmesh::RunLocalOptions;
//! ModelWorker::new(MyModel)
//!     .run_local_async(RunLocalOptions::parse("tcp://127.0.0.1:50051")?.for_episodes(100))
//!     .await
//! # }
//! ```

#![warn(missing_docs)]

mod address;
mod bound;
pub mod env;
mod error;
pub mod lifecycle;
pub mod model;
pub mod prelude;
mod single;
pub mod spaces;

pub use address::{BindAddress, ConnectAddress};
pub use env::{
    BoundEnvServer, CloseRequest, CloseResult, Env, EnvServer, EpisodeMetadata, RemoteEnv,
    RenderRequest, RenderResult, ResetRequest, ResetResult, StepRequest, StepResult,
};
pub use error::{EnvironmentError, Error, ErrorCode, ModelError, Result};
pub use lifecycle::ServeOptions;
pub use model::{
    BoundModelServer, EnvClientRuntimeEnv, ModelEpisodeEnd, ModelHandler, ModelHandlerRuntimeModel,
    ModelObservation, ModelRouteContext, ModelRouteSlot, ModelWorker, RunLocalOptions,
    ServeModelOptions,
};
pub use single::{SingleEnv, SingleEnvAdapter};
pub use spaces::{EnvContract, EnvRuntimeError, RenderFrame, SpaceSpec, SpaceValue};

#[cfg(test)]
mod tests;
