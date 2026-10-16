//! Rust SDK for RLMesh model-environment evaluation workflows.
//!
//! RLMesh connects a model to an environment over gRPC. This crate is the Rust
//! facade over the transport, wire, and runtime crates, with the Python
//! package's nouns and defaults. Python users should start with the `rlmesh`
//! Python package; use this crate when serving or driving environments and
//! models from Rust.
//!
//! # The two roles
//!
//! - **Serve an environment.** Implement [`Env`] (or [`VectorEnv`]) in
//!   Gymnasium shape and host it with [`EnvServer`], or implement
//!   [`EnvFactory`] and call [`EnvFactoryExt::serve`].
//!
//! - **Drive or serve a model.** Implement [`ModelHandler`], then run it against
//!   a remote environment with [`ModelWorker::run_local`] or serve it as an
//!   endpoint with [`ModelWorker::serve`].
//!
//! Use [`RemoteEnv`] when you want to step an environment server directly.
//!
//! # Options structs
//!
//! Python's keyword arguments are options structs whose `Default` holds
//! Python's defaults ([`EnvServerOptions`], [`ServeOptions`],
//! [`EnvFactoryServeOptions`]). Build them with `..Default::default()`: by
//! project policy appending an options field is a minor change, and a full
//! struct literal without `..` is unsupported.
//!
//! # Errors
//!
//! Fallible operations return [`Result`] (alias for `Result<T, `[`Error`]`>`).
//! Each [`Error`] variant is one of the Python package's exception classes:
//! [`Error::class`] reports the exact class as an [`ErrorClass`], whose
//! [`ErrorClass::is_subclass_of`] follows Python's inheritance, and `Display`
//! is Python's `str(exc)`. The two domain failures are
//! [`Error::Environment`] (carrying an [`ErrorCode`]) and [`Error::Model`] (a
//! failure your [`ModelHandler`] raised, carrying a [`ModelErrorCode`]). Both
//! carry an `is_recoverable` flag surfaced by [`Error::is_recoverable`], which
//! is false for every other variant.
//!
//! # Example: serve an environment
//!
//! ```no_run
//! use rlmesh::prelude::*;
//! use rlmesh::spaces::{self, BoxOptions, Discrete, DiscreteOptions};
//!
//! struct Cartpole {
//!     observation_space: Space,
//!     action_space: Space,
//! }
//!
//! impl Env for Cartpole {
//!     fn observation_space(&self) -> Space { self.observation_space.clone() }
//!     fn action_space(&self) -> Space { self.action_space.clone() }
//!     fn reset(&mut self, _seed: Option<i64>, _options: Option<&InfoDict>)
//!         -> rlmesh::Result<(Value, InfoDict)>
//!     {
//!         Ok((self.observation_space.sample()?, InfoDict::new()))
//!     }
//!     fn step(&mut self, _action: Option<Value>)
//!         -> rlmesh::Result<(Value, f64, bool, bool, InfoDict)>
//!     {
//!         Ok((self.observation_space.sample()?, 1.0, false, false, InfoDict::new()))
//!     }
//! }
//!
//! # fn run() -> rlmesh::Result<()> {
//! let env = Cartpole {
//!     observation_space: spaces::Box::new(-1.0, 1.0, [4], BoxOptions::default())?.into(),
//!     action_space: Discrete::new(2, DiscreteOptions::default())?.into(),
//! };
//! let server = EnvServer::new(
//!     env,
//!     EnvServerOptions { address: Some("tcp://127.0.0.1:0".into()), ..Default::default() },
//! )?;
//! println!("listening on {}", server.address());
//! server.serve()
//! # }
//! ```
//!
//! # Example: drive a model against that environment
//!
//! ```no_run
//! use rlmesh::prelude::*;
//! use rlmesh::binding::spaces::SpaceValue;
//!
//! struct MyModel;
//!
//! #[rlmesh::async_trait]
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
//! println!("ran {} steps", report.total_steps);
//! Ok(())
//! # }
//! ```

#![warn(missing_docs)]

pub mod adapters;
mod address;
#[doc(hidden)]
pub mod binding;
mod bound;
mod build_info;
mod dtype;
mod editions;
pub mod env;
mod error;
pub mod ext;
pub mod model;
pub mod prelude;
pub mod serve_options;
pub mod spaces;
pub mod specs;
mod tensor;
pub mod types;
mod value;
mod warnings;

#[cfg(unix)]
pub use address::remove_stale_socket;
pub use address::{BindAddress, ConnectAddress};
pub use async_trait::async_trait;
/// The wire space spec the model traits take until they take
/// [`spaces::Space`]; the shared spec type is [`spaces::SpaceSpec`].
pub use binding::spaces::SpaceSpec;
pub use binding::spaces::{EnvRuntimeError, SpaceValue};
pub use build_info::{BuildInfo, BuildSource, VERSION, build_info};
pub use dtype::{DType, Device};
pub use editions::current_workflow_edition;

/// The key a serialized describe envelope travels under (`rlmesh.describe.v1`):
/// Python's `rlmesh.DESCRIBE_METADATA_KEY`.
pub const DESCRIBE_METADATA_KEY: &str = rlmesh_adapters::v1::DESCRIBE_METADATA_KEY;

/// The describe envelope's schema version: Python's
/// `rlmesh.DESCRIBE_SCHEMA_VERSION`.
pub const DESCRIBE_SCHEMA_VERSION: i64 = rlmesh_adapters::v1::DESCRIBE_SCHEMA_VERSION as i64;
pub use env::{
    Env, EnvFactory, EnvFactoryExt, EnvFactoryServeOptions, EnvServer, EnvServerOptions, MadeEnv,
    RemoteEnv, RemoteVectorEnv, TagParam, Transport, Variant, VariantOptions, VectorEnv,
    VectorSeed, trial_index,
};
pub use error::{
    EnvironmentError, Error, ErrorClass, ErrorCode, InvalidArgumentError, ModelError,
    ModelErrorCode, Result,
};
pub use model::{
    AdaptedModelHandler, BoundModelServer, EnvClientRuntimeEnv, EpisodeInfo, HeldState,
    HistoryFrame, HistoryNeeds, ModelHandler, ModelHandlerRuntimeModel, ModelObservation,
    ModelRouteContext, ModelRouteSetup, ModelWorker, PredictFn, PredictFrames, RemoteModel,
    ResolveOptions, RouteConfig, RouteNeeds, RouteResolver, RunLocalOptions, ServeModelOptions,
    predict_seed,
};
#[doc(no_inline)]
pub use rlmesh_runtime::telemetry;
#[doc(no_inline)]
pub use rlmesh_runtime::{
    ActionReceivedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, HookError, NoopRuntimeHooks,
    ObservationEmittedEvent, RuntimeHooks, StepCompletedEvent,
};
#[doc(no_inline)]
pub use rlmesh_runtime::{ENV_RESET_OPTIONS_KEY, PeerCeiling, RuntimeReport};
pub use serve_options::{ENV_ENDPOINT_TOKEN_ENV, MODEL_ENDPOINT_TOKEN_ENV, ServeOptions};
pub use specs::EnvContract;
pub use tensor::Tensor;
pub use tokio_util::sync::CancellationToken;
pub use value::{InfoDict, Kwargs, Metadata, MetadataIter, Value, ValueKind};
pub use warnings::{Warning, WarningCategory, WarningHandler, set_warning_handler};

/// Mint one fresh routing/episode id (UUIDv7 — time-ordered, sortable by creation,
/// never repeats). The single id-format home for this crate's id authorities (the
/// direct env client and the in-process / remote model paths). The runtime driver
/// mints its own ids in `rlmesh-runtime` (it does not depend on this crate).
pub(crate) fn mint_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

#[cfg(test)]
mod tests;
