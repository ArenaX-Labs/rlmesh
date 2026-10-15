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
//! - **Serve a model.** Implement [`Policy`] (or write a predict closure),
//!   build a [`Model`] from it, and host it with [`Model::serve`].
//!
//! - **Run a model against an env.** [`Model::run`] (or [`run`]) drives
//!   episodes on the native loop and returns a [`RunResult`]; [`RunHooks`]
//!   observe it. [`Model::session`] (or [`session`]) returns a [`Session`]
//!   to drive by hand.
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
//! failure your [`Policy`] raised, carrying a [`ModelErrorCode`]). Both
//! carry an `is_recoverable` flag surfaced by [`Error::is_recoverable`], which
//! is false for every other variant.
//!
//! # Example: serve an environment
//!
//! ```no_run
//! use rlmesh::prelude::*;
//! use rlmesh::spaces::{self, BoxOptions, Discrete, DiscreteOptions};
//! use rlmesh::{EnvServer, EnvServerOptions};
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
//! # Example: serve a model
//!
//! ```no_run
//! use rlmesh::prelude::*;
//! use rlmesh::{Model, ModelOptions, ModelServeOptions};
//!
//! /// Counts its predicts per episode in the episode's state.
//! #[derive(Default)]
//! struct Counter;
//!
//! impl Policy for Counter {
//!     const CONTEXT: Corners = Corners::PREDICT;
//!
//!     fn predict_with_context(&mut self, _observation: Value, context: &mut PredictContext<'_>)
//!         -> rlmesh::Result<Value>
//!     {
//!         *context.state.get_or_insert_with("predicts", || 0u64)? += 1;
//!         Ok(Value::Int(0))
//!     }
//! }
//!
//! # fn run() -> rlmesh::Result<()> {
//! let model = Model::new(Counter, ModelOptions::default())?;
//! model.serve("tcp://127.0.0.1:50051", ModelServeOptions::default())?;
//! // A closure is a policy too.
//! let greedy = Model::new(|_observation: Value| Ok(Value::Int(0)), ModelOptions::default())?;
//! # drop(greedy);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

pub mod adapters;
mod address;
#[doc(hidden)]
pub mod binding;
mod bound;
mod build_info;
mod describe;
mod dtype;
mod editions;
pub mod env;
mod error;
pub mod ext;
mod model;
pub mod prelude;
mod run;
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
pub use build_info::{BuildInfo, BuildSource, VERSION, build_info};
pub use describe::{
    Describable, DescribeJsonOptions, DescribeKind, DescribeOptions, describe, describe_json,
};
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
    RemoteEnv, RemoteEnvOptions, RemoteVectorEnv, ServedEnv, TagParam, Transport, Variant,
    VariantOptions, VectorEnv, VectorSeed, trial_index,
};
pub use error::{
    EnvironmentError, Error, ErrorClass, ErrorCode, InvalidArgumentError, ModelError,
    ModelErrorCode, Result,
};
pub use model::{
    Corners, Model, ModelOptions, ModelServeOptions, NO_ADAPTER, Policy, RemoteModel, WithContext,
    predict_seed, with_context,
};
#[doc(no_inline)]
pub use rlmesh_runtime::ENV_RESET_OPTIONS_KEY;
pub use run::{
    EnvTarget, EpisodeResult, ModelTarget, RANDOM_SAMPLE, RandomSample, RunContext, RunHooks,
    RunOptions, RunResult, Session, SessionOptions, SessionRunOptions, StepEvent, TelemetryRow,
    run, run_async, session, session_async,
};
pub use serve_options::{ENV_ENDPOINT_TOKEN_ENV, MODEL_ENDPOINT_TOKEN_ENV, ServeOptions};
pub use specs::EnvContract;
pub use tensor::Tensor;
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
