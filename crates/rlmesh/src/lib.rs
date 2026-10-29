//! Rust SDK for RLMesh model-environment evaluation workflows.
//!
//! RLMesh connects a model to an environment over a session: participant
//! addresses such as `:0/env` (the `lockstep` feature), or `tcp://` and
//! `unix://` addresses on the v1 gRPC bridge. This crate is the Rust facade
//! over the transport, wire, and runtime crates. RLMesh uses
//! the same nouns and constructs in every language, following each
//! language's conventions. Use this crate to serve or drive environments
//! and models from Rust.
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
//!   observe it. `Model::bind` (with the `lockstep` feature) binds it to
//!   one env to step by hand.
//!
//! Use [`RemoteEnv`] when you want to step an environment server directly.
//!
//! # Options structs
//!
//! Optional arguments use options structs with documented defaults
//! ([`EnvServerOptions`], [`ServeOptions`], [`EnvFactoryServeOptions`]).
//! Build them with `..Default::default()`: by project policy appending an
//! options field is a minor change, and a full struct literal without `..`
//! is unsupported.
//!
//! # Features
//!
//! - `lockstep`: the [`lockstep`] module, Gym envs and v1 model servers
//!   over a v2 session (participant addresses for [`EnvServer`] and
//!   local models). Pulls in `rlmesh-session`.
//!
//! - `viewer` (default): the live terminal/HTTP [`viewer`] behind
//!   [`View`]. Without it a run's `view` warns and shows nothing.
//!
//! - `managed` (default): the [`managed`] module, an Experimental client for
//!   the managed RLMesh platform that shares the `rlmesh` CLI's sign-in.
//!   Pulls in `reqwest`, `rustls` and `keyring`.
//!
//! # Errors
//!
//! Fallible operations return [`Result`], an alias for `Result<T, Error>`.
//! Each [`Error`] variant identifies an error class:
//! [`Error::class`] reports the exact class as an [`ErrorClass`], whose
//! [`ErrorClass::is_subclass_of`] tests class inheritance. `Display`
//! formats the error message. The two domain failures are
//! [`Error::Environment`] (carrying an [`ErrorCode`]) and [`Error::Model`]
//! (a failure your [`Policy`] raised, carrying a [`ModelErrorCode`]). Both
//! carry an `is_recoverable` flag surfaced by [`Error::is_recoverable`],
//! which is false for every other variant.
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
mod credentials;
mod describe;
mod dlpack;
#[cfg(feature = "lockstep")]
mod doctor;
mod dtype;
mod editions;
pub mod env;
mod error;
pub mod ext;
/// The v2 session protocol engine the bindings drive (R-25 binding
/// plumbing, like [`binding`]): hidden and semver-exempt.
#[cfg(feature = "lockstep")]
#[doc(hidden)]
pub mod lockstep;
#[cfg(feature = "managed")]
pub mod managed;
mod model;
pub mod prelude;
pub mod recorder;
mod run;
pub mod serve_options;
pub mod spaces;
pub mod specs;
mod tensor;
pub mod types;
mod value;
mod view;
/// The live viewer engine behind [`View`] (R-25 binding plumbing: runs and
/// the Python binding drive it; users reach it through `View`).
#[cfg(feature = "viewer")]
#[doc(hidden)]
pub mod viewer;
mod warnings;

pub(crate) use address::{BindAddress, ConnectAddress};
pub use build_info::{BuildInfo, BuildSource, VERSION, build_info};
pub use describe::{
    Describable, DescribeJsonOptions, DescribeKind, DescribeOptions, describe, describe_json,
};
pub use dtype::{DType, Device};
pub use editions::current_workflow_edition;

/// The key a serialized describe envelope travels under
/// (`rlmesh.describe.v1`).
pub const DESCRIBE_METADATA_KEY: &str = rlmesh_adapters::v1::DESCRIBE_METADATA_KEY;

/// The describe envelope's schema version.
pub const DESCRIBE_SCHEMA_VERSION: i64 = rlmesh_adapters::v1::DESCRIBE_SCHEMA_VERSION as i64;
#[doc(no_inline)]
pub use crate::run::runtime::ENV_RESET_OPTIONS_KEY;
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
    Corners, Model, ModelOptions, ModelServeOptions, NO_ADAPTER, Policy, RemoteModel,
    RemoteModelOptions, WithContext, predict_seed, with_context,
};
pub use recorder::{
    EpisodeRecord, MediaRef, Recorder, RecorderAddOptions, RecorderCaptureOptions,
    RecorderExportOptions, RecorderOptions, RecorderToDictOptions, VideoCodec, WorkloadRecord,
};
pub use run::{
    BindOptions, EnvTarget, EpisodeResult, FrameSources, ModelTarget, RANDOM_SAMPLE, RandomSample,
    ReadFn, ReadItem, Reader, RenderFn, RunContext, RunHooks, RunOptions, RunResult,
    RunResultSaveOptions, StepEvent, TelemetryRow, run, run_async,
};
#[cfg(feature = "lockstep")]
pub use run::{Binding, bind};
pub use serve_options::{BatchOptions, ServeOptions};
pub use specs::EnvContract;
pub use tensor::{Tensor, ToDlpackOptions};
pub use value::{InfoDict, Kwargs, Metadata, MetadataIter, Value, ValueKind};
pub use view::{View, ViewBackend, ViewFormat, ViewHold};
pub use warnings::{Warning, WarningCategory, WarningHandler, set_warning_handler};

/// Mint one fresh routing/episode id (UUIDv7 — time-ordered, sortable by creation,
/// never repeats). The single id-format home for this crate's id authorities (the
/// direct env client and the in-process / remote model paths). The runtime driver
/// (`run::runtime`) mints its own.
pub(crate) fn mint_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

#[cfg(test)]
mod tests;
