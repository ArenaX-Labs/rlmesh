//! Common imports for working with rlmesh.
//!
//! `use rlmesh::prelude::*;` brings the traits, servers, request/result, and
//! error types you need to implement an [`Env`] or [`ModelHandler`] into scope,
//! plus the option types ([`RunLocalOptions`], [`ServeModelOptions`],
//! [`ServeOptions`]), the [`BindAddress`] / [`ConnectAddress`] address types,
//! and [`BinaryPayload`](crate::spaces::BinaryPayload) (the encoded action a
//! [`ModelHandler::predict`] returns). It deliberately omits the full
//! space-system surface; reach into [`crate::spaces`] for the rest.
//!
//! The crate-root [`Result`](crate::Result) alias is intentionally **not** in
//! the prelude: it would shadow [`std::result::Result`], and [`Env`] /
//! [`SingleEnv`] methods return `Result<_, EnvRuntimeError>` — a *two*-argument
//! `Result` that the single-arg alias cannot spell. With the alias out of the
//! glob, prelude users write those signatures with the plain two-arg `Result`,
//! and reach for [`crate::Result`] explicitly where they want the single-arg
//! form (e.g. a [`ModelHandler`] return type).

pub use crate::spaces::BinaryPayload;
pub use crate::{
    BindAddress, CloseRequest, CloseResult, ConnectAddress, Env, EnvContract, EnvRuntimeError,
    EnvServer, EnvironmentError, Error, ErrorCode, ModelEpisodeEnd, ModelHandler, ModelObservation,
    ModelRouteContext, ModelRouteSlot, ModelWorker, RemoteEnv, RenderFrame, RenderRequest,
    RenderResult, ResetRequest, ResetResult, RunLocalOptions, ServeModelOptions, ServeOptions,
    SingleEnv, SingleEnvAdapter, SpaceSpec, SpaceValue, StepRequest, StepResult,
};
