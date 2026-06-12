//! Common imports for working with rlmesh.
//!
//! `use rlmesh::prelude::*;` brings the traits, servers, request/result, and
//! error types you need to implement an [`Env`] or [`ModelHandler`] into scope.
//! It deliberately omits the full space-system surface; reach into
//! [`crate::spaces`] for the rest.

pub use crate::{
    BindAddress, CloseRequest, CloseResult, ConnectAddress, Env, EnvContract, EnvRuntimeError,
    EnvServer, EnvironmentError, Error, ErrorCode, ModelEpisodeEnd, ModelHandler, ModelObservation,
    ModelRouteContext, ModelRouteSlot, ModelWorker, RemoteEnv, RenderFrame, RenderRequest,
    RenderResult, ResetRequest, ResetResult, Result, ServeOptions, SingleEnv, SingleEnvAdapter,
    SpaceSpec, SpaceValue, StepRequest, StepResult,
};
