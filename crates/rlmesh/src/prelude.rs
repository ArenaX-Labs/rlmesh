//! Common imports for working with rlmesh.
//!
//! `use rlmesh::prelude::*;` brings the traits, servers, request/result, and
//! error types you need to implement an [`Env`] or [`ModelHandler`] into scope,
//! plus the option types ([`RunLocalOptions`], [`ServeModelOptions`],
//! [`ServeOptions`]), the [`BindAddress`] / [`ConnectAddress`] address types,
//! and [`BinaryPayload`](crate::spaces::BinaryPayload) (the encoded action a
//! [`ModelHandler::predict`] returns). It deliberately omits the full
//! space-system surface; reach into [`crate::spaces`] for the rest.

pub use crate::spaces::BinaryPayload;
pub use crate::{
    BindAddress, CloseRequest, CloseResult, ConnectAddress, Env, EnvContract, EnvRuntimeError,
    EnvServer, EnvironmentError, Error, ErrorCode, ModelEpisodeEnd, ModelHandler, ModelObservation,
    ModelRouteContext, ModelRouteSlot, ModelWorker, RemoteEnv, RenderFrame, RenderRequest,
    RenderResult, ResetRequest, ResetResult, Result, RunLocalOptions, ServeModelOptions,
    ServeOptions, SingleEnv, SingleEnvAdapter, SpaceSpec, SpaceValue, StepRequest, StepResult,
};
