//! Common imports for RLMesh users.
//!
//! This prelude brings in the traits, servers, request/result types, errors,
//! options, address types, and [`BinaryPayload`]
//! used by environment and model implementations. It leaves the full space
//! system under [`crate::spaces`].
//!
//! [`crate::Result`] is not included. [`Env`] and [`SingleEnv`] methods use the
//! two-argument `std::result::Result<_, EnvRuntimeError>` form.

pub use crate::spaces::BinaryPayload;
pub use crate::{
    BindAddress, BoundEnvServer, BoundModelServer, CloseRequest, CloseResult, ConnectAddress, Env,
    EnvContract, EnvRuntimeError, EnvServer, EnvironmentError, Error, ErrorCode, ModelEpisodeEnd,
    ModelHandler, ModelObservation, ModelRouteContext, ModelRouteSlot, ModelWorker, RemoteEnv,
    RenderFrame, RenderRequest, RenderResult, ResetRequest, ResetResult, RunLocalOptions,
    ServeModelOptions, ServeOptions, SingleEnv, SingleEnvAdapter, SpaceSpec, SpaceValue,
    StepRequest, StepResult,
};
