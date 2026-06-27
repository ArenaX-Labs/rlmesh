//! Common imports for RLMesh users.
//!
//! This prelude brings in the traits, servers, request/result types, errors,
//! options, address types, and [`BinaryPayload`]
//! used by environment and model implementations. It leaves the full space
//! system under [`crate::spaces`].
//!
//! [`crate::Result`] is not included. [`Env`] and [`VectorEnv`] methods use the
//! two-argument `std::result::Result<_, EnvRuntimeError>` form.

pub use crate::spaces::BinaryPayload;
pub use crate::{
    BindAddress, BoundEnvServer, BoundModelServer, CloseRequest, CloseResult, ConnectAddress, Env,
    EnvContract, EnvRuntimeError, EnvServer, EnvironmentError, Error, ErrorCode, ModelHandler,
    ModelObservation, ModelRouteContext, ModelWorker, RemoteEnv, RemoteVectorEnv, RenderFrame,
    RenderRequest, RenderResult, ResetRequest, ResetResult, RunLocalOptions, ServeModelOptions,
    ServeOptions, SpaceSpec, SpaceValue, StepRequest, StepResult, VectorCloseResult, VectorEnv,
    VectorEnvServer, VectorResetRequest, VectorResetResult, VectorStepRequest, VectorStepResult,
};
