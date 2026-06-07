mod address;
pub mod env;
mod error;
pub mod lifecycle;
pub mod model;
pub mod prelude;
mod single;
pub mod spaces;

pub use address::{BindAddress, ConnectAddress};
pub use env::{
    CloseRequest, CloseResult, Env, EnvServer, EpisodeMetadata, RemoteEnv, RenderRequest,
    RenderResult, ResetRequest, ResetResult, StepRequest, StepResult,
};
pub use error::{EnvironmentError, Error, ErrorCode, Result};
pub use lifecycle::ServeOptions;
pub use model::{
    ModelEpisodeEnd, ModelHandler, ModelObservation, ModelRouteContext, ModelRouteSlot, ModelWorker,
};
pub use single::{SingleEnv, SingleEnvAdapter};
pub use spaces::{EnvContract, EnvRuntimeError, RenderFrame, SpaceSpec, SpaceValue};

#[cfg(test)]
mod tests;
