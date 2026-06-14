pub mod client;
mod environment;
mod episode;
mod server;
mod stream;
mod wire;

pub use client::{EnvClient, EnvHandshake};
pub use environment::{
    CloseResponse, Environment, RenderRequest, RenderResponse, ResetRequest, ResetResponse,
    StepRequest, StepResponse,
};
pub use server::{GrpcEnvServer, env_service, env_service_from_shared, serve};

use crate::error::EnvError;
use rlmesh_proto::env::v1::{EnvError as ProtoEnvError, EnvErrorCode as ProtoEnvErrorCode};

pub use rlmesh_proto::PROTOCOL_GENERATION;

pub(crate) fn env_error_to_proto(e: EnvError) -> ProtoEnvError {
    ProtoEnvError {
        code: ProtoEnvErrorCode::from(e.code) as i32,
        message: e.message,
        is_recoverable: e.is_recoverable,
        debug_info: e.debug_info.unwrap_or_default(),
        interrupted_episodes: vec![],
    }
}
