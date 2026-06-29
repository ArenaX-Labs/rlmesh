//! Env wire helpers: proto<->native error mapping and request-kind labels for
//! tracing.

use rlmesh_proto::env::v1::{
    EnvError as ProtoEnvError, EnvErrorCode as ProtoEnvErrorCode, JoinRequest, join_request,
};

use crate::error::{EnvError, EnvErrorCode};

pub(super) use crate::error::status_to_grpc_error;

pub(super) fn proto_error_to_env_error(error: ProtoEnvError) -> EnvError {
    let code = ProtoEnvErrorCode::try_from(error.code)
        .map(EnvErrorCode::from)
        .unwrap_or(EnvErrorCode::Unspecified);

    EnvError {
        code,
        message: error.message,
        is_recoverable: error.is_recoverable,
        debug_info: if error.debug_info.is_empty() {
            None
        } else {
            Some(error.debug_info)
        },
    }
}

pub(super) fn join_request_kind_name(req: &JoinRequest) -> &'static str {
    match req.kind.as_ref() {
        Some(join_request::Kind::Configure(_)) => "configure",
        Some(join_request::Kind::Reset(_)) => "reset",
        Some(join_request::Kind::Step(_)) => "step",
        Some(join_request::Kind::Render(_)) => "render",
        Some(join_request::Kind::Close(_)) => "close",
        None => "empty",
    }
}
