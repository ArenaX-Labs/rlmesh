use rlmesh_proto::model::v1::{ModelError, join_request};

use crate::error::{Error as GrpcError, ProtocolError};

pub(super) fn model_error_to_protocol_error(error: ModelError) -> GrpcError {
    GrpcError::Protocol(ProtocolError::DecodeError(error.message))
}

pub(super) fn join_request_kind_name(kind: Option<&join_request::Kind>) -> &'static str {
    match kind {
        Some(join_request::Kind::ConfigureRoute(_)) => "configure_route",
        Some(join_request::Kind::Predict(_)) => "predict",
        Some(join_request::Kind::CloseRoute(_)) => "close_route",
        Some(join_request::Kind::Close(_)) => "close",
        None => "empty",
    }
}
