use rlmesh_proto::model::v1::{
    ModelError as ProtoModelError, ModelErrorCode as ProtoModelErrorCode, join_request,
};

use crate::error::{Error as GrpcError, ModelError, ModelErrorCode};

/// Convert a server-side `ModelError` into a `GrpcError`, preserving the error
/// code, recoverability, message, and debug info instead of collapsing every
/// model error into an opaque (and non-recoverable) protocol decode error.
pub(super) fn model_error_to_grpc_error(error: ProtoModelError) -> GrpcError {
    let code = ProtoModelErrorCode::try_from(error.code)
        .map(ModelErrorCode::from)
        .unwrap_or(ModelErrorCode::Unspecified);

    GrpcError::Model(ModelError {
        code,
        message: error.message,
        is_recoverable: error.is_recoverable,
        debug_info: if error.debug_info.is_empty() {
            None
        } else {
            Some(error.debug_info)
        },
    })
}

pub(super) fn join_request_kind_name(kind: Option<&join_request::Kind>) -> &'static str {
    match kind {
        Some(join_request::Kind::ResolveAdapter(_)) => "resolve_adapter",
        Some(join_request::Kind::Predict(_)) => "predict",
        Some(join_request::Kind::GroupedPredict(_)) => "grouped_predict",
        Some(join_request::Kind::ResetAdapter(_)) => "reset_adapter",
        Some(join_request::Kind::ReleaseAdapter(_)) => "release_adapter",
        Some(join_request::Kind::Close(_)) => "close",
        None => "empty",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error as GrpcError;

    #[test]
    fn preserves_recoverable_model_error_code() {
        let proto = ProtoModelError {
            code: ProtoModelErrorCode::Busy as i32,
            message: "model is busy".to_string(),
            is_recoverable: true,
            debug_info: "queue depth 42".to_string(),
        };

        let error = model_error_to_grpc_error(proto);

        // A recoverable, busy model error must surface as a recoverable Model
        // error -- not as a non-recoverable protocol decode error.
        assert!(error.is_recoverable());
        match error {
            GrpcError::Model(model) => {
                assert_eq!(model.code, ModelErrorCode::Busy);
                assert!(model.is_recoverable);
                assert_eq!(model.message, "model is busy");
                assert_eq!(model.debug_info.as_deref(), Some("queue depth 42"));
            }
            other => panic!("expected Error::Model, got {other:?}"),
        }
    }

    #[test]
    fn preserves_non_recoverable_model_error_code() {
        let proto = ProtoModelError {
            code: ProtoModelErrorCode::Internal as i32,
            message: "boom".to_string(),
            is_recoverable: false,
            debug_info: String::new(),
        };

        let error = model_error_to_grpc_error(proto);
        assert!(!error.is_recoverable());
        match error {
            GrpcError::Model(model) => {
                assert_eq!(model.code, ModelErrorCode::Internal);
                assert!(model.debug_info.is_none());
            }
            other => panic!("expected Error::Model, got {other:?}"),
        }
    }
}
