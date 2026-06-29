//! Client-side validation of model route and predict requests before they hit
//! the wire: a non-empty env/request id, and an ordered, non-empty,
//! duplicate-free episode-id vector on a predict.

use std::collections::HashSet;

use rlmesh_proto::model::v1::{AdapterContext, PredictRequest};

use crate::error::{Error as GrpcError, ProtocolError};

pub(super) fn route_request_id<F>(context: Option<&AdapterContext>, fallback: F) -> String
where
    F: FnOnce() -> String,
{
    context
        .map(|context| context.request_id.clone())
        .filter(|request_id| !request_id.is_empty())
        .unwrap_or_else(fallback)
}

pub(super) fn validate_route(context: &AdapterContext) -> Result<(), GrpcError> {
    if context.env_id.is_empty() {
        return Err(decode_error("model env_id is empty"));
    }
    if context.request_id.is_empty() {
        return Err(decode_error("model request_id is empty"));
    }
    Ok(())
}

pub(super) fn validate_predict_route(request: &PredictRequest) -> Result<(), GrpcError> {
    let context = request
        .context
        .as_ref()
        .ok_or_else(|| decode_error("predict missing adapter context"))?;
    validate_route(context)?;
    // The self-describing batch: an ordered, non-empty, duplicate-free vector of
    // per-row episode ids (replaces the old positional slots).
    if request.episode_ids.is_empty() {
        return Err(decode_error(
            "model predict must include at least one episode_id",
        ));
    }
    let mut seen = HashSet::new();
    for (index, episode_id) in request.episode_ids.iter().enumerate() {
        if episode_id.is_empty() {
            return Err(decode_error(format!(
                "model predict episode_ids[{index}] is empty"
            )));
        }
        if !seen.insert(episode_id.as_str()) {
            return Err(decode_error(format!(
                "model predict has duplicate episode_id {episode_id:?}"
            )));
        }
    }

    Ok(())
}

pub(super) fn decode_error(message: impl Into<String>) -> GrpcError {
    GrpcError::Protocol(ProtocolError::DecodeError(message.into()))
}

#[cfg(test)]
mod tests {
    use rlmesh_proto::model::v1::{AdapterContext, PredictRequest};

    use super::validate_predict_route;

    fn predict_request() -> PredictRequest {
        PredictRequest {
            context: Some(AdapterContext {
                session_id: "session-1".to_string(),
                env_id: "env-1".to_string(),
                request_id: "request-1".to_string(),
            }),
            observation: None,
            episode_ids: vec!["episode-1".to_string()],
        }
    }

    #[test]
    fn model_predict_request_accepts_episode_ids() {
        validate_predict_route(&predict_request()).unwrap();
    }

    #[test]
    fn model_predict_request_rejects_empty_episode_ids() {
        let mut request = predict_request();
        request.episode_ids.clear();

        let err = validate_predict_route(&request).unwrap_err();

        assert!(err.to_string().contains("at least one episode_id"));
    }

    #[test]
    fn model_predict_request_rejects_blank_episode_id() {
        let mut request = predict_request();
        request.episode_ids[0].clear();

        let err = validate_predict_route(&request).unwrap_err();

        assert!(err.to_string().contains("is empty"));
    }

    #[test]
    fn model_predict_request_rejects_duplicate_episode_id() {
        let mut request = predict_request();
        request.episode_ids = vec!["dup".to_string(), "dup".to_string()];

        let err = validate_predict_route(&request).unwrap_err();

        assert!(err.to_string().contains("duplicate episode_id"));
    }
}
