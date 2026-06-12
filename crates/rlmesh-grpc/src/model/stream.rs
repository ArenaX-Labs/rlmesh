use std::collections::HashMap;
use std::error::Error as StdError;
use std::sync::{Arc, Mutex};

use rlmesh_proto::model::v1::JoinResponse;
use tokio::sync::oneshot;
use tonic::Status;

/// Map of in-flight `request_id` to the oneshot that delivers its response.
///
/// Shared between the client (which inserts a sender before issuing a request)
/// and the response pump (which removes and fulfills the matching sender as each
/// response arrives). This is what makes multiple predicts concurrently
/// in-flight on one Join stream: responses are demuxed by `request_id` rather
/// than assumed to arrive in request order.
pub(super) type PendingResponses =
    Arc<Mutex<HashMap<String, oneshot::Sender<Result<JoinResponse, Status>>>>>;

/// Spawn the response pump for a Join stream.
///
/// The pump reads responses and routes each to the pending sender registered
/// under its `request_id`. A response with no matching pending entry (a late
/// response from an abandoned request, or an unknown id) is logged and dropped.
/// When the stream ends or errors, every still-pending caller is failed so no
/// `await` hangs forever.
pub(super) fn spawn_response_pump(
    mut response_stream: tonic::Streaming<JoinResponse>,
    pending: PendingResponses,
) {
    tokio::spawn(async move {
        loop {
            match response_stream.message().await {
                Ok(Some(message)) => {
                    let request_id = message.request_id.clone();
                    let sender = pending
                        .lock()
                        .expect("pending map poisoned")
                        .remove(&request_id);
                    match sender {
                        Some(sender) => {
                            // The receiver may have been dropped (caller gave up);
                            // that is fine, just drop the response.
                            let _ = sender.send(Ok(message));
                        }
                        None => {
                            tracing::warn!(
                                stale_request_id = %request_id,
                                response_kind = ?message.kind,
                                "discarding model response with no pending request id"
                            );
                        }
                    }
                }
                Ok(None) => {
                    tracing::warn!("model join stream ended");
                    fail_all_pending(&pending, || Status::unavailable("model join stream ended"));
                    break;
                }
                Err(error) => {
                    tracing::error!(
                        code = ?error.code(),
                        message = %error.message(),
                        source = ?error.source(),
                        "model join stream error from server"
                    );
                    // Surface the real Status to every pending caller instead of
                    // letting them observe only an opaque "connection closed".
                    let code = error.code();
                    let message = error.message().to_string();
                    fail_all_pending(&pending, || Status::new(code, message.clone()));
                    break;
                }
            }
        }
    });
}

/// Fail every still-pending request with a freshly built `Status`.
fn fail_all_pending(pending: &PendingResponses, status: impl Fn() -> Status) {
    let drained: Vec<_> = pending
        .lock()
        .expect("pending map poisoned")
        .drain()
        .collect();
    for (_request_id, sender) in drained {
        let _ = sender.send(Err(status()));
    }
}
