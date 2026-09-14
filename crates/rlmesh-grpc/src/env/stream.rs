//! Response pump for the env Join stream: routes each server response to the
//! caller awaiting that `request_id`, so any number of requests (one per lane)
//! can be in flight on the one stream and complete in any order.

use std::collections::HashMap;
use std::error::Error as StdError;
use std::sync::{Arc, Mutex};

use rlmesh_proto::env::v1::JoinResponse;
use tokio::sync::oneshot;
use tonic::Status;

/// Callers awaiting a response, keyed by `request_id`. `None` once the stream
/// has ended: a late registration then fails fast instead of hanging.
pub(super) type Pending =
    Arc<Mutex<Option<HashMap<String, oneshot::Sender<Result<JoinResponse, Status>>>>>>;

pub(super) fn new_pending() -> Pending {
    Arc::new(Mutex::new(Some(HashMap::new())))
}

/// Deliver one pump event: a response to its waiter, or a terminal error /
/// end-of-stream to every waiter (the map is closed either way).
pub(super) fn dispatch_response(pending: &Pending, event: Option<Result<JoinResponse, Status>>) {
    let mut guard = pending
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match event {
        Some(Ok(response)) => {
            let waiter = guard
                .as_mut()
                .and_then(|map| map.remove(&response.request_id));
            match waiter {
                Some(tx) => {
                    let _ = tx.send(Ok(response));
                }
                None => tracing::warn!(
                    request_id = %response.request_id,
                    response_kind = ?response.kind,
                    "discarding env response with no waiting request (abandoned or duplicate)"
                ),
            }
        }
        Some(Err(status)) => {
            if let Some(map) = guard.take() {
                for (_, tx) in map {
                    let _ = tx.send(Err(status.clone()));
                }
            }
        }
        // End of stream: dropping the senders wakes every waiter with a
        // receive error, which the client maps to ConnectionClosed.
        None => {
            guard.take();
        }
    }
}

pub(super) fn spawn_response_pump(
    mut response_stream: tonic::Streaming<JoinResponse>,
    pending: Pending,
) {
    tokio::spawn(async move {
        loop {
            match response_stream.message().await {
                Ok(Some(msg)) => dispatch_response(&pending, Some(Ok(msg))),
                Ok(None) => {
                    tracing::debug!("env join stream ended");
                    dispatch_response(&pending, None);
                    break;
                }
                Err(error) => {
                    tracing::error!(
                        code = ?error.code(),
                        message = %error.message(),
                        source = ?error.source(),
                        "join stream error from env server"
                    );
                    // Surface the real Status to the pending callers instead of
                    // letting them observe only an opaque "connection closed".
                    dispatch_response(&pending, Some(Err(error)));
                    break;
                }
            }
        }
    });
}
