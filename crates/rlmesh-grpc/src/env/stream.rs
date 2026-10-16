//! Response pump for the env Join stream: routes each server response to the
//! caller awaiting that `request_id`, so any number of requests (one per lane)
//! can be in flight on the one stream and complete in any order.

use std::collections::HashMap;
use std::error::Error as StdError;
use std::sync::{Arc, Mutex};

use rlmesh_proto::env::v1::{JoinResponse, join_response};
use tokio::sync::{oneshot, watch};
use tonic::Status;

/// Callers awaiting a response, keyed by `request_id`. `None` once the stream
/// has ended: a late registration then fails fast instead of hanging.
pub(super) type Pending =
    Arc<Mutex<Option<HashMap<String, oneshot::Sender<Result<JoinResponse, Status>>>>>>;

/// The terminal outcome of the response stream, independently of any reply.
pub(super) type StreamCompletion = watch::Receiver<Option<Result<(), Status>>>;

// Only sessions whose owner has started closing/dropping are registered here.
// An active client never makes another client wait instead of being refused.
static RETIRING: Mutex<Vec<(String, StreamCompletion)>> = Mutex::new(Vec::new());

pub(super) fn retire_session(address: &str, completion: StreamCompletion) {
    let mut retiring = RETIRING.lock().unwrap_or_else(|e| e.into_inner());
    retiring.retain(|(_, completion)| {
        completion.borrow().is_none() && completion.has_changed().is_ok()
    });
    retiring.push((address.to_string(), completion));
}

/// Order a new local Join behind teardown already started for this address.
/// This covers nonblocking Drop and detach, which cannot await a remote peer.
/// The grace is bounded: an operation still draining afterwards genuinely
/// owns the slot. Return false when the call budget expires so the caller can
/// report an active-session refusal. After the full grace, let the server
/// decide admission if the caller still has time.
pub(super) async fn wait_for_retiring_sessions(
    address: &str,
    deadline: Option<tokio::time::Instant>,
) -> bool {
    let sessions: Vec<_> = RETIRING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|(target, _)| target == address)
        .map(|(_, completion)| completion.clone())
        .collect();
    let grace = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    let until = deadline.map_or(grace, |deadline| deadline.min(grace));
    let completed = tokio::time::timeout_at(until, async {
        for mut completion in sessions {
            let _ = completion.wait_for(|outcome| outcome.is_some()).await;
        }
    })
    .await
    .is_ok();
    completed || deadline.is_none_or(|deadline| deadline > grace)
}

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

/// Read `response_stream` until it ends, routing each reply to its waiter.
///
/// `on_fatal` sees every non-recoverable env error before its waiter does: the
/// server ends the session after one, and that must be recorded even when no
/// one is waiting any more (a caller that dropped its request future on an
/// interrupt or deadline).
pub(super) fn spawn_response_pump(
    mut response_stream: tonic::Streaming<JoinResponse>,
    pending: Pending,
    on_fatal: impl Fn(String) + Send + 'static,
) -> StreamCompletion {
    let (finished_tx, finished_rx) = watch::channel(None);
    tokio::spawn(async move {
        let outcome = loop {
            match response_stream.message().await {
                Ok(Some(msg)) => {
                    if let Some(join_response::Kind::Error(error)) = &msg.kind
                        && !error.is_recoverable
                    {
                        on_fatal(error.message.clone());
                    }
                    dispatch_response(&pending, Some(Ok(msg)));
                }
                Ok(None) => {
                    tracing::debug!("env join stream ended");
                    dispatch_response(&pending, None);
                    break Ok(());
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
                    dispatch_response(&pending, Some(Err(error.clone())));
                    break Err(error);
                }
            }
        };
        let _ = finished_tx.send(Some(outcome));
    });
    finished_rx
}
