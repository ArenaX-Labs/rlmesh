//! Response pump for the env Join stream: forwards each server response (or a
//! terminal error status) onto a channel the client awaits.

use std::error::Error as StdError;

use rlmesh_proto::env::v1::JoinResponse;
use tokio::sync::mpsc;
use tonic::Status;

pub(super) fn spawn_response_pump(
    mut response_stream: tonic::Streaming<JoinResponse>,
) -> mpsc::Receiver<Result<JoinResponse, Status>> {
    let (resp_tx, resp_rx) = mpsc::channel::<Result<JoinResponse, Status>>(32);

    tokio::spawn(async move {
        loop {
            match response_stream.message().await {
                Ok(Some(msg)) => {
                    if resp_tx.send(Ok(msg)).await.is_err() {
                        tracing::warn!("join stream receiver dropped; stopping response pump");
                        break;
                    }
                }
                Ok(None) => {
                    tracing::debug!("env join stream ended");
                    break;
                }
                Err(error) => {
                    tracing::error!(
                        code = ?error.code(),
                        message = %error.message(),
                        source = ?error.source(),
                        "join stream error from env server"
                    );
                    // Surface the real Status to the pending caller instead of
                    // letting it observe only an opaque "connection closed".
                    let _ = resp_tx.send(Err(error)).await;
                    break;
                }
            }
        }
    });

    resp_rx
}
