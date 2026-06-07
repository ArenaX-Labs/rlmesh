use std::error::Error as StdError;

use rlmesh_proto::env::v1::JoinResponse;
use tokio::sync::mpsc;

pub(super) fn spawn_response_pump(
    mut response_stream: tonic::Streaming<JoinResponse>,
) -> mpsc::Receiver<JoinResponse> {
    let (resp_tx, resp_rx) = mpsc::channel::<JoinResponse>(32);

    tokio::spawn(async move {
        loop {
            match response_stream.message().await {
                Ok(Some(msg)) => {
                    if resp_tx.send(msg).await.is_err() {
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
                    break;
                }
            }
        }
    });

    resp_rx
}
