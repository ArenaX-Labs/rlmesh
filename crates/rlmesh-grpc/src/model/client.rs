use rlmesh_proto::{
    PROTOCOL_GENERATION, capabilities, capability_map,
    core::v1::OperationTelemetry,
    model::v1::{
        CloseRequest, CloseRouteRequest, ConfigureRouteRequest, HandshakeRequest, JoinRequest,
        JoinResponse, PredictRequest, PredictResponse, ShutdownRequest, ShutdownResponse,
        join_request, join_response, model_service_client::ModelServiceClient,
    },
    supported_workflow_editions,
};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::error::{Error as GrpcError, ProtocolError, TransportError};
use crate::helpers::normalize_tcp_session_address;
use crate::states::ClientState;

use super::protocol::{join_request_kind_name, model_error_to_grpc_error};
use super::stream::spawn_response_pump;
use super::validation::{decode_error, route_request_id, validate_predict_route, validate_route};

pub struct ModelClient {
    address: String,
    client: ModelServiceClient<tonic::transport::Channel>,
    token: String,
    state: ClientState,
    request_tx: Option<mpsc::Sender<JoinRequest>>,
    response_rx: Option<mpsc::Receiver<JoinResponse>>,
    request_counter: u64,
    last_telemetry: Option<OperationTelemetry>,
}

impl ModelClient {
    pub async fn connect(address: &str, token: &str) -> Result<Self, GrpcError> {
        let address = normalize_tcp_session_address(address)?;
        let endpoint =
            tonic::transport::Endpoint::from_shared(address.replacen("tcp://", "http://", 1))
                .map_err(|err| TransportError::InvalidAddress(err.to_string()))?;
        let channel = endpoint
            .connect()
            .await
            .map_err(|err| TransportError::ConnectFailed(err.to_string()))?;

        Ok(Self {
            address,
            client: ModelServiceClient::new(channel)
                .max_decoding_message_size(crate::MAX_MESSAGE_SIZE)
                .max_encoding_message_size(crate::MAX_MESSAGE_SIZE),
            token: token.to_string(),
            state: ClientState::Connected,
            request_tx: None,
            response_rx: None,
            request_counter: 0,
            last_telemetry: None,
        })
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    pub fn take_last_telemetry(&mut self) -> Option<OperationTelemetry> {
        self.last_telemetry.take()
    }

    pub async fn handshake(&mut self) -> Result<(), GrpcError> {
        if self.state != ClientState::Connected {
            return Err(crate::error::ClientError::NotConnected.into());
        }

        let request = self.authorized_request(HandshakeRequest {
            protocol_generation: PROTOCOL_GENERATION.to_string(),
            client_name: "rlmesh-rust-model-grpc".to_string(),
            client_version: env!("CARGO_PKG_VERSION").to_string(),
            capabilities: capability_map(&[
                capabilities::MODEL_SERVICE_V1,
                capabilities::SPACES_CORE_V1,
            ]),
            supported_workflow_editions: supported_workflow_editions(),
        })?;

        let response = self
            .client
            .handshake(request)
            .await
            .map_err(|err| TransportError::ConnectFailed(err.to_string()))?
            .into_inner();

        if !response.compatible {
            return Err(ProtocolError::HandshakeFailed(response.error_message).into());
        }

        self.setup_join_stream().await?;
        self.state = ClientState::Ready;
        Ok(())
    }

    pub async fn configure_route(
        &mut self,
        request: ConfigureRouteRequest,
    ) -> Result<(), GrpcError> {
        self.ensure_ready()?;
        validate_route(
            request
                .context
                .as_ref()
                .ok_or_else(|| decode_error("configure_route missing route context"))?,
        )?;
        let request_id = route_request_id(request.context.as_ref(), || self.next_request_id());
        let response = self
            .send_on_stream(JoinRequest {
                kind: Some(join_request::Kind::ConfigureRoute(request)),
                request_id,
            })
            .await?;
        self.last_telemetry = response.telemetry.clone();
        match response.kind {
            Some(join_response::Kind::ConfigureRoute(_)) => Ok(()),
            Some(join_response::Kind::Error(error)) => Err(model_error_to_grpc_error(error)),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "ConfigureRouteResponse".to_string(),
                actual: format!("{:?}", response.kind),
            }
            .into()),
        }
    }

    pub async fn predict(&mut self, request: PredictRequest) -> Result<PredictResponse, GrpcError> {
        self.ensure_ready()?;
        validate_predict_route(
            request
                .context
                .as_ref()
                .ok_or_else(|| decode_error("predict missing route context"))?,
        )?;
        let request_id = route_request_id(request.context.as_ref(), || self.next_request_id());
        let response = self
            .send_on_stream(JoinRequest {
                kind: Some(join_request::Kind::Predict(request)),
                request_id,
            })
            .await?;
        self.last_telemetry = response.telemetry.clone();

        match response.kind {
            Some(join_response::Kind::Predict(predict)) => Ok(predict),
            Some(join_response::Kind::Error(error)) => Err(model_error_to_grpc_error(error)),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "PredictResponse".to_string(),
                actual: format!("{:?}", response.kind),
            }
            .into()),
        }
    }

    pub async fn close_route(&mut self, request: CloseRouteRequest) -> Result<(), GrpcError> {
        self.ensure_ready()?;
        validate_route(
            request
                .context
                .as_ref()
                .ok_or_else(|| decode_error("close_route missing route context"))?,
        )?;
        let request_id = route_request_id(request.context.as_ref(), || self.next_request_id());
        let response = self
            .send_on_stream(JoinRequest {
                kind: Some(join_request::Kind::CloseRoute(request)),
                request_id,
            })
            .await?;
        self.last_telemetry = response.telemetry.clone();
        match response.kind {
            Some(join_response::Kind::CloseRoute(_)) => Ok(()),
            Some(join_response::Kind::Error(error)) => Err(model_error_to_grpc_error(error)),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "CloseRouteResponse".to_string(),
                actual: format!("{:?}", response.kind),
            }
            .into()),
        }
    }

    pub async fn close(&mut self, reason: impl Into<String>) -> Result<(), GrpcError> {
        self.close_with_timeout(reason, Duration::from_secs(5))
            .await
    }

    pub async fn close_with_timeout(
        &mut self,
        reason: impl Into<String>,
        timeout: Duration,
    ) -> Result<(), GrpcError> {
        if self.state == ClientState::Closed {
            return Err(crate::error::ClientError::NotConnected.into());
        }
        self.ensure_ready()?;

        let request = JoinRequest {
            kind: Some(join_request::Kind::Close(CloseRequest {
                reason: reason.into(),
            })),
            request_id: self.next_request_id(),
        };

        let response = tokio::time::timeout(timeout, self.send_on_stream(request))
            .await
            .map_err(|_| GrpcError::Timeout(timeout))??;
        self.last_telemetry = response.telemetry.clone();
        self.state = ClientState::Closed;

        match response.kind {
            Some(join_response::Kind::Close(_)) => Ok(()),
            Some(join_response::Kind::Error(error)) => Err(model_error_to_grpc_error(error)),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "CloseResponse".to_string(),
                actual: format!("{:?}", response.kind),
            }
            .into()),
        }
    }

    pub async fn shutdown(
        &mut self,
        reason: impl Into<String>,
    ) -> Result<ShutdownResponse, GrpcError> {
        if self.state == ClientState::Closed {
            return Err(crate::error::ClientError::NotConnected.into());
        }

        let request = self.authorized_request(ShutdownRequest {
            reason: reason.into(),
        })?;
        let response = self
            .client
            .shutdown(request)
            .await
            .map_err(|err| TransportError::ConnectFailed(err.to_string()))?
            .into_inner();

        if response.accepted {
            self.state = ClientState::Closed;
            self.request_tx.take();
            self.response_rx.take();
        }

        Ok(response)
    }

    async fn setup_join_stream(&mut self) -> Result<(), GrpcError> {
        let (tx, rx) = mpsc::channel::<JoinRequest>(32);
        let request_stream = ReceiverStream::new(rx);
        let request = self.authorized_request(request_stream)?;

        let response = self
            .client
            .join(request)
            .await
            .map_err(|err| TransportError::ConnectFailed(err.to_string()))?;

        self.request_tx = Some(tx);
        self.response_rx = Some(spawn_response_pump(response.into_inner()));
        Ok(())
    }

    async fn send_on_stream(&mut self, request: JoinRequest) -> Result<JoinResponse, GrpcError> {
        let request_id = request.request_id.clone();
        let request_kind = join_request_kind_name(request.kind.as_ref());
        let tx = self
            .request_tx
            .as_ref()
            .ok_or(crate::error::ClientError::NotHandshaked)?;
        tx.send(request)
            .await
            .map_err(|_| TransportError::ConnectionClosed)?;

        let rx = self
            .response_rx
            .as_mut()
            .ok_or(crate::error::ClientError::NotHandshaked)?;
        loop {
            let response = rx.recv().await.ok_or_else(|| {
                tracing::error!(
                    request_id = %request_id,
                    request_kind,
                    "model join stream closed while waiting for response"
                );
                GrpcError::from(TransportError::ConnectionClosed)
            })?;
            if response.request_id == request_id {
                return Ok(response);
            }
            tracing::warn!(
                request_id = %request_id,
                stale_request_id = %response.request_id,
                request_kind,
                response_kind = ?response.kind,
                "discarding stale model response from abandoned request"
            );
        }
    }

    fn ensure_ready(&self) -> Result<(), GrpcError> {
        match self.state {
            ClientState::Ready => Ok(()),
            ClientState::Disconnected => Err(crate::error::ClientError::NotConnected.into()),
            ClientState::Connected => Err(crate::error::ClientError::NotHandshaked.into()),
            ClientState::Closed => Err(crate::error::ClientError::NotConnected.into()),
        }
    }

    fn next_request_id(&mut self) -> String {
        self.request_counter += 1;
        format!("model-grpc-req-{}", self.request_counter)
    }

    fn authorized_request<T>(&self, message: T) -> Result<tonic::Request<T>, GrpcError> {
        let mut request = tonic::Request::new(message);
        if !self.token.is_empty() {
            request.metadata_mut().insert(
                "authorization",
                self.token
                    .parse()
                    .map_err(|_| TransportError::InvalidAddress("invalid token".to_string()))?,
            );
        }
        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_proto::model::v1::{CloseRouteResponse, PredictResponse};
    use tonic::transport::Endpoint;

    #[tokio::test]
    async fn send_on_stream_discards_stale_responses_until_request_id_matches() {
        let (request_tx, mut request_rx) = mpsc::channel(4);
        let (response_tx, response_rx) = mpsc::channel(4);
        let channel = Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        let mut client = ModelClient {
            address: "tcp://127.0.0.1:1".to_string(),
            client: ModelServiceClient::new(channel),
            token: String::new(),
            state: ClientState::Ready,
            request_tx: Some(request_tx),
            response_rx: Some(response_rx),
            request_counter: 0,
            last_telemetry: None,
        };

        response_tx
            .send(JoinResponse {
                request_id: "abandoned".to_string(),
                kind: Some(join_response::Kind::Predict(PredictResponse::default())),
                telemetry: None,
            })
            .await
            .unwrap();
        response_tx
            .send(JoinResponse {
                request_id: "target".to_string(),
                kind: Some(join_response::Kind::CloseRoute(CloseRouteResponse {})),
                telemetry: None,
            })
            .await
            .unwrap();

        let response = client
            .send_on_stream(JoinRequest {
                request_id: "target".to_string(),
                kind: Some(join_request::Kind::CloseRoute(CloseRouteRequest::default())),
            })
            .await
            .unwrap();

        assert!(matches!(
            response.kind,
            Some(join_response::Kind::CloseRoute(_))
        ));
        assert_eq!(request_rx.recv().await.unwrap().request_id, "target");
    }
}
