//! Environment client transport implementation using Tonic.

#[cfg(unix)]
use hyper_util::rt::TokioIo;
#[cfg(unix)]
use tokio::net::UnixStream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
#[cfg(unix)]
use tower::service_fn;

use rlmesh_proto::core::v1::OperationTelemetry;
use rlmesh_proto::env::v1::{
    CloseResponse, EnvContract, HandshakeRequest, JoinRequest, JoinResponse, RenderRequest,
    RenderResponse, ResetRequest, ResetResponse, ShutdownRequest, ShutdownResponse, StepRequest,
    StepResponse, env_service_client::EnvServiceClient, join_request, join_response,
};
use rlmesh_proto::{
    PROTOCOL_GENERATION, capabilities, capability_map, supported_workflow_editions,
};

use crate::error::{ClientError, Error as GrpcError, ProtocolError, TransportError};
use crate::helpers::address::parse_env_connect_target;
use crate::states::ClientState;

use super::protocol::{join_request_kind_name, proto_error_to_env_error};
use super::stream::spawn_response_pump;

#[derive(Debug, Clone, PartialEq)]
pub struct EnvHandshake {
    pub env_contract: EnvContract,
    pub num_envs: usize,
    pub server_protocol_generation: String,
    pub workflow_edition: String,
    pub supported_workflow_editions: Vec<String>,
}

/// Environment client that connects to an EnvService server.
pub struct EnvClient {
    /// Inner tonic client for unary RPCs (Handshake, Check).
    client: EnvServiceClient<tonic::transport::Channel>,
    /// Connected address in normalized display form.
    address: String,
    /// Bearer token sent on the `authorization` metadata header (empty = none).
    token: String,
    /// Client state.
    state: ClientState,
    /// Sender half of the Join bidi stream request channel.
    request_tx: Option<mpsc::Sender<JoinRequest>>,
    /// Receiver half of the Join bidi stream response channel.
    response_rx: Option<mpsc::Receiver<Result<JoinResponse, tonic::Status>>>,
    /// Counter for generating unique request IDs.
    request_counter: u64,
    /// Telemetry attached to the last Join response.
    last_telemetry: Option<OperationTelemetry>,
}

impl EnvClient {
    /// Connect to an EnvService server.
    ///
    /// `addr` may be `"host:port"`, `"tcp://host:port"`, `"http://host:port"`,
    /// or `"unix:///path/to/socket"` on Unix.
    pub async fn connect(addr: &str) -> Result<Self, GrpcError> {
        Self::connect_with_token(addr, "").await
    }

    /// Connect to an EnvService server, sending `token` on the `authorization`
    /// metadata header of every request. An empty token sends no header and is
    /// equivalent to [`EnvClient::connect`].
    pub async fn connect_with_token(addr: &str, token: &str) -> Result<Self, GrpcError> {
        let target = parse_env_connect_target(addr)?;

        #[cfg(unix)]
        let channel = if let Some(socket_path) = target.unix_path().cloned() {
            let endpoint = crate::configure_endpoint(
                tonic::transport::Endpoint::from_shared(target.endpoint().to_string())
                    .map_err(|e| TransportError::ConnectFailed(e.to_string()))?,
            );

            endpoint
                .connect_with_connector(service_fn(move |_: tonic::transport::Uri| {
                    let socket_path = socket_path.clone();
                    async move { UnixStream::connect(socket_path).await.map(TokioIo::new) }
                }))
                .await
                .map_err(|e| TransportError::ConnectFailed(e.to_string()))?
        } else {
            let endpoint = crate::configure_endpoint(
                tonic::transport::Endpoint::from_shared(target.endpoint().to_string())
                    .map_err(|e| TransportError::ConnectFailed(e.to_string()))?,
            );
            endpoint
                .connect()
                .await
                .map_err(|e| TransportError::ConnectFailed(e.to_string()))?
        };

        #[cfg(not(unix))]
        let channel = {
            let endpoint = crate::configure_endpoint(
                tonic::transport::Endpoint::from_shared(target.endpoint().to_string())
                    .map_err(|e| TransportError::ConnectFailed(e.to_string()))?,
            );
            endpoint
                .connect()
                .await
                .map_err(|e| TransportError::ConnectFailed(e.to_string()))?
        };

        Ok(Self {
            client: EnvServiceClient::new(channel)
                .max_decoding_message_size(crate::MAX_MESSAGE_SIZE)
                .max_encoding_message_size(crate::MAX_MESSAGE_SIZE),
            address: target.display_address().to_string(),
            token: token.to_string(),
            state: ClientState::Connected,
            request_tx: None,
            response_rx: None,
            request_counter: 0,
            last_telemetry: None,
        })
    }

    /// Connect to an EnvService server, retrying until the server accepts the
    /// connection (or the deadline/cancellation in `options` fires).
    ///
    /// This replaces hand-rolled poll-connect loops used to race a
    /// just-launched server. It retries only the transport connect; perform the
    /// handshake explicitly on the returned client.
    pub async fn connect_with_retry(
        addr: &str,
        token: &str,
        options: &crate::connect::ConnectOptions,
    ) -> Result<Self, GrpcError> {
        crate::connect::retry_connect(options, || Self::connect_with_token(addr, token)).await
    }

    /// Connected address in normalized display form.
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Current client state.
    pub fn state(&self) -> ClientState {
        self.state
    }

    /// Take telemetry attached to the most recent Join response, if any.
    pub fn take_last_telemetry(&mut self) -> Option<OperationTelemetry> {
        self.last_telemetry.take()
    }

    /// Perform the handshake RPC. The Join bidi stream (the env's exclusive
    /// session slot) is opened lazily by the first reset/step/render/close.
    #[tracing::instrument(
        name = "rlmesh.grpc.client.handshake",
        skip_all,
        fields(address = %self.address)
    )]
    pub async fn handshake(&mut self) -> Result<EnvHandshake, GrpcError> {
        if self.state != ClientState::Connected {
            return Err(ClientError::NotConnected.into());
        }

        let res = self.send_handshake().await?;

        if !res.compatible {
            return Err(ProtocolError::HandshakeFailed(res.error_message).into());
        }

        let env_contract = res.env_contract.ok_or_else(|| {
            GrpcError::from(ProtocolError::HandshakeFailed(
                "no env_contract in response".to_string(),
            ))
        })?;
        validate_env_contract(&env_contract)?;
        let num_envs = usize::try_from(env_contract.num_envs)
            .unwrap_or(usize::MAX)
            .max(1);
        let handshake = EnvHandshake {
            env_contract,
            num_envs,
            server_protocol_generation: res.server_protocol_generation,
            workflow_edition: res.selected_workflow_edition,
            supported_workflow_editions: res.supported_workflow_editions,
        };
        self.state = ClientState::Ready;

        Ok(handshake)
    }

    async fn send_handshake(
        &mut self,
    ) -> Result<rlmesh_proto::env::v1::HandshakeResponse, GrpcError> {
        let req = HandshakeRequest {
            protocol_generation: PROTOCOL_GENERATION.to_string(),
            client_name: "rlmesh-rust-grpc".to_string(),
            client_version: env!("CARGO_PKG_VERSION").to_string(),
            capabilities: capability_map(&[
                capabilities::ENV_SERVICE_V1,
                capabilities::SPACES_CORE_V1,
            ]),
            supported_workflow_editions: supported_workflow_editions(),
        };

        Ok(self
            .client
            .handshake(self.authorized_request(req)?)
            .await
            .map_err(crate::error::status_to_grpc_error)?
            .into_inner())
    }

    /// Reset the environment.
    #[tracing::instrument(
        name = "rlmesh.grpc.client.reset",
        skip_all,
        fields(address = %self.address)
    )]
    pub async fn reset(&mut self, req: ResetRequest) -> Result<ResetResponse, GrpcError> {
        self.ensure_ready()?;
        self.ensure_join_stream().await?;

        let env_req = JoinRequest {
            kind: Some(join_request::Kind::Reset(req)),
            request_id: self.next_request_id(),
        };

        let res = self.send_on_stream(env_req).await?;
        self.last_telemetry = res.telemetry.clone();

        match res.kind {
            Some(join_response::Kind::Reset(ok)) => Ok(ok),
            Some(join_response::Kind::Error(e)) => Err(proto_error_to_env_error(e).into()),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "ResetResponse".to_string(),
                actual: format!("{:?}", res.kind),
            }
            .into()),
        }
    }

    /// Take a step in the environment.
    #[tracing::instrument(
        name = "rlmesh.grpc.client.step",
        skip_all,
        fields(address = %self.address)
    )]
    pub async fn step(&mut self, req: StepRequest) -> Result<StepResponse, GrpcError> {
        self.ensure_ready()?;
        self.ensure_join_stream().await?;

        let env_req = JoinRequest {
            kind: Some(join_request::Kind::Step(req)),
            request_id: self.next_request_id(),
        };

        let res = self.send_on_stream(env_req).await?;
        self.last_telemetry = res.telemetry.clone();

        match res.kind {
            Some(join_response::Kind::Step(ok)) => Ok(ok),
            Some(join_response::Kind::Error(e)) => Err(proto_error_to_env_error(e).into()),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "StepResponse".to_string(),
                actual: format!("{:?}", res.kind),
            }
            .into()),
        }
    }

    /// Render the environment.
    #[tracing::instrument(
        name = "rlmesh.grpc.client.render",
        skip_all,
        fields(address = %self.address)
    )]
    pub async fn render(&mut self, req: RenderRequest) -> Result<RenderResponse, GrpcError> {
        self.ensure_ready()?;
        self.ensure_join_stream().await?;

        let env_req = JoinRequest {
            kind: Some(join_request::Kind::Render(req)),
            request_id: self.next_request_id(),
        };

        let res = self.send_on_stream(env_req).await?;
        self.last_telemetry = res.telemetry.clone();

        match res.kind {
            Some(join_response::Kind::Render(ok)) => Ok(ok),
            Some(join_response::Kind::Error(e)) => Err(proto_error_to_env_error(e).into()),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "RenderResponse".to_string(),
                actual: format!("{:?}", res.kind),
            }
            .into()),
        }
    }

    /// Close this client session on the remote environment server.
    /// Close this client's session on the server and tear down the local Join
    /// stream.
    ///
    /// This ends the **session**, not the **server**: the served environment
    /// detaches the session and remains available for a subsequent client to
    /// connect and run a new session. It does not stop the server process — use
    /// [`EnvClient::shutdown`] or the server's idle/drain policy for that. This
    /// reuse-across-sessions behavior is intentional (review finding #81).
    pub async fn close(&mut self) -> Result<CloseResponse, GrpcError> {
        self.ensure_ready()?;
        self.ensure_join_stream().await?;

        let env_req = JoinRequest {
            kind: Some(join_request::Kind::Close(
                rlmesh_proto::env::v1::CloseRequest {
                    reason: "client close".to_string(),
                },
            )),
            request_id: self.next_request_id(),
        };

        let res = self.send_on_stream(env_req).await?;
        self.last_telemetry = res.telemetry.clone();
        self.close_local();

        match res.kind {
            Some(join_response::Kind::Close(ok)) => Ok(ok),
            Some(join_response::Kind::Error(e)) => Err(proto_error_to_env_error(e).into()),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "CloseResponse".to_string(),
                actual: format!("{:?}", res.kind),
            }
            .into()),
        }
    }

    /// Request owner-level shutdown of the remote environment endpoint.
    pub async fn shutdown(
        &mut self,
        reason: impl Into<String>,
    ) -> Result<ShutdownResponse, GrpcError> {
        if self.state == ClientState::Closed {
            return Err(ClientError::NotConnected.into());
        }

        let response = self
            .client
            .shutdown(self.authorized_request(ShutdownRequest {
                reason: reason.into(),
            })?)
            .await
            .map_err(crate::error::status_to_grpc_error)?
            .into_inner();

        if response.accepted {
            self.close_local();
        }

        Ok(response)
    }

    // ---- Private helpers ----

    fn close_local(&mut self) {
        self.request_tx.take();
        self.response_rx.take();
        self.state = ClientState::Closed;
    }

    /// Open the Join stream on first use. The stream is the env's exclusive
    /// session slot (the server admits one Join at a time), so it is acquired
    /// lazily on the first streaming operation rather than at handshake —
    /// an idle connected client must not lock other clients out of the env.
    async fn ensure_join_stream(&mut self) -> Result<(), GrpcError> {
        if self.request_tx.is_none() || self.response_rx.is_none() {
            self.setup_join_stream().await?;
        }
        Ok(())
    }

    async fn setup_join_stream(&mut self) -> Result<(), GrpcError> {
        let (tx, rx) = mpsc::channel::<JoinRequest>(32);
        let request_stream = ReceiverStream::new(rx);

        let response = self
            .client
            .join(self.authorized_request(request_stream)?)
            .await
            .map_err(crate::error::status_to_grpc_error)?;

        self.request_tx = Some(tx);
        self.response_rx = Some(spawn_response_pump(response.into_inner()));
        Ok(())
    }

    /// Wrap a message in a `tonic::Request`, attaching the `authorization`
    /// metadata header when a token is configured.
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

    #[tracing::instrument(
        name = "rlmesh.grpc.client.join_roundtrip",
        skip_all,
        fields(
            address = %self.address,
            request_id = %req.request_id,
            request_kind = join_request_kind_name(&req)
        )
    )]
    async fn send_on_stream(&mut self, req: JoinRequest) -> Result<JoinResponse, GrpcError> {
        let request_id = req.request_id.clone();
        let request_kind = join_request_kind_name(&req);
        let tx = self.request_tx.as_ref().ok_or(ClientError::NotHandshaked)?;

        tx.send(req).await.map_err(|_| {
            tracing::error!(
                request_id = %request_id,
                request_kind,
                "failed to send request because the env join stream is closed"
            );
            TransportError::ConnectionClosed
        })?;

        let rx = self
            .response_rx
            .as_mut()
            .ok_or(ClientError::NotHandshaked)?;

        loop {
            let response = rx.recv().await.ok_or_else(|| {
                tracing::error!(
                    request_id = %request_id,
                    request_kind,
                    "env join stream closed while waiting for response"
                );
                GrpcError::from(TransportError::ConnectionClosed)
            })?;
            let response = match response {
                Ok(response) => response,
                Err(status) => {
                    tracing::error!(
                        request_id = %request_id,
                        request_kind,
                        code = ?status.code(),
                        message = %status.message(),
                        "env join stream returned an error status"
                    );
                    return Err(super::protocol::status_to_grpc_error(status));
                }
            };
            if response.request_id == request_id {
                return Ok(response);
            }
            tracing::warn!(
                request_id = %request_id,
                stale_request_id = %response.request_id,
                request_kind,
                response_kind = ?response.kind,
                "discarding stale env response from abandoned request"
            );
        }
    }

    fn ensure_ready(&self) -> Result<(), GrpcError> {
        match self.state {
            ClientState::Ready => Ok(()),
            ClientState::Disconnected => Err(ClientError::NotConnected.into()),
            ClientState::Connected => Err(ClientError::NotHandshaked.into()),
            ClientState::Closed => Err(ClientError::NotConnected.into()),
        }
    }

    fn next_request_id(&mut self) -> String {
        self.request_counter += 1;
        format!("grpc-req-{}", self.request_counter)
    }
}

fn validate_env_contract(env_contract: &EnvContract) -> Result<(), GrpcError> {
    if env_contract.observation_space.is_none() {
        return Err(ProtocolError::HandshakeFailed(
            "env_contract missing observation_space".to_string(),
        )
        .into());
    }
    if env_contract.action_space.is_none() {
        return Err(ProtocolError::HandshakeFailed(
            "env_contract missing action_space".to_string(),
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_proto::env::v1::env_service_server::{EnvService, EnvServiceServer};
    use rlmesh_proto::env::v1::{
        CloseRequest, CloseResponse, HandshakeRequest, HandshakeResponse, ShutdownRequest,
        ShutdownResponse, StepResponse,
    };
    use rlmesh_proto::spaces::v1::SpaceSpec;
    use rlmesh_proto::{
        CURRENT_WORKFLOW_EDITION, MIN_SUPPORTED_PROTOCOL_GENERATION, PROTOCOL_GENERATION,
        supported_workflow_editions,
    };
    use tokio::sync::oneshot;
    use tokio_stream::wrappers::ReceiverStream;
    use tonic::transport::Endpoint;
    use tonic::{Request, Response, Status};

    #[test]
    fn validate_env_contract_requires_spaces() {
        let valid = EnvContract {
            observation_space: Some(SpaceSpec::default()),
            action_space: Some(SpaceSpec::default()),
            ..Default::default()
        };
        assert!(validate_env_contract(&valid).is_ok());

        let missing_observation = EnvContract {
            action_space: Some(SpaceSpec::default()),
            ..Default::default()
        };
        let err = validate_env_contract(&missing_observation).unwrap_err();
        assert!(err.to_string().contains("missing observation_space"));

        let missing_action = EnvContract {
            observation_space: Some(SpaceSpec::default()),
            ..Default::default()
        };
        let err = validate_env_contract(&missing_action).unwrap_err();
        assert!(err.to_string().contains("missing action_space"));
    }

    #[tokio::test]
    async fn send_on_stream_discards_stale_responses_until_request_id_matches() {
        let (request_tx, mut request_rx) = mpsc::channel(4);
        let (response_tx, response_rx) = mpsc::channel(4);
        let channel = Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        let mut client = EnvClient {
            client: EnvServiceClient::new(channel),
            token: String::new(),
            address: "http://127.0.0.1:1".to_string(),
            state: ClientState::Ready,
            request_tx: Some(request_tx),
            response_rx: Some(response_rx),
            request_counter: 0,
            last_telemetry: None,
        };

        response_tx
            .send(Ok(JoinResponse {
                request_id: "abandoned".to_string(),
                kind: Some(join_response::Kind::Step(StepResponse::default())),
                telemetry: None,
            }))
            .await
            .unwrap();
        response_tx
            .send(Ok(JoinResponse {
                request_id: "target".to_string(),
                kind: Some(join_response::Kind::Close(CloseResponse::default())),
                telemetry: None,
            }))
            .await
            .unwrap();

        let response = client
            .send_on_stream(JoinRequest {
                request_id: "target".to_string(),
                kind: Some(join_request::Kind::Close(CloseRequest::default())),
            })
            .await
            .unwrap();

        assert!(matches!(response.kind, Some(join_response::Kind::Close(_))));
        assert_eq!(request_rx.recv().await.unwrap().request_id, "target");
    }

    #[tokio::test]
    async fn send_on_stream_surfaces_pump_status_error_to_caller() {
        let (request_tx, _request_rx) = mpsc::channel(1);
        let (response_tx, response_rx) = mpsc::channel(1);
        let channel = Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        let mut client = EnvClient {
            client: EnvServiceClient::new(channel),
            token: String::new(),
            address: "tcp://127.0.0.1:1".to_string(),
            state: ClientState::Ready,
            request_tx: Some(request_tx),
            response_rx: Some(response_rx),
            request_counter: 0,
            last_telemetry: None,
        };

        // The response pump propagates a transport Status (e.g. a response that
        // exceeded the decode limit) instead of just dropping it. The pending
        // caller must observe that status, not an opaque "connection closed".
        response_tx
            .send(Err(tonic::Status::new(
                tonic::Code::ResourceExhausted,
                "message length too large",
            )))
            .await
            .unwrap();

        let error = client
            .step(StepRequest::default())
            .await
            .expect_err("a stream status error must surface to the caller");

        let message = error.to_string();
        assert!(
            message.contains("message length too large"),
            "expected the gRPC status message to survive, got: {message}"
        );
        assert!(
            !matches!(
                error,
                GrpcError::Transport(TransportError::ConnectionClosed)
            ),
            "status error was collapsed into opaque ConnectionClosed"
        );
    }

    #[tokio::test]
    async fn close_sends_remote_close_then_closes_locally() {
        let (request_tx, mut request_rx) = mpsc::channel(1);
        let (response_tx, response_rx) = mpsc::channel(1);
        let channel = Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        let mut client = EnvClient {
            client: EnvServiceClient::new(channel),
            token: String::new(),
            address: "tcp://127.0.0.1:1".to_string(),
            state: ClientState::Ready,
            request_tx: Some(request_tx),
            response_rx: Some(response_rx),
            request_counter: 0,
            last_telemetry: None,
        };

        response_tx
            .send(Ok(JoinResponse {
                request_id: "grpc-req-1".to_string(),
                kind: Some(join_response::Kind::Close(CloseResponse::default())),
                telemetry: None,
            }))
            .await
            .unwrap();

        let response = client.close().await.unwrap();

        assert!(response.final_episodes.is_empty());
        assert_eq!(client.state(), ClientState::Closed);
        let request = request_rx.recv().await.unwrap();
        assert!(matches!(request.kind, Some(join_request::Kind::Close(_))));
        assert_eq!(request.request_id, "grpc-req-1");
    }

    #[derive(Default)]
    struct RejectJoinService;

    #[async_trait::async_trait]
    impl EnvService for RejectJoinService {
        async fn handshake(
            &self,
            _request: Request<HandshakeRequest>,
        ) -> std::result::Result<Response<HandshakeResponse>, Status> {
            Ok(Response::new(HandshakeResponse {
                compatible: true,
                server_protocol_generation: PROTOCOL_GENERATION.to_string(),
                min_supported_protocol_generation: MIN_SUPPORTED_PROTOCOL_GENERATION.to_string(),
                selected_workflow_edition: CURRENT_WORKFLOW_EDITION.to_string(),
                supported_workflow_editions: supported_workflow_editions(),
                env_contract: Some(EnvContract {
                    observation_space: Some(SpaceSpec::default()),
                    action_space: Some(SpaceSpec::default()),
                    num_envs: 1,
                    ..Default::default()
                }),
                ..Default::default()
            }))
        }

        type JoinStream = ReceiverStream<std::result::Result<JoinResponse, Status>>;

        async fn join(
            &self,
            _request: Request<tonic::Streaming<JoinRequest>>,
        ) -> std::result::Result<Response<Self::JoinStream>, Status> {
            Err(Status::unavailable("join unavailable"))
        }

        async fn shutdown(
            &self,
            _request: Request<ShutdownRequest>,
        ) -> std::result::Result<Response<ShutdownResponse>, Status> {
            Ok(Response::new(ShutdownResponse {
                accepted: true,
                ..Default::default()
            }))
        }
    }

    #[tokio::test]
    async fn connect_with_token_authenticates_against_token_server() {
        use crate::env::server::GrpcEnvServer;
        use crate::lifecycle::{ServeOptions, ShutdownTrigger};
        use rlmesh_proto::env::v1::env_service_server::EnvServiceServer;
        use rlmesh_spaces::{EnvContract as SpaceEnvContract, SpaceSpec};

        struct TokenEnv {
            contract: SpaceEnvContract,
        }
        #[async_trait::async_trait]
        impl crate::env::Environment for TokenEnv {
            fn observation_space(&self) -> &SpaceSpec {
                self.contract.observation_space.as_ref().unwrap()
            }
            fn action_space(&self) -> &SpaceSpec {
                self.contract.action_space.as_ref().unwrap()
            }
            fn num_envs(&self) -> usize {
                1
            }
            fn env_contract(&self) -> &SpaceEnvContract {
                &self.contract
            }
            async fn reset(
                &mut self,
                _req: ResetRequest,
            ) -> std::result::Result<ResetResponse, crate::error::EnvError> {
                Ok(ResetResponse::default())
            }
            async fn step(
                &mut self,
                _req: StepRequest,
            ) -> std::result::Result<StepResponse, crate::error::EnvError> {
                Ok(StepResponse::default())
            }
            async fn render(
                &mut self,
                _req: RenderRequest,
            ) -> std::result::Result<RenderResponse, crate::error::EnvError> {
                Ok(RenderResponse::default())
            }
            async fn close(
                &mut self,
            ) -> std::result::Result<CloseResponse, crate::error::EnvError> {
                Ok(CloseResponse::default())
            }
        }

        let space = SpaceSpec::default();
        let env = TokenEnv {
            contract: SpaceEnvContract {
                id: "token-env".to_string(),
                action_space: Some(space.clone()),
                observation_space: Some(space),
                metadata: None,
                render_mode: String::new(),
                num_envs: 1,
            },
        };
        let options = ServeOptions {
            token: Some("s3cret".to_string()),
            ..Default::default()
        };
        let service = EnvServiceServer::new(GrpcEnvServer::new_with_options(
            env,
            ShutdownTrigger::new(),
            options,
            None,
        ));

        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_shutdown(addr, async {
                    let _ = shutdown_rx.await;
                })
                .await
        });

        let address = format!("tcp://{addr}");

        // A client without the token is rejected at handshake.
        let connect_options =
            crate::connect::ConnectOptions::with_deadline(std::time::Duration::from_secs(5))
                .backoff(std::time::Duration::from_millis(10));
        let mut anon = EnvClient::connect_with_retry(&address, "", &connect_options)
            .await
            .expect("test server did not start");
        let err = anon.handshake().await.unwrap_err();
        assert!(
            err.to_string().contains("invalid env token"),
            "unauthenticated handshake should be rejected, got: {err}"
        );

        // A client with the correct token handshakes successfully.
        let mut authed = EnvClient::connect_with_token(&address, "s3cret")
            .await
            .unwrap();
        authed.handshake().await.expect("authorized handshake");
        assert_eq!(authed.state(), ClientState::Ready);

        let _ = shutdown_tx.send(());
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), server).await;
    }

    #[tokio::test]
    async fn join_failure_surfaces_on_first_operation_and_leaves_client_usable() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(EnvServiceServer::new(RejectJoinService))
                .serve_with_shutdown(addr, async {
                    let _ = shutdown_rx.await;
                })
                .await
        });

        let address = format!("tcp://{addr}");
        let connect_options =
            crate::connect::ConnectOptions::with_deadline(std::time::Duration::from_secs(5))
                .backoff(std::time::Duration::from_millis(10));
        let mut client = EnvClient::connect_with_retry(&address, "", &connect_options)
            .await
            .expect("test server did not start");

        // The handshake itself succeeds: the Join stream (the exclusive
        // session slot) is only acquired lazily by the first streaming op.
        client.handshake().await.expect("handshake is join-free");
        assert_eq!(client.state(), ClientState::Ready);
        assert!(client.request_tx.is_none());
        assert!(client.response_rx.is_none());

        // The join failure surfaces on the first operation and leaves the
        // client un-wedged (no half-open stream state).
        let error = client
            .reset(ResetRequest::default())
            .await
            .expect_err("join is unavailable");
        assert!(error.to_string().contains("join unavailable"));
        assert_eq!(client.state(), ClientState::Ready);
        assert!(client.request_tx.is_none());
        assert!(client.response_rx.is_none());

        let _ = shutdown_tx.send(());
        tokio::time::timeout(std::time::Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}
