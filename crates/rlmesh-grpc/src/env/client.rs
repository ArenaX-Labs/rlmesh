//! Environment client transport implementation using Tonic.

#[cfg(unix)]
use hyper_util::rt::TokioIo;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
#[cfg(unix)]
use tokio::net::UnixStream;

use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
#[cfg(unix)]
use tower::service_fn;

use rlmesh_proto::core::v1::{
    EnvContract, ShutdownRequest as CoreShutdownRequest, ShutdownResponse as CoreShutdownResponse,
};
use rlmesh_proto::env::v1::{
    CloseEnvsResponse, HandshakeRequest, HandshakeResponse, JoinRequest, JoinResponse,
    RenderRequest, RenderResponse, ResetRequest, ResetResponse, ShutdownRequest, StepRequest,
    StepResponse, env_service_client::EnvServiceClient, join_request, join_response,
};
use rlmesh_proto::{EndpointPhases, negotiate_workflow_edition, supported_workflow_editions};

use crate::error::{ClientError, Error as GrpcError, ProtocolError, TransportError};
use crate::helpers::address::parse_env_connect_target;
use crate::states::ClientState;

#[cfg(test)]
use super::stream::dispatch_response;
use super::stream::{Pending, new_pending, spawn_response_pump};
use super::wire::{join_request_kind_name, proto_error_to_env_error};

/// The result of an env handshake: the env contract plus the negotiated edition
/// and the env's advertised window.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvHandshake {
    /// The env contract (spaces, id, render mode, metadata) the server returned.
    pub env_contract: EnvContract,
    /// Number of sub-environments the server steps in lockstep (at least 1).
    pub num_envs: usize,
    /// The workflow edition this (co-located) client selected with the env — the
    /// highest edition both support.
    pub workflow_edition: String,
    /// The editions the env advertised, for the three-way session floor.
    pub supported_workflow_editions: Vec<String>,
    /// Optional features the server advertised (advisory; query with
    /// [`rlmesh_proto::has_capability`]).
    pub capabilities: std::collections::HashMap<String, String>,
}

impl EnvHandshake {
    /// The env's bind-time offer: the workflow editions it supports. Generation
    /// is gated by equality at the handshake (carried by `base.compatible`), so it
    /// is not part of the offer. Capabilities are read pairwise from
    /// [`capabilities`](Self::capabilities), not negotiated. Feeds
    /// [`rlmesh_proto::negotiate_session_floor`] as the env's offer.
    pub fn session_offer(&self) -> rlmesh_proto::SessionOffer {
        rlmesh_proto::SessionOffer {
            editions: self.supported_workflow_editions.clone(),
        }
    }
}

/// Environment client that connects to an EnvService server.
///
/// Cloning yields another handle on the **same session**: one Join stream,
/// one request-id space, one lifecycle. Each clone can have its own request in
/// flight (the stream multiplexes by `request_id`), which is how a lane-capable
/// env is driven: one handle per lane, all stepping concurrently. Closing any
/// handle closes the session for all of them.
#[derive(Clone)]
pub struct EnvClient {
    /// Inner tonic client for unary RPCs (Handshake, Check).
    client: EnvServiceClient<tonic::transport::Channel>,
    /// Connected address in normalized display form.
    address: String,
    /// Bearer token sent on the `authorization` metadata header (empty = none).
    token: String,
    /// Session state shared by every clone: lifecycle, the Join stream, ids.
    shared: Arc<Shared>,
    /// Endpoint-local op duration (ns) attached to the last Join response. The
    /// nested per-step telemetry message was replaced by this hot scalar
    /// (`JoinResponse.endpoint_total_ns`). Per handle: each lane reads its own.
    last_endpoint_total_ns: Option<u64>,
    /// The peer's split of that duration, cleared by the read.
    last_phases: EndpointPhases,
}

/// The per-session state behind every clone of an [`EnvClient`].
struct Shared {
    /// Client state (a `ClientState` discriminant).
    state: AtomicU8,
    /// The open Join stream, if any: the env's exclusive session slot.
    stream: std::sync::Mutex<Option<JoinStream>>,
    /// Serializes stream opening so racing clones cannot each open a Join
    /// (the server admits one).
    open_lock: tokio::sync::Mutex<()>,
    /// Counter for generating unique request IDs across all clones.
    request_counter: AtomicU64,
}

/// The Join bidi stream: where requests go, and who is waiting for a reply.
#[derive(Clone)]
struct JoinStream {
    tx: mpsc::Sender<JoinRequest>,
    pending: Pending,
}

impl Shared {
    fn new(state: ClientState, stream: Option<JoinStream>) -> Self {
        Self {
            state: AtomicU8::new(state as u8),
            stream: std::sync::Mutex::new(stream),
            open_lock: tokio::sync::Mutex::new(()),
            request_counter: AtomicU64::new(0),
        }
    }

    fn state(&self) -> ClientState {
        match self.state.load(Ordering::Acquire) {
            0 => ClientState::Connected,
            1 => ClientState::Ready,
            _ => ClientState::Closed,
        }
    }

    fn set_state(&self, state: ClientState) {
        self.state.store(state as u8, Ordering::Release);
    }

    fn stream(&self) -> Option<JoinStream> {
        self.stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn set_stream(&self, stream: Option<JoinStream>) {
        *self
            .stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = stream;
    }
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
            shared: Arc::new(Shared::new(ClientState::Connected, None)),
            last_endpoint_total_ns: None,
            last_phases: EndpointPhases::default(),
        })
    }

    /// Connect to an EnvService server, retrying until the server accepts the
    /// connection (or the deadline/cancellation in `options` fires).
    ///
    /// Only the transport connect is retried; perform the handshake explicitly
    /// on the returned client.
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

    /// Current client state (shared by every clone of this session).
    pub fn state(&self) -> ClientState {
        self.shared.state()
    }

    /// Take the endpoint-local op duration (ns) attached to the most recent
    /// Join response, if any (`JoinResponse.endpoint_total_ns`).
    pub fn take_last_endpoint_total_ns(&mut self) -> Option<u64> {
        self.last_endpoint_total_ns.take()
    }

    /// The peer's split of that duration; all-zero for a peer that reports none.
    pub fn take_last_phases(&mut self) -> EndpointPhases {
        std::mem::take(&mut self.last_phases)
    }

    /// Perform the handshake RPC. The Join bidi stream (the env's exclusive
    /// session slot) is opened lazily by the first reset/step/render/close.
    #[tracing::instrument(
        name = "rlmesh.grpc.client.handshake",
        skip_all,
        fields(address = %self.address)
    )]
    pub async fn handshake(&mut self) -> Result<EnvHandshake, GrpcError> {
        if self.state() != ClientState::Connected {
            return Err(ClientError::NotConnected.into());
        }

        let res = self.send_handshake().await?;

        // The env Handshake returns a thin service-specific wrapper: the
        // shared negotiation result is in `base`, the env contract alongside it.
        let env_contract = res.env_contract;
        let base = res.base.ok_or_else(|| {
            GrpcError::from(ProtocolError::HandshakeFailed(
                "handshake response missing base".to_string(),
            ))
        })?;

        // `compatible` is the server's verdict on protocol generation (plain
        // equality — a wrong generation is a hard, full-restart break). Edition
        // selection happens later, at the runtime floor. The client trusts it;
        // there is no echoed server generation.
        if !base.compatible {
            return Err(ProtocolError::HandshakeFailed(handshake_rejection_reason(
                base.error_message,
            ))
            .into());
        }

        let env_contract = env_contract.ok_or_else(|| {
            GrpcError::from(ProtocolError::HandshakeFailed(
                "no env_contract in response".to_string(),
            ))
        })?;
        validate_env_contract(&env_contract)?;
        let num_envs = usize::try_from(env_contract.num_envs)
            .unwrap_or(usize::MAX)
            .max(1);
        // The handshake declares supported editions and gates only generation; the
        // runtime (this client) is the edition decider, picking the mutual with the
        // env (`env ∩ self` — the co-located floor for the in-process/run_local
        // path). No mutual edition fails here with an all-tiers diagnostic, rather
        // than yielding an empty string that trips the runtime spec validate later.
        let workflow_edition = negotiate_workflow_edition(&base.supported_workflow_editions)
            .ok_or_else(|| {
                ProtocolError::HandshakeFailed(format!(
                    "no mutual workflow edition with the env: env offered [{}], this runtime \
                     supports [{}]",
                    base.supported_workflow_editions.join(", "),
                    supported_workflow_editions().join(", ")
                ))
            })?
            .to_string();
        let handshake = EnvHandshake {
            env_contract,
            num_envs,
            workflow_edition,
            supported_workflow_editions: base.supported_workflow_editions,
            capabilities: base.capabilities,
        };
        self.shared.set_state(ClientState::Ready);

        Ok(handshake)
    }

    async fn send_handshake(&mut self) -> Result<HandshakeResponse, GrpcError> {
        let req = HandshakeRequest {
            base: Some(rlmesh_proto::core_handshake_request("rlmesh-env", &[])),
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
        self.last_endpoint_total_ns = res.endpoint_total_ns;
        self.last_phases = EndpointPhases::from_env_response(&res);

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
        self.last_endpoint_total_ns = res.endpoint_total_ns;
        self.last_phases = EndpointPhases::from_env_response(&res);

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
        self.last_endpoint_total_ns = res.endpoint_total_ns;
        self.last_phases = EndpointPhases::from_env_response(&res);

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

    /// Close this client's session on the server and tear down the local Join
    /// stream.
    ///
    /// This ends the **session**, not the **server**: the served environment
    /// detaches the session and remains available for a subsequent client to
    /// connect and run a new session. It does not stop the server process; use
    /// [`EnvClient::shutdown`] or the server's idle/drain policy for that.
    pub async fn close(&mut self) -> Result<CloseEnvsResponse, GrpcError> {
        self.ensure_ready()?;

        // A client that never opened the Join stream holds none of the server's
        // exclusive session slot, so there is nothing to close remotely. Opening
        // a fresh Join here just to close it would race any *other* client's
        // active session and earn a FailedPrecondition from the server's
        // join_active CAS, exactly the lockout the lazy Join stream exists to
        // avoid (see `ensure_join_stream`). Short-circuit to a local-only close.
        if self.shared.stream().is_none() {
            self.close_local();
            return Ok(CloseEnvsResponse::default());
        }

        let env_req = JoinRequest {
            kind: Some(join_request::Kind::Close(
                rlmesh_proto::env::v1::CloseEnvsRequest {
                    reason: "client close".to_string(),
                },
            )),
            request_id: self.next_request_id(),
        };

        let res = self.send_on_stream(env_req).await?;
        self.last_endpoint_total_ns = res.endpoint_total_ns;
        self.last_phases = EndpointPhases::from_env_response(&res);
        self.close_local();

        match res.kind {
            Some(join_response::Kind::Close(ok)) => Ok(ok),
            Some(join_response::Kind::Error(e)) => Err(proto_error_to_env_error(e).into()),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "CloseEnvsResponse".to_string(),
                actual: format!("{:?}", res.kind),
            }
            .into()),
        }
    }

    /// Request owner-level shutdown of the remote environment endpoint.
    pub async fn shutdown(
        &mut self,
        reason: impl Into<String>,
    ) -> Result<CoreShutdownResponse, GrpcError> {
        if self.state() == ClientState::Closed {
            return Err(ClientError::NotConnected.into());
        }

        let response = self
            .client
            .shutdown(self.authorized_request(ShutdownRequest {
                base: Some(CoreShutdownRequest {
                    reason: reason.into(),
                }),
            })?)
            .await
            .map_err(crate::error::status_to_grpc_error)?
            .into_inner()
            .base
            .ok_or_else(|| {
                GrpcError::from(ProtocolError::HandshakeFailed(
                    "shutdown response missing base".to_string(),
                ))
            })?;

        if response.accepted {
            self.close_local();
        }

        Ok(response)
    }

    /// Tear down the local session state without a Close round-trip.
    ///
    /// Dropping the Join stream releases the server's exclusive session slot
    /// once the server observes the stream end. If an operation is still
    /// draining server-side, the slot frees only after it completes, so an
    /// immediate reconnect can still be rejected briefly. The server completes
    /// this session's in-flight episodes as truncated; their metadata is not
    /// returned to this client. Use when a graceful [`EnvClient::close`] is
    /// not possible (e.g. it timed out behind a long-draining operation).
    pub fn detach(&mut self) {
        self.close_local();
    }

    fn close_local(&mut self) {
        // Dropping the request sender ends the Join stream; the pump then
        // closes the pending map, waking any in-flight waiter on other clones.
        self.shared.set_stream(None);
        self.shared.set_state(ClientState::Closed);
    }

    /// Open the Join stream on first use. The stream is the env's exclusive
    /// session slot (the server admits one Join at a time), so it is acquired
    /// lazily on the first streaming operation rather than at handshake;
    /// an idle connected client must not lock other clients out of the env.
    async fn ensure_join_stream(&mut self) -> Result<(), GrpcError> {
        if self.shared.stream().is_some() {
            return Ok(());
        }
        let _opening = self.shared.open_lock.lock().await;
        if self.shared.stream().is_some() {
            return Ok(());
        }
        let (tx, rx) = mpsc::channel::<JoinRequest>(32);
        let request_stream = ReceiverStream::new(rx);

        let response = self
            .client
            .join(self.authorized_request(request_stream)?)
            .await
            .map_err(crate::error::status_to_grpc_error)?;

        let pending = new_pending();
        spawn_response_pump(response.into_inner(), pending.clone());
        self.shared.set_stream(Some(JoinStream { tx, pending }));
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

    /// Send one request on the Join stream and await its reply. Other requests
    /// (from this or any clone) may be in flight at the same time; replies are
    /// routed by `request_id`, so they can arrive in any order.
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
        let stream = self.shared.stream().ok_or(ClientError::NotHandshaked)?;

        let (reply_tx, reply_rx) = oneshot::channel();
        {
            let mut pending = stream
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let map = pending.as_mut().ok_or_else(|| {
                tracing::error!(
                    request_id = %request_id,
                    request_kind,
                    "env join stream already ended; cannot send request"
                );
                GrpcError::from(TransportError::ConnectionClosed)
            })?;
            map.insert(request_id.clone(), reply_tx);
        }

        if stream.tx.send(req).await.is_err() {
            tracing::error!(
                request_id = %request_id,
                request_kind,
                "failed to send request because the env join stream is closed"
            );
            if let Some(map) = stream
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_mut()
            {
                map.remove(&request_id);
            }
            return Err(TransportError::ConnectionClosed.into());
        }

        match reply_rx.await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(status)) => {
                tracing::error!(
                    request_id = %request_id,
                    request_kind,
                    code = ?status.code(),
                    message = %status.message(),
                    "env join stream returned an error status"
                );
                Err(super::wire::status_to_grpc_error(status))
            }
            Err(_) => {
                tracing::error!(
                    request_id = %request_id,
                    request_kind,
                    "env join stream closed while waiting for response"
                );
                Err(TransportError::ConnectionClosed.into())
            }
        }
    }

    fn ensure_ready(&self) -> Result<(), GrpcError> {
        match self.state() {
            ClientState::Ready => Ok(()),
            ClientState::Connected => Err(ClientError::NotHandshaked.into()),
            ClientState::Closed => Err(ClientError::NotConnected.into()),
        }
    }

    fn next_request_id(&mut self) -> String {
        let n = self.shared.request_counter.fetch_add(1, Ordering::AcqRel) + 1;
        format!("grpc-req-{n}")
    }
}

fn validate_env_contract(env_contract: &EnvContract) -> Result<(), GrpcError> {
    let spec = env_contract.spec.as_ref().ok_or_else(|| {
        GrpcError::from(ProtocolError::HandshakeFailed(
            "env_contract missing spec".to_string(),
        ))
    })?;
    if spec.observation_space.is_none() {
        return Err(ProtocolError::HandshakeFailed(
            "env_contract missing observation_space".to_string(),
        )
        .into());
    }
    if spec.action_space.is_none() {
        return Err(ProtocolError::HandshakeFailed(
            "env_contract missing action_space".to_string(),
        )
        .into());
    }
    Ok(())
}

/// The reason to surface for a peer that answered `compatible = false`.
///
/// `error_message` is optional on the wire, so a peer that rejects the
/// handshake without filling it in would otherwise surface as a handshake
/// failure with an empty reason. The fallback names what this build speaks, so
/// the user can compare it against the peer.
fn handshake_rejection_reason(error_message: Option<String>) -> String {
    error_message
        .filter(|message| !message.trim().is_empty())
        .unwrap_or_else(|| {
            format!(
                "peer rejected the handshake without a reason; this build speaks protocol \
                 generation {} and workflow editions [{}]",
                rlmesh_proto::PROTOCOL_GENERATION,
                rlmesh_proto::supported_workflow_editions().join(", ")
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_proto::core::v1::{
        EnvSpec, HandshakeResponse as CoreHandshakeResponse,
        ShutdownResponse as CoreShutdownResponse,
    };
    use rlmesh_proto::env::v1::env_service_server::{EnvService, EnvServiceServer};
    use rlmesh_proto::env::v1::{ShutdownResponse, StepResponse};
    use rlmesh_proto::spaces::v1::SpaceSpec;
    use rlmesh_proto::supported_workflow_editions;
    use tokio::sync::oneshot;
    use tokio_stream::wrappers::ReceiverStream;
    use tonic::transport::Endpoint;
    use tonic::{Request, Response, Status};

    use rlmesh_spaces::{EnvContract as SpaceEnvContract, SpaceSpec as NativeSpaceSpec};

    /// A no-op single-lane env used by the integration-style server tests below.
    struct PlainEnv {
        contract: SpaceEnvContract,
    }

    impl PlainEnv {
        fn new(id: &str) -> Self {
            let space = NativeSpaceSpec::default();
            Self {
                contract: SpaceEnvContract {
                    id: id.to_string(),
                    autoreset_mode: Default::default(),
                    action_space: Some(space.clone()),
                    observation_space: Some(space),
                    metadata: None,
                    render_mode: String::new(),
                    num_envs: 1,
                },
            }
        }
    }

    #[async_trait::async_trait]
    impl crate::env::Environment for PlainEnv {
        fn observation_space(&self) -> &NativeSpaceSpec {
            self.contract.observation_space.as_ref().unwrap()
        }
        fn action_space(&self) -> &NativeSpaceSpec {
            self.contract.action_space.as_ref().unwrap()
        }
        fn num_envs(&self) -> usize {
            1
        }
        fn env_contract(&self) -> &SpaceEnvContract {
            &self.contract
        }
        async fn reset(
            &self,
            _req: ResetRequest,
        ) -> std::result::Result<(ResetResponse, EndpointPhases), crate::error::EnvError> {
            Ok((ResetResponse::default(), EndpointPhases::default()))
        }
        async fn step(
            &self,
            _req: StepRequest,
        ) -> std::result::Result<(StepResponse, EndpointPhases), crate::error::EnvError> {
            Ok((StepResponse::default(), EndpointPhases::default()))
        }
        async fn render(
            &self,
            _req: RenderRequest,
        ) -> std::result::Result<(RenderResponse, EndpointPhases), crate::error::EnvError> {
            Ok((RenderResponse::default(), EndpointPhases::default()))
        }
        async fn close(&self) -> std::result::Result<CloseEnvsResponse, crate::error::EnvError> {
            Ok(CloseEnvsResponse::default())
        }
    }

    #[test]
    fn validate_env_contract_requires_spaces() {
        let valid = EnvContract {
            spec: Some(EnvSpec {
                observation_space: Some(SpaceSpec::default()),
                action_space: Some(SpaceSpec::default()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(validate_env_contract(&valid).is_ok());

        let missing_observation = EnvContract {
            spec: Some(EnvSpec {
                action_space: Some(SpaceSpec::default()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let err = validate_env_contract(&missing_observation).unwrap_err();
        assert!(err.to_string().contains("missing observation_space"));

        let missing_action = EnvContract {
            spec: Some(EnvSpec {
                observation_space: Some(SpaceSpec::default()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let err = validate_env_contract(&missing_action).unwrap_err();
        assert!(err.to_string().contains("missing action_space"));
    }

    /// A Ready client over a fake Join stream: requests land on `request_rx`,
    /// replies are injected with `dispatch_response` on the returned pending map.
    fn ready_client_with_stream() -> (EnvClient, mpsc::Receiver<JoinRequest>, Pending) {
        let (request_tx, request_rx) = mpsc::channel(4);
        let pending = new_pending();
        let channel = Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        let client = EnvClient {
            client: EnvServiceClient::new(channel),
            token: String::new(),
            address: "tcp://127.0.0.1:1".to_string(),
            shared: Arc::new(Shared::new(
                ClientState::Ready,
                Some(JoinStream {
                    tx: request_tx,
                    pending: pending.clone(),
                }),
            )),
            last_endpoint_total_ns: None,
            last_phases: EndpointPhases::default(),
        };
        (client, request_rx, pending)
    }

    #[tokio::test]
    async fn send_on_stream_routes_replies_by_request_id_in_any_order() {
        let (client, mut request_rx, pending) = ready_client_with_stream();

        // Two clones (lanes) with requests in flight at once; the replies come
        // back in the opposite order and each lands on its own waiter.
        let mut lane_a = client.clone();
        let mut lane_b = client;
        let a = tokio::spawn(async move { lane_a.step(StepRequest::default()).await });
        let b = tokio::spawn(async move { lane_b.step(StepRequest::default()).await });
        let first = request_rx.recv().await.unwrap();
        let second = request_rx.recv().await.unwrap();
        assert_ne!(first.request_id, second.request_id);

        // A reply nobody asked for is dropped, not misdelivered.
        dispatch_response(
            &pending,
            Some(Ok(JoinResponse {
                request_id: "abandoned".to_string(),
                kind: Some(join_response::Kind::Step(StepResponse::default())),
                ..Default::default()
            })),
        );
        for id in [second.request_id, first.request_id] {
            dispatch_response(
                &pending,
                Some(Ok(JoinResponse {
                    request_id: id,
                    kind: Some(join_response::Kind::Step(StepResponse {
                        rewards: vec![1.0],
                        ..Default::default()
                    })),
                    ..Default::default()
                })),
            );
        }
        assert_eq!(a.await.unwrap().unwrap().rewards, vec![1.0]);
        assert_eq!(b.await.unwrap().unwrap().rewards, vec![1.0]);
    }

    #[tokio::test]
    async fn send_on_stream_surfaces_pump_status_error_to_caller() {
        let (mut client, mut request_rx, pending) = ready_client_with_stream();

        let step = tokio::spawn(async move { client.step(StepRequest::default()).await });
        let _ = request_rx.recv().await.unwrap();
        // The response pump propagates a transport Status (e.g. a response that
        // exceeded the decode limit) instead of just dropping it. The pending
        // caller must observe that status, not an opaque "connection closed".
        dispatch_response(
            &pending,
            Some(Err(tonic::Status::new(
                tonic::Code::ResourceExhausted,
                "message length too large",
            ))),
        );

        let error = step
            .await
            .unwrap()
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
    async fn stream_end_wakes_waiters_with_connection_closed() {
        let (mut client, mut request_rx, pending) = ready_client_with_stream();
        let step = tokio::spawn(async move { client.step(StepRequest::default()).await });
        let _ = request_rx.recv().await.unwrap();
        dispatch_response(&pending, None);
        assert!(matches!(
            step.await.unwrap().unwrap_err(),
            GrpcError::Transport(TransportError::ConnectionClosed)
        ));
    }

    #[tokio::test]
    async fn close_sends_remote_close_then_closes_locally() {
        let (mut client, mut request_rx, pending) = ready_client_with_stream();
        let other = client.clone();

        let close = tokio::spawn(async move { client.close().await });
        let request = request_rx.recv().await.unwrap();
        assert!(matches!(request.kind, Some(join_request::Kind::Close(_))));
        assert_eq!(request.request_id, "grpc-req-1");
        dispatch_response(
            &pending,
            Some(Ok(JoinResponse {
                request_id: "grpc-req-1".to_string(),
                kind: Some(join_response::Kind::Close(CloseEnvsResponse::default())),
                ..Default::default()
            })),
        );

        let response = close.await.unwrap().unwrap();
        assert!(response.final_episodes.is_empty());
        // Closing one handle closes the session for every clone.
        assert_eq!(other.state(), ClientState::Closed);
        assert!(other.shared.stream().is_none());
    }

    #[tokio::test]
    async fn close_on_never_used_client_is_local_only_and_opens_no_join() {
        // A client that handshook but never ran an operation has no Join stream.
        // close() must not open one just to tear it down; that would race any
        // other client's active session (server join_active CAS). It should
        // short-circuit to a local-only close.
        let channel = Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        let mut client = EnvClient {
            client: EnvServiceClient::new(channel),
            token: String::new(),
            address: "tcp://127.0.0.1:1".to_string(),
            shared: Arc::new(Shared::new(ClientState::Ready, None)),
            last_endpoint_total_ns: None,
            last_phases: EndpointPhases::default(),
        };

        let response = client.close().await.unwrap();

        assert!(response.final_episodes.is_empty());
        assert_eq!(client.state(), ClientState::Closed);
        // No Join stream was ever opened, and the request counter was not bumped
        // (no JoinRequest was minted).
        assert!(client.shared.stream().is_none());
        assert_eq!(client.shared.request_counter.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn idle_client_close_does_not_lock_out_an_active_session() {
        use crate::env::server::GrpcEnvServer;
        use crate::lifecycle::{ServeOptions, ShutdownTrigger};
        use rlmesh_proto::env::v1::env_service_server::EnvServiceServer;

        let service = EnvServiceServer::new(GrpcEnvServer::new_with_options(
            PlainEnv::new("plain-env"),
            ShutdownTrigger::new(),
            ServeOptions::default(),
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
        let connect_options =
            crate::connect::ConnectOptions::with_deadline(std::time::Duration::from_secs(5))
                .backoff(std::time::Duration::from_millis(10));

        // Client A holds the env's single Join session (opened lazily by reset).
        let mut client_a = EnvClient::connect_with_retry(&address, "", &connect_options)
            .await
            .expect("test server did not start");
        client_a.handshake().await.expect("handshake A");
        client_a
            .reset(ResetRequest::default())
            .await
            .expect("A reset");
        assert!(client_a.shared.stream().is_some());

        let mut client_b = EnvClient::connect_with_retry(&address, "", &connect_options)
            .await
            .expect("test server did not start");
        client_b.handshake().await.expect("handshake B");
        assert!(client_b.shared.stream().is_none());

        client_b
            .close()
            .await
            .expect("idle client close must not contend for the active Join slot");
        assert_eq!(client_b.state(), ClientState::Closed);
        assert!(client_b.shared.stream().is_none());

        // Client A's session is undisturbed and remains usable.
        client_a
            .step(StepRequest::default())
            .await
            .expect("A still usable after B close");
        client_a.close().await.expect("A graceful close");

        let _ = shutdown_tx.send(());
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), server).await;
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
                base: Some(CoreHandshakeResponse {
                    compatible: true,
                    supported_workflow_editions: supported_workflow_editions(),
                    ..Default::default()
                }),
                env_contract: Some(EnvContract {
                    spec: Some(EnvSpec {
                        observation_space: Some(SpaceSpec::default()),
                        action_space: Some(SpaceSpec::default()),
                        ..Default::default()
                    }),
                    num_envs: 1,
                    ..Default::default()
                }),
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
                base: Some(CoreShutdownResponse {
                    accepted: true,
                    ..Default::default()
                }),
            }))
        }
    }

    #[tokio::test]
    async fn connect_with_token_authenticates_against_token_server() {
        use crate::env::server::GrpcEnvServer;
        use crate::lifecycle::{ServeOptions, ShutdownTrigger};
        use rlmesh_proto::env::v1::env_service_server::EnvServiceServer;

        let options = ServeOptions {
            token: Some("s3cret".to_string()),
            ..Default::default()
        };
        let service = EnvServiceServer::new(GrpcEnvServer::new_with_options(
            PlainEnv::new("token-env"),
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
        assert!(client.shared.stream().is_none());

        // The join failure surfaces on the first operation and leaves the
        // client un-wedged (no half-open stream state).
        let error = client
            .reset(ResetRequest::default())
            .await
            .expect_err("join is unavailable");
        assert!(error.to_string().contains("join unavailable"));
        assert_eq!(client.state(), ClientState::Ready);
        assert!(client.shared.stream().is_none());

        let _ = shutdown_tx.send(());
        tokio::time::timeout(std::time::Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[test]
    fn handshake_rejection_reason_names_this_build_when_the_peer_gives_none() {
        assert_eq!(
            handshake_rejection_reason(Some("bad generation".into())),
            "bad generation"
        );
        let fallback = handshake_rejection_reason(None);
        assert_eq!(handshake_rejection_reason(Some("  ".into())), fallback);
        assert!(
            fallback.contains(rlmesh_proto::PROTOCOL_GENERATION),
            "{fallback}"
        );
        assert!(
            fallback.contains(rlmesh_proto::CURRENT_WORKFLOW_EDITION),
            "{fallback}"
        );
    }
}
