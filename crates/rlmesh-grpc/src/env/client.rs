//! Environment client transport implementation using Tonic.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use rlmesh_proto::core::v1::{
    EnvContract, ShutdownRequest as CoreShutdownRequest, ShutdownResponse as CoreShutdownResponse,
};
use rlmesh_proto::env::v1::{
    CloseEnvsResponse, ConfigureEnvRequest, HandshakeRequest, HandshakeResponse, JoinRequest,
    JoinResponse, RenderRequest, RenderResponse, ResetRequest, ResetResponse, ShutdownRequest,
    StepRequest, StepResponse, env_service_client::EnvServiceClient, join_request, join_response,
};
use rlmesh_proto::{
    Edition, EndpointPhases, SessionOffer, negotiate_workflow_edition, parse_retained_edition,
};

use crate::error::{
    ClientError, EnvError, EnvErrorCode, Error as GrpcError, ProtocolError, TransportError,
};
use crate::helpers::address::parse_env_connect_target;
use crate::states::ClientState;

#[cfg(test)]
use super::stream::dispatch_response;
use super::stream::{
    Pending, StreamCompletion, new_pending, retire_session, spawn_response_pump,
    wait_for_retiring_sessions,
};
use super::wire::{join_request_kind_name, proto_error_to_env_error};

tokio::task_local! {
    static REQUEST_DEADLINE: tokio::time::Instant;
}

/// Carry a facade's absolute call deadline into Join admission, so local
/// teardown waits share the caller's remaining budget and refusal semantics.
#[doc(hidden)]
pub async fn with_request_deadline<F: std::future::Future>(
    deadline: tokio::time::Instant,
    future: F,
) -> F::Output {
    REQUEST_DEADLINE.scope(deadline, future).await
}

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
    pub workflow_edition: Edition,
    /// The exact wire spelling that selection was made under (e.g.
    /// `2026.06-0.1.0-rc.12`), kept alongside the typed value because the two are
    /// not interchangeable on the wire: [`workflow_edition`](Self::workflow_edition)
    /// drives logic and displays its bare base, while an env accepts only the
    /// cohort spelling it offered. This is what goes back on
    /// `ConfigureEnvRequest.selected_workflow_edition`, and what telemetry names.
    pub selected_workflow_edition: String,
    /// CAN: the editions the env advertised, for the three-way session floor.
    pub supported_workflow_editions: Vec<String>,
    /// WANT: the edition the env declared, or `None` when it declared none (an
    /// empty wire value — every peer built before the field existed). Negotiation
    /// reads an undeclared WANT as `max(supported_workflow_editions)`.
    pub preferred_workflow_edition: Option<String>,
    /// Optional features the server advertised (advisory; query with
    /// [`rlmesh_proto::has_capability`]).
    pub capabilities: std::collections::HashMap<String, String>,
}

impl EnvHandshake {
    /// The env's bind-time offer: the editions it CAN drive plus the one it
    /// declared (WANT). Generation is gated by equality at the handshake (carried
    /// by `base.compatible`), so it is not part of the offer. Capabilities are read
    /// pairwise from [`capabilities`](Self::capabilities), not negotiated. Feeds
    /// [`rlmesh_proto::negotiate_session_floor`] as the env's offer.
    pub fn session_offer(&self) -> SessionOffer {
        SessionOffer {
            editions: self.supported_workflow_editions.clone(),
            preferred: self.preferred_workflow_edition.clone(),
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
    /// Unlike edition observers, only client clones keep the session alive.
    _lifetime: Arc<ClientLifetime>,
    /// Endpoint-local op duration (ns) attached to the last Join response. The
    /// nested per-step telemetry message was replaced by this hot scalar
    /// (`JoinResponse.endpoint_total_ns`). Per handle: each lane reads its own.
    last_endpoint_total_ns: Option<u64>,
    /// The peer's split of that duration, cleared by the read.
    last_phases: EndpointPhases,
    /// WANT: the workflow edition the RUNTIME declares on this leg, set by
    /// [`declare_workflow_edition`](Self::declare_workflow_edition) before
    /// [`handshake`](Self::handshake). `None` declares this build's current
    /// edition, which is its `max(can)` and therefore caps no env.
    declared_workflow_edition: Option<String>,
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
    /// The exact edition spelling the Join stream opens with (its first message,
    /// `ConfigureEnvRequest`): the env-leg selection from the handshake, or the
    /// session floor a model leg lowered it to. Empty until the handshake.
    pinned_workflow_edition: std::sync::Mutex<String>,
    /// The message of the non-recoverable env error that ended the last Join
    /// session, until a reset opens a fresh one. The server closes the stream
    /// after such an error, so later ops on it could only report a closed
    /// connection; this names the real cause instead.
    ended_by: std::sync::Mutex<Option<String>>,
}

/// A shared read handle on the edition an [`EnvClient`]'s Join opens with
/// (the handshake selection, or the last pin).
#[derive(Clone)]
pub struct PinnedEdition(Arc<Shared>);

impl PinnedEdition {
    /// The current pinned edition spelling.
    pub fn get(&self) -> String {
        self.0.pinned_workflow_edition()
    }
}

impl std::fmt::Debug for PinnedEdition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PinnedEdition").field(&self.get()).finish()
    }
}

/// The Join bidi stream: where requests go, and who is waiting for a reply.
#[derive(Clone)]
struct JoinStream {
    tx: mpsc::Sender<JoinRequest>,
    pending: Pending,
    retirement: Arc<JoinRetirement>,
}

struct JoinRetirement {
    finished: StreamCompletion,
    address: String,
    retiring: AtomicBool,
}

impl JoinStream {
    fn retire(&self) {
        if !self.retirement.retiring.swap(true, Ordering::AcqRel) {
            retire_session(&self.retirement.address, self.retirement.finished.clone());
        }
    }
}

/// Closing is terminal even when the caller drops the future or sending fails.
struct CloseGuard(Arc<Shared>);

impl Drop for CloseGuard {
    fn drop(&mut self) {
        self.0.set_stream(None);
        self.0.set_state(ClientState::Closed);
    }
}

struct ClientLifetime(Arc<Shared>);

impl Drop for ClientLifetime {
    fn drop(&mut self) {
        // The final client half-closes even if an edition observer survives.
        // set_stream registers the EOF barrier before dropping the sender.
        self.0.set_stream(None);
        self.0.set_state(ClientState::Closed);
    }
}

impl Shared {
    fn new(state: ClientState, stream: Option<JoinStream>) -> Self {
        Self {
            state: AtomicU8::new(state as u8),
            stream: std::sync::Mutex::new(stream),
            open_lock: tokio::sync::Mutex::new(()),
            request_counter: AtomicU64::new(0),
            pinned_workflow_edition: std::sync::Mutex::new(String::new()),
            ended_by: std::sync::Mutex::new(None),
        }
    }

    fn pinned_workflow_edition(&self) -> String {
        self.pinned_workflow_edition
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn set_pinned_workflow_edition(&self, edition: String) {
        *self
            .pinned_workflow_edition
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = edition;
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
        let mut slot = self
            .stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(previous) = slot.as_ref() {
            previous.retire();
        }
        *slot = stream;
    }

    fn ended_by(&self) -> Option<String> {
        self.ended_by
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn set_ended_by(&self, message: Option<String>) {
        *self
            .ended_by
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = message;
    }

    /// Drop the Join stream whose waiters are `pending` after a
    /// non-recoverable error ended it, unless it is no longer this session's
    /// stream: a stale stream's pump must not end a newer session.
    fn end_session(&self, pending: &Pending, message: String) {
        let mut slot = self
            .stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(current) = slot.as_ref()
            && Arc::ptr_eq(&current.pending, pending)
        {
            current.retire();
            *slot = None;
            self.set_ended_by(Some(message));
        }
    }

    /// The error for an op on a Join stream that has gone away: the fatal env
    /// error that ended the session, when that is why, else a closed connection.
    fn stream_gone_error(&self) -> GrpcError {
        match self.ended_by() {
            Some(message) => session_ended_error(&message),
            None => TransportError::ConnectionClosed.into(),
        }
    }
}

fn session_ended_error(message: &str) -> GrpcError {
    EnvError {
        code: EnvErrorCode::Closed,
        message: format!(
            "env session ended after a non-recoverable env error: {message}; reset to start a \
             new session"
        ),
        is_recoverable: false,
        debug_info: None,
    }
    .into()
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
        Self::connect_with(addr, &crate::Credentials::token(token)).await
    }

    /// Connect to an EnvService server with `credentials`: the token every
    /// request carries and the TLS trust of an `https://` address.
    pub async fn connect_with(
        addr: &str,
        credentials: &crate::Credentials,
    ) -> Result<Self, GrpcError> {
        let target = parse_env_connect_target(addr)?;
        let channel = crate::dial::dial(&target, &credentials.trust).await?;
        let token = credentials.token.as_str();

        let shared = Arc::new(Shared::new(ClientState::Connected, None));
        Ok(Self {
            client: crate::ACCEPTED_COMPRESSION.into_iter().fold(
                EnvServiceClient::new(channel)
                    .max_decoding_message_size(crate::MAX_MESSAGE_SIZE)
                    .max_encoding_message_size(crate::MAX_MESSAGE_SIZE),
                |client, encoding| client.accept_compressed(encoding),
            ),
            address: target.display_address().to_string(),
            token: token.to_string(),
            shared: shared.clone(),
            _lifetime: Arc::new(ClientLifetime(shared)),
            last_endpoint_total_ns: None,
            last_phases: EndpointPhases::default(),
            declared_workflow_edition: None,
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

    /// Declare the workflow edition the runtime brings to this leg (its WANT),
    /// or `None` to declare nothing. Must be set before
    /// [`handshake`](Self::handshake): it is what the request carries and what
    /// the env-leg negotiation caps the session at. Any cohort spelling of a
    /// retained base is accepted; an edition this build cannot drive is refused
    /// by the negotiation, naming both sets.
    pub fn declare_workflow_edition(&mut self, declared: Option<String>) {
        self.declared_workflow_edition = declared
            .map(|edition| edition.trim().to_string())
            .filter(|edition| !edition.is_empty());
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
        let env_offer = SessionOffer {
            editions: base.supported_workflow_editions,
            preferred: Some(base.preferred_workflow_edition.trim().to_string())
                .filter(|want| !want.is_empty()),
        };
        let runtime_offer = SessionOffer::this_build(self.declared_workflow_edition.as_deref());
        let selected =
            negotiate_workflow_edition(&env_offer, &runtime_offer).map_err(|refusal| {
                ProtocolError::HandshakeFailed(format!(
                    "no mutual workflow edition with the env: {refusal}{}",
                    refusal.hint()
                ))
            })?;
        // The wire carries a name; the session carries the typed edition. This is
        // the runtime's one parse of an env-leg edition, and it goes through the
        // same retained-list boundary every other caller uses — `selected` came
        // from this build's own retained list, so a failure here means that list
        // names an edition this build cannot actually drive.
        let workflow_edition =
            parse_retained_edition(&selected).map_err(ProtocolError::HandshakeFailed)?;
        self.shared.set_pinned_workflow_edition(selected.clone());
        let handshake = EnvHandshake {
            env_contract,
            num_envs,
            workflow_edition,
            selected_workflow_edition: selected,
            supported_workflow_editions: env_offer.editions,
            preferred_workflow_edition: env_offer.preferred,
            capabilities: base.capabilities,
        };
        self.shared.set_state(ClientState::Ready);

        Ok(handshake)
    }

    /// Pin the env to the session edition: the exact spelling the Join stream
    /// opens with, sent as its first message and acked before the first reset.
    ///
    /// The handshake sets it to the env-leg selection
    /// ([`EnvHandshake::selected_workflow_edition`]); a session with a model leg
    /// calls this with the three-way floor ([`crate::env_floor`]) before its
    /// first operation, so both legs are pinned to the same value. A pin the env
    /// cannot run is refused when the stream opens, naming the pin and the env's
    /// editions, and no reset is sent.
    pub fn pin_workflow_edition(&mut self, selected_workflow_edition: String) {
        self.shared
            .set_pinned_workflow_edition(selected_workflow_edition.trim().to_string());
    }

    /// A handle on the edition this session's Join opens with, shared by
    /// every clone: it reads each pin as soon as it is made.
    pub fn pinned_edition(&self) -> PinnedEdition {
        PinnedEdition(Arc::clone(&self.shared))
    }

    async fn send_handshake(&mut self) -> Result<HandshakeResponse, GrpcError> {
        let req = HandshakeRequest {
            base: Some(rlmesh_proto::core_handshake_request(
                "rlmesh-env",
                &[],
                self.declared_workflow_edition.as_deref(),
            )),
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
    ///
    /// After a non-recoverable env error ended the session, this opens a fresh
    /// Join session, so the env is usable again.
    pub async fn reset(&mut self, req: ResetRequest) -> Result<ResetResponse, GrpcError> {
        self.ensure_ready()?;
        self.ensure_join_stream(req.timeout_ms).await?;

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
        self.ensure_session_not_ended()?;
        self.ensure_join_stream(req.timeout_ms).await?;

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
        self.ensure_session_not_ended()?;
        self.ensure_join_stream(req.timeout_ms).await?;

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
    /// stream, waiting for its terminal status so the server has released its
    /// exclusive Join slot before another client connects.
    ///
    /// This ends the **session**, not the **server**: the served environment
    /// detaches the session and remains available for a subsequent client to
    /// connect and run a new session. It does not stop the server process; use
    /// [`EnvClient::shutdown`] or the server's idle/drain policy for that.
    pub async fn close(&mut self) -> Result<CloseEnvsResponse, GrpcError> {
        self.ensure_ready()?;
        let _close_guard = CloseGuard(self.shared.clone());
        self.prepare_close();

        // A client that never opened the Join stream holds none of the server's
        // exclusive session slot, so there is nothing to close remotely. Opening
        // a fresh Join here just to close it would race any *other* client's
        // active session and earn a FailedPrecondition from the server's
        // join_active CAS, exactly the lockout the lazy Join stream exists to
        // avoid (see `ensure_join_stream`). Short-circuit to a local-only close.
        let Some(mut finished) = self
            .shared
            .stream()
            .map(|stream| stream.retirement.finished.clone())
        else {
            self.close_local();
            return Ok(CloseEnvsResponse::default());
        };

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

        // A Close reply acknowledges the request, but older servers may still
        // be draining the session's tracker afterwards. Half-close our request
        // stream and wait for the response stream's terminal status as well.
        // The Close reply confirms success even if the final transport status
        // is lost or non-OK. Still await termination to support older servers.
        let _ = finished.wait_for(|outcome| outcome.is_some()).await;

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
    /// draining server-side, the slot frees only after it completes. A later
    /// local Join waits up to five seconds for this stream to finish; an
    /// operation still draining after that can still cause a refusal. The
    /// server completes this session's in-flight episodes as truncated; their
    /// metadata is not returned to this client. Use when a graceful
    /// [`EnvClient::close`] is not possible (e.g. it timed out behind a
    /// long-draining operation).
    pub fn detach(&mut self) {
        self.close_local();
    }

    /// Order later local clients behind this session's pending teardown.
    /// The owner must subsequently call `close` or `detach`; this hook lets
    /// facade Drop implementations register before spawning async cleanup.
    #[doc(hidden)]
    pub fn prepare_close(&self) {
        if let Some(stream) = self.shared.stream() {
            stream.retire();
        }
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
    /// Its first message pins the env to the session edition; the stream is
    /// not handed out until the env acks that, and a refusal releases it.
    async fn ensure_join_stream(&mut self, timeout_ms: u64) -> Result<(), GrpcError> {
        let deadline = REQUEST_DEADLINE
            .try_with(|deadline| *deadline)
            .ok()
            .or_else(|| {
                (timeout_ms > 0).then(|| {
                    tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms)
                })
            });
        if self.shared.stream().is_some() {
            return Ok(());
        }
        let shared = Arc::clone(&self.shared);
        let _opening = shared.open_lock.lock().await;
        if self.shared.stream().is_some() {
            return Ok(());
        }
        let (tx, response) = self.open_join(deadline).await?;

        let pending = new_pending();
        // The pump, not the request future, records a fatal reply: a caller
        // that dropped its future before the reply arrived must still find the
        // session ended, so its next reset opens a fresh one.
        let session = Arc::downgrade(&self.shared);
        let stream_pending = pending.clone();
        let finished =
            spawn_response_pump(response.into_inner(), pending.clone(), move |message| {
                if let Some(shared) = session.upgrade() {
                    shared.end_session(&stream_pending, message);
                }
            });
        self.shared.set_stream(Some(JoinStream {
            tx,
            pending,
            retirement: Arc::new(JoinRetirement {
                finished,
                address: self.address.clone(),
                retiring: AtomicBool::new(false),
            }),
        }));
        if let Err(error) = self.configure_session_edition().await {
            self.shared.set_stream(None);
            return Err(error);
        }
        self.shared.set_ended_by(None);
        Ok(())
    }

    /// Open a Join after any locally retired session finishes. Never retry a
    /// refusal: a genuinely concurrent client still owns the exclusive slot.
    async fn open_join(
        &self,
        deadline: Option<tokio::time::Instant>,
    ) -> Result<
        (
            mpsc::Sender<JoinRequest>,
            tonic::Response<tonic::Streaming<JoinResponse>>,
        ),
        GrpcError,
    > {
        if !wait_for_retiring_sessions(&self.address, deadline).await {
            return Err(crate::error::status_to_grpc_error(
                super::wire::active_join_status(),
            ));
        }
        let (tx, rx) = mpsc::channel::<JoinRequest>(32);
        let response = self
            .client
            .clone()
            .join(self.authorized_request(ReceiverStream::new(rx))?)
            .await
            .map_err(crate::error::status_to_grpc_error)?;
        Ok((tx, response))
    }

    /// Refuse a step/render once a non-recoverable error ended the session:
    /// only a reset starts a new one.
    fn ensure_session_not_ended(&self) -> Result<(), GrpcError> {
        if self.shared.stream().is_none()
            && let Some(message) = self.shared.ended_by()
        {
            return Err(session_ended_error(&message));
        }
        Ok(())
    }

    async fn configure_session_edition(&mut self) -> Result<(), GrpcError> {
        let selected_workflow_edition = self.shared.pinned_workflow_edition();
        if selected_workflow_edition.is_empty() {
            return Ok(());
        }
        let env_req = JoinRequest {
            kind: Some(join_request::Kind::Configure(ConfigureEnvRequest {
                selected_workflow_edition: selected_workflow_edition.clone(),
            })),
            request_id: self.next_request_id(),
        };
        let res = self.send_on_stream(env_req).await?;
        match res.kind {
            Some(join_response::Kind::Configure(_)) => {
                tracing::debug!(
                    selected_workflow_edition = %selected_workflow_edition,
                    "env pinned to session edition"
                );
                Ok(())
            }
            Some(join_response::Kind::Error(e)) => Err(proto_error_to_env_error(e).into()),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "ConfigureEnvResponse".to_string(),
                actual: format!("{:?}", res.kind),
            }
            .into()),
        }
    }

    /// Wrap a message in a `tonic::Request`, attaching the `authorization`
    /// metadata header when a token is configured.
    fn authorized_request<T>(&self, message: T) -> Result<tonic::Request<T>, GrpcError> {
        let mut request = tonic::Request::new(message);
        crate::helpers::auth::authorize(&mut request, &self.token)?;
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
                self.shared.stream_gone_error()
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
            return Err(self.shared.stream_gone_error());
        }

        match reply_rx.await {
            // A non-recoverable error reply already ended the session in the
            // response pump.
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
                Err(self.shared.stream_gone_error())
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
    use rlmesh_proto::env::v1::{ConfigureEnvResponse, ShutdownResponse, StepResponse};
    use rlmesh_proto::spaces::v1::SpaceSpec;
    use rlmesh_proto::supported_workflow_editions;
    use tokio::sync::oneshot;
    use tokio_stream::wrappers::ReceiverStream;
    use tonic::transport::Endpoint;
    use tonic::{Request, Response, Status};

    use rlmesh_spaces::{EnvContract as SpaceEnvContract, SpaceSpec as NativeSpaceSpec};

    /// A no-op single-lane env used by the integration-style server tests below.
    /// It records the edition pins and resets it receives, in order.
    struct PlainEnv {
        contract: SpaceEnvContract,
        events: Arc<std::sync::Mutex<Vec<String>>>,
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
                events: Arc::default(),
            }
        }

        fn record(&self, event: String) {
            self.events.lock().unwrap().push(event);
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
        fn pin_workflow_edition(&self, edition: Edition) {
            self.record(format!("pin:{edition}"));
        }
        async fn reset(
            &self,
            _req: ResetRequest,
        ) -> std::result::Result<(ResetResponse, EndpointPhases), crate::error::EnvError> {
            self.record("reset".to_string());
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
        let shared = Arc::new(Shared::new(
            ClientState::Ready,
            Some(JoinStream {
                tx: request_tx,
                pending: pending.clone(),
                retirement: Arc::new(JoinRetirement {
                    finished: tokio::sync::watch::channel(Some(Ok(()))).1,
                    address: "tcp://127.0.0.1:1".to_string(),
                    retiring: AtomicBool::new(false),
                }),
            }),
        ));
        let client = EnvClient {
            client: EnvServiceClient::new(channel),
            token: String::new(),
            address: "tcp://127.0.0.1:1".to_string(),
            shared: shared.clone(),
            _lifetime: Arc::new(ClientLifetime(shared)),
            last_endpoint_total_ns: None,
            last_phases: EndpointPhases::default(),
            declared_workflow_edition: None,
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
    async fn a_stale_streams_fatal_error_does_not_end_a_newer_session() {
        let (client, _request_rx, current) = ready_client_with_stream();

        // A pump still draining an older stream reports its fatal error late.
        client
            .shared
            .end_session(&new_pending(), "old session".to_string());
        assert!(
            client.shared.stream().is_some(),
            "the newer session survives"
        );
        assert!(client.shared.ended_by().is_none());

        client
            .shared
            .end_session(&current, "this session".to_string());
        assert!(client.shared.stream().is_none());
        assert_eq!(client.shared.ended_by().as_deref(), Some("this session"));
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
    async fn close_send_error_detaches_all_clones() {
        let (mut client, request_rx, _) = ready_client_with_stream();
        let other = client.clone();
        drop(request_rx);
        assert!(client.close().await.is_err());
        assert_eq!(other.state(), ClientState::Closed);
        assert!(other.shared.stream().is_none());
    }

    #[tokio::test]
    async fn close_reply_error_detaches_all_clones() {
        let (mut client, mut request_rx, pending) = ready_client_with_stream();
        let mut closing = Box::pin(client.close());
        assert!(futures::poll!(closing.as_mut()).is_pending());
        request_rx.recv().await.unwrap();
        dispatch_response(&pending, Some(Err(Status::unavailable("reply lost"))));
        assert!(closing.await.is_err());
        assert_eq!(client.state(), ClientState::Closed);
        assert!(client.shared.stream().is_none());
    }

    #[tokio::test]
    async fn close_cancellation_detaches_while_sending_or_awaiting_reply() {
        for full_channel in [false, true] {
            let (mut client, mut request_rx, _) = ready_client_with_stream();
            let other = client.clone();
            if full_channel {
                let stream = client.shared.stream().unwrap();
                for _ in 0..4 {
                    stream.tx.send(JoinRequest::default()).await.unwrap();
                }
            }
            let mut closing = Box::pin(client.close());
            assert!(futures::poll!(closing.as_mut()).is_pending());
            if !full_channel {
                let request = request_rx.recv().await.unwrap();
                assert!(matches!(request.kind, Some(join_request::Kind::Close(_))));
            }
            drop(closing);
            assert_eq!(other.state(), ClientState::Closed);
            assert!(other.shared.stream().is_none());
            while request_rx.recv().await.is_some() {}
        }
    }

    #[tokio::test]
    async fn close_success_survives_a_non_ok_terminal_stream_status() {
        let (mut client, mut request_rx, pending) = ready_client_with_stream();
        let stream = client.shared.stream().unwrap();
        let (finished, completion) = tokio::sync::watch::channel(None);
        client.shared.set_stream(Some(JoinStream {
            retirement: Arc::new(JoinRetirement {
                finished: completion,
                address: client.address.clone(),
                retiring: AtomicBool::new(false),
            }),
            ..stream
        }));
        let mut closing = Box::pin(client.close());
        assert!(futures::poll!(closing.as_mut()).is_pending());
        let request = request_rx.recv().await.unwrap();
        dispatch_response(
            &pending,
            Some(Ok(JoinResponse {
                request_id: request.request_id,
                kind: Some(join_response::Kind::Close(CloseEnvsResponse::default())),
                ..Default::default()
            })),
        );
        assert!(futures::poll!(closing.as_mut()).is_pending());
        finished
            .send(Some(Err(Status::unavailable("final status lost"))))
            .unwrap();
        assert!(closing.await.is_ok());
    }

    #[tokio::test]
    async fn close_on_never_used_client_is_local_only_and_opens_no_join() {
        // A client that handshook but never ran an operation has no Join stream.
        // close() must not open one just to tear it down; that would race any
        // other client's active session (server join_active CAS). It should
        // short-circuit to a local-only close.
        let channel = Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        let shared = Arc::new(Shared::new(ClientState::Ready, None));
        let mut client = EnvClient {
            client: EnvServiceClient::new(channel),
            token: String::new(),
            address: "tcp://127.0.0.1:1".to_string(),
            shared: shared.clone(),
            _lifetime: Arc::new(ClientLifetime(shared)),
            last_endpoint_total_ns: None,
            last_phases: EndpointPhases::default(),
            declared_workflow_edition: None,
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

    /// A fake env service: the handshake offers this build's editions; Join
    /// either fails to open (`rejecting`) or acks every request in order and
    /// records what it saw (`recording`).
    struct FakeEnvService {
        recorded: Option<Arc<std::sync::Mutex<Vec<JoinRequest>>>>,
        close_barrier: Option<Arc<tokio::sync::Notify>>,
    }

    impl FakeEnvService {
        fn rejecting() -> Self {
            Self {
                recorded: None,
                close_barrier: None,
            }
        }

        fn recording(recorded: Arc<std::sync::Mutex<Vec<JoinRequest>>>) -> Self {
            Self {
                recorded: Some(recorded),
                close_barrier: None,
            }
        }
    }

    #[async_trait::async_trait]
    impl EnvService for FakeEnvService {
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
            request: Request<tonic::Streaming<JoinRequest>>,
        ) -> std::result::Result<Response<Self::JoinStream>, Status> {
            let Some(recorded) = self.recorded.clone() else {
                return Err(Status::unavailable("join unavailable"));
            };
            let mut incoming = request.into_inner();
            let close_barrier = self.close_barrier.clone();
            let (tx, rx) = mpsc::channel(8);
            tokio::spawn(async move {
                while let Ok(Some(req)) = incoming.message().await {
                    let close = matches!(req.kind, Some(join_request::Kind::Close(_)));
                    let kind = match &req.kind {
                        Some(join_request::Kind::Configure(_)) => {
                            join_response::Kind::Configure(ConfigureEnvResponse::default())
                        }
                        Some(join_request::Kind::Reset(_)) => {
                            join_response::Kind::Reset(ResetResponse::default())
                        }
                        Some(join_request::Kind::Step(_)) => {
                            join_response::Kind::Step(StepResponse::default())
                        }
                        _ => join_response::Kind::Close(CloseEnvsResponse::default()),
                    };
                    let response = JoinResponse {
                        request_id: req.request_id.clone(),
                        kind: Some(kind),
                        ..Default::default()
                    };
                    recorded.lock().unwrap().push(req);
                    if tx.send(Ok(response)).await.is_err() {
                        break;
                    }
                    if close {
                        break;
                    }
                }
                if let Some(barrier) = &close_barrier {
                    barrier.notified().await;
                }
            });
            Ok(Response::new(ReceiverStream::new(rx)))
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

    /// A throwaway CA and a `localhost` / `127.0.0.1` server certificate:
    /// `(ca_pem, cert_pem, key_pem)`.
    fn test_pki() -> (String, String, String) {
        use rcgen::{BasicConstraints, CertificateParams, IsCa, Issuer, KeyPair};
        let ca_key = KeyPair::generate().unwrap();
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_key).unwrap();
        let issuer = Issuer::new(ca_params, ca_key);
        let key = KeyPair::generate().unwrap();
        let cert = CertificateParams::new(vec!["localhost".to_string(), "127.0.0.1".to_string()])
            .unwrap()
            .signed_by(&key, &issuer)
            .unwrap();
        (ca_cert.pem(), cert.pem(), key.serialize_pem())
    }

    /// Serve `PlainEnv` over TLS (or plaintext with `tls: None`) on a
    /// loopback port, recording each request's `authorization` header.
    async fn serve_env(
        tls: Option<(String, String)>,
    ) -> (
        std::net::SocketAddr,
        Arc<std::sync::Mutex<Vec<String>>>,
        oneshot::Sender<()>,
    ) {
        use crate::env::server::GrpcEnvServer;
        use crate::lifecycle::{ServeOptions, ShutdownTrigger};
        use rlmesh_proto::env::v1::env_service_server::EnvServiceServer;

        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let record = Arc::clone(&seen);
        let service = tonic::service::interceptor::InterceptedService::new(
            EnvServiceServer::new(GrpcEnvServer::new_with_options(
                PlainEnv::new("tls-env"),
                ShutdownTrigger::new(),
                ServeOptions::default(),
                None,
            )),
            move |request: tonic::Request<()>| {
                if let Some(value) = request.metadata().get("authorization") {
                    record
                        .lock()
                        .unwrap()
                        .push(value.to_str().unwrap_or_default().to_string());
                }
                Ok(request)
            },
        );
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let mut builder = tonic::transport::Server::builder();
        if let Some((cert, key)) = tls {
            builder = builder
                .tls_config(
                    tonic::transport::ServerTlsConfig::new()
                        .identity(tonic::transport::Identity::from_pem(cert, key)),
                )
                .unwrap();
        }
        tokio::spawn(async move {
            builder
                .add_service(service)
                .serve_with_incoming_shutdown(
                    tokio_stream::wrappers::TcpListenerStream::new(listener),
                    async {
                        let _ = shutdown_rx.await;
                    },
                )
                .await
        });
        (addr, seen, shutdown_tx)
    }

    fn trust(pem: &str) -> crate::TlsTrust {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rlmesh-test-ca-{}-{}.pem",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, pem).unwrap();
        let trust = crate::TlsTrust::from_ca_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        trust
    }

    #[tokio::test]
    async fn https_dials_tls_and_sends_a_bearer_token() {
        let (ca, cert, key) = test_pki();
        let (addr, seen, shutdown) = serve_env(Some((cert, key))).await;
        let credentials = crate::Credentials {
            token: "s3cret".to_string(),
            trust: trust(&ca),
        };
        let address = format!("https://localhost:{}", addr.port());
        let mut client = EnvClient::connect_with(&address, &credentials)
            .await
            .unwrap();
        assert_eq!(client.address(), address);
        client.handshake().await.unwrap();
        assert_eq!(seen.lock().unwrap().as_slice(), ["Bearer s3cret"]);
        let _ = shutdown.send(());
    }

    #[tokio::test]
    async fn tls_failures_name_the_handshake() {
        let (_, cert, key) = test_pki();
        let (other_ca, _, _) = test_pki();
        let (addr, _, shutdown) = serve_env(Some((cert, key))).await;
        let untrusted = crate::Credentials {
            token: String::new(),
            trust: trust(&other_ca),
        };
        let error =
            EnvClient::connect_with(&format!("https://localhost:{}", addr.port()), &untrusted)
                .await
                .err()
                .expect("an untrusted certificate must fail")
                .to_string();
        assert!(
            error.contains(&format!(
                "TLS handshake with localhost:{} failed",
                addr.port()
            )),
            "{error}"
        );
        assert!(error.contains("invalid peer certificate"), "{error}");
        let _ = shutdown.send(());

        // https:// against a plaintext server fails the handshake too.
        let (plain, _, shutdown) = serve_env(None).await;
        let error = EnvClient::connect_with(&format!("https://{plain}"), &untrusted)
            .await
            .err()
            .expect("a plaintext server cannot complete TLS")
            .to_string();
        assert!(error.contains("TLS handshake with 127.0.0.1:"), "{error}");
        let _ = shutdown.send(());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn model_client_dials_a_unix_socket() {
        let path = std::env::temp_dir().join(format!("rlmesh-model-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let (_reporter, health) = crate::health::serving_health_service().await;
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(health)
                .serve_with_incoming_shutdown(
                    tokio_stream::wrappers::UnixListenerStream::new(listener),
                    async {
                        let _ = shutdown_rx.await;
                    },
                )
                .await
        });
        let address = format!("unix://{}", path.display());
        let client = crate::ModelClient::connect(&address, "").await.unwrap();
        assert_eq!(client.address(), address);
        let _ = shutdown_tx.send(());
        let _ = std::fs::remove_file(&path);
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
        let handshake = authed.handshake().await.expect("authorized handshake");
        assert_eq!(authed.state(), ClientState::Ready);
        // The env's WANT rides back next to its CAN list and reaches the offer
        // the three-way floor is computed from.
        assert_eq!(
            handshake.preferred_workflow_edition.as_deref(),
            Some(rlmesh_proto::CURRENT_WORKFLOW_EDITION)
        );
        assert_eq!(
            handshake.session_offer(),
            rlmesh_proto::SessionOffer::this_build(None)
        );

        let _ = shutdown_tx.send(());
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), server).await;
    }

    /// A declared pin that admits no mutual edition is refused at the
    /// handshake, before any Join stream opens, naming every tier's WANT and
    /// CAN. With one retained edition no value passes the user-facing
    /// construction guards yet refuses here, so the pin is set on the bare
    /// `ServeOptions` struct, which only trims.
    #[tokio::test]
    async fn a_declared_pin_with_no_mutual_edition_is_refused_before_join() {
        use crate::env::server::GrpcEnvServer;
        use crate::lifecycle::{ServeOptions, ShutdownTrigger};
        use rlmesh_proto::env::v1::env_service_server::EnvServiceServer;

        let stale_base = "2025.01";
        let options = ServeOptions {
            workflow_edition: Some(format!(" {stale_base} ")),
            ..Default::default()
        };
        let service = EnvServiceServer::new(GrpcEnvServer::new_with_options(
            PlainEnv::new("pinned-env"),
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
        let mut client = EnvClient::connect_with_retry(&address, "", &connect_options)
            .await
            .expect("test server did not start");
        let err = client.handshake().await.unwrap_err();
        let message = err.to_string();
        let can = format!("{:?}", supported_workflow_editions());
        for expected in [
            "no mutual workflow edition with the env",
            &format!("env wants {stale_base:?} and can {can}"),
            &format!(
                "runtime wants {:?} and can {can}",
                rlmesh_proto::CURRENT_WORKFLOW_EDITION
            ),
        ] {
            assert!(
                message.contains(expected),
                "missing {expected:?} in: {message}"
            );
        }
        assert_ne!(client.state(), ClientState::Ready);
        assert!(client.shared.stream().is_none(), "no Join stream may open");

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
                .add_service(EnvServiceServer::new(FakeEnvService::rejecting()))
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

    /// Serve `service` on a free port until the returned sender drops.
    fn serve(service: impl EnvService) -> (String, oneshot::Sender<()>) {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(EnvServiceServer::new(service))
                .serve_with_shutdown(addr, async {
                    let _ = shutdown_rx.await;
                })
                .await
        });
        (format!("tcp://{addr}"), shutdown_tx)
    }

    async fn ready_client(address: &str) -> (EnvClient, EnvHandshake) {
        let connect_options =
            crate::connect::ConnectOptions::with_deadline(std::time::Duration::from_secs(5))
                .backoff(std::time::Duration::from_millis(10));
        let mut client = EnvClient::connect_with_retry(address, "", &connect_options)
            .await
            .expect("test server did not start");
        let handshake = client.handshake().await.expect("handshake");
        (client, handshake)
    }

    #[tokio::test]
    async fn retiring_join_uses_the_remaining_call_budget() {
        let barrier = Arc::new(tokio::sync::Notify::new());
        let recorded = Arc::default();
        let (address, _shutdown) = serve(FakeEnvService {
            recorded: Some(Arc::clone(&recorded)),
            close_barrier: Some(barrier.clone()),
        });
        let (mut previous, _) = ready_client(&address).await;
        previous.reset(ResetRequest::default()).await.unwrap();
        previous.detach();
        let (mut next, _) = ready_client(&address).await;
        let count = recorded.lock().unwrap().len();
        // Direct transport callers carry their budget on the wire request.
        let error = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            next.reset(ResetRequest {
                timeout_ms: 10,
                ..Default::default()
            }),
        )
        .await
        .expect("must honor the direct request budget")
        .unwrap_err();
        assert!(
            matches!(
                error,
                GrpcError::Transport(TransportError::Status {
                    code: tonic::Code::FailedPrecondition,
                    ..
                })
            ),
            "{error:?}"
        );
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(30);
        let error = with_request_deadline(deadline, async {
            // Earlier work has already consumed this call's entire budget.
            tokio::time::sleep_until(deadline).await;
            tokio::time::timeout(
                std::time::Duration::from_millis(100),
                next.reset(ResetRequest {
                    timeout_ms: 1000,
                    ..Default::default()
                }),
            )
            .await
            .expect("must use the remaining budget, not a fresh second")
        })
        .await
        .unwrap_err();
        assert!(
            matches!(
                error,
                GrpcError::Transport(TransportError::Status {
                    code: tonic::Code::FailedPrecondition,
                    ..
                })
            ),
            "{error:?}"
        );
        assert!(
            error
                .to_string()
                .contains("already has an active Join session")
        );
        assert_eq!(
            recorded.lock().unwrap().len(),
            count,
            "no new Join was opened"
        );
        barrier.notify_one();
        // A refusal does not wedge the new client once the old stream drains.
        next.reset(ResetRequest::default()).await.unwrap();
        barrier.notify_one();
        next.close().await.unwrap();
    }

    #[tokio::test]
    async fn close_waits_for_stream_termination_after_the_reply() {
        let barrier = Arc::new(tokio::sync::Notify::new());
        let recorded = Arc::default();
        let (address, _shutdown) = serve(FakeEnvService {
            recorded: Some(recorded),
            close_barrier: Some(barrier.clone()),
        });
        for _ in 0..20 {
            let (mut client, _) = ready_client(&address).await;
            client.reset(ResetRequest::default()).await.unwrap();
            let pending = client.shared.stream().unwrap().pending;
            let mut closing = Box::pin(client.close());
            assert!(futures::poll!(closing.as_mut()).is_pending());
            // Model a server paused between its Close reply and releasing the slot.
            // Wait until the pump delivered the reply, then poll close directly:
            // it must still be pending, independent of scheduler speed or load.
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while !pending.lock().unwrap().as_ref().unwrap().is_empty() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            let early = futures::poll!(closing.as_mut());
            barrier.notify_one();
            assert!(
                early.is_pending(),
                "close returned before the server ended its Join stream"
            );
            tokio::time::timeout(std::time::Duration::from_secs(2), closing)
                .await
                .unwrap()
                .unwrap();
        }
    }

    #[tokio::test]
    async fn last_client_drop_orders_the_next_join_even_with_an_edition_observer() {
        let barrier = Arc::new(tokio::sync::Notify::new());
        let (address, _shutdown) = serve(FakeEnvService {
            recorded: Some(Arc::default()),
            close_barrier: Some(barrier.clone()),
        });
        let (client, _) = ready_client(&address).await;
        let observer = client.pinned_edition();
        let mut clone = client.clone();
        drop(client);
        // A surviving client clone still owns and can drive the session.
        clone.reset(ResetRequest::default()).await.unwrap();
        drop(clone);
        assert!(
            observer.0.stream().is_none(),
            "observers must not retain the Join sender"
        );

        let (mut next, _) = ready_client(&address).await;
        let mut reset = Box::pin(next.reset(ResetRequest::default()));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(100), &mut reset)
                .await
                .is_err(),
            "the next Join overtook the dropped stream's EOF"
        );
        barrier.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(2), reset)
            .await
            .unwrap()
            .unwrap();
        barrier.notify_one();
        next.close().await.unwrap();
    }

    fn recorded_kinds(recorded: &std::sync::Mutex<Vec<JoinRequest>>) -> Vec<join_request::Kind> {
        recorded
            .lock()
            .unwrap()
            .iter()
            .filter_map(|req| req.kind.clone())
            .collect()
    }

    #[tokio::test]
    async fn join_opens_with_the_env_leg_pin_then_reset() {
        let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (address, _server) = serve(FakeEnvService::recording(Arc::clone(&recorded)));
        let (mut client, handshake) = ready_client(&address).await;

        client
            .reset(ResetRequest::default())
            .await
            .expect("reset after the pin ack");
        assert_eq!(
            recorded_kinds(&recorded),
            vec![
                join_request::Kind::Configure(ConfigureEnvRequest {
                    selected_workflow_edition: handshake.selected_workflow_edition.clone(),
                }),
                join_request::Kind::Reset(ResetRequest::default()),
            ]
        );

        // The pin rides the stream open, not every operation.
        client.step(StepRequest::default()).await.expect("step");
        assert_eq!(recorded_kinds(&recorded).len(), 3);
    }

    #[tokio::test]
    async fn a_session_floor_pin_replaces_the_env_leg_pin_verbatim() {
        // A model leg lowers the session to the three-way floor; the env is
        // pinned to that exact spelling (the bare base here, which is not the
        // cohort spelling the handshake selected on a prerelease build).
        let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (address, _server) = serve(FakeEnvService::recording(Arc::clone(&recorded)));
        let (mut client, handshake) = ready_client(&address).await;
        let floor = handshake.workflow_edition.base();

        client.pin_workflow_edition(format!(" {floor} "));
        client.reset(ResetRequest::default()).await.expect("reset");
        assert_eq!(
            recorded_kinds(&recorded).first(),
            Some(&join_request::Kind::Configure(ConfigureEnvRequest {
                selected_workflow_edition: floor.to_string(),
            }))
        );
    }

    #[tokio::test]
    async fn the_served_env_is_pinned_before_its_first_reset() {
        use crate::env::server::GrpcEnvServer;

        let env = PlainEnv::new("pinned-env");
        let events = Arc::clone(&env.events);
        let (address, _server) = serve(GrpcEnvServer::new(env));
        let (mut client, handshake) = ready_client(&address).await;

        client.reset(ResetRequest::default()).await.expect("reset");
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                format!("pin:{}", handshake.workflow_edition),
                "reset".to_string()
            ]
        );
    }

    #[tokio::test]
    async fn a_refused_pin_sends_no_reset_and_names_both_sides() {
        use crate::env::server::GrpcEnvServer;

        let env = PlainEnv::new("pinned-env");
        let events = Arc::clone(&env.events);
        let (address, _server) = serve(GrpcEnvServer::new(env));
        let (mut client, _) = ready_client(&address).await;

        client.pin_workflow_edition("2099.01".to_string());
        let error = client
            .reset(ResetRequest::default())
            .await
            .expect_err("the env cannot run 2099.01");
        let message = error.to_string();
        for expected in ["2099.01", &format!("{:?}", supported_workflow_editions())] {
            assert!(
                message.contains(expected),
                "missing {expected:?} in: {message}"
            );
        }
        let seen = events.lock().unwrap().clone();
        assert!(
            seen.is_empty(),
            "a refused pin must reach the env as neither pin nor reset: {seen:?}"
        );
        assert!(
            client.shared.stream().is_none(),
            "the refused Join is released"
        );
        assert_eq!(client.state(), ClientState::Ready);
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
