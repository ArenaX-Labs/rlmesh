use rlmesh_proto::{
    capabilities,
    core::v1::{ShutdownRequest as CoreShutdownRequest, ShutdownResponse as CoreShutdownResponse},
    model::v1::{
        CloseParticipantRequest, GroupedPredictRequest, GroupedPredictResponse, JoinRequest,
        JoinResponse, PredictRequest, PredictResponse, ReleaseAdapterRequest, ResetAdapterRequest,
        ResolveAdapterRequest, ShutdownRequest, join_request, join_response,
        model_service_client::ModelServiceClient,
    },
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use crate::error::{Error as GrpcError, ProtocolError, TransportError};
use crate::helpers::normalize_tcp_session_address;
use crate::states::ClientState;

use super::stream::{PendingResponses, spawn_response_pump};
use super::validation::{decode_error, route_request_id, validate_predict_route, validate_route};
use super::wire::{join_request_kind_name, model_error_to_grpc_error};

/// Client for a ModelService server's Join bidi stream.
///
/// # Concurrency: demux by `request_id`
///
/// Responses are demultiplexed by `request_id` through a shared pending map, so
/// **multiple requests can be in flight on one connection at once**. A response
/// pump routes each response to the matching waiter; a response with no pending
/// waiter (a late one from an abandoned request, or an unknown id) is logged and
/// dropped.
///
/// The public per-request methods ([`predict`](Self::predict),
/// [`configure_route`](Self::configure_route), [`close_route`](Self::close_route),
/// [`close`](Self::close)) take `&mut self` and await their own response, so used
/// alone they behave exactly as before (one request at a time, response matched
/// by id). To actually overlap predicts on one connection, use
/// [`predict_concurrent`](Self::predict_concurrent), which takes `&self` and may
/// be called from multiple tasks concurrently. The matching server advertises
/// the `rlmesh.model.concurrent_predict.v1` capability when it pipelines.
pub struct ModelClient {
    address: String,
    client: ModelServiceClient<tonic::transport::Channel>,
    token: String,
    state: ClientState,
    request_tx: Option<mpsc::Sender<JoinRequest>>,
    pending: PendingResponses,
    request_counter: Arc<AtomicU64>,
    /// Endpoint-local op duration (ns) attached to the last Join response. The
    /// nested per-step telemetry message was replaced by this hot scalar
    /// (`JoinResponse.endpoint_total_ns`).
    last_endpoint_total_ns: Option<u64>,
    server_capabilities: HashMap<String, String>,
    /// The model's offered workflow editions, learned at handshake. Feeds the
    /// three-way session-floor reconciliation.
    server_supported_editions: Vec<String>,
}

impl ModelClient {
    pub async fn connect(address: &str, token: &str) -> Result<Self, GrpcError> {
        let address = normalize_tcp_session_address(address)?;
        let endpoint = crate::configure_endpoint(
            tonic::transport::Endpoint::from_shared(address.replacen("tcp://", "http://", 1))
                .map_err(|err| TransportError::InvalidAddress(err.to_string()))?,
        );
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
            pending: Default::default(),
            request_counter: Arc::new(AtomicU64::new(0)),
            last_endpoint_total_ns: None,
            server_capabilities: HashMap::new(),
            server_supported_editions: Vec::new(),
        })
    }

    /// Connect to a ModelService server, retrying until the server accepts the
    /// connection (or the deadline/cancellation in `options` fires).
    ///
    /// Only the transport connect is retried; perform the handshake explicitly
    /// on the returned client.
    pub async fn connect_with_retry(
        address: &str,
        token: &str,
        options: &crate::connect::ConnectOptions,
    ) -> Result<Self, GrpcError> {
        crate::connect::retry_connect(options, || Self::connect(address, token)).await
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    /// Take the endpoint-local op duration (ns) attached to the most recent
    /// Join response, if any (`JoinResponse.endpoint_total_ns`).
    pub fn take_last_endpoint_total_ns(&mut self) -> Option<u64> {
        self.last_endpoint_total_ns.take()
    }

    /// Whether the server advertised that it pipelines Join-stream predicts
    /// (`rlmesh.model.concurrent_predict.v1`). Advisory: overlapping predicts via
    /// [`predict_concurrent`](Self::predict_concurrent) work either way, but
    /// serialize behind the handler when this is false.
    pub fn server_pipelines_predict(&self) -> bool {
        rlmesh_proto::has_capability(
            &self.server_capabilities,
            capabilities::MODEL_CONCURRENT_PREDICT_V1,
        )
    }

    /// The model's bind-time offer learned at handshake: the workflow editions it
    /// supports. The runtime (client to both the env and the model) feeds this
    /// into [`rlmesh_proto::negotiate_session_floor`] to pick the route edition.
    /// Empty before [`handshake`](Self::handshake) completes. Capabilities are read
    /// pairwise (see [`server_pipelines_predict`](Self::server_pipelines_predict)), not here.
    pub fn model_session_offer(&self) -> rlmesh_proto::SessionOffer {
        rlmesh_proto::SessionOffer {
            editions: self.server_supported_editions.clone(),
        }
    }

    pub async fn handshake(&mut self) -> Result<(), GrpcError> {
        if self.state != ClientState::Connected {
            return Err(crate::error::ClientError::NotConnected.into());
        }

        let request = self.authorized_request(rlmesh_proto::model::v1::HandshakeRequest {
            base: Some(rlmesh_proto::core_handshake_request("rlmesh-model", &[])),
        })?;

        let response = self
            .client
            .handshake(request)
            .await
            .map_err(crate::error::status_to_grpc_error)?
            .into_inner()
            .base
            .ok_or_else(|| {
                GrpcError::from(ProtocolError::HandshakeFailed(
                    "handshake response missing base".to_string(),
                ))
            })?;

        // `compatible` is the server's verdict on protocol generation (plain
        // equality — a wrong generation is a hard, full-restart break) and edition
        // mutuality. The client trusts it; there is no echoed server generation to
        // re-verify.
        if !response.compatible {
            return Err(
                ProtocolError::HandshakeFailed(response.error_message.unwrap_or_default()).into(),
            );
        }
        self.server_supported_editions = response.supported_workflow_editions;
        self.server_capabilities = response.capabilities;

        self.setup_join_stream().await?;
        self.state = ClientState::Ready;
        Ok(())
    }

    pub async fn resolve_adapter(
        &mut self,
        request: ResolveAdapterRequest,
    ) -> Result<(), GrpcError> {
        self.ensure_ready()?;
        validate_route(
            request
                .context
                .as_ref()
                .ok_or_else(|| decode_error("resolve_adapter missing adapter context"))?,
        )?;
        let request_id = route_request_id(request.context.as_ref(), || self.next_request_id());
        let response = self
            .send_on_stream(JoinRequest {
                kind: Some(join_request::Kind::ResolveAdapter(request)),
                request_id,
            })
            .await?;
        self.last_endpoint_total_ns = response.endpoint_total_ns;
        match response.kind {
            Some(join_response::Kind::ResolveAdapter(_)) => Ok(()),
            Some(join_response::Kind::Error(error)) => Err(model_error_to_grpc_error(error)),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "ResolveAdapterResponse".to_string(),
                actual: format!("{:?}", response.kind),
            }
            .into()),
        }
    }

    pub async fn predict(&mut self, request: PredictRequest) -> Result<PredictResponse, GrpcError> {
        self.ensure_ready()?;
        validate_predict_route(&request)?;
        let request_id = route_request_id(request.context.as_ref(), || self.next_request_id());
        let response = self
            .send_on_stream(JoinRequest {
                kind: Some(join_request::Kind::Predict(request)),
                request_id,
            })
            .await?;
        self.last_endpoint_total_ns = response.endpoint_total_ns;

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

    pub async fn reset_adapter(&mut self, request: ResetAdapterRequest) -> Result<(), GrpcError> {
        self.ensure_ready()?;
        validate_route(
            request
                .context
                .as_ref()
                .ok_or_else(|| decode_error("reset_adapter missing adapter context"))?,
        )?;
        let request_id = route_request_id(request.context.as_ref(), || self.next_request_id());
        let response = self
            .send_on_stream(JoinRequest {
                kind: Some(join_request::Kind::ResetAdapter(request)),
                request_id,
            })
            .await?;
        self.last_endpoint_total_ns = response.endpoint_total_ns;
        match response.kind {
            Some(join_response::Kind::ResetAdapter(_)) => Ok(()),
            Some(join_response::Kind::Error(error)) => Err(model_error_to_grpc_error(error)),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "ResetAdapterResponse".to_string(),
                actual: format!("{:?}", response.kind),
            }
            .into()),
        }
    }

    pub async fn release_adapter(
        &mut self,
        request: ReleaseAdapterRequest,
    ) -> Result<(), GrpcError> {
        self.ensure_ready()?;
        validate_route(
            request
                .context
                .as_ref()
                .ok_or_else(|| decode_error("release_adapter missing adapter context"))?,
        )?;
        let request_id = route_request_id(request.context.as_ref(), || self.next_request_id());
        let response = self
            .send_on_stream(JoinRequest {
                kind: Some(join_request::Kind::ReleaseAdapter(request)),
                request_id,
            })
            .await?;
        self.last_endpoint_total_ns = response.endpoint_total_ns;
        match response.kind {
            Some(join_response::Kind::ReleaseAdapter(_)) => Ok(()),
            Some(join_response::Kind::Error(error)) => Err(model_error_to_grpc_error(error)),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "ReleaseAdapterResponse".to_string(),
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
            kind: Some(join_request::Kind::Close(CloseParticipantRequest {
                reason: reason.into(),
            })),
            request_id: self.next_request_id(),
        };

        let response = tokio::time::timeout(timeout, self.send_on_stream(request))
            .await
            .map_err(|_| GrpcError::Timeout(timeout))??;
        self.last_endpoint_total_ns = response.endpoint_total_ns;
        self.state = ClientState::Closed;

        match response.kind {
            Some(join_response::Kind::Close(_)) => Ok(()),
            Some(join_response::Kind::Error(error)) => Err(model_error_to_grpc_error(error)),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "CloseParticipantResponse".to_string(),
                actual: format!("{:?}", response.kind),
            }
            .into()),
        }
    }

    pub async fn shutdown(
        &mut self,
        reason: impl Into<String>,
    ) -> Result<CoreShutdownResponse, GrpcError> {
        if self.state == ClientState::Closed {
            return Err(crate::error::ClientError::NotConnected.into());
        }

        let request = self.authorized_request(ShutdownRequest {
            base: Some(CoreShutdownRequest {
                reason: reason.into(),
            }),
        })?;
        let response = self
            .client
            .shutdown(request)
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
            self.state = ClientState::Closed;
            self.request_tx.take();
            // Drop the request stream sender; the pump will then see the stream
            // end and fail any still-pending waiters.
            self.pending.lock().expect("pending map poisoned").clear();
        }

        Ok(response)
    }

    /// Issue a predict that may overlap other in-flight requests on the same
    /// connection. Takes `&self`, so it can be called from multiple tasks
    /// concurrently; responses are demuxed by `request_id`.
    ///
    /// Unlike [`predict`](Self::predict), this does not record
    /// `last_telemetry` (that field is single-threaded `&mut self` state); read
    /// per-call telemetry from the returned response if needed. The server only
    /// pipelines these when it advertises `rlmesh.model.concurrent_predict.v1`;
    /// against a serial server they still complete correctly, just serialized.
    pub async fn predict_concurrent(
        &self,
        request: PredictRequest,
    ) -> Result<PredictResponse, GrpcError> {
        self.ensure_ready()?;
        validate_predict_route(&request)?;
        let request_id = route_request_id(request.context.as_ref(), || self.next_request_id());
        let response = self
            .send_on_stream(JoinRequest {
                kind: Some(join_request::Kind::Predict(request)),
                request_id,
            })
            .await?;
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

    /// Issue a control-plane-grouped predict: one request carrying several
    /// already-routed predict groups, each processed by the server against its
    /// own route's spec (and optionally fused into one forward pass). Takes
    /// `&self`, so it can overlap other in-flight requests on the connection;
    /// the response is demuxed by the grouped request's own envelope id.
    ///
    /// Per-group *handler* errors come back inside the `GroupedPredictResponse`
    /// (one `GroupedPredictResult` per group), so a single bad group does not
    /// fail this call. A malformed group (missing/empty route context) or an
    /// empty batch fails the whole call up front, before anything is sent.
    pub async fn grouped_predict(
        &self,
        request: GroupedPredictRequest,
    ) -> Result<GroupedPredictResponse, GrpcError> {
        self.ensure_ready()?;
        if request.groups.is_empty() {
            return Err(decode_error(
                "grouped predict must include at least one group",
            ));
        }
        for group in &request.groups {
            validate_predict_route(group)?;
        }
        // A grouped request spans multiple routes, so it gets its OWN envelope
        // request_id (never a group's): the demux is one response per request_id.
        let request_id = self.next_request_id();
        let response = self
            .send_on_stream(JoinRequest {
                kind: Some(join_request::Kind::GroupedPredict(request)),
                request_id,
            })
            .await?;
        match response.kind {
            Some(join_response::Kind::GroupedPredict(grouped)) => Ok(grouped),
            Some(join_response::Kind::Error(error)) => Err(model_error_to_grpc_error(error)),
            _ => Err(ProtocolError::UnexpectedMessage {
                expected: "GroupedPredictResponse".to_string(),
                actual: format!("{:?}", response.kind),
            }
            .into()),
        }
    }

    async fn setup_join_stream(&mut self) -> Result<(), GrpcError> {
        let (tx, rx) = mpsc::channel::<JoinRequest>(32);
        let request_stream = ReceiverStream::new(rx);
        let request = self.authorized_request(request_stream)?;

        let response = self
            .client
            .join(request)
            .await
            .map_err(crate::error::status_to_grpc_error)?;

        self.request_tx = Some(tx);
        spawn_response_pump(response.into_inner(), Arc::clone(&self.pending));
        Ok(())
    }

    /// Send one request and await its response, matched by `request_id` through
    /// the shared pending map. Takes `&self` so both the `&mut self` public
    /// methods and the concurrent `predict_concurrent` path can use it.
    async fn send_on_stream(&self, request: JoinRequest) -> Result<JoinResponse, GrpcError> {
        let request_id = request.request_id.clone();
        let request_kind = join_request_kind_name(request.kind.as_ref());
        let tx = self
            .request_tx
            .clone()
            .ok_or(crate::error::ClientError::NotHandshaked)?;

        // Register the waiter *before* sending so a fast response cannot race
        // ahead of the pending insert.
        let (response_tx, response_rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock().expect("pending map poisoned");
            // The demux is keyed by request_id; silently overwriting a live
            // entry would strand the first caller until stream end. Reject a
            // duplicate caller-supplied id instead.
            if pending.contains_key(&request_id) {
                return Err(crate::error::ProtocolError::DecodeError(format!(
                    "request_id {request_id:?} is already in flight on this stream"
                ))
                .into());
            }
            pending.insert(request_id.clone(), response_tx);
        }

        if tx.send(request).await.is_err() {
            // The stream is gone; clean up our pending entry.
            self.pending
                .lock()
                .expect("pending map poisoned")
                .remove(&request_id);
            return Err(TransportError::ConnectionClosed.into());
        }

        match response_rx.await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(status)) => {
                tracing::error!(
                    request_id = %request_id,
                    request_kind,
                    code = ?status.code(),
                    message = %status.message(),
                    "model join stream returned an error status"
                );
                Err(crate::error::status_to_grpc_error(status))
            }
            Err(_) => {
                // The pump dropped our sender without sending; the stream closed.
                tracing::error!(
                    request_id = %request_id,
                    request_kind,
                    "model join stream closed while waiting for response"
                );
                Err(TransportError::ConnectionClosed.into())
            }
        }
    }

    fn ensure_ready(&self) -> Result<(), GrpcError> {
        match self.state {
            ClientState::Ready => Ok(()),
            ClientState::Connected => Err(crate::error::ClientError::NotHandshaked.into()),
            ClientState::Closed => Err(crate::error::ClientError::NotConnected.into()),
        }
    }

    fn next_request_id(&self) -> String {
        let id = self.request_counter.fetch_add(1, Ordering::Relaxed) + 1;
        format!("model-grpc-req-{id}")
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
    use rlmesh_proto::model::v1::{AdapterContext, PredictResponse};
    use tonic::transport::Endpoint;

    /// Build a `Ready` client wired to an in-memory request channel, plus a
    /// "fake pump" handle: a closure-friendly clone of the pending map and the
    /// receiver of outgoing requests. This drives the real `send_on_stream`
    /// demux without a transport.
    fn ready_client() -> (ModelClient, mpsc::Receiver<JoinRequest>, PendingResponses) {
        let (request_tx, request_rx) = mpsc::channel(8);
        let channel = Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        let pending: PendingResponses = Default::default();
        let client = ModelClient {
            address: "tcp://127.0.0.1:1".to_string(),
            client: ModelServiceClient::new(channel),
            token: String::new(),
            state: ClientState::Ready,
            request_tx: Some(request_tx),
            pending: Arc::clone(&pending),
            request_counter: Arc::new(AtomicU64::new(0)),
            last_endpoint_total_ns: None,
            server_capabilities: HashMap::new(),
            server_supported_editions: Vec::new(),
        };
        (client, request_rx, pending)
    }

    /// Route a response into the pending map exactly as the real pump would.
    fn deliver(pending: &PendingResponses, request_id: &str, response: JoinResponse) {
        let sender = pending
            .lock()
            .unwrap()
            .remove(request_id)
            .expect("expected a pending waiter for the request id");
        sender.send(Ok(response)).expect("waiter still alive");
    }

    fn predict_response_for(request_id: &str) -> JoinResponse {
        JoinResponse {
            request_id: request_id.to_string(),
            kind: Some(join_response::Kind::Predict(PredictResponse::default())),
            endpoint_total_ns: None,
        }
    }

    #[tokio::test]
    async fn send_on_stream_resolves_by_request_id() {
        let (client, mut request_rx, pending) = ready_client();

        let send = tokio::spawn(async move {
            client
                .send_on_stream(JoinRequest {
                    request_id: "target".to_string(),
                    kind: Some(join_request::Kind::Predict(PredictRequest::default())),
                })
                .await
        });

        // The request reaches the stream and a waiter is registered.
        let sent = request_rx.recv().await.unwrap();
        assert_eq!(sent.request_id, "target");
        deliver(&pending, "target", predict_response_for("target"));

        let response = send.await.unwrap().unwrap();
        assert_eq!(response.request_id, "target");
    }

    #[tokio::test]
    async fn two_overlapping_requests_demux_out_of_order() {
        let (client, mut request_rx, pending) = ready_client();
        let client = Arc::new(client);

        // Two predicts in flight at once on the same connection.
        let c1 = Arc::clone(&client);
        let first = tokio::spawn(async move {
            c1.send_on_stream(JoinRequest {
                request_id: "req-1".to_string(),
                kind: Some(join_request::Kind::Predict(PredictRequest::default())),
            })
            .await
        });
        let c2 = Arc::clone(&client);
        let second = tokio::spawn(async move {
            c2.send_on_stream(JoinRequest {
                request_id: "req-2".to_string(),
                kind: Some(join_request::Kind::Predict(PredictRequest::default())),
            })
            .await
        });

        // Both requests are sent before either response arrives.
        let mut sent_ids = vec![
            request_rx.recv().await.unwrap().request_id,
            request_rx.recv().await.unwrap().request_id,
        ];
        sent_ids.sort();
        assert_eq!(sent_ids, vec!["req-1".to_string(), "req-2".to_string()]);

        // Deliver responses out of order: req-2 first, then req-1.
        deliver(&pending, "req-2", predict_response_for("req-2"));
        deliver(&pending, "req-1", predict_response_for("req-1"));

        // Each waiter gets exactly its own response, regardless of order.
        assert_eq!(first.await.unwrap().unwrap().request_id, "req-1");
        assert_eq!(second.await.unwrap().unwrap().request_id, "req-2");
    }

    /// A routed, single-row predict usable as a grouped-predict member.
    fn valid_member(request_id: &str, episode: &str) -> PredictRequest {
        PredictRequest {
            context: Some(AdapterContext {
                session_id: "s".to_string(),
                env_id: "r".to_string(),
                request_id: request_id.to_string(),
            }),
            observation: None,
            episode_ids: vec![episode.to_string()],
        }
    }

    #[tokio::test]
    async fn grouped_predict_uses_a_fresh_envelope_id_and_resolves_by_it() {
        let (client, mut request_rx, pending) = ready_client();

        let send = tokio::spawn(async move {
            client
                .grouped_predict(GroupedPredictRequest {
                    groups: vec![valid_member("group-0", "ep-0")],
                })
                .await
        });

        let sent = request_rx.recv().await.unwrap();
        assert!(matches!(
            sent.kind,
            Some(join_request::Kind::GroupedPredict(_))
        ));
        // The grouped request carries its OWN envelope id, never a group's, so
        // the one-response-per-request_id demux invariant holds.
        assert_ne!(sent.request_id, "group-0");

        deliver(
            &pending,
            &sent.request_id,
            JoinResponse {
                request_id: sent.request_id.clone(),
                kind: Some(join_response::Kind::GroupedPredict(
                    GroupedPredictResponse::default(),
                )),
                endpoint_total_ns: None,
            },
        );

        let response = send.await.unwrap().unwrap();
        assert!(response.results.is_empty());
    }

    #[tokio::test]
    async fn grouped_and_single_predict_demux_out_of_order() {
        let (client, mut request_rx, pending) = ready_client();
        let client = Arc::new(client);

        // A grouped predict and a single predict in flight at once on one stream.
        let c1 = Arc::clone(&client);
        let grouped = tokio::spawn(async move {
            c1.grouped_predict(GroupedPredictRequest {
                groups: vec![valid_member("group-0", "ep-g")],
            })
            .await
        });
        let c2 = Arc::clone(&client);
        let single =
            tokio::spawn(
                async move { c2.predict_concurrent(valid_member("single", "ep-s")).await },
            );

        // Capture both outgoing envelope ids (the single predict's id is "single"
        // from its context; the grouped predict's is a fresh counter id).
        let sent_a = request_rx.recv().await.unwrap();
        let sent_b = request_rx.recv().await.unwrap();
        let (grouped_id, single_id) = match (&sent_a.kind, &sent_b.kind) {
            (Some(join_request::Kind::GroupedPredict(_)), _) => {
                (sent_a.request_id.clone(), sent_b.request_id.clone())
            }
            _ => (sent_b.request_id.clone(), sent_a.request_id.clone()),
        };
        assert_eq!(single_id, "single");

        // Deliver out of order: the single predict first, then the grouped one.
        deliver(&pending, &single_id, predict_response_for(&single_id));
        deliver(
            &pending,
            &grouped_id,
            JoinResponse {
                request_id: grouped_id.clone(),
                kind: Some(join_response::Kind::GroupedPredict(
                    GroupedPredictResponse::default(),
                )),
                endpoint_total_ns: None,
            },
        );

        // Each waiter resolves to its own typed response despite the overlap and
        // the out-of-order delivery: grouped -> GroupedPredictResponse, single ->
        // PredictResponse. Both completing proves they demuxed independently.
        assert!(grouped.await.unwrap().is_ok());
        assert!(single.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn grouped_predict_rejects_empty_groups() {
        let (client, _request_rx, _pending) = ready_client();
        let err = client
            .grouped_predict(GroupedPredictRequest { groups: vec![] })
            .await
            .unwrap_err();
        assert!(err.to_string().contains("at least one group"), "got: {err}");
    }

    #[tokio::test]
    async fn send_on_stream_errors_when_waiter_dropped_by_stream_close() {
        let (client, _request_rx, pending) = ready_client();

        let send = tokio::spawn(async move {
            client
                .send_on_stream(JoinRequest {
                    request_id: "orphan".to_string(),
                    kind: Some(join_request::Kind::Predict(PredictRequest::default())),
                })
                .await
        });

        // Simulate the pump dropping every pending sender on stream close.
        tokio::time::sleep(Duration::from_millis(20)).await;
        pending.lock().unwrap().clear();

        let result = send.await.unwrap();
        assert!(result.is_err(), "a closed stream must fail the waiter");
    }
}
