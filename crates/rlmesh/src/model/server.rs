//! Served model endpoint: the tonic `ModelService` implementation.
//!
//! # Why this lives in the facade (vs. EnvService in rlmesh-grpc)
//!
//! The symmetric `EnvService` implementation lives one layer down in
//! `rlmesh-grpc` because it is parameterized over `rlmesh_grpc::env::
//! Environment`, a trait *defined in that crate*. The model service, by
//! contrast, is parameterized over the user-facing [`crate::ModelHandler`]
//! trait (and its [`crate::ModelObservation`] / route / lifecycle types), all
//! defined here in the facade. Moving this impl into `rlmesh-grpc` would require
//! either making `rlmesh-grpc` depend on `rlmesh` (a dependency cycle, since the
//! facade already depends on `rlmesh-grpc`) or relocating the entire
//! `ModelHandler` family — a public-API-breaking, Python-rippling refactor and
//! net-new "grpc-level model handler trait" design. Review finding #74 judged
//! that churn out of scope; the asymmetry is intentional and documented here.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rlmesh_grpc::lifecycle::{IdleActivity, await_close_with_timeout, start_idle_shutdown};
use rlmesh_grpc::wire::env_contract_from_proto;
use rlmesh_proto::model::v1::{
    CloseResponse, CloseRouteResponse, ConfigureRouteRequest, ConfigureRouteResponse,
    HandshakeRequest, HandshakeResponse, JoinRequest, JoinResponse, PredictRequest,
    ShutdownRequest, ShutdownResponse, join_request, join_response,
    model_service_server::{ModelService as ModelServiceTrait, ModelServiceServer},
};
use rlmesh_proto::{
    MIN_SUPPORTED_PROTOCOL_GENERATION, PROTOCOL_GENERATION, capabilities, capability_map,
    negotiate_workflow_edition, supported_workflow_editions,
};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use super::handler::ModelHandler;
use super::lifecycle::{finish_lifecycle, finish_route_lifecycle, update_lifecycle};
use super::wire::{
    ModelAction, model_action_to_endpoint_response, model_error, model_error_from_error,
    model_join_request_operation, model_observation_from_endpoint_request,
    model_operation_telemetry,
};
use crate::bound::BoundListener;
use crate::{BindAddress, Error, Result, ServeOptions, spaces};

/// A model server that has bound its listener but not yet started serving.
///
/// Created by [`crate::ModelWorker::bind_async`]. Use
/// [`BoundModelServer::local_addr`]
/// to read the resolved bind address (e.g. the OS-assigned port for TCP port
/// 0), then [`BoundModelServer::serve`] to run until shutdown.
pub struct BoundModelServer {
    listener: BoundListener,
    router: tonic::transport::server::Router,
    shutdown: rlmesh_grpc::lifecycle::ShutdownTrigger,
    handler: Arc<Mutex<dyn ModelHandler>>,
    local_addr: BindAddress,
    drain_timeout: Option<Duration>,
    close_timeout: Option<Duration>,
}

impl BoundModelServer {
    /// The resolved address the server is bound to.
    pub fn local_addr(&self) -> &BindAddress {
        &self.local_addr
    }

    /// Serve until shutdown, then run the handler close hook.
    pub async fn serve(self) -> Result<()> {
        let serve_result = self
            .listener
            .serve(self.router, self.shutdown, self.drain_timeout)
            .await;
        let close_result = close_model(self.handler, self.close_timeout).await;
        match (serve_result, close_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(err), Ok(())) => Err(err),
            (Ok(()), Err(err)) => Err(err),
            (Err(serve_err), Err(close_err)) => Err(Error::Internal(format!(
                "model server failed: {serve_err}; close hook failed: {close_err}"
            ))),
        }
    }
}

pub(super) async fn bind_model_with_options<H>(
    handler: H,
    address: BindAddress,
    token: &str,
    options: ServeOptions,
) -> Result<BoundModelServer>
where
    H: ModelHandler + 'static,
{
    let handler = Arc::new(Mutex::new(handler));
    let shutdown = rlmesh_grpc::lifecycle::ShutdownTrigger::new();
    let activity_tx = start_idle_shutdown(options.idle_timeout, shutdown.clone());
    let drain_timeout = options.drain_timeout;
    let close_timeout = options.close_timeout;
    let service = model_service(
        Arc::clone(&handler),
        token.to_string(),
        activity_tx,
        shutdown.clone(),
        options,
    );

    let listener = BoundListener::bind(address).await?;
    let local_addr = listener.local_addr()?;
    // Always-on standard gRPC health service (`grpc.health.v1`). The listener
    // is already bound (bind-first), so the overall server health is marked
    // SERVING immediately (review finding #57).
    let (_health_reporter, health_service) = rlmesh_grpc::health::serving_health_service().await;
    let router = tonic::transport::Server::builder()
        .add_service(health_service)
        .add_service(service);
    // Upcast so the bound handle does not leak the handler generic; only the
    // close hook needs the handler afterward.
    let handler: Arc<Mutex<dyn ModelHandler>> = handler;

    Ok(BoundModelServer {
        listener,
        router,
        shutdown,
        handler,
        local_addr,
        drain_timeout,
        close_timeout,
    })
}

async fn close_model(
    handler: Arc<Mutex<dyn ModelHandler>>,
    close_timeout: Option<Duration>,
) -> Result<()> {
    let close = async { handler.lock().await.on_close().await };
    await_close_with_timeout(close, close_timeout)
        .await
        .map_err(Error::Timeout)?
}

struct ServedModelServer<H> {
    handler: Arc<Mutex<H>>,
    active_episodes: Arc<Mutex<HashMap<(String, i32), String>>>,
    route_configs: Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
    token: String,
    activity_tx: Option<mpsc::UnboundedSender<IdleActivity>>,
    shutdown: rlmesh_grpc::lifecycle::ShutdownTrigger,
    serve_options: ServeOptions,
}

#[derive(Debug, Clone)]
pub(super) struct ModelRouteConfig {
    pub(super) env_contract: Option<spaces::EnvContract>,
    pub(super) num_envs: usize,
}

fn model_service<H>(
    handler: Arc<Mutex<H>>,
    token: String,
    activity_tx: Option<mpsc::UnboundedSender<IdleActivity>>,
    shutdown: rlmesh_grpc::lifecycle::ShutdownTrigger,
    serve_options: ServeOptions,
) -> ModelServiceServer<ServedModelServer<H>>
where
    H: ModelHandler + 'static,
{
    ModelServiceServer::new(ServedModelServer {
        handler,
        active_episodes: Arc::new(Mutex::new(HashMap::new())),
        route_configs: Arc::new(Mutex::new(HashMap::new())),
        token,
        activity_tx,
        shutdown,
        serve_options,
    })
    .max_decoding_message_size(rlmesh_grpc::MAX_MESSAGE_SIZE)
    .max_encoding_message_size(rlmesh_grpc::MAX_MESSAGE_SIZE)
}

#[tonic::async_trait]
impl<H> ModelServiceTrait for ServedModelServer<H>
where
    H: ModelHandler + 'static,
{
    async fn handshake(
        &self,
        request: Request<HandshakeRequest>,
    ) -> std::result::Result<Response<HandshakeResponse>, Status> {
        self.authenticate(&request)?;
        let request = request.into_inner();
        let protocol_compatible = rlmesh_proto::is_protocol_generation_compatible(
            &request.protocol_generation,
            PROTOCOL_GENERATION,
        );
        let selected_edition = negotiate_workflow_edition(&request.supported_workflow_editions);
        let compatible = protocol_compatible && selected_edition.is_some();
        Ok(Response::new(HandshakeResponse {
            compatible,
            server_protocol_generation: PROTOCOL_GENERATION.to_string(),
            min_supported_protocol_generation: MIN_SUPPORTED_PROTOCOL_GENERATION.to_string(),
            error_message: if compatible {
                String::new()
            } else if !protocol_compatible {
                format!(
                    "protocol generation {} not compatible with server {}",
                    request.protocol_generation, PROTOCOL_GENERATION
                )
            } else if request.supported_workflow_editions.is_empty() {
                format!(
                    "client offered no workflow editions (clients from 0.1.0-beta.2 or older predate edition negotiation and are not supported); server supports [{}]",
                    supported_workflow_editions().join(", ")
                )
            } else {
                format!(
                    "no mutually supported workflow edition; client offered [{}], server supports [{}]",
                    request.supported_workflow_editions.join(", "),
                    supported_workflow_editions().join(", ")
                )
            },
            capabilities: capability_map(&[
                capabilities::MODEL_SERVICE_V1,
                capabilities::SPACES_CORE_V1,
                // The served model pipelines Join-stream requests (see `join`).
                // Advisory: lets clients detect that overlapping predicts will
                // actually pipeline rather than serialize behind the handler.
                capabilities::MODEL_CONCURRENT_PREDICT_V1,
            ]),
            selected_workflow_edition: if compatible {
                selected_edition.unwrap_or_default().to_string()
            } else {
                String::new()
            },
            supported_workflow_editions: supported_workflow_editions(),
        }))
    }

    type JoinStream =
        tokio_stream::wrappers::ReceiverStream<std::result::Result<JoinResponse, Status>>;

    /// Handle a Join bidi stream.
    ///
    /// # Concurrency contract (pipelined predict)
    ///
    /// Requests on the stream are **pipelined**: the read loop spawns a task per
    /// request rather than awaiting each to completion before reading the next,
    /// so decode/validation, the handler critical section, encode, and the
    /// response pump overlap across requests. Responses are tagged with their
    /// originating `request_id` and emitted in **completion order**, which may
    /// differ from arrival order (a slow request no longer head-of-line-blocks a
    /// fast one). The matching [`crate::model`] client demuxes responses by
    /// `request_id`. The capability `model.concurrent_predict.v1` advertises this
    /// behavior at handshake.
    ///
    /// ## What is serialized, and why
    ///
    /// The user handler is an `Arc<Mutex<H>>` with `&mut self` hooks, so every
    /// handler/lifecycle critical section still runs one at a time (option (a):
    /// we keep the mutex rather than break the public `ModelHandler` trait to
    /// `&self`). The win is real even so: decode/encode and the stream pump no
    /// longer block behind the handler.
    ///
    /// ## Ordering guarantees
    ///
    /// - **Per-route order is preserved.** Each request is assigned, in arrival
    ///   order, a slot on a per-route chain of completion signals; a request's
    ///   handler critical section waits for the previous same-route request to
    ///   finish before acquiring the handler lock. So for a given route,
    ///   `ConfigureRoute` → `Predict` → … → `CloseRoute` run their lifecycle
    ///   updates and predicts in exactly the order the client sent them. This
    ///   keeps `active_episodes` / `update_lifecycle` correct and prevents a
    ///   `CloseRoute` from overtaking an in-flight `Predict` for the same route
    ///   and dropping its episode accounting.
    /// - **Different routes interleave.** Critical sections for distinct routes
    ///   may run in either order (still one at a time under the mutex); per-route
    ///   episode accounting is independent, so this is safe and is what yields
    ///   out-of-order completion.
    /// - **`Close` is a barrier.** A whole-session `Close` drains *all* routes'
    ///   episodes, so it waits for every outstanding request on the stream to
    ///   finish before draining, then ends the stream. It can never overtake an
    ///   in-flight predict on any route.
    ///
    /// Idle-shutdown `IdleActivity::Started`/`Finished` is emitted as a balanced
    /// pair around each request's processing (inside the per-request task), so
    /// the active count tracks genuinely in-flight work even while pipelined.
    async fn join(
        &self,
        request: Request<Streaming<JoinRequest>>,
    ) -> std::result::Result<Response<Self::JoinStream>, Status> {
        self.authenticate(&request)?;
        let mut request_stream = request.into_inner();
        let handler = Arc::clone(&self.handler);
        let active_episodes = Arc::clone(&self.active_episodes);
        let route_configs = Arc::clone(&self.route_configs);
        let activity_tx = self.activity_tx.clone();
        let concurrency = self
            .serve_options
            .predict_concurrency
            .unwrap_or(rlmesh_grpc::DEFAULT_PREDICT_CONCURRENCY)
            .max(1);
        let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<JoinResponse, Status>>(64);

        tokio::spawn(async move {
            // Per-route tail of completion signals. Reading happens on this single
            // task, in arrival order, so the map needs no locking. Each entry is a
            // receiver that fires when the most recently enqueued request for that
            // route finishes its handler critical section.
            let mut route_tails: HashMap<String, tokio::sync::oneshot::Receiver<()>> =
                HashMap::new();

            while let Some(request_result) = request_stream.next().await {
                let request = match request_result {
                    Ok(request) => request,
                    Err(error) => {
                        log_join_stream_error(&error);
                        break;
                    }
                };
                let close_after = matches!(request.kind, Some(join_request::Kind::Close(_)));
                let route_key = join_request_route_key(&request);

                // Compute the gate this request must wait on before entering its
                // handler critical section, and the signal it fires on completion.
                let (gate, done_tx): (RequestGate, tokio::sync::oneshot::Sender<()>) =
                    if close_after {
                        // Close drains every route: wait for all outstanding requests.
                        let prev = route_tails.drain().map(|(_, rx)| rx).collect::<Vec<_>>();
                        let (done_tx, _done_rx) = tokio::sync::oneshot::channel();
                        (RequestGate::All(prev), done_tx)
                    } else {
                        // Chain this request after the previous one on its route (if
                        // any). Requests with no route key (malformed) are ungated.
                        let prev = route_key.as_ref().and_then(|key| route_tails.remove(key));
                        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
                        if let Some(key) = &route_key {
                            route_tails.insert(key.clone(), done_rx);
                        }
                        (RequestGate::Prev(prev), done_tx)
                    };

                // Acquire a permit before spawning so the number of outstanding
                // per-request tasks stays bounded; held for the task's lifetime.
                let permit = match Arc::clone(&semaphore).acquire_owned().await {
                    Ok(permit) => permit,
                    Err(_) => break,
                };

                if let Some(activity_tx) = &activity_tx {
                    let _ = activity_tx.send(IdleActivity::Started);
                }

                let handler = Arc::clone(&handler);
                let active_episodes = Arc::clone(&active_episodes);
                let route_configs = Arc::clone(&route_configs);
                let activity_tx = activity_tx.clone();
                let tx = tx.clone();

                tokio::spawn(async move {
                    let _permit = permit;
                    // Wait for predecessors so the handler critical section runs in
                    // per-route arrival order (or, for Close, after every route).
                    gate.wait().await;

                    let response =
                        handle_model_request(request, handler, active_episodes, route_configs)
                            .await;

                    // Release successors on this route *before* sending the
                    // response, so per-route ordering does not depend on the
                    // unbounded response channel draining.
                    let _ = done_tx.send(());

                    if let Some(activity_tx) = &activity_tx {
                        let _ = activity_tx.send(IdleActivity::Finished);
                    }
                    if tx.send(Ok(response)).await.is_err() {
                        tracing::warn!(
                            "model join response receiver closed before response could be delivered"
                        );
                    }
                });

                if close_after {
                    break;
                }
            }
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
            rx,
        )))
    }

    async fn shutdown(
        &self,
        request: Request<ShutdownRequest>,
    ) -> std::result::Result<Response<ShutdownResponse>, Status> {
        self.authenticate(&request)?;
        let request = request.into_inner();
        if !self.serve_options.allow_remote_shutdown {
            return Ok(Response::new(ShutdownResponse {
                accepted: false,
                message: "remote shutdown is disabled for this model endpoint".to_string(),
            }));
        }
        self.shutdown.trigger(if request.reason.is_empty() {
            "remote shutdown".to_string()
        } else {
            request.reason.clone()
        });
        Ok(Response::new(ShutdownResponse {
            accepted: true,
            message: if request.reason.is_empty() {
                "shutdown accepted".to_string()
            } else {
                format!("shutdown accepted: {}", request.reason)
            },
        }))
    }
}

pub(super) async fn handle_model_request<H: ModelHandler + 'static>(
    request: JoinRequest,
    handler: Arc<Mutex<H>>,
    active_episodes: Arc<Mutex<HashMap<(String, i32), String>>>,
    route_configs: Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
) -> JoinResponse {
    let request_id = request.request_id.clone();
    let operation = model_join_request_operation(request.kind.as_ref());
    let started_at = Instant::now();

    let kind = match request.kind {
        Some(join_request::Kind::ConfigureRoute(request)) => {
            handle_configure_route(request, route_configs).await
        }
        Some(join_request::Kind::Predict(request)) => {
            handle_predict(request, handler, active_episodes, route_configs).await
        }
        Some(join_request::Kind::CloseRoute(request)) => {
            let route_key = request.context.as_ref().and_then(route_config_key);
            match route_key {
                Some(route_key) => {
                    let drain_result = {
                        let mut handler = handler.lock().await;
                        let mut active_episodes = active_episodes.lock().await;
                        finish_route_lifecycle(&mut *handler, &mut active_episodes, &route_key)
                            .await
                    };
                    if let Err(error) = drain_result {
                        return JoinResponse {
                            kind: Some(model_error(error.to_string())),
                            telemetry: Some(model_operation_telemetry(operation, started_at)),
                            request_id,
                        };
                    }
                    route_configs.lock().await.remove(&route_key);
                    Some(join_response::Kind::CloseRoute(CloseRouteResponse {}))
                }
                _ => Some(model_error("close_route missing route_id")),
            }
        }
        Some(join_request::Kind::Close(_request)) => {
            let drain_result = {
                let mut handler = handler.lock().await;
                let mut active_episodes = active_episodes.lock().await;
                finish_lifecycle(&mut *handler, &mut active_episodes).await
            };
            if let Err(error) = drain_result {
                return JoinResponse {
                    kind: Some(model_error(error.to_string())),
                    telemetry: Some(model_operation_telemetry(operation, started_at)),
                    request_id,
                };
            }
            Some(join_response::Kind::Close(CloseResponse {}))
        }
        None => Some(model_error("empty model request")),
    };

    JoinResponse {
        kind,
        telemetry: Some(model_operation_telemetry(operation, started_at)),
        request_id,
    }
}

async fn handle_configure_route(
    request: ConfigureRouteRequest,
    route_configs: Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
) -> Option<join_response::Kind> {
    let route = match request.context {
        Some(route) => route,
        None => return Some(model_error("configure_route missing route context")),
    };
    let route_id = route.route_id.clone();
    if route_id.is_empty() {
        return Some(model_error("configure_route route_id is empty"));
    }
    let route_key = match route_config_key(&route) {
        Some(route_key) => route_key,
        None => {
            return Some(model_error(
                "configure_route missing session_id or route_id",
            ));
        }
    };
    let env_contract = match request.env_contract {
        Some(env_contract) => env_contract,
        None => return Some(model_error("configure_route missing env_contract")),
    };
    let env_contract = match env_contract_from_proto(env_contract) {
        Ok(env_contract) => env_contract,
        Err(error) => return Some(model_error(error.to_string())),
    };
    if env_contract.num_envs == 0 {
        return Some(model_error(
            "configure_route env_contract.num_envs must be positive",
        ));
    }
    let num_envs = env_contract.num_envs as usize;
    route_configs.lock().await.insert(
        route_key,
        ModelRouteConfig {
            env_contract: Some(env_contract),
            num_envs,
        },
    );
    Some(join_response::Kind::ConfigureRoute(
        ConfigureRouteResponse {},
    ))
}

async fn handle_predict<H: ModelHandler + 'static>(
    request: PredictRequest,
    handler: Arc<Mutex<H>>,
    active_episodes: Arc<Mutex<HashMap<(String, i32), String>>>,
    route_configs: Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
) -> Option<join_response::Kind> {
    async {
        let mut observation = model_observation_from_endpoint_request(request)?;
        let route = observation.route.clone();
        let route_key = model_route_config_key(&route);
        let config = route_configs
            .lock()
            .await
            .get(&route_key)
            .cloned()
            .ok_or_else(|| Error::model("model route was not configured"))?;
        observation.env_contract = config.env_contract;
        observation.num_envs = config.num_envs;
        if observation.route.slots.len() > config.num_envs {
            return Err(Error::model(
                "predict route has more slots than configured route",
            ));
        }

        let mut handler = handler.lock().await;
        let mut active_episodes = active_episodes.lock().await;
        update_lifecycle(&mut *handler, &mut active_episodes, &observation).await?;
        let action = handler.predict(observation).await?;
        Ok::<_, Error>(join_response::Kind::Predict(
            model_action_to_endpoint_response(ModelAction {
                action: Some(action),
                route,
                telemetry: None,
            }),
        ))
    }
    .await
    .unwrap_or_else(|error| model_error_from_error(&error))
    .into()
}

/// The set of predecessor completion signals a request must await before it may
/// enter its handler critical section, preserving per-route arrival order.
enum RequestGate {
    /// Wait for the previous same-route request (if any). A dropped/`None`
    /// sender resolves immediately, so a failed predecessor never deadlocks a
    /// successor.
    Prev(Option<tokio::sync::oneshot::Receiver<()>>),
    /// Wait for every outstanding request across all routes (whole-session
    /// `Close` barrier).
    All(Vec<tokio::sync::oneshot::Receiver<()>>),
}

impl RequestGate {
    async fn wait(self) {
        match self {
            RequestGate::Prev(prev) => {
                if let Some(rx) = prev {
                    // A `RecvError` means the predecessor's sender was dropped
                    // (it finished or its task panicked); either way we proceed.
                    let _ = rx.await;
                }
            }
            RequestGate::All(prev) => {
                for rx in prev {
                    let _ = rx.await;
                }
            }
        }
    }
}

/// Route key for ordering a Join request on its per-route chain.
///
/// `ConfigureRoute` / `Predict` / `CloseRoute` are keyed by their
/// `session_id:route_id`; whole-session `Close` and malformed requests (missing
/// context or ids) return `None` — `Close` is handled as an all-routes barrier
/// by the caller, and ungated malformed requests still produce an in-band error.
fn join_request_route_key(request: &JoinRequest) -> Option<String> {
    let context = match request.kind.as_ref()? {
        join_request::Kind::ConfigureRoute(request) => request.context.as_ref()?,
        join_request::Kind::Predict(request) => request.context.as_ref()?,
        join_request::Kind::CloseRoute(request) => request.context.as_ref()?,
        join_request::Kind::Close(_) => return None,
    };
    route_config_key(context)
}

fn route_config_key(context: &rlmesh_proto::model::v1::PredictContext) -> Option<String> {
    if context.session_id.is_empty() || context.route_id.is_empty() {
        return None;
    }
    Some(format!("{}:{}", context.session_id, context.route_id))
}

fn model_route_config_key(route: &super::types::ModelRouteContext) -> String {
    format!("{}:{}", route.session_id, route.route_id)
}

/// Log an inbound Join-stream error meaningfully instead of swallowing it.
///
/// Mirrors the env server's handling so a client that aborts or sends a
/// malformed Join stream leaves a diagnostic trace rather than disappearing
/// silently.
fn log_join_stream_error(error: &Status) {
    tracing::debug!("model join stream closed: {}", error);
}

impl<H> ServedModelServer<H> {
    /// Reject the request unless its `authorization` metadata matches the
    /// configured route token (constant-time compare). An empty configured
    /// token disables authentication: every request is accepted.
    fn authenticate<T>(&self, request: &Request<T>) -> std::result::Result<(), Status> {
        let provided = request
            .metadata()
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        if rlmesh_grpc::helpers::bearer_token_matches(&self.token, provided) {
            Ok(())
        } else {
            Err(Status::unauthenticated("invalid route token"))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use async_trait::async_trait;
    use rlmesh_proto::CURRENT_WORKFLOW_EDITION;
    use tracing::field::{Field, Visit};
    use tracing::subscriber::with_default;
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::{Layer, Registry};

    use super::*;

    #[derive(Clone, Default)]
    struct CaptureLayer {
        messages: Arc<StdMutex<Vec<String>>>,
    }

    struct MessageVisitor<'a>(&'a mut Vec<String>);

    impl Visit for MessageVisitor<'_> {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.0.push(format!("{value:?}"));
            }
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for CaptureLayer {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
            let mut messages = self.messages.lock().unwrap();
            let mut visitor = MessageVisitor(&mut messages);
            event.record(&mut visitor);
        }
    }

    #[test]
    fn inbound_join_stream_error_is_logged_not_swallowed() {
        let layer = CaptureLayer::default();
        let messages = Arc::clone(&layer.messages);
        let subscriber = Registry::default().with(layer);

        with_default(subscriber, || {
            log_join_stream_error(&Status::aborted("client went away"));
        });

        let messages = messages.lock().unwrap();
        assert!(
            messages
                .iter()
                .any(|message| message.contains("model join stream closed")
                    && message.contains("client went away")),
            "expected a diagnostic log for the inbound stream error, got {messages:?}"
        );
    }

    #[derive(Default)]
    struct NoopModelHandler;

    #[async_trait]
    impl ModelHandler for NoopModelHandler {
        async fn predict(
            &mut self,
            _observation: super::super::types::ModelObservation,
        ) -> Result<spaces::BinaryPayload> {
            Ok(spaces::BinaryPayload { data: Vec::new() })
        }
    }

    fn test_server() -> ServedModelServer<NoopModelHandler> {
        ServedModelServer {
            handler: Arc::new(Mutex::new(NoopModelHandler)),
            active_episodes: Arc::new(Mutex::new(HashMap::new())),
            route_configs: Arc::new(Mutex::new(HashMap::new())),
            token: String::new(),
            activity_tx: None,
            shutdown: rlmesh_grpc::lifecycle::ShutdownTrigger::new(),
            serve_options: ServeOptions::default(),
        }
    }

    fn handshake_request(offered_editions: &[&str]) -> HandshakeRequest {
        HandshakeRequest {
            protocol_generation: PROTOCOL_GENERATION.to_string(),
            client_name: "client".to_string(),
            client_version: "0.1.0-beta.2".to_string(),
            capabilities: Default::default(),
            supported_workflow_editions: offered_editions
                .iter()
                .map(|edition| edition.to_string())
                .collect(),
        }
    }

    #[tokio::test]
    async fn handshake_selects_highest_mutual_edition() {
        let server = test_server();

        for offer in [
            &[CURRENT_WORKFLOW_EDITION][..],
            &["2025.01", CURRENT_WORKFLOW_EDITION, "2031.12"][..],
        ] {
            let response =
                ModelServiceTrait::handshake(&server, Request::new(handshake_request(offer)))
                    .await
                    .unwrap()
                    .into_inner();

            assert!(response.compatible, "offer {offer:?} must be accepted");
            assert_eq!(response.selected_workflow_edition, CURRENT_WORKFLOW_EDITION);
            assert_eq!(
                response.supported_workflow_editions,
                supported_workflow_editions()
            );
        }
    }

    #[tokio::test]
    async fn handshake_rejects_offer_without_mutual_edition() {
        let server = test_server();

        for offer in [&[][..], &["2026"][..], &["2026.11", "next"][..]] {
            let response =
                ModelServiceTrait::handshake(&server, Request::new(handshake_request(offer)))
                    .await
                    .unwrap()
                    .into_inner();

            assert!(!response.compatible, "offer {offer:?} must be rejected");
            assert!(response.error_message.contains("workflow edition"));
            if offer.is_empty() {
                assert!(
                    response
                        .error_message
                        .contains("predate edition negotiation")
                );
            }
            assert!(response.selected_workflow_edition.is_empty());
        }
    }
}
