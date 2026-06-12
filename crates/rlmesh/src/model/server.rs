use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rlmesh_grpc::lifecycle::{
    IdleActivity, await_close_with_timeout, await_server_shutdown, start_idle_shutdown,
};
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
use tokio::net::TcpListener;
#[cfg(unix)]
use tokio::net::UnixListener;
use tokio::sync::{Mutex, mpsc};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::TcpListenerStream;
#[cfg(unix)]
use tokio_stream::wrappers::UnixListenerStream;
use tonic::{Request, Response, Status, Streaming};

use super::handler::ModelHandler;
use super::lifecycle::{finish_lifecycle, finish_route_lifecycle, update_lifecycle};
use super::wire::{
    ModelAction, model_action_to_endpoint_response, model_error, model_error_from_error,
    model_join_request_operation, model_observation_from_endpoint_request,
    model_operation_telemetry,
};
use crate::{BindAddress, Error, Result, ServeOptions, spaces};

pub(super) async fn serve_model_with_options<H>(
    handler: H,
    address: BindAddress,
    token: &str,
    options: ServeOptions,
) -> Result<()>
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
    let serve_result = match address {
        BindAddress::Tcp { host, port } => {
            let listener = TcpListener::bind((host.as_str(), port))
                .await
                .map_err(|err| Error::Server(err.to_string()))?;
            await_server_shutdown(
                tonic::transport::Server::builder()
                    .add_service(service)
                    .serve_with_incoming_shutdown(
                        TcpListenerStream::new(listener),
                        shutdown.cancelled_owned(),
                    ),
                shutdown.clone(),
                drain_timeout,
            )
            .await
            .map_err(|err| Error::Server(err.to_string()))
        }
        BindAddress::Unix { path } => {
            #[cfg(not(unix))]
            {
                let _ = path;
                return Err(Error::Address(
                    "unix sockets are not supported on Windows; use tcp://host:port instead"
                        .to_string(),
                ));
            }

            #[cfg(unix)]
            {
                crate::address::remove_stale_socket(&path)?;
                let listener =
                    UnixListener::bind(&path).map_err(|err| Error::Server(err.to_string()))?;
                let result = await_server_shutdown(
                    tonic::transport::Server::builder()
                        .add_service(service)
                        .serve_with_incoming_shutdown(
                            UnixListenerStream::new(listener),
                            shutdown.cancelled_owned(),
                        ),
                    shutdown.clone(),
                    drain_timeout,
                )
                .await
                .map_err(|err| Error::Server(err.to_string()));
                // Unlink the socket file on shutdown so a subsequent serve on
                // the same path does not fail with AddrInUse.
                let _ = std::fs::remove_file(&path);
                result
            }
        }
    };
    let close_result = close_model(handler, close_timeout).await;
    match (serve_result, close_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(err), Ok(())) => Err(err),
        (Ok(()), Err(err)) => Err(err),
        (Err(serve_err), Err(close_err)) => Err(Error::Internal(format!(
            "model server failed: {serve_err}; close hook failed: {close_err}"
        ))),
    }
}

async fn close_model<H: ModelHandler + 'static>(
    handler: Arc<Mutex<H>>,
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
        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<JoinResponse, Status>>(64);

        tokio::spawn(async move {
            while let Some(request_result) = request_stream.next().await {
                let request = match request_result {
                    Ok(request) => request,
                    Err(error) => {
                        log_join_stream_error(&error);
                        break;
                    }
                };
                let close_after = matches!(request.kind, Some(join_request::Kind::Close(_)));
                if let Some(activity_tx) = &activity_tx {
                    let _ = activity_tx.send(IdleActivity::Started);
                }
                let response = handle_model_request(
                    request,
                    Arc::clone(&handler),
                    Arc::clone(&active_episodes),
                    Arc::clone(&route_configs),
                )
                .await;
                if let Some(activity_tx) = &activity_tx {
                    let _ = activity_tx.send(IdleActivity::Finished);
                }
                let send_result = tx.send(Ok(response)).await;
                if send_result.is_err() {
                    tracing::warn!(
                        "model join response receiver closed before response could be delivered"
                    );
                    break;
                }
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
