//! Served model endpoint: the tonic `ModelService` implementation.
//!
//! This implementation stays in the facade because it is parameterized over the
//! public [`crate::ModelHandler`] family. Moving it into `rlmesh-grpc` would
//! introduce a dependency cycle or force a new lower-level model trait.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rlmesh_grpc::lifecycle::{
    ActivityFinishedGuard, IdleActivity, await_close_with_timeout, start_idle_shutdown,
};
use rlmesh_grpc::wire::env_spec_from_proto;
use rlmesh_proto::model::v1::{
    CloseParticipantResponse, CloseRouteResponse, ConfigureRouteRequest, ConfigureRouteResponse,
    GroupedPredictRequest, GroupedPredictResponse, GroupedPredictResult, HandshakeRequest,
    HandshakeResponse, JoinRequest, JoinResponse, PredictRequest, PredictResponse, ShutdownRequest,
    ShutdownResponse, grouped_predict_result, join_request, join_response,
    model_service_server::{ModelService as ModelServiceTrait, ModelServiceServer},
};
use rlmesh_proto::{
    PROTOCOL_GENERATION, capabilities, capability_map, evaluate_handshake, peer_info,
    supported_workflow_editions,
};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use super::handler::{ModelHandler, ModelRouteSetup};
use super::lifecycle::{finish_lifecycle, finish_route_lifecycle, update_lifecycle};
use super::types::{ModelObservation, ModelRouteContext};
use super::wire::{
    ModelAction, check_actions_conform, model_action_to_endpoint_response, model_endpoint_total_ns,
    model_error, model_error_from_error, model_error_value,
    model_observation_from_endpoint_request,
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
        crate::error::join_results(serve_result, close_result, "model server failed")
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
    // Obtain the route setup once, before serving: ConfigureRoute then resolves
    // routes through it without taking the per-request handler lock, so
    // configuring one route never blocks on an in-flight predict on another.
    let route_setup = handler.lock().await.route_setup();
    let shutdown = rlmesh_grpc::lifecycle::ShutdownTrigger::new();
    let activity_tx = start_idle_shutdown(options.idle_timeout, shutdown.clone());
    let drain_timeout = options.drain_timeout;
    let close_timeout = options.close_timeout;
    let service = model_service(
        Arc::clone(&handler),
        route_setup,
        token.to_string(),
        activity_tx,
        shutdown.clone(),
        options,
    );

    let listener = BoundListener::bind(address).await?;
    let local_addr = listener.local_addr()?;
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
    route_setup: Option<Arc<dyn ModelRouteSetup>>,
    active_episodes: Arc<Mutex<HashMap<(String, i32), String>>>,
    route_configs: Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
    token: String,
    activity_tx: Option<mpsc::UnboundedSender<IdleActivity>>,
    shutdown: rlmesh_grpc::lifecycle::ShutdownTrigger,
    serve_options: ServeOptions,
}

#[derive(Debug, Clone)]
pub(super) struct ModelRouteConfig {
    pub(super) env_contract: Option<Arc<spaces::EnvContract>>,
    /// The reconciled three-way (relay) session floor the runtime pinned this
    /// route to via `ConfigureRoute`. Authoritative over the model's own
    /// (pairwise) handshake result. Stored/logged here; full enforcement is
    /// minimal while the build holds a single generation and edition.
    pub(super) floor: Option<RouteFloor>,
}

/// Floor the runtime pinned a route to (from `ConfigureRouteRequest`). Generation
/// is not part of the floor — it is gated by equality at the handshake.
#[derive(Debug, Clone)]
pub(super) struct RouteFloor {
    pub(super) selected_workflow_edition: String,
    pub(super) active_capabilities: HashMap<String, String>,
}

fn model_service<H>(
    handler: Arc<Mutex<H>>,
    route_setup: Option<Arc<dyn ModelRouteSetup>>,
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
        route_setup,
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
        let request = request
            .into_inner()
            .base
            .ok_or_else(|| Status::invalid_argument("handshake request missing base"))?;
        let compat = evaluate_handshake(
            &request.protocol_generation,
            &request.supported_workflow_editions,
        );
        // Moving prerelease/dev editions are pinned by their cohort suffix; an
        // exact name match in negotiation is the compatibility check.
        let compatible = compat.is_compatible();
        Ok(Response::new(HandshakeResponse {
            base: Some(rlmesh_proto::core::v1::HandshakeResponse {
                compatible,
                peer_info: Some(peer_info("rlmesh-model")),
                server_protocol_generation: PROTOCOL_GENERATION.to_string(),
                error_message: if compatible {
                    String::new()
                } else if !compat.protocol_compatible {
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
                    compat.selected_edition.unwrap_or_default().to_string()
                } else {
                    String::new()
                },
                supported_workflow_editions: supported_workflow_editions(),
            }),
        }))
    }

    type JoinStream =
        tokio_stream::wrappers::ReceiverStream<std::result::Result<JoinResponse, Status>>;

    /// Handle a pipelined Join stream.
    ///
    /// Responses may complete out of arrival order, but each route is serialized
    /// through a per-route chain so lifecycle updates cannot overtake earlier
    /// requests on the same route. A whole-session `Close` waits for all in-flight
    /// route work before draining final episode accounting.
    async fn join(
        &self,
        request: Request<Streaming<JoinRequest>>,
    ) -> std::result::Result<Response<Self::JoinStream>, Status> {
        self.authenticate(&request)?;
        let mut request_stream = request.into_inner();
        let handler = Arc::clone(&self.handler);
        let route_setup = self.route_setup.clone();
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
            // Per-route tail of completion signals, kept bounded across the
            // stream's lifetime even as it cycles fresh route keys per episode.
            // See [`RouteTails`] for the ordering and reaping invariants.
            let mut route_tails = RouteTails::new();

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
                // `CloseRoute` is keyed like any other request: it replaces its
                // route's tail so a later `ConfigureRoute` reopening the same key
                // chains after it, and its fired tail is later reaped, keeping the
                // map bounded for long-lived streams.
                let (gate, done_tx): (RequestGate, tokio::sync::oneshot::Sender<()>) =
                    if close_after {
                        // Close drains every route: wait for all outstanding requests.
                        route_tails.close_all_gate()
                    } else {
                        // Chain this request after the previous one on its route (if
                        // any). Requests with no route key (malformed) are ungated.
                        route_tails.next_keyed_gate(route_key.as_deref())
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
                // RAII pairing: the matching Finished must fire even if the
                // spawned request task panics, or the idle-shutdown in-flight
                // count stays elevated forever and idle shutdown never fires.
                let activity_guard = ActivityFinishedGuard::new(activity_tx.clone());

                let handler = Arc::clone(&handler);
                let route_setup = route_setup.clone();
                let active_episodes = Arc::clone(&active_episodes);
                let route_configs = Arc::clone(&route_configs);
                let tx = tx.clone();

                tokio::spawn(async move {
                    let _permit = permit;
                    let _activity_guard = activity_guard;
                    // Wait for predecessors so the handler critical section runs in
                    // per-route arrival order (or, for Close, after every route).
                    gate.wait().await;

                    let response = handle_model_request(
                        request,
                        handler,
                        route_setup,
                        active_episodes,
                        route_configs,
                    )
                    .await;

                    // Release successors on this route *before* sending the
                    // response, so per-route ordering does not depend on the
                    // unbounded response channel draining.
                    let _ = done_tx.send(());

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
        let request = request
            .into_inner()
            .base
            .ok_or_else(|| Status::invalid_argument("shutdown request missing base"))?;
        if !self.serve_options.allow_remote_shutdown {
            return Ok(Response::new(ShutdownResponse {
                base: Some(rlmesh_proto::core::v1::ShutdownResponse {
                    accepted: false,
                    message: "remote shutdown is disabled for this model endpoint".to_string(),
                }),
            }));
        }
        self.shutdown.trigger(if request.reason.is_empty() {
            "remote shutdown".to_string()
        } else {
            request.reason.clone()
        });
        Ok(Response::new(ShutdownResponse {
            base: Some(rlmesh_proto::core::v1::ShutdownResponse {
                accepted: true,
                message: if request.reason.is_empty() {
                    "shutdown accepted".to_string()
                } else {
                    format!("shutdown accepted: {}", request.reason)
                },
            }),
        }))
    }
}

pub(super) async fn handle_model_request<H: ModelHandler + 'static>(
    request: JoinRequest,
    handler: Arc<Mutex<H>>,
    route_setup: Option<Arc<dyn ModelRouteSetup>>,
    active_episodes: Arc<Mutex<HashMap<(String, i32), String>>>,
    route_configs: Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
) -> JoinResponse {
    let request_id = request.request_id.clone();
    let started_at = Instant::now();

    let kind = match request.kind {
        Some(join_request::Kind::ConfigureRoute(request)) => {
            handle_configure_route(request, route_setup.as_deref(), route_configs).await
        }
        Some(join_request::Kind::Predict(request)) => {
            handle_predict(request, handler, active_episodes, route_configs).await
        }
        Some(join_request::Kind::GroupedPredict(request)) => {
            handle_grouped_predict(request, handler, active_episodes, route_configs).await
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
                            endpoint_total_ns: Some(model_endpoint_total_ns(started_at)),
                            request_id,
                        };
                    }
                    route_configs.lock().await.remove(&route_key);
                    if let Some(route_setup) = route_setup.as_deref()
                        && let Err(error) = route_setup.close_route(&route_key).await
                    {
                        return JoinResponse {
                            kind: Some(model_error(error.to_string())),
                            endpoint_total_ns: Some(model_endpoint_total_ns(started_at)),
                            request_id,
                        };
                    }
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
                    endpoint_total_ns: Some(model_endpoint_total_ns(started_at)),
                    request_id,
                };
            }
            // Close drains every route globally above, so tear down every
            // route's per-route setup and config too rather than leaking them
            // for the server's lifetime. Drop the configs guard before the
            // async close_route calls; re-lock only to clear.
            let route_keys: Vec<String> = route_configs.lock().await.keys().cloned().collect();
            if let Some(route_setup) = route_setup.as_deref() {
                for route_key in &route_keys {
                    if let Err(error) = route_setup.close_route(route_key).await {
                        return JoinResponse {
                            kind: Some(model_error_from_error(&error)),
                            endpoint_total_ns: Some(model_endpoint_total_ns(started_at)),
                            request_id,
                        };
                    }
                }
            }
            route_configs.lock().await.clear();
            Some(join_response::Kind::Close(CloseParticipantResponse {}))
        }
        None => Some(model_error("empty model request")),
    };

    JoinResponse {
        kind,
        endpoint_total_ns: Some(model_endpoint_total_ns(started_at)),
        request_id,
    }
}

async fn handle_configure_route(
    request: ConfigureRouteRequest,
    route_setup: Option<&dyn ModelRouteSetup>,
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
    let env_spec = match request.env_spec {
        Some(env_spec) => env_spec,
        None => return Some(model_error("configure_route missing env_spec")),
    };
    // The runtime is the binding authority for the three-way (relay) floor: it
    // decode-rebuilds env<->model envelopes, so these values override the model's
    // own (pairwise, possibly higher) handshake result. Today there is a single
    // edition, so we store/log the floor rather than re-deriving behavior from it;
    // full enforcement lands when a second edition exists.
    let floor = if request.selected_workflow_edition.is_empty() {
        // Older runtime that predates floor propagation: nothing to pin.
        None
    } else {
        let floor = RouteFloor {
            selected_workflow_edition: request.selected_workflow_edition,
            active_capabilities: request.active_capabilities,
        };
        // Minimal enforcement: the floor is authoritative over this model's own
        // handshake. Reject a route the runtime pinned to an edition this build
        // cannot drive, instead of silently running it under the wrong semantics.
        // (With a single edition this never fires; the path exists so the floor
        // is honored, not merely logged, once a second edition lands.)
        if let Err(error) = enforce_route_floor(&floor) {
            return Some(model_error(error));
        }
        tracing::debug!(
            route = %route_id,
            selected_workflow_edition = %floor.selected_workflow_edition,
            active_capabilities = ?floor.active_capabilities,
            "model route pinned to runtime-reconciled session floor"
        );
        Some(floor)
    };
    // The model receives only the stable EnvSpec; the orchestration knobs
    // (num_envs/render_mode/autoreset) are runtime-owned and default here.
    let env_contract = match env_spec_from_proto(env_spec) {
        Ok(env_contract) => env_contract,
        Err(error) => return Some(model_error(error.to_string())),
    };
    // The worker encodes every predict's action against this route's action
    // space (typed predict, D10). It is carried by the EnvSpec in this very
    // handshake, so a missing one dooms every future predict on the route —
    // reject it once, here at setup, instead of failing late on first predict.
    // (observation_space stays optional: a None observation is a valid relay.)
    if env_contract.action_space.is_none() {
        return Some(model_error(
            "route EnvSpec has no action_space; a model worker cannot encode actions without it",
        ));
    }
    // Resolve the route's adapter before storing the config: a failure here
    // fails configuration, so the client never predicts against an unresolved
    // adapter. This runs off the predict-serialization lock (see
    // `ModelRouteSetup`), so configuring one route never blocks on an in-flight
    // predict on another.
    if let Some(route_setup) = route_setup
        && let Err(error) = route_setup.configure_route(&route_key, &env_contract).await
    {
        return Some(model_error_from_error(&error));
    }
    route_configs.lock().await.insert(
        route_key,
        ModelRouteConfig {
            env_contract: Some(Arc::new(env_contract)),
            floor,
        },
    );
    Some(join_response::Kind::ConfigureRoute(
        ConfigureRouteResponse {},
    ))
}

/// Honor the runtime-reconciled three-way floor on the route: the runtime is the
/// binding authority, so a floor naming an edition this model build cannot speak
/// is a hard configuration error, not a silently-downgraded run. Generation is
/// not checked here — it was already gated by equality at the handshake.
///
/// Minimal by design while a single edition exists: it checks equality against
/// this build's edition. The capability intersection is advisory (absence-neutral)
/// and is carried for diagnostics; it never fails configuration here.
fn enforce_route_floor(floor: &RouteFloor) -> std::result::Result<(), String> {
    if floor.selected_workflow_edition != rlmesh_proto::CURRENT_WORKFLOW_EDITION {
        return Err(format!(
            "runtime pinned this route to workflow edition {:?}, which this model build does \
             not implement (implements {:?})",
            floor.selected_workflow_edition,
            rlmesh_proto::CURRENT_WORKFLOW_EDITION,
        ));
    }
    // The active capability set is advisory; absence is semantically neutral, so
    // an unrecognized active capability never fails the route. Touch it so the
    // floor is read in full (and to anchor future per-capability gating).
    let _active_capabilities = &floor.active_capabilities;
    Ok(())
}

/// Everything a single predict needs once its route is resolved and its
/// lifecycle has run: the observation to hand the handler, plus the action
/// space / lane count / route to encode the result against. Split out of
/// [`handle_predict`] so a grouped predict can prepare every group, batch the
/// observations into one [`ModelHandler::predict_grouped`] call, then encode
/// each group's actions against its OWN route's space.
struct PreparedPredict {
    observation: ModelObservation,
    action_space: spaces::SpaceSpec,
    num_envs: usize,
    route: ModelRouteContext,
}

/// Resolve a predict's route config and run its lifecycle — everything up to
/// (but not including) the handler `predict` call. Must run with the handler and
/// active-episodes locks already held: `update_lifecycle` drives the handler's
/// `&mut self` hooks, and the per-route episode map is shared.
async fn prepare_predict_locked<H: ModelHandler>(
    request: PredictRequest,
    handler: &mut H,
    active_episodes: &mut HashMap<(String, i32), String>,
    route_configs: &Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
) -> Result<PreparedPredict> {
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
    // The route runs at the runtime-reconciled floor (authoritative over this
    // model's own handshake). With a single edition this is the build's edition;
    // trace it so the active session value is observable.
    if let Some(floor) = config.floor.as_ref() {
        tracing::trace!(
            selected_workflow_edition = %floor.selected_workflow_edition,
            "predict on route pinned to session floor"
        );
    }
    // The route config no longer carries num_envs (the model gets only the
    // stable EnvSpec); the per-predict lane count comes from the slots, already
    // set by `model_observation_from_endpoint_request`.
    update_lifecycle(handler, active_episodes, &observation).await?;
    // Capture the action space + lane count before predict consumes the obs;
    // the worker owns the typed->wire encode (D10).
    let num_envs = observation.num_envs;
    let action_space = observation
        .env_contract
        .as_ref()
        .and_then(|contract| contract.action_space.clone())
        .ok_or_else(|| Error::model("model route contract missing action space"))?;
    Ok(PreparedPredict {
        observation,
        action_space,
        num_envs,
        route,
    })
}

/// Turn one group's predicted actions into a wire [`PredictResponse`]: enforce
/// the `== num_envs` lane count and structural conformance, then encode against
/// this route's action space.
fn finish_predict(
    actions: Vec<spaces::SpaceValue>,
    num_envs: usize,
    action_space: &spaces::SpaceSpec,
    route: ModelRouteContext,
) -> Result<PredictResponse> {
    if actions.len() != num_envs {
        return Err(Error::model(format!(
            "predict returned {} actions for {num_envs} lanes",
            actions.len()
        )));
    }
    check_actions_conform(action_space, &actions)?;
    let wire = rlmesh_grpc::wire::encode_batched_partial_values(&actions, action_space)
        .map_err(|err| Error::model(err.to_string()))?;
    Ok(model_action_to_endpoint_response(ModelAction {
        action: Some(wire),
        route,
    }))
}

async fn handle_predict<H: ModelHandler + 'static>(
    request: PredictRequest,
    handler: Arc<Mutex<H>>,
    active_episodes: Arc<Mutex<HashMap<(String, i32), String>>>,
    route_configs: Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
) -> Option<join_response::Kind> {
    let result = async {
        let mut handler = handler.lock().await;
        let mut active_episodes = active_episodes.lock().await;
        let prepared =
            prepare_predict_locked(request, &mut *handler, &mut active_episodes, &route_configs)
                .await?;
        let PreparedPredict {
            observation,
            action_space,
            num_envs,
            route,
        } = prepared;
        let actions = handler.predict(observation).await?;
        finish_predict(actions, num_envs, &action_space, route)
    }
    .await;
    Some(match result {
        Ok(response) => join_response::Kind::Predict(response),
        Err(error) => model_error_from_error(&error),
    })
}

/// Process a control-plane-grouped predict: one request carrying N groups, each
/// already routed to its own configured route. Every group's lifecycle/decode
/// runs under one handler-lock acquisition (atomic w.r.t. other predicts); the
/// prepared observations are handed to [`ModelHandler::predict_grouped`] in one
/// batch (the fusion seam — the default fans out sequentially), then each
/// group's actions are encoded against its OWN route's action space. A group
/// that fails to prepare or predict reports its own error in `results[i]` and
/// never sinks the others.
async fn handle_grouped_predict<H: ModelHandler + 'static>(
    request: GroupedPredictRequest,
    handler: Arc<Mutex<H>>,
    active_episodes: Arc<Mutex<HashMap<(String, i32), String>>>,
    route_configs: Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
) -> Option<join_response::Kind> {
    // A finished group is either a prepare-time failure to report verbatim, or a
    // prepared group awaiting its slice of the batched predict's actions.
    enum Finisher {
        Failed(Error),
        Pending {
            num_envs: usize,
            action_space: spaces::SpaceSpec,
            route: ModelRouteContext,
        },
    }

    let mut handler = handler.lock().await;
    let mut active_episodes = active_episodes.lock().await;

    // Prepare every group under the lock (per-route lookup + lifecycle). A group
    // whose route is unconfigured (or otherwise fails to prepare) records its own
    // error and is excluded from the batched predict.
    let mut batch: Vec<ModelObservation> = Vec::with_capacity(request.groups.len());
    let mut finishers: Vec<Finisher> = Vec::with_capacity(request.groups.len());
    for group in request.groups {
        match prepare_predict_locked(group, &mut *handler, &mut active_episodes, &route_configs)
            .await
        {
            Ok(prepared) => {
                let PreparedPredict {
                    observation,
                    action_space,
                    num_envs,
                    route,
                } = prepared;
                batch.push(observation);
                finishers.push(Finisher::Pending {
                    num_envs,
                    action_space,
                    route,
                });
            }
            Err(error) => finishers.push(Finisher::Failed(error)),
        }
    }

    // One batched predict over the prepared observations. The default
    // `predict_grouped` runs them sequentially; a fusing handler overrides it to
    // run a single forward pass. Results align 1:1 and in order with `batch`.
    let mut actions = handler.predict_grouped(batch).await.into_iter();

    let results = finishers
        .into_iter()
        .map(|finisher| {
            let outcome = match finisher {
                Finisher::Failed(error) => Err(error),
                Finisher::Pending {
                    num_envs,
                    action_space,
                    route,
                } => match actions.next() {
                    Some(Ok(actions)) => finish_predict(actions, num_envs, &action_space, route),
                    Some(Err(error)) => Err(error),
                    // A correct `predict_grouped` returns one result per prepared
                    // group; a short Vec is a handler-contract violation.
                    None => Err(Error::model(
                        "predict_grouped returned fewer action sets than prepared groups",
                    )),
                },
            };
            grouped_predict_result(outcome)
        })
        .collect();

    Some(join_response::Kind::GroupedPredict(
        GroupedPredictResponse { results },
    ))
}

/// Wrap one group's outcome into a `GroupedPredictResult`, reusing the single-
/// predict error mapping so a per-group error carries the same code/recoverable
/// flag it would as a standalone predict.
fn grouped_predict_result(outcome: Result<PredictResponse>) -> GroupedPredictResult {
    GroupedPredictResult {
        outcome: Some(match outcome {
            Ok(response) => grouped_predict_result::Outcome::Response(response),
            Err(error) => grouped_predict_result::Outcome::Error(model_error_value(&error)),
        }),
    }
}

/// Per-route tail of completion signals for a single Join stream.
///
/// Each entry maps `session_id:route_id` to a receiver that fires when the most
/// recently enqueued request for that route finishes its handler critical
/// section. The reader task is single-threaded and consults this in arrival
/// order, so it needs no locking.
///
/// # Bounded growth
///
/// A keyed request (`ConfigureRoute` / `Predict` / `CloseRoute`) always replaces
/// its route's tail so the next request on that route, including a
/// `ConfigureRoute` that *reopens* a key after `CloseRoute`, chains correctly
/// after the in-flight predecessor. A `CloseRoute` is the typical last request
/// on a route, so without pruning its fired tail would linger forever and a
/// long-lived stream cycling fresh `session_id:route_id` keys per episode would
/// leak one entry per closed route, growing unboundedly over days. To bound the
/// map, [`RouteTails::next_gate`] reaps every entry whose receiver has already
/// completed (sender fired or dropped) on each call: a completed tail means that
/// route's last request already finished, so a future reopen needs nothing to
/// wait on. The map therefore holds at most the routes with work still in
/// flight (plus any completed-but-not-yet-reaped tails since the last call),
/// independent of how many routes the stream has cycled through.
#[derive(Default)]
struct RouteTails {
    tails: HashMap<String, tokio::sync::oneshot::Receiver<()>>,
}

impl RouteTails {
    fn new() -> Self {
        Self::default()
    }

    /// Compute the gate a keyed (`route_key = Some`) or malformed (`None`)
    /// request must await, and the sender it fires on completion. The request's
    /// fresh tail replaces its route's previous tail; completed tails for *other*
    /// routes are reaped so the map stays bounded over the stream's lifetime.
    fn next_keyed_gate(
        &mut self,
        route_key: Option<&str>,
    ) -> (RequestGate, tokio::sync::oneshot::Sender<()>) {
        let prev = route_key.and_then(|key| self.tails.remove(key));
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        if let Some(key) = route_key {
            self.tails.insert(key.to_string(), done_rx);
        }
        self.reap_completed();
        (RequestGate::Prev(prev), done_tx)
    }

    /// Compute the whole-session `Close` barrier: drain every outstanding route
    /// tail to wait on, then hand back a discarded sender for type uniformity.
    fn close_all_gate(&mut self) -> (RequestGate, tokio::sync::oneshot::Sender<()>) {
        let prev = self.tails.drain().map(|(_, rx)| rx).collect::<Vec<_>>();
        let (done_tx, _done_rx) = tokio::sync::oneshot::channel();
        (RequestGate::All(prev), done_tx)
    }

    /// Drop tails whose receiver has already completed, meaning the sender fired
    /// or was dropped and that route's last enqueued request has finished. A fired
    /// `oneshot::Receiver` resolves immediately, so reaping it never relaxes
    /// ordering: a route reopened after its tail was reaped had its predecessor
    /// already complete, hence nothing left to wait on.
    fn reap_completed(&mut self) {
        self.tails.retain(|_, rx| {
            matches!(
                rx.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            )
        });
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.tails.len()
    }
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
/// context or ids) return `None`. `Close` is handled as an all-routes barrier
/// by the caller, and ungated malformed requests still produce an in-band error.
fn join_request_route_key(request: &JoinRequest) -> Option<String> {
    let context = match request.kind.as_ref()? {
        join_request::Kind::ConfigureRoute(request) => request.context.as_ref()?,
        join_request::Kind::Predict(request) => request.context.as_ref()?,
        join_request::Kind::CloseRoute(request) => request.context.as_ref()?,
        join_request::Kind::Close(_) => return None,
        // A grouped predict spans multiple routes, so it is not pinned to any one
        // route's chain; it runs under the handler Mutex (which serializes it
        // against every other predict) rather than a per-route gate.
        join_request::Kind::GroupedPredict(_) => return None,
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
        ) -> Result<Vec<spaces::SpaceValue>> {
            Ok(Vec::new())
        }
    }

    fn test_server() -> ServedModelServer<NoopModelHandler> {
        ServedModelServer {
            handler: Arc::new(Mutex::new(NoopModelHandler)),
            route_setup: None,
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
            base: Some(rlmesh_proto::core::v1::HandshakeRequest {
                protocol_generation: PROTOCOL_GENERATION.to_string(),
                peer_info: Some(peer_info("rlmesh-model-test-client")),
                capabilities: Default::default(),
                supported_workflow_editions: offered_editions
                    .iter()
                    .map(|edition| edition.to_string())
                    .collect(),
            }),
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

            let base = response.base.expect("handshake response includes base");
            assert!(base.compatible, "offer {offer:?} must be accepted");
            assert_eq!(base.selected_workflow_edition, CURRENT_WORKFLOW_EDITION);
            assert_eq!(
                base.supported_workflow_editions,
                supported_workflow_editions()
            );
        }
    }

    /// Drive a tail through its lifecycle: take the gate, fire the sender, and
    /// confirm the gate it handed out (the *predecessor* of this request) is
    /// already satisfied where expected.
    fn fire(done_tx: tokio::sync::oneshot::Sender<()>) {
        let _ = done_tx.send(());
    }

    #[tokio::test]
    async fn route_tails_reaps_closed_routes_so_the_map_stays_bounded() {
        // A long-lived Join stream cycling fresh `session_id:route_id` keys
        // (ConfigureRoute → Predict → CloseRoute per episode) must not leak one
        // tail entry per closed route. After each episode's requests complete,
        // the next episode's first request reaps the prior fired tails.
        let mut tails = RouteTails::new();

        for episode in 0..1_000 {
            let key = format!("session:{episode}");

            // ConfigureRoute: no predecessor on a fresh key, installs a tail.
            let (gate, configure_done) = tails.next_keyed_gate(Some(&key));
            assert!(matches!(gate, RequestGate::Prev(None)));
            // Predict: gated on ConfigureRoute, replaces the tail.
            let (gate, predict_done) = tails.next_keyed_gate(Some(&key));
            assert!(matches!(gate, RequestGate::Prev(Some(_))));
            // CloseRoute: gated on Predict, replaces the tail once more.
            let (gate, close_done) = tails.next_keyed_gate(Some(&key));
            assert!(matches!(gate, RequestGate::Prev(Some(_))));

            // Requests complete (in handler order) and fire their tails.
            fire(configure_done);
            fire(predict_done);
            fire(close_done);

            // The CloseRoute's fired tail still lingers until the next call reaps
            // it; the map never holds more than this single episode's tail.
            assert!(
                tails.len() <= 1,
                "episode {episode}: route tails grew to {} entries",
                tails.len()
            );
        }

        // After the final episode's CloseRoute fired, one more keyed request on a
        // brand-new route reaps the leftover tail, leaving only the new one.
        let (_gate, _done) = tails.next_keyed_gate(Some("session:final"));
        assert_eq!(
            tails.len(),
            1,
            "only the in-flight route should remain after reaping"
        );
    }

    #[tokio::test]
    async fn route_tails_reopen_after_close_still_sequences() {
        // Reopening a route key after CloseRoute must gate the new ConfigureRoute
        // on the still-in-flight CloseRoute, never overtake it.
        let mut tails = RouteTails::new();
        let key = "session:route";

        let (_g, configure_done) = tails.next_keyed_gate(Some(key));
        let (_g, close_done) = tails.next_keyed_gate(Some(key));
        fire(configure_done);

        // CloseRoute is still in flight (close_done not fired): a reopening
        // ConfigureRoute on the same key must chain after it.
        let (reopen_gate, _reopen_done) = tails.next_keyed_gate(Some(key));
        let mut reopen_prev = match reopen_gate {
            RequestGate::Prev(Some(rx)) => rx,
            RequestGate::Prev(None) => {
                panic!("reopen must gate on the in-flight CloseRoute, got an ungated request")
            }
            RequestGate::All(_) => panic!("a keyed request must never produce an All gate"),
        };

        // The reopen gate is unresolved until CloseRoute completes.
        assert!(matches!(
            reopen_prev.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        // Once CloseRoute fires, the reopen gate resolves and the reopen proceeds.
        fire(close_done);
        assert!(reopen_prev.try_recv().is_ok());
    }

    #[tokio::test]
    async fn route_tails_reopen_after_reaped_close_is_ungated() {
        // If the CloseRoute already completed *and* was reaped, a reopen on the
        // same key has nothing to wait on; ordering is still correct because the
        // predecessor genuinely finished.
        let mut tails = RouteTails::new();
        let key = "session:route";

        let (_g, configure_done) = tails.next_keyed_gate(Some(key));
        let (_g, close_done) = tails.next_keyed_gate(Some(key));
        fire(configure_done);
        fire(close_done);
        // A keyed request on another route triggers reaping of the fired tail.
        let (_g, _d) = tails.next_keyed_gate(Some("other:route"));

        // Reopen the original key: its tail was reaped, so no predecessor gate.
        let (reopen_gate, _d) = tails.next_keyed_gate(Some(key));
        assert!(
            matches!(reopen_gate, RequestGate::Prev(None)),
            "reaped-then-reopened route should be ungated"
        );
    }

    #[tokio::test]
    async fn route_tails_close_drains_every_route() {
        // Whole-session Close is a barrier over all outstanding routes and clears
        // the map regardless of how many routes are in flight.
        let mut tails = RouteTails::new();
        let (_g, _d0) = tails.next_keyed_gate(Some("a:1"));
        let (_g, _d1) = tails.next_keyed_gate(Some("b:1"));
        assert_eq!(tails.len(), 2);

        let (gate, _close_done) = tails.close_all_gate();
        match gate {
            RequestGate::All(prev) => assert_eq!(prev.len(), 2),
            RequestGate::Prev(_) => panic!("Close must produce an All gate over every route"),
        }
        assert_eq!(tails.len(), 0, "Close must clear every route tail");
    }

    #[tokio::test]
    async fn close_tears_down_every_route_config() {
        // Whole-session Close drains all episodes globally, so it must also
        // clear every route's config rather than leaking it for the server's
        // lifetime (a CloseRoute only tears down its own route).
        let server = test_server();
        {
            let mut configs = server.route_configs.lock().await;
            for key in ["session:a", "session:b"] {
                configs.insert(
                    key.to_string(),
                    ModelRouteConfig {
                        env_contract: None,
                        floor: None,
                    },
                );
            }
        }

        let response = handle_model_request(
            JoinRequest {
                request_id: "close-1".to_string(),
                kind: Some(join_request::Kind::Close(
                    rlmesh_proto::model::v1::CloseParticipantRequest::default(),
                )),
            },
            Arc::clone(&server.handler),
            server.route_setup.clone(),
            Arc::clone(&server.active_episodes),
            Arc::clone(&server.route_configs),
        )
        .await;

        assert!(matches!(response.kind, Some(join_response::Kind::Close(_))));
        assert!(
            server.route_configs.lock().await.is_empty(),
            "Close must clear every route config"
        );
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

            let base = response.base.expect("handshake response includes base");
            assert!(!base.compatible, "offer {offer:?} must be rejected");
            assert!(base.error_message.contains("workflow edition"));
            if offer.is_empty() {
                assert!(base.error_message.contains("predate edition negotiation"));
            }
            assert!(base.selected_workflow_edition.is_empty());
        }
    }
}
