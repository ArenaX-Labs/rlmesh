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
    CloseParticipantResponse, GroupedPredictRequest, GroupedPredictResponse, GroupedPredictResult,
    HandshakeRequest, HandshakeResponse, JoinRequest, JoinResponse, PredictRequest, PredictResponse,
    ReleaseAdapterResponse, ResetAdapterResponse, ResolveAdapterRequest, ResolveAdapterResponse,
    ShutdownRequest, ShutdownResponse, grouped_predict_result, join_request, join_response,
    model_service_server::{ModelService as ModelServiceTrait, ModelServiceServer},
};
use rlmesh_proto::{
    capabilities, capability_map, evaluate_handshake, generation_mismatch_message, peer_info,
    supported_workflow_editions,
};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use super::handler::{ModelHandler, ModelRouteSetup, PredictFrames};
use super::types::{ModelObservation, ModelRouteContext};
use super::wire::{
    ModelAction, check_actions_conform, encode_replay_frames, model_action_to_endpoint_response,
    model_endpoint_total_ns, model_error, model_error_from_error, model_error_value,
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
    route_configs: Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
    token: String,
    activity_tx: Option<mpsc::UnboundedSender<IdleActivity>>,
    shutdown: rlmesh_grpc::lifecycle::ShutdownTrigger,
    serve_options: ServeOptions,
}

#[derive(Debug, Clone)]
pub(super) struct ModelRouteConfig {
    pub(super) env_contract: Option<Arc<spaces::EnvContract>>,
    /// The env edition the runtime pinned via `ResolveAdapter` (the floor — the
    /// highest edition env, model, and runtime all support). Authoritative over the
    /// model's own (pairwise) handshake result. Stored/logged here; full enforcement
    /// is minimal while the build holds a single generation and edition.
    pub(super) floor: Option<RouteFloor>,
}

/// Edition the runtime pinned an env to (from `ResolveAdapterRequest`).
/// Generation is gated by equality at the handshake; capabilities are advisory
/// and pairwise, so neither is part of the env floor.
#[derive(Debug, Clone)]
pub(super) struct RouteFloor {
    pub(super) selected_workflow_edition: String,
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
        // The handshake decides ONE thing: protocol generation. The edition is the
        // runtime's call (the floor); a generation-ok peer is compatible regardless
        // of editions and fails later at the floor if there is no mutual one.
        let compatible = evaluate_handshake(&request.protocol_generation);
        Ok(Response::new(HandshakeResponse {
            base: Some(rlmesh_proto::core::v1::HandshakeResponse {
                compatible,
                peer_info: Some(peer_info("rlmesh-model")),
                error_message: (!compatible)
                    .then(|| generation_mismatch_message(&request.protocol_generation)),
                capabilities: capability_map(&[
                    // The served model pipelines Join-stream requests (see `join`).
                    // Advisory: lets clients detect that overlapping predicts will
                    // actually pipeline rather than serialize behind the handler.
                    capabilities::MODEL_CONCURRENT_PREDICT_V1,
                ]),
                supported_workflow_editions: supported_workflow_editions(),
            }),
            // No model contract advertised yet; the field exists for a future
            // model-side spec but has no native source today.
            model_contract: None,
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
                let (gate, dones): (RequestGate, Vec<tokio::sync::oneshot::Sender<()>>) =
                    if close_after {
                        // Close drains every route: wait for all outstanding requests.
                        route_tails.close_all_gate()
                    } else if let Some(keys) = grouped_predict_route_keys(&request) {
                        // A grouped predict spans multiple routes, so it can't ride a
                        // single route's chain: gate it on every route it references
                        // (and register a tail on each so a later same-route request
                        // orders after it). Without this it would run ungated and
                        // could race a route's `ConfigureRoute`/`CloseRoute`.
                        route_tails.next_multi_keyed_gate(&keys)
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
                let route_configs = Arc::clone(&route_configs);
                let tx = tx.clone();

                tokio::spawn(async move {
                    let _permit = permit;
                    let _activity_guard = activity_guard;
                    // Wait for predecessors so the handler critical section runs in
                    // per-route arrival order (or, for Close, after every route).
                    gate.wait().await;

                    let response =
                        handle_model_request(request, handler, route_setup, route_configs).await;

                    // Release successors on this request's route(s) *before* sending
                    // the response, so per-route ordering does not depend on the
                    // unbounded response channel draining. A grouped predict fires
                    // one signal per referenced route.
                    for done in dones {
                        let _ = done.send(());
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
    route_configs: Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
) -> JoinResponse {
    let request_id = request.request_id.clone();
    let started_at = Instant::now();

    let kind = match request.kind {
        Some(join_request::Kind::ResolveAdapter(request)) => {
            handle_resolve_adapter(request, route_setup.as_deref(), route_configs).await
        }
        Some(join_request::Kind::Predict(request)) => {
            handle_predict(request, handler, route_configs).await
        }
        Some(join_request::Kind::GroupedPredict(request)) => {
            handle_grouped_predict(request, handler, route_configs).await
        }
        Some(join_request::Kind::ResetAdapter(request)) => {
            // Explicit GC (R2): evict the named episodes' per-episode adapter
            // state. Empty episode_ids means evict all of this env's state.
            let env_id = request.context.as_ref().map(|context| context.env_id.clone());
            match env_id.filter(|env_id| !env_id.is_empty()) {
                Some(env_id) => {
                    let result = handler
                        .lock()
                        .await
                        .reset_adapter(&env_id, request.episode_ids)
                        .await;
                    match result {
                        Ok(()) => Some(join_response::Kind::ResetAdapter(ResetAdapterResponse {})),
                        Err(error) => Some(model_error_from_error(&error)),
                    }
                }
                None => Some(model_error("reset_adapter missing env_id")),
            }
        }
        Some(join_request::Kind::ReleaseAdapter(request)) => {
            let env_id = request.context.as_ref().and_then(route_config_key);
            match env_id {
                Some(env_id) => {
                    // ReleaseAdapter removes the adapter entirely (implies
                    // reset-all): drop the config and tear down its setup.
                    route_configs.lock().await.remove(&env_id);
                    if let Some(route_setup) = route_setup.as_deref()
                        && let Err(error) = route_setup.release_adapter(&env_id).await
                    {
                        return JoinResponse {
                            kind: Some(model_error(error.to_string())),
                            endpoint_total_ns: Some(model_endpoint_total_ns(started_at)),
                            request_id,
                        };
                    }
                    Some(join_response::Kind::ReleaseAdapter(ReleaseAdapterResponse {}))
                }
                _ => Some(model_error("release_adapter missing env_id")),
            }
        }
        Some(join_request::Kind::Close(_request)) => {
            // Tear down every env's adapter rather than leaking them for the
            // server's lifetime. Drop the configs guard before the async
            // release_adapter calls; re-lock only to clear.
            let env_ids: Vec<String> = route_configs.lock().await.keys().cloned().collect();
            if let Some(route_setup) = route_setup.as_deref() {
                for env_id in &env_ids {
                    if let Err(error) = route_setup.release_adapter(env_id).await {
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

async fn handle_resolve_adapter(
    request: ResolveAdapterRequest,
    route_setup: Option<&dyn ModelRouteSetup>,
    route_configs: Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
) -> Option<join_response::Kind> {
    let route = match request.context {
        Some(route) => route,
        None => return Some(model_error("resolve_adapter missing adapter context")),
    };
    let env_id = route.env_id.clone();
    if env_id.is_empty() {
        return Some(model_error("resolve_adapter env_id is empty"));
    }
    let route_key = match route_config_key(&route) {
        Some(route_key) => route_key,
        None => return Some(model_error("resolve_adapter missing env_id")),
    };
    let env_spec = match request.env_spec {
        Some(env_spec) => env_spec,
        None => return Some(model_error("resolve_adapter missing env_spec")),
    };
    // The runtime pins the env edition (the floor — highest mutual across env,
    // model, and runtime), authoritative over this model's own (pairwise, possibly
    // higher) handshake result.
    let floor = if request.selected_workflow_edition.is_empty() {
        // Empty pin: a legacy/older runtime that predates edition propagation.
        // Harmless while a single edition exists (the floor could only be CURRENT).
        // Once the support window grows, an unpinned env could silently run newest
        // semantics, so reject it then — production must select the floor.
        if rlmesh_proto::SUPPORTED_WORKFLOW_EDITIONS.len() > 1 {
            return Some(model_error(
                "resolve_adapter arrived without a workflow edition pin; the runtime must \
                 select the session floor once more than one edition is supported",
            ));
        }
        None
    } else {
        let floor = RouteFloor {
            selected_workflow_edition: request.selected_workflow_edition,
        };
        // Minimal enforcement: the pinned edition is authoritative over this
        // model's own handshake. Reject an env the runtime pinned to an edition
        // this build cannot drive, instead of silently running the wrong semantics.
        // (With a single edition this never fires; the path exists so the edition
        // is honored, not merely logged, once a second edition lands.)
        if let Err(error) = enforce_route_floor(&floor) {
            return Some(model_error(error));
        }
        tracing::debug!(
            env_id = %env_id,
            selected_workflow_edition = %floor.selected_workflow_edition,
            "model adapter pinned to runtime-selected edition"
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
            "env EnvSpec has no action_space; a model worker cannot encode actions without it",
        ));
    }
    // Resolve the env's adapter before storing the config: a failure here fails
    // resolution, so the client never predicts against an unresolved adapter.
    // This runs off the predict-serialization lock (see `ModelRouteSetup`), so
    // resolving one env's adapter never blocks on an in-flight predict on another.
    // Runtime-chosen replay horizon pinned on this env (1 = no chunking).
    let action_horizon = request.action_horizon;
    if let Some(route_setup) = route_setup
        && let Err(error) = route_setup
            .resolve_adapter(&route_key, &env_contract, action_horizon)
            .await
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
    Some(join_response::Kind::ResolveAdapter(
        ResolveAdapterResponse {},
    ))
}

/// Honor the runtime-pinned route edition: the runtime selects it (the floor —
/// highest mutual across env, model, and runtime) and it is authoritative, so an
/// edition this model build cannot drive is a hard configuration error, not a
/// silently-downgraded run. Generation is not checked here — it was already gated
/// by equality at the handshake.
///
/// Checks **membership** in the support window, not equality with `CURRENT`: the
/// floor is deliberately allowed to land on a supported older edition (a graceful
/// downgrade), so equality would reject a route this build can actually drive.
fn enforce_route_floor(floor: &RouteFloor) -> std::result::Result<(), String> {
    if !rlmesh_proto::is_supported_edition(&floor.selected_workflow_edition) {
        return Err(format!(
            "runtime pinned this route to workflow edition {:?}, which this model build does \
             not implement (implements {:?})",
            floor.selected_workflow_edition,
            rlmesh_proto::SUPPORTED_WORKFLOW_EDITIONS,
        ));
    }
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

/// Resolve a predict's adapter config — everything up to (but not including) the
/// handler `predict` call. No lifecycle hooks run here anymore (R2): per-episode
/// state is lazy-seeded on first predict and evicted via `reset_adapter`, so
/// there is no position-diff and no shared active-episodes map. The handler lock
/// is held by the caller across prepare + predict for per-handler ordering.
async fn prepare_predict_locked(
    request: PredictRequest,
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
        .ok_or_else(|| Error::model("model env adapter was not resolved"))?;
    observation.env_contract = config.env_contract;
    // The adapter runs at the runtime-reconciled floor (authoritative over this
    // model's own handshake). With a single edition this is the build's edition;
    // trace it so the active session value is observable.
    if let Some(floor) = config.floor.as_ref() {
        tracing::trace!(
            selected_workflow_edition = %floor.selected_workflow_edition,
            "predict on adapter pinned to session floor"
        );
    }
    // The adapter config no longer carries num_envs (the model gets only the
    // stable EnvSpec); the per-predict row count comes from episode_ids, already
    // set by `model_observation_from_endpoint_request`.
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

/// Turn one group's predicted frames into a wire [`PredictResponse`]: enforce the
/// `== num_envs` lane count and structural conformance on frame 0, then encode it
/// (the action for this step) plus any chunk replay frames against this route's
/// action space, into the single ordered `actions` list (`actions[0]` is this
/// step, `actions[1..]` the replay frames).
fn finish_predict(
    frames: PredictFrames,
    num_envs: usize,
    action_space: &spaces::SpaceSpec,
    route: ModelRouteContext,
) -> Result<PredictResponse> {
    let PredictFrames { actions, replay } = frames;
    if actions.len() != num_envs {
        return Err(Error::model(format!(
            "predict returned {} actions for {num_envs} lanes",
            actions.len()
        )));
    }
    check_actions_conform(action_space, &actions)?;
    let frame0 = rlmesh_grpc::wire::encode_batched_partial_values(&actions, action_space)
        .map_err(|err| Error::model(err.to_string()))?;
    let mut wire_actions = Vec::with_capacity(1 + replay.len());
    wire_actions.push(frame0);
    wire_actions.extend(encode_replay_frames(&replay, num_envs, action_space)?);
    Ok(model_action_to_endpoint_response(ModelAction {
        actions: wire_actions,
        route,
    }))
}

async fn handle_predict<H: ModelHandler + 'static>(
    request: PredictRequest,
    handler: Arc<Mutex<H>>,
    route_configs: Arc<Mutex<HashMap<String, ModelRouteConfig>>>,
) -> Option<join_response::Kind> {
    let result = async {
        let mut handler = handler.lock().await;
        let prepared = prepare_predict_locked(request, &route_configs).await?;
        let PreparedPredict {
            observation,
            action_space,
            num_envs,
            route,
        } = prepared;
        let frames = handler.predict_chunked(observation).await?;
        finish_predict(frames, num_envs, &action_space, route)
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

    // Prepare every group under the lock (per-env adapter lookup). A group whose
    // env adapter is unresolved (or otherwise fails to prepare) records its own
    // error and is excluded from the batched predict.
    let mut batch: Vec<ModelObservation> = Vec::with_capacity(request.groups.len());
    let mut finishers: Vec<Finisher> = Vec::with_capacity(request.groups.len());
    for group in request.groups {
        match prepare_predict_locked(group, &route_configs).await {
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
                    // Grouped predict does not chunk: each group yields one action
                    // per lane (no replay frames).
                    Some(Ok(actions)) => finish_predict(
                        PredictFrames {
                            actions,
                            replay: Vec::new(),
                        },
                        num_envs,
                        &action_space,
                        route,
                    ),
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
    ) -> (RequestGate, Vec<tokio::sync::oneshot::Sender<()>>) {
        let prev = route_key.and_then(|key| self.tails.remove(key));
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        if let Some(key) = route_key {
            self.tails.insert(key.to_string(), done_rx);
        }
        self.reap_completed();
        (RequestGate::Prev(prev), vec![done_tx])
    }

    /// Compute the gate a grouped predict must await: it references several routes
    /// at once, so wait on the latest outstanding request of EACH referenced route
    /// and register a fresh tail on each so a later same-route request (e.g. a
    /// `CloseRoute`) chains after it. `keys` must be deduplicated. Mirrors
    /// `next_keyed_gate` but fans the single completion out to every route.
    fn next_multi_keyed_gate(
        &mut self,
        keys: &[String],
    ) -> (RequestGate, Vec<tokio::sync::oneshot::Sender<()>>) {
        let mut prev = Vec::with_capacity(keys.len());
        let mut dones = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some(rx) = self.tails.remove(key) {
                prev.push(rx);
            }
            let (done_tx, done_rx) = tokio::sync::oneshot::channel();
            self.tails.insert(key.clone(), done_rx);
            dones.push(done_tx);
        }
        self.reap_completed();
        (RequestGate::All(prev), dones)
    }

    /// Compute the whole-session `Close` barrier: drain every outstanding route
    /// tail to wait on. `Close` ends the session, so it registers no tail of its
    /// own (there are no successors to release).
    fn close_all_gate(&mut self) -> (RequestGate, Vec<tokio::sync::oneshot::Sender<()>>) {
        let prev = self.tails.drain().map(|(_, rx)| rx).collect::<Vec<_>>();
        (RequestGate::All(prev), Vec::new())
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
        join_request::Kind::ResolveAdapter(request) => request.context.as_ref()?,
        join_request::Kind::Predict(request) => request.context.as_ref()?,
        join_request::Kind::ResetAdapter(request) => request.context.as_ref()?,
        join_request::Kind::ReleaseAdapter(request) => request.context.as_ref()?,
        join_request::Kind::Close(_) => return None,
        // A grouped predict spans multiple envs, so it has no single key; the
        // dispatch loop gates it on all of its referenced envs via
        // `grouped_predict_route_keys` + `next_multi_keyed_gate` instead.
        join_request::Kind::GroupedPredict(_) => return None,
    };
    route_config_key(context)
}

/// The deduplicated route keys a grouped predict references, in first-seen order,
/// or `None` for any other request kind. A group with a missing/malformed context
/// contributes no key (it reports its own in-band error during handling), so an
/// empty `Vec` means there is nothing to gate on.
fn grouped_predict_route_keys(request: &JoinRequest) -> Option<Vec<String>> {
    let join_request::Kind::GroupedPredict(grouped) = request.kind.as_ref()? else {
        return None;
    };
    let mut keys = Vec::new();
    for group in &grouped.groups {
        if let Some(key) = group.context.as_ref().and_then(route_config_key)
            && !keys.contains(&key)
        {
            keys.push(key);
        }
    }
    Some(keys)
}

fn route_config_key(context: &rlmesh_proto::model::v1::AdapterContext) -> Option<String> {
    if context.env_id.is_empty() {
        return None;
    }
    // env_id is globally unique (UUIDv7), so it alone keys the adapter.
    Some(context.env_id.clone())
}

fn model_route_config_key(route: &super::types::ModelRouteContext) -> String {
    route.env_id.clone()
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
                protocol_generation: rlmesh_proto::PROTOCOL_GENERATION.to_string(),
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
            assert_eq!(
                base.supported_workflow_editions,
                supported_workflow_editions()
            );
        }
    }

    /// Drive a tail through its lifecycle: take the gate, fire the sender, and
    /// confirm the gate it handed out (the *predecessor* of this request) is
    /// already satisfied where expected.
    fn fire(dones: Vec<tokio::sync::oneshot::Sender<()>>) {
        for done in dones {
            let _ = done.send(());
        }
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
    async fn route_tails_grouped_predict_gates_each_route_and_chains_successors() {
        // A grouped predict referencing routes a and b must (1) wait on the
        // in-flight request of each, and (2) install a tail on each so a later
        // same-route request chains after it instead of racing it (the bug: a
        // grouped predict used to run ungated, racing a route's configure/close).
        let mut tails = RouteTails::new();
        let (_g, _a_prev) = tails.next_keyed_gate(Some("s:a"));
        let (_g, _b_prev) = tails.next_keyed_gate(Some("s:b"));

        let (gate, grouped_done) =
            tails.next_multi_keyed_gate(&["s:a".to_owned(), "s:b".to_owned()]);
        match gate {
            RequestGate::All(prev) => assert_eq!(prev.len(), 2, "must gate on both routes"),
            RequestGate::Prev(_) => panic!("a grouped predict must produce an All gate"),
        }

        // A CloseRoute on route a arriving after the grouped predict must chain
        // behind it (the grouped predict replaced a's tail), not overtake it.
        let (close_gate, _close_done) = tails.next_keyed_gate(Some("s:a"));
        let mut close_prev = match close_gate {
            RequestGate::Prev(Some(rx)) => rx,
            _ => panic!("CloseRoute after a grouped predict must gate on it, not run ungated"),
        };
        assert!(matches!(
            close_prev.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        // Completing the grouped predict fires its tail on every referenced route,
        // releasing the CloseRoute.
        fire(grouped_done);
        assert!(close_prev.try_recv().is_ok());
    }

    #[test]
    fn grouped_predict_route_keys_dedups_referenced_routes() {
        let group = |env_id: &str| PredictRequest {
            context: Some(rlmesh_proto::model::v1::AdapterContext {
                session_id: "s".to_owned(),
                env_id: env_id.to_owned(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let grouped = JoinRequest {
            kind: Some(join_request::Kind::GroupedPredict(GroupedPredictRequest {
                groups: vec![group("a"), group("b"), group("a")],
            })),
            ..Default::default()
        };
        assert_eq!(
            grouped_predict_route_keys(&grouped),
            Some(vec!["a".to_owned(), "b".to_owned()])
        );
        // A non-grouped request carries a single route key, so this returns None.
        let predict = JoinRequest {
            kind: Some(join_request::Kind::Predict(group("a"))),
            ..Default::default()
        };
        assert_eq!(grouped_predict_route_keys(&predict), None);
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
    async fn handshake_accepts_any_generation_compatible_offer() {
        // The handshake decides ONE thing: protocol generation. The edition is the
        // runtime's call (the floor), so a generation-ok client is compatible even
        // with no mutual edition — it fails later at the floor, not here.
        let server = test_server();

        for offer in [&[][..], &["2026"][..], &["2026.11", "next"][..]] {
            let response =
                ModelServiceTrait::handshake(&server, Request::new(handshake_request(offer)))
                    .await
                    .unwrap()
                    .into_inner();

            let base = response.base.expect("handshake response includes base");
            assert!(
                base.compatible,
                "generation-ok offer {offer:?} is compatible"
            );
            assert!(base.error_message.is_none());
        }
    }
}
