//! Tonic transport server for the EnvService.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use prost::Message;
use rlmesh_proto::spaces::v1::meta_value::Kind as MetaKind;
use rlmesh_proto::spaces::v1::{MetaMap, MetaValue};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

use crate::env::Environment;
use crate::env::episode::EpisodeTracker;
use crate::error::EnvError;
use crate::lifecycle::{ActivityFinishedGuard, IdleActivity, ServeOptions, ShutdownTrigger};
use crate::wire::spaces::env_contract_to_proto;
use crate::wire::value_bytes_ref;
use rlmesh_spaces::AutoresetMode;

use rlmesh_proto::core::v1::{OperationMetric, OperationTelemetry, operation_metric};
use rlmesh_proto::env::v1::{
    CloseResponse, EnvError as ProtoEnvError, EnvErrorCode as ProtoEnvErrorCode, HandshakeRequest,
    HandshakeResponse, JoinRequest, JoinResponse, ShutdownRequest, ShutdownResponse,
    env_service_server::EnvService, join_request, join_response,
};
use rlmesh_proto::{
    MIN_SUPPORTED_PROTOCOL_GENERATION, PROTOCOL_GENERATION, capabilities, capability_map,
    negotiate_workflow_edition, supported_workflow_editions,
};

use super::{env_error_to_proto, is_protocol_generation_compatible};

/// Run an environment operation under a deadline, then drain it before the next
/// request may access the same environment.
///
/// Python-backed environments are not generally cancellable. On timeout this
/// keeps polling the operation to completion while the caller still holds the
/// environment mutex, preventing overlapping access to the wrapped environment.
async fn run_env_op_with_deadline<F, T>(
    op: F,
    timeout_ms: i64,
    operation: &str,
) -> Result<T, EnvError>
where
    F: std::future::Future<Output = Result<T, EnvError>>,
{
    if timeout_ms <= 0 {
        return op.await;
    }

    let timeout_duration = Duration::from_millis(timeout_ms as u64);
    tokio::pin!(op);

    match tokio::time::timeout(timeout_duration, op.as_mut()).await {
        Ok(result) => result,
        Err(_) => {
            // Deadline elapsed. The underlying env op cannot be cancelled, so we
            // must drive it to completion before releasing the env mutex; only
            // then is it safe for the next request to access the env.
            tracing::warn!(
                operation,
                timeout_ms,
                "{operation} exceeded {timeout_ms}ms deadline; draining the in-flight \
                 operation to completion before serving the next request to avoid concurrent \
                 access to the environment"
            );
            match op.await {
                Ok(_) => tracing::warn!(
                    operation,
                    "{operation} completed after its deadline; result discarded (client already \
                     received a timeout error)"
                ),
                Err(error) => tracing::warn!(
                    operation,
                    error = %error,
                    "{operation} failed after its deadline; error discarded (client already \
                     received a timeout error)"
                ),
            }
            Err(EnvError::new(
                crate::error::EnvErrorCode::Timeout,
                format!("{operation} timed out after {timeout_ms}ms"),
            ))
        }
    }
}

/// A transport server that implements the `EnvService` tonic trait.
pub struct GrpcEnvServer<E: Environment> {
    env: Arc<Mutex<E>>,
    episode_tracker: Arc<Mutex<EpisodeTracker>>,
    shutdown: ShutdownTrigger,
    serve_options: ServeOptions,
    /// Optional bearer token required on the `authorization` metadata header.
    /// Empty means authentication is disabled.
    token: String,
    activity_tx: Option<mpsc::UnboundedSender<IdleActivity>>,
    /// Whether a Join stream is currently active. The env protocol has no
    /// session identity and a single env / episode tracker is shared across all
    /// streams, so concurrent Join streams would interleave reset/step on the
    /// same non-thread-safe environment and one client's Close would complete
    /// every other client's episodes. We therefore admit only one Join stream at
    /// a time and reject the rest until the active one ends.
    join_active: Arc<std::sync::atomic::AtomicBool>,
}

/// RAII guard that releases the single-Join-stream slot when dropped.
struct JoinSlotGuard(Arc<std::sync::atomic::AtomicBool>);

impl Drop for JoinSlotGuard {
    fn drop(&mut self) {
        self.0.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

impl<E: Environment> GrpcEnvServer<E> {
    /// Create a new environment server wrapping the given environment.
    pub fn new(env: E) -> Self {
        Self::new_with_options(env, ShutdownTrigger::new(), ServeOptions::default(), None)
    }

    #[doc(hidden)]
    pub fn new_with_options(
        env: E,
        shutdown: ShutdownTrigger,
        serve_options: ServeOptions,
        activity_tx: Option<mpsc::UnboundedSender<IdleActivity>>,
    ) -> Self {
        Self::from_shared(
            Arc::new(Mutex::new(env)),
            shutdown,
            serve_options,
            activity_tx,
        )
    }

    #[doc(hidden)]
    pub fn from_shared(
        env: Arc<Mutex<E>>,
        shutdown: ShutdownTrigger,
        serve_options: ServeOptions,
        activity_tx: Option<mpsc::UnboundedSender<IdleActivity>>,
    ) -> Self {
        let token = serve_options.token.clone().unwrap_or_default();
        Self {
            env,
            episode_tracker: Arc::new(Mutex::new(EpisodeTracker::new())),
            shutdown,
            serve_options,
            token,
            activity_tx,
            join_active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Reject the request when a token is configured and the request's
    /// `authorization` metadata does not match it. Mirrors the model service's
    /// bearer-token check. A no-op when no token is configured.
    fn authenticate<T>(&self, request: &Request<T>) -> Result<(), Status> {
        let provided = request
            .metadata()
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        if crate::helpers::bearer_token_matches(&self.token, provided) {
            Ok(())
        } else {
            Err(Status::unauthenticated("invalid env token"))
        }
    }
}

pub fn env_service<E: Environment + 'static>(
    env: E,
) -> rlmesh_proto::env::v1::env_service_server::EnvServiceServer<GrpcEnvServer<E>> {
    rlmesh_proto::env::v1::env_service_server::EnvServiceServer::new(GrpcEnvServer::new(env))
        .max_decoding_message_size(crate::MAX_MESSAGE_SIZE)
        .max_encoding_message_size(crate::MAX_MESSAGE_SIZE)
}

#[doc(hidden)]
pub fn env_service_from_shared<E: Environment + 'static>(
    env: Arc<Mutex<E>>,
    shutdown: ShutdownTrigger,
    serve_options: ServeOptions,
    activity_tx: Option<mpsc::UnboundedSender<IdleActivity>>,
) -> rlmesh_proto::env::v1::env_service_server::EnvServiceServer<GrpcEnvServer<E>> {
    rlmesh_proto::env::v1::env_service_server::EnvServiceServer::new(GrpcEnvServer::from_shared(
        env,
        shutdown,
        serve_options,
        activity_tx,
    ))
    .max_decoding_message_size(crate::MAX_MESSAGE_SIZE)
    .max_encoding_message_size(crate::MAX_MESSAGE_SIZE)
}

#[tonic::async_trait]
impl<E: Environment + 'static> EnvService for GrpcEnvServer<E> {
    #[tracing::instrument(
        name = "rlmesh.grpc.server.handshake",
        skip_all,
        fields(
            client_name = %request.get_ref().client_name,
            client_version = %request.get_ref().client_version
        )
    )]
    async fn handshake(
        &self,
        request: Request<HandshakeRequest>,
    ) -> Result<Response<HandshakeResponse>, Status> {
        self.authenticate(&request)?;
        let req = request.into_inner();

        tracing::info!(
            "Handshake from {} v{} (protocol {}, offered editions [{}])",
            req.client_name,
            req.client_version,
            req.protocol_generation,
            req.supported_workflow_editions.join(", ")
        );

        let protocol_compatible =
            is_protocol_generation_compatible(&req.protocol_generation, PROTOCOL_GENERATION);
        let selected_edition = negotiate_workflow_edition(&req.supported_workflow_editions);
        let compatible = protocol_compatible && selected_edition.is_some();

        let env_contract = if compatible {
            let env = self.env.lock().await;
            let mut contract = env_contract_to_proto(env.env_contract());
            contract.num_envs = env.num_envs() as u32;
            Some(contract)
        } else {
            None
        };

        let res = HandshakeResponse {
            compatible,
            server_protocol_generation: PROTOCOL_GENERATION.to_string(),
            min_supported_protocol_generation: MIN_SUPPORTED_PROTOCOL_GENERATION.to_string(),
            error_message: if compatible {
                String::new()
            } else if !protocol_compatible {
                format!(
                    "protocol generation {} not compatible with server {}",
                    req.protocol_generation, PROTOCOL_GENERATION
                )
            } else if req.supported_workflow_editions.is_empty() {
                format!(
                    "client offered no workflow editions (clients from 0.1.0-beta.2 or older predate edition negotiation and are not supported); server supports [{}]",
                    supported_workflow_editions().join(", ")
                )
            } else {
                format!(
                    "no mutually supported workflow edition; client offered [{}], server supports [{}]",
                    req.supported_workflow_editions.join(", "),
                    supported_workflow_editions().join(", ")
                )
            },
            capabilities: capability_map(&[
                capabilities::ENV_SERVICE_V1,
                capabilities::SPACES_CORE_V1,
            ]),
            env_contract,
            selected_workflow_edition: if compatible {
                selected_edition.unwrap_or_default().to_string()
            } else {
                String::new()
            },
            supported_workflow_editions: supported_workflow_editions(),
        };

        Ok(Response::new(res))
    }

    type JoinStream = ReceiverStream<Result<JoinResponse, Status>>;

    async fn join(
        &self,
        request: Request<Streaming<JoinRequest>>,
    ) -> Result<Response<Self::JoinStream>, Status> {
        self.authenticate(&request)?;
        // Reject a second concurrent Join stream: the env protocol carries no
        // session identity and the env + episode tracker are shared, so two
        // streams cannot safely coexist.
        if self
            .join_active
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
            )
            .is_err()
        {
            tracing::warn!("rejecting Join stream: environment already has an active session");
            return Err(Status::failed_precondition(
                "environment already has an active Join session; only one client may be connected \
                 at a time",
            ));
        }
        let join_slot = JoinSlotGuard(self.join_active.clone());

        let mut req_stream = request.into_inner();
        let env = self.env.clone();
        let episode_tracker = self.episode_tracker.clone();
        let activity_tx = self.activity_tx.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<JoinResponse, Status>>(64);

        tokio::spawn(async move {
            // Hold the slot guard for the lifetime of this stream; dropping it
            // (on normal completion, error, or task cancellation) frees the slot.
            let _join_slot = join_slot;
            while let Some(req_result) = req_stream.next().await {
                let req = match req_result {
                    Ok(req) => req,
                    Err(e) => {
                        tracing::debug!("join stream closed: {}", e);
                        break;
                    }
                };

                let close_after = matches!(req.kind, Some(join_request::Kind::Close(_)));
                if let Some(activity_tx) = &activity_tx {
                    let _ = activity_tx.send(IdleActivity::Started);
                }
                // RAII pairing: the matching Finished must fire even if the
                // served environment's reset/step/render panics and unwinds out
                // of this loop, or the idle-shutdown in-flight count stays
                // elevated forever and idle shutdown never fires. The guard's
                // scope ends at the bottom of this loop iteration (request done).
                let res = {
                    let _activity_guard = ActivityFinishedGuard::new(activity_tx.clone());
                    handle_env_request(req, env.clone(), episode_tracker.clone()).await
                };

                let send_result = tx.send(Ok(res)).await;

                if send_result.is_err() {
                    tracing::warn!(
                        "env join response receiver closed before response could be delivered"
                    );
                    break;
                }

                if close_after {
                    break;
                }
            }

            // The session is over, however it ended. A graceful Close already
            // drained the tracker; an abrupt end (client drop/detach/network)
            // must not leak this session's episodes into the next session's
            // accounting, so complete anything still active as truncated. The
            // metadata has no recipient (the stream is gone) and is dropped.
            let leftover = episode_tracker.lock().await.complete_all("session ended");
            if !leftover.is_empty() {
                tracing::info!(
                    episodes = leftover.len(),
                    "completed episodes left active by an abruptly-ended session"
                );
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn shutdown(
        &self,
        request: Request<ShutdownRequest>,
    ) -> Result<Response<ShutdownResponse>, Status> {
        self.authenticate(&request)?;
        let request = request.into_inner();

        if !self.serve_options.allow_remote_shutdown {
            return Ok(Response::new(ShutdownResponse {
                accepted: false,
                message: "remote shutdown is disabled for this environment endpoint".to_string(),
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

#[tracing::instrument(
    name = "rlmesh.grpc.server.handle_request",
    skip_all,
    fields(
        request_id = %req.request_id,
        request_kind = join_request_kind_name(req.kind.as_ref())
    )
)]
async fn handle_env_request<E: Environment>(
    req: JoinRequest,
    env: Arc<Mutex<E>>,
    episode_tracker: Arc<Mutex<EpisodeTracker>>,
) -> JoinResponse {
    let request_id = req.request_id.clone();
    let operation = join_request_operation(req.kind.as_ref());
    let endpoint_started = Instant::now();

    let kind = match req.kind {
        Some(join_request::Kind::Reset(reset_req)) => {
            let mut env = env.lock().await;

            let num_envs = env.num_envs();
            // Record the seed honestly: an empty seeds vector means the
            // environment seeds itself from entropy, so the episode has no seed
            // rather than a fabricated 0.
            let seeds = reset_req.seeds.clone();
            // Non-empty env_indices is an explicit partial / subenv reset: only
            // those lanes restart, with `seeds` positionally aligned to them.
            let env_indices = reset_req.env_indices.clone();
            let partial = !env_indices.is_empty();

            let timeout_ms = reset_req.timeout_ms;
            let result = if partial {
                run_env_op_with_deadline(
                    env.reset_subset(reset_req),
                    timeout_ms,
                    "env.reset_subset",
                )
                .await
            } else {
                run_env_op_with_deadline(env.reset(reset_req), timeout_ms, "env.reset").await
            };

            match result {
                Ok(mut ok) => {
                    let mut tracker = episode_tracker.lock().await;
                    let episode_ids: Vec<String> = if partial {
                        // Start a fresh episode only for the reset lanes; other
                        // lanes keep their active id. Returned ids are full-width.
                        for (i, &env_idx) in env_indices.iter().enumerate() {
                            let seed = seeds.get(i).copied();
                            tracker.start_episode(env_idx, seed);
                        }
                        (0..num_envs)
                            .map(|env_idx| {
                                tracker
                                    .active_episode_id(env_idx as i32)
                                    .unwrap_or_default()
                                    .to_string()
                            })
                            .collect()
                    } else {
                        (0..num_envs)
                            .map(|env_idx| {
                                let seed = seeds.get(env_idx).copied();
                                tracker.start_episode(env_idx as i32, seed)
                            })
                            .collect()
                    };
                    ok.episode_ids = episode_ids;
                    let obs_bytes = space_value_len(ok.observation.as_ref());
                    let info_bytes = ok.infos.as_ref().map(MetaMap::encoded_len).unwrap_or(0);
                    tracing::debug!(
                        obs_bytes,
                        info_bytes,
                        episode_count = ok.episode_ids.len(),
                        "env reset completed"
                    );
                    Some(join_response::Kind::Reset(ok))
                }
                Err(e) => {
                    tracing::error!(error = %e, "env reset failed");
                    Some(join_response::Kind::Error(env_error_to_proto(e)))
                }
            }
        }
        Some(join_request::Kind::Step(step_req)) => {
            let mut env = env.lock().await;
            let num_envs = env.num_envs();
            let autoreset_mode = env.env_contract().autoreset_mode;

            // Subset-stepping is reserved on the wire (StepRequest.env_indices)
            // but not yet honored: fail loud rather than silently treat it as a
            // full-width step.
            if !step_req.env_indices.is_empty() {
                tracing::error!("StepRequest.env_indices set but subset stepping is unsupported");
                Some(join_response::Kind::Error(env_error_to_proto(
                    EnvError::new(
                        crate::error::EnvErrorCode::Internal,
                        "subset stepping (StepRequest.env_indices) is not supported",
                    ),
                )))
            } else {
                let timeout_ms = step_req.timeout_ms;
                let result =
                    run_env_op_with_deadline(env.step(step_req), timeout_ms, "env.step").await;

                match result {
                    Ok(mut ok) => {
                        let mut tracker = episode_tracker.lock().await;
                        // Episodes interrupted by a replacing reset surface here so
                        // their accounting is not lost.
                        let mut completed_episodes = tracker.drain_interrupted();
                        let shared_info = decode_info_struct(ok.infos.as_ref());

                        // Per-lane episode ids. The id flips only at the NEXT_STEP
                        // boundary (t+1, the fresh-obs step), never on the done step
                        // t, so the terminal observation stays labelled with the
                        // episode that just ended.
                        let mut episode_ids = vec![String::new(); num_envs];

                        for (env_idx, episode_id) in episode_ids.iter_mut().enumerate() {
                            let terminated = ok
                                .terminated_mask
                                .get(env_idx)
                                .map(|&b| b != 0)
                                .unwrap_or(false);
                            let truncated = ok
                                .truncated_mask
                                .get(env_idx)
                                .map(|&b| b != 0)
                                .unwrap_or(false);
                            let done = terminated || truncated;
                            let was_active = tracker.active_episode_id(env_idx as i32).is_some();

                            if was_active {
                                // A real action-step on a running lane: its reward counts.
                                let reward = ok.rewards.get(env_idx).copied().unwrap_or(0.0);
                                tracker.record_step(env_idx as i32, reward);

                                if done {
                                    // Done step t: complete the episode. The terminal obs
                                    // belongs to the OLD episode, so episode_ids keeps the
                                    // completed id; the new id (if any) rolls at t+1.
                                    if let Some(metadata) = tracker.complete_episode(
                                        env_idx as i32,
                                        terminated,
                                        truncated,
                                        extract_env_final_info(
                                            shared_info.as_ref(),
                                            env_idx,
                                            num_envs,
                                        ),
                                    ) {
                                        *episode_id = metadata.episode_id.clone();
                                        completed_episodes.push(metadata);
                                    }
                                } else {
                                    *episode_id = tracker
                                        .active_episode_id(env_idx as i32)
                                        .unwrap_or_default()
                                        .to_string();
                                }
                            } else if autoreset_mode == AutoresetMode::NextStep && !done {
                                // Fresh-obs step t+1: the env auto-reset this lane and is
                                // delivering the first observation of a new episode. Roll
                                // the new id now (step 0). This is NOT a reward-bearing
                                // step, so do not record_step; gym reseeds autoreset from
                                // entropy (seed None).
                                let reward = ok.rewards.get(env_idx).copied().unwrap_or(0.0);
                                if reward != 0.0 {
                                    // The autoreset obs is assumed to carry reward 0. A
                                    // non-zero reward here would belong to the fresh
                                    // episode if recorded, corrupting it — surface the
                                    // anomaly but drop the value.
                                    tracing::warn!(
                                        env_index = env_idx,
                                        reward,
                                        "non-zero reward on a NEXT_STEP autoreset observation is being dropped"
                                    );
                                }
                                *episode_id = tracker.start_episode(env_idx as i32, None);
                            } else if autoreset_mode == AutoresetMode::NextStep
                                && !was_active
                                && done
                            {
                                // A NEXT_STEP lane that is inactive yet reports a terminal
                                // fresh-obs step is unsupported: the completion cannot be
                                // attributed to any episode. Leave the id empty (as before)
                                // but make the dropped completion visible.
                                tracing::warn!(
                                    env_index = env_idx,
                                    "NEXT_STEP env reported a terminal fresh-obs step for an inactive lane; this is unsupported and the completion is dropped"
                                );
                            }
                            // else: a DISABLED lane awaiting an explicit reset (or another
                            // inactive lane) — leave the id empty.
                        }

                        ok.completed_episodes = completed_episodes;
                        ok.episode_ids = episode_ids;
                        let obs_bytes = space_value_len(ok.observation.as_ref());
                        let info_bytes = ok.infos.as_ref().map(MetaMap::encoded_len).unwrap_or(0);
                        tracing::trace!(
                            obs_bytes,
                            info_bytes,
                            completed_episodes = ok.completed_episodes.len(),
                            "env step completed"
                        );
                        Some(join_response::Kind::Step(ok))
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "env step failed");
                        Some(join_response::Kind::Error(env_error_to_proto(e)))
                    }
                }
            }
        }
        Some(join_request::Kind::Render(render_req)) => {
            let mut env = env.lock().await;

            let timeout_ms = render_req.timeout_ms;
            let result =
                run_env_op_with_deadline(env.render(render_req), timeout_ms, "env.render").await;

            match result {
                Ok(ok) => {
                    let frame_bytes = ok.png_frame.as_ref().map(Vec::len).unwrap_or(0);
                    tracing::debug!(frame_bytes, "env render completed");
                    Some(join_response::Kind::Render(ok))
                }
                Err(e) => {
                    tracing::error!(error = %e, "env render failed");
                    Some(join_response::Kind::Error(env_error_to_proto(e)))
                }
            }
        }
        Some(join_request::Kind::Close(_close_req)) => {
            let mut tracker = episode_tracker.lock().await;
            let final_episodes = tracker.complete_all("client close");

            Some(join_response::Kind::Close(CloseResponse { final_episodes }))
        }
        None => Some(join_response::Kind::Error(ProtoEnvError {
            code: ProtoEnvErrorCode::Internal as i32,
            message: "empty request".to_string(),
            is_recoverable: false,
            debug_info: String::new(),
            interrupted_episodes: vec![],
        })),
    };

    let response = JoinResponse {
        kind,
        telemetry: Some(operation_telemetry(operation, endpoint_started.elapsed())),
        request_id,
    };
    tracing::debug!(
        response_kind = join_response_kind_name(response.kind.as_ref()),
        payload_bytes = join_response_payload_bytes(&response),
        "env join response prepared"
    );
    response
}

fn join_request_kind_name(kind: Option<&join_request::Kind>) -> &'static str {
    match kind {
        Some(join_request::Kind::Reset(_)) => "reset",
        Some(join_request::Kind::Step(_)) => "step",
        Some(join_request::Kind::Render(_)) => "render",
        Some(join_request::Kind::Close(_)) => "close",
        None => "empty",
    }
}

fn join_request_operation(kind: Option<&join_request::Kind>) -> &'static str {
    match kind {
        Some(join_request::Kind::Reset(_)) => "env.reset",
        Some(join_request::Kind::Step(_)) => "env.step",
        Some(join_request::Kind::Render(_)) => "env.render",
        Some(join_request::Kind::Close(_)) => "env.close",
        None => "env.unknown",
    }
}

fn join_response_kind_name(kind: Option<&join_response::Kind>) -> &'static str {
    match kind {
        Some(join_response::Kind::Reset(_)) => "reset_ok",
        Some(join_response::Kind::Step(_)) => "step_ok",
        Some(join_response::Kind::Render(_)) => "render_ok",
        Some(join_response::Kind::Close(_)) => "close_ok",
        Some(join_response::Kind::Error(_)) => "error",
        None => "empty",
    }
}

fn join_response_payload_bytes(response: &JoinResponse) -> usize {
    match response.kind.as_ref() {
        Some(join_response::Kind::Reset(ok)) => {
            space_value_len(ok.observation.as_ref())
                + ok.infos.as_ref().map(MetaMap::encoded_len).unwrap_or(0)
        }
        Some(join_response::Kind::Step(ok)) => {
            space_value_len(ok.observation.as_ref())
                + ok.infos.as_ref().map(MetaMap::encoded_len).unwrap_or(0)
        }
        Some(join_response::Kind::Render(ok)) => ok.png_frame.as_ref().map(Vec::len).unwrap_or(0),
        Some(join_response::Kind::Error(error)) => error.message.len() + error.debug_info.len(),
        _ => 0,
    }
}

fn space_value_len(payload: Option<&rlmesh_proto::spaces::v1::SpaceValue>) -> usize {
    value_bytes_ref(payload)
        .ok()
        .flatten()
        .map(|payload| payload.data.len())
        .unwrap_or(0)
}

fn decode_info_struct(info: Option<&MetaMap>) -> Option<MetaMap> {
    info.cloned()
}

fn operation_telemetry(operation: &str, endpoint_total: Duration) -> OperationTelemetry {
    OperationTelemetry {
        operation: operation.to_string(),
        component_id: String::new(),
        metrics: vec![OperationMetric {
            name: "endpoint.total".to_string(),
            labels: HashMap::new(),
            value: Some(operation_metric::Value::DurationNs(
                endpoint_total.as_nanos().try_into().unwrap_or(u64::MAX),
            )),
        }],
    }
}

fn extract_env_final_info(
    info: Option<&MetaMap>,
    env_idx: usize,
    num_envs: usize,
) -> Option<MetaMap> {
    let info = info?;
    let final_info = info.entries.get("final_info")?;
    let is_present = match info.entries.get("_final_info") {
        Some(mask) => value_bool_at(mask, env_idx).unwrap_or(false),
        None => num_envs == 1,
    };

    if !is_present {
        return None;
    }

    match &final_info.kind {
        Some(MetaKind::Map(map)) => Some(map.clone()),
        Some(MetaKind::List(list)) => {
            let entry = list.items.get(env_idx)?;
            match &entry.kind {
                Some(MetaKind::Map(map)) => Some(map.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

fn value_bool_at(value: &MetaValue, env_idx: usize) -> Option<bool> {
    match &value.kind {
        Some(MetaKind::List(list)) => {
            let entry = list.items.get(env_idx)?;
            if let Some(MetaKind::Bool(flag)) = &entry.kind {
                Some(*flag)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Serve the environment at the given address.
pub async fn serve<E: Environment + 'static>(
    env: E,
    addr: impl Into<std::net::SocketAddr>,
) -> Result<(), tonic::transport::Error> {
    let (_health_reporter, health_service) = crate::health::serving_health_service().await;
    tonic::transport::Server::builder()
        .add_service(health_service)
        .add_service(env_service(env))
        .serve(addr.into())
        .await
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use rlmesh_proto::env::v1::env_service_server::EnvService;
    use rlmesh_proto::env::v1::{
        CloseResponse, HandshakeRequest, RenderRequest, RenderResponse, ResetRequest,
        ResetResponse, StepRequest, StepResponse,
    };
    use rlmesh_proto::{
        CURRENT_WORKFLOW_EDITION, MIN_SUPPORTED_PROTOCOL_GENERATION, PROTOCOL_GENERATION,
        capabilities, supported_workflow_editions,
    };
    use rlmesh_spaces::{EnvContract as SpaceEnvContract, SpaceSpec};
    use tonic::Request;

    use super::{Environment, GrpcEnvServer};
    use crate::error::EnvError;

    #[tokio::test]
    async fn handshake_enforces_optional_bearer_token() {
        use crate::lifecycle::{ServeOptions, ShutdownTrigger};

        let options = ServeOptions {
            token: Some("secret-token".to_string()),
            ..Default::default()
        };
        let server = GrpcEnvServer::new_with_options(
            HandshakeOnlyEnv::default(),
            ShutdownTrigger::new(),
            options,
            None,
        );

        // No token -> rejected.
        let err = EnvService::handshake(
            &server,
            Request::new(handshake_request(
                PROTOCOL_GENERATION,
                &[CURRENT_WORKFLOW_EDITION],
            )),
        )
        .await
        .expect_err("missing token must be rejected");
        assert_eq!(err.code(), tonic::Code::Unauthenticated);

        // Wrong token -> rejected.
        let mut wrong = Request::new(handshake_request(
            PROTOCOL_GENERATION,
            &[CURRENT_WORKFLOW_EDITION],
        ));
        wrong
            .metadata_mut()
            .insert("authorization", "nope".parse().unwrap());
        let err = EnvService::handshake(&server, wrong)
            .await
            .expect_err("wrong token must be rejected");
        assert_eq!(err.code(), tonic::Code::Unauthenticated);

        // Correct token -> accepted.
        let mut ok = Request::new(handshake_request(
            PROTOCOL_GENERATION,
            &[CURRENT_WORKFLOW_EDITION],
        ));
        ok.metadata_mut()
            .insert("authorization", "secret-token".parse().unwrap());
        let response = EnvService::handshake(&server, ok)
            .await
            .expect("correct token must be accepted")
            .into_inner();
        assert!(response.compatible);
    }

    #[tokio::test]
    async fn handshake_without_token_is_unauthenticated_by_default() {
        // A server with no configured token accepts unauthenticated requests.
        let server = GrpcEnvServer::new(HandshakeOnlyEnv::default());
        let response = EnvService::handshake(
            &server,
            Request::new(handshake_request(
                PROTOCOL_GENERATION,
                &[CURRENT_WORKFLOW_EDITION],
            )),
        )
        .await
        .expect("no-token server accepts requests")
        .into_inner();
        assert!(response.compatible);
    }

    struct HandshakeOnlyEnv {
        contract: SpaceEnvContract,
    }

    impl Default for HandshakeOnlyEnv {
        fn default() -> Self {
            let space = SpaceSpec::default();
            Self {
                contract: SpaceEnvContract {
                    id: "handshake-only".to_string(),
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

    #[async_trait]
    impl Environment for HandshakeOnlyEnv {
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

        async fn reset(&mut self, _req: ResetRequest) -> Result<ResetResponse, EnvError> {
            unreachable!("handshake test does not call reset")
        }

        async fn step(&mut self, _req: StepRequest) -> Result<StepResponse, EnvError> {
            unreachable!("handshake test does not call step")
        }

        async fn render(&mut self, _req: RenderRequest) -> Result<RenderResponse, EnvError> {
            unreachable!("handshake test does not call render")
        }

        async fn close(&mut self) -> Result<CloseResponse, EnvError> {
            unreachable!("handshake test does not call close")
        }
    }

    fn handshake_request(protocol_generation: &str, offered_editions: &[&str]) -> HandshakeRequest {
        HandshakeRequest {
            protocol_generation: protocol_generation.to_string(),
            client_name: "client".to_string(),
            client_version: "0.1.0-beta.2".to_string(),
            capabilities: Default::default(),
            supported_workflow_editions: offered_editions
                .iter()
                .map(|edition| edition.to_string())
                .collect(),
        }
    }

    /// An env whose `step` sleeps and asserts it is never entered concurrently.
    struct SlowConcurrencyEnv {
        contract: SpaceEnvContract,
        step_delay: std::time::Duration,
        in_op: std::sync::Arc<std::sync::atomic::AtomicBool>,
        overlap_detected: std::sync::Arc<std::sync::atomic::AtomicBool>,
        completed_steps: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl SlowConcurrencyEnv {
        fn new(step_delay: std::time::Duration) -> Self {
            let space = SpaceSpec::default();
            Self {
                contract: SpaceEnvContract {
                    id: "slow".to_string(),
                    autoreset_mode: Default::default(),
                    action_space: Some(space.clone()),
                    observation_space: Some(space),
                    metadata: None,
                    render_mode: String::new(),
                    num_envs: 1,
                },
                step_delay,
                in_op: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                overlap_detected: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                completed_steps: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl Environment for SlowConcurrencyEnv {
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
        async fn reset(&mut self, _req: ResetRequest) -> Result<ResetResponse, EnvError> {
            Ok(ResetResponse::default())
        }
        async fn step(&mut self, _req: StepRequest) -> Result<StepResponse, EnvError> {
            use std::sync::atomic::Ordering;
            let in_op = self.in_op.clone();
            let overlap = self.overlap_detected.clone();
            let completed = self.completed_steps.clone();
            let delay = self.step_delay;
            let handle = tokio::spawn(async move {
                if in_op.swap(true, Ordering::SeqCst) {
                    overlap.store(true, Ordering::SeqCst);
                }
                tokio::time::sleep(delay).await;
                in_op.store(false, Ordering::SeqCst);
                completed.fetch_add(1, Ordering::SeqCst);
            });
            let _ = handle.await;
            Ok(StepResponse::default())
        }
        async fn render(&mut self, _req: RenderRequest) -> Result<RenderResponse, EnvError> {
            Ok(RenderResponse::default())
        }
        async fn close(&mut self) -> Result<CloseResponse, EnvError> {
            Ok(CloseResponse::default())
        }
    }

    #[tokio::test]
    async fn timed_out_step_drains_before_next_request_runs() {
        use std::sync::Arc;
        use std::sync::atomic::Ordering;
        use tokio::sync::Mutex;

        use rlmesh_proto::env::v1::{JoinRequest, join_request, join_response};

        let env = SlowConcurrencyEnv::new(std::time::Duration::from_millis(200));
        let overlap = env.overlap_detected.clone();
        let completed = env.completed_steps.clone();
        let env = Arc::new(Mutex::new(env));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        let step_req = |timeout_ms: i64, id: &str| JoinRequest {
            kind: Some(join_request::Kind::Step(StepRequest {
                timeout_ms,
                ..Default::default()
            })),
            request_id: id.to_string(),
        };

        // Dispatch the second step while the first is still draining.
        let first = {
            let env = env.clone();
            let tracker = tracker.clone();
            tokio::spawn(async move {
                super::handle_env_request(step_req(50, "first"), env, tracker).await
            })
        };
        tokio::time::sleep(std::time::Duration::from_millis(75)).await;
        let second = {
            let env = env.clone();
            let tracker = tracker.clone();
            tokio::spawn(async move {
                super::handle_env_request(step_req(0, "second"), env, tracker).await
            })
        };

        let first_res = first.await.unwrap();
        let second_res = second.await.unwrap();

        // The first call returned a Timeout error to the client...
        assert!(matches!(
            first_res.kind,
            Some(join_response::Kind::Error(ref e))
                if e.code == rlmesh_proto::env::v1::EnvErrorCode::Timeout as i32
        ));
        // ...but the orphaned op was drained, and the second ran without overlap.
        assert!(matches!(
            second_res.kind,
            Some(join_response::Kind::Step(_))
        ));
        assert!(
            !overlap.load(Ordering::SeqCst),
            "two env.step calls overlapped against the same environment"
        );
        // Both steps actually executed to completion in the env.
        assert_eq!(completed.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn join_rejects_a_second_concurrent_stream() {
        use rlmesh_proto::env::v1::JoinRequest;
        use rlmesh_proto::env::v1::env_service_client::EnvServiceClient;
        use rlmesh_proto::env::v1::env_service_server::EnvServiceServer;
        use tokio::sync::oneshot;
        use tokio_stream::wrappers::ReceiverStream;

        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(EnvServiceServer::new(GrpcEnvServer::new(
                    HandshakeOnlyEnv::default(),
                )))
                .serve_with_shutdown(addr, async {
                    let _ = shutdown_rx.await;
                })
                .await
        });

        let endpoint = format!("http://{addr}");
        let mut client = loop {
            match EnvServiceClient::connect(endpoint.clone()).await {
                Ok(client) => break client,
                Err(_) if !server.is_finished() => {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
                Err(error) => panic!("test server did not start: {error}"),
            }
        };
        let mut client2 = EnvServiceClient::connect(endpoint.clone()).await.unwrap();

        // First Join stream: the slot is claimed synchronously in the join
        // handler before it returns, so holding the request channel open keeps
        // the stream active on the server.
        let (tx1, rx1) = tokio::sync::mpsc::channel::<JoinRequest>(1);
        let first = client
            .join(ReceiverStream::new(rx1))
            .await
            .expect("first join accepted")
            .into_inner();

        // Second Join stream must be rejected.
        let (_tx2, rx2) = tokio::sync::mpsc::channel::<JoinRequest>(1);
        let second = client2.join(ReceiverStream::new(rx2)).await;
        let status = second.expect_err("second concurrent join must be rejected");
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);

        // After the first stream ends, a new Join is admitted again.
        drop(tx1);
        drop(first);
        let mut admitted = None;
        for _ in 0..50 {
            let (_tx3, rx3) = tokio::sync::mpsc::channel::<JoinRequest>(1);
            match client2.join(ReceiverStream::new(rx3)).await {
                Ok(stream) => {
                    admitted = Some((_tx3, stream));
                    break;
                }
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(10)).await,
            }
        }
        assert!(
            admitted.is_some(),
            "slot was not released after the first stream ended"
        );

        let _ = shutdown_tx.send(());
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), server).await;
    }

    #[tokio::test]
    async fn handshake_reports_protocol_edition_and_capabilities() {
        let server = GrpcEnvServer::new(HandshakeOnlyEnv::default());

        let response = EnvService::handshake(
            &server,
            Request::new(handshake_request(
                PROTOCOL_GENERATION,
                &[CURRENT_WORKFLOW_EDITION],
            )),
        )
        .await
        .unwrap()
        .into_inner();

        assert!(response.compatible);
        assert_eq!(response.server_protocol_generation, PROTOCOL_GENERATION);
        assert_eq!(
            response.min_supported_protocol_generation,
            MIN_SUPPORTED_PROTOCOL_GENERATION
        );
        assert_eq!(response.selected_workflow_edition, CURRENT_WORKFLOW_EDITION);
        assert_eq!(
            response.supported_workflow_editions,
            supported_workflow_editions()
        );
        assert!(response.env_contract.is_some());
        assert!(
            response
                .capabilities
                .contains_key(capabilities::ENV_SERVICE_V1)
        );
        assert!(
            response
                .capabilities
                .contains_key(capabilities::SPACES_CORE_V1)
        );
    }

    #[tokio::test]
    async fn handshake_rejects_unsupported_protocol_generation() {
        let server = GrpcEnvServer::new(HandshakeOnlyEnv::default());

        let response = EnvService::handshake(
            &server,
            Request::new(handshake_request(
                "rlmesh.protocol.v2",
                &[CURRENT_WORKFLOW_EDITION],
            )),
        )
        .await
        .unwrap()
        .into_inner();

        assert!(!response.compatible);
        assert!(response.error_message.contains("protocol generation"));
        assert!(response.selected_workflow_edition.is_empty());
        assert!(response.env_contract.is_none());
    }

    #[tokio::test]
    async fn handshake_selects_highest_mutual_edition_from_offer() {
        let server = GrpcEnvServer::new(HandshakeOnlyEnv::default());

        let response = EnvService::handshake(
            &server,
            Request::new(handshake_request(
                PROTOCOL_GENERATION,
                &["2025.01", CURRENT_WORKFLOW_EDITION, "2031.12"],
            )),
        )
        .await
        .unwrap()
        .into_inner();

        assert!(response.compatible);
        assert_eq!(response.selected_workflow_edition, CURRENT_WORKFLOW_EDITION);
        assert!(response.env_contract.is_some());
    }

    #[tokio::test]
    async fn handshake_rejects_offer_without_mutual_edition() {
        let server = GrpcEnvServer::new(HandshakeOnlyEnv::default());

        for offer in [&[][..], &["2026"][..], &["2026.11", "2027.01"][..]] {
            let response = EnvService::handshake(
                &server,
                Request::new(handshake_request(PROTOCOL_GENERATION, offer)),
            )
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
            assert_eq!(
                response.supported_workflow_editions,
                supported_workflow_editions()
            );
            assert!(response.selected_workflow_edition.is_empty());
            assert!(response.env_contract.is_none());
        }
    }

    /// A 2-lane vector env whose first step terminates lane 0 and whose later
    /// steps never terminate, modelling a non-autoresetting vector env (e.g.
    /// gymnasium `AutoresetMode::DISABLED`) that keeps accepting steps.
    struct TerminatingVectorEnv {
        contract: SpaceEnvContract,
        step_count: std::sync::atomic::AtomicUsize,
    }

    impl TerminatingVectorEnv {
        fn with_mode(autoreset_mode: rlmesh_spaces::AutoresetMode) -> Self {
            let space = SpaceSpec::default();
            Self {
                contract: SpaceEnvContract {
                    id: "terminating".to_string(),
                    autoreset_mode,
                    action_space: Some(space.clone()),
                    observation_space: Some(space),
                    metadata: None,
                    render_mode: String::new(),
                    num_envs: 2,
                },
                step_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        /// DISABLED: a done lane stays inactive until an explicit reset.
        fn new() -> Self {
            Self::with_mode(rlmesh_spaces::AutoresetMode::Disabled)
        }

        /// NEXT_STEP: the env auto-resets a done lane and delivers its fresh obs
        /// (terminated=false) on the following step — exactly what this mock's
        /// step sequence already produces (terminal at n==0, fresh at n>=1).
        fn next_step() -> Self {
            Self::with_mode(rlmesh_spaces::AutoresetMode::NextStep)
        }
    }

    #[async_trait]
    impl Environment for TerminatingVectorEnv {
        fn observation_space(&self) -> &SpaceSpec {
            self.contract.observation_space.as_ref().unwrap()
        }
        fn action_space(&self) -> &SpaceSpec {
            self.contract.action_space.as_ref().unwrap()
        }
        fn num_envs(&self) -> usize {
            2
        }
        fn env_contract(&self) -> &SpaceEnvContract {
            &self.contract
        }
        async fn reset(&mut self, _req: ResetRequest) -> Result<ResetResponse, EnvError> {
            Ok(ResetResponse::default())
        }
        async fn step(&mut self, _req: StepRequest) -> Result<StepResponse, EnvError> {
            let n = self
                .step_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Lane 0 terminates on the first step only; lane 1 never terminates.
            let terminated_mask = if n == 0 {
                vec![1u8, 0u8]
            } else {
                vec![0u8, 0u8]
            };
            Ok(StepResponse {
                rewards: vec![1.0, 1.0],
                terminated_mask,
                truncated_mask: vec![0u8, 0u8],
                ..Default::default()
            })
        }
        async fn render(&mut self, _req: RenderRequest) -> Result<RenderResponse, EnvError> {
            Ok(RenderResponse::default())
        }
        async fn close(&mut self) -> Result<CloseResponse, EnvError> {
            Ok(CloseResponse::default())
        }
    }

    #[tokio::test]
    async fn terminated_lane_starts_no_phantom_episode_until_reset() {
        // 2026.06: terminated lanes stay inactive until explicit reset.
        use std::sync::Arc;
        use tokio::sync::Mutex;

        use rlmesh_proto::env::v1::{
            JoinRequest, ResetRequest as ProtoResetRequest, join_request, join_response,
        };

        let env = Arc::new(Mutex::new(TerminatingVectorEnv::new()));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        // Reset both lanes.
        let reset = JoinRequest {
            kind: Some(join_request::Kind::Reset(ProtoResetRequest::default())),
            request_id: "reset".to_string(),
        };
        let _ = super::handle_env_request(reset, env.clone(), tracker.clone()).await;

        let step_req = |id: &str| JoinRequest {
            kind: Some(join_request::Kind::Step(StepRequest::default())),
            request_id: id.to_string(),
        };

        // Step 1: lane 0 terminates. Its episode completes exactly once. The
        // terminal observation still belongs to the episode that just ended, so
        // lane 0's `episode_id` is the COMPLETED id (not empty). This env is
        // DISABLED, so no replacement episode is started.
        let first = super::handle_env_request(step_req("s1"), env.clone(), tracker.clone()).await;
        let first = match first.kind {
            Some(join_response::Kind::Step(ok)) => ok,
            other => panic!("expected step response, got {other:?}"),
        };
        assert_eq!(first.completed_episodes.len(), 1, "lane 0 should complete");
        assert_eq!(first.episode_ids.len(), 2);
        assert_eq!(
            first.episode_ids[0], first.completed_episodes[0].episode_id,
            "the terminal step labels lane 0 with the episode that just ended"
        );
        assert!(!first.episode_ids[0].is_empty());
        assert!(
            !first.episode_ids[1].is_empty(),
            "lane 1 keeps its active episode"
        );

        // Step 2: no phantom episode is delivered for lane 0 (no spurious
        // truncated 0-step completion), and lane 0 still has no active episode.
        let second = super::handle_env_request(step_req("s2"), env.clone(), tracker.clone()).await;
        let second = match second.kind {
            Some(join_response::Kind::Step(ok)) => ok,
            other => panic!("expected step response, got {other:?}"),
        };
        assert!(
            second.completed_episodes.is_empty(),
            "no phantom episode may be delivered for the terminated lane"
        );
        assert!(second.episode_ids[0].is_empty());
        assert!(!second.episode_ids[1].is_empty());

        // An explicit Reset re-establishes a tracked episode for every lane.
        let _ = super::handle_env_request(
            JoinRequest {
                kind: Some(join_request::Kind::Reset(ProtoResetRequest::default())),
                request_id: "reset2".to_string(),
            },
            env.clone(),
            tracker.clone(),
        )
        .await;
        let tracker = tracker.lock().await;
        assert!(
            tracker.active_episode_id(0).is_some(),
            "Reset must re-establish lane 0's tracked episode"
        );
    }

    #[tokio::test]
    async fn next_step_rolls_episode_id_at_t_plus_1_not_at_done_step() {
        // BLOCKER-1 regression guard. Under NEXT_STEP the env returns the
        // terminal obs at the done step `t` and the fresh obs at `t+1`. The
        // server must roll the episode id at `t+1` (so the terminal obs stays
        // labelled with the episode that ended), NOT at the done step.
        use std::sync::Arc;
        use tokio::sync::Mutex;

        use rlmesh_proto::env::v1::{
            JoinRequest, ResetRequest as ProtoResetRequest, join_request, join_response,
        };

        let env = Arc::new(Mutex::new(TerminatingVectorEnv::next_step()));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        let _ = super::handle_env_request(
            JoinRequest {
                kind: Some(join_request::Kind::Reset(ProtoResetRequest::default())),
                request_id: "reset".to_string(),
            },
            env.clone(),
            tracker.clone(),
        )
        .await;

        let step = |id: &str| JoinRequest {
            kind: Some(join_request::Kind::Step(StepRequest::default())),
            request_id: id.to_string(),
        };
        let step_ok = |r: super::JoinResponse| match r.kind {
            Some(join_response::Kind::Step(ok)) => ok,
            other => panic!("expected step response, got {other:?}"),
        };

        // Done step t: lane 0 terminates. The terminal obs keeps the OLD id; the
        // id has NOT rolled yet.
        let t = step_ok(super::handle_env_request(step("s1"), env.clone(), tracker.clone()).await);
        assert_eq!(t.completed_episodes.len(), 1, "lane 0 completes at t");
        let old_id = t.completed_episodes[0].episode_id.clone();
        assert_eq!(
            t.episode_ids[0], old_id,
            "terminal obs at t must keep the completed (old) episode id"
        );

        // Fresh-obs step t+1: the env auto-reset lane 0. NOW the id rolls to a new
        // episode (step 0), with no spurious completion.
        let tp1 =
            step_ok(super::handle_env_request(step("s2"), env.clone(), tracker.clone()).await);
        assert!(
            tp1.completed_episodes.is_empty(),
            "no completion on the fresh-obs step"
        );
        assert!(
            !tp1.episode_ids[0].is_empty(),
            "lane 0 has a fresh active episode at t+1"
        );
        assert_ne!(
            tp1.episode_ids[0], old_id,
            "the fresh obs at t+1 carries a NEW episode id"
        );
    }
}
