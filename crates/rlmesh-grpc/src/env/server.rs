//! Tonic transport server for the EnvService.

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
use crate::env::episode::{EpisodeTracker, LaneState};
use crate::error::EnvError;
use crate::lifecycle::{ActivityFinishedGuard, IdleActivity, ServeOptions, ShutdownTrigger};
use crate::wire::spaces::env_contract_to_proto;
use crate::wire::value_leaves;
use rlmesh_spaces::AutoresetMode;

use rlmesh_proto::env::v1::{
    CloseEnvsResponse, ConfigureEnvResponse, EnvError as ProtoEnvError,
    EnvErrorCode as ProtoEnvErrorCode, HandshakeRequest, HandshakeResponse, JoinRequest,
    JoinResponse, ShutdownRequest, ShutdownResponse, env_service_server::EnvService, join_request,
    join_response,
};
use rlmesh_proto::{
    capability_map, evaluate_handshake, generation_mismatch_message, peer_info,
    supported_workflow_editions,
};

use super::env_error_to_proto;

/// Run an environment operation under a deadline, then drain it before the next
/// request may access the same environment.
///
/// Python-backed environments are not generally cancellable. On timeout this
/// keeps polling the operation to completion while the caller still holds the
/// environment mutex, preventing overlapping access to the wrapped environment.
async fn run_env_op_with_deadline<F, T>(
    op: F,
    timeout_ms: u64,
    operation: &str,
) -> Result<T, EnvError>
where
    F: std::future::Future<Output = Result<T, EnvError>>,
{
    // timeout_ms comes off the wire as uint64; 0 means "no timeout".
    if timeout_ms == 0 {
        return op.await;
    }

    let timeout_duration = Duration::from_millis(timeout_ms);
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

/// Build a tonic `EnvServiceServer` hosting `env`, with the RLMesh message-size
/// limits applied.
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
            peer_component = %request.get_ref().base.as_ref().and_then(|b| b.peer_info.as_ref()).map(|p| p.component.as_str()).unwrap_or(""),
            peer_version = %request.get_ref().base.as_ref().and_then(|b| b.peer_info.as_ref()).map(|p| p.package_version.as_str()).unwrap_or("")
        )
    )]
    async fn handshake(
        &self,
        request: Request<HandshakeRequest>,
    ) -> Result<Response<HandshakeResponse>, Status> {
        self.authenticate(&request)?;
        let req = request
            .into_inner()
            .base
            .ok_or_else(|| Status::invalid_argument("handshake request missing base"))?;

        // Peer identity/version now ride PeerInfo (subsuming the old
        // client_name/client_version scalars); missing PeerInfo reports empty.
        let peer = req.peer_info.clone().unwrap_or_default();
        tracing::info!(
            "Handshake from {} v{} (protocol {}, offered editions [{}])",
            peer.component,
            peer.package_version,
            req.protocol_generation,
            req.supported_workflow_editions.join(", ")
        );

        // The handshake decides ONE thing: protocol generation. The edition is the
        // runtime's call (the floor); a generation-ok client is compatible and the
        // env contract is returned, even if no edition is mutual — the runtime then
        // fails at the floor with an all-tiers diagnostic.
        let compatible = evaluate_handshake(&req.protocol_generation);

        let env_contract = if compatible {
            let env = self.env.lock().await;
            let mut contract = env_contract_to_proto(env.env_contract());
            contract.num_envs = env.num_envs() as u32;
            Some(contract)
        } else {
            None
        };

        let base = rlmesh_proto::core::v1::HandshakeResponse {
            compatible,
            peer_info: Some(peer_info("rlmesh-env")),
            error_message: (!compatible)
                .then(|| generation_mismatch_message(&req.protocol_generation)),
            capabilities: capability_map(&[]),
            supported_workflow_editions: supported_workflow_editions(),
        };

        Ok(Response::new(HandshakeResponse {
            base: Some(base),
            env_contract,
        }))
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

                // TODO(#6): a non-recoverable `Kind::Error` (e.g. a NEXT_STEP
                // lane-contract violation from handle_env_request) is delivered to
                // the client but does not end the stream. Only a send failure or
                // a Close breaks this loop, so a lenient client can keep stepping.
                // Tracker state stays consistent, so this is a transport-policy
                // decision deferred to its own change.
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
        let request = request
            .into_inner()
            .base
            .ok_or_else(|| Status::invalid_argument("shutdown request missing base"))?;

        if !self.serve_options.allow_remote_shutdown {
            return Ok(Response::new(ShutdownResponse {
                base: Some(rlmesh_proto::core::v1::ShutdownResponse {
                    accepted: false,
                    message: "remote shutdown is disabled for this environment endpoint"
                        .to_string(),
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
    let endpoint_started = Instant::now();

    let kind = match req.kind {
        Some(join_request::Kind::Reset(reset_req)) => {
            let mut env = env.lock().await;

            let num_envs = env.num_envs();
            // Record the seed honestly: an empty seeds vector means the
            // environment seeds itself from entropy, so the episode has no seed
            // rather than a fabricated 0.
            let seeds = reset_req.seeds.clone();
            // Authoritative episode ids minted by the runtime and pushed down
            // (R1): the env adopts these and never mints. Positional — aligned to
            // `env_indices` for a partial reset, to lanes 0..num_envs for a full one.
            let pushed_ids = reset_req.episode_ids.clone();
            // Non-empty env_indices is an explicit partial / subenv reset: only
            // those lanes restart, with `seeds` positionally aligned to them.
            let env_indices = reset_req.env_indices.clone();
            let partial = !env_indices.is_empty();

            // env_indices/seeds arrive straight off the wire, so a foreign or
            // buggy client can send out-of-range, negative, duplicate, or
            // length-mismatched lanes. Validate before touching the env or the
            // tracker: silently deduping/truncating would start phantom or
            // misaligned episodes (seeds are positionally aligned to lanes).
            let timeout_ms = reset_req.timeout_ms;
            let result = match validate_partial_reset(
                partial,
                &env_indices,
                &seeds,
                &pushed_ids,
                num_envs,
            ) {
                Err(message) => Err(EnvError::new(
                    crate::error::EnvErrorCode::InvalidAction,
                    message,
                )),
                Ok(()) if partial => {
                    run_env_op_with_deadline(
                        env.reset_subset(reset_req),
                        timeout_ms,
                        "env.reset_subset",
                    )
                    .await
                }
                Ok(()) => {
                    run_env_op_with_deadline(env.reset(reset_req), timeout_ms, "env.reset").await
                }
            };

            match result {
                Ok(ok) => {
                    let mut tracker = episode_tracker.lock().await;
                    let episode_count = if partial {
                        // Start a fresh episode only for the reset lanes, adopting
                        // the runtime-pushed ids (aligned to env_indices). Other
                        // lanes keep their active id.
                        for (i, &env_idx) in env_indices.iter().enumerate() {
                            let seed = seeds.get(i).copied();
                            let id = pushed_ids.get(i).cloned().unwrap_or_default();
                            // Wire env_index is uint32; the tracker keys on i32.
                            tracker.start_episode(env_idx as i32, seed, id);
                        }
                        env_indices.len()
                    } else {
                        for env_idx in 0..num_envs {
                            let seed = seeds.get(env_idx).copied();
                            let id = pushed_ids.get(env_idx).cloned().unwrap_or_default();
                            tracker.start_episode(env_idx as i32, seed, id);
                        }
                        num_envs
                    };
                    let obs_bytes = space_value_len(ok.observation.as_ref());
                    let info_bytes = ok.infos.as_ref().map(MetaMap::encoded_len).unwrap_or(0);
                    tracing::debug!(obs_bytes, info_bytes, episode_count, "env reset completed");
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
                        crate::error::EnvErrorCode::InvalidAction,
                        "subset stepping (StepRequest.env_indices) is not supported",
                    ),
                )))
            } else {
                let timeout_ms = step_req.timeout_ms;
                // Authoritative per-lane ids pushed down by the runtime (R1).
                // Under NEXT_STEP the env adopts this lane's id when it rolls the
                // fresh autoreset episode; it never mints its own.
                let pushed_ids = step_req.episode_ids.clone();
                let result =
                    run_env_op_with_deadline(env.step(step_req), timeout_ms, "env.step").await;

                match result {
                    Ok(mut ok) => {
                        let mut tracker = episode_tracker.lock().await;
                        // TODO(#6): SameStep falls through to the DISABLED/Idle
                        // path here (next_step is only true for NextStep) and is
                        // silently mishandled. It is rejected upstream at spec /
                        // Python validation today, so it cannot reach a
                        // runtime-driven server; add an explicit guard (or real
                        // SameStep support) here once that contract is defined.
                        let next_step = autoreset_mode == AutoresetMode::NextStep;

                        // Validate the per-lane NEXT_STEP lane lifecycle BEFORE
                        // mutating anything. A contract violation must leave the
                        // tracker (and the interrupted buffer) untouched so the
                        // error is side-effect-free: the step is refused whole
                        // rather than half-applied, and re-stepping reports the
                        // same violation against consistent state. (A `Kind::Error`
                        // payload does not by itself tear down the stream.)
                        if let Some(e) = validate_step_lanes(&ok, num_envs, next_step, &tracker) {
                            tracing::error!(error = %e, "env step contract violation");
                            Some(join_response::Kind::Error(env_error_to_proto(e)))
                        } else {
                            // Validated: every lane takes a legal transition.
                            // Episodes interrupted by a replacing reset surface
                            // here so their accounting is not lost.
                            let mut completed_episodes = tracker.drain_interrupted();
                            let shared_info = ok.infos.clone();

                            // The env no longer emits per-lane episode ids (R1):
                            // the runtime is authoritative and keys lifecycle by
                            // env_index. We still advance tracker state, surface
                            // completed metadata (tagged with the runtime-pushed
                            // id), and adopt the pushed id when a lane rolls.
                            for env_idx in 0..num_envs {
                                let terminated = lane_bit(&ok.terminated_mask, env_idx);
                                let truncated = lane_bit(&ok.truncated_mask, env_idx);
                                let done = terminated || truncated;
                                let reward = ok.rewards.get(env_idx).copied().unwrap_or(0.0);
                                let lane = env_idx as i32;

                                match tracker.lane_state(lane) {
                                    LaneState::Active => {
                                        // A real action-step on a running lane: reward counts.
                                        tracker.record_step(lane, reward);
                                        if done {
                                            // Done step t: complete the episode (its metadata
                                            // carries the runtime id we adopted at start).
                                            if let Some(metadata) = tracker.complete_episode(
                                                lane,
                                                terminated,
                                                truncated,
                                                extract_env_final_info(
                                                    shared_info.as_ref(),
                                                    env_idx,
                                                    num_envs,
                                                ),
                                            ) {
                                                completed_episodes.push(metadata);
                                            }
                                            // Under NEXT_STEP the env owes this lane a fresh
                                            // autoreset observation next step; mark it so that
                                            // step is recognised as the roll.
                                            if next_step {
                                                tracker.expect_autoreset(lane);
                                            }
                                        }
                                    }
                                    LaneState::PendingAutoreset => {
                                        // Validated as the fresh autoreset observation
                                        // (non-terminal, reward 0). Roll the new episode
                                        // (step 0) adopting the runtime-pushed id; not a
                                        // reward-bearing step, so no record_step; gym reseeds
                                        // autoreset from entropy (seed None).
                                        let id =
                                            pushed_ids.get(env_idx).cloned().unwrap_or_default();
                                        tracker.start_episode(lane, None, id);
                                    }
                                    LaneState::Idle => {
                                        // DISABLED only (validation rejects an Idle NEXT_STEP
                                        // lane): an inactive lane awaits an explicit reset; no
                                        // phantom episode.
                                    }
                                }
                            }

                            ok.completed_episodes = completed_episodes;
                            let obs_bytes = space_value_len(ok.observation.as_ref());
                            let info_bytes =
                                ok.infos.as_ref().map(MetaMap::encoded_len).unwrap_or(0);
                            tracing::trace!(
                                obs_bytes,
                                info_bytes,
                                completed_episodes = ok.completed_episodes.len(),
                                "env step completed"
                            );
                            Some(join_response::Kind::Step(ok))
                        }
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
                    let frame_bytes = ok.frame.as_ref().map(Vec::len).unwrap_or(0);
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

            Some(join_response::Kind::Close(CloseEnvsResponse {
                final_episodes,
            }))
        }
        Some(join_request::Kind::Configure(configure_req)) => {
            // Pin the env to the runtime-selected route edition (the standard bind
            // step, sent first). Reject one this build cannot drive (membership in
            // the support window), mirroring the model's enforce_route_floor; honor
            // it as a no-op while a single edition exists (the floor is always
            // CURRENT). An empty pin is a legacy/unset runtime — accepted as a no-op.
            let edition = configure_req.selected_workflow_edition;
            if !edition.is_empty() && !rlmesh_proto::is_supported_edition(&edition) {
                Some(join_response::Kind::Error(ProtoEnvError {
                    code: ProtoEnvErrorCode::InvalidAction as i32,
                    message: format!(
                        "runtime pinned this env to workflow edition {edition:?}, which this env \
                         build does not implement (implements {:?})",
                        rlmesh_proto::SUPPORTED_WORKFLOW_EDITIONS
                    ),
                    is_recoverable: false,
                    debug_info: String::new(),
                    interrupted_episodes: vec![],
                }))
            } else {
                Some(join_response::Kind::Configure(ConfigureEnvResponse {}))
            }
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
        // Hot per-step endpoint-local duration carried as the bare scalar
        // (replaces the old nested telemetry message). The empty-component_id
        // and dead-labels bugs vanish with the nested shape.
        endpoint_total_ns: Some(
            endpoint_started
                .elapsed()
                .as_nanos()
                .min(u128::from(u64::MAX)) as u64,
        ),
        request_id,
    };
    tracing::debug!(
        response_kind = join_response_kind_name(response.kind.as_ref()),
        payload_bytes = join_response_payload_bytes(&response),
        "env join response prepared"
    );
    response
}

/// Validate an explicit partial reset (`ResetRequest.env_indices`) before it
/// reaches the env or the episode tracker. A full reset (empty `env_indices`)
/// is always allowed. For a partial reset every lane must be in `0..num_envs`
/// and unique. Because `seeds` and `episode_ids` are positionally aligned to
/// `env_indices`, each must either be empty or match that length. Duplicates and
/// length mismatches are rejected rather than deduped/truncated: the intent is
/// ambiguous and silently guessing would start phantom, seed-misaligned, or
/// empty-id episodes (the latter then rejected mid-run by the model predict
/// validator, so the two sides of the wire would disagree).
fn validate_partial_reset(
    partial: bool,
    env_indices: &[u32],
    seeds: &[i64],
    episode_ids: &[String],
    num_envs: usize,
) -> Result<(), String> {
    if !partial {
        return Ok(());
    }
    if !seeds.is_empty() && seeds.len() != env_indices.len() {
        return Err(format!(
            "partial reset: seeds length {} does not match env_indices length {} \
             (provide one seed per reset lane, or none)",
            seeds.len(),
            env_indices.len(),
        ));
    }
    if !episode_ids.is_empty() && episode_ids.len() != env_indices.len() {
        return Err(format!(
            "partial reset: episode_ids length {} does not match env_indices length {} \
             (provide one episode id per reset lane, or none)",
            episode_ids.len(),
            env_indices.len(),
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(env_indices.len());
    for &idx in env_indices {
        // Wire env_index is uint32, so a negative index is unrepresentable.
        if idx as usize >= num_envs {
            return Err(format!(
                "partial reset: env_index {idx} out of range for num_envs {num_envs}"
            ));
        }
        if !seen.insert(idx) {
            return Err(format!("partial reset: duplicate env_index {idx}"));
        }
    }
    Ok(())
}

/// Read lane `idx`'s flag from a per-lane byte mask, treating a missing entry as
/// `false`. Vector widths are validated up front by [`validate_step_lanes`], so a
/// missing entry only occurs for an intentionally-empty (all-false) mask.
fn lane_bit(mask: &[u8], idx: usize) -> bool {
    mask.get(idx).map(|&b| b != 0).unwrap_or(false)
}

/// Validate a `StepResponse` against the per-lane NEXT_STEP autoreset contract
/// BEFORE any tracker mutation, returning the first violation as a
/// non-recoverable `Internal` error (or `None` if the step is legal).
///
/// Two classes of violation are caught:
/// - **Malformed width:** each per-lane vector (`rewards`, `terminated_mask`,
///   `truncated_mask`) must be either empty (interpreted as all-false / reward 0)
///   or exactly `num_envs` long. A partial-width vector is rejected so it cannot
///   silently read missing lanes as not-done / reward-0 and mask a real
///   completion or reward.
/// - **Illegal NEXT_STEP transition:** a `PendingAutoreset` lane must deliver a
///   fresh (non-terminal) reward-0 observation, and an `Idle` lane must not be
///   stepped at all (it was never reset, or reported a stray terminal). This
///   replaces the old behaviour of fabricating a phantom episode or silently
///   dropping the reward/completion.
fn validate_step_lanes(
    ok: &rlmesh_proto::env::v1::StepResponse,
    num_envs: usize,
    next_step: bool,
    tracker: &EpisodeTracker,
) -> Option<EnvError> {
    let internal =
        |message: String| Some(EnvError::new(crate::error::EnvErrorCode::Internal, message));

    for (label, len) in [
        ("rewards", ok.rewards.len()),
        ("terminated_mask", ok.terminated_mask.len()),
        ("truncated_mask", ok.truncated_mask.len()),
    ] {
        if len != 0 && len != num_envs {
            return internal(format!(
                "StepResponse.{label} has length {len}, which is neither empty nor the env's lane \
                 count {num_envs}; a partial per-lane vector would silently mask lanes"
            ));
        }
    }

    if !next_step {
        return None;
    }

    for env_idx in 0..num_envs {
        let done = lane_bit(&ok.terminated_mask, env_idx) || lane_bit(&ok.truncated_mask, env_idx);
        let reward = ok.rewards.get(env_idx).copied().unwrap_or(0.0);
        match tracker.lane_state(env_idx as i32) {
            LaneState::PendingAutoreset => {
                if done {
                    return internal(format!(
                        "NEXT_STEP lane {env_idx} reported a terminal step where its autoreset \
                         observation was expected; after a completion the env must deliver one \
                         fresh (non-terminal) observation"
                    ));
                }
                if reward != 0.0 {
                    return internal(format!(
                        "NEXT_STEP lane {env_idx} carried a non-zero reward ({reward}) on its \
                         autoreset observation; the fresh observation after a completion must \
                         carry reward 0"
                    ));
                }
            }
            LaneState::Idle => {
                return internal(format!(
                    "NEXT_STEP lane {env_idx} was stepped with no active episode and no pending \
                     autoreset; reset the lane before stepping it, and the env may only autoreset \
                     the step after a completion"
                ));
            }
            LaneState::Active => {}
        }
    }

    None
}

fn join_request_kind_name(kind: Option<&join_request::Kind>) -> &'static str {
    match kind {
        Some(join_request::Kind::Configure(_)) => "configure",
        Some(join_request::Kind::Reset(_)) => "reset",
        Some(join_request::Kind::Step(_)) => "step",
        Some(join_request::Kind::Render(_)) => "render",
        Some(join_request::Kind::Close(_)) => "close",
        None => "empty",
    }
}

fn join_response_kind_name(kind: Option<&join_response::Kind>) -> &'static str {
    match kind {
        Some(join_response::Kind::Configure(_)) => "configure_ok",
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
        Some(join_response::Kind::Render(ok)) => ok.frame.as_ref().map(Vec::len).unwrap_or(0),
        Some(join_response::Kind::Error(error)) => error.message.len() + error.debug_info.len(),
        _ => 0,
    }
}

fn space_value_len(payload: Option<&rlmesh_proto::spaces::v1::SpaceValue>) -> usize {
    value_leaves(payload)
        .map(|leaves| leaves.iter().map(|leaf| leaf.len()).sum())
        .unwrap_or(0)
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
        CloseEnvsResponse, HandshakeRequest, RenderRequest, RenderResponse, ResetRequest,
        ResetResponse, StepRequest, StepResponse,
    };
    use rlmesh_proto::{
        CURRENT_WORKFLOW_EDITION, PROTOCOL_GENERATION, peer_info, supported_workflow_editions,
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
            ScriptedVectorEnv::handshake_only(),
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
        assert!(response.base.unwrap().compatible);
    }

    #[tokio::test]
    async fn handshake_without_token_is_unauthenticated_by_default() {
        // A server with no configured token accepts unauthenticated requests.
        let server = GrpcEnvServer::new(ScriptedVectorEnv::handshake_only());
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
        assert!(response.base.unwrap().compatible);
    }

    fn contract(
        id: &str,
        num_envs: u32,
        autoreset_mode: rlmesh_spaces::AutoresetMode,
    ) -> SpaceEnvContract {
        let space = SpaceSpec::default();
        SpaceEnvContract {
            id: id.to_string(),
            autoreset_mode,
            action_space: Some(space.clone()),
            observation_space: Some(space),
            metadata: None,
            render_mode: String::new(),
            num_envs,
        }
    }

    /// A probe that asserts `step` is never entered concurrently and counts the
    /// steps that ran to completion (with a configurable per-step delay).
    #[derive(Clone)]
    struct ConcurrencyProbe {
        step_delay: std::time::Duration,
        in_op: std::sync::Arc<std::sync::atomic::AtomicBool>,
        overlap_detected: std::sync::Arc<std::sync::atomic::AtomicBool>,
        completed_steps: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    /// The single env mock used across these tests. It replays a pre-scripted
    /// sequence of `StepResponse`s (yielding the default all-zero step once the
    /// script is exhausted), or runs a [`ConcurrencyProbe`] step when one is set.
    struct ScriptedVectorEnv {
        contract: SpaceEnvContract,
        steps: std::collections::VecDeque<StepResponse>,
        probe: Option<ConcurrencyProbe>,
    }

    impl ScriptedVectorEnv {
        fn new(
            num_envs: usize,
            mode: rlmesh_spaces::AutoresetMode,
            steps: Vec<StepResponse>,
        ) -> Self {
            Self {
                contract: contract("scripted", num_envs as u32, mode),
                steps: steps.into(),
                probe: None,
            }
        }

        /// A handshake-only env: 1 lane, no scripted steps.
        fn handshake_only() -> Self {
            Self::new(1, Default::default(), vec![])
        }

        fn concurrency_probe(step_delay: std::time::Duration) -> Self {
            let mut env = Self::new(1, Default::default(), vec![]);
            env.probe = Some(ConcurrencyProbe {
                step_delay,
                in_op: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                overlap_detected: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                completed_steps: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            });
            env
        }
    }

    #[async_trait]
    impl Environment for ScriptedVectorEnv {
        fn observation_space(&self) -> &SpaceSpec {
            self.contract.observation_space.as_ref().unwrap()
        }
        fn action_space(&self) -> &SpaceSpec {
            self.contract.action_space.as_ref().unwrap()
        }
        fn num_envs(&self) -> usize {
            self.contract.num_envs as usize
        }
        fn env_contract(&self) -> &SpaceEnvContract {
            &self.contract
        }
        async fn reset(&mut self, _req: ResetRequest) -> Result<ResetResponse, EnvError> {
            Ok(ResetResponse::default())
        }
        async fn step(&mut self, _req: StepRequest) -> Result<StepResponse, EnvError> {
            if let Some(probe) = &self.probe {
                use std::sync::atomic::Ordering;
                let probe = probe.clone();
                let handle = tokio::spawn(async move {
                    if probe.in_op.swap(true, Ordering::SeqCst) {
                        probe.overlap_detected.store(true, Ordering::SeqCst);
                    }
                    tokio::time::sleep(probe.step_delay).await;
                    probe.in_op.store(false, Ordering::SeqCst);
                    probe.completed_steps.fetch_add(1, Ordering::SeqCst);
                });
                let _ = handle.await;
                return Ok(StepResponse::default());
            }
            Ok(self.steps.pop_front().unwrap_or_default())
        }
        async fn render(&mut self, _req: RenderRequest) -> Result<RenderResponse, EnvError> {
            Ok(RenderResponse::default())
        }
        async fn close(&mut self) -> Result<CloseEnvsResponse, EnvError> {
            Ok(CloseEnvsResponse::default())
        }
    }

    fn step_resp(rewards: Vec<f64>, terminated: Vec<u8>, truncated: Vec<u8>) -> StepResponse {
        StepResponse {
            rewards,
            terminated_mask: terminated,
            truncated_mask: truncated,
            ..Default::default()
        }
    }

    fn handshake_request(protocol_generation: &str, offered_editions: &[&str]) -> HandshakeRequest {
        HandshakeRequest {
            base: Some(rlmesh_proto::core::v1::HandshakeRequest {
                protocol_generation: protocol_generation.to_string(),
                peer_info: Some(peer_info("rlmesh-env-test-client")),
                capabilities: Default::default(),
                supported_workflow_editions: offered_editions
                    .iter()
                    .map(|edition| edition.to_string())
                    .collect(),
            }),
        }
    }

    #[tokio::test]
    async fn timed_out_step_drains_before_next_request_runs() {
        use std::sync::Arc;
        use std::sync::atomic::Ordering;
        use tokio::sync::Mutex;

        use rlmesh_proto::env::v1::{JoinRequest, join_request, join_response};

        let env = ScriptedVectorEnv::concurrency_probe(std::time::Duration::from_millis(200));
        let probe = env.probe.clone().unwrap();
        let overlap = probe.overlap_detected;
        let completed = probe.completed_steps;
        let env = Arc::new(Mutex::new(env));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        let step_req = |timeout_ms: u64, id: &str| JoinRequest {
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

        // The first call returned a Timeout error to the client.
        assert!(matches!(
            first_res.kind,
            Some(join_response::Kind::Error(ref e))
                if e.code == rlmesh_proto::env::v1::EnvErrorCode::Timeout as i32
        ));
        // The orphaned op was drained, and the second ran without overlap.
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
                    ScriptedVectorEnv::handshake_only(),
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
        let server = GrpcEnvServer::new(ScriptedVectorEnv::handshake_only());

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

        assert!(response.env_contract.is_some());
        let base = response.base.unwrap();
        assert!(base.compatible);
        assert_eq!(
            base.supported_workflow_editions,
            supported_workflow_editions()
        );
        // PeerInfo is populated both directions; the response names the env.
        assert_eq!(base.peer_info.as_ref().unwrap().component, "rlmesh-env");
        // The env advertises no capabilities; the map is the pairwise channel,
        // but there is no behavior-bearing env capability to declare.
        assert!(base.capabilities.is_empty());
    }

    #[tokio::test]
    async fn handshake_carries_python_supplied_peer_info() {
        use std::collections::HashMap;

        use rlmesh_proto::{PeerInfoOverride, set_peer_info_override};

        // Simulate the Python SDK stamping its runtime identity at import: a
        // python-hosted peer reports language/version + framework versions. This
        // is the process-wide override the handshake builder merges in.
        let mut frameworks = HashMap::new();
        frameworks.insert("numpy".to_string(), "1.26.4".to_string());
        set_peer_info_override(PeerInfoOverride {
            language: "python".to_string(),
            language_version: "3.11.4".to_string(),
            package_version: "0.1.0rc1".to_string(),
            os: "linux".to_string(),
            os_version: "ubuntu-22.04".to_string(),
            arch: "x86_64".to_string(),
            framework_versions: frameworks,
            extra: HashMap::new(),
        });

        let server = GrpcEnvServer::new(ScriptedVectorEnv::handshake_only());
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

        let peer = response.base.unwrap().peer_info.unwrap();
        // The component still names this call site; the rest is python-supplied.
        assert_eq!(peer.component, "rlmesh-env");
        assert_eq!(peer.language, "python");
        assert!(!peer.language_version.is_empty());
        assert_eq!(peer.language_version, "3.11.4");
        assert!(
            peer.framework_versions.contains_key("numpy"),
            "python handshake should carry framework versions, got {:?}",
            peer.framework_versions
        );
    }

    #[tokio::test]
    async fn handshake_rejects_unsupported_protocol_generation() {
        let server = GrpcEnvServer::new(ScriptedVectorEnv::handshake_only());

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

        assert!(response.env_contract.is_none());
        let base = response.base.unwrap();
        assert!(!base.compatible);
        assert!(
            base.error_message
                .as_deref()
                .unwrap_or_default()
                .contains("protocol generation")
        );
    }

    #[tokio::test]
    async fn handshake_selects_highest_mutual_edition_from_offer() {
        let server = GrpcEnvServer::new(ScriptedVectorEnv::handshake_only());

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

        // The handshake declares editions; mutual selection happens at bind. A
        // compatible response means a mutual edition exists.
        assert!(response.env_contract.is_some());
        let base = response.base.unwrap();
        assert!(base.compatible);
    }

    #[tokio::test]
    async fn handshake_accepts_any_generation_compatible_offer() {
        // The handshake gates only generation; the edition is the runtime's call
        // (the floor). A generation-ok client is compatible and the env contract is
        // returned, even with no mutual edition — it fails later at the floor.
        let server = GrpcEnvServer::new(ScriptedVectorEnv::handshake_only());

        for offer in [&[][..], &["2026"][..], &["2026.11", "2027.01"][..]] {
            let response = EnvService::handshake(
                &server,
                Request::new(handshake_request(PROTOCOL_GENERATION, offer)),
            )
            .await
            .unwrap()
            .into_inner();

            let base = response.base.unwrap();
            assert!(
                base.compatible,
                "generation-ok offer {offer:?} is compatible"
            );
            assert!(base.error_message.is_none());
            assert!(response.env_contract.is_some());
            assert_eq!(
                base.supported_workflow_editions,
                supported_workflow_editions()
            );
        }
    }

    #[tokio::test]
    async fn configure_env_pins_edition_and_rejects_unsupported() {
        use std::sync::Arc;

        use rlmesh_proto::env::v1::{
            ConfigureEnvRequest, JoinRequest, join_request, join_response,
        };
        use tokio::sync::Mutex;

        let env = Arc::new(Mutex::new(terminating_env(
            rlmesh_spaces::AutoresetMode::Disabled,
        )));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));
        let configure = |edition: &str| JoinRequest {
            kind: Some(join_request::Kind::Configure(ConfigureEnvRequest {
                selected_workflow_edition: edition.to_string(),
            })),
            request_id: "configure".to_string(),
        };

        // The current edition (the only one in the window) is accepted; an empty
        // pin (legacy/unset runtime) is accepted as a no-op.
        for pin in [rlmesh_proto::CURRENT_WORKFLOW_EDITION, ""] {
            let ok = super::handle_env_request(configure(pin), env.clone(), tracker.clone()).await;
            assert!(
                matches!(ok.kind, Some(join_response::Kind::Configure(_))),
                "pin {pin:?} should be accepted"
            );
        }

        // An edition this build cannot drive is rejected.
        let bad =
            super::handle_env_request(configure("2099.01"), env.clone(), tracker.clone()).await;
        assert!(matches!(bad.kind, Some(join_response::Kind::Error(_))));
    }

    /// A 2-lane vector env whose first step terminates lane 0 and whose later
    /// steps never terminate, modelling a non-autoresetting vector env that
    /// keeps accepting steps. Lane 0's fresh autoreset obs (reward 0) lands at
    /// the step after its termination; lane 1 keeps stepping normally.
    fn terminating_env(mode: rlmesh_spaces::AutoresetMode) -> ScriptedVectorEnv {
        ScriptedVectorEnv::new(
            2,
            mode,
            vec![
                step_resp(vec![1.0, 1.0], vec![1, 0], vec![0, 0]),
                step_resp(vec![0.0, 1.0], vec![0, 0], vec![0, 0]),
            ],
        )
    }

    #[tokio::test]
    async fn terminated_lane_starts_no_phantom_episode_until_reset() {
        // 2026.06: terminated lanes stay inactive until explicit reset.
        use std::sync::Arc;
        use tokio::sync::Mutex;

        use rlmesh_proto::env::v1::{
            JoinRequest, ResetRequest as ProtoResetRequest, join_request, join_response,
        };

        let env = Arc::new(Mutex::new(terminating_env(
            rlmesh_spaces::AutoresetMode::Disabled,
        )));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        // Reset both lanes, pushing the runtime-minted ids the env adopts (R1).
        let reset = JoinRequest {
            kind: Some(join_request::Kind::Reset(ProtoResetRequest {
                episode_ids: vec!["E0".to_string(), "E1".to_string()],
                ..Default::default()
            })),
            request_id: "reset".to_string(),
        };
        let _ = super::handle_env_request(reset, env.clone(), tracker.clone()).await;

        let step_req = |id: &str| JoinRequest {
            kind: Some(join_request::Kind::Step(StepRequest::default())),
            request_id: id.to_string(),
        };

        // Step 1: lane 0 terminates. Its episode completes exactly once, carrying
        // the adopted id (the env no longer emits per-lane response ids). This env
        // is DISABLED, so no replacement episode is started.
        let first = super::handle_env_request(step_req("s1"), env.clone(), tracker.clone()).await;
        let first = match first.kind {
            Some(join_response::Kind::Step(ok)) => ok,
            other => panic!("expected step response, got {other:?}"),
        };
        assert_eq!(first.completed_episodes.len(), 1, "lane 0 should complete");
        assert_eq!(
            first.completed_episodes[0].episode_id, "E0",
            "the completed episode carries the runtime-pushed id"
        );
        {
            let t = tracker.lock().await;
            assert!(
                t.active_episode_id(0).is_none(),
                "DISABLED: terminated lane 0 has no active episode"
            );
            assert_eq!(
                t.active_episode_id(1),
                Some("E1"),
                "lane 1 keeps its episode"
            );
        }

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
        {
            let t = tracker.lock().await;
            assert!(t.active_episode_id(0).is_none());
            assert_eq!(t.active_episode_id(1), Some("E1"));
        }

        // An explicit Reset re-establishes a tracked episode for every lane.
        let _ = super::handle_env_request(
            JoinRequest {
                kind: Some(join_request::Kind::Reset(ProtoResetRequest {
                    episode_ids: vec!["F0".to_string(), "F1".to_string()],
                    ..Default::default()
                })),
                request_id: "reset2".to_string(),
            },
            env.clone(),
            tracker.clone(),
        )
        .await;
        let tracker = tracker.lock().await;
        assert_eq!(
            tracker.active_episode_id(0),
            Some("F0"),
            "Reset must re-establish lane 0's tracked episode"
        );
    }

    #[tokio::test]
    async fn partial_reset_rejects_invalid_env_indices() {
        // ResetRequest.env_indices arrives straight off the wire, so the server
        // must reject out-of-range, negative, duplicate, and seed-misaligned
        // lanes BEFORE touching the env or the tracker. Silently deduping or
        // truncating would start phantom or seed-misaligned episodes.
        use std::sync::Arc;
        use tokio::sync::Mutex;

        use rlmesh_proto::env::v1::{
            EnvErrorCode as ProtoEnvErrorCode, JoinRequest, ResetRequest as ProtoResetRequest,
            join_request, join_response,
        };

        // num_envs == 2.
        let env = Arc::new(Mutex::new(terminating_env(
            rlmesh_spaces::AutoresetMode::Disabled,
        )));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        let reset = |env_indices: Vec<u32>, seeds: Vec<i64>| JoinRequest {
            kind: Some(join_request::Kind::Reset(ProtoResetRequest {
                env_indices,
                seeds,
                ..Default::default()
            })),
            request_id: "partial".to_string(),
        };

        let expect_invalid = |resp: super::JoinResponse, needle: &str| match resp.kind {
            Some(join_response::Kind::Error(e)) => {
                assert_eq!(
                    e.code,
                    ProtoEnvErrorCode::InvalidAction as i32,
                    "invalid partial reset must report InvalidAction, got code {} ({})",
                    e.code,
                    e.message,
                );
                assert!(
                    e.message.contains(needle),
                    "error message {:?} should mention {:?}",
                    e.message,
                    needle,
                );
            }
            other => panic!("expected error response, got {other:?}"),
        };

        // Out of range: lane 2 does not exist for num_envs == 2.
        expect_invalid(
            super::handle_env_request(reset(vec![2], vec![]), env.clone(), tracker.clone()).await,
            "out of range",
        );
        // Negative lanes are unrepresentable now that env_indices is uint32.
        // Duplicate lane.
        expect_invalid(
            super::handle_env_request(reset(vec![0, 0], vec![]), env.clone(), tracker.clone())
                .await,
            "duplicate",
        );
        // Seeds present but misaligned with env_indices (2 lanes, 1 seed).
        expect_invalid(
            super::handle_env_request(reset(vec![0, 1], vec![7]), env.clone(), tracker.clone())
                .await,
            "seeds length",
        );

        // A rejected partial reset must not have started any tracked episode.
        let tracker = tracker.lock().await;
        assert!(
            tracker.active_episode_id(0).is_none() && tracker.active_episode_id(1).is_none(),
            "a rejected partial reset must not start any episode"
        );
    }

    #[tokio::test]
    async fn next_step_rolls_episode_id_at_t_plus_1_not_at_done_step() {
        // BLOCKER-1 regression guard. Under NEXT_STEP the env returns the
        // terminal obs at the done step `t` and the fresh obs at `t+1`. The
        // server must roll the episode id at `t+1` (so the terminal obs stays
        // labelled with the episode that ended), not at the done step.
        use std::sync::Arc;
        use tokio::sync::Mutex;

        use rlmesh_proto::env::v1::{
            JoinRequest, ResetRequest as ProtoResetRequest, join_request, join_response,
        };

        let env = Arc::new(Mutex::new(terminating_env(
            rlmesh_spaces::AutoresetMode::NextStep,
        )));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        let _ = super::handle_env_request(
            JoinRequest {
                kind: Some(join_request::Kind::Reset(ProtoResetRequest {
                    episode_ids: vec!["A".to_string()],
                    ..Default::default()
                })),
                request_id: "reset".to_string(),
            },
            env.clone(),
            tracker.clone(),
        )
        .await;

        // A step that pushes the runtime-minted ids the env adopts on a roll.
        let step = |id: &str, episode_ids: Vec<String>| JoinRequest {
            kind: Some(join_request::Kind::Step(StepRequest {
                episode_ids,
                ..Default::default()
            })),
            request_id: id.to_string(),
        };
        let step_ok = |r: super::JoinResponse| match r.kind {
            Some(join_response::Kind::Step(ok)) => ok,
            other => panic!("expected step response, got {other:?}"),
        };

        // Done step t: lane 0 terminates, completing the old episode "A". The id
        // has not rolled yet (the lane is now pending-autoreset).
        let t = step_ok(
            super::handle_env_request(
                step("s1", vec!["A".to_string()]),
                env.clone(),
                tracker.clone(),
            )
            .await,
        );
        assert_eq!(t.completed_episodes.len(), 1, "lane 0 completes at t");
        assert_eq!(t.completed_episodes[0].episode_id, "A");
        {
            let tk = tracker.lock().await;
            assert!(
                tk.active_episode_id(0).is_none(),
                "lane 0 is pending-autoreset at t, no active episode"
            );
        }

        // Fresh-obs step t+1: the env auto-resets lane 0, adopting the rolled id
        // "B" the runtime pushed. No spurious completion.
        let tp1 = step_ok(
            super::handle_env_request(
                step("s2", vec!["B".to_string()]),
                env.clone(),
                tracker.clone(),
            )
            .await,
        );
        assert!(
            tp1.completed_episodes.is_empty(),
            "no completion on the fresh-obs step"
        );
        let tk = tracker.lock().await;
        assert_eq!(
            tk.active_episode_id(0),
            Some("B"),
            "the fresh obs at t+1 adopts the NEW pushed episode id"
        );
    }

    #[tokio::test]
    async fn next_step_nonzero_reward_on_autoreset_obs_is_an_error() {
        use std::sync::Arc;

        use rlmesh_proto::env::v1::{
            JoinRequest, ResetRequest as ProtoResetRequest, join_request, join_response,
        };
        use tokio::sync::Mutex;

        // Terminal at s1, then a fresh-obs step carrying reward 3.0. The
        // autoreset observation must be reward 0, so this is a hard error rather
        // than a silently dropped reward.
        let env = Arc::new(Mutex::new(ScriptedVectorEnv::new(
            1,
            rlmesh_spaces::AutoresetMode::NextStep,
            vec![
                step_resp(vec![1.0], vec![1], vec![0]),
                step_resp(vec![3.0], vec![0], vec![0]),
            ],
        )));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        let reset = JoinRequest {
            kind: Some(join_request::Kind::Reset(ProtoResetRequest::default())),
            request_id: "r".to_string(),
        };
        let _ = super::handle_env_request(reset, env.clone(), tracker.clone()).await;

        let step = |id: &str| JoinRequest {
            kind: Some(join_request::Kind::Step(StepRequest::default())),
            request_id: id.to_string(),
        };
        // Done step: completes lane 0 and marks it pending-autoreset.
        let _ = super::handle_env_request(step("s1"), env.clone(), tracker.clone()).await;
        // Fresh-obs step with a non-zero reward: hard error.
        let resp = super::handle_env_request(step("s2"), env.clone(), tracker.clone()).await;
        match resp.kind {
            Some(join_response::Kind::Error(e)) => assert!(
                e.message.contains("non-zero reward"),
                "expected a reward-on-autoreset error, got: {}",
                e.message
            ),
            other => panic!("expected error response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn next_step_terminal_when_autoreset_expected_is_an_error() {
        use std::sync::Arc;

        use rlmesh_proto::env::v1::{
            JoinRequest, ResetRequest as ProtoResetRequest, join_request, join_response,
        };
        use tokio::sync::Mutex;

        // Terminal at s1, then terminal again at s2. The env never delivered the
        // fresh autoreset observation. A sticky-terminal env must fail loud, not
        // silently drop the second completion.
        let env = Arc::new(Mutex::new(ScriptedVectorEnv::new(
            1,
            rlmesh_spaces::AutoresetMode::NextStep,
            vec![
                step_resp(vec![1.0], vec![1], vec![0]),
                step_resp(vec![0.0], vec![1], vec![0]),
            ],
        )));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        let reset = JoinRequest {
            kind: Some(join_request::Kind::Reset(ProtoResetRequest::default())),
            request_id: "r".to_string(),
        };
        let _ = super::handle_env_request(reset, env.clone(), tracker.clone()).await;
        let step = |id: &str| JoinRequest {
            kind: Some(join_request::Kind::Step(StepRequest::default())),
            request_id: id.to_string(),
        };
        let _ = super::handle_env_request(step("s1"), env.clone(), tracker.clone()).await;
        let resp = super::handle_env_request(step("s2"), env.clone(), tracker.clone()).await;
        match resp.kind {
            Some(join_response::Kind::Error(e)) => assert!(
                e.message
                    .contains("terminal step where its autoreset observation was expected"),
                "expected a terminal-when-autoreset-expected error, got: {}",
                e.message
            ),
            other => panic!("expected error response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn next_step_step_before_reset_is_an_error() {
        use std::sync::Arc;

        use rlmesh_proto::env::v1::{JoinRequest, join_request, join_response};
        use tokio::sync::Mutex;

        // No reset: lane 0 is Idle. Stepping a NEXT_STEP lane with no active
        // episode and no pending autoreset is a hard error. The old behavior
        // fabricated a phantom episode here.
        let env = Arc::new(Mutex::new(ScriptedVectorEnv::new(
            1,
            rlmesh_spaces::AutoresetMode::NextStep,
            vec![step_resp(vec![1.0], vec![0], vec![0])],
        )));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        let step = JoinRequest {
            kind: Some(join_request::Kind::Step(StepRequest::default())),
            request_id: "s1".to_string(),
        };
        let resp = super::handle_env_request(step, env.clone(), tracker.clone()).await;
        match resp.kind {
            Some(join_response::Kind::Error(e)) => assert!(
                e.message
                    .contains("no active episode and no pending autoreset"),
                "expected a step-before-reset error, got: {}",
                e.message
            ),
            other => panic!("expected error response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn next_step_cycles_cleanly_through_multiple_episodes() {
        use std::sync::Arc;

        use rlmesh_proto::env::v1::{
            JoinRequest, ResetRequest as ProtoResetRequest, join_request, join_response,
        };
        use tokio::sync::Mutex;

        // s1 normal, s2 terminal (completes A), s3 fresh obs (rolls B),
        // s4 terminal (completes B), s5 fresh obs (rolls C). Two completions
        // total, three distinct episode ids, no phantom episodes.
        let env = Arc::new(Mutex::new(ScriptedVectorEnv::new(
            1,
            rlmesh_spaces::AutoresetMode::NextStep,
            vec![
                step_resp(vec![1.0], vec![0], vec![0]),
                step_resp(vec![1.0], vec![1], vec![0]),
                step_resp(vec![0.0], vec![0], vec![0]),
                step_resp(vec![1.0], vec![1], vec![0]),
                step_resp(vec![0.0], vec![0], vec![0]),
            ],
        )));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        let reset = JoinRequest {
            kind: Some(join_request::Kind::Reset(ProtoResetRequest {
                episode_ids: vec!["A".to_string()],
                ..Default::default()
            })),
            request_id: "r".to_string(),
        };
        let _ = super::handle_env_request(reset, env.clone(), tracker.clone()).await;
        // Steps push the runtime-minted ids the env adopts on each roll.
        let step = |id: &str, episode_ids: Vec<String>| JoinRequest {
            kind: Some(join_request::Kind::Step(StepRequest {
                episode_ids,
                ..Default::default()
            })),
            request_id: id.to_string(),
        };
        let step_ok = |r: super::JoinResponse| match r.kind {
            Some(join_response::Kind::Step(ok)) => ok,
            other => panic!("expected step response, got {other:?}"),
        };

        let _ = step_ok(
            super::handle_env_request(
                step("s1", vec!["A".to_string()]),
                env.clone(),
                tracker.clone(),
            )
            .await,
        );
        let s2 = step_ok(
            super::handle_env_request(
                step("s2", vec!["A".to_string()]),
                env.clone(),
                tracker.clone(),
            )
            .await,
        );
        assert_eq!(s2.completed_episodes.len(), 1, "episode A completes at s2");
        assert_eq!(s2.completed_episodes[0].episode_id, "A");

        // s3 rolls episode B (the env adopts the pushed id).
        let s3 = step_ok(
            super::handle_env_request(
                step("s3", vec!["B".to_string()]),
                env.clone(),
                tracker.clone(),
            )
            .await,
        );
        assert!(
            s3.completed_episodes.is_empty(),
            "no completion on the roll"
        );
        {
            let tk = tracker.lock().await;
            assert_eq!(
                tk.active_episode_id(0),
                Some("B"),
                "s3 rolls a new episode B"
            );
        }

        let s4 = step_ok(
            super::handle_env_request(
                step("s4", vec!["B".to_string()]),
                env.clone(),
                tracker.clone(),
            )
            .await,
        );
        assert_eq!(s4.completed_episodes.len(), 1, "episode B completes at s4");
        assert_eq!(s4.completed_episodes[0].episode_id, "B");

        let s5 = step_ok(
            super::handle_env_request(
                step("s5", vec!["C".to_string()]),
                env.clone(),
                tracker.clone(),
            )
            .await,
        );
        assert!(s5.completed_episodes.is_empty());
        let tk = tracker.lock().await;
        assert_eq!(
            tk.active_episode_id(0),
            Some("C"),
            "s5 rolls a new episode C"
        );
    }

    #[tokio::test]
    async fn next_step_contract_violation_is_side_effect_free() {
        use std::sync::Arc;

        use rlmesh_proto::env::v1::{
            JoinRequest, ResetRequest as ProtoResetRequest, join_request, join_response,
        };
        use tokio::sync::Mutex;

        // A violation on any lane must abort the whole step without mutating the
        // tracker; earlier lanes must not be half-completed. num_envs=2: lane 1
        // is driven into PendingAutoreset, then a step reports lane 0 terminal
        // (which would complete it) and lane 1 terminal-when-autoreset-expected
        // (a violation). The step must error with lane 0 left untouched.
        let env = Arc::new(Mutex::new(ScriptedVectorEnv::new(
            2,
            rlmesh_spaces::AutoresetMode::NextStep,
            vec![
                // s1: lane 1 terminates -> completes, becomes pending-autoreset.
                step_resp(vec![1.0, 1.0], vec![0, 1], vec![0, 0]),
                // s2: lane 0 terminal (would complete) + lane 1 terminal again
                //     (pending-autoreset violation) -> the whole step must error.
                step_resp(vec![1.0, 0.0], vec![1, 1], vec![0, 0]),
            ],
        )));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        let reset = JoinRequest {
            kind: Some(join_request::Kind::Reset(ProtoResetRequest::default())),
            request_id: "r".to_string(),
        };
        let _ = super::handle_env_request(reset, env.clone(), tracker.clone()).await;
        let step = |id: &str| JoinRequest {
            kind: Some(join_request::Kind::Step(StepRequest::default())),
            request_id: id.to_string(),
        };
        // s1 puts lane 1 into pending-autoreset.
        let _ = super::handle_env_request(step("s1"), env.clone(), tracker.clone()).await;

        // Snapshot lane 0's episode id before the violating step.
        let lane0_before = {
            let t = tracker.lock().await;
            t.active_episode_id(0).map(|s| s.to_string())
        };
        assert!(
            lane0_before.is_some(),
            "lane 0 is active before the violating step"
        );

        // s2 violates on lane 1; the whole step must error.
        let resp = super::handle_env_request(step("s2"), env.clone(), tracker.clone()).await;
        assert!(
            matches!(resp.kind, Some(join_response::Kind::Error(_))),
            "a violating step must return an error"
        );

        // Lane 0 must not have been completed or rolled: same active episode.
        let t = tracker.lock().await;
        assert_eq!(
            t.lane_state(0),
            super::super::episode::LaneState::Active,
            "lane 0 stays active; the aborted step did not half-apply"
        );
        assert_eq!(
            t.active_episode_id(0).map(|s| s.to_string()),
            lane0_before,
            "lane 0's episode is untouched by the aborted step"
        );
    }

    #[tokio::test]
    async fn step_rejects_partial_width_masks() {
        use std::sync::Arc;

        use rlmesh_proto::env::v1::{
            JoinRequest, ResetRequest as ProtoResetRequest, join_request, join_response,
        };
        use tokio::sync::Mutex;

        // A per-lane vector that is neither empty nor full-width is rejected so a
        // missing lane cannot be silently read as not-done / reward-0.
        let env = Arc::new(Mutex::new(ScriptedVectorEnv::new(
            2,
            rlmesh_spaces::AutoresetMode::NextStep,
            // terminated_mask has length 1 for a 2-lane env: partial width.
            vec![step_resp(vec![1.0, 1.0], vec![0], vec![0, 0])],
        )));
        let tracker = Arc::new(Mutex::new(super::super::episode::EpisodeTracker::new()));

        let reset = JoinRequest {
            kind: Some(join_request::Kind::Reset(ProtoResetRequest::default())),
            request_id: "r".to_string(),
        };
        let _ = super::handle_env_request(reset, env.clone(), tracker.clone()).await;
        let step = JoinRequest {
            kind: Some(join_request::Kind::Step(StepRequest::default())),
            request_id: "s1".to_string(),
        };
        let resp = super::handle_env_request(step, env.clone(), tracker.clone()).await;
        match resp.kind {
            Some(join_response::Kind::Error(e)) => assert!(
                e.message.contains("neither empty nor"),
                "expected a partial-width error, got: {}",
                e.message
            ),
            other => panic!("expected error response, got {other:?}"),
        }
    }
}
