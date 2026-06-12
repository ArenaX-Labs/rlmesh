//! Tonic transport server for the EnvService.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use prost::Message;
use prost_types::{Struct, Value, value};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

use crate::env::Environment;
use crate::env::episode::EpisodeTracker;
use crate::error::EnvError;
use crate::lifecycle::{ServeOptions, ShutdownTrigger};
use crate::wire::spaces::env_contract_to_proto;
use crate::wire::value_bytes_ref;

use rlmesh_proto::core::v1::{OperationMetric, OperationTelemetry, operation_metric};
use rlmesh_proto::env::v1::{
    CloseResponse, EnvError as ProtoEnvError, EnvErrorCode as ProtoEnvErrorCode, HandshakeRequest,
    HandshakeResponse, JoinRequest, JoinResponse, ShutdownRequest, ShutdownResponse,
    env_service_server::EnvService, join_request, join_response,
};
use rlmesh_proto::{
    CURRENT_WORKFLOW_EDITION, MIN_SUPPORTED_PROTOCOL_GENERATION, PROTOCOL_GENERATION, capabilities,
    capability_map, is_workflow_edition_compatible, supported_workflow_editions,
};

use super::{env_error_to_proto, is_protocol_generation_compatible};

/// A transport server that implements the `EnvService` tonic trait.
pub struct GrpcEnvServer<E: Environment> {
    env: Arc<Mutex<E>>,
    episode_tracker: Arc<Mutex<EpisodeTracker>>,
    shutdown: ShutdownTrigger,
    serve_options: ServeOptions,
    activity_tx: Option<mpsc::UnboundedSender<()>>,
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
        activity_tx: Option<mpsc::UnboundedSender<()>>,
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
        activity_tx: Option<mpsc::UnboundedSender<()>>,
    ) -> Self {
        Self {
            env,
            episode_tracker: Arc::new(Mutex::new(EpisodeTracker::new())),
            shutdown,
            serve_options,
            activity_tx,
        }
    }
}

pub fn env_service<E: Environment + 'static>(
    env: E,
) -> rlmesh_proto::env::v1::env_service_server::EnvServiceServer<GrpcEnvServer<E>> {
    rlmesh_proto::env::v1::env_service_server::EnvServiceServer::new(GrpcEnvServer::new(env))
}

#[doc(hidden)]
pub fn env_service_from_shared<E: Environment + 'static>(
    env: Arc<Mutex<E>>,
    shutdown: ShutdownTrigger,
    serve_options: ServeOptions,
    activity_tx: Option<mpsc::UnboundedSender<()>>,
) -> rlmesh_proto::env::v1::env_service_server::EnvServiceServer<GrpcEnvServer<E>> {
    rlmesh_proto::env::v1::env_service_server::EnvServiceServer::new(GrpcEnvServer::from_shared(
        env,
        shutdown,
        serve_options,
        activity_tx,
    ))
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
        let req = request.into_inner();

        tracing::info!(
            "Handshake from {} v{} (protocol {}, edition {})",
            req.client_name,
            req.client_version,
            req.protocol_generation,
            req.workflow_edition
        );

        let protocol_compatible =
            is_protocol_generation_compatible(&req.protocol_generation, PROTOCOL_GENERATION);
        let edition_compatible = is_workflow_edition_compatible(&req.workflow_edition);
        let compatible = protocol_compatible && edition_compatible;

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
            } else {
                format!(
                    "workflow edition {} is not compatible; advertised editions: {}",
                    req.workflow_edition,
                    supported_workflow_editions().join(", ")
                )
            },
            capabilities: capability_map(&[
                capabilities::ENV_SERVICE_V1,
                capabilities::SPACES_CORE_V1,
            ]),
            env_contract,
            workflow_edition: CURRENT_WORKFLOW_EDITION.to_string(),
            supported_workflow_editions: supported_workflow_editions(),
        };

        Ok(Response::new(res))
    }

    type JoinStream = ReceiverStream<Result<JoinResponse, Status>>;

    async fn join(
        &self,
        request: Request<Streaming<JoinRequest>>,
    ) -> Result<Response<Self::JoinStream>, Status> {
        let mut req_stream = request.into_inner();
        let env = self.env.clone();
        let episode_tracker = self.episode_tracker.clone();
        let activity_tx = self.activity_tx.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<JoinResponse, Status>>(64);

        tokio::spawn(async move {
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
                    let _ = activity_tx.send(());
                }

                let res = handle_env_request(req, env.clone(), episode_tracker.clone()).await;

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
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn shutdown(
        &self,
        request: Request<ShutdownRequest>,
    ) -> Result<Response<ShutdownResponse>, Status> {
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
            let seeds = if reset_req.seeds.is_empty() {
                vec![0; num_envs]
            } else {
                reset_req.seeds.clone()
            };

            let timeout_ms = reset_req.timeout_ms;
            let result = if timeout_ms > 0 {
                let timeout_duration = Duration::from_millis(timeout_ms as u64);
                match tokio::time::timeout(timeout_duration, env.reset(reset_req)).await {
                    Ok(result) => result,
                    Err(_) => Err(EnvError::new(
                        crate::error::EnvErrorCode::Timeout,
                        format!("reset timed out after {}ms", timeout_ms),
                    )),
                }
            } else {
                env.reset(reset_req).await
            };

            match result {
                Ok(mut ok) => {
                    let mut tracker = episode_tracker.lock().await;
                    let episode_ids: Vec<String> = (0..num_envs)
                        .map(|env_idx| {
                            let seed = seeds.get(env_idx).copied().unwrap_or(0);
                            tracker.start_episode(env_idx as i32, seed)
                        })
                        .collect();
                    ok.episode_ids = episode_ids;
                    let obs_bytes = space_value_len(ok.observation.as_ref());
                    let info_bytes = ok.infos.as_ref().map(Struct::encoded_len).unwrap_or(0);
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

            let timeout_ms = step_req.timeout_ms;
            let result = if timeout_ms > 0 {
                let timeout_duration = Duration::from_millis(timeout_ms as u64);
                match tokio::time::timeout(timeout_duration, env.step(step_req)).await {
                    Ok(result) => result,
                    Err(_) => Err(EnvError::new(
                        crate::error::EnvErrorCode::Timeout,
                        format!("step timed out after {}ms", timeout_ms),
                    )),
                }
            } else {
                env.step(step_req).await
            };

            match result {
                Ok(mut ok) => {
                    let mut tracker = episode_tracker.lock().await;
                    let mut completed_episodes = Vec::new();
                    let shared_info = decode_info_struct(ok.infos.as_ref());

                    for env_idx in 0..num_envs {
                        let reward = ok.rewards.get(env_idx).copied().unwrap_or(0.0);
                        tracker.record_step(env_idx as i32, reward, None);

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

                        if (terminated || truncated)
                            && let Some(metadata) = tracker.complete_episode(
                                env_idx as i32,
                                terminated,
                                truncated,
                                extract_env_final_info(shared_info.as_ref(), env_idx, num_envs),
                            )
                        {
                            completed_episodes.push(metadata);
                        }
                    }

                    let episode_ids = (0..num_envs)
                        .map(|env_idx| {
                            tracker
                                .active_episode_id(env_idx as i32)
                                .unwrap_or_default()
                                .to_string()
                        })
                        .collect();

                    ok.completed_episodes = completed_episodes;
                    ok.episode_ids = episode_ids;
                    let obs_bytes = space_value_len(ok.observation.as_ref());
                    let info_bytes = ok.infos.as_ref().map(Struct::encoded_len).unwrap_or(0);
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
        Some(join_request::Kind::Render(render_req)) => {
            let mut env = env.lock().await;

            let timeout_ms = render_req.timeout_ms;
            let result = if timeout_ms > 0 {
                let timeout_duration = Duration::from_millis(timeout_ms as u64);
                match tokio::time::timeout(timeout_duration, env.render(render_req)).await {
                    Ok(result) => result,
                    Err(_) => Err(EnvError::new(
                        crate::error::EnvErrorCode::Timeout,
                        format!("render timed out after {}ms", timeout_ms),
                    )),
                }
            } else {
                env.render(render_req).await
            };

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
                + ok.infos.as_ref().map(Struct::encoded_len).unwrap_or(0)
        }
        Some(join_response::Kind::Step(ok)) => {
            space_value_len(ok.observation.as_ref())
                + ok.infos.as_ref().map(Struct::encoded_len).unwrap_or(0)
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

fn decode_info_struct(info: Option<&Struct>) -> Option<Struct> {
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
    info: Option<&Struct>,
    env_idx: usize,
    num_envs: usize,
) -> Option<Struct> {
    let info = info?;
    let final_info = info.fields.get("final_info")?;
    let is_present = match info.fields.get("_final_info") {
        Some(mask) => value_bool_at(mask, env_idx).unwrap_or(false),
        None => num_envs == 1,
    };

    if !is_present {
        return None;
    }

    match &final_info.kind {
        Some(value::Kind::StructValue(struct_value)) => Some(struct_value.clone()),
        Some(value::Kind::ListValue(list_value)) => {
            let entry = list_value.values.get(env_idx)?;
            match &entry.kind {
                Some(value::Kind::StructValue(struct_value)) => Some(struct_value.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

fn value_bool_at(value: &Value, env_idx: usize) -> Option<bool> {
    match &value.kind {
        Some(value::Kind::ListValue(list_value)) => {
            let entry = list_value.values.get(env_idx)?;
            if let Some(value::Kind::BoolValue(flag)) = &entry.kind {
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
    use rlmesh_proto::env::v1::env_service_server::EnvServiceServer;

    let server = GrpcEnvServer::new(env);
    tonic::transport::Server::builder()
        .add_service(EnvServiceServer::new(server))
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
        CURRENT_WORKFLOW_EDITION, LEGACY_WORKFLOW_EDITION_2026, MIN_SUPPORTED_PROTOCOL_GENERATION,
        PROTOCOL_GENERATION, capabilities, supported_workflow_editions,
    };
    use rlmesh_spaces::{EnvContract as SpaceEnvContract, SpaceSpec};
    use tonic::Request;

    use super::{Environment, GrpcEnvServer};
    use crate::error::EnvError;

    struct HandshakeOnlyEnv {
        contract: SpaceEnvContract,
    }

    impl Default for HandshakeOnlyEnv {
        fn default() -> Self {
            let space = SpaceSpec::default();
            Self {
                contract: SpaceEnvContract {
                    id: "handshake-only".to_string(),
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

    #[tokio::test]
    async fn handshake_reports_protocol_edition_and_capabilities() {
        let server = GrpcEnvServer::new(HandshakeOnlyEnv::default());

        let response = EnvService::handshake(
            &server,
            Request::new(HandshakeRequest {
                protocol_generation: PROTOCOL_GENERATION.to_string(),
                client_name: "client".to_string(),
                client_version: "0.1.0-beta.2".to_string(),
                capabilities: Default::default(),
                workflow_edition: CURRENT_WORKFLOW_EDITION.to_string(),
            }),
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
        assert_eq!(response.workflow_edition, CURRENT_WORKFLOW_EDITION);
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
            Request::new(HandshakeRequest {
                protocol_generation: "rlmesh.protocol.v2".to_string(),
                client_name: "client".to_string(),
                client_version: "0.1.0-beta.2".to_string(),
                capabilities: Default::default(),
                workflow_edition: CURRENT_WORKFLOW_EDITION.to_string(),
            }),
        )
        .await
        .unwrap()
        .into_inner();

        assert!(!response.compatible);
        assert!(response.error_message.contains("protocol generation"));
        assert!(response.env_contract.is_none());
    }

    #[tokio::test]
    async fn handshake_accepts_legacy_workflow_edition_alias() {
        let server = GrpcEnvServer::new(HandshakeOnlyEnv::default());

        let response = EnvService::handshake(
            &server,
            Request::new(HandshakeRequest {
                protocol_generation: PROTOCOL_GENERATION.to_string(),
                client_name: "client".to_string(),
                client_version: "0.1.0-beta.2".to_string(),
                capabilities: Default::default(),
                workflow_edition: LEGACY_WORKFLOW_EDITION_2026.to_string(),
            }),
        )
        .await
        .unwrap()
        .into_inner();

        assert!(response.compatible);
        assert_eq!(response.workflow_edition, CURRENT_WORKFLOW_EDITION);
        assert_eq!(
            response.supported_workflow_editions,
            supported_workflow_editions()
        );
        assert!(response.env_contract.is_some());
    }

    #[tokio::test]
    async fn handshake_accepts_future_workflow_edition() {
        let server = GrpcEnvServer::new(HandshakeOnlyEnv::default());

        let response = EnvService::handshake(
            &server,
            Request::new(HandshakeRequest {
                protocol_generation: PROTOCOL_GENERATION.to_string(),
                client_name: "client".to_string(),
                client_version: "0.1.0-beta.2".to_string(),
                capabilities: Default::default(),
                workflow_edition: "2026.11".to_string(),
            }),
        )
        .await
        .unwrap()
        .into_inner();

        assert!(response.compatible);
        assert_eq!(response.workflow_edition, CURRENT_WORKFLOW_EDITION);
        assert!(response.env_contract.is_some());
    }

    #[tokio::test]
    async fn handshake_rejects_malformed_workflow_edition() {
        let server = GrpcEnvServer::new(HandshakeOnlyEnv::default());

        let response = EnvService::handshake(
            &server,
            Request::new(HandshakeRequest {
                protocol_generation: PROTOCOL_GENERATION.to_string(),
                client_name: "client".to_string(),
                client_version: "0.1.0-beta.2".to_string(),
                capabilities: Default::default(),
                workflow_edition: "next".to_string(),
            }),
        )
        .await
        .unwrap()
        .into_inner();

        assert!(!response.compatible);
        assert!(response.error_message.contains("workflow edition"));
        assert!(response.env_contract.is_none());
    }
}
