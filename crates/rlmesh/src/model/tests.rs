use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use rlmesh_proto::model::v1::{
    CloseRequest, CloseRouteRequest, ConfigureRouteRequest, JoinRequest, PredictRequest,
    join_request, join_response,
};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::TcpListenerStream;

use super::lifecycle::update_lifecycle;
use super::server::{ModelRouteConfig, handle_model_request};
use super::*;
use crate::{BindAddress, ConnectAddress, Result, ServeOptions, spaces};

#[derive(Default)]
struct RecordingHandler {
    resets: Vec<(String, i32)>,
    episode_ends: Vec<ModelEpisodeEnd>,
    predictions: Vec<(String, i64, i32)>,
}

#[async_trait]
impl ModelHandler for RecordingHandler {
    async fn predict(&mut self, observation: ModelObservation) -> Result<spaces::BinaryPayload> {
        self.predictions.push((
            observation.episode_id().to_string(),
            observation.step(),
            observation.env_index(),
        ));
        Ok(spaces::BinaryPayload {
            data: vec![1, 2, 3],
        })
    }

    async fn on_reset(&mut self, observation: &ModelObservation) -> Result<()> {
        self.resets.push((
            observation.episode_id().to_string(),
            observation.env_index(),
        ));
        Ok(())
    }

    async fn on_episode_end(&mut self, event: ModelEpisodeEnd) -> Result<()> {
        self.episode_ends.push(event);
        Ok(())
    }
}

fn raw_observation(
    episode_id: &str,
    step: i64,
    env_index: i32,
    is_reset: bool,
) -> ModelObservation {
    ModelObservation {
        observation: Some(spaces::BinaryPayload {
            data: vec![env_index as u8, step as u8],
        }),
        route: ModelRouteContext {
            session_id: "test-session".to_string(),
            route_id: "test-route".to_string(),
            request_id: format!("request-{episode_id}-{step}"),
            slots: vec![ModelRouteSlot {
                episode_id: episode_id.to_string(),
                env_index,
                step,
                reset: is_reset,
            }],
        },
        reset: is_reset,
        num_envs: 1,
        env_contract: None,
    }
}

struct SmokeEnv {
    obs_space: spaces::SpaceSpec,
    action_space: spaces::SpaceSpec,
    env_contract: spaces::EnvContract,
}

impl SmokeEnv {
    fn new() -> Self {
        let obs_space = spaces::spaces::BoxSpaceBuilder::scalar(0.0, 255.0, vec![1])
            .dtype(spaces::DType::Uint8)
            .build()
            .unwrap();
        let action_space = spaces::spaces::BoxSpaceBuilder::scalar(0.0, 1.0, vec![1])
            .dtype(spaces::DType::Uint8)
            .build()
            .unwrap();
        let env_contract = spaces::EnvContract {
            id: "SmokeEnv-v0".to_string(),
            observation_space: Some(obs_space.clone()),
            action_space: Some(action_space.clone()),
            metadata: None,
            render_mode: String::new(),
            num_envs: 1,
        };
        Self {
            obs_space,
            action_space,
            env_contract,
        }
    }
}

#[async_trait]
impl crate::SingleEnv for SmokeEnv {
    fn observation_space(&self) -> &spaces::SpaceSpec {
        &self.obs_space
    }

    fn action_space(&self) -> &spaces::SpaceSpec {
        &self.action_space
    }

    fn env_contract(&self) -> &spaces::EnvContract {
        &self.env_contract
    }

    async fn reset(
        &mut self,
        _req: spaces::request::ResetRequest,
    ) -> std::result::Result<spaces::request::ResetResult, spaces::EnvRuntimeError> {
        Ok(spaces::request::ResetResult {
            observation: Some(spaces::SpaceValue::Box(
                spaces::Tensor::from_vec(vec![0], vec![1], spaces::DType::Uint8).unwrap(),
            )),
            info: None,
            episode_id: Some("ep-smoke".to_string()),
        })
    }

    async fn step(
        &mut self,
        _req: spaces::request::StepRequest,
    ) -> std::result::Result<spaces::request::StepResult, spaces::EnvRuntimeError> {
        Ok(spaces::request::StepResult {
            observation: Some(spaces::SpaceValue::Box(
                spaces::Tensor::from_vec(vec![1], vec![1], spaces::DType::Uint8).unwrap(),
            )),
            reward: 1.0,
            terminated: true,
            truncated: false,
            info: None,
        })
    }

    async fn render(
        &mut self,
        _req: spaces::RenderRequest,
    ) -> std::result::Result<spaces::RenderResult, spaces::EnvRuntimeError> {
        Ok(spaces::RenderResult::default())
    }

    async fn close(
        &mut self,
        _req: spaces::CloseRequest,
    ) -> std::result::Result<spaces::request::CloseResult, spaces::EnvRuntimeError> {
        Ok(spaces::request::CloseResult)
    }
}

struct SmokeModel {
    predicts: Arc<AtomicUsize>,
    closes: Arc<AtomicUsize>,
}

#[async_trait]
impl ModelHandler for SmokeModel {
    async fn predict(&mut self, _observation: ModelObservation) -> Result<spaces::BinaryPayload> {
        self.predicts.fetch_add(1, Ordering::SeqCst);
        Ok(spaces::BinaryPayload { data: vec![0] })
    }

    async fn on_close(&mut self) -> Result<()> {
        self.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
#[ignore = "requires local socket bind support"]
async fn run_local_smoke_uses_in_process_model() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let env_address = format!("tcp://{}", listener.local_addr().unwrap());
    let env_server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(rlmesh_grpc::env::env_service(
                crate::env::WireEnvAdapter::new(crate::SingleEnvAdapter::new(SmokeEnv::new())),
            ))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap()
    });

    let predicts = Arc::new(AtomicUsize::new(0));
    let closes = Arc::new(AtomicUsize::new(0));
    ModelWorker::new(SmokeModel {
        predicts: Arc::clone(&predicts),
        closes: Arc::clone(&closes),
    })
    .run_local_to_async_for_episodes(ConnectAddress::Tcp(env_address), 1)
    .await
    .unwrap();

    assert!(predicts.load(Ordering::SeqCst) >= 1);
    assert_eq!(closes.load(Ordering::SeqCst), 1);
    env_server.abort();
}

#[tokio::test]
async fn served_model_configure_route_requires_env_contract() {
    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::ConfigureRoute(ConfigureRouteRequest {
                context: Some(rlmesh_proto::model::v1::PredictContext {
                    session_id: "session-1".to_string(),
                    route_id: "route-1".to_string(),
                    request_id: "configure-1".to_string(),
                    ..Default::default()
                }),
                env_contract: None,
            })),
            request_id: "configure-1".to_string(),
        },
        Arc::new(Mutex::new(SmokeModel {
            predicts: Arc::new(AtomicUsize::new(0)),
            closes: Arc::new(AtomicUsize::new(0)),
        })),
        Arc::new(Mutex::new(HashMap::new())),
        Arc::new(Mutex::new(HashMap::new())),
    )
    .await;

    assert!(matches!(response.kind, Some(join_response::Kind::Error(_))));
}

#[tokio::test]
async fn served_model_predict_mirrors_route_context() {
    let context = rlmesh_proto::model::v1::PredictContext {
        session_id: "session-1".to_string(),
        route_id: "route-1".to_string(),
        request_id: "request-1".to_string(),
        slots: vec![rlmesh_proto::model::v1::PredictSlot {
            episode_id: "episode-1".to_string(),
            env_index: 0,
            step: 7,
            reset: false,
        }],
    };
    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::Predict(PredictRequest {
                context: Some(context.clone()),
                observation: None,
            })),
            request_id: "request-1".to_string(),
        },
        Arc::new(Mutex::new(SmokeModel {
            predicts: Arc::new(AtomicUsize::new(0)),
            closes: Arc::new(AtomicUsize::new(0)),
        })),
        Arc::new(Mutex::new(HashMap::new())),
        Arc::new(Mutex::new(HashMap::from([(
            "session-1:route-1".to_string(),
            ModelRouteConfig {
                env_contract: None,
                num_envs: 1,
            },
        )]))),
    )
    .await;

    match response.kind {
        Some(join_response::Kind::Predict(response)) => {
            assert_eq!(response.context, Some(context));
        }
        other => panic!("expected predict response, got {other:?}"),
    }
}

#[tokio::test]
async fn served_model_predict_rejects_route_wider_than_opened_route() {
    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::Predict(PredictRequest {
                context: Some(rlmesh_proto::model::v1::PredictContext {
                    session_id: "session-1".to_string(),
                    route_id: "route-1".to_string(),
                    request_id: "request-1".to_string(),
                    slots: vec![
                        rlmesh_proto::model::v1::PredictSlot {
                            episode_id: "episode-0".to_string(),
                            env_index: 0,
                            ..Default::default()
                        },
                        rlmesh_proto::model::v1::PredictSlot {
                            episode_id: "episode-1".to_string(),
                            env_index: 1,
                            ..Default::default()
                        },
                    ],
                }),
                observation: None,
            })),
            request_id: "request-1".to_string(),
        },
        Arc::new(Mutex::new(SmokeModel {
            predicts: Arc::new(AtomicUsize::new(0)),
            closes: Arc::new(AtomicUsize::new(0)),
        })),
        Arc::new(Mutex::new(HashMap::new())),
        Arc::new(Mutex::new(HashMap::from([(
            "session-1:route-1".to_string(),
            ModelRouteConfig {
                env_contract: None,
                num_envs: 1,
            },
        )]))),
    )
    .await;

    assert!(matches!(response.kind, Some(join_response::Kind::Error(_))));
}

#[tokio::test]
async fn served_model_close_route_drains_route_episodes() {
    let handler = Arc::new(Mutex::new(RecordingHandler::default()));
    let active_episodes = Arc::new(Mutex::new(HashMap::from([
        (
            ("session-1:route-1".to_string(), 0),
            "episode-route-1".to_string(),
        ),
        (
            ("session-1:route-2".to_string(), 0),
            "episode-route-2".to_string(),
        ),
    ])));
    let route_configs = Arc::new(Mutex::new(HashMap::from([
        (
            "session-1:route-1".to_string(),
            ModelRouteConfig {
                env_contract: None,
                num_envs: 1,
            },
        ),
        (
            "session-1:route-2".to_string(),
            ModelRouteConfig {
                env_contract: None,
                num_envs: 1,
            },
        ),
    ])));

    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::CloseRoute(CloseRouteRequest {
                context: Some(rlmesh_proto::model::v1::PredictContext {
                    session_id: "session-1".to_string(),
                    route_id: "route-1".to_string(),
                    request_id: "close-route-1".to_string(),
                    ..Default::default()
                }),
                reason: "route complete".to_string(),
            })),
            request_id: "close-route-1".to_string(),
        },
        Arc::clone(&handler),
        Arc::clone(&active_episodes),
        Arc::clone(&route_configs),
    )
    .await;

    assert!(matches!(
        response.kind,
        Some(join_response::Kind::CloseRoute(_))
    ));
    assert_eq!(
        handler.lock().await.episode_ends,
        vec![ModelEpisodeEnd {
            episode_id: "episode-route-1".to_string(),
            env_index: 0,
        }]
    );
    let active_episodes = active_episodes.lock().await;
    assert!(!active_episodes.contains_key(&("session-1:route-1".to_string(), 0)));
    assert_eq!(
        active_episodes.get(&("session-1:route-2".to_string(), 0)),
        Some(&"episode-route-2".to_string())
    );
    drop(active_episodes);
    let route_configs = route_configs.lock().await;
    assert!(!route_configs.contains_key("session-1:route-1"));
    assert!(route_configs.contains_key("session-1:route-2"));
}

#[tokio::test]
async fn served_model_close_drains_all_active_episodes() {
    let handler = Arc::new(Mutex::new(RecordingHandler::default()));
    let active_episodes = Arc::new(Mutex::new(HashMap::from([
        (
            ("session-1:route-1".to_string(), 0),
            "episode-route-1".to_string(),
        ),
        (
            ("session-1:route-2".to_string(), 1),
            "episode-route-2".to_string(),
        ),
    ])));

    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::Close(CloseRequest {
                reason: "session complete".to_string(),
            })),
            request_id: "close-1".to_string(),
        },
        Arc::clone(&handler),
        Arc::clone(&active_episodes),
        Arc::new(Mutex::new(HashMap::new())),
    )
    .await;

    assert!(matches!(response.kind, Some(join_response::Kind::Close(_))));
    assert_eq!(
        handler.lock().await.episode_ends,
        vec![
            ModelEpisodeEnd {
                episode_id: "episode-route-1".to_string(),
                env_index: 0,
            },
            ModelEpisodeEnd {
                episode_id: "episode-route-2".to_string(),
                env_index: 1,
            },
        ]
    );
    assert!(active_episodes.lock().await.is_empty());
}

#[tokio::test]
async fn served_model_close_detaches_and_shutdown_runs_close_hook_once() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let predicts = Arc::new(AtomicUsize::new(0));
    let closes = Arc::new(AtomicUsize::new(0));
    let server_predicts = Arc::clone(&predicts);
    let server_closes = Arc::clone(&closes);
    let server = tokio::spawn(async move {
        ModelWorker::new(SmokeModel {
            predicts: server_predicts,
            closes: server_closes,
        })
        .serve_to_async_with_options(
            BindAddress::Tcp {
                host: "127.0.0.1".to_string(),
                port,
            },
            "route-token",
            ServeOptions {
                allow_remote_shutdown: true,
                ..ServeOptions::default()
            },
        )
        .await
    });

    let address = format!("tcp://127.0.0.1:{port}");
    let connect_options = rlmesh_grpc::ConnectOptions::with_deadline(Duration::from_secs(5))
        .backoff(Duration::from_millis(10));
    let mut client =
        rlmesh_grpc::ModelClient::connect_with_retry(&address, "route-token", &connect_options)
            .await
            .expect("model server did not start");
    client.handshake().await.unwrap();

    client.close("client session complete").await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!server.is_finished());
    assert_eq!(closes.load(Ordering::SeqCst), 0);

    let mut second = rlmesh_grpc::ModelClient::connect(&address, "route-token")
        .await
        .unwrap();
    second.handshake().await.unwrap();
    let shutdown = second.shutdown("test complete").await.unwrap();
    assert!(shutdown.accepted);

    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();

    assert_eq!(predicts.load(Ordering::SeqCst), 0);
    assert_eq!(closes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn public_env_runtime_adapter_drives_a_remote_env_with_telemetry() {
    use rlmesh_runtime::RuntimeEnv;

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let env_address = format!("tcp://{}", listener.local_addr().unwrap());
    let env_server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(rlmesh_grpc::env::env_service(
                crate::env::WireEnvAdapter::new(crate::SingleEnvAdapter::new(SmokeEnv::new())),
            ))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap()
    });

    let mut client = rlmesh_grpc::EnvClient::connect(&env_address).await.unwrap();
    client.handshake().await.unwrap();

    // The public adapter lets external RuntimeDriver embedders drive a remote
    // env without re-implementing the take_last_telemetry choreography.
    let mut adapter = crate::EnvClientRuntimeEnv::new(client);
    let reset = adapter
        .reset(rlmesh_proto::env::v1::ResetRequest {
            seeds: vec![7],
            options: None,
            timeout_ms: 0,
        })
        .await
        .expect("adapter reset must succeed");
    assert!(
        reset.telemetry.is_some(),
        "adapter must encapsulate and surface per-call telemetry"
    );

    env_server.abort();
}

#[tokio::test]
async fn model_bind_resolves_port_zero_before_serving() {
    let predicts = Arc::new(AtomicUsize::new(0));
    let closes = Arc::new(AtomicUsize::new(0));
    let bound = ModelWorker::new(SmokeModel {
        predicts: Arc::clone(&predicts),
        closes: Arc::clone(&closes),
    })
    .bind_to_async_with_options(
        BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        },
        "route-token",
        ServeOptions {
            allow_remote_shutdown: true,
            ..ServeOptions::default()
        },
    )
    .await
    .unwrap();

    // The OS-assigned port is known before serving begins.
    let port = match bound.local_addr().clone() {
        BindAddress::Tcp { port, .. } => port,
        other => panic!("expected tcp bind address, got {other:?}"),
    };
    assert_ne!(port, 0, "port 0 must resolve to a real port");

    let server = tokio::spawn(async move { bound.serve().await });

    // No poll-connect race: the resolved address is immediately usable.
    let address = format!("tcp://127.0.0.1:{port}");
    let mut client = rlmesh_grpc::ModelClient::connect(&address, "route-token")
        .await
        .unwrap();
    client.handshake().await.unwrap();
    let shutdown = client.shutdown("test complete").await.unwrap();
    assert!(shutdown.accepted);

    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(closes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn served_lifecycle_allows_predict_for_every_observation() {
    let observations = vec![
        raw_observation("ep-1", 0, 0, true),
        raw_observation("ep-1", 1, 0, false),
    ];
    let mut handler = RecordingHandler::default();
    let mut active_episodes = HashMap::new();

    for observation in observations {
        update_lifecycle(&mut handler, &mut active_episodes, &observation)
            .await
            .unwrap();
        handler.predict(observation).await.unwrap();
    }

    assert_eq!(
        handler.predictions,
        vec![("ep-1".to_string(), 0, 0), ("ep-1".to_string(), 1, 0)]
    );
    assert_eq!(handler.resets, vec![("ep-1".to_string(), 0)]);
}

#[tokio::test]
async fn served_lifecycle_closes_previous_episode_on_reset_boundary() {
    let observations = vec![
        raw_observation("ep-1", 0, 0, true),
        raw_observation("ep-1", 1, 0, false),
        raw_observation("ep-2", 0, 0, true),
        raw_observation("ep-2", 1, 0, false),
    ];
    let mut handler = RecordingHandler::default();
    let mut active_episodes = HashMap::new();

    for observation in observations {
        update_lifecycle(&mut handler, &mut active_episodes, &observation)
            .await
            .unwrap();
    }

    assert_eq!(
        handler.resets,
        vec![("ep-1".to_string(), 0), ("ep-2".to_string(), 0)]
    );
    assert_eq!(
        handler.episode_ends,
        vec![ModelEpisodeEnd {
            episode_id: "ep-1".to_string(),
            env_index: 0,
        }]
    );
}

#[tokio::test]
async fn served_lifecycle_tracks_episode_boundaries_per_env_index() {
    let observations = vec![
        raw_observation("ep-a", 0, 0, true),
        raw_observation("ep-b", 0, 1, true),
        raw_observation("ep-c", 0, 0, true),
    ];
    let mut handler = RecordingHandler::default();
    let mut active_episodes = HashMap::new();

    for observation in observations {
        update_lifecycle(&mut handler, &mut active_episodes, &observation)
            .await
            .unwrap();
    }

    assert_eq!(
        handler.episode_ends,
        vec![ModelEpisodeEnd {
            episode_id: "ep-a".to_string(),
            env_index: 0,
        }]
    );
}

#[tokio::test]
async fn idle_shutdown_arms_immediately_and_activity_extends_window() {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let shutdown = tokio::spawn(async move {
        rlmesh_grpc::lifecycle::wait_for_idle_shutdown(&mut rx, Duration::from_millis(25)).await;
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!shutdown.is_finished());

    tx.send(rlmesh_grpc::lifecycle::IdleActivity::Started)
        .expect("idle activity receiver should be open");
    tx.send(rlmesh_grpc::lifecycle::IdleActivity::Finished)
        .expect("idle activity receiver should be open");
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!shutdown.is_finished());

    tokio::time::timeout(Duration::from_millis(100), shutdown)
        .await
        .unwrap()
        .unwrap();
}
