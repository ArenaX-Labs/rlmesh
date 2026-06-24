use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use rlmesh_proto::model::v1::{
    CloseParticipantRequest, CloseRouteRequest, ConfigureRouteRequest, JoinRequest, PredictRequest,
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
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<spaces::SpaceValue>> {
        self.predictions.push((
            observation.episode_id().to_string(),
            observation.step(),
            observation.env_index(),
        ));
        Ok(vec![spaces::SpaceValue::Discrete(0)])
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
        observation: None,
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
    reset_seeds: Option<Arc<Mutex<Vec<Option<i64>>>>>,
}

impl SmokeEnv {
    /// A smoke env that records the reset seed it was asked to use into `sink`.
    fn recording(sink: Arc<Mutex<Vec<Option<i64>>>>) -> Self {
        Self {
            reset_seeds: Some(sink),
            ..Self::new()
        }
    }

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
            autoreset_mode: Default::default(),
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
            reset_seeds: None,
        }
    }
}

#[async_trait]
impl crate::Env for SmokeEnv {
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
        req: spaces::request::ResetRequest,
    ) -> std::result::Result<spaces::request::ResetResult, spaces::EnvRuntimeError> {
        if let Some(seeds) = &self.reset_seeds {
            seeds.lock().await.push(req.seed);
        }
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
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<spaces::SpaceValue>> {
        self.predicts.fetch_add(1, Ordering::SeqCst);
        // One action per lane; the action space is a Uint8 Box[1] (SmokeEnv).
        Ok((0..observation.num_envs)
            .map(|_| {
                spaces::SpaceValue::Box(
                    spaces::Tensor::from_vec(vec![0u8], vec![1], spaces::DType::Uint8).unwrap(),
                )
            })
            .collect())
    }

    async fn on_close(&mut self) -> Result<()> {
        self.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Spawn a bound model server, returning its resolved port and the serve task.
fn spawn_bound_server(bound: BoundModelServer) -> (u16, tokio::task::JoinHandle<Result<()>>) {
    let port = match bound.local_addr().clone() {
        BindAddress::Tcp { port, .. } => port,
        other => panic!("expected tcp bind address, got {other:?}"),
    };
    assert_ne!(port, 0, "port 0 must resolve to a real port");
    let server = tokio::spawn(async move { bound.serve().await });
    (port, server)
}

/// Await a serve task that should exit cleanly after a shutdown request.
async fn shutdown_and_join(server: tokio::task::JoinHandle<Result<()>>) {
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn user_set_base_seed_reaches_the_env_reset_seeds() {
    async fn run_with_base_seed(base_seed: Option<i64>) -> Vec<Option<i64>> {
        let reset_seeds = Arc::new(Mutex::new(Vec::new()));
        let env = SmokeEnv::recording(Arc::clone(&reset_seeds));
        // Bind first so the listener is accepting before run_local connects.
        let bound = crate::EnvServer::new(env)
            .bind(BindAddress::Tcp {
                host: "127.0.0.1".to_string(),
                port: 0,
            })
            .await
            .unwrap();
        let address = bound.local_addr().to_string();
        let server = tokio::spawn(async move { bound.serve().await });

        let mut options = RunLocalOptions::parse(&address).unwrap().for_episodes(1);
        options.base_seed = base_seed;

        ModelWorker::new(SmokeModel {
            predicts: Arc::new(AtomicUsize::new(0)),
            closes: Arc::new(AtomicUsize::new(0)),
        })
        .run_local_async(options)
        .await
        .unwrap();

        // The served env stays up across client sessions (see the close()
        // contract); abort the server task now that the run is done.
        server.abort();

        reset_seeds.lock().await.clone()
    }

    // With a user-set base_seed, the env's first reset receives a concrete
    // (deterministic) seed rather than None.
    let seeded = run_with_base_seed(Some(4242)).await;
    assert!(
        seeded.first().map(Option::is_some).unwrap_or(false),
        "expected a concrete reset seed, got {seeded:?}"
    );

    // Determinism: the same base_seed yields the same reset seed.
    let seeded_again = run_with_base_seed(Some(4242)).await;
    assert_eq!(
        seeded.first(),
        seeded_again.first(),
        "the same base_seed must produce the same reset seed"
    );

    // Without a base_seed, no seed is injected (the env decides).
    let unseeded = run_with_base_seed(None).await;
    assert_eq!(
        unseeded.first(),
        Some(&None),
        "no base_seed must leave the reset seed unset, got {unseeded:?}"
    );
}

#[test]
fn run_local_and_serve_options_cover_all_axes() {
    // run_local: address + for_episodes + base_seed axes via one options struct.
    let run = RunLocalOptions::parse("tcp://env:50051")
        .unwrap()
        .for_episodes(5)
        .base_seed(123);
    assert_eq!(run.max_episodes, Some(5));
    assert_eq!(run.base_seed, Some(123));
    assert_eq!(
        run.env_address,
        ConnectAddress::parse("tcp://env:50051").unwrap()
    );

    // Defaults: unbounded, no seed.
    let default_run = RunLocalOptions::new(ConnectAddress::parse("tcp://env:1").unwrap());
    assert_eq!(default_run.max_episodes, None);
    assert_eq!(default_run.base_seed, None);

    // serve: address + token + serve options axes via one options struct.
    let serve = ServeModelOptions::parse("tcp://0.0.0.0:50061")
        .unwrap()
        .token("secret")
        .serve_options(ServeOptions {
            allow_remote_shutdown: true,
            ..ServeOptions::default()
        });
    assert_eq!(serve.token, "secret");
    assert!(serve.serve.allow_remote_shutdown);

    // No token => empty (auth disabled), default serve options.
    let default_serve = ServeModelOptions::new(BindAddress::parse("tcp://0.0.0.0:1").unwrap());
    assert!(default_serve.token.is_empty());
    assert_eq!(default_serve.serve, ServeOptions::default());
}

#[tokio::test]
async fn served_model_configure_route_requires_env_spec() {
    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::ConfigureRoute(ConfigureRouteRequest {
                context: Some(rlmesh_proto::model::v1::PredictContext {
                    session_id: "session-1".to_string(),
                    route_id: "route-1".to_string(),
                    request_id: "configure-1".to_string(),
                    ..Default::default()
                }),
                env_spec: None,
                ..Default::default()
            })),
            request_id: "configure-1".to_string(),
        },
        Arc::new(Mutex::new(SmokeModel {
            predicts: Arc::new(AtomicUsize::new(0)),
            closes: Arc::new(AtomicUsize::new(0)),
        })),
        None,
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
        None,
        Arc::new(Mutex::new(HashMap::new())),
        Arc::new(Mutex::new(HashMap::from([(
            "session-1:route-1".to_string(),
            ModelRouteConfig {
                env_contract: Some(std::sync::Arc::new(SmokeEnv::new().env_contract)),
                floor: None,
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

// NOTE: the former `served_model_predict_rejects_route_wider_than_opened_route`
// test was removed in the C8 EnvSpec split. The model now receives only the
// stable `EnvSpec` (no `num_envs`), so the configured route no longer carries a
// lane bound to clamp predict slots against; the per-predict lane count is taken
// from the request's slots. There is no contract-derived width to reject.
#[tokio::test]
async fn served_model_predict_uses_slot_count_as_lane_count() {
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
        None,
        Arc::new(Mutex::new(HashMap::new())),
        Arc::new(Mutex::new(HashMap::from([(
            "session-1:route-1".to_string(),
            ModelRouteConfig {
                env_contract: Some(std::sync::Arc::new(SmokeEnv::new().env_contract)),
                floor: None,
            },
        )]))),
    )
    .await;

    // A two-slot predict against a configured route now succeeds: the model
    // takes the lane count from the slots rather than a contract bound.
    assert!(matches!(
        response.kind,
        Some(join_response::Kind::Predict(_))
    ));
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
                floor: None,
            },
        ),
        (
            "session-1:route-2".to_string(),
            ModelRouteConfig {
                env_contract: None,
                floor: None,
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
        None,
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
            kind: Some(join_request::Kind::Close(CloseParticipantRequest {
                reason: "session complete".to_string(),
            })),
            request_id: "close-1".to_string(),
        },
        Arc::clone(&handler),
        None,
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
        .serve_async(
            ServeModelOptions::new(BindAddress::Tcp {
                host: "127.0.0.1".to_string(),
                port,
            })
            .token("route-token")
            .serve_options(ServeOptions {
                allow_remote_shutdown: true,
                ..ServeOptions::default()
            }),
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

    shutdown_and_join(server).await;

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
                crate::env::WireEnvAdapter::new(crate::env::ScalarEnvAdapter::new(SmokeEnv::new())),
            ))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap()
    });

    let mut client = rlmesh_grpc::EnvClient::connect(&env_address).await.unwrap();
    client.handshake().await.unwrap();

    // The public adapter lets external RuntimeDriver embedders drive a remote
    // env without re-implementing the take_last_endpoint_total_ns choreography.
    let mut adapter = crate::EnvClientRuntimeEnv::new(client);
    let reset = adapter
        .reset(rlmesh_proto::env::v1::ResetRequest {
            seeds: vec![7],
            options: None,
            timeout_ms: 0,
            env_indices: vec![],
        })
        .await
        .expect("adapter reset must succeed");
    assert!(
        reset.endpoint_total_ns.is_some(),
        "adapter must encapsulate and surface the per-call endpoint duration"
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
    .bind_async(
        ServeModelOptions::new(BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        })
        .token("route-token")
        .serve_options(ServeOptions {
            allow_remote_shutdown: true,
            ..ServeOptions::default()
        }),
    )
    .await
    .unwrap();

    // The OS-assigned port is known before serving begins.
    let (port, server) = spawn_bound_server(bound);

    // No poll-connect race: the resolved address is immediately usable.
    let address = format!("tcp://127.0.0.1:{port}");
    let mut client = rlmesh_grpc::ModelClient::connect(&address, "route-token")
        .await
        .unwrap();
    client.handshake().await.unwrap();
    let shutdown = client.shutdown("test complete").await.unwrap();
    assert!(shutdown.accepted);

    shutdown_and_join(server).await;
    assert_eq!(closes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn remote_model_connects_resets_and_predicts() {
    // The public RemoteModel mirrors RemoteEnv: connect to a served policy,
    // begin an episode with reset(), then drive it by hand with predict(). It
    // configures the route once from the env contract and round-trips the
    // observation/action through the value codec.
    let predicts = Arc::new(AtomicUsize::new(0));
    let closes = Arc::new(AtomicUsize::new(0));
    let bound = ModelWorker::new(SmokeModel {
        predicts: Arc::clone(&predicts),
        closes: Arc::clone(&closes),
    })
    .bind_async(
        ServeModelOptions::new(BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        })
        .serve_options(ServeOptions {
            allow_remote_shutdown: true,
            ..ServeOptions::default()
        }),
    )
    .await
    .unwrap();
    let (port, server) = spawn_bound_server(bound);

    let address = format!("tcp://127.0.0.1:{port}");
    let env_contract = SmokeEnv::new().env_contract;
    let mut model = crate::RemoteModel::connect(&address, env_contract)
        .await
        .expect("model server did not start");

    let observe = || {
        spaces::SpaceValue::Box(
            spaces::Tensor::from_vec(vec![5], vec![1], spaces::DType::Uint8).unwrap(),
        )
    };

    model.reset();
    // SmokeModel returns the raw action byte vec![0]; it decodes against the
    // contract's Uint8 Box action space.
    let action = model.predict(observe()).await.unwrap();
    assert!(matches!(action, spaces::SpaceValue::Box(_)));
    // A second predict on the same episode does not re-configure the route.
    model.predict(observe()).await.unwrap();
    assert_eq!(predicts.load(Ordering::SeqCst), 2);

    model.close().await.unwrap();
    // close() now ends only this route (CloseRoute), leaving the bidi stream
    // open; dropping the model closes it so the server can drain on shutdown.
    drop(model);

    // The route was configured exactly once; a fresh client could still connect.
    let mut shutdown_client = rlmesh_grpc::ModelClient::connect(&address, "")
        .await
        .unwrap();
    shutdown_client.handshake().await.unwrap();
    assert!(shutdown_client.shutdown("done").await.unwrap().accepted);
    shutdown_and_join(server).await;
}

#[tokio::test]
async fn remote_model_reconciles_three_way_floor_and_pins_route() {
    // The runtime is client to both peers. connect_with_env_offer handshakes the
    // model, reconciles the three-way floor against the env's offer, and pins the
    // route to it. With a single generation/edition the floor is the build's
    // values, and configure_route succeeds (the served model honors the floor).
    let predicts = Arc::new(AtomicUsize::new(0));
    let closes = Arc::new(AtomicUsize::new(0));
    let bound = ModelWorker::new(SmokeModel {
        predicts: Arc::clone(&predicts),
        closes: Arc::clone(&closes),
    })
    .bind_async(
        ServeModelOptions::new(BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        })
        .serve_options(ServeOptions {
            allow_remote_shutdown: true,
            ..ServeOptions::default()
        }),
    )
    .await
    .unwrap();
    let (port, server) = spawn_bound_server(bound);

    let address = format!("tcp://127.0.0.1:{port}");
    // An env that offers exactly this build's edition.
    let env_offer = rlmesh_proto::SessionOffer::new(
        &[rlmesh_proto::CURRENT_WORKFLOW_EDITION],
        &[rlmesh_proto::capabilities::SPACES_CORE_V1],
    );
    let mut model = crate::RemoteModel::connect_with_env_offer(
        &address,
        "",
        SmokeEnv::new().env_contract,
        env_offer,
    )
    .await
    .expect("model server did not start");

    let floor = model.session_floor();
    assert_eq!(
        floor.selected_workflow_edition,
        rlmesh_proto::CURRENT_WORKFLOW_EDITION
    );

    // The route configures (the pinned floor is accepted by the served model).
    model.reset();
    let action = model
        .predict(spaces::SpaceValue::Box(
            spaces::Tensor::from_vec(vec![5], vec![1], spaces::DType::Uint8).unwrap(),
        ))
        .await
        .unwrap();
    assert!(matches!(action, spaces::SpaceValue::Box(_)));
    drop(model);

    let mut shutdown_client = rlmesh_grpc::ModelClient::connect(&address, "")
        .await
        .unwrap();
    shutdown_client.handshake().await.unwrap();
    assert!(shutdown_client.shutdown("done").await.unwrap().accepted);
    shutdown_and_join(server).await;
}

#[tokio::test]
async fn remote_model_fails_fast_when_no_three_way_floor() {
    // If the env offers no edition this runtime/model can speak, the session
    // fails before any route is configured, with a diagnostic naming all three
    // offers. (The model handshake still happens; the floor reconciliation is
    // what fails.)
    let predicts = Arc::new(AtomicUsize::new(0));
    let closes = Arc::new(AtomicUsize::new(0));
    let bound = ModelWorker::new(SmokeModel {
        predicts: Arc::clone(&predicts),
        closes: Arc::clone(&closes),
    })
    .bind_async(
        ServeModelOptions::new(BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        })
        .serve_options(ServeOptions {
            allow_remote_shutdown: true,
            ..ServeOptions::default()
        }),
    )
    .await
    .unwrap();
    let (port, server) = spawn_bound_server(bound);

    let address = format!("tcp://127.0.0.1:{port}");
    // The env offers only a future edition no peer here implements.
    let env_offer = rlmesh_proto::SessionOffer::new(&["2099.01"], &[]);
    let message = match crate::RemoteModel::connect_with_env_offer(
        &address,
        "",
        SmokeEnv::new().env_contract,
        env_offer,
    )
    .await
    {
        Ok(_) => panic!("a session with no mutual edition must fail to connect"),
        Err(err) => err.to_string(),
    };
    assert!(
        message.contains("no mutual session floor") && message.contains("2099.01"),
        "expected a three-way floor diagnostic naming the offers, got: {message}"
    );

    server.abort();
}

#[tokio::test]
async fn remote_model_predict_requires_reset() {
    let predicts = Arc::new(AtomicUsize::new(0));
    let closes = Arc::new(AtomicUsize::new(0));
    let bound = ModelWorker::new(SmokeModel {
        predicts: Arc::clone(&predicts),
        closes: Arc::clone(&closes),
    })
    .bind_async(
        ServeModelOptions::new(BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        })
        .serve_options(ServeOptions {
            allow_remote_shutdown: true,
            ..ServeOptions::default()
        }),
    )
    .await
    .unwrap();
    let (port, server) = spawn_bound_server(bound);

    let address = format!("tcp://127.0.0.1:{port}");
    let mut model = crate::RemoteModel::connect(&address, SmokeEnv::new().env_contract)
        .await
        .unwrap();

    // predict() before reset() is a usage error, and no predict reaches the server.
    let err = model
        .predict(spaces::SpaceValue::Box(
            spaces::Tensor::from_vec(vec![0], vec![1], spaces::DType::Uint8).unwrap(),
        ))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("reset()"));
    assert_eq!(predicts.load(Ordering::SeqCst), 0);

    server.abort();
}

#[tokio::test]
async fn two_remote_models_in_one_process_use_distinct_route_keys() {
    // Two RemoteModels connected to the same server from one process must not
    // collide on the served model's session_id:route_id-keyed caches. route_id
    // is stable per instance ("remote-model"), so the session id is what keeps
    // the keys distinct; a regression would make both clients share one key and
    // clobber each other's contract/adapter/lifecycle.
    #[derive(Clone)]
    struct KeyRecordingModel {
        keys: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl ModelHandler for KeyRecordingModel {
        async fn predict(
            &mut self,
            observation: ModelObservation,
        ) -> Result<Vec<spaces::SpaceValue>> {
            let key = format!(
                "{}:{}",
                observation.route.session_id, observation.route.route_id
            );
            self.keys.lock().await.push(key);
            Ok(vec![spaces::SpaceValue::Box(
                spaces::Tensor::from_vec(vec![0u8], vec![1], spaces::DType::Uint8).unwrap(),
            )])
        }
    }

    let keys = Arc::new(Mutex::new(Vec::new()));
    let bound = ModelWorker::new(KeyRecordingModel {
        keys: Arc::clone(&keys),
    })
    .bind_async(
        ServeModelOptions::new(BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        })
        .serve_options(ServeOptions {
            allow_remote_shutdown: true,
            ..ServeOptions::default()
        }),
    )
    .await
    .unwrap();
    let (port, server) = spawn_bound_server(bound);
    let address = format!("tcp://127.0.0.1:{port}");

    let observe = || {
        spaces::SpaceValue::Box(
            spaces::Tensor::from_vec(vec![5], vec![1], spaces::DType::Uint8).unwrap(),
        )
    };

    let mut first = crate::RemoteModel::connect(&address, SmokeEnv::new().env_contract)
        .await
        .unwrap();
    let mut second = crate::RemoteModel::connect(&address, SmokeEnv::new().env_contract)
        .await
        .unwrap();

    first.reset();
    first.predict(observe()).await.unwrap();
    second.reset();
    second.predict(observe()).await.unwrap();

    let recorded = keys.lock().await.clone();
    assert_eq!(recorded.len(), 2);
    assert_ne!(
        recorded[0], recorded[1],
        "two sessions in one process collided on a single route key: {recorded:?}"
    );

    first.close().await.unwrap();
    second.close().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn served_model_reports_grpc_health_serving() {
    use tonic_health::ServingStatus;
    use tonic_health::pb::HealthCheckRequest;
    use tonic_health::pb::health_client::HealthClient;

    let predicts = Arc::new(AtomicUsize::new(0));
    let closes = Arc::new(AtomicUsize::new(0));
    let bound = ModelWorker::new(SmokeModel {
        predicts: Arc::clone(&predicts),
        closes: Arc::clone(&closes),
    })
    .bind_async(
        ServeModelOptions::new(BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        })
        .token("route-token")
        .serve_options(ServeOptions {
            allow_remote_shutdown: true,
            ..ServeOptions::default()
        }),
    )
    .await
    .unwrap();
    let (port, server) = spawn_bound_server(bound);

    // The standard health service requires no route token (it is a distinct
    // gRPC service from ModelService) and reports overall health = SERVING.
    let channel = tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut health = HealthClient::new(channel);
    let response = health
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.status, ServingStatus::Serving as i32);

    let mut client =
        rlmesh_grpc::ModelClient::connect(&format!("tcp://127.0.0.1:{port}"), "route-token")
            .await
            .unwrap();
    assert!(client.shutdown("done").await.unwrap().accepted);
    shutdown_and_join(server).await;
}

/// A handler whose `predict` latency is controlled per-request by the slot
/// `step` field (`step == 0` sleeps `slow_delay`, otherwise returns promptly),
/// recording the order in which predicts *complete* and the per-route order in
/// which lifecycle hooks and predicts *enter* the critical section.
#[derive(Clone)]
struct OrderingHandler {
    slow_delay: Duration,
    /// `(route_id, step)` in handler-entry order for predicts.
    predict_order: Arc<Mutex<Vec<(String, i64)>>>,
    /// `(route_id, event)` where event is one of "reset"/"predict"/"close" in
    /// handler-entry order, to assert per-route lifecycle ordering.
    route_events: Arc<Mutex<Vec<(String, String)>>>,
}

#[async_trait]
impl ModelHandler for OrderingHandler {
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<spaces::SpaceValue>> {
        let route_id = observation.route.route_id.clone();
        let step = observation.step();
        self.predict_order
            .lock()
            .await
            .push((route_id.clone(), step));
        self.route_events
            .lock()
            .await
            .push((route_id, "predict".to_string()));
        if step == 0 {
            tokio::time::sleep(self.slow_delay).await;
        }
        Ok(vec![spaces::SpaceValue::Discrete(step)])
    }

    async fn on_reset(&mut self, observation: &ModelObservation) -> Result<()> {
        self.route_events
            .lock()
            .await
            .push((observation.route.route_id.clone(), "reset".to_string()));
        Ok(())
    }

    async fn on_episode_end(&mut self, _event: ModelEpisodeEnd) -> Result<()> {
        Ok(())
    }
}

/// Spin up a real served model on an ephemeral port and return its address plus
/// a low-level `ModelServiceClient` Join stream that lets the test send multiple
/// requests before reading any response (the public `ModelClient` is
/// single-flight by API and cannot overlap sends).
async fn bound_ordering_server(
    handler: OrderingHandler,
    predict_concurrency: Option<usize>,
) -> (
    tokio::task::JoinHandle<crate::Result<()>>,
    rlmesh_proto::model::v1::model_service_client::ModelServiceClient<tonic::transport::Channel>,
    u16,
) {
    let bound = ModelWorker::new(handler)
        .bind_async(
            ServeModelOptions::new(BindAddress::Tcp {
                host: "127.0.0.1".to_string(),
                port: 0,
            })
            .serve_options(ServeOptions {
                allow_remote_shutdown: true,
                predict_concurrency,
                ..ServeOptions::default()
            }),
        )
        .await
        .unwrap();
    let (port, server) = spawn_bound_server(bound);

    let channel = tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let client = rlmesh_proto::model::v1::model_service_client::ModelServiceClient::new(channel);
    (server, client, port)
}

fn predict_context(
    route_id: &str,
    request_id: &str,
    step: i64,
) -> rlmesh_proto::model::v1::PredictContext {
    rlmesh_proto::model::v1::PredictContext {
        session_id: "session".to_string(),
        route_id: route_id.to_string(),
        request_id: request_id.to_string(),
        slots: vec![rlmesh_proto::model::v1::PredictSlot {
            episode_id: format!("ep-{route_id}"),
            env_index: 0,
            step,
            reset: step == 0,
        }],
    }
}

fn configure_route_request(route_id: &str, request_id: &str) -> JoinRequest {
    JoinRequest {
        kind: Some(join_request::Kind::ConfigureRoute(ConfigureRouteRequest {
            context: Some(rlmesh_proto::model::v1::PredictContext {
                session_id: "session".to_string(),
                route_id: route_id.to_string(),
                request_id: request_id.to_string(),
                ..Default::default()
            }),
            env_spec: Some(rlmesh_proto::core::v1::EnvSpec {
                id: "Ordering-v0".to_string(),
                observation_space: None,
                // The typed worker encodes the action, so the route needs a real
                // action space; OrderingHandler returns Discrete.
                action_space: Some(rlmesh_proto::spaces::v1::SpaceSpec {
                    shape: vec![],
                    dtype: rlmesh_proto::spaces::v1::DType::Int64 as i32,
                    spec: Some(rlmesh_proto::spaces::v1::space_spec::Spec::Discrete(
                        rlmesh_proto::spaces::v1::DiscreteSpec { n: 1000, start: 0 },
                    )),
                }),
                metadata: None,
            }),
            ..Default::default()
        })),
        request_id: request_id.to_string(),
    }
}

fn predict_join_request(route_id: &str, request_id: &str, step: i64) -> JoinRequest {
    JoinRequest {
        kind: Some(join_request::Kind::Predict(PredictRequest {
            context: Some(predict_context(route_id, request_id, step)),
            observation: None,
        })),
        request_id: request_id.to_string(),
    }
}

#[tokio::test]
async fn pipelined_requests_complete_out_of_order() {
    // Under option (a) the handler mutex is held across `predict`, so two
    // *predicts* serialize at the handler. Pipelining still removes head-of-line
    // blocking for work that does not touch the handler: `ConfigureRoute` only
    // mutates the route table. A configure sent *after* a slow in-flight predict
    // therefore completes *first*. That is the out-of-order guarantee the
    // server delivers and that the single-flight design could not.
    let handler = OrderingHandler {
        slow_delay: Duration::from_millis(300),
        predict_order: Arc::new(Mutex::new(Vec::new())),
        route_events: Arc::new(Mutex::new(Vec::new())),
    };
    let (server, mut client, _port) = bound_ordering_server(handler, None).await;

    let (req_tx, req_rx) = mpsc::channel::<JoinRequest>(8);
    let request_stream = tokio_stream::wrappers::ReceiverStream::new(req_rx);
    let mut responses = client
        .join(tonic::Request::new(request_stream))
        .await
        .unwrap()
        .into_inner();

    // Configure the "slow" route and read its ack.
    req_tx
        .send(configure_route_request("slow", "cfg-slow"))
        .await
        .unwrap();
    let ack = responses.message().await.unwrap().unwrap();
    assert!(matches!(
        ack.kind,
        Some(join_response::Kind::ConfigureRoute(_))
    ));

    // Send a slow predict on the "slow" route, then a ConfigureRoute on a
    // different route. The configure response must arrive first.
    req_tx
        .send(predict_join_request("slow", "predict-slow", 0))
        .await
        .unwrap();
    req_tx
        .send(configure_route_request("other", "cfg-other"))
        .await
        .unwrap();

    let first = responses.message().await.unwrap().unwrap();
    assert_eq!(
        first.request_id, "cfg-other",
        "a configure must not be head-of-line-blocked by an in-flight slow predict"
    );
    assert!(matches!(
        first.kind,
        Some(join_response::Kind::ConfigureRoute(_))
    ));
    let second = responses.message().await.unwrap().unwrap();
    assert_eq!(second.request_id, "predict-slow");

    drop(req_tx);
    let _ = responses.message().await;
    server.abort();
}

#[tokio::test]
async fn pipelined_predicts_preserve_per_route_order() {
    let route_events = Arc::new(Mutex::new(Vec::new()));
    let handler = OrderingHandler {
        slow_delay: Duration::from_millis(150),
        predict_order: Arc::new(Mutex::new(Vec::new())),
        route_events: Arc::clone(&route_events),
    };
    let (server, mut client, _port) = bound_ordering_server(handler, None).await;

    let (req_tx, req_rx) = mpsc::channel::<JoinRequest>(8);
    let request_stream = tokio_stream::wrappers::ReceiverStream::new(req_rx);
    let mut responses = client
        .join(tonic::Request::new(request_stream))
        .await
        .unwrap()
        .into_inner();

    req_tx
        .send(configure_route_request("r", "cfg"))
        .await
        .unwrap();
    let ack = responses.message().await.unwrap().unwrap();
    assert!(matches!(
        ack.kind,
        Some(join_response::Kind::ConfigureRoute(_))
    ));

    // Two predicts on the same route: the first is slow (step 0, reset), the
    // second fast (step 1). Per-route order must keep predict step 0 before 1.
    req_tx
        .send(predict_join_request("r", "p0", 0))
        .await
        .unwrap();
    req_tx
        .send(predict_join_request("r", "p1", 1))
        .await
        .unwrap();

    // Same-route responses also arrive in order.
    let first = responses.message().await.unwrap().unwrap();
    assert_eq!(first.request_id, "p0");
    let second = responses.message().await.unwrap().unwrap();
    assert_eq!(second.request_id, "p1");

    drop(req_tx);
    let _ = responses.message().await;

    let events = route_events.lock().await.clone();
    // For route "r": reset (from step-0 predict) then predict p0 then predict p1.
    let r_events: Vec<&str> = events
        .iter()
        .filter(|(route, _)| route == "r")
        .map(|(_, event)| event.as_str())
        .collect();
    assert_eq!(
        r_events,
        vec!["reset", "predict", "predict"],
        "per-route lifecycle order must match send order: {events:?}"
    );
    server.abort();
}

#[tokio::test]
async fn close_drains_after_in_flight_same_route_predict() {
    let route_events = Arc::new(Mutex::new(Vec::new()));
    let predict_order = Arc::new(Mutex::new(Vec::new()));
    let handler = OrderingHandler {
        slow_delay: Duration::from_millis(250),
        predict_order: Arc::clone(&predict_order),
        route_events: Arc::clone(&route_events),
    };
    let (server, mut client, _port) = bound_ordering_server(handler, None).await;

    let (req_tx, req_rx) = mpsc::channel::<JoinRequest>(8);
    let request_stream = tokio_stream::wrappers::ReceiverStream::new(req_rx);
    let mut responses = client
        .join(tonic::Request::new(request_stream))
        .await
        .unwrap()
        .into_inner();

    req_tx
        .send(configure_route_request("r", "cfg"))
        .await
        .unwrap();
    let _ = responses.message().await.unwrap().unwrap();

    // A slow predict followed immediately by a whole-session Close. The Close
    // barrier must not overtake the in-flight predict.
    req_tx
        .send(predict_join_request("r", "p0", 0))
        .await
        .unwrap();
    req_tx
        .send(JoinRequest {
            kind: Some(join_request::Kind::Close(CloseParticipantRequest {
                reason: "done".to_string(),
            })),
            request_id: "close".to_string(),
        })
        .await
        .unwrap();

    let first = responses.message().await.unwrap().unwrap();
    assert_eq!(
        first.request_id, "p0",
        "the in-flight predict must complete before Close drains"
    );
    let second = responses.message().await.unwrap().unwrap();
    assert_eq!(second.request_id, "close");
    assert!(matches!(second.kind, Some(join_response::Kind::Close(_))));

    // The predict's handler entry happened before the Close drain.
    let order = predict_order.lock().await.clone();
    assert_eq!(order, vec![("r".to_string(), 0)]);

    drop(req_tx);
    server.abort();
}

#[tokio::test]
async fn public_client_predict_concurrent_demuxes_overlapping_predicts() {
    // End-to-end: the high-level ModelClient issues two overlapping predicts via
    // predict_concurrent against the real pipelined server; both must resolve to
    // their own response (demux by request_id), even though they overlap.
    let handler = OrderingHandler {
        slow_delay: Duration::from_millis(100),
        predict_order: Arc::new(Mutex::new(Vec::new())),
        route_events: Arc::new(Mutex::new(Vec::new())),
    };
    let bound = ModelWorker::new(handler)
        .bind_async(
            ServeModelOptions::new(BindAddress::Tcp {
                host: "127.0.0.1".to_string(),
                port: 0,
            })
            .token("tok")
            .serve_options(ServeOptions {
                allow_remote_shutdown: true,
                ..ServeOptions::default()
            }),
        )
        .await
        .unwrap();
    let (port, server) = spawn_bound_server(bound);

    let address = format!("tcp://127.0.0.1:{port}");
    let mut client = rlmesh_grpc::ModelClient::connect(&address, "tok")
        .await
        .unwrap();
    client.handshake().await.unwrap();
    client
        .configure_route(ConfigureRouteRequest {
            context: Some(rlmesh_proto::model::v1::PredictContext {
                session_id: "s".to_string(),
                route_id: "r".to_string(),
                request_id: "cfg".to_string(),
                ..Default::default()
            }),
            env_spec: Some(rlmesh_proto::core::v1::EnvSpec {
                id: "Ordering-v0".to_string(),
                observation_space: None,
                // The typed worker encodes the action, so the route needs a real
                // action space; OrderingHandler returns Discrete.
                action_space: Some(rlmesh_proto::spaces::v1::SpaceSpec {
                    shape: vec![],
                    dtype: rlmesh_proto::spaces::v1::DType::Int64 as i32,
                    spec: Some(rlmesh_proto::spaces::v1::space_spec::Spec::Discrete(
                        rlmesh_proto::spaces::v1::DiscreteSpec { n: 1000, start: 0 },
                    )),
                }),
                metadata: None,
            }),
            ..Default::default()
        })
        .await
        .unwrap();

    let client = Arc::new(client);
    let make_predict = |request_id: &str, step: i64| PredictRequest {
        context: Some(rlmesh_proto::model::v1::PredictContext {
            session_id: "s".to_string(),
            route_id: "r".to_string(),
            request_id: request_id.to_string(),
            slots: vec![rlmesh_proto::model::v1::PredictSlot {
                episode_id: "ep".to_string(),
                env_index: 0,
                step,
                reset: step == 0,
            }],
        }),
        observation: None,
    };

    let c1 = Arc::clone(&client);
    let p1 = make_predict("predict-1", 0);
    let first = tokio::spawn(async move { c1.predict_concurrent(p1).await });
    let c2 = Arc::clone(&client);
    let p2 = make_predict("predict-2", 1);
    let second = tokio::spawn(async move { c2.predict_concurrent(p2).await });

    let r1 = first.await.unwrap().unwrap();
    let r2 = second.await.unwrap().unwrap();
    assert_eq!(r1.context.unwrap().request_id, "predict-1");
    assert_eq!(r2.context.unwrap().request_id, "predict-2");

    server.abort();
}

#[tokio::test]
async fn pipelined_idle_activity_stays_balanced() {
    // Every request emits exactly one Started/Finished pair. If pipelining ever
    // leaked an unbalanced Started, the idle-shutdown counter would never return
    // to zero and the idle timer would never fire. We assert the server *does*
    // idle-shut-down shortly after a batch of pipelined predicts drains, which is
    // only possible if every pair balanced.
    let handler = OrderingHandler {
        slow_delay: Duration::from_millis(20),
        predict_order: Arc::new(Mutex::new(Vec::new())),
        route_events: Arc::new(Mutex::new(Vec::new())),
    };
    let bound = ModelWorker::new(handler)
        .bind_async(
            ServeModelOptions::new(BindAddress::Tcp {
                host: "127.0.0.1".to_string(),
                port: 0,
            })
            .serve_options(ServeOptions {
                idle_timeout: Some(Duration::from_millis(150)),
                ..ServeOptions::default()
            }),
        )
        .await
        .unwrap();
    let (port, server) = spawn_bound_server(bound);

    {
        let channel = tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{port}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        let mut client =
            rlmesh_proto::model::v1::model_service_client::ModelServiceClient::new(channel);
        let (req_tx, req_rx) = mpsc::channel::<JoinRequest>(16);
        let request_stream = tokio_stream::wrappers::ReceiverStream::new(req_rx);
        let mut responses = client
            .join(tonic::Request::new(request_stream))
            .await
            .unwrap()
            .into_inner();
        req_tx
            .send(configure_route_request("r", "cfg"))
            .await
            .unwrap();
        let _ = responses.message().await.unwrap().unwrap();
        // Fire several overlapping predicts.
        for i in 0..5 {
            req_tx
                .send(predict_join_request("r", &format!("p{i}"), i as i64))
                .await
                .unwrap();
        }
        for _ in 0..5 {
            let _ = responses.message().await.unwrap().unwrap();
        }
        // Drop the stream so no further activity is generated.
        drop(req_tx);
        let _ = responses.message().await;
    }

    // With balanced activity, the idle timer fires and the server shuts down.
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .expect("server must idle-shut-down (balanced idle activity)")
        .unwrap()
        .unwrap();
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
