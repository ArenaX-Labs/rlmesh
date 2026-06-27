use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use rlmesh_proto::model::v1::{
    AdapterContext, CloseParticipantRequest, GroupedPredictRequest, GroupedPredictResponse,
    GroupedPredictResult, JoinRequest, PredictRequest, ReleaseAdapterRequest, ResolveAdapterRequest,
    grouped_predict_result, join_request, join_response,
};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::TcpListenerStream;

use super::server::{ModelRouteConfig, handle_model_request};
use super::*;
use crate::{BindAddress, ConnectAddress, Result, ServeOptions, spaces};

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
    // (derived) seed rather than None.
    let seeded = run_with_base_seed(Some(4242)).await;
    assert!(
        seeded.first().map(Option::is_some).unwrap_or(false),
        "expected a concrete reset seed, got {seeded:?}"
    );

    // base_seed is reproducible: the derived per-lane reset seed depends only on
    // (base_seed, session_id, reset_generation, lane) — NOT the per-attach random
    // env_id — so two runs with the same base_seed in the same session replay the
    // identical derived seed.
    let seeded_again = run_with_base_seed(Some(4242)).await;
    assert_eq!(
        seeded.first(),
        seeded_again.first(),
        "the same base_seed must derive the same reset seed (reproducibility)"
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
async fn served_model_resolve_adapter_requires_env_spec() {
    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::ResolveAdapter(ResolveAdapterRequest {
                context: Some(AdapterContext {
                    session_id: "session-1".to_string(),
                    env_id: "env-1".to_string(),
                    request_id: "resolve-1".to_string(),
                }),
                env_spec: None,
                ..Default::default()
            })),
            request_id: "resolve-1".to_string(),
        },
        Arc::new(Mutex::new(SmokeModel {
            predicts: Arc::new(AtomicUsize::new(0)),
            closes: Arc::new(AtomicUsize::new(0)),
        })),
        None,
        Arc::new(Mutex::new(HashMap::new())),
    )
    .await;

    assert!(matches!(response.kind, Some(join_response::Kind::Error(_))));
}

#[tokio::test]
async fn served_model_predict_mirrors_route_context() {
    let context = AdapterContext {
        session_id: "session-1".to_string(),
        env_id: "env-1".to_string(),
        request_id: "request-1".to_string(),
    };
    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::Predict(PredictRequest {
                context: Some(context.clone()),
                observation: None,
                episode_ids: vec!["episode-1".to_string()],
            })),
            request_id: "request-1".to_string(),
        },
        Arc::new(Mutex::new(SmokeModel {
            predicts: Arc::new(AtomicUsize::new(0)),
            closes: Arc::new(AtomicUsize::new(0)),
        })),
        None,
        Arc::new(Mutex::new(HashMap::from([(
            "env-1".to_string(),
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
// stable `EnvSpec` (no `num_envs`), so the resolved adapter no longer carries a
// lane bound to clamp the batch against; the per-predict row count is taken from
// the request's `episode_ids`. There is no contract-derived width to reject.
#[tokio::test]
async fn served_model_predict_uses_episode_id_count_as_lane_count() {
    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::Predict(PredictRequest {
                context: Some(AdapterContext {
                    session_id: "session-1".to_string(),
                    env_id: "env-1".to_string(),
                    request_id: "request-1".to_string(),
                }),
                observation: None,
                episode_ids: vec!["episode-0".to_string(), "episode-1".to_string()],
            })),
            request_id: "request-1".to_string(),
        },
        Arc::new(Mutex::new(SmokeModel {
            predicts: Arc::new(AtomicUsize::new(0)),
            closes: Arc::new(AtomicUsize::new(0)),
        })),
        None,
        Arc::new(Mutex::new(HashMap::from([(
            "env-1".to_string(),
            ModelRouteConfig {
                env_contract: Some(std::sync::Arc::new(SmokeEnv::new().env_contract)),
                floor: None,
            },
        )]))),
    )
    .await;

    // A two-row predict against a resolved adapter now succeeds: the model takes
    // the row count from the `episode_ids` vector rather than a contract bound.
    assert!(matches!(
        response.kind,
        Some(join_response::Kind::Predict(_))
    ));
}

/// Returns a per-lane action that conforms to the adapter's pinned action space,
/// keyed by env_id, so a grouped predict across adapters with *different* action
/// spaces round-trips each group against its own space. Records
/// `(env_id, episode_id)` in handler-entry order.
#[derive(Clone, Default)]
struct PerRouteActionHandler {
    seen: Arc<Mutex<Vec<(String, String)>>>,
}

#[async_trait]
impl ModelHandler for PerRouteActionHandler {
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<spaces::SpaceValue>> {
        self.seen.lock().await.push((
            observation.route.env_id.clone(),
            observation.episode_id().to_string(),
        ));
        // "env-disc" pinned a Discrete action space; every other env here is the
        // SmokeEnv Uint8 Box[1]. Returning the action that conforms to *this*
        // env's space proves each group is finished against its own spec — the
        // Discrete action would fail conformance if encoded against the Box env.
        let action = if observation.route.env_id == "env-disc" {
            spaces::SpaceValue::Discrete(0)
        } else {
            spaces::SpaceValue::Box(
                spaces::Tensor::from_vec(vec![0u8], vec![1], spaces::DType::Uint8).unwrap(),
            )
        };
        Ok((0..observation.num_envs).map(|_| action.clone()).collect())
    }
}

/// A native env contract whose action space is `Discrete(1000)`, built from a
/// proto `EnvSpec` so the test does not hand-assemble native specs.
fn discrete_action_contract() -> spaces::EnvContract {
    rlmesh_grpc::wire::env_spec_from_proto(rlmesh_proto::core::v1::EnvSpec {
        id: "Disc-v0".to_string(),
        observation_space: None,
        action_space: Some(rlmesh_proto::spaces::v1::SpaceSpec {
            shape: vec![],
            dtype: rlmesh_proto::spaces::v1::DType::Int64 as i32,
            spec: Some(rlmesh_proto::spaces::v1::space_spec::Spec::Discrete(
                rlmesh_proto::spaces::v1::DiscreteSpec { n: 1000, start: 0 },
            )),
        }),
        metadata: None,
    })
    .expect("discrete env spec builds a contract")
}

/// One group of a grouped predict: a routed, single-row `PredictRequest`.
fn grouped_member(env_id: &str, request_id: &str, episode_id: &str) -> PredictRequest {
    PredictRequest {
        context: Some(AdapterContext {
            session_id: "session-1".to_string(),
            env_id: env_id.to_string(),
            request_id: request_id.to_string(),
        }),
        observation: None,
        episode_ids: vec![episode_id.to_string()],
    }
}

fn expect_group_response(
    result: &GroupedPredictResult,
) -> rlmesh_proto::model::v1::PredictResponse {
    match result.outcome.as_ref().expect("group has an outcome") {
        grouped_predict_result::Outcome::Response(response) => response.clone(),
        grouped_predict_result::Outcome::Error(error) => {
            panic!(
                "expected a per-group response, got error: {}",
                error.message
            )
        }
    }
}

#[tokio::test]
async fn grouped_predict_processes_each_group_against_its_own_route() {
    // Two groups on routes with DIFFERENT action spaces (Box vs Discrete). One
    // grouped request must produce one ordered result per group, each mirroring
    // its route and encoded against that route's own space.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let handler = Arc::new(Mutex::new(PerRouteActionHandler {
        seen: Arc::clone(&seen),
    }));
    let configs = Arc::new(Mutex::new(HashMap::from([
        (
            "env-box".to_string(),
            ModelRouteConfig {
                env_contract: Some(Arc::new(SmokeEnv::new().env_contract)),
                floor: None,
            },
        ),
        (
            "env-disc".to_string(),
            ModelRouteConfig {
                env_contract: Some(Arc::new(discrete_action_contract())),
                floor: None,
            },
        ),
    ])));

    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::GroupedPredict(GroupedPredictRequest {
                groups: vec![
                    grouped_member("env-box", "p-box", "ep-box"),
                    grouped_member("env-disc", "p-disc", "ep-disc"),
                ],
            })),
            request_id: "grouped-1".to_string(),
        },
        Arc::clone(&handler),
        None,
        Arc::clone(&configs),
    )
    .await;

    let results = match response.kind {
        Some(join_response::Kind::GroupedPredict(GroupedPredictResponse { results })) => results,
        other => panic!("expected grouped predict response, got {other:?}"),
    };
    assert_eq!(results.len(), 2, "one result per group, in order");

    let box_response = expect_group_response(&results[0]);
    assert_eq!(box_response.context.as_ref().unwrap().env_id, "env-box");
    assert_eq!(
        box_response.actions.len(),
        1,
        "the box group's action was encoded against its own action space (no chunking)"
    );
    let disc_response = expect_group_response(&results[1]);
    assert_eq!(disc_response.context.as_ref().unwrap().env_id, "env-disc");
    assert_eq!(disc_response.actions.len(), 1);

    // Both groups reached the handler in group order, each decoded against its
    // own adapter config.
    assert_eq!(
        *seen.lock().await,
        vec![
            ("env-box".to_string(), "ep-box".to_string()),
            ("env-disc".to_string(), "ep-disc".to_string()),
        ]
    );
}

#[tokio::test]
async fn grouped_predict_isolates_a_single_group_failure() {
    // The second group references an UNCONFIGURED route. It must report its own
    // error without sinking the first group's result (per-group results).
    let seen = Arc::new(Mutex::new(Vec::new()));
    let handler = Arc::new(Mutex::new(PerRouteActionHandler {
        seen: Arc::clone(&seen),
    }));
    let configs = Arc::new(Mutex::new(HashMap::from([(
        "env-box".to_string(),
        ModelRouteConfig {
            env_contract: Some(Arc::new(SmokeEnv::new().env_contract)),
            floor: None,
        },
    )])));

    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::GroupedPredict(GroupedPredictRequest {
                groups: vec![
                    grouped_member("env-box", "p-box", "ep-box"),
                    grouped_member("env-missing", "p-missing", "ep-missing"),
                ],
            })),
            request_id: "grouped-2".to_string(),
        },
        Arc::clone(&handler),
        None,
        Arc::clone(&configs),
    )
    .await;

    let results = match response.kind {
        Some(join_response::Kind::GroupedPredict(GroupedPredictResponse { results })) => results,
        other => panic!("expected grouped predict response, got {other:?}"),
    };
    assert_eq!(results.len(), 2);
    assert!(
        matches!(
            results[0].outcome.as_ref().unwrap(),
            grouped_predict_result::Outcome::Response(_)
        ),
        "the configured group still succeeds"
    );
    match results[1].outcome.as_ref().unwrap() {
        grouped_predict_result::Outcome::Error(error) => {
            assert!(
                error.message.contains("was not resolved"),
                "got: {}",
                error.message
            );
        }
        other => panic!("expected an error for the unresolved group, got {other:?}"),
    }
    // Only the configured group reached the handler.
    assert_eq!(
        *seen.lock().await,
        vec![("env-box".to_string(), "ep-box".to_string())]
    );
}

/// A `ModelRouteSetup` that records which envs it was asked to release, so a
/// teardown test can assert `release_adapter` fired for exactly the right envs.
#[derive(Clone, Default)]
struct ReleaseRecordingSetup {
    released: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl ModelRouteSetup for ReleaseRecordingSetup {
    async fn resolve_adapter(
        &self,
        _env_id: &str,
        _env_contract: &spaces::EnvContract,
        _action_horizon: u32,
    ) -> Result<()> {
        Ok(())
    }

    async fn release_adapter(&self, env_id: &str) -> Result<()> {
        self.released.lock().await.push(env_id.to_string());
        Ok(())
    }
}

#[tokio::test]
async fn served_model_release_adapter_tears_down_only_its_env() {
    // ReleaseAdapter removes exactly the named env's adapter (config dropped +
    // its setup released), leaving every other env's adapter intact.
    let released = Arc::new(Mutex::new(Vec::new()));
    let route_setup: Arc<dyn ModelRouteSetup> = Arc::new(ReleaseRecordingSetup {
        released: Arc::clone(&released),
    });
    let route_configs = Arc::new(Mutex::new(HashMap::from([
        (
            "env-1".to_string(),
            ModelRouteConfig {
                env_contract: None,
                floor: None,
            },
        ),
        (
            "env-2".to_string(),
            ModelRouteConfig {
                env_contract: None,
                floor: None,
            },
        ),
    ])));

    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::ReleaseAdapter(ReleaseAdapterRequest {
                context: Some(AdapterContext {
                    session_id: "session-1".to_string(),
                    env_id: "env-1".to_string(),
                    request_id: "release-1".to_string(),
                }),
                reason: "env complete".to_string(),
            })),
            request_id: "release-1".to_string(),
        },
        Arc::new(Mutex::new(SmokeModel {
            predicts: Arc::new(AtomicUsize::new(0)),
            closes: Arc::new(AtomicUsize::new(0)),
        })),
        Some(Arc::clone(&route_setup)),
        Arc::clone(&route_configs),
    )
    .await;

    assert!(matches!(
        response.kind,
        Some(join_response::Kind::ReleaseAdapter(_))
    ));
    // The setup was torn down for exactly the released env.
    assert_eq!(*released.lock().await, vec!["env-1".to_string()]);
    // Only env-1's config was dropped; env-2 stays resolved.
    let route_configs = route_configs.lock().await;
    assert!(!route_configs.contains_key("env-1"));
    assert!(route_configs.contains_key("env-2"));
}

#[tokio::test]
async fn served_model_close_releases_every_adapter() {
    // Whole-session Close tears down every resolved adapter (config cleared +
    // each setup released) rather than leaking them for the server's lifetime.
    let released = Arc::new(Mutex::new(Vec::new()));
    let route_setup: Arc<dyn ModelRouteSetup> = Arc::new(ReleaseRecordingSetup {
        released: Arc::clone(&released),
    });
    let route_configs = Arc::new(Mutex::new(HashMap::from([
        (
            "env-1".to_string(),
            ModelRouteConfig {
                env_contract: None,
                floor: None,
            },
        ),
        (
            "env-2".to_string(),
            ModelRouteConfig {
                env_contract: None,
                floor: None,
            },
        ),
    ])));

    let response = handle_model_request(
        JoinRequest {
            kind: Some(join_request::Kind::Close(CloseParticipantRequest {
                reason: "session complete".to_string(),
            })),
            request_id: "close-1".to_string(),
        },
        Arc::new(Mutex::new(SmokeModel {
            predicts: Arc::new(AtomicUsize::new(0)),
            closes: Arc::new(AtomicUsize::new(0)),
        })),
        Some(Arc::clone(&route_setup)),
        Arc::clone(&route_configs),
    )
    .await;

    assert!(matches!(response.kind, Some(join_response::Kind::Close(_))));
    // Every adapter's setup was released (order is map iteration, so sort).
    let mut released = released.lock().await.clone();
    released.sort();
    assert_eq!(released, vec!["env-1".to_string(), "env-2".to_string()]);
    assert!(route_configs.lock().await.is_empty());
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
            episode_ids: vec![],
        })
        .await
        .expect("adapter reset must succeed");
    assert!(
        reset.endpoint_total_ns.is_some(),
        "adapter must encapsulate and surface the per-call endpoint duration"
    );

    env_server.abort();
}

/// A served NEXT_STEP vector env with a per-lane terminal schedule. It mimics gym
/// autoreset: a lane terminates at its scheduled step, then on the FOLLOWING step
/// the env delivers a fresh (non-terminal, reward-0) observation. The env never
/// mints ids — the runtime pushes them down and the env server's tracker adopts
/// them; this env only drives the masks the tracker keys on.
struct AutoresetVectorEnv {
    obs_space: spaces::SpaceSpec,
    action_space: spaces::SpaceSpec,
    env_contract: spaces::EnvContract,
    terminal_after: Vec<usize>,
    lane_step: Vec<usize>,
    pending: Vec<bool>,
}

impl AutoresetVectorEnv {
    fn new(terminal_after: Vec<usize>) -> Self {
        let n = terminal_after.len();
        let obs_space = spaces::spaces::BoxSpaceBuilder::scalar(-1.0, 1.0, vec![1])
            .dtype(spaces::DType::Float32)
            .build()
            .unwrap();
        let action_space = spaces::spaces::DiscreteBuilder::new(2).build().unwrap();
        let env_contract = spaces::EnvContract {
            id: "AutoresetVectorEnv-v0".to_string(),
            autoreset_mode: spaces::types::AutoresetMode::NextStep,
            observation_space: Some(obs_space.clone()),
            action_space: Some(action_space.clone()),
            metadata: None,
            render_mode: String::new(),
            num_envs: n as u32,
        };
        Self {
            obs_space,
            action_space,
            env_contract,
            lane_step: vec![0; n],
            pending: vec![false; n],
            terminal_after,
        }
    }

    fn lane_obs() -> spaces::SpaceValue {
        spaces::SpaceValue::Box(
            spaces::Tensor::from_vec(vec![0u8; 4], vec![1], spaces::DType::Float32).unwrap(),
        )
    }
}

#[async_trait]
impl crate::VectorEnv for AutoresetVectorEnv {
    fn observation_space(&self) -> &spaces::SpaceSpec {
        &self.obs_space
    }
    fn action_space(&self) -> &spaces::SpaceSpec {
        &self.action_space
    }
    fn num_envs(&self) -> usize {
        self.terminal_after.len()
    }
    fn env_contract(&self) -> &spaces::EnvContract {
        &self.env_contract
    }

    async fn reset(
        &mut self,
        _req: crate::VectorResetRequest,
    ) -> std::result::Result<crate::VectorResetResult, spaces::EnvRuntimeError> {
        let n = self.terminal_after.len();
        self.lane_step = vec![0; n];
        self.pending = vec![false; n];
        Ok(crate::VectorResetResult {
            observations: (0..n).map(|_| Self::lane_obs()).collect(),
            info: None,
            episode_ids: Vec::new(),
        })
    }

    async fn step(
        &mut self,
        _req: crate::VectorStepRequest,
    ) -> std::result::Result<crate::VectorStepResult, spaces::EnvRuntimeError> {
        let n = self.terminal_after.len();
        let mut rewards = vec![1.0; n];
        let mut terminated = vec![false; n];
        for lane in 0..n {
            if self.pending[lane] {
                // Fresh autoreset observation: non-terminal, reward 0.
                self.pending[lane] = false;
                self.lane_step[lane] = 0;
                rewards[lane] = 0.0;
            } else {
                self.lane_step[lane] += 1;
                if self.lane_step[lane] >= self.terminal_after[lane] {
                    terminated[lane] = true;
                    self.pending[lane] = true;
                }
            }
        }
        Ok(crate::VectorStepResult {
            observations: (0..n).map(|_| Self::lane_obs()).collect(),
            rewards,
            terminated,
            truncated: vec![false; n],
            info: None,
            completed_episodes: Vec::new(),
            episode_ids: Vec::new(),
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
    ) -> std::result::Result<crate::VectorCloseResult, spaces::EnvRuntimeError> {
        Ok(crate::VectorCloseResult::default())
    }
}

/// Records the per-row episode ids it sees in predict and the ids it is asked to
/// evict via the explicit ResetAdapter op.
#[derive(Clone, Default)]
struct IdRecordingHandler {
    predict_ids: Arc<Mutex<Vec<Vec<String>>>>,
    evicted_ids: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl ModelHandler for IdRecordingHandler {
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<spaces::SpaceValue>> {
        self.predict_ids.lock().await.push(observation.episode_ids());
        Ok((0..observation.num_envs)
            .map(|_| spaces::SpaceValue::Discrete(0))
            .collect())
    }

    async fn reset_adapter(&mut self, _env_id: &str, episode_ids: Vec<String>) -> Result<()> {
        self.evicted_ids.lock().await.extend(episode_ids);
        Ok(())
    }
}

#[tokio::test]
async fn next_step_episode_ids_round_trip_through_the_real_env_server() {
    // End-to-end across the REAL gRPC stack (runtime driver -> EnvClient -> env
    // server -> EpisodeTracker), not a mock: the runtime mints UUIDv7 ids and
    // pushes them down, the env server adopts them and (under NEXT_STEP) rolls a
    // lane on autoreset, and the model sees the rolled ids — proving the
    // pending_roll push-down + adoption + echo + resolve loop works on the wire.
    let bound = crate::VectorEnvServer::new(AutoresetVectorEnv::new(vec![2, 3]))
        .bind(BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        })
        .await
        .unwrap();
    let address = bound.local_addr().to_string();
    let server = tokio::spawn(async move { bound.serve().await });

    let handler = IdRecordingHandler::default();
    let predict_ids = Arc::clone(&handler.predict_ids);
    let evicted_ids = Arc::clone(&handler.evicted_ids);

    ModelWorker::new(handler)
        .run_local_async(RunLocalOptions::parse(&address).unwrap().for_episodes(6))
        .await
        .unwrap();
    server.abort();

    let predicts = predict_ids.lock().await.clone();
    assert!(!predicts.is_empty(), "the model ran at least one predict");
    // Every id the model ever saw is a runtime-minted UUIDv7 (round-tripped
    // through the real env server), never empty or an env placeholder.
    for row in &predicts {
        assert_eq!(row.len(), 2, "two lanes per predict");
        for id in row {
            assert_eq!(id.len(), 36, "episode id must be a UUID, got {id:?}");
            assert_eq!(id.as_bytes()[14], b'7', "UUIDv7 version nibble: {id:?}");
        }
    }
    // Lane 0 rolled: its id changed across the run (the autoreset boundary minted
    // a fresh id the env adopted), so the model saw more than one distinct lane-0 id.
    let lane0: std::collections::BTreeSet<&str> =
        predicts.iter().map(|row| row[0].as_str()).collect();
    assert!(
        lane0.len() >= 2,
        "lane 0's episode id must roll across autoreset, saw {lane0:?}"
    );
    // Completed episodes were evicted via ResetAdapter, each a real minted id.
    let evicted = evicted_ids.lock().await.clone();
    assert!(
        !evicted.is_empty(),
        "ResetAdapter must evict completed episodes"
    );
    for id in &evicted {
        assert_eq!(id.len(), 36, "evicted a UUID id, got {id:?}");
    }
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
    let env_offer = rlmesh_proto::SessionOffer::new(&[rlmesh_proto::CURRENT_WORKFLOW_EDITION]);
    let mut model = crate::RemoteModel::connect_with_env_offer(
        &address,
        "",
        SmokeEnv::new().env_contract,
        env_offer,
    )
    .await
    .expect("model server did not start");

    assert_eq!(
        model.selected_workflow_edition(),
        rlmesh_proto::CURRENT_WORKFLOW_EDITION
    );

    // The route configures (the pinned edition is accepted by the served model).
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
async fn remote_model_fails_fast_when_no_mutual_edition() {
    // If the env offers no edition the model can speak, the session fails before
    // any route is configured, with a diagnostic naming both offers. (The model
    // handshake still happens; the 2-way edition pick is what fails.)
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
    let env_offer = rlmesh_proto::SessionOffer::new(&["2099.01"]);
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
        message.contains("no mutual workflow edition") && message.contains("2099.01"),
        "expected an all-tiers edition diagnostic naming the offers, got: {message}"
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
async fn two_remote_models_in_one_process_use_distinct_env_keys() {
    // Two RemoteModels connected to the same server from one process must not
    // collide on the served model's env_id-keyed caches. Each RemoteModel mints
    // its own env_id (UUIDv7) on connect, so the keys are distinct; a regression
    // would make both clients share one key and clobber each other's
    // contract/adapter state.
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
            self.keys.lock().await.push(observation.route.env_id.clone());
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

/// A handler whose `predict` latency is controlled per-request by the row's
/// episode id (an id containing `"slow"` sleeps `slow_delay`, otherwise returns
/// promptly), recording the order in which predicts *enter* the critical section
/// (both globally and per-env) so a test can assert pipelining/ordering.
#[derive(Clone)]
struct OrderingHandler {
    slow_delay: Duration,
    /// `(env_id, request_id)` in handler-entry order for predicts.
    predict_order: Arc<Mutex<Vec<(String, String)>>>,
    /// `(env_id, event)` where event is one of "predict"/"close" in
    /// handler-entry order, to assert per-env ordering.
    route_events: Arc<Mutex<Vec<(String, String)>>>,
}

#[async_trait]
impl ModelHandler for OrderingHandler {
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<spaces::SpaceValue>> {
        let env_id = observation.route.env_id.clone();
        let request_id = observation.route.request_id.clone();
        let slow = observation.episode_id().contains("slow");
        self.predict_order
            .lock()
            .await
            .push((env_id.clone(), request_id));
        self.route_events
            .lock()
            .await
            .push((env_id, "predict".to_string()));
        if slow {
            tokio::time::sleep(self.slow_delay).await;
        }
        Ok(vec![spaces::SpaceValue::Discrete(0)])
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

fn resolve_adapter_request(env_id: &str, request_id: &str) -> JoinRequest {
    JoinRequest {
        kind: Some(join_request::Kind::ResolveAdapter(ResolveAdapterRequest {
            context: Some(AdapterContext {
                session_id: "session".to_string(),
                env_id: env_id.to_string(),
                request_id: request_id.to_string(),
            }),
            env_spec: Some(rlmesh_proto::core::v1::EnvSpec {
                id: "Ordering-v0".to_string(),
                observation_space: None,
                // The typed worker encodes the action, so the adapter needs a real
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

/// A single-row predict for `env_id`. `slow` rides on the row's episode id so
/// the [`OrderingHandler`] knows to sleep for it (the only way to mark a request
/// slow now that there is no positional slot/step on the wire).
fn predict_join_request(env_id: &str, request_id: &str, slow: bool) -> JoinRequest {
    let suffix = if slow { "-slow" } else { "" };
    JoinRequest {
        kind: Some(join_request::Kind::Predict(PredictRequest {
            context: Some(AdapterContext {
                session_id: "session".to_string(),
                env_id: env_id.to_string(),
                request_id: request_id.to_string(),
            }),
            observation: None,
            episode_ids: vec![format!("ep-{env_id}-{request_id}{suffix}")],
        })),
        request_id: request_id.to_string(),
    }
}

#[tokio::test]
async fn pipelined_requests_complete_out_of_order() {
    // Under option (a) the handler mutex is held across `predict`, so two
    // *predicts* serialize at the handler. Pipelining still removes head-of-line
    // blocking for work that does not touch the handler: `ResolveAdapter` only
    // mutates the adapter table. A resolve sent *after* a slow in-flight predict
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

    // Resolve the "slow" env's adapter and read its ack.
    req_tx
        .send(resolve_adapter_request("slow", "cfg-slow"))
        .await
        .unwrap();
    let ack = responses.message().await.unwrap().unwrap();
    assert!(matches!(
        ack.kind,
        Some(join_response::Kind::ResolveAdapter(_))
    ));

    // Send a slow predict on the "slow" env, then a ResolveAdapter on a
    // different env. The resolve response must arrive first.
    req_tx
        .send(predict_join_request("slow", "predict-slow", true))
        .await
        .unwrap();
    req_tx
        .send(resolve_adapter_request("other", "cfg-other"))
        .await
        .unwrap();

    let first = responses.message().await.unwrap().unwrap();
    assert_eq!(
        first.request_id, "cfg-other",
        "a resolve must not be head-of-line-blocked by an in-flight slow predict"
    );
    assert!(matches!(
        first.kind,
        Some(join_response::Kind::ResolveAdapter(_))
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
        .send(resolve_adapter_request("r", "cfg"))
        .await
        .unwrap();
    let ack = responses.message().await.unwrap().unwrap();
    assert!(matches!(
        ack.kind,
        Some(join_response::Kind::ResolveAdapter(_))
    ));

    // Two predicts on the same env: the first is slow, the second fast. Per-env
    // order must keep predict p0 before p1.
    req_tx.send(predict_join_request("r", "p0", true)).await.unwrap();
    req_tx
        .send(predict_join_request("r", "p1", false))
        .await
        .unwrap();

    // Same-env responses also arrive in order.
    let first = responses.message().await.unwrap().unwrap();
    assert_eq!(first.request_id, "p0");
    let second = responses.message().await.unwrap().unwrap();
    assert_eq!(second.request_id, "p1");

    drop(req_tx);
    let _ = responses.message().await;

    let events = route_events.lock().await.clone();
    // For env "r": predict p0 (slow) entered the handler before predict p1.
    let r_events: Vec<&str> = events
        .iter()
        .filter(|(env_id, _)| env_id == "r")
        .map(|(_, event)| event.as_str())
        .collect();
    assert_eq!(
        r_events,
        vec!["predict", "predict"],
        "per-env predict order must match send order: {events:?}"
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
        .send(resolve_adapter_request("r", "cfg"))
        .await
        .unwrap();
    let _ = responses.message().await.unwrap().unwrap();

    // A slow predict followed immediately by a whole-session Close. The Close
    // barrier must not overtake the in-flight predict.
    req_tx
        .send(predict_join_request("r", "p0", true))
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
    assert_eq!(order, vec![("r".to_string(), "p0".to_string())]);

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
        .resolve_adapter(ResolveAdapterRequest {
            context: Some(AdapterContext {
                session_id: "s".to_string(),
                env_id: "r".to_string(),
                request_id: "cfg".to_string(),
            }),
            env_spec: Some(rlmesh_proto::core::v1::EnvSpec {
                id: "Ordering-v0".to_string(),
                observation_space: None,
                // The typed worker encodes the action, so the adapter needs a real
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
    let make_predict = |request_id: &str, slow: bool| PredictRequest {
        context: Some(AdapterContext {
            session_id: "s".to_string(),
            env_id: "r".to_string(),
            request_id: request_id.to_string(),
        }),
        observation: None,
        episode_ids: vec![format!(
            "ep-{request_id}{}",
            if slow { "-slow" } else { "" }
        )],
    };

    let c1 = Arc::clone(&client);
    let p1 = make_predict("predict-1", true);
    let first = tokio::spawn(async move { c1.predict_concurrent(p1).await });
    let c2 = Arc::clone(&client);
    let p2 = make_predict("predict-2", false);
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
            .send(resolve_adapter_request("r", "cfg"))
            .await
            .unwrap();
        let _ = responses.message().await.unwrap().unwrap();
        // Fire several overlapping predicts (the first is slow).
        for i in 0..5 {
            req_tx
                .send(predict_join_request("r", &format!("p{i}"), i == 0))
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
