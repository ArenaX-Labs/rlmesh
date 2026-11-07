use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio_stream::wrappers::TcpListenerStream;

use super::*;
use crate::{BindAddress, ConnectAddress, Result, ServeOptions, binding::spaces};

struct SmokeEnv {
    obs_space: spaces::SpaceSpec,
    action_space: spaces::SpaceSpec,
    env_contract: spaces::EnvContract,
    reset_seeds: Option<Arc<Mutex<Vec<Option<i64>>>>>,
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
impl crate::binding::env::Env for SmokeEnv {
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

#[test]
fn episode_budget_resolves_like_python() {
    let base = RunLocalOptions::parse("tcp://env:1").unwrap();
    // Neither given: one episode (Python's `episodes=None`).
    assert_eq!(base.episode_budget().unwrap(), 1);
    // Seeds only: one episode per seed.
    let seeded = base.clone().episode_seeds(vec![3, 4, 5]);
    assert_eq!(seeded.episode_budget().unwrap(), 3);
    // An explicit count wins, zero included (runs nothing).
    assert_eq!(base.clone().for_episodes(7).episode_budget().unwrap(), 7);
    assert_eq!(base.clone().for_episodes(0).episode_budget().unwrap(), 0);
    // Both given: the counts must match.
    assert_eq!(seeded.clone().for_episodes(3).episode_budget().unwrap(), 3);
    let error = seeded.for_episodes(2).episode_budget().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not match len(episode_seeds)=3"),
        "{error}"
    );
}

/// `for_episodes(0)` runs nothing and never dials the env; a mismatched seed
/// count is refused before dialing.
#[tokio::test]
async fn run_local_zero_episodes_runs_nothing() {
    // Nothing listens on port 1: dialing would fail the run.
    let unreachable = "tcp://127.0.0.1:1";
    let predicts = Arc::new(AtomicUsize::new(0));
    let report = ModelWorker::new(SmokeModel {
        predicts: Arc::clone(&predicts),
        closes: Arc::new(AtomicUsize::new(0)),
    })
    .run_local_async(RunLocalOptions::parse(unreachable).unwrap().for_episodes(0))
    .await
    .unwrap();
    assert_eq!(report.total_episodes, 0);
    assert!(report.episodes.is_empty());
    assert_eq!(predicts.load(Ordering::SeqCst), 0);

    let error = ModelWorker::new(SmokeModel {
        predicts: Arc::clone(&predicts),
        closes: Arc::new(AtomicUsize::new(0)),
    })
    .run_local_async(
        RunLocalOptions::parse(unreachable)
            .unwrap()
            .for_episodes(2)
            .episode_seeds(vec![1]),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("does not match"), "{error}");
}

#[test]
fn run_local_options_cover_all_axes() {
    let run = RunLocalOptions::parse("tcp://env:50051")
        .unwrap()
        .for_episodes(5)
        .base_seed(123)
        .execution_horizon(8)
        .prefetch_lead(2);
    assert_eq!(run.max_episodes, Some(5));
    assert_eq!(run.base_seed, Some(123));
    assert_eq!(run.execution_horizon, 8);
    assert_eq!(run.prefetch_lead, 2);
    assert_eq!(
        run.env_address,
        ConnectAddress::parse("tcp://env:50051").unwrap()
    );

    let default_run = RunLocalOptions::new(ConnectAddress::parse("tcp://env:1").unwrap());
    assert_eq!(default_run.max_episodes, None);
    assert_eq!(default_run.base_seed, None);
    assert_eq!(default_run.execution_horizon, 1);
    assert_eq!(default_run.prefetch_lead, 0);
    assert_eq!(default_run.execution_horizon(0).execution_horizon, 1);
}

#[tokio::test]
async fn public_env_runtime_adapter_drives_a_remote_env_with_telemetry() {
    use crate::run::runtime::RuntimeEnv;

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let env_address = format!("tcp://{}", listener.local_addr().unwrap());
    let env_server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(rlmesh_grpc::env::env_service(
                crate::binding::env::WireLaneAdapter::new(vec![SmokeEnv::new()]).unwrap(),
            ))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap()
    });

    let mut client = rlmesh_grpc::EnvClient::connect(&env_address).await.unwrap();
    client.handshake().await.unwrap();

    // The public adapter lets external RuntimeDriver embedders drive a remote
    // env without re-implementing the take_last_endpoint_total_ns choreography.
    let mut adapter = super::local::EnvClientRuntimeEnv::new(client);
    let reset = adapter
        .reset(crate::run::runtime::EnvResetRequest {
            seeds: vec![7],
            options: None,
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

/// A declaration no peer can honor is refused at the env handshake, before any
/// Join stream opens, and the message names both sides' WANT and CAN.
///
/// The declaration is set on the bare `ServeOptions` field, which takes no
/// validation (the user-facing surfaces refuse such a value where it is typed),
/// so this is the path that MUST end in a clean negotiation refusal. `2020.01`
/// is below every edition any build offers, which makes the test independent of
/// whether this build is sealed or a prerelease cohort.
#[tokio::test]
async fn a_declared_edition_no_peer_can_run_is_refused_at_the_env_handshake() {
    let bound = crate::binding::env::WireEnvServer::new(SmokeEnv::new())
        .bind_with_options(
            BindAddress::Tcp {
                host: "127.0.0.1".to_string(),
                port: 0,
            },
            ServeOptions {
                workflow_edition: Some("2020.01".to_string()),
                ..ServeOptions::default()
            },
        )
        .await
        .unwrap();
    let address = bound.local_addr().to_string();
    let server = tokio::spawn(async move { bound.serve().await });

    let connect = crate::binding::env::EnvClientCore::connect(&address).await;
    server.abort();
    let Err(error) = connect else {
        panic!("the env declares an edition this build cannot run")
    };
    // A refused handshake is the ProtocolException class (RMT-12).
    assert_eq!(error.class(), crate::ErrorClass::ProtocolException);
    let error = error.to_string();

    assert!(
        error.contains("no mutual workflow edition with the env"),
        "{error}"
    );
    for half in [
        "env wants \"2020.01\"",
        "runtime wants",
        rlmesh_proto::CURRENT_WORKFLOW_EDITION,
    ] {
        assert!(error.contains(half), "expected {half:?} in: {error}");
    }
}
