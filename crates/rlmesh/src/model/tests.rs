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
    /// A smoke env that records the reset seed it was asked to use into `sink`.
    #[cfg(not(feature = "lockstep"))]
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

// The v1 run path, which the v2 build no longer dials.
#[cfg(not(feature = "lockstep"))]
#[tokio::test]
async fn user_set_base_seed_reaches_the_env_reset_seeds() {
    async fn run_with_base_seed(base_seed: Option<i64>) -> Vec<Option<i64>> {
        let reset_seeds = Arc::new(Mutex::new(Vec::new()));
        let env = SmokeEnv::recording(Arc::clone(&reset_seeds));
        // Bind first so the listener is accepting before run_local connects.
        let bound = crate::binding::env::WireEnvServer::new(env)
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

/// A run with no explicit budget runs one episode, not "until the env ends".
// The v1 run path, which the v2 build no longer dials.
#[cfg(not(feature = "lockstep"))]
#[tokio::test]
async fn run_local_defaults_to_one_episode() {
    let bound = crate::binding::env::WireEnvServer::new(SmokeEnv::new())
        .bind(BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        })
        .await
        .unwrap();
    let address = bound.local_addr().to_string();
    let server = tokio::spawn(async move { bound.serve().await });

    let report = ModelWorker::new(SmokeModel {
        predicts: Arc::new(AtomicUsize::new(0)),
        closes: Arc::new(AtomicUsize::new(0)),
    })
    .run_local_async(RunLocalOptions::parse(&address).unwrap())
    .await
    .unwrap();
    server.abort();
    assert_eq!(report.total_episodes, 1);
    assert_eq!(report.episodes.len(), 1);
}

/// A run borrows the model: `on_close` fires only when the caller asks with
/// `close_model(true)`.
// The v1 run path, which the v2 build no longer dials.
#[cfg(not(feature = "lockstep"))]
#[tokio::test]
async fn run_local_borrows_the_model_unless_asked_to_close_it() {
    let bound = crate::binding::env::WireEnvServer::new(SmokeEnv::new())
        .bind(BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        })
        .await
        .unwrap();
    let address = bound.local_addr().to_string();
    let server = tokio::spawn(async move { bound.serve().await });

    assert!(!RunLocalOptions::parse(&address).unwrap().close_model);
    let closes = Arc::new(AtomicUsize::new(0));
    for (close_model, expected) in [(None, 0), (Some(false), 0), (Some(true), 1)] {
        closes.store(0, Ordering::SeqCst);
        let mut options = RunLocalOptions::parse(&address).unwrap();
        if let Some(close_model) = close_model {
            options = options.close_model(close_model);
        }
        ModelWorker::new(SmokeModel {
            predicts: Arc::new(AtomicUsize::new(0)),
            closes: Arc::clone(&closes),
        })
        .run_local_async(options)
        .await
        .unwrap();
        assert_eq!(
            closes.load(Ordering::SeqCst),
            expected,
            "close_model={close_model:?}"
        );
    }
    server.abort();
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

/// A served NEXT_STEP vector env with a per-lane terminal schedule. It mimics gym
/// autoreset: a lane terminates at its scheduled step, then on the FOLLOWING step
/// the env delivers a fresh (non-terminal, reward-0) observation. The env never
/// mints ids — the runtime pushes them down and the env server's tracker adopts
/// them; this env only drives the masks the tracker keys on.
#[cfg(not(feature = "lockstep"))]
struct AutoresetVectorEnv {
    obs_space: spaces::SpaceSpec,
    action_space: spaces::SpaceSpec,
    env_contract: spaces::EnvContract,
    terminal_after: Vec<usize>,
    lane_step: Vec<usize>,
    pending: Vec<bool>,
}

#[cfg(not(feature = "lockstep"))]
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
#[cfg(not(feature = "lockstep"))]
impl crate::binding::env::VectorEnv for AutoresetVectorEnv {
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
        _req: crate::binding::env::VectorResetRequest,
    ) -> std::result::Result<crate::binding::env::VectorResetResult, spaces::EnvRuntimeError> {
        let n = self.terminal_after.len();
        self.lane_step = vec![0; n];
        self.pending = vec![false; n];
        Ok(crate::binding::env::VectorResetResult {
            observations: (0..n).map(|_| Self::lane_obs()).collect(),
            info: None,
            episode_ids: Vec::new(),
        })
    }

    async fn step(
        &mut self,
        _req: crate::binding::env::VectorStepRequest,
    ) -> std::result::Result<crate::binding::env::VectorStepResult, spaces::EnvRuntimeError> {
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
        Ok(crate::binding::env::VectorStepResult {
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
    ) -> std::result::Result<crate::binding::env::VectorCloseResult, spaces::EnvRuntimeError> {
        Ok(crate::binding::env::VectorCloseResult::default())
    }
}

/// Records the per-row episode ids it sees in predict and the ids it is asked to
/// evict via the explicit ResetAdapter op.
#[cfg(not(feature = "lockstep"))]
#[derive(Clone, Default)]
struct IdRecordingHandler {
    predict_ids: Arc<Mutex<Vec<Vec<String>>>>,
    evicted_ids: Arc<Mutex<Vec<String>>>,
}

#[cfg(not(feature = "lockstep"))]
#[async_trait]
impl ModelHandler for IdRecordingHandler {
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<spaces::SpaceValue>> {
        self.predict_ids
            .lock()
            .await
            .push(observation.episode_ids());
        Ok((0..observation.num_envs)
            .map(|_| spaces::SpaceValue::Discrete(0))
            .collect())
    }

    async fn reset_adapter(&mut self, _env_id: &str, episode_ids: Vec<String>) -> Result<()> {
        self.evicted_ids.lock().await.extend(episode_ids);
        Ok(())
    }
}

// The v1 run path, which the v2 build no longer dials.
#[cfg(not(feature = "lockstep"))]
#[tokio::test]
async fn run_local_refuses_chunking_on_a_vector_env() {
    // Chunk replay is whole-batch: one lane ending mid-chunk discards every
    // lane's buffered frames. The served engine cannot see this (ResolveAdapter
    // carries no lane count), so `run_local` — where the handshake's num_envs and
    // the pinned horizon are both in scope — refuses the pairing before resolving.
    let bound = crate::binding::env::WireVectorEnvServer::new(AutoresetVectorEnv::new(vec![2, 3]))
        .bind(BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        })
        .await
        .unwrap();
    let address = bound.local_addr().to_string();
    let server = tokio::spawn(async move { bound.serve().await });

    let error = ModelWorker::new(IdRecordingHandler::default())
        .run_local_async(
            RunLocalOptions::parse(&address)
                .unwrap()
                .for_episodes(2)
                .execution_horizon(2),
        )
        .await
        .expect_err("a chunked vector route is refused");
    server.abort();

    let message = error.to_string();
    assert!(
        message.contains("num_envs=2") && message.contains("execution_horizon=2"),
        "the refusal should name both numbers: {message}"
    );
}

// The v1 run path, which the v2 build no longer dials.
#[cfg(not(feature = "lockstep"))]
#[tokio::test]
async fn next_step_episode_ids_round_trip_through_the_real_env_server() {
    // End-to-end across the REAL gRPC stack (runtime driver -> EnvClient -> env
    // server -> EpisodeTracker), not a mock: the runtime mints UUIDv7 ids and
    // pushes them down, the env server adopts them and (under NEXT_STEP) rolls a
    // lane on autoreset, and the model sees the rolled ids — proving the
    // pending_roll push-down + adoption + echo + resolve loop works on the wire.
    let bound = crate::binding::env::WireVectorEnvServer::new(AutoresetVectorEnv::new(vec![2, 3]))
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
/// `RunLocalOptions::token` is the only way an in-process run can reach a
/// token-protected env server, so the env client must be built with it.
// The v1 run path, which the v2 build no longer dials.
#[cfg(not(feature = "lockstep"))]
#[tokio::test]
async fn run_local_options_token_reaches_a_token_protected_env() {
    let bound = crate::binding::env::WireEnvServer::new(SmokeEnv::new())
        .bind_with_ext(
            BindAddress::Tcp {
                host: "127.0.0.1".to_string(),
                port: 0,
            },
            ServeOptions::default(),
            crate::ext::EnvServerOptions {
                token: Some("env-token".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let address = bound.local_addr().to_string();
    let server = tokio::spawn(async move { bound.serve().await });

    let untokened = ModelWorker::new(SmokeModel {
        predicts: Arc::new(AtomicUsize::new(0)),
        closes: Arc::new(AtomicUsize::new(0)),
    })
    .run_local_async(RunLocalOptions::parse(&address).unwrap().for_episodes(1))
    .await
    .expect_err("a token-protected env must reject an untokened run_local");
    assert!(
        untokened.to_string().contains("rejected the API key"),
        "expected an unauthenticated rejection, got {untokened}"
    );

    let predicts = Arc::new(AtomicUsize::new(0));
    ModelWorker::new(SmokeModel {
        predicts: Arc::clone(&predicts),
        closes: Arc::new(AtomicUsize::new(0)),
    })
    .run_local_async(
        RunLocalOptions::parse(&address)
            .unwrap()
            .token("env-token")
            .for_episodes(1),
    )
    .await
    .unwrap();
    assert!(predicts.load(Ordering::SeqCst) > 0);

    server.abort();
}
