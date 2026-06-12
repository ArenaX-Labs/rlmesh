#[cfg(unix)]
use crate::env::WireEnvAdapter;
use crate::*;
#[cfg(unix)]
use rlmesh_grpc::env::env_service;
use rlmesh_grpc::error::{EnvError, EnvErrorCode};
#[cfg(unix)]
use tokio::net::UnixListener;
#[cfg(unix)]
use tokio_stream::wrappers::UnixListenerStream;

#[cfg(unix)]
struct DummyEnv {
    obs_space: spaces::SpaceSpec,
    action_space: spaces::SpaceSpec,
    env_contract: spaces::EnvContract,
}

#[cfg(unix)]
impl DummyEnv {
    fn new() -> Self {
        let obs_space = spaces::spaces::BoxSpaceBuilder::scalar(-1.0, 1.0, vec![4])
            .dtype(spaces::DType::Uint8)
            .build()
            .unwrap();
        let action_space = spaces::spaces::DiscreteBuilder::new(2).build().unwrap();
        let env_contract = spaces::EnvContract {
            id: "DummyEnv-v1".to_string(),
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

    fn observation(&self) -> spaces::Tensor {
        spaces::Tensor::from_vec(vec![0, 1, 2, 3], vec![4], spaces::DType::Uint8).unwrap()
    }
}

#[cfg(unix)]
#[async_trait::async_trait]
impl Env for DummyEnv {
    fn observation_space(&self) -> &spaces::SpaceSpec {
        &self.obs_space
    }

    fn action_space(&self) -> &spaces::SpaceSpec {
        &self.action_space
    }

    fn env_contract(&self) -> &spaces::EnvContract {
        &self.env_contract
    }

    fn num_envs(&self) -> usize {
        1
    }

    async fn reset(
        &mut self,
        _req: ResetRequest,
    ) -> std::result::Result<ResetResult, spaces::EnvRuntimeError> {
        Ok(ResetResult {
            observations: vec![spaces::SpaceValue::Box(self.observation())],
            info: None,
            episode_ids: vec!["ep-1".to_string()],
        })
    }

    async fn step(
        &mut self,
        _req: StepRequest,
    ) -> std::result::Result<StepResult, spaces::EnvRuntimeError> {
        Ok(StepResult {
            observations: vec![spaces::SpaceValue::Box(self.observation())],
            rewards: vec![1.0],
            terminated: vec![false],
            truncated: vec![false],
            info: None,
            completed_episodes: vec![],
            episode_ids: vec!["ep-1".to_string()],
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
    ) -> std::result::Result<CloseResult, spaces::EnvRuntimeError> {
        Ok(CloseResult {
            final_episodes: vec![],
        })
    }
}

#[tokio::test]
#[cfg(unix)]
#[ignore = "requires local socket bind support"]
async fn remote_env_smoke_test() {
    let socket_path =
        std::env::temp_dir().join(format!("rlmesh-facade-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path).unwrap();

    let server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(env_service(WireEnvAdapter::new(DummyEnv::new())))
            .serve_with_incoming(UnixListenerStream::new(listener))
            .await
            .unwrap()
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let mut env = RemoteEnv::connect(&format!("unix://{}", socket_path.display()))
        .await
        .unwrap();
    let spec = env.env_contract();
    assert_eq!(spec.id, "DummyEnv-v1");

    let reset = env.reset(ResetRequest::default()).await.unwrap();
    assert_eq!(reset.observations.len(), 1);

    let step = env
        .step(StepRequest {
            actions: vec![spaces::SpaceValue::Discrete(0)],
            timeout_ms: 0,
        })
        .await
        .unwrap();
    assert_eq!(step.rewards, vec![1.0]);

    let _ = env.render(spaces::RenderRequest::default()).await.unwrap();
    let _ = env.close().await.unwrap();

    server.abort();
    let _ = std::fs::remove_file(socket_path);
}

#[test]
fn reexports_space_types() {
    let spec = spaces::EnvContract {
        id: "test-env".to_string(),
        observation_space: Some(
            spaces::spaces::BoxSpaceBuilder::scalar(-1.0, 1.0, vec![4])
                .build()
                .unwrap(),
        ),
        action_space: Some(spaces::spaces::DiscreteBuilder::new(2).build().unwrap()),
        num_envs: 1,
        ..Default::default()
    };

    assert!(spec.observation_space.is_some());
    assert!(spec.action_space.is_some());
}

#[test]
fn parses_facade_addresses() {
    assert_eq!(
        ConnectAddress::parse("localhost:50051").unwrap(),
        ConnectAddress::Tcp("tcp://localhost:50051".to_string())
    );
    #[cfg(unix)]
    assert_eq!(
        ConnectAddress::parse("unix:///tmp/rlmesh.sock").unwrap(),
        ConnectAddress::Unix("/tmp/rlmesh.sock".into())
    );
    assert_eq!(
        BindAddress::parse("7000").unwrap(),
        BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 7000,
        }
    );
}

#[cfg(not(unix))]
#[test]
fn rejects_unix_facade_addresses_on_windows() {
    let err = ConnectAddress::parse("unix:///tmp/rlmesh.sock").unwrap_err();
    assert!(err.to_string().contains("use tcp://host:port instead"));

    let err = BindAddress::parse("unix:///tmp/rlmesh.sock").unwrap_err();
    assert!(err.to_string().contains("use tcp://host:port instead"));
}

#[test]
fn facade_error_mapping_hides_core_error_types() {
    let err = Error::from(rlmesh_grpc::error::Error::Environment(EnvError::new(
        EnvErrorCode::InvalidAction,
        "bad action",
    )));
    assert_eq!(
        err,
        Error::Environment(EnvironmentError {
            code: ErrorCode::InvalidAction,
            message: "bad action".to_string(),
            is_recoverable: true,
        })
    );
}

#[test]
fn curated_spaces_facade_separates_the_two_request_families() {
    // The crate-root request types are the vectorized env-layer family.
    let env_layer = crate::ResetRequest {
        seeds: vec![1, 2, 3],
        ..Default::default()
    };
    assert_eq!(env_layer.seeds.len(), 3);

    // The single-env request family is namespaced under `spaces::request`,
    // resolving the previous same-named `rlmesh::ResetRequest` vs
    // `rlmesh::spaces::ResetRequest` glob collision.
    let single_env = crate::spaces::request::ResetRequest {
        seed: Some(7),
        ..Default::default()
    };
    assert_eq!(single_env.seed, Some(7));

    // The two families are genuinely distinct types reachable through curated
    // (non-glob) paths.
    fn assert_distinct<A: 'static, B: 'static>() {
        assert_ne!(
            std::any::TypeId::of::<A>(),
            std::any::TypeId::of::<B>(),
            "env-layer and single-env ResetRequest must be distinct types"
        );
    }
    assert_distinct::<crate::ResetRequest, crate::spaces::request::ResetRequest>();
}
