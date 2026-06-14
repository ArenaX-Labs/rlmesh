use crate::*;
use rlmesh_grpc::error::{EnvError, EnvErrorCode};

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
