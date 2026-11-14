use crate::data::{EnvError, EnvErrorCode};
use crate::*;

/// Session address used by env tests.
pub(crate) const ENV_ADDRESS: &str = ":0/env";

pub(crate) fn env_serve_options() -> ServeOptions {
    ServeOptions::default()
}

#[test]
fn reexports_space_types() {
    let spec = binding::spaces::EnvContract {
        id: "test-env".to_string(),
        observation_space: Some(
            binding::spaces::spaces::BoxSpaceBuilder::scalar(-1.0, 1.0, vec![4])
                .build()
                .unwrap(),
        ),
        action_space: Some(
            binding::spaces::spaces::DiscreteBuilder::new(2)
                .build()
                .unwrap(),
        ),
        num_envs: 1,
        ..Default::default()
    };

    assert!(spec.observation_space.is_some());
    assert!(spec.action_space.is_some());
}

#[test]
fn facade_error_mapping_hides_core_error_types() {
    let err = Error::Environment(EnvironmentError::from_wire(EnvError::new(
        EnvErrorCode::InvalidAction,
        "bad action",
    )));
    assert_eq!(
        err,
        Error::Environment(EnvironmentError {
            code: ErrorCode::InvalidAction,
            message: "bad action".to_string(),
            is_recoverable: true,
            class: None,
        })
    );
}

#[test]
fn curated_spaces_facade_separates_the_two_request_families() {
    // The crate-root request types are the scalar env family.
    let env_layer = crate::binding::env::ResetRequest {
        seed: Some(7),
        ..Default::default()
    };
    assert_eq!(env_layer.seed, Some(7));

    // The vector env request family is explicit.
    let vector_env = crate::binding::env::VectorResetRequest {
        seeds: vec![1, 2, 3],
        ..Default::default()
    };
    assert_eq!(vector_env.seeds.len(), 3);

    // The two families are genuinely distinct types reachable through curated
    // (non-glob) paths.
    fn assert_distinct<A: 'static, B: 'static>() {
        assert_ne!(
            std::any::TypeId::of::<A>(),
            std::any::TypeId::of::<B>(),
            "env-layer and single-env ResetRequest must be distinct types"
        );
    }
    assert_distinct::<crate::binding::env::ResetRequest, crate::binding::env::VectorResetRequest>();
}

#[test]
fn describe_and_reset_keys_match_python() {
    assert_eq!(crate::DESCRIBE_METADATA_KEY, "rlmesh.describe.v1");
    assert_eq!(crate::DESCRIBE_SCHEMA_VERSION, 1);
    assert_eq!(crate::ENV_RESET_OPTIONS_KEY, "rlmesh.env.v1.reset_options");
}

/// The cargo features the crate docs list (`//! - `name`` under
/// `# Features`), and the ones the manifest declares (`default` aside).
#[test]
fn the_crate_docs_list_exactly_the_manifest_features() {
    let mut documented: Vec<&str> = include_str!("lib.rs")
        .lines()
        .skip_while(|line| *line != "//! # Features")
        .skip(1)
        .take_while(|line| !line.starts_with("//! # "))
        .filter_map(|line| line.strip_prefix("//! - `"))
        .filter_map(|line| line.split('`').next())
        .collect();
    let mut declared: Vec<&str> = include_str!("../Cargo.toml")
        .lines()
        .skip_while(|line| *line != "[features]")
        .skip(1)
        .take_while(|line| !line.starts_with('['))
        .filter_map(|line| line.split_once(" = ").map(|(name, _)| name.trim()))
        .filter(|name| !name.starts_with('#') && *name != "default")
        .collect();
    documented.sort_unstable();
    declared.sort_unstable();
    assert!(!declared.is_empty());
    assert_eq!(
        documented, declared,
        "the `# Features` list in crates/rlmesh/src/lib.rs and [features] in its Cargo.toml differ"
    );
}
