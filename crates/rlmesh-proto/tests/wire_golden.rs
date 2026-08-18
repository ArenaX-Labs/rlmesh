//! Serialized-message goldens for the frozen `rlmesh-wire-v1` encoding.
//!
//! Each test encodes a canonical, fully-populated instance of a key wire
//! message and asserts the exact byte vector. Unlike the regen-resettable text
//! baseline and round-trip tests, these bytes catch proto field-number/tag
//! drift even when both peers renumber in sync. If a test here fails you are
//! BREAKING THE WIRE CONTRACT: the fix is a protocol generation bump
//! (`rlmesh-wire-v2`), never an update of the expected bytes. Map-typed fields
//! carry at most a single entry so the encoding stays deterministic.

use prost::Message;
use rlmesh_proto::{core, env, model, spaces};

fn assert_frozen(name: &str, got: Vec<u8>, want: &[u8]) {
    assert_eq!(
        got.as_slice(),
        want,
        "{name}: frozen rlmesh-wire-v1 bytes changed (this is a wire break, not a golden \
         update); got {got:?}"
    );
}

fn space_value() -> spaces::v1::SpaceValue {
    spaces::v1::SpaceValue {
        leaves: vec![vec![1, 2, 3].into(), vec![4, 5].into()],
    }
}

fn adapter_context() -> model::v1::AdapterContext {
    model::v1::AdapterContext {
        session_id: "sess-1".to_string(),
        env_id: "env-1".to_string(),
        request_id: "req-1".to_string(),
    }
}

fn box_space_spec() -> spaces::v1::SpaceSpec {
    spaces::v1::SpaceSpec {
        shape: vec![2, 3],
        dtype: spaces::v1::DataType::Float32 as i32,
        spec: Some(spaces::v1::space_spec::Spec::Box(spaces::v1::BoxSpec {
            bounds: Some(spaces::v1::box_spec::Bounds::Uniform(
                spaces::v1::UniformBounds {
                    low: vec![0x00, 0x00, 0x80, 0xBF],
                    high: vec![0x00, 0x00, 0x80, 0x3F],
                },
            )),
        })),
    }
}

fn meta_map() -> spaces::v1::MetaMap {
    let mut entries = std::collections::HashMap::new();
    entries.insert(
        "k".to_string(),
        spaces::v1::MetaValue {
            kind: Some(spaces::v1::meta_value::Kind::Integer(7)),
        },
    );
    spaces::v1::MetaMap { entries }
}

fn predict_request() -> model::v1::PredictRequest {
    model::v1::PredictRequest {
        context: Some(adapter_context()),
        observation: Some(space_value()),
        episode_ids: vec!["ep-1".to_string(), "ep-2".to_string()],
        episode_seeds: vec![
            model::v1::EpisodeSeed {
                episode_id: "ep-1".to_string(),
                seed: Some(7),
            },
            model::v1::EpisodeSeed {
                episode_id: "ep-2".to_string(),
                seed: None,
            },
        ],
    }
}

fn episode_metadata() -> env::v1::EpisodeMetadata {
    env::v1::EpisodeMetadata {
        episode_id: "ep-1".to_string(),
        seed: Some(7),
        env_index: 3,
        step_count: 42,
        cumulative_reward: 1.25,
        terminated: true,
        truncated: true,
        start_timestamp_ns: 1_000,
        end_timestamp_ns: 2_000,
        final_info: Some(meta_map()),
    }
}

/// Hand-checkable anchor: `leaves` is field 1 wire-type LEN, so every leaf is
/// tag byte 0x0A = (1 << 3) | 2 followed by its length and raw bytes.
#[test]
fn spaces_space_value_bytes_are_frozen() {
    assert_frozen(
        "spaces.v1.SpaceValue",
        space_value().encode_to_vec(),
        &[10, 3, 1, 2, 3, 10, 2, 4, 5],
    );
}

#[test]
fn spaces_space_spec_bytes_are_frozen() {
    assert_frozen(
        "spaces.v1.SpaceSpec",
        box_space_spec().encode_to_vec(),
        &[
            10, 2, 2, 3, 16, 11, 82, 14, 18, 12, 10, 4, 0, 0, 128, 191, 18, 4, 0, 0, 128, 63,
        ],
    );
}

#[test]
fn model_predict_request_bytes_are_frozen() {
    assert_frozen(
        "model.v1.PredictRequest",
        predict_request().encode_to_vec(),
        &[
            10, 22, 10, 6, 115, 101, 115, 115, 45, 49, 18, 5, 101, 110, 118, 45, 49, 26, 5, 114,
            101, 113, 45, 49, 18, 9, 10, 3, 1, 2, 3, 10, 2, 4, 5, 26, 4, 101, 112, 45, 49, 26, 4,
            101, 112, 45, 50, 34, 8, 10, 4, 101, 112, 45, 49, 16, 7, 34, 6, 10, 4, 101, 112, 45,
            50,
        ],
    );
}

#[test]
fn model_resolve_adapter_request_bytes_are_frozen() {
    let message = model::v1::ResolveAdapterRequest {
        context: Some(adapter_context()),
        env_spec: Some(core::v1::EnvSpec {
            id: "env-spec-1".to_string(),
            action_space: Some(box_space_spec()),
            observation_space: Some(box_space_spec()),
            metadata: Some(meta_map()),
        }),
        selected_workflow_edition: "2026.06".to_string(),
        execution_horizon: 4,
    };
    assert_frozen(
        "model.v1.ResolveAdapterRequest",
        message.encode_to_vec(),
        &[
            10, 22, 10, 6, 115, 101, 115, 115, 45, 49, 18, 5, 101, 110, 118, 45, 49, 26, 5, 114,
            101, 113, 45, 49, 18, 71, 10, 10, 101, 110, 118, 45, 115, 112, 101, 99, 45, 49, 18, 22,
            10, 2, 2, 3, 16, 11, 82, 14, 18, 12, 10, 4, 0, 0, 128, 191, 18, 4, 0, 0, 128, 63, 26,
            22, 10, 2, 2, 3, 16, 11, 82, 14, 18, 12, 10, 4, 0, 0, 128, 191, 18, 4, 0, 0, 128, 63,
            34, 9, 10, 7, 10, 1, 107, 18, 2, 8, 7, 26, 7, 50, 48, 50, 54, 46, 48, 54, 32, 4,
        ],
    );
}

#[test]
fn model_join_request_bytes_are_frozen() {
    let message = model::v1::JoinRequest {
        kind: Some(model::v1::join_request::Kind::Predict(predict_request())),
        request_id: "req-1".to_string(),
    };
    assert_frozen(
        "model.v1.JoinRequest",
        message.encode_to_vec(),
        &[
            18, 65, 10, 22, 10, 6, 115, 101, 115, 115, 45, 49, 18, 5, 101, 110, 118, 45, 49, 26, 5,
            114, 101, 113, 45, 49, 18, 9, 10, 3, 1, 2, 3, 10, 2, 4, 5, 26, 4, 101, 112, 45, 49, 26,
            4, 101, 112, 45, 50, 34, 8, 10, 4, 101, 112, 45, 49, 16, 7, 34, 6, 10, 4, 101, 112, 45,
            50, 42, 5, 114, 101, 113, 45, 49,
        ],
    );
}

#[test]
fn model_join_response_bytes_are_frozen() {
    let message = model::v1::JoinResponse {
        kind: Some(model::v1::join_response::Kind::Predict(
            model::v1::PredictResponse {
                context: Some(adapter_context()),
                actions: vec![space_value()],
            },
        )),
        request_id: "req-1".to_string(),
        endpoint_total_ns: Some(1234),
    };
    assert_frozen(
        "model.v1.JoinResponse",
        message.encode_to_vec(),
        &[
            18, 35, 10, 22, 10, 6, 115, 101, 115, 115, 45, 49, 18, 5, 101, 110, 118, 45, 49, 26, 5,
            114, 101, 113, 45, 49, 18, 9, 10, 3, 1, 2, 3, 10, 2, 4, 5, 42, 5, 114, 101, 113, 45,
            49, 48, 210, 9,
        ],
    );
}

#[test]
fn env_step_request_bytes_are_frozen() {
    let message = env::v1::StepRequest {
        action: Some(space_value()),
        timeout_ms: 250,
        env_indices: vec![1],
        episode_ids: vec!["ep-1".to_string(), "ep-2".to_string()],
    };
    assert_frozen(
        "env.v1.StepRequest",
        message.encode_to_vec(),
        &[
            10, 9, 10, 3, 1, 2, 3, 10, 2, 4, 5, 16, 250, 1, 26, 1, 1, 34, 4, 101, 112, 45, 49, 34,
            4, 101, 112, 45, 50,
        ],
    );
}

#[test]
fn env_step_response_bytes_are_frozen() {
    let message = env::v1::StepResponse {
        observation: Some(space_value()),
        rewards: vec![1.5, -0.5],
        terminated_mask: vec![1, 0],
        truncated_mask: vec![0, 1],
        infos: Some(meta_map()),
        completed_episodes: vec![episode_metadata()],
        env_indices: vec![0, 1],
    };
    assert_frozen(
        "env.v1.StepResponse",
        message.encode_to_vec(),
        &[
            10, 9, 10, 3, 1, 2, 3, 10, 2, 4, 5, 18, 16, 0, 0, 0, 0, 0, 0, 248, 63, 0, 0, 0, 0, 0,
            0, 224, 191, 26, 2, 1, 0, 34, 2, 0, 1, 42, 9, 10, 7, 10, 1, 107, 18, 2, 8, 7, 50, 42,
            10, 4, 101, 112, 45, 49, 16, 7, 24, 3, 32, 42, 41, 0, 0, 0, 0, 0, 0, 244, 63, 48, 1,
            56, 1, 64, 232, 7, 72, 208, 15, 82, 9, 10, 7, 10, 1, 107, 18, 2, 8, 7, 58, 2, 0, 1,
        ],
    );
}

#[test]
fn env_join_request_bytes_are_frozen() {
    let message = env::v1::JoinRequest {
        kind: Some(env::v1::join_request::Kind::Step(env::v1::StepRequest {
            action: Some(space_value()),
            timeout_ms: 250,
            env_indices: vec![1],
            episode_ids: vec!["ep-1".to_string(), "ep-2".to_string()],
        })),
        request_id: "req-1".to_string(),
    };
    assert_frozen(
        "env.v1.JoinRequest",
        message.encode_to_vec(),
        &[
            18, 29, 10, 9, 10, 3, 1, 2, 3, 10, 2, 4, 5, 16, 250, 1, 26, 1, 1, 34, 4, 101, 112, 45,
            49, 34, 4, 101, 112, 45, 50, 42, 5, 114, 101, 113, 45, 49,
        ],
    );
}

#[test]
fn core_handshake_request_bytes_are_frozen() {
    let mut frameworks = std::collections::HashMap::new();
    frameworks.insert("numpy".to_string(), "1.26".to_string());
    let mut extra = std::collections::HashMap::new();
    extra.insert("k".to_string(), "v".to_string());
    let mut capabilities = std::collections::HashMap::new();
    capabilities.insert(
        "rlmesh.model.concurrent_predict.v1".to_string(),
        "true".to_string(),
    );
    let message = core::v1::HandshakeRequest {
        protocol_generation: "rlmesh-wire-v1".to_string(),
        peer_info: Some(core::v1::PeerInfo {
            component: "rlmesh-model".to_string(),
            package_version: "0.1.0".to_string(),
            language: "rust".to_string(),
            language_version: "1.88.0".to_string(),
            os: "linux".to_string(),
            os_version: "6.1".to_string(),
            arch: "x86_64".to_string(),
            framework_versions: frameworks,
            extra,
        }),
        capabilities,
        supported_workflow_editions: vec!["2026.06".to_string()],
    };
    assert_frozen(
        "core.v1.HandshakeRequest",
        message.encode_to_vec(),
        &[
            10, 14, 114, 108, 109, 101, 115, 104, 45, 119, 105, 114, 101, 45, 118, 49, 18, 78, 10,
            12, 114, 108, 109, 101, 115, 104, 45, 109, 111, 100, 101, 108, 18, 5, 48, 46, 49, 46,
            48, 26, 4, 114, 117, 115, 116, 34, 6, 49, 46, 56, 56, 46, 48, 42, 5, 108, 105, 110,
            117, 120, 50, 3, 54, 46, 49, 58, 6, 120, 56, 54, 95, 54, 52, 66, 13, 10, 5, 110, 117,
            109, 112, 121, 18, 4, 49, 46, 50, 54, 122, 6, 10, 1, 107, 18, 1, 118, 26, 42, 10, 34,
            114, 108, 109, 101, 115, 104, 46, 109, 111, 100, 101, 108, 46, 99, 111, 110, 99, 117,
            114, 114, 101, 110, 116, 95, 112, 114, 101, 100, 105, 99, 116, 46, 118, 49, 18, 4, 116,
            114, 117, 101, 34, 7, 50, 48, 50, 54, 46, 48, 54,
        ],
    );
}
