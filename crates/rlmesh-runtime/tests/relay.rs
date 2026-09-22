//! The relay seam: per-leg ceilings, the refusing default, and a converting
//! policy's advisories reaching the report and the hooks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use prost::bytes::Bytes;
use rlmesh_proto::Edition;
use rlmesh_proto::env::v1::{
    EpisodeMetadata, ResetRequest, ResetResponse, StepRequest, StepResponse,
};
use rlmesh_proto::spaces::v1::{BoxSpec, DataType, DictSpec, SpaceSpec, box_spec, space_spec};
use rlmesh_runtime::{
    ActionReceivedEvent, Advisory, DType, EndpointPhases, HookError, Leg, ObservationEmittedEvent,
    PayloadFacts, PeerCeiling, RefusingRelayPolicy, RelayAdvisoryEvent, RelayDecision, RelayPolicy,
    RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep, RuntimeError, RuntimeHooks,
    RuntimeSessionSpec,
};

mod common;

use common::*;

const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

fn ceiling() -> PeerCeiling {
    PeerCeiling::wire_v1(Edition::E2026_06, HashMap::new(), MAX_MESSAGE_SIZE)
}

fn ceiling_without(dtype: DType) -> PeerCeiling {
    let mut ceiling = ceiling();
    ceiling.dtypes.remove(&dtype);
    ceiling
}

fn float64_box() -> SpaceSpec {
    unbounded_box(DataType::Float64)
}

fn unbounded_box(dtype: DataType) -> SpaceSpec {
    SpaceSpec {
        shape: vec![1],
        dtype: dtype as i32,
        spec: Some(space_spec::Spec::Box(BoxSpec {
            bounds: Some(box_spec::Bounds::Unbounded(true)),
        })),
    }
}

/// `one_episode_spec` whose observations are float64 and whose model leg
/// is served behind `model_ceiling`.
fn float64_spec(model_ceiling: PeerCeiling) -> RuntimeSessionSpec {
    let mut spec = one_episode_spec();
    if let Some(env_spec) = spec.env_contract.spec.as_mut() {
        env_spec.observation_space = Some(float64_box());
    }
    spec.env_ceiling = Some(ceiling());
    spec.model_ceiling = Some(model_ceiling);
    spec
}

fn float64_leaf(value: f64) -> Bytes {
    Bytes::copy_from_slice(&value.to_le_bytes())
}

/// `one_episode_spec` whose actions are int64 and whose env leg is served
/// behind `env_ceiling`.
fn int64_action_spec(env_ceiling: PeerCeiling) -> RuntimeSessionSpec {
    let mut spec = one_episode_spec();
    if let Some(env_spec) = spec.env_contract.spec.as_mut() {
        env_spec.action_space = Some(unbounded_box(DataType::Int64));
    }
    spec.env_ceiling = Some(env_ceiling);
    spec.model_ceiling = Some(ceiling());
    spec
}

/// A one-step episode whose observations are single float64 leaves.
#[derive(Clone, Default)]
struct Float64Env {
    episode_id: Arc<Mutex<String>>,
    resets: Arc<Mutex<usize>>,
    actions: Arc<Mutex<Vec<Vec<Bytes>>>>,
}

#[async_trait]
impl RuntimeEnv for Float64Env {
    async fn reset(&mut self, request: ResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        *self.resets.lock().expect("reset counter poisoned") += 1;
        *self.episode_id.lock().expect("episode id poisoned") =
            request.episode_ids.first().cloned().unwrap_or_default();
        Ok(RuntimeEnvReset {
            response: ResetResponse {
                observation: Some(leaves_value(float64_leaf(1.5))),
                infos: None,
            },
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn step(&mut self, request: StepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        self.actions.lock().expect("action recorder poisoned").push(
            request
                .action
                .map(|action| action.leaves)
                .unwrap_or_default(),
        );
        Ok(RuntimeEnvStep {
            response: StepResponse {
                observation: Some(leaves_value(float64_leaf(2.5))),
                rewards: vec![1.0],
                terminated_mask: vec![1],
                truncated_mask: vec![0],
                infos: None,
                completed_episodes: vec![EpisodeMetadata {
                    episode_id: self.episode_id.lock().expect("episode id poisoned").clone(),
                    step_count: 1,
                    cumulative_reward: 1.0,
                    terminated: true,
                    ..Default::default()
                }],
                env_indices: vec![],
            },
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }
}

/// A closed-platform-style policy: downcasts float64 leaves to float32 for a
/// model whose ceiling lacks float64, raising a Caution.
struct DowncastFloat64;

fn downcast_caution() -> Advisory {
    Advisory::caution("float64 observation downcast to float32 for the model")
}

impl RelayPolicy for DowncastFloat64 {
    fn reconcile(&self, leg: Leg, ceiling: &PeerCeiling, payload: &PayloadFacts) -> RelayDecision {
        if leg != Leg::EnvToModel || ceiling.dtypes.contains(&DType::Float64) {
            return RelayDecision::Forward;
        }
        let leaves = payload
            .leaves
            .iter()
            .map(|leaf| {
                let wide = f64::from_le_bytes(leaf[..].try_into().expect("an f64 leaf"));
                Bytes::copy_from_slice(&(wide as f32).to_le_bytes())
            })
            .collect();
        RelayDecision::Convert {
            leaves,
            space: Some(Arc::new(unbounded_box(DataType::Float32))),
            advisory: downcast_caution(),
        }
    }
}

/// Replaces every action bound for an env whose ceiling lacks int64.
struct ReplaceInt64Action;

impl RelayPolicy for ReplaceInt64Action {
    fn reconcile(&self, leg: Leg, ceiling: &PeerCeiling, _payload: &PayloadFacts) -> RelayDecision {
        if leg != Leg::ModelToEnv || ceiling.dtypes.contains(&DType::Int64) {
            return RelayDecision::Forward;
        }
        RelayDecision::Convert {
            leaves: vec![payload([7])],
            space: Some(Arc::new(unbounded_box(DataType::Int32))),
            advisory: Advisory::caution("int64 action narrowed for the env"),
        }
    }
}

/// Records advisories and the space each emitted observation/action carries.
#[derive(Default)]
struct AdvisoryHooks {
    advisories: Mutex<Vec<RelayAdvisoryEvent>>,
    spaces: Mutex<Vec<(&'static str, i32)>>,
}

#[async_trait]
impl RuntimeHooks for AdvisoryHooks {
    async fn relay_advisory(&self, event: RelayAdvisoryEvent) -> Result<(), HookError> {
        self.advisories
            .lock()
            .expect("advisory recorder poisoned")
            .push(event);
        Ok(())
    }

    async fn observation_emitted(&self, event: ObservationEmittedEvent) -> Result<(), HookError> {
        self.spaces
            .lock()
            .expect("space recorder poisoned")
            .push(("observation", event.observation_space.dtype));
        Ok(())
    }

    async fn action_received(&self, event: ActionReceivedEvent) -> Result<(), HookError> {
        self.spaces
            .lock()
            .expect("space recorder poisoned")
            .push(("action", event.action_space.dtype));
        Ok(())
    }
}

#[test]
fn default_policy_refuses_only_what_the_ceiling_does_not_cover() {
    let inside = PayloadFacts::new(
        Arc::new(unbounded_box(DataType::Float32)),
        vec![payload([0; 4])],
    );
    let outside = PayloadFacts {
        space: Arc::new(float64_box()),
        ..inside.clone()
    };
    let narrow = ceiling_without(DType::Float64);

    assert_eq!(
        RefusingRelayPolicy.reconcile(Leg::EnvToModel, &narrow, &inside),
        RelayDecision::Forward
    );
    let RelayDecision::Refuse(reason) =
        RefusingRelayPolicy.reconcile(Leg::EnvToModel, &narrow, &outside)
    else {
        panic!("a dtype outside the ceiling must be refused");
    };
    assert!(reason.contains("float64"), "{reason}");

    let oversized = PayloadFacts {
        byte_len: MAX_MESSAGE_SIZE + 1,
        ..inside.clone()
    };
    assert!(matches!(
        RefusingRelayPolicy.reconcile(Leg::ModelToEnv, &ceiling(), &oversized),
        RelayDecision::Refuse(_)
    ));
}

#[test]
fn a_nested_leaf_outside_the_ceiling_is_named() {
    let mixed = SpaceSpec {
        spec: Some(space_spec::Spec::Dict(DictSpec {
            keys: vec!["pose".to_string(), "depth".to_string()],
            spaces: vec![unbounded_box(DataType::Float32), float64_box()],
        })),
        ..Default::default()
    };
    let facts = PayloadFacts::new(Arc::new(mixed), Vec::new());

    assert_eq!(facts.exceeds(&ceiling()), None);
    let reason = facts
        .exceeds(&ceiling_without(DType::Float64))
        .expect("the depth leaf is float64");
    assert!(reason.contains("float64"), "{reason}");
}

#[tokio::test]
async fn default_policy_forwards_every_payload_today() {
    let env = TestEnv::default();
    let model = TestModel::default();
    let spec = RuntimeSessionSpec {
        env_ceiling: Some(ceiling()),
        model_ceiling: Some(ceiling()),
        ..one_episode_spec()
    };

    let report = RuntimeDriver::new(
        spec,
        env.clone(),
        model.clone(),
        Arc::new(AdvisoryHooks::default()),
    )
    .run()
    .await
    .expect("a payload inside both ceilings relays unchanged");

    assert_eq!(report.total_episodes, 1);
    assert!(report.advisories.is_empty());
    assert_eq!(*model.seen_observations.lock().unwrap(), vec![vec![1]]);
}

#[tokio::test]
async fn default_policy_refuses_a_contract_the_model_cannot_decode() {
    let env = Float64Env::default();
    let hooks = Arc::new(RecordingHooks::default());

    let error = RuntimeDriver::new(
        float64_spec(ceiling_without(DType::Float64)),
        env.clone(),
        TestModel::default(),
        hooks.clone(),
    )
    .run()
    .await
    .expect_err("the model's ceiling lacks float64");

    let message = error.to_string();
    assert!(
        message.contains("cannot relay this contract to the model") && message.contains("float64"),
        "{message}"
    );
    assert_eq!(*env.resets.lock().unwrap(), 0, "refused before any reset");
    assert_eq!(hooks.failed.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_converting_policy_downcasts_and_reports_its_caution_once() {
    let model = TestModel::default();
    let hooks = Arc::new(AdvisoryHooks::default());

    let report = RuntimeDriver::new(
        float64_spec(ceiling_without(DType::Float64)),
        Float64Env::default(),
        model.clone(),
        hooks.clone(),
    )
    .with_relay_policy(Arc::new(DowncastFloat64))
    .run()
    .await
    .expect("the policy converts instead of refusing");

    assert_eq!(
        *model.seen_observations.lock().unwrap(),
        vec![1.5f32.to_le_bytes().to_vec()],
        "the model receives the downcast leaf"
    );
    assert_eq!(report.advisories, vec![downcast_caution()]);
    let streamed = hooks.advisories.lock().unwrap();
    assert_eq!(streamed.len(), 1, "one event per distinct advisory");
    assert_eq!(streamed[0].leg, Leg::EnvToModel);
    assert_eq!(streamed[0].advisory, downcast_caution());
    assert!(
        hooks
            .spaces
            .lock()
            .unwrap()
            .iter()
            .filter(|(kind, _)| *kind == "observation")
            .all(|(_, dtype)| *dtype == DataType::Float32 as i32),
        "emitted observations carry the converted space"
    );
}

#[tokio::test]
async fn default_policy_refuses_an_action_the_env_cannot_decode() {
    let env = Float64Env::default();
    let hooks = Arc::new(RecordingHooks::default());

    let error = RuntimeDriver::new(
        int64_action_spec(ceiling_without(DType::Int64)),
        env.clone(),
        TestModel::default(),
        hooks.clone(),
    )
    .run()
    .await
    .expect_err("the env's ceiling lacks int64");

    let message = error.to_string();
    assert!(
        message.contains("cannot relay this payload to the env") && message.contains("int64"),
        "{message}"
    );
    assert!(
        env.actions.lock().unwrap().is_empty(),
        "no action reached the env"
    );
    assert_eq!(hooks.failed.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_converting_policy_rewrites_the_action_the_env_receives() {
    let env = Float64Env::default();
    let hooks = Arc::new(AdvisoryHooks::default());

    let report = RuntimeDriver::new(
        int64_action_spec(ceiling_without(DType::Int64)),
        env.clone(),
        TestModel::default(),
        hooks.clone(),
    )
    .with_relay_policy(Arc::new(ReplaceInt64Action))
    .run()
    .await
    .expect("the policy converts the action instead of refusing");

    assert_eq!(*env.actions.lock().unwrap(), vec![vec![payload([7])]]);
    assert_eq!(report.advisories.len(), 1);
    assert_eq!(hooks.advisories.lock().unwrap()[0].leg, Leg::ModelToEnv);
    assert!(
        hooks
            .spaces
            .lock()
            .unwrap()
            .contains(&("action", DataType::Int32 as i32)),
        "the emitted action carries the converted space"
    );
}
