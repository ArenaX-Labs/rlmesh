//! Runtime hook transforms: failing, action, per-lane and instruction
//! transforms.

use super::*;

#[tokio::test]
async fn a_failing_transform_fails_only_its_own_route() {
    let hooks = Arc::new(RecordingHooks {
        fail_transform_on: Some("perturbed".to_string()),
        ..Default::default()
    });
    let route = |env_id: &str| {
        let mut spec = one_episode_spec();
        spec.env_id = env_id.to_string();
        spec.max_episodes = Some(3);
        let env = TestEnv::default();
        let model = TestModel::default();
        let driver = new_driver(spec, env.clone(), model.clone(), hooks.clone());
        (driver, env, model)
    };
    let (healthy, _, _) = route("healthy");
    let (perturbed, perturbed_env, perturbed_model) = route("perturbed");

    let (healthy, perturbed) = tokio::join!(healthy.run(), perturbed.run());

    assert_eq!(
        healthy.expect("the sibling route completes").total_episodes,
        3
    );
    let error = perturbed.expect_err("the failing transform fails its route");
    assert!(matches!(error, RuntimeError::Hook(_)), "{error:?}");
    assert!(
        error
            .to_string()
            .contains("transform_observation: perturbation engine refused"),
        "the error names the hook and its cause: {error}"
    );
    assert!(perturbed_env.closed.load(Ordering::SeqCst));
    assert!(perturbed_model.closed.load(Ordering::SeqCst));
    assert_eq!(hooks.failed.load(Ordering::SeqCst), 1);
    assert_eq!(hooks.ended.load(Ordering::SeqCst), 1);
}

/// An env whose episodes only the runtime's step cap ends, recording the
/// action byte it was stepped with.
#[derive(Clone, Default)]
struct ActionLog(Arc<Mutex<Vec<u8>>>);

#[async_trait]
impl RuntimeEnv for ActionLog {
    async fn reset(&mut self, _request: EnvResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        Ok(RuntimeEnvReset {
            observation: vec![payload([0])],
            infos: None,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn step(&mut self, request: EnvStepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        self.0
            .lock()
            .expect("recorder lock poisoned")
            .push(request.action[0][0]);
        Ok(RuntimeEnvStep {
            observation: vec![payload([0])],
            rewards: vec![0.0],
            terminated: vec![false],
            truncated: vec![false],
            infos: None,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }
}

/// The latency ops a perturbation plugin runs on actions, per episode: each
/// keeps the model's actions and the ones it sent.
#[derive(Clone, Copy)]
enum ActionOp {
    /// Send the model's action from `n` steps ago (the first, until there is one).
    Delay(usize),
    /// At these 1-based steps, hold the previously sent action.
    Drop(&'static [i64]),
    /// From step `at`, hold that step's action for `len` steps.
    Freeze { at: i64, len: i64 },
}

#[derive(Default)]
struct ActionOpHooks {
    op: Option<ActionOp>,
    clean: Mutex<std::collections::HashMap<String, Vec<Vec<Bytes>>>>,
    sent: Mutex<std::collections::HashMap<String, Vec<Bytes>>>,
    /// `(raw, sent)` first action byte per `action_received`.
    seen: Mutex<Vec<(Option<u8>, u8)>>,
}

#[async_trait]
impl RuntimeHooks for ActionOpHooks {
    async fn transform_action(
        &self,
        event: &ActionReceivedEvent,
    ) -> Result<Option<Vec<Bytes>>, HookError> {
        let Some(op) = self.op else {
            return Ok(None);
        };
        let action = event.action.clone().expect("the driver sends an action");
        let mut clean = self.clean.lock().expect("recorder lock poisoned");
        let history = clean.entry(event.episode_id.clone()).or_default();
        history.push(action.clone());
        let mut sent = self.sent.lock().expect("recorder lock poisoned");
        let last = sent.get(&event.episode_id).cloned();
        let out = match op {
            ActionOp::Delay(n) => history[history.len().saturating_sub(n + 1)].clone(),
            ActionOp::Drop(steps) if steps.contains(&event.step) => last.unwrap_or(action),
            ActionOp::Freeze { at, len } if (at + 1..at + len).contains(&event.step) => {
                last.unwrap_or(action)
            }
            _ => action,
        };
        sent.insert(event.episode_id.clone(), out.clone());
        Ok(Some(out))
    }

    async fn action_received(&self, event: ActionReceivedEvent) -> Result<(), HookError> {
        let first = |leaves: &Vec<Bytes>| leaves[0][0];
        self.seen.lock().expect("recorder lock poisoned").push((
            event.raw_action.as_ref().map(first),
            first(event.action.as_ref().expect("an action")),
        ));
        Ok(())
    }
}

#[tokio::test]
async fn an_action_transform_expresses_delay_drop_and_freeze() {
    // One predict and seven replayed frames per episode: the model's actions
    // are 0, 100..=106 on steps 1..=8.
    let cases: [(&str, Option<ActionOp>, [u8; 8]); 4] = [
        ("none", None, [0, 100, 101, 102, 103, 104, 105, 106]),
        (
            "delay",
            Some(ActionOp::Delay(2)),
            [0, 0, 0, 100, 101, 102, 103, 104],
        ),
        (
            "drop",
            Some(ActionOp::Drop(&[3, 5])),
            [0, 100, 100, 102, 102, 104, 105, 106],
        ),
        (
            "freeze",
            Some(ActionOp::Freeze { at: 2, len: 3 }),
            [0, 100, 100, 100, 103, 104, 105, 106],
        ),
    ];
    for (name, op, sent) in cases {
        let mut spec = one_episode_spec();
        spec.max_episode_steps = Some(8);
        let env = ActionLog::default();
        let model = TestModel {
            replay_frames: 7,
            ..Default::default()
        };
        let hooks = Arc::new(ActionOpHooks {
            op,
            ..Default::default()
        });
        new_driver(spec, env.clone(), model, hooks.clone())
            .run()
            .await
            .unwrap();

        assert_eq!(
            *env.0.lock().expect("recorder lock poisoned"),
            sent,
            "{name}: the env's actions"
        );
        let seen = hooks.seen.lock().expect("recorder lock poisoned").clone();
        let model_actions = [0, 100, 101, 102, 103, 104, 105, 106];
        for (step, (raw, action)) in seen.into_iter().enumerate() {
            assert_eq!(action, sent[step], "{name}: observers see the sent action");
            match op {
                // Unchanged payloads carry no raw copy.
                None => assert_eq!(raw, None, "{name}"),
                Some(_) => assert_eq!(raw, Some(model_actions[step]), "{name}: raw action"),
            }
        }
    }
}

/// A perturbation hook using the runtime events: bail before any
/// decode on a route with no engine, decode the group's lanes from the
/// event's own space, transform each lane under its (episode, lane, step)
/// identity, and encode the lanes back.
#[derive(Default)]
struct PerturbationLikeHooks {
    enabled: std::collections::HashSet<String>,
    /// `(episode id, lane, step)` per transformed lane.
    lanes: Mutex<Vec<(String, i32, i64)>>,
}

#[async_trait]
impl RuntimeHooks for PerturbationLikeHooks {
    async fn transform_observation(
        &self,
        event: &rlmesh::binding::runtime::ObservationEmittedEvent,
    ) -> Result<Option<Vec<Bytes>>, HookError> {
        if !self.enabled.contains(&event.route.env_id) {
            return Ok(None);
        }
        let fail = |err: &dyn std::fmt::Display| HookError::Message(err.to_string());
        let space = rlmesh::binding::data::space_spec_from_proto(SpaceSpec::clone(
            &event.observation_space,
        ))
        .map_err(|err| fail(&err))?;
        let rows = event.episode_ids.len();
        let leaves = event.observation.as_deref().unwrap_or_default();
        let values = rlmesh_spaces::leaves::decode_leaf_slab(leaves, &space, rows)
            .map_err(|err| fail(&err))?;
        let mut lanes = self.lanes.lock().expect("recorder lock poisoned");
        let perturbed: Vec<SpaceValue> = values
            .into_iter()
            .enumerate()
            .map(|(row, _)| {
                // A lane group's event names its lane; a vector group's rows are
                // its lanes in order.
                let lane = if rows == 1 {
                    event.env_index
                } else {
                    row as i32
                };
                lanes.push((event.episode_ids[row].clone(), lane, event.step));
                action(10 + lane as u8)
            })
            .collect();
        rlmesh_spaces::leaves::encode_leaf_slab(&perturbed, &space)
            .map(Some)
            .map_err(|err| fail(&err))
    }
}

#[tokio::test]
async fn transforms_apply_per_lane_inside_a_fused_predict() {
    // Four lane groups whose resets land while the first predict is still
    // in flight: they reach the model as one fused predict.
    let lanes = 4;
    let mut spec = lane_spec(lanes, lanes as u64, Vec::new());
    let space = rlmesh::binding::data::space_spec_to_proto(
        contract(&spec).observation_space.as_ref().unwrap(),
    );
    let env_spec = spec.env_contract.spec.as_mut().unwrap();
    env_spec.observation_space = Some(space.clone());
    env_spec.action_space = Some(space);
    let env = LaneTestEnv::new(vec![(1, Duration::ZERO); lanes]);
    let model = TestModel {
        predict_delay: Some(Duration::from_millis(20)),
        ..Default::default()
    };
    let hooks = Arc::new(PerturbationLikeHooks {
        enabled: [spec.env_id.clone()].into(),
        ..Default::default()
    });

    new_driver(spec, env, model.clone(), hooks.clone())
        .run()
        .await
        .unwrap();

    let batches = model
        .batches
        .lock()
        .expect("recorder lock poisoned")
        .clone();
    let fused = batches
        .iter()
        .find(|batch| batch.len() > 1)
        .expect("a fused predict");
    let mut marks: Vec<u8> = fused.iter().map(|leaves| leaves[0][0]).collect();
    marks.sort_unstable();
    marks.dedup();
    assert_eq!(
        marks.len(),
        fused.len(),
        "each lane carries its own transform"
    );
    let all: Vec<u8> = batches
        .iter()
        .flatten()
        .map(|leaves| leaves[0][0])
        .collect();
    let mut all_sorted = all.clone();
    all_sorted.sort_unstable();
    assert_eq!(all_sorted, [10, 11, 12, 13], "every lane, transformed once");
    let identities = hooks.lanes.lock().expect("recorder lock poisoned").clone();
    assert_eq!(identities.len(), lanes);
    assert!(
        identities
            .iter()
            .all(|(id, _, step)| !id.is_empty() && *step == 0)
    );
}

/// A one-lane env whose observation is `{state, task}` with the env's own
/// instruction in `task`; its episodes end after two steps.
#[derive(Clone)]
struct InstructionEnv {
    space: rlmesh_spaces::SpaceSpec,
    steps: Arc<AtomicUsize>,
}

impl InstructionEnv {
    fn observation(&self, step: u8) -> Vec<Bytes> {
        let value = SpaceValue::Dict(
            [
                ("state".to_string(), action(step)),
                (
                    "task".to_string(),
                    SpaceValue::Text("pick up the bowl".into()),
                ),
            ]
            .into(),
        );
        rlmesh_spaces::leaves::encode_leaf_slab(&[value], &self.space)
            .expect("an encodable observation")
    }
}

#[async_trait]
impl RuntimeEnv for InstructionEnv {
    async fn reset(&mut self, _request: EnvResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        self.steps.store(0, Ordering::SeqCst);
        Ok(RuntimeEnvReset {
            observation: self.observation(0),
            infos: None,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn step(&mut self, _request: EnvStepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        let step = self.steps.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(RuntimeEnvStep {
            observation: self.observation(step as u8),
            rewards: vec![0.0],
            terminated: vec![step >= 3],
            truncated: vec![false],
            infos: None,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }
}

#[derive(Default)]
struct InstructionHooks {
    /// `(episode index, the env's instruction)` per call.
    asked: Mutex<Vec<(i64, String)>>,
    /// The raw instruction leaf of every observation, when one was rewritten.
    raw: Mutex<Vec<Option<Bytes>>>,
}

#[async_trait]
impl RuntimeHooks for InstructionHooks {
    async fn transform_instruction(
        &self,
        episode: &rlmesh::binding::runtime::EpisodeStartedEvent,
        instruction: &str,
    ) -> Result<Option<String>, HookError> {
        self.asked
            .lock()
            .expect("recorder lock poisoned")
            .push((episode.episode_index, instruction.to_string()));
        Ok(Some(format!(
            "{instruction}, episode {}",
            episode.episode_index
        )))
    }

    async fn observation_emitted(
        &self,
        event: rlmesh::binding::runtime::ObservationEmittedEvent,
    ) -> Result<(), HookError> {
        self.raw
            .lock()
            .expect("recorder lock poisoned")
            .push(event.raw_observation.map(|raw| raw[1].clone()));
        Ok(())
    }
}

#[tokio::test]
async fn the_instruction_transform_rewrites_each_episode_from_its_start() {
    use rlmesh::adapters::{Action, Actuator, EnvTags, INSTRUCTION, ObsNode, TextTag};
    use rlmesh_spaces::{DictSpec, SpaceKind};
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(2);
    let mut native = contract(&spec);
    let state = native.observation_space.clone().unwrap();
    let space = rlmesh_spaces::SpaceSpec {
        spec: Some(SpaceKind::Dict(DictSpec {
            keys: vec!["state".into(), "task".into()],
            spaces: vec![state, text_space()],
        })),
        ..Default::default()
    };
    native.observation_space = Some(space.clone());
    let tags = EnvTags::new(
        ObsNode::dict([("task", ObsNode::from(TextTag::new(INSTRUCTION)))]),
        Action::new([Actuator {
            dim: 1,
            ..Default::default()
        }]),
    )
    .unwrap();
    native.metadata =
        Some(rlmesh::binding::metadata_to_wire(&tags.to_metadata().unwrap()).unwrap());
    spec.env_contract = rlmesh::binding::data::env_contract_to_proto(&native);
    let env = InstructionEnv {
        space: space.clone(),
        steps: Arc::default(),
    };
    let model = TestModel::default();
    let hooks = Arc::new(InstructionHooks::default());

    rlmesh::binding::runtime::RuntimeDriver::new(spec, env, model.clone(), native, hooks.clone())
        .run()
        .await
        .unwrap();

    assert_eq!(
        *hooks.asked.lock().expect("recorder lock poisoned"),
        [
            (1, "pick up the bowl".to_string()),
            (2, "pick up the bowl".to_string())
        ],
        "asked once per episode, with the env's instruction"
    );
    let tasks: Vec<String> = model
        .batches
        .lock()
        .unwrap()
        .iter()
        .flatten()
        .map(|leaves| {
            match &rlmesh_spaces::leaves::decode_leaf_slab(leaves, &space, 1).unwrap()[0] {
                SpaceValue::Dict(map) => match &map["task"] {
                    SpaceValue::Text(text) => text.clone(),
                    other => panic!("task is text: {other:?}"),
                },
                other => panic!("a dict observation: {other:?}"),
            }
        })
        .collect();
    let episode = |n| format!("pick up the bowl, episode {n}");
    assert_eq!(
        tasks,
        [
            episode(1),
            episode(1),
            episode(1),
            episode(2),
            episode(2),
            episode(2)
        ],
        "every observation of an episode carries its instruction"
    );
    let raw = hooks.raw.lock().expect("recorder lock poisoned").clone();
    assert_eq!(raw.len(), tasks.len());
    for leaf in raw {
        let leaf = leaf.expect("the raw observation rides beside the rewrite");
        let values = rlmesh_spaces::leaves::decode_leaf_slab(&[leaf], &text_space(), 1).unwrap();
        assert_eq!(values, [SpaceValue::Text("pick up the bowl".into())]);
    }
}

fn text_space() -> rlmesh_spaces::SpaceSpec {
    rlmesh_spaces::SpaceSpec {
        spec: Some(rlmesh_spaces::SpaceKind::Text(rlmesh_spaces::TextSpec {
            min_length: 0,
            max_length: 64,
            charset: String::new(),
        })),
        ..Default::default()
    }
}
