//! The Gym view over a v2-native robot that serves lockstep.

#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rlmesh::lockstep::view::{Entry, Frame, commands};
use rlmesh::lockstep::{ConnectOptions, GymError, LockstepHandler, Request};
use rlmesh_session::spec::Policy;
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::wire::{DType, LockstepReply, PartSpec};

const UR10E: &str = r#"{
    "arm": {"kind": "joints", "count": 6, "fields": ["pos", "vel"], "hz": 100,
            "command": "position", "safe": "stop"},
    "wrist/ft": {"kind": "wrench", "frame": "tool0", "hz": 100},
    "task": {"kind": "text"}
}"#;

const POLICY: &str = r#"{
    "hz": 25,
    "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"], "history": 2},
               "ft": {"kind": "wrench", "at": "wrist/ft", "frame": "tool0"},
               "prompt": {"kind": "text", "at": "task"}},
    "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position",
                        "hz": 100, "chunk": 4}}
}"#;

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// A first-order arm: q moves halfway to the commanded target each step.
struct Arm {
    parts: Vec<PartSpec>,
    state: Mutex<([f32; 6], u64, Vec<f32>)>,
}

impl Arm {
    fn frame(&self, q: &[f32; 6], t: u64, target: &[f32]) -> Result<Frame, GymError> {
        let mut frame = Frame::new(self.parts.clone());
        let vel: Vec<f32> = q.iter().zip(target).map(|(q, x)| x - q).collect();
        frame.publish("arm", &[&f32s(q), &f32s(&vel)])?;
        frame.publish(
            "wrist/ft",
            &[&f32s(&[t as f32, 0.0, 0.0]), &f32s(&[0.0; 3])],
        )?;
        frame.publish("task", &[b"reach"])?;
        let err: f32 = q.iter().zip(target).map(|(q, x)| (x - q).abs()).sum();
        frame.reward = -f64::from(err);
        frame.truncated = t >= 10;
        frame.info_json = format!("{{\"t\":{t}}}");
        Ok(frame)
    }
}

#[async_trait]
impl LockstepHandler for Arm {
    async fn handle(&self, request: Request) -> Result<LockstepReply, GymError> {
        let mut state = self.state.lock().unwrap();
        match request {
            Request::Reset(reset) => {
                *state = ([0.0; 6], 0, vec![0.0; 6]);
                Ok(self.frame(&state.0, 0, &state.2)?.reply(reset.epoch, 0))
            }
            Request::Step(step) => {
                for (path, fields) in commands(&self.parts, &step)? {
                    assert_eq!(path, "arm");
                    state.2 = floats(&fields[0]);
                }
                let target = state.2.clone();
                for (q, x) in state.0.iter_mut().zip(&target) {
                    *q += 0.5 * (x - *q);
                }
                state.1 += 1;
                Ok(self
                    .frame(&state.0, state.1, &target)?
                    .reply(step.epoch, step.step))
            }
        }
    }
}

async fn serve() -> (String, tokio_util::sync::CancellationToken) {
    let binding = rlmesh::lockstep::bind(":0/ur10e").await.unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    tokio::spawn(binding.serve_robot(
        UR10E,
        rlmesh::lockstep::ServeOptions::default(),
        cancel.clone(),
        |declared| {
            Ok(Arc::new(Arm {
                parts: declared.parts,
                state: Mutex::new(([0.0; 6], 0, vec![0.0; 6])),
            }) as Arc<dyn LockstepHandler>)
        },
    ));
    (address, cancel)
}

fn target(values: [f32; 6]) -> Value {
    Value::Tensor(Tensor {
        dtype: DType::F32,
        shape: vec![6],
        data: f32s(&values),
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_a_policy_the_view_is_keyed_by_path_and_steps_once_per_step() {
    let (address, cancel) = serve().await;
    for direct in [true, false] {
        let view = rlmesh::lockstep::connect_view(
            &address,
            None,
            ConnectOptions {
                direct,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let obs = view.observation_entries();
        let names: Vec<&str> = obs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["arm", "wrist.ft", "task"]);
        assert!(matches!(&obs[0].1, Entry::Dict(fields) if fields.len() == 2));
        assert_eq!(obs[2].1, Entry::Text);
        assert_eq!(
            view.action_entries(),
            [(
                "arm".to_string(),
                Entry::Array {
                    dtype: DType::F32,
                    shape: vec![6]
                }
            )]
        );
        let first = view.reset(Some(1), "").await.unwrap();
        assert_eq!(first.steps, 0);
        let step = view
            .step(&[("arm".into(), target([1.0; 6]))])
            .await
            .unwrap();
        assert_eq!(step.steps, 1);
        let Value::Dict(arm) = &step.observation[0].1 else {
            panic!("arm is a dict of fields");
        };
        assert_eq!(floats(&arm[0].1.data), [0.5; 6], "halfway to the target");
        assert_eq!(step.reward, -3.0);
        assert_eq!(step.ages_ns[0].1, Some(0));
        assert_eq!(step.edge_modes, [("arm".to_string(), "driving")]);
        assert_eq!(step.info_json, "{\"t\":1}");
        assert_eq!(step.observation[2].1, Value::Text("reach".into()));
        // The view holds the lease; dropping it lets the next view take it.
        drop(view);
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    }
    cancel.cancel();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn through_a_policy_one_view_step_plays_the_whole_chunk() {
    let (address, cancel) = serve().await;
    let policy = Policy::parse(POLICY).unwrap();
    let view =
        rlmesh::lockstep::connect_view(&address, Some((&policy, "reach")), Default::default())
            .await
            .unwrap();
    assert!(view.table().is_some());
    let obs = view.observation_entries();
    assert_eq!(
        obs[0],
        (
            "q".to_string(),
            Entry::Array {
                dtype: DType::F32,
                shape: vec![2, 6]
            }
        ),
        "a 2-sample window"
    );
    assert_eq!(
        view.action_entries(),
        [(
            "arm".to_string(),
            Entry::Array {
                dtype: DType::F32,
                shape: vec![4, 6]
            }
        )]
    );
    let first = view.reset(None, "").await.unwrap();
    let Value::Tensor(q) = &first.observation[0].1 else {
        panic!("q");
    };
    assert_eq!(floats(&q.data), [0.0; 12], "padded with the oldest sample");
    let chunk = Value::Tensor(Tensor {
        dtype: DType::F32,
        shape: vec![4, 6],
        data: f32s(&[1.0; 24]),
    });
    let step = view.step(&[("arm".into(), chunk.clone())]).await.unwrap();
    assert_eq!(step.steps, 4, "a 4-action chunk is 4 lockstep steps");
    let Value::Tensor(q) = &step.observation[0].1 else {
        panic!("q");
    };
    let q = floats(&q.data);
    assert!(
        (q[0] - 0.875).abs() < 1e-6 && (q[6] - 0.9375).abs() < 1e-6,
        "{q:?}"
    );
    assert_eq!(step.observation[2].1, Value::Text("reach".into()));
    // The episode truncates at t = 10: the third chunk stops after 2.
    view.step(&[("arm".into(), chunk.clone())]).await.unwrap();
    let last = view.step(&[("arm".into(), chunk)]).await.unwrap();
    assert_eq!(last.steps, 2);
    assert!(last.truncated);
    cancel.cancel();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_robot_whose_startup_fails_leaves_its_name_free_to_rebind() {
    // Another binding hosts the session, so the failed robot's host
    // closing does not take its participant with it.
    let host = rlmesh::lockstep::bind(":0/host").await.unwrap();
    let address = format!("{}/ur10e", host.address().rsplit_once('/').unwrap().0);
    let binding = rlmesh::lockstep::bind(&address).await.unwrap();
    let failed = binding
        .serve_robot(
            UR10E,
            rlmesh::lockstep::ServeOptions::default(),
            tokio_util::sync::CancellationToken::new(),
            |_| Err(GymError::Config("no handler".into())),
        )
        .await;
    assert!(failed.is_err());
    rlmesh::lockstep::bind(&address).await.unwrap();
}

const HEAD: &str = r#"{
    "head": {"kind": "joints", "names": ["pan", "tilt"], "fields": ["pos"], "hz": 50,
             "command": "position", "safe": "hold",
             "limits": {"pos": [[-0.6, 0.6], [-0.4, 0.4]]}},
    "temp": {"kind": "value", "hz": 50, "unit": "degC", "range": [-40, 125]}
}"#;

/// A pan/tilt head that records every command its step receives.
struct Head {
    parts: Vec<PartSpec>,
    seen: Arc<Mutex<Vec<Vec<f32>>>>,
}

#[async_trait]
impl LockstepHandler for Head {
    async fn handle(&self, request: Request) -> Result<LockstepReply, GymError> {
        let mut frame = Frame::new(self.parts.clone());
        let (epoch, step) = match &request {
            Request::Reset(reset) => (reset.epoch, 0),
            Request::Step(step) => {
                for (_, fields) in commands(&self.parts, step)? {
                    self.seen.lock().unwrap().push(floats(&fields[0]));
                }
                (step.epoch, step.step)
            }
        };
        let last = self.seen.lock().unwrap().last().cloned();
        frame.publish("head", &[&f32s(&last.unwrap_or(vec![0.0; 2]))])?;
        frame.publish("temp", &[&f32s(&[21.0])])?;
        Ok(frame.reply(epoch, step))
    }
}

fn head(values: &[f32]) -> Value {
    Value::Tensor(Tensor {
        dtype: DType::F32,
        shape: vec![2],
        data: f32s(values),
    })
}

fn box_bounds(space: &rlmesh_spaces::SpaceSpec, key: &str) -> rlmesh_spaces::BoxBounds {
    use rlmesh_spaces::SpaceKind;
    let Some(SpaceKind::Dict(dict)) = &space.spec else {
        panic!("a dict space");
    };
    let i = dict.keys.iter().position(|k| k == key).unwrap();
    let Some(SpaceKind::Box(spec)) = &dict.spaces[i].spec else {
        panic!("{key} is a box");
    };
    spec.bounds.clone().unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lockstep_robot_holds_its_commands_to_its_parts_limits() {
    use rlmesh_spaces::{BoxBounds, ElementwiseBounds, UniformBounds};
    let binding = rlmesh::lockstep::bind(":0/head").await.unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let robot_seen = seen.clone();
    tokio::spawn(binding.serve_robot(
        HEAD,
        rlmesh::lockstep::ServeOptions::default(),
        cancel.clone(),
        move |declared| {
            Ok(Arc::new(Head {
                parts: declared.parts,
                seen: robot_seen,
            }) as Arc<dyn LockstepHandler>)
        },
    ));
    for direct in [true, false] {
        let view = rlmesh::lockstep::connect_view(
            &address,
            None,
            ConnectOptions {
                direct,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        // The view's spaces carry the declared limits and range.
        let (observations, actions) = view.spaces();
        assert_eq!(
            box_bounds(&actions, "head"),
            BoxBounds::Elementwise(ElementwiseBounds {
                low: vec![-0.6, -0.4],
                high: vec![0.6, 0.4],
            })
        );
        assert_eq!(
            box_bounds(&observations, "temp"),
            BoxBounds::Uniform(UniformBounds {
                low: -40.0,
                high: 125.0,
            })
        );
        assert_eq!(
            box_bounds(&observations, "head"),
            BoxBounds::Uniform(UniformBounds {
                low: f64::NEG_INFINITY,
                high: f64::INFINITY,
            }),
            "a part with no range keeps the dtype's span"
        );

        // An out-of-limit action reaches the robot clamped, and `info`
        // counts it.
        view.reset(None, "").await.unwrap();
        seen.lock().unwrap().clear();
        let step = view
            .step(&[("head".into(), head(&[1.26, -1.70]))])
            .await
            .unwrap();
        assert_eq!(seen.lock().unwrap().as_slice(), [vec![0.6, -0.4]]);
        assert_eq!(step.clamped, [("head".to_string(), 1)]);
        let info = rlmesh::lockstep::view::info(&step).unwrap();
        assert_eq!(
            info["clamped"],
            rlmesh_spaces::MetaValue::Map(
                [("head".into(), rlmesh_spaces::MetaValue::Int(1))].into()
            )
        );
        let within = view
            .step(&[("head".into(), head(&[0.1, -0.2]))])
            .await
            .unwrap();
        assert_eq!(seen.lock().unwrap()[1], [0.1, -0.2]);
        assert_eq!(within.clamped, [("head".to_string(), 0)]);

        // An invalid action is refused before the robot's step sees it.
        let refused = view
            .step(&[("head".into(), head(&[f32::NAN, 0.0]))])
            .await
            .unwrap_err();
        assert!(
            refused.to_string().contains("head: a non-finite value"),
            "{refused}"
        );
        assert_eq!(seen.lock().unwrap().len(), 2);
        drop(view);
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    }
    cancel.cancel();
}
