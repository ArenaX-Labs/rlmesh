//! The Gym view over a v2-native robot that serves lockstep (§12.6).

#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rlmesh_gym::view::{Entry, Frame, commands};
use rlmesh_gym::{ConnectOptions, GymError, LockstepHandler, Request};
use rlmesh_session::spec::Policy;
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::wire::{DType, LockstepReply, PartSpec};

const UR10E: &str = r#"{
    "arm": {"kind": "joints", "count": 6, "fields": ["pos", "vel"], "hz": 100,
            "command": "position"},
    "wrist/ft": {"kind": "wrench", "frame": "tool0", "hz": 100},
    "task": {"kind": "text"}
}"#;

const POLICY: &str = r#"{
    "hz": 25,
    "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"], "history": 2},
               "ft": {"kind": "wrench", "at": "wrist/ft", "frame": "tool0"},
               "prompt": {"kind": "text", "at": "task"}},
    "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position",
                        "hz": 100, "horizon": 4}}
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
        frame.reward = -err;
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
    let binding = rlmesh_gym::bind("lockstep://127.0.0.1:0/ur10e")
        .await
        .unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    tokio::spawn(binding.serve_robot(
        UR10E,
        rlmesh_gym::ServeOptions::default(),
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
        let view = rlmesh_gym::connect_view(
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
    let view = rlmesh_gym::connect_view(&address, Some((&policy, "reach")), Default::default())
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
