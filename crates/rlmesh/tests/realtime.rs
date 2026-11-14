//! One policy, one `predict`, on either clock: `rlmesh::realtime` follows
//! the robot's declared clock.

#![allow(clippy::unwrap_used)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use rlmesh::lockstep::view::{Frame, commands};
use rlmesh::lockstep::{GymError, LockstepHandler, Request};
use rlmesh::realtime::{
    self, BoundPolicy, LockstepRunError, PolicyRunError, RunOutcome, RunSettings,
};
use rlmesh_session::SessionAddress;
use rlmesh_session::cache::Observation;
use rlmesh_session::controller::{Predictor, Resume, RunMode, Step};
use rlmesh_session::sdk::BindPolicyError;
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::spec::{Policy, RunOptions};
use rlmesh_session::wire::RobotClock as Clock;
use rlmesh_session::wire::{DType, LockstepReply, PartSpec, PriorityClass};

const WAIT: Duration = Duration::from_secs(10);

const ARM: &str = r#"{
    "arm": {"kind": "joints", "count": 2, "fields": ["pos"], "hz": 100,
            "command": "position", "safe": "stop", "max_action_age_ms": 1000}
}"#;

/// Reads the joints, commands a 4-action chunk at 100 Hz.
const REACH: &str = r#"{
    "hz": 25,
    "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]}},
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

/// What the lockstep robot saw.
#[derive(Default)]
struct Seen {
    seeds: Vec<Option<u64>>,
    rewards: Vec<f64>,
}

/// A first-order arm: q moves halfway to its target each step; the reward
/// is minus the distance to 1, and an episode is cut at 10 steps.
struct Sim {
    parts: Vec<PartSpec>,
    state: Mutex<([f32; 2], u64)>,
    seen: Arc<Mutex<Seen>>,
}

impl Sim {
    fn frame(&self, q: [f32; 2], t: u64) -> Result<Frame, GymError> {
        let mut frame = Frame::new(self.parts.clone());
        frame.publish("arm", &[&f32s(&q)])?;
        frame.reward = -f64::from((1.0 - q[0]).abs() + (1.0 - q[1]).abs());
        frame.truncated = t >= 10;
        Ok(frame)
    }
}

#[async_trait]
impl LockstepHandler for Sim {
    async fn handle(&self, request: Request) -> Result<LockstepReply, GymError> {
        let mut state = self.state.lock().unwrap();
        match request {
            Request::Reset(reset) => {
                let seed = reset.seeds.first().copied();
                self.seen.lock().unwrap().seeds.push(seed);
                // The seed sets where the arm starts.
                let start = seed.map_or(0.0, |s| (s % 7) as f32 / 10.0);
                *state = ([start; 2], 0);
                Ok(self.frame(state.0, 0)?.reply(reset.epoch, 0))
            }
            Request::Step(step) => {
                for (_, fields) in commands(&self.parts, &step)? {
                    let target = floats(&fields[0]);
                    for (q, x) in state.0.iter_mut().zip(target) {
                        *q += 0.5 * (x - *q);
                    }
                }
                state.1 += 1;
                let frame = self.frame(state.0, state.1)?;
                self.seen.lock().unwrap().rewards.push(frame.reward);
                Ok(frame.reply(step.epoch, step.step))
            }
        }
    }
}

/// Serves the lockstep robot `sim`; returns its session and what it saw.
async fn serve_sim() -> (
    SessionAddress,
    Arc<Mutex<Seen>>,
    tokio_util::sync::CancellationToken,
) {
    let binding = rlmesh::lockstep::bind(":0/sim").await.unwrap();
    let session = rlmesh_session::address::Address::participant(&binding.address())
        .unwrap()
        .joining()
        .unwrap();
    let cancel = tokio_util::sync::CancellationToken::new();
    let seen = Arc::new(Mutex::new(Seen::default()));
    let robot_seen = seen.clone();
    tokio::spawn(binding.serve_robot(
        ARM,
        rlmesh::lockstep::ServeOptions::default(),
        cancel.clone(),
        move |declared| {
            Ok(Arc::new(Sim {
                parts: declared.parts,
                state: Mutex::new(([0.0; 2], 0)),
                seen: robot_seen,
            }) as Arc<dyn LockstepHandler>)
        },
    ));
    (session, seen, cancel)
}

/// The one model: reach for 1 on both joints, whatever the clock.
#[derive(Default)]
struct Reach {
    predicts: u64,
    ages_ns: Vec<Option<i64>>,
    played: Vec<usize>,
}

impl Predictor for Reach {
    type Error = String;

    fn predict(
        &mut self,
        observation: Observation,
        _: i64,
    ) -> Result<Vec<(String, Value)>, String> {
        self.predicts += 1;
        self.ages_ns.push(observation.meta.inputs[0].age_ns);
        let chunk = Tensor {
            dtype: DType::F32,
            shape: vec![4, 2],
            data: f32s(&[1.0; 8]),
        };
        Ok(vec![("arm".into(), Value::Tensor(chunk))])
    }

    fn step(&mut self, step: &Step) -> Result<(), String> {
        self.played.push(step.actions[0].packed.len() / 8);
        Ok(())
    }
}

fn reach() -> Policy {
    Policy::parse(REACH).unwrap()
}

async fn bind(session: &SessionAddress, robot: &str) -> BoundPolicy {
    let options = RunOptions::named("reach").robot(robot).unwrap();
    realtime::bind(
        session,
        "reach",
        &reach(),
        &options,
        PriorityClass::Model,
        WAIT,
    )
    .await
    .unwrap()
}

/// Runs `bound` on a plain thread, as a run must.
fn run(
    bound: &BoundPolicy,
    predictor: &mut Reach,
    settings: RunSettings,
) -> Result<RunOutcome, PolicyRunError<String>> {
    std::thread::scope(|s| s.spawn(|| bound.run(predictor, settings)).join().unwrap())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lockstep_robot_runs_episodes_of_whole_chunks_with_their_rewards_summed() {
    let (session, seen, cancel) = serve_sim().await;
    let bound = bind(&session, "sim").await;
    assert_eq!(bound.clock(), Clock::Lockstep);
    let BoundPolicy::Lockstep(policy) = &bound else {
        panic!("a lockstep robot");
    };
    // One step is one action period of the policy (100 Hz).
    assert_eq!(policy.tick_ns(), 10_000_000);
    let settings = RunSettings {
        episodes: Some(2),
        seed: Some(7),
        ..RunSettings::default()
    };
    let mut model = Reach::default();
    let Ok(RunOutcome::Lockstep(first)) = run(&bound, &mut model, settings.clone()) else {
        panic!("a lockstep run");
    };
    // 10 steps an episode: chunks of 4, 4 and 2, the last cut at the end.
    assert_eq!(first.episodes.len(), 2);
    assert_eq!(first.starts, 6);
    assert_eq!(model.played, [4, 4, 2, 4, 4, 2]);
    for episode in &first.episodes {
        assert_eq!(episode.steps, 10);
        assert!(episode.truncated && !episode.terminated);
    }
    let rewards = seen.lock().unwrap().rewards.clone();
    let sums: Vec<f64> = rewards.chunks(10).map(|c| c.iter().sum()).collect();
    assert_eq!(
        first.episodes.iter().map(|e| e.reward).collect::<Vec<_>>(),
        sums
    );
    // Each episode is reset with the seed a model run derives.
    let seeds = seen.lock().unwrap().seeds.clone();
    let derived: Vec<Option<u64>> = (0..2)
        .map(|i| Some(policy.episode_seed(7, i) as u64))
        .collect();
    assert_eq!(seeds, derived);
    assert_ne!(seeds[0], seeds[1]);
    // Ages are session time, and the arm publishes every step.
    assert!(model.ages_ns.iter().all(|age| *age == Some(0)));

    // The same seed gives the same episodes.
    let Ok(RunOutcome::Lockstep(again)) = run(&bound, &mut Reach::default(), settings) else {
        panic!("a lockstep run");
    };
    assert_eq!(again, first);
    bound.close().await;
    cancel.cancel();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lockstep_run_refuses_what_only_the_wall_clock_means() {
    let (session, _, cancel) = serve_sim().await;
    let bound = bind(&session, "sim").await;
    for (settings, says) in [
        (
            RunSettings {
                duration: Some(Duration::from_secs(1)),
                ..RunSettings::default()
            },
            "duration=",
        ),
        (
            RunSettings {
                mode: RunMode::Shadow,
                ..RunSettings::default()
            },
            "mode=shadow",
        ),
        (
            RunSettings {
                resume: Some(Resume::Auto),
                ..RunSettings::default()
            },
            "resume=",
        ),
    ] {
        let Err(PolicyRunError::Settings(why)) = run(&bound, &mut Reach::default(), settings)
        else {
            panic!("{says} is refused");
        };
        assert!(why.starts_with(says) && why.contains("lockstep"), "{why}");
    }
    // A run straight on the lockstep policy checks them too.
    let BoundPolicy::Lockstep(policy) = &bound else {
        panic!("a lockstep robot");
    };
    let refused = std::thread::scope(|s| {
        s.spawn(|| {
            let settings = RunSettings {
                episodes: Some(0),
                ..RunSettings::default()
            };
            policy.run(&mut Reach::default(), &settings)
        })
        .join()
        .unwrap()
    });
    assert!(matches!(refused, Err(LockstepRunError::Settings(_))));
    bound.close().await;
    cancel.cancel();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_same_predictor_drives_a_wall_clock_robot() {
    let (session, _, cancel) = serve_sim().await;
    // A wall-clock robot in the same session, publishing and playing what
    // is due at 100 Hz.
    let robot = Arc::new(
        rlmesh_session::sdk::connect_robot(&session, "ur10e", ARM, WAIT)
            .await
            .unwrap(),
    );
    // The lockstep robot armed the session: one declaring after is held
    // until re-armed by name.
    let ur10e = rlmesh_session::wire::ParticipantId::new("ur10e").unwrap();
    robot.participant().rearm(&ur10e).await.unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let played = Arc::new(AtomicU64::new(0));
    let control = {
        let (robot, stop, played) = (robot.clone(), stop.clone(), played.clone());
        std::thread::spawn(move || {
            let clock = rlmesh_session::SystemMonotonicClock;
            while !stop.load(Ordering::Relaxed) {
                let now = rlmesh_session::Clock::now_ns(&clock);
                rlmesh_session::sdk::publish(robot.published(), "arm", now, vec![f32s(&[0.0; 2])])
                    .unwrap();
                if robot
                    .due("arm")
                    .unwrap()
                    .is_some_and(|d| d.action.is_some())
                {
                    played.fetch_add(1, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        })
    };
    let bound = bind(&session, "ur10e").await;
    assert_eq!(bound.clock(), Clock::Wall);
    let mut model = Reach::default();
    // Episodes and seeds are lockstep's.
    let refused = run(
        &bound,
        &mut model,
        RunSettings {
            episodes: Some(3),
            ..RunSettings::default()
        },
    );
    assert!(
        matches!(&refused, Err(PolicyRunError::Settings(why)) if why.starts_with("episodes=")),
        "{refused:?}"
    );
    let ran = run(
        &bound,
        &mut model,
        RunSettings {
            duration: Some(Duration::from_millis(1500)),
            ..RunSettings::default()
        },
    );
    assert!(matches!(ran, Ok(RunOutcome::Wall(_))), "{ran:?}");
    assert!(model.predicts > 0);
    assert!(
        played.load(Ordering::Relaxed) > 0,
        "the edge played the chunks"
    );
    bound.close().await;
    stop.store(true, Ordering::Relaxed);
    control.join().unwrap();
    rlmesh_session::sdk::close_robot(robot).await.unwrap();
    cancel.cancel();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_policy_binding_a_lockstep_robot_among_others_is_refused() {
    let (session, _, cancel) = serve_sim().await;
    let board = rlmesh_session::sdk::connect_robot(&session, "board", ARM, WAIT)
        .await
        .unwrap();
    let both = Policy::parse(
        r#"{"hz": 25,
            "inputs": {"sim": {"kind": "joints", "at": "sim:arm", "fields": ["pos"]},
                       "board": {"kind": "joints", "at": "board:arm", "fields": ["pos"]}}}"#,
    )
    .unwrap();
    let refused = realtime::bind(
        &session,
        "both",
        &both,
        &RunOptions::named("both"),
        PriorityClass::Model,
        WAIT,
    )
    .await;
    let Err(realtime::BindError::Policy(BindPolicyError::Controller(err))) = refused else {
        panic!("refused: {:?}", refused.err());
    };
    let why = err.to_string();
    assert!(
        why.contains("board, sim") && why.contains("sim runs in lockstep"),
        "{why}"
    );
    drop(board);
    cancel.cancel();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_policy_bound_before_its_lockstep_robot_serves_runs_it_in_lockstep() {
    // Attached and declared lockstep, not serving: nothing answers `l/sim`.
    let binding = rlmesh::lockstep::bind(":0/sim").await.unwrap();
    let session = rlmesh_session::address::Address::participant(&binding.address())
        .unwrap()
        .joining()
        .unwrap();
    let binding_session = session.clone();
    let bound = tokio::spawn(async move { bind(&binding_session, "sim").await });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let cancel = tokio_util::sync::CancellationToken::new();
    let seen = Arc::new(Mutex::new(Seen::default()));
    tokio::spawn(binding.serve_robot(
        ARM,
        rlmesh::lockstep::ServeOptions::default(),
        cancel.clone(),
        move |declared| {
            Ok(Arc::new(Sim {
                parts: declared.parts,
                state: Mutex::new(([0.0; 2], 0)),
                seen,
            }) as Arc<dyn LockstepHandler>)
        },
    ));
    let bound = bound.await.unwrap();
    assert_eq!(bound.clock(), Clock::Lockstep);
    let ran = run(&bound, &mut Reach::default(), RunSettings::default());
    let Ok(RunOutcome::Lockstep(stats)) = ran else {
        panic!("a lockstep run: {ran:?}");
    };
    assert_eq!(stats.episodes.len(), 1, "one episode by default");
    bound.close().await;
    cancel.cancel();
}

/// A Gym env of two steps an episode that records the `trial_index` each
/// reset carried; it names the option among its reset options when
/// `declares`.
struct Trials {
    declares: bool,
    t: AtomicU64,
    seen: Arc<Mutex<Vec<Option<rlmesh_spaces::MetaValue>>>>,
}

impl Trials {
    fn zeros() -> rlmesh_spaces::SpaceValue {
        let data = f32s(&[0.0; 2]);
        rlmesh_spaces::SpaceValue::Box(
            rlmesh_spaces::Tensor::from_vec(data, vec![2], rlmesh_spaces::DType::Float32).unwrap(),
        )
    }
}

#[async_trait]
impl rlmesh::lockstep::Served for Trials {
    fn contract(&self) -> rlmesh_spaces::EnvContract {
        let space = rlmesh_spaces::spaces::BoxSpaceBuilder::scalar(-1.0, 1.0, vec![2])
            .dtype(rlmesh_spaces::DType::Float32)
            .build()
            .unwrap();
        let declared = rlmesh_spaces::MetaValue::List(vec![rlmesh_spaces::MetaValue::String(
            "trial_index".into(),
        )]);
        rlmesh_spaces::EnvContract {
            id: "Trials-v0".into(),
            observation_space: Some(space.clone()),
            action_space: Some(space),
            metadata: self
                .declares
                .then(|| [(rlmesh::ENV_RESET_OPTIONS_KEY.to_string(), declared)].into()),
            render_mode: String::new(),
            num_envs: 1,
            autoreset_mode: rlmesh_spaces::AutoresetMode::Disabled,
        }
    }

    fn subset_step(&self) -> bool {
        false
    }

    async fn reset(
        &self,
        _: &[u32],
        _: Vec<i64>,
        options: Option<rlmesh_spaces::MetaMap>,
    ) -> Result<rlmesh::lockstep::Reset, rlmesh::EnvironmentError> {
        self.seen
            .lock()
            .unwrap()
            .push(options.and_then(|o| o.get("trial_index").cloned()));
        self.t.store(0, Ordering::SeqCst);
        Ok(rlmesh::lockstep::Reset {
            observations: vec![Self::zeros()],
            info: None,
        })
    }

    async fn step(
        &self,
        _: &[u32],
        _: Vec<rlmesh_spaces::SpaceValue>,
    ) -> Result<rlmesh::lockstep::Step, rlmesh::EnvironmentError> {
        let t = self.t.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(rlmesh::lockstep::Step {
            observations: vec![Self::zeros()],
            rewards: vec![1.0],
            terminated: vec![t >= 2],
            truncated: vec![false],
            info: None,
        })
    }

    async fn render(&self, _: u32) -> Result<Option<Vec<u8>>, rlmesh::EnvironmentError> {
        Ok(None)
    }

    async fn close(&self) -> Result<(), rlmesh::EnvironmentError> {
        Ok(())
    }
}

/// Acts zeros on a Gym env's box.
struct Still;

impl Predictor for Still {
    type Error = String;

    fn predict(&mut self, _: Observation, _: i64) -> Result<Vec<(String, Value)>, String> {
        let chunk = Tensor {
            dtype: DType::F32,
            shape: vec![1, 2],
            data: f32s(&[0.0; 2]),
        };
        Ok(vec![("a".into(), Value::Tensor(chunk))])
    }
}

/// The `trial_index` each of three episodes' resets carried, on a Gym env
/// that declares the option or not.
async fn trials_seen(declares: bool) -> Vec<Option<rlmesh_spaces::MetaValue>> {
    let binding = rlmesh::lockstep::bind(":0/trials").await.unwrap();
    let session = rlmesh_session::address::Address::participant(&binding.address())
        .unwrap()
        .joining()
        .unwrap();
    let cancel = tokio_util::sync::CancellationToken::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let env = Trials {
        declares,
        t: AtomicU64::new(0),
        seen: seen.clone(),
    };
    tokio::spawn(binding.serve(
        Arc::new(env),
        rlmesh::lockstep::ServeOptions::default(),
        cancel.clone(),
    ));
    let policy = Policy::parse(
        r#"{"hz": 10,
            "inputs": {"x": {"kind": "gym/box", "at": "obs",
                             "fields": {"value": ["float32", [2]]}}},
            "actions": {"a": {"kind": "gym/box", "at": "action",
                              "fields": {"value": ["float32", [2]]}}}}"#,
    )
    .unwrap();
    let options = RunOptions::named("p").robot("trials").unwrap();
    let bound = realtime::bind(&session, "p", &policy, &options, PriorityClass::Model, WAIT)
        .await
        .unwrap();
    assert_eq!(bound.clock(), Clock::Lockstep);
    let settings = RunSettings {
        episodes: Some(3),
        seed: Some(5),
        ..RunSettings::default()
    };
    let ran = std::thread::scope(|s| s.spawn(|| bound.run(&mut Still, settings)).join().unwrap());
    let Ok(RunOutcome::Lockstep(stats)) = ran else {
        panic!("a lockstep run: {ran:?}");
    };
    assert_eq!(stats.episodes.len(), 3);
    bound.close().await;
    cancel.cancel();
    seen.lock().unwrap().clone()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_gym_env_declaring_trial_index_gets_each_episode_s_ordinal_as_a_model_run_sends_it() {
    use rlmesh_spaces::MetaValue::Int;
    // Numbered from 0, one a reset, as `Model::run` numbers them.
    assert_eq!(
        trials_seen(true).await,
        [Some(Int(0)), Some(Int(1)), Some(Int(2))]
    );
    // An env that did not declare it is never handed the key.
    assert_eq!(trials_seen(false).await, [None, None, None]);
}
