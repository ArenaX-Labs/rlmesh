//! Runs, sessions and hooks driven end to end against local and served envs.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::spaces::{Discrete, DiscreteOptions, Space};
use crate::{
    EnvFactory, EnvServer, EnvServerOptions, ErrorClass, InfoDict, Kwargs, MadeEnv, Metadata,
    ModelOptions, Result, Value,
};

/// One recorded `reset(seed, options)`.
type Reset = (Option<i64>, Option<InfoDict>);

/// A three-step env: observations count steps, reward is the action.
struct Steps {
    step: i64,
    closes: Arc<AtomicUsize>,
    resets: Arc<Mutex<Vec<Reset>>>,
    metadata: Option<Metadata>,
}

impl Steps {
    fn new() -> Steps {
        Steps {
            step: 0,
            closes: Arc::default(),
            resets: Arc::default(),
            metadata: None,
        }
    }
}

impl Env for Steps {
    fn observation_space(&self) -> Space {
        Discrete::new(5, DiscreteOptions::default()).unwrap().into()
    }
    fn action_space(&self) -> Space {
        Discrete::new(2, DiscreteOptions::default()).unwrap().into()
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.resets.lock().unwrap().push((seed, options.cloned()));
        self.step = 0;
        Ok((Value::Int(0), InfoDict::new()))
    }
    fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.step += 1;
        let reward = match action {
            Some(Value::Int(action)) => action as f64,
            _ => -1.0,
        };
        let mut info = InfoDict::new();
        info.insert("is_success", Value::Bool(true));
        Ok((Value::Int(self.step), reward, self.step >= 3, false, info))
    }
    fn close(&mut self) -> Result<()> {
        self.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn metadata(&self) -> Option<Metadata> {
        self.metadata.clone()
    }
}

fn ones() -> Model {
    Model::new(
        |_observation: Value| Ok(Value::Int(1)),
        ModelOptions::default(),
    )
    .unwrap()
}

fn quick() -> EnvServerOptions {
    EnvServerOptions {
        options: Some(crate::ServeOptions {
            drain_timeout: Some(Duration::from_millis(100)),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Records every hook call, in order.
#[derive(Default)]
struct Log {
    calls: Vec<String>,
    context: Option<Arc<dyn RunContext>>,
    async_context: Vec<bool>,
    fail_on_step: Option<i64>,
}

impl RunHooks for Log {
    fn on_run_start(&mut self, context: Arc<dyn RunContext>) -> Result<()> {
        self.calls.push(format!(
            "run_start contract={} num_envs={}",
            context.contract()?.is_some(),
            context.num_envs()?
        ));
        self.async_context.push(drive::in_async_context());
        self.context = Some(context);
        Ok(())
    }
    fn on_episode_start(&mut self, episode: i64, seed: Option<i64>) -> Result<()> {
        self.calls.push(format!("episode_start {episode} {seed:?}"));
        Ok(())
    }
    fn on_step(&mut self, event: &StepEvent) -> Result<()> {
        self.async_context.push(drive::in_async_context());
        self.calls.push(format!(
            "step {} {} obs={} action={} reward={}",
            event.episode,
            event.step,
            crate::env::factory::py_repr(&event.observation),
            crate::env::factory::py_repr(&event.action),
            event.reward
        ));
        if self.fail_on_step == Some(event.step) {
            return Err(Error::invalid_argument("hook failed"));
        }
        Ok(())
    }
    fn on_episode_end(&mut self, result: &EpisodeResult) -> Result<()> {
        self.calls.push(format!(
            "episode_end {} steps={} reward={}",
            result.index, result.steps, result.reward
        ));
        Ok(())
    }
    fn on_run_end(&mut self, result: &RunResult) -> Result<()> {
        self.calls
            .push(format!("run_end episodes={}", result.num_episodes()));
        Ok(())
    }
}

fn expected_two_episodes() -> Vec<String> {
    let mut calls = vec!["run_start contract=false num_envs=1".to_string()];
    for episode in 0..2 {
        calls.push(format!("episode_start {episode} Some({})", 10 + episode));
        for step in 0..3 {
            calls.push(format!(
                "step {episode} {step} obs={step} action=1 reward=1"
            ));
        }
        calls.push(format!("episode_end {episode} steps=3 reward=3"));
    }
    calls.push("run_end episodes=2".to_string());
    calls
}

#[test]
fn model_run_drives_a_borrowed_local_env_and_calls_hooks_on_this_thread() {
    let model = ones();
    let mut env = Steps::new();
    let closes = Arc::clone(&env.closes);
    let mut log = Log::default();
    let result = model
        .run(
            &mut env,
            RunOptions {
                seeds: Some(vec![10, 11]),
                hooks: Some(&mut log),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(result.num_episodes(), 2);
    assert_eq!(result.total_steps(), 6);
    assert_eq!(result.mean_reward(), 3.0);
    assert_eq!(result.success_rate(), Some(1.0));
    let first = &result.episodes[0];
    assert_eq!(
        (first.index, first.seed, first.trial),
        (0, Some(10), Some(0))
    );
    assert!(first.terminated && !first.truncated);
    assert!(!result.telemetry.is_empty());
    assert_eq!(log.calls, expected_two_episodes());
    assert!(
        log.async_context.iter().all(|inside| !inside),
        "sync hooks run outside any async context"
    );
    let context = log.context.take().unwrap();
    assert_eq!(
        context.num_envs().unwrap(),
        1,
        "a native context outlives its run"
    );
    assert!(context.contract().unwrap().is_some());
    assert_eq!(closes.load(Ordering::SeqCst), 0, "the env is borrowed");
    // The model and the env stay usable.
    model.run(&mut env, RunOptions::default()).unwrap();
    model
        .run(
            &mut env,
            RunOptions {
                close_env: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(closes.load(Ordering::SeqCst), 1, "close_env closes it");
    model.close().unwrap();
}

#[test]
fn model_run_async_runs_under_any_executor_with_hooks_inside_the_future() {
    let model = ones();
    let mut env = Steps::new();
    let mut log = Log::default();
    let future = model.run_async(
        &mut env,
        RunOptions {
            seeds: Some(vec![10, 11]),
            hooks: Some(&mut log),
            ..Default::default()
        },
    );
    fn assert_send<T: Send>(_: &T) {}
    assert_send(&future);
    let result = futures::executor::block_on(future).unwrap();
    assert_eq!(result.num_episodes(), 2);
    assert_eq!(log.calls, expected_two_episodes());
    assert!(log.async_context.iter().all(|inside| *inside));
}

#[test]
fn a_run_dials_an_address_or_an_env_server() {
    let model = ones();
    let server = EnvServer::new(Steps::new(), quick()).unwrap();
    server.start().unwrap();
    let by_address = model.run(server.address(), RunOptions::default()).unwrap();
    assert_eq!(by_address.num_episodes(), 1);
    let by_server = run(
        &model,
        &server,
        RunOptions {
            episodes: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(by_server.num_episodes(), 2);
    server.shutdown().unwrap();
}

#[test]
fn zero_episodes_never_touch_the_env_but_fire_the_run_hooks() {
    let model = ones();
    let mut env = Steps::new();
    let resets = Arc::clone(&env.resets);
    let mut log = Log::default();
    let result = model
        .run(
            &mut env,
            RunOptions {
                episodes: Some(0),
                hooks: Some(&mut log),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(result, RunResult::default());
    assert!(resets.lock().unwrap().is_empty());
    assert_eq!(
        log.calls,
        vec!["run_start contract=false num_envs=1", "run_end episodes=0"]
    );
    let result = model
        .run(
            "tcp://127.0.0.1:1",
            RunOptions {
                seeds: Some(Vec::new()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(result.num_episodes(), 0, "seeds=[] is zero episodes");
}

#[test]
fn a_hook_error_aborts_the_run_and_on_run_end_still_fires() {
    let model = ones();
    let mut env = Steps::new();
    let mut log = Log {
        fail_on_step: Some(1),
        ..Default::default()
    };
    let error = model
        .run(
            &mut env,
            RunOptions {
                episodes: Some(2),
                hooks: Some(&mut log),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(error.to_string(), "hook failed");
    assert_eq!(error.class(), ErrorClass::ValueError);
    assert_eq!(log.calls.last().unwrap(), "run_end episodes=0");
    // The model is free for the next run.
    model.run(&mut env, RunOptions::default()).unwrap();
}

#[test]
fn options_are_checked_before_anything_dials() {
    let model = ones();
    let refused = |options: RunOptions<'_>| {
        let error = model.run("tcp://127.0.0.1:1", options).unwrap_err();
        assert_eq!(error.class(), ErrorClass::ValueError, "{error}");
        error.to_string()
    };
    assert_eq!(
        refused(RunOptions {
            episodes: Some(2),
            seeds: Some(vec![1]),
            ..Default::default()
        }),
        "episodes=2 does not match len(seeds)=1: pass one seed per episode, or only one of \
         the two"
    );
    assert_eq!(
        refused(RunOptions {
            execution_horizon: 0,
            ..Default::default()
        }),
        "execution_horizon must be >= 1, got 0"
    );
    assert_eq!(
        refused(RunOptions {
            execution_horizon: 1025,
            ..Default::default()
        }),
        "execution_horizon must be <= 1024, got 1025"
    );
    assert_eq!(
        refused(RunOptions {
            max_episode_steps: Some(0),
            ..Default::default()
        }),
        "max_episode_steps must be >= 1, got 0"
    );
    assert!(
        refused(RunOptions {
            max_episode_steps: Some(u64::MAX),
            ..Default::default()
        })
        .starts_with("max_episode_steps must be <= 9223372036854775807")
    );
    assert_eq!(
        refused(RunOptions {
            max_episode: Some(Duration::ZERO),
            ..Default::default()
        }),
        "max_episode_seconds must be > 0, got 0.0"
    );
    assert!(
        refused(RunOptions {
            prefetch_lead: u64::from(u32::MAX) + 1,
            ..Default::default()
        })
        .starts_with("prefetch_lead must be <= 4294967295")
    );
}

#[test]
fn a_model_runs_one_run_at_a_time() {
    struct Reenter<'m>(&'m Model, Option<Error>);
    impl RunHooks for Reenter<'_> {
        fn on_episode_start(&mut self, _episode: i64, _seed: Option<i64>) -> Result<()> {
            let mut env = Steps::new();
            self.1 = self.0.run(&mut env, RunOptions::default()).err();
            Ok(())
        }
    }
    let model = ones();
    let mut env = Steps::new();
    let mut hooks = Reenter(&model, None);
    model
        .run(
            &mut env,
            RunOptions {
                hooks: Some(&mut hooks),
                ..Default::default()
            },
        )
        .unwrap();
    let error = hooks.1.unwrap();
    assert_eq!(error.class(), ErrorClass::RuntimeError, "{error}");
    assert!(error.to_string().contains("already in progress"), "{error}");
}

#[test]
fn a_fired_cancel_token_fails_the_call() {
    let model = ones();
    let mut env = Steps::new();
    let cancel = crate::ext::CancelToken::new();
    cancel.cancel();
    let error = model
        .run(
            &mut env,
            RunOptions {
                cancel: Some(cancel.clone()),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(error.class(), ErrorClass::KeyboardInterrupt);
    // It re-armed.
    model
        .run(
            &mut env,
            RunOptions {
                cancel: Some(cancel),
                ..Default::default()
            },
        )
        .unwrap();
}

#[test]
fn sync_calls_refuse_async_code_naming_the_twin() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let model = ones();
    let error = runtime.block_on(async {
        tokio::spawn(async move {
            let mut env = Steps::new();
            model.run(&mut env, RunOptions::default()).unwrap_err()
        })
        .await
        .unwrap()
    });
    assert_eq!(error.class(), ErrorClass::RuntimeError);
    assert_eq!(
        error.to_string(),
        "Model::run() blocks; call Model::run_async() from async code"
    );
}

/// A factory whose made envs report their closes.
struct Factory {
    closes: Arc<AtomicUsize>,
    made: usize,
}

impl EnvFactory for Factory {
    fn make(&mut self, _kwargs: &Kwargs) -> Result<MadeEnv> {
        self.made += 1;
        let mut env = Steps::new();
        env.closes = Arc::clone(&self.closes);
        Ok(MadeEnv::env(env))
    }
}

#[test]
fn a_factory_target_builds_one_env_when_needed_and_closes_it() {
    let model = ones();
    let mut factory = Factory {
        closes: Arc::default(),
        made: 0,
    };
    model
        .run(
            EnvTarget::factory(&mut factory),
            RunOptions {
                episodes: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(factory.made, 0, "zero episodes builds none");
    let result = model
        .run(EnvTarget::factory(&mut factory), RunOptions::default())
        .unwrap();
    assert_eq!(result.num_episodes(), 1);
    assert_eq!(factory.made, 1);
    assert_eq!(factory.closes.load(Ordering::SeqCst), 1);
}

#[test]
fn a_session_drives_by_hand_runs_and_closes() {
    let ended = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen = Arc::clone(&ended);
    let model = Model::new(
        |_observation: Value| Ok(Value::Int(1)),
        ModelOptions {
            on_episode_end: Some(Box::new(move |id: &str| {
                seen.lock().unwrap().push(id.to_string());
                Ok(())
            })),
            ..Default::default()
        },
    )
    .unwrap();
    let mut env = Steps::new();
    let closes = Arc::clone(&env.closes);
    {
        let mut session = model.session(&mut env, SessionOptions::default()).unwrap();
        let (mut observation, _info) = session.reset(Some(3), None).unwrap();
        assert_eq!(observation, Value::Int(0));
        let mut steps = 0;
        while !session.done() {
            let action = session.predict(observation.clone()).unwrap();
            assert_eq!(action, Value::Int(1));
            let (next, reward, ..) = session.step(action).unwrap();
            assert_eq!(reward, 1.0);
            observation = next;
            steps += 1;
        }
        assert_eq!(steps, 3);
        assert_eq!(session.num_envs().unwrap(), 1);
        assert_eq!(
            session.selected_workflow_edition().unwrap_err().class(),
            ErrorClass::RuntimeError
        );
        let mut log = Log::default();
        let result = session
            .run(SessionRunOptions {
                seeds: Some(vec![10, 11]),
                hooks: Some(&mut log),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(result.num_episodes(), 2);
        assert_eq!(result.episodes[1].trial, Some(1));
        let mut expected = expected_two_episodes();
        expected[0] = "run_start contract=true num_envs=1".to_string();
        assert_eq!(log.calls, expected);
        let ops: Vec<_> = result.telemetry.iter().map(|row| row.op.as_str()).collect();
        assert_eq!(
            ops,
            ["model.predict", "env.reset", "env.step", "runner.round"]
        );
        let context = session.run_context().unwrap();
        session.close().unwrap();
        session.close().unwrap();
        assert_eq!(
            session.reset(None, None).unwrap_err().to_string(),
            "session is closed"
        );
        assert_eq!(
            context.num_envs().unwrap_err().to_string(),
            "session is closed"
        );
    }
    assert_eq!(
        ended.lock().unwrap().len(),
        3,
        "one episode end per episode"
    );
    assert_eq!(closes.load(Ordering::SeqCst), 0, "the env is borrowed");
    model.close().unwrap();
}

#[test]
fn a_session_dials_an_address_and_reports_its_edition() {
    let model = ones();
    let server = EnvServer::new(Steps::new(), quick()).unwrap();
    server.start().unwrap();
    let mut session = model
        .session(server.address(), SessionOptions::default())
        .unwrap();
    let result = session.run(SessionRunOptions::default()).unwrap();
    assert_eq!(result.num_episodes(), 1);
    assert_eq!(result.episodes[0].reward, 3.0);
    assert!(!session.selected_workflow_edition().unwrap().is_empty());
    session.close().unwrap();
    drop(session);
    server.shutdown().unwrap();
}

#[test]
fn random_sample_runs_on_the_session_loop() {
    let mut env = Steps::new();
    let result = run(
        RANDOM_SAMPLE,
        &mut env,
        RunOptions {
            episodes: Some(2),
            max_episode_steps: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.num_episodes(), 2);
    assert!(
        result
            .episodes
            .iter()
            .all(|episode| episode.steps == 2 && episode.truncated)
    );
    let error = run(
        RANDOM_SAMPLE,
        &mut env,
        RunOptions {
            prefetch_lead: 1,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
    assert_eq!(RANDOM_SAMPLE.to_string(), "RANDOM_SAMPLE");
}

#[test]
fn a_session_drops_cleanly_and_ends_its_episode() {
    let ended = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&ended);
    let model = Model::new(
        |_observation: Value| Ok(Value::Int(0)),
        ModelOptions {
            on_episode_end: Some(Box::new(move |_id: &str| {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })),
            ..Default::default()
        },
    )
    .unwrap();
    let mut env = Steps::new();
    let mut session = model.session(&mut env, SessionOptions::default()).unwrap();
    session.reset(None, None).unwrap();
    drop(session);
    assert_eq!(ended.load(Ordering::SeqCst), 1);
}

#[test]
fn results_print_as_python_does() {
    let episode = EpisodeResult {
        index: 0,
        seed: Some(3),
        steps: 2,
        reward: 1.5,
        terminated: true,
        truncated: false,
        success: None,
        duration_s: 0.25,
        predict_ms: Some(1.0),
        step_ms: None,
        trial: Some(0),
    };
    assert_eq!(
        episode.to_string(),
        "EpisodeResult(index=0, seed=3, steps=2, reward=1.5, terminated=True, truncated=False, \
         success=None, duration_s=0.25, predict_ms=1.0, step_ms=None, trial=0)"
    );
    let result = RunResult {
        advisories: Vec::new(),
        episodes: vec![episode],
        telemetry: vec![TelemetryRow {
            op: "env.step".into(),
            component: "env".into(),
            metric: "rpc.total".into(),
            unit: "ms".into(),
            count: 3,
            avg: 1.0,
            p50: 1.0,
            p95: 2.5,
            p99: 2.5,
        }],
    };
    assert_eq!(
        result.to_string(),
        "RunResult(episodes=1, mean_reward=1.500, total_steps=2)"
    );
    assert_eq!(result.success_rate(), None);
    assert_eq!(
        result.format_telemetry(),
        "op        metric     unit  count   avg   p50   p95   p99\n\
         env.step  rpc.total  ms        3  1.00  1.00  2.50  2.50"
    );
    assert_eq!(
        RunResult::default().format_telemetry(),
        "(no telemetry recorded)"
    );
    assert_eq!(
        result.telemetry[0].to_string(),
        "TelemetryRow(op='env.step', component='env', metric='rpc.total', unit='ms', count=3, \
         avg=1.0, p50=1.0, p95=2.5, p99=2.5)"
    );
}

/// Two lanes that each end after three steps and restart on the next step
/// (`NEXT_STEP` autoreset).
struct Lanes {
    counts: Vec<i64>,
    ended: Vec<bool>,
}

impl crate::VectorEnv for Lanes {
    fn num_envs(&self) -> u64 {
        self.counts.len() as u64
    }
    fn single_observation_space(&self) -> Space {
        Discrete::new(5, DiscreteOptions::default()).unwrap().into()
    }
    fn single_action_space(&self) -> Space {
        Discrete::new(2, DiscreteOptions::default()).unwrap().into()
    }
    fn reset(
        &mut self,
        _seed: Option<crate::VectorSeed>,
        _options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.counts.iter_mut().for_each(|count| *count = 0);
        self.ended.iter_mut().for_each(|ended| *ended = false);
        Ok((self.observations(), InfoDict::new()))
    }
    fn step(
        &mut self,
        _actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        let mut rewards = Vec::new();
        let mut terminated = Vec::new();
        for lane in 0..self.counts.len() {
            if self.ended[lane] {
                self.counts[lane] = 0;
                self.ended[lane] = false;
                rewards.push(0.0);
                terminated.push(false);
            } else {
                self.counts[lane] += 1;
                self.ended[lane] = self.counts[lane] >= 3;
                rewards.push(1.0);
                terminated.push(self.ended[lane]);
            }
        }
        let truncated = vec![false; self.counts.len()];
        Ok((
            self.observations(),
            rewards,
            terminated,
            truncated,
            InfoDict::new(),
        ))
    }
    fn metadata(&self) -> Option<Metadata> {
        let mut metadata = Metadata::new();
        metadata.insert("autoreset_mode", Value::Text("NextStep".into()));
        Some(metadata)
    }
}

impl Lanes {
    fn observations(&self) -> Value {
        Value::List(self.counts.iter().map(|count| Value::Int(*count)).collect())
    }
}

#[test]
fn a_borrowed_vector_env_runs_its_lanes_on_the_native_loop() {
    // A batch of observations takes one action per lane.
    let model = Model::new(
        |observation: Value| {
            let lanes = match &observation {
                Value::Tensor(tensor) => tensor.shape().first().copied().unwrap_or(1),
                Value::List(items) => items.len(),
                _ => return Ok(Value::Int(1)),
            };
            Ok(Value::List(vec![Value::Int(1); lanes]))
        },
        ModelOptions::default(),
    )
    .unwrap();
    let mut env = Lanes {
        counts: vec![0; 2],
        ended: vec![false; 2],
    };
    let result = model
        .run(
            EnvTarget::vector(&mut env),
            RunOptions {
                episodes: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(result.num_episodes(), 2);
    assert!(result.episodes.iter().all(|episode| episode.steps == 3));
    assert!(
        result
            .episodes
            .iter()
            .all(|episode| episode.trial.is_none())
    );
}

#[test]
fn a_session_has_async_twins_and_closes_a_factory_built_env() {
    let model = ones();
    let mut factory = Factory {
        closes: Arc::default(),
        made: 0,
    };
    let closes = Arc::clone(&factory.closes);
    futures::executor::block_on(async {
        let mut session = model
            .session_async(EnvTarget::factory(&mut factory), SessionOptions::default())
            .await
            .unwrap();
        let (observation, _) = session.reset_async(Some(1), None).await.unwrap();
        let action = session.predict_async(observation).await.unwrap();
        let (_, reward, ..) = session.step_async(action).await.unwrap();
        assert_eq!(reward, 1.0);
        let mut log = Log::default();
        let result = session
            .run_async(SessionRunOptions {
                hooks: Some(&mut log),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(result.num_episodes(), 1);
        assert!(log.async_context.iter().all(|inside| *inside));
        session.close_async().await.unwrap();
    });
    assert_eq!(factory.made, 1);
    assert_eq!(
        closes.load(Ordering::SeqCst),
        1,
        "a built env closes with the session"
    );
}

#[test]
fn a_session_refuses_a_vector_env() {
    let model = ones();
    let mut env = Lanes {
        counts: vec![0; 2],
        ended: vec![false; 2],
    };
    let mut session = model
        .session(EnvTarget::vector(&mut env), SessionOptions::default())
        .unwrap();
    let error = session.reset(None, None).unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
    assert!(error.to_string().contains("num_envs=2"), "{error}");
}

/// A model served on a free port by a library thread, stopped by `cancel`.
struct ServedModel {
    address: String,
    cancel: crate::ext::CancelToken,
    thread: Option<std::thread::JoinHandle<Result<()>>>,
}

impl ServedModel {
    fn start() -> ServedModel {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let address = format!("tcp://127.0.0.1:{port}");
        let cancel = crate::ext::CancelToken::new();
        let serve_address = address.clone();
        let serve_cancel = cancel.clone();
        let thread = std::thread::spawn(move || {
            ones().serve(
                &serve_address,
                crate::ModelServeOptions {
                    cancel: Some(serve_cancel),
                    ..Default::default()
                },
            )
        });
        // Wait until the model answers a handshake.
        for _ in 0..200 {
            let probe = drive::block_on_library(async {
                crate::binding::model::ModelClient::connect(
                    &address,
                    crate::env::sync_wire::env_contract_of(&Steps::new())
                        .unwrap()
                        .to_wire()
                        .unwrap(),
                )
                .await
            });
            if probe.is_ok() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        ServedModel {
            address,
            cancel,
            thread: Some(thread),
        }
    }
}

impl Drop for ServedModel {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn steps_server() -> EnvServer {
    let server = EnvServer::new(Steps::new(), quick()).unwrap();
    server.start().unwrap();
    server
}

#[test]
fn remote_model_construction_checks_without_dialing() {
    let err = crate::RemoteModel::new(crate::RemoteModelOptions::default()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert_eq!(err.to_string(), "address or host/port is required");
    let err = crate::RemoteModel::new(crate::RemoteModelOptions::default().address("bogus://x"))
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert!(err.to_string().starts_with("invalid address: "), "{err}");
    // Kept as given, and nothing listens there: construction does no I/O.
    let model =
        crate::RemoteModel::new(crate::RemoteModelOptions::default().address("127.0.0.1:1"))
            .unwrap();
    assert_eq!(model.address(), "127.0.0.1:1");
    assert_eq!(model.to_string(), "RemoteModel(address='127.0.0.1:1')");
    let shared = model.clone();
    assert_eq!(shared.address(), model.address());
    let err = crate::ext::remote_model_with_token(
        crate::RemoteModelOptions::default().address("127.0.0.1:1"),
        " padded",
    )
    .unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
}

#[test]
fn remote_model_sessions_refuse_bad_targets_before_dialing() {
    // Nothing serves the model: every refusal below precedes the dial.
    let model =
        crate::RemoteModel::new(crate::RemoteModelOptions::default().address("127.0.0.1:1"))
            .unwrap();
    let mut env = Steps::new();
    let err = model
        .session(&mut env, SessionOptions::default())
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::TypeError);
    let server = steps_server();
    let err = model
        .session(
            &server,
            SessionOptions {
                trust_entrypoints: Some(true),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    let err = model
        .session(
            &server,
            SessionOptions {
                execution_horizon: 0,
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    let lanes = EnvServer::lanes(vec![Steps::new(), Steps::new()], quick()).unwrap();
    let err = model
        .session(&lanes, SessionOptions::default())
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert!(err.to_string().contains("num_envs=2"), "{err}");
    // An unreachable model is the ConnectionError class, named.
    let err = model
        .session(server.address(), SessionOptions::default())
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::ConnectionError, "{err}");
    assert!(
        err.to_string()
            .starts_with("could not connect to a model at '127.0.0.1:1'"),
        "{err}"
    );
}

#[test]
fn remote_model_sessions_drive_every_remote_target() {
    let served = ServedModel::start();
    let model =
        crate::RemoteModel::new(crate::RemoteModelOptions::default().address(&served.address))
            .unwrap();
    let server = steps_server();

    // An address: dialed now, owned by the session.
    let mut session = model
        .session(server.address(), SessionOptions::default())
        .unwrap();
    assert!(!session.selected_workflow_edition().unwrap().is_empty());
    let (observation, _) = session.reset(Some(1), None).unwrap();
    let action = session.predict(observation).unwrap();
    assert_eq!(action, Value::Int(1));
    let (_, reward, _, _, _) = session.step(action).unwrap();
    assert_eq!(reward, 1.0);
    session.close().unwrap();

    // A borrowed RemoteEnv: driven through its own connection, handed back
    // pinned to the session's floor.
    let mut remote =
        crate::RemoteEnv::new(crate::RemoteEnvOptions::default().address(server.address()))
            .unwrap();
    let floor = {
        let mut session = model
            .session(&mut remote, SessionOptions::default())
            .unwrap();
        let result = session.run(SessionRunOptions::default()).unwrap();
        assert_eq!(result.num_episodes(), 1);
        assert_eq!(result.episodes[0].reward, 3.0);
        session.selected_workflow_edition().unwrap()
    };
    assert_eq!(remote.selected_workflow_edition(), floor);
    remote.reset(None, None).unwrap();
    remote.close().unwrap();

    // An EnvServer: dialed at first env use.
    let mut session = model.session(&server, SessionOptions::default()).unwrap();
    session.reset(None, None).unwrap();
    drop(session);

    // rlmesh::run over a served model runs the session loop.
    let result = crate::run(
        &model,
        server.address(),
        RunOptions {
            episodes: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.num_episodes(), 2);

    // The async twin runs on any executor.
    let address = server.address().to_string();
    let result = futures::executor::block_on(async {
        let mut session = model
            .session_async(address.as_str(), SessionOptions::default())
            .await?;
        let out = session.run_async(SessionRunOptions::default()).await;
        session.close_async().await?;
        out
    })
    .unwrap();
    assert_eq!(result.num_episodes(), 1);
    server.shutdown().unwrap();
}

#[test]
fn a_native_run_dials_a_borrowed_remote_env_s_address() {
    let server = steps_server();
    let mut remote =
        crate::RemoteEnv::new(crate::RemoteEnvOptions::default().address(server.address()))
            .unwrap();
    // An idle handle holds no session slot: the run dials its address.
    let result = ones()
        .run(
            &mut remote,
            RunOptions {
                episodes: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(result.num_episodes(), 1);
    // A handle that was driven holds the slot, and the run says so.
    remote.reset(None, None).unwrap();
    let err = ones()
        .run(
            &mut remote,
            RunOptions {
                episodes: Some(1),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::RuntimeError, "{err}");
    assert!(
        err.to_string()
            .starts_with("the target env already has an active session"),
        "{err}"
    );
    remote.close().unwrap();
    server.shutdown().unwrap();
}

struct AdapterSteps {
    step: i64,
}
impl Env for AdapterSteps {
    fn observation_space(&self) -> Space {
        crate::spaces::Box::new(-10.0, 10.0, vec![1], Default::default())
            .unwrap()
            .into()
    }
    fn action_space(&self) -> Space {
        crate::spaces::Box::new(-10.0, 10.0, vec![1], Default::default())
            .unwrap()
            .into()
    }
    fn reset(
        &mut self,
        _seed: Option<i64>,
        _options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.step = 0;
        Ok((Value::List(vec![Value::Float(2.)]), InfoDict::new()))
    }
    fn step(&mut self, _action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.step += 1;
        Ok((
            Value::List(vec![Value::Float(3.)]),
            1.,
            true,
            false,
            InfoDict::new(),
        ))
    }
    fn render_mode(&self) -> Option<String> {
        Some("rgb_array".into())
    }
    fn render(&mut self) -> Result<Option<Value>> {
        Ok(Some(Value::Int(self.step)))
    }
}
fn adapter_action() -> crate::adapters::Action {
    use crate::adapters::*;
    Action::new([Actuator {
        role: Some(ACTION_GRIPPER.into()),
        dim: 1,
        ..Default::default()
    }])
}
fn adapter_steps() -> crate::adapters::Tagged<AdapterSteps> {
    use crate::adapters::*;
    tag(
        AdapterSteps { step: 0 },
        &EnvTags::new(StateTag::new(GRIPPER_POS), adapter_action()).unwrap(),
        Default::default(),
    )
    .unwrap()
}
struct TextPolicy;
impl crate::Policy for TextPolicy {
    fn predict(&mut self, observation: Value) -> Result<Value> {
        let input = observation.as_dict()?;
        assert_eq!(
            input.get("instruction"),
            Some(&Value::List(vec![Value::Text("override".into())]))
        );
        assert!(input.contains_key("state"));
        Ok(Value::List(vec![Value::Float(0.)]))
    }
}
fn text_model() -> Model {
    use crate::adapters::*;
    Model::new(
        TextPolicy,
        ModelOptions {
            spec: Some(
                ModelSpec::new(
                    InputNode::dict([
                        ("state", State::new(GRIPPER_POS).into()),
                        (
                            "missing_camera",
                            Image {
                                optional: true,
                                height: Some(1),
                                width: Some(1),
                                channels: Some(3),
                                ..Image::new(IMAGE_PRIMARY)
                            }
                            .into(),
                        ),
                        (
                            "instruction",
                            Text {
                                container: TextContainer::List,
                                fill: Some("fallback".into()),
                                ..Text::new(INSTRUCTION)
                            }
                            .into(),
                        ),
                    ]),
                    adapter_action(),
                )
                .unwrap()
                .into(),
            ),
            ..Default::default()
        },
    )
    .unwrap()
}
#[derive(Default)]
struct ReadHooks {
    context: Option<Arc<dyn RunContext>>,
    reads: usize,
}
impl RunHooks for ReadHooks {
    fn on_run_start(&mut self, context: Arc<dyn RunContext>) -> Result<()> {
        self.context = Some(context);
        Ok(())
    }
    fn on_step(&mut self, event: &StepEvent) -> Result<()> {
        let value = event.read(crate::adapters::GRIPPER_POS)?;
        assert!(matches!(value, Value::Tensor(_)));
        let context = self.context.as_ref().unwrap();
        let mut sources = context.frame_sources()?;
        assert_eq!(sources.render_label.as_deref(), Some("render"));
        assert!(sources.render.as_mut().is_some());
        assert_eq!(sources.render.as_mut().unwrap().call()?, Value::Int(1));
        self.reads += 1;
        Ok(())
    }
    fn on_run_end(&mut self, _result: &RunResult) -> Result<()> {
        let mut sources = self.context.as_ref().unwrap().frame_sources()?;
        assert!(sources.render.as_mut().is_some());
        assert_eq!(sources.render.as_mut().unwrap().call()?, Value::Int(1));
        Ok(())
    }
}

#[test]
fn adapter_session_readers_instruction_and_scoped_render() {
    let model = text_model();
    let mut env = adapter_steps();
    let mut session = model
        .session(&mut env, SessionOptions::default().instruction("override"))
        .unwrap();
    let (observation, _) = session.reset(None, None).unwrap();
    let reader = session
        .reader(&[crate::adapters::GRIPPER_POS.into()])
        .unwrap();
    assert_eq!(reader.roles(), [crate::adapters::GRIPPER_POS]);
    assert!(
        reader
            .call(&observation)
            .unwrap()
            .contains_key(crate::adapters::GRIPPER_POS)
    );
    assert_eq!(
        session.observation_roles().unwrap().states,
        [crate::adapters::GRIPPER_POS]
    );
    assert!(session.reader(&[]).is_err());
    let context = session.run_context().unwrap();
    assert!(context.frame_sources().unwrap().render.is_none());
    {
        let mut sources = session.frame_sources().unwrap();
        assert_eq!(
            sources.render.as_mut().unwrap().call().unwrap(),
            Value::Int(0)
        );
    }
    let mut hooks = ReadHooks::default();
    let result = session
        .run(SessionRunOptions {
            hooks: Some(&mut hooks),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(hooks.reads, 1);
    assert!(!result.advisories.is_empty());
    session.close().unwrap();
    assert_eq!(
        context.frame_sources().unwrap_err().class(),
        ErrorClass::RuntimeError
    );
}
#[test]
fn native_adapter_instruction_and_kept_context_read() {
    let model = text_model();
    let mut env = adapter_steps();
    let mut hooks = ReadHooks::default();
    let result = model
        .run(
            &mut env,
            RunOptions {
                instruction: Some("override".into()),
                hooks: Some(&mut hooks),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(hooks.reads, 1);
    assert!(!result.advisories.is_empty());
    let context = hooks.context.unwrap();
    assert!(context.contract().unwrap().is_some());
    assert!(context.frame_sources().unwrap().render.is_none());
    assert!(
        context
            .read(
                &Value::List(vec![Value::Float(5.)]),
                &crate::adapters::GRIPPER_POS.into()
            )
            .is_ok()
    );
}

#[test]
fn closure_resolution_does_not_consume_the_authored_warning() {
    let _lock = crate::warnings::tests::HANDLER_LOCK.lock().unwrap();
    let warnings = Arc::new(AtomicUsize::new(0));
    let count = warnings.clone();
    crate::set_warning_handler(Some(Box::new(move |warning| {
        if warning.category == crate::WarningCategory::WorkflowEditionWarning {
            count.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    })));
    let closure = ones();
    let mut env = Steps::new();
    let session = closure
        .session(&mut env, SessionOptions::default())
        .unwrap();
    drop(session);
    closure.run(&mut env, RunOptions::default()).unwrap();
    assert_eq!(warnings.load(Ordering::SeqCst), 0);
    struct Authored;
    impl crate::Policy for Authored {
        fn predict(&mut self, _: Value) -> Result<Value> {
            Ok(Value::Int(0))
        }
    }
    let authored = Model::new(Authored, ModelOptions::default()).unwrap();
    let session = authored
        .session(&mut env, SessionOptions::default())
        .unwrap();
    drop(session);
    crate::set_warning_handler(None);
    assert_eq!(warnings.load(Ordering::SeqCst), 1);
}

#[test]
fn short_chunk_warnings_run_on_the_polling_thread_and_can_abort() {
    let _lock = crate::warnings::tests::HANDLER_LOCK.lock().unwrap();
    struct Chunk;
    impl crate::Policy for Chunk {
        const CORNERS: crate::Corners =
            crate::Corners::PREDICT.union(crate::Corners::PREDICT_CHUNK);
        fn predict(&mut self, _: Value) -> Result<Value> {
            Ok(Value::Int(0))
        }
        fn predict_chunk(&mut self, _: Value, _: u32) -> Result<Value> {
            Ok(Value::List(vec![Value::Int(0)]))
        }
    }
    let model = Model::new(Chunk, ModelOptions::default()).unwrap();
    let mut env = Steps::new();
    let mut session = model
        .session(
            &mut env,
            SessionOptions {
                execution_horizon: 2,
                ..Default::default()
            },
        )
        .unwrap();
    session.reset(None, None).unwrap();
    let caller = std::thread::current().id();
    crate::set_warning_handler(Some(Box::new(move |warning| {
        if warning.category == crate::WarningCategory::RuntimeWarning {
            assert_eq!(std::thread::current().id(), caller);
            return Err(crate::Error::Runtime("warning refused".into()));
        }
        Ok(())
    })));
    let error = session.predict(Value::Int(0)).unwrap_err();
    crate::set_warning_handler(None);
    assert_eq!(error, crate::Error::Runtime("warning refused".into()));
}

#[test]
fn dropped_predict_never_starts_a_queued_model_callback() {
    let starts = Arc::new(AtomicUsize::new(0));
    let count = starts.clone();
    let model = Model::new(
        move |_: Value| {
            count.fetch_add(1, Ordering::SeqCst);
            Ok(Value::Int(0))
        },
        ModelOptions::default(),
    )
    .unwrap();
    let mut env = Steps::new();
    let mut session = model.session(&mut env, SessionOptions::default()).unwrap();
    session.reset(None, None).unwrap();
    let shared = model.shared().clone();
    let (entered, started) = std::sync::mpsc::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let busy = std::thread::spawn(move || {
        shared.call(move |_| {
            entered.send(()).unwrap();
            blocked.recv().unwrap();
            Ok(())
        })
    });
    started.recv().unwrap();
    super::drive::block_on_library(async {
        assert!(
            tokio::time::timeout(
                Duration::from_millis(50),
                session.predict_async(Value::Int(0))
            )
            .await
            .is_err()
        );
    });
    release.send(()).unwrap();
    busy.join().unwrap().unwrap();
    model.shared().call(|_| Ok(())).unwrap();
    assert_eq!(starts.load(Ordering::SeqCst), 0);
    session.predict(Value::Int(0)).unwrap();
    assert_eq!(starts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn async_session_drop_enqueues_episode_end_before_model_close() {
    tokio::spawn(async move {
        let ended = Arc::new(AtomicUsize::new(0));
        let count = ended.clone();
        let model = Model::new(
            |_: Value| Ok(Value::Int(0)),
            ModelOptions {
                on_episode_end: Some(Box::new(move |_| {
                    count.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })),
                ..Default::default()
            },
        )
        .unwrap();
        let mut env = Steps::new();
        let mut session = model
            .session_async(&mut env, SessionOptions::default())
            .await
            .unwrap();
        session.reset_async(None, None).await.unwrap();
        drop(session);
        let closing = model.clone();
        std::thread::spawn(move || closing.close())
            .join()
            .unwrap()
            .unwrap();
        assert_eq!(ended.load(Ordering::SeqCst), 1);
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn async_session_drop_closes_factory_envs() {
    tokio::spawn(async move {
        let model = ones();
        let mut factory = Factory {
            closes: Arc::default(),
            made: 0,
        };
        let closes = factory.closes.clone();
        let mut session = model
            .session_async(EnvTarget::factory(&mut factory), SessionOptions::default())
            .await
            .unwrap();
        session.reset_async(None, None).await.unwrap();
        drop(session);
        for _ in 0..100 {
            if closes.load(Ordering::SeqCst) != 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    })
    .await
    .unwrap();
}

#[test]
fn cancelling_a_slow_local_step_discards_its_result() {
    struct Slow {
        token: crate::ext::CancelToken,
        steps: usize,
    }
    impl crate::Env for Slow {
        fn observation_space(&self) -> Space {
            Discrete::new(4, DiscreteOptions::default()).unwrap().into()
        }
        fn action_space(&self) -> Space {
            self.observation_space()
        }
        fn reset(&mut self, _: Option<i64>, _: Option<&InfoDict>) -> Result<(Value, InfoDict)> {
            Ok((Value::Int(0), InfoDict::new()))
        }
        fn step(&mut self, _: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
            self.steps += 1;
            if self.steps == 1 {
                self.token.cancel();
            }
            Ok((Value::Int(1), 1., true, false, InfoDict::new()))
        }
    }
    let token = crate::ext::CancelToken::new();
    let mut env = Slow {
        token: token.clone(),
        steps: 0,
    };
    let model = ones();
    let mut session = model
        .session(
            &mut env,
            SessionOptions {
                cancel: Some(token),
                ..Default::default()
            },
        )
        .unwrap();
    session.reset(None, None).unwrap();
    assert_eq!(
        session.step(Value::Int(0)).unwrap_err(),
        crate::Error::Cancelled
    );
    assert!(
        !session.done(),
        "cancelled results do not finish the episode"
    );
    session.step(Value::Int(0)).unwrap();
    assert!(session.done());
}

#[test]
fn one_firing_cancels_two_sessions_sharing_a_token() {
    let token = crate::ext::CancelToken::new();
    let (ready, entered) = std::sync::mpsc::channel();
    let mut threads = Vec::new();
    for _ in 0..2 {
        let ready = ready.clone();
        let token = token.clone();
        threads.push(std::thread::spawn(move || {
            struct Slow(std::sync::mpsc::Sender<()>);
            impl crate::Env for Slow {
                fn observation_space(&self) -> Space {
                    Discrete::new(2, DiscreteOptions::default()).unwrap().into()
                }
                fn action_space(&self) -> Space {
                    self.observation_space()
                }
                fn reset(
                    &mut self,
                    _: Option<i64>,
                    _: Option<&InfoDict>,
                ) -> Result<(Value, InfoDict)> {
                    Ok((Value::Int(0), InfoDict::new()))
                }
                fn step(&mut self, _: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
                    self.0.send(()).unwrap();
                    std::thread::sleep(Duration::from_millis(150));
                    Ok((Value::Int(0), 0., false, false, InfoDict::new()))
                }
            }
            let model = ones();
            let mut env = Slow(ready);
            let mut session = model
                .session(
                    &mut env,
                    SessionOptions {
                        cancel: Some(token),
                        ..Default::default()
                    },
                )
                .unwrap();
            session.reset(None, None).unwrap();
            session.step(Value::Int(0))
        }));
    }
    for _ in 0..2 {
        entered.recv_timeout(Duration::from_secs(2)).unwrap();
    }
    token.cancel();
    for thread in threads {
        assert_eq!(thread.join().unwrap().unwrap_err(), crate::Error::Cancelled);
    }
    token.cancel();
    assert!(token.take());
    assert!(!token.take());
}

#[test]
fn dropping_remote_session_creation_returns_the_borrowed_env() {
    let server = steps_server();
    let mut remote =
        crate::RemoteEnv::new(crate::RemoteEnvOptions::default().address(server.address()))
            .unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let model = crate::RemoteModel::new(
        crate::RemoteModelOptions::default().address(listener.local_addr().unwrap().to_string()),
    )
    .unwrap();
    super::drive::block_on_library(async {
        assert!(
            tokio::time::timeout(
                Duration::from_millis(60),
                model.session_async(&mut remote, SessionOptions::default())
            )
            .await
            .is_err()
        );
    });
    remote
        .reset(None, None)
        .expect("a cancelled session dial returns the borrowed client core");
    remote.close().unwrap();
    server.shutdown().unwrap();
}

#[test]
fn cancelling_a_failed_model_dial_does_not_poison_the_next_call() {
    let server = steps_server();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let model = crate::RemoteModel::new(
        crate::RemoteModelOptions::default()
            .address(listener.local_addr().unwrap().to_string())
            .connect_timeout(Duration::from_millis(100)),
    )
    .unwrap();
    let cancel = crate::ext::CancelToken::new();
    let fired = cancel.clone();
    let thread = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(30));
        fired.cancel();
    });
    let error = model
        .session(
            &server,
            SessionOptions {
                cancel: Some(cancel.clone()),
                ..Default::default()
            },
        )
        .unwrap_err();
    thread.join().unwrap();
    assert_eq!(error, crate::Error::Cancelled);
    let error = model
        .session(
            &server,
            SessionOptions {
                cancel: Some(cancel),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_ne!(
        error,
        crate::Error::Cancelled,
        "the next dial must see its own error"
    );
    server.shutdown().unwrap();
}

#[test]
fn dropped_native_run_closes_its_factory_env() {
    let model = Model::new(
        |_: Value| {
            std::thread::sleep(Duration::from_millis(100));
            Ok(Value::Int(0))
        },
        ModelOptions::default(),
    )
    .unwrap();
    let mut factory = Factory {
        closes: Arc::default(),
        made: 0,
    };
    let closes = factory.closes.clone();
    super::drive::block_on_library(async {
        assert!(
            tokio::time::timeout(
                Duration::from_millis(50),
                model.run_async(EnvTarget::factory(&mut factory), RunOptions::default())
            )
            .await
            .is_err()
        );
        for _ in 0..100 {
            if closes.load(Ordering::SeqCst) > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });
    assert_eq!(closes.load(Ordering::SeqCst), 1);
}

#[test]
fn cancellation_wins_over_a_failed_env_dial_too() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let cancel = crate::ext::CancelToken::new();
    let fired = cancel.clone();
    let peer = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        fired.cancel();
        drop(stream);
    });
    let model =
        crate::RemoteModel::new(crate::RemoteModelOptions::default().address("127.0.0.1:1"))
            .unwrap();
    let result = model.session(
        address.as_str(),
        SessionOptions {
            cancel: Some(cancel),
            ..Default::default()
        },
    );
    peer.join().unwrap();
    assert_eq!(result.unwrap_err(), crate::Error::Cancelled);
}

struct ViewedSteps(Steps);
impl Env for ViewedSteps {
    fn observation_space(&self) -> Space {
        self.0.observation_space()
    }
    fn action_space(&self) -> Space {
        self.0.action_space()
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.0.reset(seed, options)
    }
    fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.0.step(action)
    }
    fn render_mode(&self) -> Option<String> {
        Some("rgb_array".into())
    }
    fn render(&mut self) -> Result<Option<Value>> {
        Ok(Some(Value::Tensor(crate::Tensor::new(
            &[128; 48],
            &[4, 4, 3],
            crate::DType::Uint8,
        )?)))
    }
}
fn http_view() -> crate::View {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    crate::View {
        backend: crate::ViewBackend::Http,
        port: listener.local_addr().unwrap().port(),
        ..Default::default()
    }
}
fn viewer_request(port: u16, route: &str) -> String {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    write!(
        stream,
        "GET {route} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    response
}
#[test]
fn native_run_refuses_view_and_session_validates_before_env_use() {
    let model = ones();
    let mut env = ViewedSteps(Steps::new());
    let err = model
        .run(
            &mut env,
            RunOptions {
                view: Some(crate::View::default()),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::TypeError);
    assert!(env.0.resets.lock().unwrap().is_empty());
    let err = model
        .session(
            &mut env,
            SessionOptions {
                view: Some(crate::View {
                    fps: 0,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
}
#[test]
fn viewer_skip_truncates_and_quit_returns_a_partial_success() {
    let view = http_view();
    let port = view.port;
    let mut env = ViewedSteps(Steps::new());
    let model = ones();
    let mut session = model
        .session(
            &mut env,
            SessionOptions {
                view: Some(view),
                ..Default::default()
            },
        )
        .unwrap();
    session.reset(None, None).unwrap();
    assert!(viewer_request(port, "/skip").contains("200 OK"));
    let (_, _, terminated, truncated, _) = session.step(Value::Int(1)).unwrap();
    assert!(!terminated && truncated);
    assert!(session.done());
    assert!(viewer_request(port, "/quit").contains("200 OK"));
    let result = session
        .run(SessionRunOptions {
            episodes: Some(5),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(result.num_episodes(), 1);
    assert!(result.episodes[0].truncated);
    session.close().unwrap();
}
#[test]
fn view_paces_steps_without_charging_telemetry_and_holds_only_success() {
    let mut view = http_view();
    view.step_hz = Some(5.);
    view.hold = crate::ViewHold::For(Duration::from_millis(70));
    let mut env = ViewedSteps(Steps::new());
    let model = ones();
    let mut session = model
        .session(
            &mut env,
            SessionOptions {
                view: Some(view),
                ..Default::default()
            },
        )
        .unwrap();
    let started = std::time::Instant::now();
    let result = session.run(SessionRunOptions::default()).unwrap();
    assert!(started.elapsed() >= Duration::from_millis(300));
    for row in &result.telemetry {
        assert!(row.avg < 100., "{} included pacing: {}", row.op, row.avg);
    }
    let started = std::time::Instant::now();
    session.close().unwrap();
    assert!(started.elapsed() >= Duration::from_millis(70));

    let mut view = http_view();
    view.hold = crate::ViewHold::For(Duration::from_secs(5));
    let mut env = ViewedSteps(Steps::new());
    let mut session = model
        .session(
            &mut env,
            SessionOptions {
                view: Some(view),
                ..Default::default()
            },
        )
        .unwrap();
    let mut hooks = Log {
        fail_on_step: Some(0),
        ..Default::default()
    };
    assert!(
        session
            .run(SessionRunOptions {
                hooks: Some(&mut hooks),
                ..Default::default()
            })
            .is_err()
    );
    let started = std::time::Instant::now();
    session.close().unwrap();
    assert!(started.elapsed() < Duration::from_secs(2));
}
#[tokio::test]
async fn async_view_run_and_dropping_a_pacing_future_leave_session_usable() {
    let mut view = http_view();
    view.step_hz = Some(0.1);
    let mut env = ViewedSteps(Steps::new());
    let model = ones();
    let mut session = model
        .session(
            &mut env,
            SessionOptions {
                view: Some(view),
                ..Default::default()
            },
        )
        .unwrap();
    session.reset_async(None, None).await.unwrap();
    session.step_async(Value::Int(1)).await.unwrap();
    let mut step = Box::pin(session.step_async(Value::Int(1)));
    assert!(futures::poll!(&mut step).is_pending());
    drop(step);
    session.reset_async(None, None).await.unwrap();
    session.close_async().await.unwrap();
}
