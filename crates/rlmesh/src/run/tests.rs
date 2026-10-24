//! Runs, bindings and hooks driven end to end against local and served envs.

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
        address: Some(crate::tests::ENV_ADDRESS.into()),
        options: Some(crate::tests::env_serve_options()),
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
            execution_horizon: Some(0),
            ..Default::default()
        }),
        "execution_horizon must be >= 1, got 0"
    );
    assert_eq!(
        refused(RunOptions {
            execution_horizon: Some(1025),
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
fn random_sample_runs_on_the_native_loop() {
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
    assert_eq!(RANDOM_SAMPLE.to_string(), "RANDOM_SAMPLE");
}

#[test]
fn a_binding_steps_a_local_env_on_this_thread_with_the_model_or_a_given_action() {
    struct OnThread(Steps, std::thread::ThreadId);
    impl Env for OnThread {
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
            assert_eq!(std::thread::current().id(), self.1);
            self.0.reset(seed, options)
        }
        fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
            assert_eq!(std::thread::current().id(), self.1);
            self.0.step(action)
        }
        fn close(&mut self) -> Result<()> {
            self.0.close()
        }
    }
    let model = ones();
    let mut env = OnThread(Steps::new(), std::thread::current().id());
    let resets = Arc::clone(&env.0.resets);
    let closes = Arc::clone(&env.0.closes);
    let mut binding = model.bind(&mut env, BindOptions::default()).unwrap();
    assert!(binding.done());
    assert_eq!(binding.num_envs(), 1);
    let mut options = InfoDict::new();
    options.insert("level", Value::Int(3));
    let (observation, _) = binding.reset(Some(4), Some(&options)).unwrap();
    assert_eq!(observation, Value::Int(0));
    assert!(!binding.done());
    // The model's action (1), then a given one (0).
    assert_eq!(binding.step(None).unwrap().1, 1.0);
    assert_eq!(binding.step(Some(Value::Int(0))).unwrap().1, 0.0);
    let (observation, _, terminated, _, info) = binding.step(None).unwrap();
    assert_eq!(observation, Value::Int(3));
    assert!(terminated && binding.done());
    assert_eq!(info.get("is_success"), Some(&Value::Bool(true)));
    let error = binding.step(None).unwrap_err();
    assert_eq!(error.class(), ErrorClass::RuntimeError);
    binding.close().unwrap();
    binding.close().unwrap();
    assert_eq!(
        binding.reset(None, None).unwrap_err().class(),
        ErrorClass::RuntimeError
    );
    drop(binding);
    assert_eq!(*resets.lock().unwrap(), vec![(Some(4), Some(options))]);
    // Closing the binding closes neither the borrowed env nor the model.
    assert_eq!(closes.load(Ordering::SeqCst), 0);
    model.run(&mut env, RunOptions::default()).unwrap();
}

#[test]
fn a_policy_that_writes_reset_options_adds_them_to_each_reset_it_starts() {
    struct Levelled;
    impl crate::Policy for Levelled {
        const RESET_OPTIONS: bool = true;
        fn predict(&mut self, _observation: Value) -> Result<Value> {
            Ok(Value::Int(1))
        }
        fn reset_options(&mut self, episode: &Metadata) -> Result<Option<Metadata>> {
            let mut options = Metadata::new();
            options.insert("level", episode.get("seed").cloned().unwrap_or(Value::None));
            assert!(matches!(episode.get("episode_id"), Some(Value::Text(_))));
            Ok(Some(options))
        }
    }
    let model = Model::new(Levelled, ModelOptions::default()).unwrap();
    let mut env = Steps::new();
    let resets = Arc::clone(&env.resets);
    model
        .run(
            &mut env,
            RunOptions {
                seeds: Some(vec![5, 6]),
                ..Default::default()
            },
        )
        .unwrap();
    let levels: Vec<_> = resets
        .lock()
        .unwrap()
        .iter()
        .map(|(_, options)| options.as_ref().and_then(|o| o.get("level").cloned()))
        .collect();
    assert_eq!(levels, vec![Some(Value::Int(5)), Some(Value::Int(6))]);
}

#[test]
fn a_bindings_hooks_see_its_episodes_a_reset_over_one_as_truncated_and_the_run_end_at_close() {
    let model = ones();
    let mut env = Steps::new();
    let mut log = Log::default();
    {
        let mut binding = model
            .bind(
                &mut env,
                BindOptions {
                    hooks: Some(&mut log),
                    ..Default::default()
                },
            )
            .unwrap();
        binding.reset(Some(10), None).unwrap();
        for _ in 0..3 {
            binding.step(None).unwrap();
        }
        // Reset over the second episode after one step: it ends truncated.
        binding.reset(Some(11), None).unwrap();
        binding.step(Some(Value::Int(0))).unwrap();
        binding.reset(None, None).unwrap();
        binding.close().unwrap();
    }
    let context = log.context.take().unwrap();
    assert_eq!(
        context.num_envs().unwrap_err().class(),
        ErrorClass::RuntimeError,
        "the binding's context fails once it closes"
    );
    assert_eq!(
        log.calls,
        vec![
            "run_start contract=true num_envs=1",
            "episode_start 0 Some(10)",
            "step 0 0 obs=0 action=1 reward=1",
            "step 0 1 obs=1 action=1 reward=1",
            "step 0 2 obs=2 action=1 reward=1",
            "episode_end 0 steps=3 reward=3",
            "episode_start 1 Some(11)",
            "step 1 0 obs=0 action=0 reward=0",
            "episode_end 1 steps=1 reward=0",
            "episode_start 2 None",
            "episode_end 2 steps=0 reward=0",
            "run_end episodes=3",
        ]
    );
}

#[test]
fn a_binding_closes_a_factory_built_env_and_a_local_env_only_under_close_env() {
    let model = ones();
    let mut factory = Factory {
        closes: Arc::default(),
        made: 0,
    };
    let closes = Arc::clone(&factory.closes);
    let mut binding = model
        .bind(EnvTarget::factory(&mut factory), BindOptions::default())
        .unwrap();
    binding.reset(None, None).unwrap();
    binding.close().unwrap();
    assert_eq!(closes.load(Ordering::SeqCst), 1);
    let mut env = Steps::new();
    let closes = Arc::clone(&env.closes);
    let options = BindOptions {
        close_env: true,
        ..Default::default()
    };
    drop(model.bind(&mut env, options).unwrap());
    assert_eq!(closes.load(Ordering::SeqCst), 1);
}

#[test]
fn a_binding_reaches_a_served_env_and_refuses_what_it_cannot_drive() {
    let model = ones();
    let server = steps_server();
    let mut binding = model.bind(&server, BindOptions::default()).unwrap();
    binding.reset(Some(1), None).unwrap();
    assert_eq!(binding.step(None).unwrap().0, Value::Int(1));
    binding.close().unwrap();
    // A v1 endpoint, and a served model at one.
    let error = model
        .bind("127.0.0.1:1", BindOptions::default())
        .unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
    let v1 = crate::RemoteModel::new(crate::RemoteModelOptions::default().address("127.0.0.1:1"))
        .unwrap();
    let error = bind(&v1, &server, BindOptions::default()).unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
    server.shutdown().unwrap();
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

fn steps_server() -> EnvServer {
    let server = EnvServer::new(Steps::new(), quick()).unwrap();
    server.start().unwrap();
    server
}

#[cfg(feature = "lockstep")]
#[tokio::test]
async fn a_v2_handle_connects_and_drops_inside_a_current_thread_runtime() {
    let server = crate::EnvServer::new(Steps::new(), quick()).unwrap();
    server.start().unwrap();
    let mut remote =
        crate::RemoteEnv::new_async(crate::RemoteEnvOptions::default().address(server.address()))
            .await
            .unwrap();
    remote.reset_async(None, None).await.unwrap();
    drop(remote);
    server.shutdown_async().await.unwrap();
}

#[cfg(not(feature = "lockstep"))]
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

#[cfg(feature = "lockstep")]
#[test]
fn a_native_run_through_an_open_v2_handle_says_the_handle_holds_the_env() {
    let server = steps_server();
    let mut remote =
        crate::RemoteEnv::new(crate::RemoteEnvOptions::default().address(server.address()))
            .unwrap();
    // A v2 handle holds its env's lease from connect: the run refuses it
    // with Python's upgrade error.
    let err = ones()
        .run(
            &mut remote,
            RunOptions {
                episodes: Some(1),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::TypeError, "{err}");
    assert!(
        err.to_string()
            .starts_with("run(RemoteEnv(address)) is gone in rlmesh 0.2"),
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
/// A chunked spec'd policy recording the horizon each chunk is asked at.
struct Paced {
    seen: Arc<Mutex<Vec<u32>>>,
    loaded: Option<u32>,
}
impl crate::Policy for Paced {
    const CORNERS: crate::Corners = crate::Corners::PREDICT.union(crate::Corners::PREDICT_CHUNK);
    fn execution_horizon(&self) -> Option<u32> {
        self.loaded
    }
    fn predict(&mut self, _observation: Value) -> Result<Value> {
        // The runtime takes the single-action corner at horizon 1.
        self.seen.lock().unwrap().push(1);
        Ok(Value::List(vec![Value::Float(0.)]))
    }
    fn predict_chunk(&mut self, _observation: Value, execution_horizon: u32) -> Result<Value> {
        self.seen.lock().unwrap().push(execution_horizon);
        Ok(Value::List(vec![
            Value::List(vec![Value::Float(0.)]);
            execution_horizon as usize
        ]))
    }
}
fn paced(loaded: Option<u32>, timing: Option<u32>) -> (Model, Arc<Mutex<Vec<u32>>>) {
    use crate::adapters::*;
    let seen = Arc::default();
    let spec = ModelSpec::new(State::new(GRIPPER_POS), adapter_action())
        .unwrap()
        .with_timing(Timing { execution: timing })
        .unwrap();
    let model = Model::new(
        Paced {
            seen: Arc::clone(&seen),
            loaded,
        },
        ModelOptions {
            spec: Some(spec.into()),
            ..Default::default()
        },
    )
    .unwrap();
    (model, seen)
}

#[test]
fn a_run_naming_no_horizon_executes_at_the_loaded_one_then_the_spec_timing_then_one() {
    for (loaded, timing, explicit, expected) in [
        (Some(3), Some(2), None, 3),
        (None, Some(2), None, 2),
        (None, None, None, 1),
        (Some(3), Some(2), Some(1), 1),
    ] {
        let (model, seen) = paced(loaded, timing);
        let mut env = adapter_steps();
        model
            .run(
                &mut env,
                RunOptions {
                    execution_horizon: explicit,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            *seen.lock().unwrap(),
            vec![expected],
            "{loaded:?} {timing:?} {explicit:?}"
        );
    }
    let (model, seen) = paced(Some(3), None);
    let mut env = adapter_steps();
    let mut binding = model.bind(&mut env, BindOptions::default()).unwrap();
    binding.reset(None, None).unwrap();
    binding.step(None).unwrap();
    drop(binding);
    assert_eq!(*seen.lock().unwrap(), vec![3]);
    let envelope = crate::describe_json(&model, Default::default()).unwrap();
    assert!(envelope.contains(r#""execution_horizon":3"#), "{envelope}");
}

#[test]
fn a_spec_timing_outside_one_to_1024_is_refused() {
    use crate::adapters::*;
    let spec = ModelSpec::new(State::new(GRIPPER_POS), adapter_action()).unwrap();
    for bad in [0, 1025] {
        let error = spec
            .clone()
            .with_timing(Timing {
                execution: Some(bad),
            })
            .unwrap_err();
        assert_eq!(error.class(), ErrorClass::ValueError);
    }
    let timed = spec.with_timing(Timing { execution: Some(8) }).unwrap();
    let back = ModelSpec::from_json(&timed.to_json().unwrap()).unwrap();
    assert_eq!(back.timing(), timed.timing());
    // State and its one-part Concat have the same canonical wire form.
    assert_eq!(back.to_json().unwrap(), timed.to_json().unwrap());
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
    drop(closure.bind(&mut env, BindOptions::default()).unwrap());
    closure.run(&mut env, RunOptions::default()).unwrap();
    assert_eq!(warnings.load(Ordering::SeqCst), 0);
    struct Authored;
    impl crate::Policy for Authored {
        fn predict(&mut self, _: Value) -> Result<Value> {
            Ok(Value::Int(0))
        }
    }
    let authored = Model::new(Authored, ModelOptions::default()).unwrap();
    drop(authored.bind(&mut env, BindOptions::default()).unwrap());
    crate::set_warning_handler(None);
    assert_eq!(warnings.load(Ordering::SeqCst), 1);
}

#[test]
fn plaintext_api_keys_warn_once_per_address() {
    let _lock = crate::warnings::tests::HANDLER_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let record = Arc::clone(&seen);
    crate::warnings::set_warning_handler(Some(Box::new(move |w: &crate::Warning| {
        record.lock().unwrap().push(w.message.clone());
        Ok(())
    })));
    let address = "10.255.255.7:5555";
    for _ in 0..2 {
        crate::RemoteModel::new(
            crate::RemoteModelOptions::default()
                .address(address)
                .api_key("some-key"),
        )
        .unwrap();
    }
    crate::warnings::set_warning_handler(None);
    assert_eq!(
        seen.lock().unwrap().as_slice(),
        [
            "api_key is sent unencrypted to tcp://10.255.255.7:5555; use an https:// address to \
          keep it private"
        ]
    );
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
fn run_completion_observes_a_token_fired_by_the_final_hook() {
    struct Fire(crate::ext::CancelToken);
    impl RunHooks for Fire {
        fn on_run_end(&mut self, _: &RunResult) -> Result<()> {
            self.0.cancel();
            Ok(())
        }
    }
    let model = ones();
    let token = crate::ext::CancelToken::new();
    let mut hook = Fire(token.clone());
    let mut env = Steps::new();
    assert_eq!(
        model
            .run(
                &mut env,
                RunOptions {
                    cancel: Some(token.clone()),
                    hooks: Some(&mut hook),
                    ..Default::default()
                }
            )
            .unwrap_err(),
        crate::Error::Cancelled
    );
}

#[test]
fn run_hook_error_wins_over_cancellation_from_on_run_end() {
    struct FailThenFire(crate::ext::CancelToken);
    impl RunHooks for FailThenFire {
        fn on_episode_end(&mut self, _: &EpisodeResult) -> Result<()> {
            Err(Error::invalid_argument("first hook error"))
        }
        fn on_run_end(&mut self, _: &RunResult) -> Result<()> {
            self.0.cancel();
            Ok(())
        }
    }
    let model = ones();
    let mut env = Steps::new();
    let token = crate::ext::CancelToken::new();
    let mut hooks = FailThenFire(token.clone());
    let result = model.run(
        &mut env,
        RunOptions {
            cancel: Some(token),
            hooks: Some(&mut hooks),
            ..Default::default()
        },
    );
    assert_eq!(
        result.unwrap_err(),
        Error::invalid_argument("first hook error")
    );
}
