//! Model runs end to end: hooks, options, one run at a time, cancellation,
//! factory targets, and how results print.

use super::*;

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

/// One model served in process answers four routes at once. The route whose
/// env dies mid-episode fails alone; its siblings finish, and their trials
/// and seeds are the ones a single route over the whole range would draw.
#[test]
fn a_failed_route_leaves_its_siblings_running_against_a_shared_served_model() {
    use crate::binding::routes::{Route, merge_results, run_routes};
    use crate::lockstep::{ModelBinding, ModelService};

    let runtime = drive::runtime();
    let binding = runtime.block_on(ModelBinding::bind(":0/random")).unwrap();
    let address = binding.address();
    let stop = tokio_util::sync::CancellationToken::new();
    let service = ModelService::new(
        crate::model::random::RandomSampleHandler::default(),
        String::new(),
    );
    let served =
        runtime.spawn(binding.serve(Arc::new(service), Duration::from_secs(5), stop.clone()));
    let model =
        crate::RemoteModel::new(crate::RemoteModelOptions::default().address(address)).unwrap();
    let servers: Vec<EnvServer> = (0..4)
        .map(|route| {
            let mut env = Steps::new();
            env.fail_at = (route == 2).then_some(2);
            let server = EnvServer::new(env, quick()).unwrap();
            server.start().unwrap();
            server
        })
        .collect();
    let routes = |base_seed| {
        servers
            .iter()
            .enumerate()
            .map(|(route, server)| Route {
                base_seed,
                ..Route::new(
                    server,
                    RunOptions {
                        episodes: Some(3),
                        trial_index_base: 3 * route as u64,
                        ..Default::default()
                    },
                )
            })
            .collect::<Vec<_>>()
    };

    let results = run_routes(&model, routes(Some(7))).unwrap();

    let verdicts: Vec<bool> = results.iter().map(Result::is_ok).collect();
    assert_eq!(verdicts, [true, true, false, true]);
    let failure = results[2].as_ref().unwrap_err().to_string();
    assert!(failure.contains("env gone"), "{failure}");
    let merged = merge_results(results.iter().flatten());
    let trials: Vec<Option<i64>> = merged.episodes.iter().map(|e| e.trial).collect();
    assert_eq!(trials, [0, 1, 2, 3, 4, 5, 9, 10, 11].map(Some));
    let indices: Vec<i64> = merged.episodes.iter().map(|e| e.index).collect();
    assert_eq!(indices, (0..9).collect::<Vec<_>>());
    let contract_id = servers[0].env_contract().id().to_string();
    for episode in &merged.episodes {
        let slot = episode.trial.unwrap() as u64;
        assert_eq!(
            episode.seed,
            Some(crate::run::runtime::episode_seed(7, &contract_id, slot))
        );
    }

    // An in-process model drives one route at a time.
    let error = run_routes(&ones(), routes(None)).unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError, "{error}");

    for server in servers {
        server.shutdown().unwrap();
    }
    stop.cancel();
    runtime.block_on(served).unwrap().unwrap();
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
    // An unknown outcome beside a reported one counts as a failure, as the
    // report and `rlmesh eval report` count it.
    assert_eq!(crate::run::success_rate_of([Some(true), None]), Some(0.5));
    assert_eq!(crate::run::success_rate_of([]), None);
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
