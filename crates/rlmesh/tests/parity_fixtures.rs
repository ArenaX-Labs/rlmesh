//! The shared TOML scenarios, projected losslessly by tests/parity/support.py.

// Fixture setup and synchronization panic on unexpected state.
#![allow(clippy::unwrap_used)]
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use rlmesh::spaces::{Discrete, DiscreteOptions, Space};
use rlmesh::types::LoadKey;
use rlmesh::*;
use serde_json::{Value as Json, json};

#[derive(Default)]
struct Counts {
    closes: AtomicUsize,
    model_closes: AtomicUsize,
    ends: AtomicUsize,
    env_starts: AtomicUsize,
    model_starts: AtomicUsize,
    reset: Mutex<(Option<i64>, Option<i64>)>,
    cleanup: (Mutex<()>, Condvar),
}
struct ScriptedEnv {
    counts: Arc<Counts>,
    steps: i64,
    cancel: Option<ext::CancelToken>,
    dict_action: bool,
}
impl ScriptedEnv {
    fn new(counts: &Arc<Counts>) -> Self {
        Self {
            counts: counts.clone(),
            steps: 0,
            cancel: None,
            dict_action: false,
        }
    }
}
impl Env for ScriptedEnv {
    fn metadata(&self) -> Option<Metadata> {
        Some(
            [(
                ENV_RESET_OPTIONS_KEY,
                Value::List(vec![Value::Text("trial_index".into())]),
            )]
            .into_iter()
            .collect(),
        )
    }
    fn observation_space(&self) -> Space {
        Discrete::new(3, DiscreteOptions::default()).unwrap().into()
    }
    fn action_space(&self) -> Space {
        if self.dict_action {
            let child =
                spaces::Dict::new([("k", self.observation_space().spec().clone())]).unwrap();
            return spaces::Dict::new([("path", child.spec().clone())])
                .unwrap()
                .into();
        }
        self.observation_space()
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        *self.counts.reset.lock().unwrap() = (seed, trial_index(options, None)?);
        self.counts.env_starts.fetch_add(1, Ordering::SeqCst);
        self.steps = 0;
        Ok((Value::Int(0), InfoDict::new()))
    }
    fn step(&mut self, _: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.counts.env_starts.fetch_add(1, Ordering::SeqCst);
        self.steps += 1;
        if let Some(cancel) = self.cancel.take() {
            cancel.cancel();
        }
        Ok((
            Value::Int(self.steps),
            1.0,
            self.steps >= 2,
            false,
            InfoDict::new(),
        ))
    }
    fn close(&mut self) -> Result<()> {
        let _guard = self.counts.cleanup.0.lock().unwrap();
        self.counts.closes.fetch_add(1, Ordering::SeqCst);
        self.counts.cleanup.1.notify_all();
        Ok(())
    }
}
struct ScriptedPolicy {
    counts: Arc<Counts>,
    failure: Arc<Mutex<Option<Error>>>,
}
impl Policy for ScriptedPolicy {
    fn predict(&mut self, _: Value) -> Result<Value> {
        self.counts.model_starts.fetch_add(1, Ordering::SeqCst);
        if let Some(error) = self.failure.lock().unwrap().clone() {
            return Err(error);
        }
        Ok(Value::Int(0))
    }
    fn on_episode_end(&mut self, _: &str) -> Result<()> {
        self.counts.ends.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        self.counts.model_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
fn model(counts: &Arc<Counts>) -> Model {
    Model::new(
        ScriptedPolicy {
            counts: counts.clone(),
            failure: Arc::default(),
        },
        ModelOptions::default(),
    )
    .unwrap()
}
fn error_record(error: Error) -> Json {
    let code = match &error {
        Error::Environment(e) => e.code.as_str(),
        Error::Model(e) => e.code.as_str(),
        _ => "",
    };
    json!({"error": error.class().name(), "text": error.to_string(), "code": code,
        "recoverable": error.is_recoverable(),
        "runtime_base": error.class().is_subclass_of(ErrorClass::RuntimeError),
        "os_base": error.class().is_subclass_of(ErrorClass::OSError),
        "value_base": error.class().is_subclass_of(ErrorClass::ValueError),
        "rlmesh_base": error.class().is_subclass_of(ErrorClass::RLMeshException),
        "environment_base": error.class().is_subclass_of(ErrorClass::EnvironmentException),
        "self_base": error.class().is_subclass_of(error.class())})
}
fn scripted_error(name: &str) -> Error {
    let text = "scripted failure";
    match name {
        "RLMeshException" => Error::Internal(text.into()),
        "ProtocolException" => Error::Protocol(text.into()),
        "EnvironmentException" => Error::environment(text),
        "RecoverableEnvironmentException" => Error::environment_recoverable(text),
        "ModelException" => Error::model(text),
        "ModelRecoverable" => Error::model_recoverable(text),
        "ValueError" | "ParamError" => Error::invalid_argument(text),
        "AdapterResolutionError" => Error::InvalidArgument(InvalidArgumentError {
            message: text.into(),
            adapter_resolution: Some(adapters::AdapterResolutionError::new()),
        }),
        "TypeError" => Error::Type(text.into()),
        "RuntimeError" | "NotImplementedError" => Error::Runtime(text.into()),
        "OSError" | "FileExistsError" | "PermissionError" => Error::Os(text.into()),
        "ConnectionError" => Error::Connection(text.into()),
        "TimeoutError" => binding::AttachedError::with_message(
            Error::Timeout(Duration::from_secs(5)),
            "authored timeout: checkpoint shard violet".into(),
        ),
        "KeyboardInterrupt" => Error::Cancelled,
        _ => panic!("unknown error {name}"),
    }
}
#[derive(Default)]
struct Hooks {
    ends: usize,
    partial: usize,
    fail: bool,
}
impl RunHooks for Hooks {
    fn on_step(&mut self, _: &StepEvent) -> Result<()> {
        if self.fail {
            Err(Error::Type("hook abort".into()))
        } else {
            Ok(())
        }
    }
    fn on_run_end(&mut self, result: &RunResult) -> Result<()> {
        self.ends += 1;
        self.partial = result.num_episodes();
        Ok(())
    }
}
fn count(value: &AtomicUsize) -> usize {
    value.load(Ordering::SeqCst)
}
fn execute(case: &Json) -> Json {
    let counts = Arc::new(Counts::default());
    let mut env = ScriptedEnv::new(&counts);
    let model = model(&counts);
    let op = case["operation"].as_str().unwrap();
    match op {
        "dict_error" => {
            env.dict_action = true;
            let server = EnvServer::new(
                env,
                EnvServerOptions {
                    address: Some("tcp://127.0.0.1:0".into()),
                    ..Default::default()
                },
            )
            .unwrap();
            server.start().unwrap();
            let mut remote =
                RemoteEnv::new(RemoteEnvOptions::default().address(server.address())).unwrap();
            remote.reset(Some(0), None).unwrap();
            let action = Value::Dict(
                [("path", Value::Dict(Metadata::new()))]
                    .into_iter()
                    .collect(),
            );
            let out = error_record(remote.step(action).unwrap_err());
            remote.close().unwrap();
            server.shutdown().unwrap();
            out
        }
        "reset_order" => {
            let mut session = model.session(&mut env, SessionOptions::default()).unwrap();
            let (observation, info) = session.reset(Some(37), Some(11)).unwrap();
            let (seed, trial) = *counts.reset.lock().unwrap();
            json!({"seed": seed, "trial_index": trial, "observation": observation.as_int().unwrap(), "info_empty": info.is_empty()})
        }
        "config_error" => {
            #[derive(Default)]
            struct Keyed;
            impl Policy for Keyed {
                const LOAD_KEYS: Option<&'static [LoadKey]> =
                    Some(&[LoadKey::required("checkpoint")]);
                fn load(&mut self, _: &Kwargs) -> Result<()> {
                    panic!("invalid config reached load")
                }
            }
            let mut config = Kwargs::new();
            if case["variant"] == "unknown" {
                config.insert("checkpoint", Value::Int(1));
                config.insert("scripted", Value::Int(2));
            }
            error_record(Model::from_config::<Keyed>(config).unwrap_err())
        }
        "validation" => {
            let variant = case["variant"].as_str().unwrap();
            let error = if variant.starts_with("horizon") {
                model
                    .run(
                        &mut env,
                        RunOptions {
                            execution_horizon: if variant == "horizon_zero" { 0 } else { 1025 },
                            ..Default::default()
                        },
                    )
                    .unwrap_err()
            } else {
                let mut view = View::default();
                match variant {
                    "view_port" => view.port = 0,
                    "view_fps" => view.fps = 0,
                    "view_quality" => view.quality = 0,
                    "view_step_hz" => view.step_hz = Some(0.0),
                    _ => panic!("unknown validation"),
                }
                view.validate().unwrap_err()
            };
            error_record(error)
        }
        "run" => {
            let close_env = case["close_env"].as_bool().unwrap_or(false);
            let result = model.run(
                &mut env,
                RunOptions {
                    episodes: case["episodes"].as_u64(),
                    seeds: case.get("seeds").map(|seeds| {
                        seeds
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|s| s.as_i64().unwrap())
                            .collect()
                    }),
                    close_env,
                    ..Default::default()
                },
            );
            match result {
                Err(error) => error_record(error),
                Ok(result) => {
                    let mut observed = json!({"episodes": result.num_episodes(), "steps": result.total_steps(),
                        "reward": result.mean_reward(), "env_closes": count(&counts.closes),
                        "model_closes": count(&counts.model_closes)});
                    model
                        .run(&mut ScriptedEnv::new(&counts), RunOptions::default())
                        .unwrap();
                    observed["reusable"] = json!(true);
                    observed
                }
            }
        }
        "busy_session" => {
            let server = EnvServer::new(
                env,
                EnvServerOptions {
                    address: Some("tcp://127.0.0.1:0".into()),
                    ..Default::default()
                },
            )
            .unwrap();
            server.start().unwrap();
            let mut first = model.session(&server, SessionOptions::default()).unwrap();
            first.reset(Some(0), None).unwrap();
            let error = match model.session(&server, SessionOptions::default()) {
                Err(error) => error,
                Ok(mut second) => second.reset(Some(0), None).unwrap_err(),
            };
            let mut out = error_record(error);
            first.step(Value::Int(0)).unwrap();
            first.close().unwrap();
            out["reusable"] = json!(true);
            drop(first);
            server.shutdown().unwrap();
            out
        }
        "remote_auth" => remote_auth(case["variant"].as_str().unwrap(), &mut env, &model),
        "busy_remote" => {
            let server = EnvServer::new(
                env,
                EnvServerOptions {
                    address: Some("tcp://127.0.0.1:0".into()),
                    ..Default::default()
                },
            )
            .unwrap();
            server.start().unwrap();
            let mut remote =
                RemoteEnv::new(RemoteEnvOptions::default().address(server.address())).unwrap();
            remote.reset(Some(0), None).unwrap();
            remote.step(Value::Int(0)).unwrap();
            let mut out = error_record(model.run(&mut remote, RunOptions::default()).unwrap_err());
            remote.step(Value::Int(0)).unwrap();
            out["reusable"] = json!(true);
            remote.close().unwrap();
            server.shutdown().unwrap();
            out
        }
        "borrow" => {
            let mut session = model.session(&mut env, SessionOptions::default()).unwrap();
            session.reset(Some(0), None).unwrap();
            drop(session);
            let mut out = json!({"env_closes": count(&counts.closes), "model_closes": count(&counts.model_closes), "episode_ends": count(&counts.ends)});
            model.run(&mut env, RunOptions::default()).unwrap();
            out["reusable"] = json!(true);
            out
        }
        "cancel_idle" | "cancel_step" | "cancel_run" | "hook_error" => {
            let token = ext::CancelToken::new();
            if op == "cancel_step" || op == "cancel_run" {
                env.cancel = Some(token.clone());
            }
            let mut session = model
                .session(
                    &mut env,
                    SessionOptions {
                        cancel: Some(token.clone()),
                        ..Default::default()
                    },
                )
                .unwrap();
            let obs = if op == "cancel_run" || op == "hook_error" {
                Value::Int(0)
            } else {
                session.reset(Some(0), None).unwrap().0
            };
            if op == "cancel_idle" {
                let mut out = Json::Null;
                let mut cancelled_call_starts = 0;
                let mut failures = 0;
                for _ in 0..2 {
                    let before = count(&counts.model_starts);
                    token.cancel();
                    token.cancel();
                    out = error_record(session.predict(obs.clone()).unwrap_err());
                    failures += 1;
                    cancelled_call_starts += count(&counts.model_starts) - before;
                    session.predict(obs.clone()).unwrap();
                }
                out["cancelled_call_starts"] = json!(cancelled_call_starts);
                out["failures"] = json!(failures);
                out["reusable"] = json!(true);
                return out;
            }
            let mut hooks = Hooks {
                fail: op == "hook_error",
                ..Default::default()
            };
            let error = if op == "cancel_step" {
                session.step(Value::Int(0)).unwrap_err()
            } else {
                session
                    .run(SessionRunOptions {
                        hooks: Some(&mut hooks),
                        ..Default::default()
                    })
                    .unwrap_err()
            };
            let mut out = error_record(error);
            out["episode_ends_before_reset"] = json!(count(&counts.ends));
            if op == "cancel_step" {
                session.reset(Some(0), None).unwrap();
            }
            out["episode_ends"] = json!(count(&counts.ends));
            out["run_ends"] = json!(hooks.ends);
            out["partial_episodes"] = json!(hooks.partial);
            session.step(Value::Int(0)).unwrap();
            out["reusable"] = json!(true);
            out
        }
        "error" => {
            let failure = scripted_error(case["variant"].as_str().unwrap());
            let model = Model::new(
                move |_: Value| Err(failure.clone()),
                ModelOptions::default(),
            )
            .unwrap();
            let mut session = model.session(&mut env, SessionOptions::default()).unwrap();
            let (obs, _) = session.reset(Some(0), None).unwrap();
            error_record(session.predict(obs).unwrap_err())
        }
        "warning" => warning(case["variant"].as_str().unwrap()),
        "factory_cancel" => drop_serve(if case["variant"] == "serving" {
            "factory_cancel"
        } else {
            "factory_idle"
        }),
        "drop_queued_predict" => drop_queued_predict(),
        "drop_owned_cleanup" => drop_owned_cleanup(),
        "model_callback" => {
            model_callback(case.get("variant").and_then(Json::as_str) == Some("timeout"))
        }
        "serve_drop_nonblocking" => serve_drop_nonblocking(),
        "drop_run" => drop_run(),
        "context" => context(case["variant"].as_str().unwrap()),
        "drop_session" => drop_session(case["variant"].as_str().unwrap().to_owned()),
        "drop_serve" => drop_serve(case["variant"].as_str().unwrap()),
        other => panic!("unknown operation {other}"),
    }
}

#[test]
fn shared_behavioural_fixtures() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new("python")
        .arg(root.join("tests/parity/support.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Json> = serde_json::from_slice(&output.stdout).unwrap();
    for case in cases {
        if case
            .get("languages")
            .is_some_and(|languages| !languages.as_array().unwrap().iter().any(|l| l == "rust"))
        {
            continue;
        }
        let observed = execute(&case);
        eprintln!("{}: {observed}", case["id"]);
        for (key, expected) in case["expect"].as_object().unwrap() {
            assert_eq!(
                &observed[key], expected,
                "scenario {} field {key}: {observed}",
                case["id"]
            );
        }
        if case["expect"].get("error").is_none() {
            assert!(observed.get("error").is_none(), "{observed}");
        }
    }
}

fn context(variant: &str) -> Json {
    fn call() -> Result<RunResult> {
        let counts = Arc::default();
        model(&counts).run(&mut ScriptedEnv::new(&counts), RunOptions::default())
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let result = match variant {
        "task" => runtime.block_on(async { tokio::spawn(async { call() }).await.unwrap() }),
        "block_in_place" => runtime.block_on(async {
            tokio::spawn(async { tokio::task::block_in_place(call) })
                .await
                .unwrap()
        }),
        "spawn_blocking" => {
            runtime.block_on(async { tokio::task::spawn_blocking(call).await.unwrap() })
        }
        "thread" => std::thread::spawn(call).join().unwrap(),
        "entered_thread" => {
            let handle = runtime.handle().clone();
            std::thread::spawn(move || {
                let _guard = handle.enter();
                call()
            })
            .join()
            .unwrap()
        }
        "root" => runtime.block_on(async { call() }),
        "sync_hook" | "async_hook" => {
            struct Nested;
            impl RunHooks for Nested {
                fn on_step(&mut self, _: &StepEvent) -> Result<()> {
                    call().map(|_| ())
                }
            }
            let counts = Arc::default();
            let model = model(&counts);
            let mut env = ScriptedEnv::new(&counts);
            let mut hook = Nested;
            let options = RunOptions {
                hooks: Some(&mut hook),
                ..Default::default()
            };
            if variant == "sync_hook" {
                model.run(&mut env, options)
            } else {
                futures::executor::block_on(model.run_async(&mut env, options))
            }
        }
        _ => panic!("unknown context {variant}"),
    };
    match result {
        Ok(result) => json!({"episodes": result.num_episodes()}),
        Err(error) => error_record(error),
    }
}
fn drop_session(variant: String) -> Json {
    let counts = Arc::new(Counts::default());
    let observed = counts.clone();
    let inside = variant != "outside";
    let blocking = variant == "spawn_blocking";
    let work = move || {
        let model = model(&counts);
        let mut env = ScriptedEnv::new(&counts);
        if variant == "outside" {
            let mut session = model
                .session(
                    &mut env,
                    SessionOptions {
                        close_env: true,
                        ..Default::default()
                    },
                )
                .unwrap();
            session.reset(Some(0), None).unwrap();
            drop(session);
        } else {
            futures::executor::block_on(async {
                let mut session = model
                    .session_async(
                        &mut env,
                        SessionOptions {
                            close_env: true,
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
                session.reset_async(Some(0), None).await.unwrap();
                drop(session);
            });
        }
        // Explicit close is a FIFO fence for Session Drop's owned episode-end call.
        futures::executor::block_on(model.close_async()).unwrap();
    };
    if !inside {
        work();
    } else {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                if blocking {
                    tokio::task::spawn_blocking(work).await.unwrap();
                } else {
                    tokio::spawn(async move {
                        work();
                    })
                    .await
                    .unwrap();
                }
            });
    }
    json!({"env_closes": count(&observed.closes), "episode_ends": count(&observed.ends)})
}
struct Factory {
    counts: Arc<Counts>,
    prepares: usize,
    cancel_in_make: Option<ext::CancelToken>,
}
impl EnvFactory for Factory {
    fn prepare(&mut self) -> Result<()> {
        self.prepares += 1;
        Ok(())
    }
    fn make(&mut self, _: &Kwargs) -> Result<MadeEnv> {
        if let Some(token) = self.cancel_in_make.take() {
            token.cancel();
        }
        Ok(MadeEnv::env(ScriptedEnv::new(&self.counts)))
    }
}
fn drop_serve(variant: &str) -> Json {
    let counts = Arc::new(Counts::default());
    if variant == "factory_cancel" || variant == "factory_idle" || variant == "factory_drop" {
        let token = ext::CancelToken::new();
        let mut factory = Factory {
            counts: counts.clone(),
            prepares: 0,
            cancel_in_make: (variant == "factory_cancel").then(|| token.clone()),
        };
        if variant == "factory_idle" {
            token.cancel();
        }
        let mut out = if variant == "factory_drop" {
            futures::executor::block_on(async {
                let mut serving = Box::pin(
                    factory.serve_async("tcp://127.0.0.1:0", EnvFactoryServeOptions::default()),
                );
                assert!(futures::poll!(serving.as_mut()).is_pending());
                drop(serving);
            });
            json!({})
        } else {
            error_record(
                futures::executor::block_on(factory.serve_async(
                    "tcp://127.0.0.1:0",
                    EnvFactoryServeOptions {
                        cancel: Some(token),
                        ..Default::default()
                    },
                ))
                .unwrap_err(),
            )
        };
        // Closing happens detached; use a bounded condition variable rather than sleeps.
        if variant != "factory_idle" {
            wait_closed(&counts, 1);
        }
        out["env_closes"] = json!(count(&counts.closes));
        out["prepares"] = json!(factory.prepares);
        return out;
    }
    let server = EnvServer::new(
        ScriptedEnv::new(&counts),
        EnvServerOptions {
            address: Some("tcp://127.0.0.1:0".into()),
            ..Default::default()
        },
    )
    .unwrap();
    if variant == "wait" {
        server.start().unwrap();
        futures::executor::block_on(async {
            let mut waiting = Box::pin(server.wait_async(None));
            assert!(futures::poll!(waiting.as_mut()).is_pending());
            drop(waiting);
        });
        let running = !server.wait(Some(Duration::ZERO)).unwrap();
        server.shutdown().unwrap();
        json!({"running": running, "env_closes": count(&counts.closes)})
    } else {
        futures::executor::block_on(async {
            let mut serving = Box::pin(server.serve_async());
            assert!(futures::poll!(serving.as_mut()).is_pending());
            drop(serving);
        });
        assert!(server.wait(Some(Duration::from_secs(10))).unwrap());
        drop(server);
        json!({"stopped": true, "env_closes": count(&counts.closes)})
    }
}
fn wait_closed(counts: &Counts, expected: usize) {
    let guard = counts.cleanup.0.lock().unwrap();
    let (_guard, _) = counts
        .cleanup
        .1
        .wait_timeout_while(guard, Duration::from_secs(10), |_| {
            count(&counts.closes) < expected
        })
        .unwrap();
    assert_eq!(count(&counts.closes), expected, "cleanup did not finish");
}

fn drop_run() -> Json {
    let counts = Arc::new(Counts::default());
    let mut env = ScriptedEnv::new(&counts);
    let (started, start) = tokio::sync::oneshot::channel();
    let (release, blocked) = std::sync::mpsc::sync_channel(1);
    let mut first = Some(started);
    let seen = counts.clone();
    let model = Model::new(
        move |_: Value| {
            seen.model_starts.fetch_add(1, Ordering::SeqCst);
            if let Some(started) = first.take() {
                started.send(()).unwrap();
                blocked.recv_timeout(Duration::from_secs(10)).unwrap();
            }
            Ok(Value::Int(0))
        },
        ModelOptions {
            on_episode_end: {
                let c = counts.clone();
                Some(Box::new(move |_| {
                    c.ends.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }))
            },
            ..Default::default()
        },
    )
    .unwrap();
    let mut session = model.session(&mut env, SessionOptions::default()).unwrap();
    let mut hooks = Hooks::default();
    futures::executor::block_on(async {
        let mut running = Box::pin(session.run_async(SessionRunOptions {
            hooks: Some(&mut hooks),
            ..Default::default()
        }));
        tokio::select! { result = &mut running => panic!("run completed before drop: {result:?}"), _ = start => {} }
        drop(running);
    });
    release.send(()).unwrap();
    // A subsequent model call fences the in-flight callback without ending the episode.
    session.predict(Value::Int(0)).unwrap();
    assert_eq!(
        count(&counts.env_starts),
        1,
        "a step must not start after drop"
    );
    assert_eq!(
        count(&counts.model_starts),
        2,
        "only the callback in flight and our explicit probe ran"
    );
    assert_eq!(count(&counts.ends), 0);
    assert_eq!(hooks.ends, 0);
    session.reset(Some(0), None).unwrap();
    let ends = count(&counts.ends);
    session.step(Value::Int(0)).unwrap();
    json!({"run_ends": hooks.ends, "episode_ends_before_reset": 0, "episode_ends": ends, "no_calls_after_drop": true, "reusable": true})
}

fn warning(variant: &str) -> Json {
    let caller = std::thread::current().id();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let fail = matches!(variant, "fail" | "runtime_fail" | "recorder");
    set_warning_handler(Some(Box::new(move |warning| {
        sink.lock().unwrap().push((
            warning.category.name(),
            std::thread::current().id() == caller,
        ));
        if fail {
            Err(Error::Type("warning abort".into()))
        } else {
            Ok(())
        }
    })));
    let counts = Arc::default();
    let mut env = ScriptedEnv::new(&counts);
    struct Chunk;
    impl Policy for Chunk {
        const CORNERS: Corners = Corners::PREDICT.union(Corners::PREDICT_CHUNK);
        fn predict(&mut self, _: Value) -> Result<Value> {
            Ok(Value::Int(0))
        }
        fn predict_chunk(&mut self, _: Value, _: u32) -> Result<Value> {
            Ok(Value::List(vec![Value::Int(0)]))
        }
    }
    let result = if variant == "workflow" {
        // Binding hosts emit on the worker; delivery still belongs to the caller.
        let mut emitted = false;
        let model = Model::new(
            move |_: Value| {
                if !emitted {
                    emitted = true;
                    binding::emit_warning(
                        WarningCategory::WorkflowEditionWarning,
                        "scripted floating edition".into(),
                    )?;
                }
                Ok(Value::Int(0))
            },
            ModelOptions::default(),
        )
        .unwrap();
        model.run(&mut env, RunOptions::default())
    } else if variant.starts_with("runtime") {
        Model::new(Chunk, ModelOptions::default()).unwrap().run(
            &mut env,
            RunOptions {
                execution_horizon: 2,
                ..Default::default()
            },
        )
    } else if variant == "recorder" {
        let mut recorder = Recorder::new(RecorderOptions::default()).unwrap();
        let mut capture = recorder
            .capture(
                "model",
                "env",
                RecorderCaptureOptions {
                    cameras: Some(vec!["absent".into()]),
                    ..Default::default()
                },
            )
            .unwrap();
        model(&counts).run(
            &mut env,
            RunOptions {
                hooks: Some(capture.as_mut()),
                ..Default::default()
            },
        )
    } else {
        model(&counts).run(
            &mut env,
            RunOptions {
                execution_horizon: 2,
                ..Default::default()
            },
        )
    };
    set_warning_handler(None);
    let mut out = match result {
        Err(error) => error_record(error),
        Ok(_) => json!({}),
    };
    let warnings = seen.lock().unwrap();
    out["category"] = json!(warnings.last().unwrap().0);
    out["calling_thread"] = json!(warnings.iter().all(|w| w.1));
    out["warnings"] = json!(warnings.len());
    out
}

fn model_callback(timeout: bool) -> Json {
    use binding::model::{ModelHandler, ModelObservation, ModelWorker, ServeModelOptions};
    struct Constant {
        timeout: bool,
    }
    #[async_trait::async_trait]
    impl ModelHandler for Constant {
        async fn predict(
            &mut self,
            observation: ModelObservation,
        ) -> Result<Vec<binding::spaces::SpaceValue>> {
            if self.timeout {
                return Err(binding::AttachedError::with_message(
                    Error::Timeout(Duration::from_secs(5)),
                    "authored timeout: checkpoint shard violet".into(),
                ));
            }
            Ok((0..observation.num_envs)
                .map(|_| binding::spaces::SpaceValue::Discrete(0))
                .collect())
        }
    }
    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap(),
    );
    let bound = runtime
        .block_on(
            ModelWorker::new(Constant { timeout })
                .bind_async(ServeModelOptions::parse("tcp://127.0.0.1:0").unwrap()),
        )
        .unwrap();
    let address = bound.local_addr().to_string();
    let shutdown = bound.shutdown_trigger();
    let serving = std::thread::spawn(move || runtime.block_on(bound.serve()));
    let server = EnvServer::new(
        ScriptedEnv::new(&Arc::default()),
        EnvServerOptions {
            address: Some("tcp://127.0.0.1:0".into()),
            ..Default::default()
        },
    )
    .unwrap();
    server.start().unwrap();
    let remote = RemoteModel::new(
        RemoteModelOptions::default()
            .address(address)
            .connect_timeout(Duration::from_secs(5)),
    )
    .unwrap();
    let callback = Model::new(
        move |_: Value| {
            let mut session = session(&remote, &server, SessionOptions::default())?;
            let (obs, _) = session.reset(Some(0), None)?;
            session.predict(obs)
        },
        ModelOptions::default(),
    )
    .unwrap();
    let result = callback.run(
        &mut ScriptedEnv::new(&Arc::default()),
        RunOptions::default(),
    );
    callback.close().unwrap();
    shutdown.trigger("fixture finished");
    serving.join().unwrap().unwrap();
    if timeout {
        error_record(result.unwrap_err())
    } else {
        json!({"episodes": result.unwrap().num_episodes()})
    }
}
fn serve_drop_nonblocking() -> Json {
    struct Gated {
        env: ScriptedEnv,
        release: std::sync::mpsc::Receiver<()>,
    }
    impl Env for Gated {
        fn observation_space(&self) -> Space {
            self.env.observation_space()
        }
        fn action_space(&self) -> Space {
            self.env.action_space()
        }
        fn reset(
            &mut self,
            seed: Option<i64>,
            options: Option<&InfoDict>,
        ) -> Result<(Value, InfoDict)> {
            self.env.reset(seed, options)
        }
        fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
            self.env.step(action)
        }
        fn close(&mut self) -> Result<()> {
            self.release.recv_timeout(Duration::from_secs(10)).unwrap();
            self.env.close()
        }
    }
    let counts = Arc::default();
    let (release, blocked) = std::sync::mpsc::channel();
    let server = Arc::new(
        EnvServer::new(
            Gated {
                env: ScriptedEnv::new(&counts),
                release: blocked,
            },
            EnvServerOptions {
                address: Some("tcp://127.0.0.1:0".into()),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let served = server.clone();
    let (dropped, drop_done) = std::sync::mpsc::channel();
    let task = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            tokio::spawn(async move {
                let mut serving = Box::pin(served.serve_async());
                assert!(futures::poll!(serving.as_mut()).is_pending());
                drop(serving);
                dropped.send(()).unwrap();
            })
            .await
            .unwrap();
        });
    });
    let nonblocking = drop_done.recv_timeout(Duration::from_secs(5)).is_ok();
    release.send(()).unwrap();
    task.join().unwrap();
    assert!(server.wait(Some(Duration::from_secs(10))).unwrap());
    drop(server);
    json!({"nonblocking": nonblocking, "env_closes": count(&counts.closes)})
}

fn drop_owned_cleanup() -> Json {
    let counts = Arc::default();
    let model = model(&counts);
    let server = EnvServer::new(
        ScriptedEnv::new(&counts),
        EnvServerOptions {
            address: Some("tcp://127.0.0.1:0".into()),
            ..Default::default()
        },
    )
    .unwrap();
    server.start().unwrap();
    let address = server.address().to_owned();
    let borrowed = model.clone();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        tokio::spawn(async move {
            let mut session = borrowed
                .session_async(address.as_str(), SessionOptions::default())
                .await
                .unwrap();
            session.reset_async(Some(0), None).await.unwrap();
            drop(session);
        })
        .await
        .unwrap();
    });
    // The next policy call waits behind the owned episode-end cleanup.
    let mut local = ScriptedEnv::new(&Arc::default());
    let mut probe = model
        .session(&mut local, SessionOptions::default())
        .unwrap();
    let (obs, _) = probe.reset(Some(0), None).unwrap();
    probe.predict(obs).unwrap();
    let episode_ends = count(&counts.ends);
    drop(probe);
    let mut remote = RemoteEnv::new(
        RemoteEnvOptions::default()
            .address(server.address())
            .request_timeout(Duration::from_secs(2)),
    )
    .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        match remote.reset(Some(0), None) {
            Ok(_) => break,
            Err(error) => {
                assert!(error.to_string().contains("active Join session"), "{error}");
                assert!(
                    std::time::Instant::now() < deadline,
                    "owned connection was not closed"
                );
                std::thread::yield_now();
            }
        }
    }
    remote.step(Value::Int(0)).unwrap();
    remote.close().unwrap();
    let out = json!({"episode_ends": episode_ends, "env_closes": count(&counts.closes), "model_closes": count(&counts.model_closes), "reusable": true});
    server.shutdown().unwrap();
    out
}

/// Replace a run-specific address part (a port, a socket path) in the text.
fn masked(mut record: Json, value: &str, mask: &str) -> Json {
    let text = record["text"].as_str().unwrap().replace(value, mask);
    record["text"] = json!(text);
    record
}

/// A served model reached with an api_key, over TLS, or driving a local env.
fn remote_auth(variant: &str, env: &mut ScriptedEnv, model: &Model) -> Json {
    match variant {
        "invalid_header" => {
            return error_record(
                RemoteModel::new(
                    RemoteModelOptions::default()
                        .address("https://localhost:1")
                        .api_key("bad\nkey"),
                )
                .unwrap_err(),
            );
        }
        "https_to_plaintext" => {
            let server = EnvServer::new(
                ScriptedEnv::new(&Arc::default()),
                EnvServerOptions {
                    address: Some("tcp://127.0.0.1:0".into()),
                    ..Default::default()
                },
            )
            .unwrap();
            server.start().unwrap();
            let port = server.address().rsplit(':').next().unwrap().to_string();
            let served = RemoteModel::new(
                RemoteModelOptions::default().address(format!("https://127.0.0.1:{port}")),
            )
            .unwrap();
            let error = run(&served, &mut *env, RunOptions::default()).unwrap_err();
            server.shutdown().unwrap();
            return masked(error_record(error), &port, "PORT");
        }
        _ => {}
    }
    let dir = std::env::temp_dir().join(format!("rlmesh-parity-{}", mint_parity_id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.sock").display().to_string();
    let address = format!("unix://{path}");
    let cancel = ext::CancelToken::new();
    let token =
        matches!(variant, "api_key_rejected" | "api_key_accepted").then(|| "right-key".to_string());
    let serving = {
        let (model, address, cancel) = (model.clone(), address.clone(), cancel.clone());
        std::thread::spawn(move || {
            model.serve(
                &address,
                ModelServeOptions {
                    cancel: Some(cancel),
                    ext: Some(ext::ModelServeOptions {
                        token,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )
        })
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while std::os::unix::net::UnixStream::connect(&path).is_err() {
        assert!(
            std::time::Instant::now() < deadline,
            "the model never served"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let mut options = RemoteModelOptions::default().address(&address);
    options.api_key = match variant {
        "api_key_rejected" => Some("wrong-key".into()),
        "api_key_accepted" => Some("right-key".into()),
        _ => None,
    };
    let served = RemoteModel::new(options).unwrap();
    let out = match run(&served, &mut *env, RunOptions::default()) {
        Ok(result) => json!({
            "episodes": result.num_episodes(),
            "steps": result.total_steps(),
            "env_closes": count(&env.counts.closes),
        }),
        Err(error) => masked(error_record(error), &path, "SOCKET"),
    };
    cancel.cancel();
    let _ = serving.join().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    out
}

/// A process-unique suffix for scratch paths.
fn mint_parity_id() -> String {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::SeqCst)
    )
}

fn drop_queued_predict() -> Json {
    let counts = Arc::new(Counts::default());
    let seen = counts.clone();
    let (started, start) = tokio::sync::oneshot::channel();
    let (release, blocked) = std::sync::mpsc::sync_channel(1);
    let mut signal = Some(started);
    let model = Model::new(
        move |_: Value| {
            seen.model_starts.fetch_add(1, Ordering::SeqCst);
            if let Some(signal) = signal.take() {
                signal.send(()).unwrap();
                blocked.recv_timeout(Duration::from_secs(10)).unwrap();
            }
            Ok(Value::Int(0))
        },
        ModelOptions::default(),
    )
    .unwrap();
    let mut env1 = ScriptedEnv::new(&counts);
    let mut env2 = ScriptedEnv::new(&counts);
    let mut first_session = model.session(&mut env1, SessionOptions::default()).unwrap();
    let mut second_session = model.session(&mut env2, SessionOptions::default()).unwrap();
    first_session.reset(Some(0), None).unwrap();
    second_session.reset(Some(0), None).unwrap();
    futures::executor::block_on(async {
        let mut first = Box::pin(first_session.predict_async(Value::Int(0)));
        tokio::select! { result = &mut first => panic!("predict completed before release: {result:?}"), _ = start => {} }
        let mut queued = Box::pin(second_session.predict_async(Value::Int(0)));
        assert!(futures::poll!(queued.as_mut()).is_pending());
        drop(queued);
        release.send(()).unwrap();
        assert_eq!(first.await.unwrap(), Value::Int(0));
    });
    first_session.predict(Value::Int(0)).unwrap();
    assert_eq!(count(&counts.model_starts), 2);
    assert_eq!(count(&counts.env_starts), 2);
    second_session.predict(Value::Int(0)).unwrap();
    drop(first_session);
    drop(second_session);
    model.close().unwrap();
    assert_eq!(count(&counts.model_starts), 3);
    json!({"no_calls_after_drop": true, "reusable": true})
}
