//! The Gymnasium-shaped env traits served end to end.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::ext::TensorExt;
use crate::spaces::{BoxOptions, Discrete, DiscreteOptions};
use crate::{DType, Error, ErrorClass, Kwargs, Tensor};

thread_local! {
    /// One client runtime per test thread: a client's connection lives on the
    /// runtime that dialed it.
    static CLIENT: tokio::runtime::Runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
}

/// Default options with a short drain, so a test does not wait out the 5 s
/// cap on its own client's idle connection.
fn quick() -> EnvServerOptions {
    EnvServerOptions {
        options: Some(crate::ServeOptions {
            drain_timeout: Some(Duration::from_millis(100)),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn client<T>(future: impl std::future::Future<Output = T>) -> T {
    CLIENT.with(|runtime| runtime.block_on(future))
}

#[derive(Default)]
struct Counters {
    closes: AtomicUsize,
    threads: Mutex<Vec<std::thread::ThreadId>>,
}

/// A three-step counter env with scripted failures.
struct Counter {
    counters: Arc<Counters>,
    step: i64,
    fail_step: Option<Error>,
    metadata: Option<Metadata>,
    /// How long each step takes.
    slow: Option<Duration>,
}

impl Counter {
    fn new(counters: &Arc<Counters>) -> Counter {
        Counter {
            counters: Arc::clone(counters),
            step: 0,
            fail_step: None,
            metadata: None,
            slow: None,
        }
    }

    fn record(&self) {
        self.counters
            .threads
            .lock()
            .unwrap()
            .push(std::thread::current().id());
    }
}

impl Env for Counter {
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
        self.record();
        self.step = seed.unwrap_or(0);
        let mut info = InfoDict::new();
        if let Some(trial) = trial_index(options, None)? {
            info.insert("trial", trial);
        }
        Ok((Value::Int(self.step), info))
    }
    fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.record();
        if let Some(err) = self.fail_step.clone() {
            return Err(err);
        }
        if let Some(slow) = self.slow {
            std::thread::sleep(slow);
        }
        self.step += 1;
        let reward = match action {
            Some(Value::Int(action)) => action as f64,
            _ => -1.0,
        };
        Ok((
            Value::Int(self.step % 5),
            reward,
            self.step >= 3,
            false,
            InfoDict::new(),
        ))
    }
    fn close(&mut self) -> Result<()> {
        self.record();
        self.counters.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn metadata(&self) -> Option<Metadata> {
        self.metadata.clone()
    }
    fn id(&self) -> Option<String> {
        Some("Counter-v0".to_string())
    }
}

fn reset(remote: &mut RemoteEnv, seed: Option<i64>) -> (Value, InfoDict) {
    remote.reset(seed, None).unwrap()
}

#[allow(clippy::type_complexity)]
fn step(remote: &mut RemoteEnv, action: i64) -> crate::Result<(Value, f64, bool, bool, InfoDict)> {
    remote.step(Value::Int(action))
}

fn dial(address: &str) -> crate::Result<RemoteEnv> {
    RemoteEnv::new(RemoteEnvOptions::default().address(address))
}

#[test]
fn a_started_server_serves_the_gymnasium_shape_and_closes_the_env() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    assert!(
        server.address().starts_with("tcp://127.0.0.1:"),
        "{}",
        server.address()
    );
    assert_eq!(server.env_contract().id(), "Counter-v0");
    assert_eq!(server.env_contract().num_envs(), 1);
    assert_eq!(
        server.to_string(),
        format!("EnvServer(address='{}')", server.address())
    );
    let err = server.wait(Some(Duration::ZERO)).unwrap_err();
    assert_eq!(err.class(), ErrorClass::RuntimeError);
    assert_eq!(err.to_string(), "wait() called before start()");

    server.start().unwrap();
    let err = server.start().unwrap_err();
    assert_eq!(err.class(), ErrorClass::RuntimeError);
    let mut remote = dial(server.address()).unwrap();
    let (observation, info) = reset(&mut remote, Some(1));
    assert_eq!(observation, Value::Int(1));
    assert!(matches!(info.get("episode_ids"), Some(Value::List(ids)) if ids.len() == 1));
    let (observation, reward, _, _, info) = step(&mut remote, 1).unwrap();
    assert_eq!(observation, Value::Int(2));
    assert_eq!(reward, 1.0);
    assert_eq!(info.get("completed_episodes"), Some(&Value::Int(0)));
    remote.close().unwrap();
    // close is idempotent; other I/O on a closed client is misuse (RMT-13).
    remote.close().unwrap();
    let err = remote.reset(None, None).unwrap_err();
    assert_eq!(err.class(), ErrorClass::RuntimeError);
    assert_eq!(err.to_string(), "RemoteEnv is closed");
    assert_eq!(remote.env_contract().id(), "Counter-v0");
    drop(remote);

    assert!(!server.wait(Some(Duration::from_millis(20))).unwrap());
    server.shutdown().unwrap();
    assert!(server.wait(None).unwrap());
    assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
    // Every callback, close included, ran on the lane's one thread.
    let threads = counters.threads.lock().unwrap();
    assert!(
        threads.windows(2).all(|pair| pair[0] == pair[1]),
        "{threads:?}"
    );
    assert_ne!(threads[0], std::thread::current().id());
    let err = server.serve().unwrap_err();
    assert_eq!(
        err.to_string(),
        "serve() called after the server has been stopped"
    );
}

#[test]
fn close_env_on_shutdown_false_leaves_the_env_open() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(
        Counter::new(&counters),
        EnvServerOptions {
            close_env_on_shutdown: false,
            ..Default::default()
        },
    )
    .unwrap();
    server.start().unwrap();
    server.shutdown().unwrap();
    assert_eq!(counters.closes.load(Ordering::SeqCst), 0);

    // A never-started server closes its env on shutdown, and drop does too.
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    drop(server);
    assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
}

#[test]
fn an_authored_failure_reads_op_failed_at_the_client() {
    let counters = Arc::new(Counters::default());
    let mut env = Counter::new(&counters);
    env.fail_step = Some(Error::environment_recoverable("try again"));
    let server = EnvServer::new(env, quick()).unwrap();
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    reset(&mut remote, None);
    let err = step(&mut remote, 0).unwrap_err();
    assert_eq!(err.class(), ErrorClass::RecoverableEnvironmentException);
    assert_eq!(err.to_string(), "[INTERNAL] step failed: try again");
    drop(remote);
    server.shutdown().unwrap();

    let mut env = Counter::new(&counters);
    env.fail_step = Some(Error::invalid_argument("boom"));
    let server = EnvServer::new(env, quick()).unwrap();
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    reset(&mut remote, None);
    let err = step(&mut remote, 0).unwrap_err();
    assert_eq!(err.class(), ErrorClass::EnvironmentException);
    assert_eq!(err.to_string(), "[INTERNAL] step failed: boom");
    drop(remote);
    server.shutdown().unwrap();
}

#[test]
fn attached_authored_failures_keep_their_wire_class_and_raw_text() {
    for (error, class, text) in [
        (
            Error::environment_recoverable("try again"),
            ErrorClass::RecoverableEnvironmentException,
            "[INTERNAL] step failed: try again",
        ),
        (
            Error::environment("failed"),
            ErrorClass::EnvironmentException,
            "[INTERNAL] step failed: failed",
        ),
        (
            Error::invalid_argument("bad value"),
            ErrorClass::EnvironmentException,
            "[INTERNAL] step failed: bad value",
        ),
    ] {
        let counters = Arc::new(Counters::default());
        let mut env = Counter::new(&counters);
        env.fail_step = Some(crate::binding::AttachedError::attach(error, ()));
        let server = EnvServer::new(env, quick()).unwrap();
        server.start().unwrap();
        let mut remote = dial(server.address()).unwrap();
        reset(&mut remote, None);
        let err = step(&mut remote, 0).unwrap_err();
        assert_eq!(err.class(), class);
        assert_eq!(err.to_string(), text);
        drop(remote);
        server.shutdown().unwrap();
    }
}

#[test]
fn lanes_serve_one_endpoint_and_must_agree() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::lanes(
        vec![Counter::new(&counters), Counter::new(&counters)],
        quick(),
    )
    .unwrap();
    assert_eq!(server.env_contract().num_envs(), 2);
    server.shutdown().unwrap();
    assert_eq!(counters.closes.load(Ordering::SeqCst), 2);

    let mut odd = Counter::new(&counters);
    let mut metadata = Metadata::new();
    metadata.insert("render_fps", 30);
    odd.metadata = Some(metadata);
    let err = EnvServer::lanes(vec![Counter::new(&counters), odd], quick()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert!(err.to_string().contains("lane 1 disagrees"), "{err}");

    let err = EnvServer::lanes(Vec::<Counter>::new(), quick()).unwrap_err();
    assert_eq!(err.to_string(), "EnvServer needs at least one environment");
}

#[test]
fn addresses_follow_python_s_normalization() {
    let counters = Arc::new(Counters::default());
    let refused = |options: EnvServerOptions| {
        let err = EnvServer::new(Counter::new(&counters), options).unwrap_err();
        assert_eq!(err.class(), ErrorClass::ValueError, "{err}");
        err.to_string()
    };
    assert_eq!(
        refused(EnvServerOptions {
            address: Some(String::new()),
            ..Default::default()
        }),
        "address cannot be empty"
    );
    refused(EnvServerOptions {
        address: Some("tcp://127.0.0.1:0".into()),
        port: Some(0),
        ..Default::default()
    });
    refused(EnvServerOptions {
        path: Some("/tmp/x.sock".into()),
        transport: Some(Transport::Tcp),
        ..Default::default()
    });
    refused(EnvServerOptions {
        transport: Some(Transport::Unix),
        ..Default::default()
    });
    let server = EnvServer::new(
        Counter::new(&counters),
        EnvServerOptions {
            host: Some("127.0.0.1".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(server.address().starts_with("tcp://127.0.0.1:"));
    assert_eq!(Transport::Unix.to_string(), "unix");
}

#[test]
fn a_zero_timeout_is_refused_at_construction() {
    let counters = Arc::new(Counters::default());
    let err = EnvServer::new(
        Counter::new(&counters),
        EnvServerOptions {
            options: Some(crate::ServeOptions {
                close_timeout: Some(Duration::ZERO),
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert_eq!(err.to_string(), "close_timeout_seconds must be positive");
}

#[test]
fn a_foreground_serve_runs_lane_0_on_the_caller() {
    let counters = Arc::new(Counters::default());
    let server = Arc::new(EnvServer::new(Counter::new(&counters), quick()).unwrap());
    let address = server.address().to_string();
    let serving = {
        let server = Arc::clone(&server);
        std::thread::spawn(move || (std::thread::current().id(), server.serve()))
    };
    let mut remote = loop {
        if let Ok(remote) = dial(&address) {
            break remote;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    reset(&mut remote, None);
    drop(remote);
    let err = server.wait(None).unwrap_err();
    assert!(err.to_string().contains("foreground"), "{err}");
    server.shutdown().unwrap();
    let (thread, result) = serving.join().unwrap();
    result.unwrap();
    assert!(server.wait(None).unwrap());
    let threads = counters.threads.lock().unwrap();
    assert!(threads.iter().all(|id| *id == thread), "{threads:?}");
    assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
}

#[test]
fn serve_async_drops_stop_the_server() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    client(async {
        let serving = server.serve_async();
        futures::pin_mut!(serving);
        // Poll once so the server starts, then drop the future.
        assert!(futures::poll!(serving.as_mut()).is_pending());
    });
    assert!(server.wait(Some(Duration::from_secs(5))).unwrap());
    drop(server);
    assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
}

/// A two-lane vector env over Discrete actions.
struct Pair {
    num_envs: u64,
    seen: Arc<Mutex<Vec<Value>>>,
}

impl VectorEnv for Pair {
    fn num_envs(&self) -> u64 {
        self.num_envs
    }
    fn single_observation_space(&self) -> Space {
        crate::spaces::Box::new(-1.0, 1.0, [2], BoxOptions::default())
            .unwrap()
            .into()
    }
    fn single_action_space(&self) -> Space {
        Discrete::new(3, DiscreteOptions::default()).unwrap().into()
    }
    fn reset(
        &mut self,
        seed: Option<VectorSeed>,
        _options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.seen
            .lock()
            .unwrap()
            .push(Value::Text(format!("{seed:?}")));
        Ok((self.observations()?, InfoDict::new()))
    }
    fn step(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        self.seen.lock().unwrap().push(actions);
        let n = self.num_envs as usize;
        Ok((
            self.observations()?,
            vec![0.5; n],
            vec![false; n],
            vec![false; n],
            InfoDict::new(),
        ))
    }
}

impl Pair {
    fn observations(&self) -> Result<Value> {
        let n = self.num_envs as usize;
        let data: Vec<u8> = std::iter::repeat_n(0.25f32, 2 * n)
            .flat_map(f32::to_le_bytes)
            .collect();
        Ok(Value::Tensor(Tensor::from_vec(
            data,
            &[n, 2],
            DType::Float32,
        )?))
    }
}

#[test]
fn a_vector_env_sees_the_batched_layout() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let server = EnvServer::vector(
        Pair {
            num_envs: 2,
            seen: Arc::clone(&seen),
        },
        quick(),
    )
    .unwrap();
    assert_eq!(server.env_contract().num_envs(), 2);
    server.start().unwrap();
    let mut remote =
        RemoteVectorEnv::new(RemoteEnvOptions::default().address(server.address())).unwrap();
    assert_eq!(remote.num_envs(), 2);
    assert!(std::ptr::eq(
        remote.observation_space(),
        remote.single_observation_space()
    ));
    let (observation, info) = remote.reset(Some(VectorSeed::One(7)), None).unwrap();
    assert_eq!(observation.as_tensor().unwrap().shape(), &[2, 2]);
    assert!(matches!(info.get("episode_ids"), Some(Value::List(ids)) if ids.len() == 2));
    let (_, rewards, terminated, _, info) = remote
        .step(Value::List(vec![Value::Int(1), Value::Int(2)]))
        .unwrap();
    assert_eq!(rewards, vec![0.5, 0.5]);
    assert_eq!(terminated, vec![false, false]);
    assert_eq!(info.get("completed_episodes"), Some(&Value::Int(0)));
    let err =
        crate::ext::reset_lanes(&mut remote, &[u64::from(u32::MAX) + 1], None, None).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    // A vector endpoint refuses the single-env client.
    let err = dial(server.address()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert!(
        err.to_string().contains("Use RemoteVectorEnv instead"),
        "{err}"
    );
    drop(remote);
    server.shutdown().unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0], Value::Text("Some(One(7))".into()));
    let actions = seen[1].as_tensor().unwrap();
    assert_eq!(actions.shape(), &[2]);
    assert_eq!(actions.dtype(), DType::Int64);
}

#[test]
fn a_one_lane_vector_env_serves_as_a_scalar_lane() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let server = EnvServer::vector(
        Pair {
            num_envs: 1,
            seen: Arc::clone(&seen),
        },
        quick(),
    )
    .unwrap();
    assert_eq!(server.env_contract().num_envs(), 1);
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    reset(&mut remote, Some(3));
    let (_, reward, _, _, _) = step(&mut remote, 2).unwrap();
    assert_eq!(reward, 0.5);
    // A one-lane endpoint refuses the vector client.
    let err =
        RemoteVectorEnv::new(RemoteEnvOptions::default().address(server.address())).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    drop(remote);
    server.shutdown().unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0], Value::Text("Some(One(3))".into()));
    // Python's `_batch1`: a Discrete action is a one-item list.
    assert_eq!(seen[1], Value::List(vec![Value::Int(2)]));

    let err = EnvServer::vector(
        Pair {
            num_envs: 0,
            seen: Arc::new(Mutex::new(Vec::new())),
        },
        quick(),
    )
    .unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
}

#[test]
fn autoreset_mode_follows_env_27() {
    let counters = Arc::new(Counters::default());
    for (mode, ok) in [
        (Value::Text("NextStep".into()), true),
        (Value::Text("Disabled".into()), true),
        (Value::Text("SameStep".into()), false),
        (Value::Text("Bogus".into()), false),
        (Value::None, false),
        (Value::Int(1), false),
    ] {
        let mut env = Counter::new(&counters);
        let mut metadata = Metadata::new();
        metadata.insert("autoreset_mode", mode.clone());
        env.metadata = Some(metadata);
        let result = EnvServer::new(env, quick());
        assert_eq!(result.is_ok(), ok, "{mode:?}");
        if let Err(err) = result {
            assert_eq!(err.class(), ErrorClass::ValueError);
        }
    }
}

/// A factory with one discriminant.
#[derive(Default)]
struct Factory {
    made: Vec<Kwargs>,
}

impl EnvFactory for Factory {
    fn tag_params() -> Vec<TagParam> {
        vec![TagParam {
            name: "mode".into(),
            default_: Value::Text("a".into()),
            choices: vec![Value::Text("a".into()), Value::Text("b".into())],
        }]
    }
    fn reset_options() -> Vec<String> {
        vec!["trial_index".into()]
    }
    fn make(&mut self, kwargs: &Kwargs) -> Result<MadeEnv> {
        self.made.push(kwargs.clone());
        Ok(MadeEnv::env(Counter::new(&Arc::new(Counters::default()))))
    }
}

#[test]
fn make_env_binds_discriminants_and_stamps() {
    let mut factory = Factory::default();
    let MadeEnv::Env(env) = factory.make_env(&Kwargs::new()).unwrap() else {
        panic!("a scalar env")
    };
    assert_eq!(factory.made[0].get("mode"), Some(&Value::Text("a".into())));
    let metadata = env.metadata().unwrap();
    let mut branch = Kwargs::new();
    branch.insert("mode", "a");
    assert_eq!(
        metadata.get(rlmesh_adapters::v1::ENV_BRANCH_METADATA_KEY),
        Some(&Value::Dict(branch))
    );
    assert_eq!(
        metadata.get(crate::ENV_RESET_OPTIONS_KEY),
        Some(&Value::List(vec![Value::Text("trial_index".into())]))
    );

    let mut kwargs = Kwargs::new();
    kwargs.insert("mode", "c");
    let err = factory.make_env(&kwargs).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert_eq!(
        err.to_string(),
        "Factory.make() got mode='c', which is not one of the contract discriminant's choices \
         ['a', 'b']"
    );
    assert_eq!(
        factory.made.len(),
        1,
        "make must not run for a refused binding"
    );
}

#[derive(Default)]
struct Unbounded;

impl EnvFactory for Unbounded {
    fn tag_params() -> Vec<TagParam> {
        let choices: Vec<Value> = (0..3).map(Value::Int).collect();
        ["x", "y"]
            .into_iter()
            .map(|name| TagParam {
                name: name.into(),
                default_: Value::Int(0),
                choices: choices.clone(),
            })
            .collect()
    }
    fn make(&mut self, _: &Kwargs) -> Result<MadeEnv> {
        Ok(MadeEnv::env(Counter::new(&Arc::new(Counters::default()))))
    }
}

#[test]
fn the_branch_table_is_checked() {
    let err = Unbounded.make_env(&Kwargs::new()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::TypeError);
    assert!(err.to_string().contains("9 contract branches"), "{err}");
    let branches = super::factory::branches::<Factory>();
    assert_eq!(branches.len(), 2);
    assert_eq!(branches[0].get("mode"), Some(&Value::Text("a".into())));
}

#[test]
fn an_already_fired_cancel_runs_no_prepare() {
    #[derive(Default)]
    struct Prepared(usize);
    impl EnvFactory for Prepared {
        fn prepare(&mut self) -> Result<()> {
            self.0 += 1;
            Ok(())
        }
        fn make(&mut self, _: &Kwargs) -> Result<MadeEnv> {
            Ok(MadeEnv::env(Counter::new(&Arc::new(Counters::default()))))
        }
    }
    let cancel = crate::ext::CancelToken::new();
    cancel.cancel();
    let mut factory = Prepared::default();
    let err = factory
        .serve(
            "tcp://127.0.0.1:0",
            EnvFactoryServeOptions {
                cancel: Some(cancel.clone()),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::KeyboardInterrupt);
    assert_eq!(factory.0, 0);

    // A cancel fired while serving drains, closes and fails with the class.
    let canceller = cancel.clone();
    let thread = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        canceller.cancel();
    });
    let err = factory
        .serve(
            "tcp://127.0.0.1:0",
            EnvFactoryServeOptions {
                cancel: Some(cancel),
                ..Default::default()
            },
        )
        .unwrap_err();
    thread.join().unwrap();
    assert_eq!(err.class(), ErrorClass::KeyboardInterrupt);
    assert_eq!(factory.0, 1);
}

#[test]
fn variants_print_python_s_repr() {
    let mut params = Kwargs::new();
    params.insert("task", 1);
    let mut metadata = Kwargs::new();
    metadata.insert("name", "Pick it's");
    let variant = Variant::new("a/b", params, VariantOptions { metadata });
    assert_eq!(
        variant.to_string(),
        "Variant(id='a/b', params={'task': 1}, metadata={'name': \"Pick it's\"})"
    );
    assert_eq!(variant.clone(), variant);
}

#[test]
fn blocking_calls_refuse_an_async_context() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let err = runtime
        .block_on(async { tokio::spawn(async move { server.serve() }).await })
        .unwrap()
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::RuntimeError);
    assert_eq!(
        err.to_string(),
        "EnvServer::serve() blocks; call serve_async() from async code"
    );
}

#[test]
fn a_session_borrows_a_remote_env_and_hands_it_back() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    let edition = remote.selected_workflow_edition().to_string();
    assert!(!edition.is_empty());
    {
        let mut session = crate::session(
            crate::RANDOM_SAMPLE,
            &mut remote,
            crate::SessionOptions::default(),
        )
        .unwrap();
        let (observation, info) = session.reset(Some(2), None).unwrap();
        assert_eq!(observation, Value::Int(2));
        assert!(info.contains_key("episode_ids"));
        session.step(Value::Int(1)).unwrap();
        assert_eq!(session.selected_workflow_edition().unwrap(), edition);
        session.close().unwrap();
    }
    // The core is back: the handle drives on and reports the same edition.
    assert_eq!(remote.selected_workflow_edition(), edition);
    let (observation, _) = reset(&mut remote, Some(0));
    assert_eq!(observation, Value::Int(0));
    // A dropped session hands it back too.
    drop(
        crate::session(
            crate::RANDOM_SAMPLE,
            &mut remote,
            crate::SessionOptions::default(),
        )
        .unwrap(),
    );
    step(&mut remote, 1).unwrap();
    remote.close().unwrap();
    server.shutdown().unwrap();
}

#[test]
fn async_twins_run_on_any_executor() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    server.start().unwrap();
    let address = server.address().to_string();
    futures::executor::block_on(async {
        let mut remote = RemoteEnv::new_async(RemoteEnvOptions::default().address(&address))
            .await
            .unwrap();
        let (observation, _) = remote.reset_async(Some(1), None).await.unwrap();
        assert_eq!(observation, Value::Int(1));
        let (_, reward, _, _, _) = remote.step_async(Value::Int(1)).await.unwrap();
        assert_eq!(reward, 1.0);
        assert_eq!(remote.render_async(None).await.unwrap(), None);
        remote.close_async().await.unwrap();
    });
    // A sync call inside a runtime names its twin instead of blocking.
    let inside = address.clone();
    let err =
        client(async { tokio::spawn(async move { dial(&inside) }).await.unwrap() }).unwrap_err();
    assert_eq!(err.class(), ErrorClass::RuntimeError);
    assert!(err.to_string().contains("RemoteEnv::new_async()"), "{err}");
    server.shutdown().unwrap();
}

#[test]
fn a_request_past_its_timeout_is_a_timeout_error() {
    let counters = Arc::new(Counters::default());
    let mut env = Counter::new(&counters);
    env.slow = Some(Duration::from_millis(400));
    let server = EnvServer::new(env, quick()).unwrap();
    server.start().unwrap();
    let mut remote = RemoteEnv::new(
        RemoteEnvOptions::default()
            .address(server.address())
            .request_timeout(Duration::from_millis(50)),
    )
    .unwrap();
    reset(&mut remote, None);
    let err = step(&mut remote, 1).unwrap_err();
    assert_eq!(err.class(), ErrorClass::TimeoutError);
    assert_eq!(err.to_string(), "operation timed out after 50ms");
    drop(remote);
    server.shutdown().unwrap();
}

#[test]
fn a_fired_cancel_handle_fails_one_call() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    remote.cancel_handle().cancel();
    let err = remote.reset(None, None).unwrap_err();
    assert_eq!(err.class(), ErrorClass::KeyboardInterrupt);
    reset(&mut remote, None);
    drop(remote);
    server.shutdown().unwrap();
}

#[test]
fn construction_errors_follow_rmt_12() {
    let err = RemoteEnv::new(RemoteEnvOptions::default()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert_eq!(err.to_string(), "address or host/port is required");
    let err = dial("bogus://x").unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert!(err.to_string().starts_with("invalid address: "), "{err}");
    let err = RemoteEnv::new(
        RemoteEnvOptions::default()
            .address("tcp://127.0.0.1:1")
            .api_key("s3cret\n"),
    )
    .unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    // Nothing listens on port 1.
    let err = RemoteEnv::new(
        RemoteEnvOptions::default()
            .port(1)
            .connect_timeout(Duration::from_secs(5)),
    )
    .unwrap_err();
    assert!(
        matches!(
            err.class(),
            ErrorClass::ConnectionError | ErrorClass::TimeoutError
        ),
        "{err:?}"
    );
    assert!(
        err.to_string()
            .starts_with("could not connect to an env at 'tcp://127.0.0.1:1'"),
        "{err}"
    );
}

#[test]
fn empty_connect_host_defaults_to_loopback() {
    assert_eq!(
        super::client::normalize_connect_address(None, Some(""), Some(1234), None, None).unwrap(),
        "tcp://127.0.0.1:1234"
    );
}

#[test]
fn foreground_callback_shutdown_marks_the_server_stopped_at_once() {
    struct Stops(Arc<Mutex<Option<std::sync::Weak<EnvServer>>>>);
    impl Env for Stops {
        fn observation_space(&self) -> Space {
            Discrete::new(2, DiscreteOptions::default()).unwrap().into()
        }
        fn action_space(&self) -> Space {
            self.observation_space()
        }
        fn reset(&mut self, _: Option<i64>, _: Option<&InfoDict>) -> Result<(Value, InfoDict)> {
            let server = self.0.lock().unwrap().as_ref().unwrap().upgrade().unwrap();
            server.shutdown()?;
            assert!(server.wait(Some(Duration::ZERO))?);
            Ok((Value::Int(0), InfoDict::new()))
        }
        fn step(&mut self, _: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
            unreachable!()
        }
    }
    let owner = Arc::new(Mutex::new(None));
    let server = Arc::new(EnvServer::new(Stops(owner.clone()), quick()).unwrap());
    *owner.lock().unwrap() = Some(Arc::downgrade(&server));
    let serving = server.clone();
    let thread = std::thread::spawn(move || serving.serve());
    let mut remote =
        crate::RemoteEnv::new(crate::RemoteEnvOptions::default().address(server.address()))
            .unwrap();
    remote.reset(None, None).unwrap();
    remote.close().unwrap();
    thread.join().unwrap().unwrap();
}
