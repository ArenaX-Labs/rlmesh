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
/// A v1 server: these tests cover the gRPC wire's own serving options.
fn quick() -> EnvServerOptions {
    EnvServerOptions {
        address: Some("tcp://127.0.0.1:0".into()),
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

#[test]
fn concurrent_shutdown_and_wait_join_the_in_progress_drain() {
    use std::future::Future;
    use std::sync::{Condvar, mpsc};

    struct HeldClose {
        entered: mpsc::Sender<()>,
        release: Arc<(Mutex<bool>, Condvar)>,
        server: Arc<Mutex<std::sync::Weak<EnvServer>>>,
    }
    impl Env for HeldClose {
        fn observation_space(&self) -> Space {
            Discrete::new(5, DiscreteOptions::default()).unwrap().into()
        }
        fn action_space(&self) -> Space {
            self.observation_space()
        }
        fn reset(&mut self, _: Option<i64>, _: Option<&InfoDict>) -> Result<(Value, InfoDict)> {
            unreachable!()
        }
        fn step(&mut self, _: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
            unreachable!()
        }
        fn close(&mut self) -> crate::Result<()> {
            // Ready cleanup must mark its callbacks too: a shutdown requested
            // by close only signals and never waits for its own drain lock.
            self.server.lock().unwrap().upgrade().unwrap().shutdown()?;
            self.entered.send(()).unwrap();
            let (lock, ready) = &*self.release;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = ready.wait(released).unwrap();
            }
            Ok(())
        }
    }
    for started in [false, true] {
        for asynchronous in [0, 1, 2] {
            let (entered, closing) = mpsc::channel();
            let release = Arc::new((Mutex::new(false), Condvar::new()));
            let served = Arc::new(Mutex::new(std::sync::Weak::new()));
            let server = Arc::new(
                EnvServer::new(
                    HeldClose {
                        entered,
                        release: Arc::clone(&release),
                        server: Arc::clone(&served),
                    },
                    quick(),
                )
                .unwrap(),
            );
            *served.lock().unwrap() = Arc::downgrade(&server);
            if started {
                server.start().unwrap();
            }
            let first = {
                let server = Arc::clone(&server);
                std::thread::spawn(move || {
                    if asynchronous == 2 {
                        let mut shutdown = Box::pin(server.shutdown_async());
                        let waker = futures::task::noop_waker();
                        let mut context = std::task::Context::from_waker(&waker);
                        assert!(shutdown.as_mut().poll(&mut context).is_pending());
                        drop(shutdown);
                        Ok(())
                    } else if asynchronous == 1 {
                        futures::executor::block_on(server.shutdown_async())
                    } else {
                        server.shutdown()
                    }
                })
            };
            closing.recv_timeout(Duration::from_secs(5)).unwrap();
            let (completed, observed) = mpsc::channel();
            let second = {
                let server = Arc::clone(&server);
                std::thread::spawn(move || {
                    completed
                        .send(futures::executor::block_on(async { server.shutdown() }))
                        .unwrap();
                })
            };
            let early = observed.recv_timeout(Duration::from_millis(50));
            let polled = server.wait(Some(Duration::ZERO));
            // Always release the close, even if a regression made an observer
            // return early; no test failure may strand a library thread.
            *release.0.lock().unwrap() = true;
            release.1.notify_all();
            first.join().unwrap().unwrap();
            second.join().unwrap();
            assert!(matches!(early, Err(mpsc::RecvTimeoutError::Timeout)));
            assert!(!polled.unwrap());
            observed
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap();
            assert!(server.wait(Some(Duration::ZERO)).unwrap());
        }
    }
}

// Shutdown regressions use channels so the test controls the exact drain
// phase, including the two reentrant hangs reported during review.
struct ShutdownProbe {
    server: Arc<Mutex<std::sync::Weak<EnvServer>>>,
    closing: Option<std::sync::mpsc::Sender<()>>,
    release: Option<std::sync::mpsc::Receiver<()>>,
    dropped: std::sync::mpsc::Sender<()>,
    fail: bool,
    reenter_drop: bool,
}
impl Env for ShutdownProbe {
    fn observation_space(&self) -> Space {
        Discrete::new(5, DiscreteOptions::default()).unwrap().into()
    }
    fn action_space(&self) -> Space {
        self.observation_space()
    }
    fn reset(&mut self, _: Option<i64>, _: Option<&InfoDict>) -> Result<(Value, InfoDict)> {
        unreachable!()
    }
    fn step(&mut self, _: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        unreachable!()
    }
    fn close(&mut self) -> Result<()> {
        if let Some(server) = self.server.lock().unwrap().upgrade() {
            server.shutdown()?;
            assert!(server.wait(None)?);
        }
        if let Some(entered) = &self.closing {
            entered.send(()).unwrap();
        }
        if let Some(release) = &self.release {
            release.recv().unwrap();
        }
        if self.fail {
            Err(Error::Internal("held close failure".into()))
        } else {
            Ok(())
        }
    }
}
impl Drop for ShutdownProbe {
    fn drop(&mut self) {
        if self.reenter_drop
            && let Some(server) = self.server.lock().unwrap().upgrade()
        {
            server.shutdown().unwrap();
            assert!(server.wait(None).unwrap());
        }
        let _ = self.dropped.send(());
    }
}

#[test]
fn shutdown_marks_reentrant_destruction_and_callback_wait() {
    for started in [false, true] {
        for close in [false, true] {
            let (dropped, observed) = std::sync::mpsc::channel();
            let weak = Arc::new(Mutex::new(std::sync::Weak::new()));
            let env = ShutdownProbe {
                server: Arc::clone(&weak),
                closing: None,
                release: None,
                dropped,
                fail: false,
                reenter_drop: true,
            };
            let server = Arc::new(
                EnvServer::new(
                    env,
                    EnvServerOptions {
                        close_env_on_shutdown: close,
                        ..quick()
                    },
                )
                .unwrap(),
            );
            *weak.lock().unwrap() = Arc::downgrade(&server);
            if started {
                server.start().unwrap();
            }
            let worker = {
                let server = Arc::clone(&server);
                std::thread::spawn(move || server.shutdown())
            };
            observed
                .recv_timeout(Duration::from_secs(5))
                .expect("env destructor reentered its own drain");
            worker.join().unwrap().unwrap();
        }
    }
}

#[test]
fn serve_async_owns_its_error_while_other_observers_join() {
    use std::future::Future;
    for observer in [0, 1, 2] {
        let (closing, entered) = std::sync::mpsc::channel();
        let (release, blocked) = std::sync::mpsc::channel();
        let (dropped, _) = std::sync::mpsc::channel();
        let weak = Arc::new(Mutex::new(std::sync::Weak::new()));
        let server = Arc::new(
            EnvServer::new(
                ShutdownProbe {
                    server: Arc::clone(&weak),
                    closing: Some(closing),
                    release: Some(blocked),
                    dropped,
                    fail: true,
                    reenter_drop: false,
                },
                quick(),
            )
            .unwrap(),
        );
        *weak.lock().unwrap() = Arc::downgrade(&server);
        let mut serving = Box::pin(server.serve_async());
        let waker = futures::task::noop_waker();
        assert!(
            serving
                .as_mut()
                .poll(&mut std::task::Context::from_waker(&waker))
                .is_pending()
        );
        server.shutdown().unwrap();
        entered.recv_timeout(Duration::from_secs(5)).unwrap();
        let (completed, observed) = std::sync::mpsc::channel();
        let worker = {
            let server = Arc::clone(&server);
            std::thread::spawn(move || {
                let result = match observer {
                    0 => server.shutdown(),
                    1 => futures::executor::block_on(server.shutdown_async()),
                    _ => server.wait(None).map(|stopped| assert!(stopped)),
                };
                completed.send(result).unwrap();
            })
        };
        let early = observed.recv_timeout(Duration::from_millis(50));
        release.send(()).unwrap();
        let result = futures::executor::block_on(serving);
        worker.join().unwrap();
        if observer == 2 {
            assert!(early.is_err());
            observed
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap();
        } else {
            early
                .expect("repeated foreground shutdown must only signal")
                .unwrap();
        }
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("held close failure")
        );
    }
}

#[test]
fn synchronous_shutdown_joins_an_unpolled_async_drain() {
    use std::future::Future;
    let (closing, entered) = std::sync::mpsc::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let (dropped, _) = std::sync::mpsc::channel();
    let server = EnvServer::new(
        ShutdownProbe {
            server: Arc::new(Mutex::new(std::sync::Weak::new())),
            closing: Some(closing),
            release: Some(blocked),
            dropped,
            fail: false,
            reenter_drop: false,
        },
        quick(),
    )
    .unwrap();
    let mut shutdown = Box::pin(server.shutdown_async());
    struct Wakes(AtomicUsize);
    impl futures::task::ArcWake for Wakes {
        fn wake_by_ref(this: &Arc<Self>) {
            this.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    let wakes = Arc::new(Wakes(AtomicUsize::new(0)));
    let waker = futures::task::waker(Arc::clone(&wakes));
    assert!(
        shutdown
            .as_mut()
            .poll(&mut std::task::Context::from_waker(&waker))
            .is_pending()
    );
    entered.recv_timeout(Duration::from_secs(5)).unwrap();
    // A held close supplies no completion notification. Timer polling would
    // wake this unpolled future after 10ms; the drain must remain asleep.
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(wakes.0.load(Ordering::Relaxed), 0);
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        release.send(()).unwrap();
    });
    server.shutdown().unwrap();
    drop(shutdown);
    release.join().unwrap();
}

#[test]
fn rust_drop_detaches_a_close_beyond_its_timeout() {
    let (closing, entered) = std::sync::mpsc::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let (dropped, destroyed) = std::sync::mpsc::channel();
    let options = EnvServerOptions {
        options: Some(crate::ServeOptions {
            close_timeout: Some(Duration::from_millis(20)),
            ..Default::default()
        }),
        ..quick()
    };
    let server = EnvServer::new(
        ShutdownProbe {
            server: Arc::new(Mutex::new(std::sync::Weak::new())),
            closing: Some(closing),
            release: Some(blocked),
            dropped,
            fail: false,
            reenter_drop: false,
        },
        options,
    )
    .unwrap();
    let (done, joined) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        drop(server);
        done.send(()).unwrap();
    });
    entered.recv_timeout(Duration::from_secs(5)).unwrap();
    let early = joined.recv_timeout(Duration::from_secs(1));
    release.send(()).unwrap();
    worker.join().unwrap();
    early.expect("Rust Drop must detach the still-wedged close after its timeout");
    destroyed.recv_timeout(Duration::from_secs(5)).unwrap();
}

#[test]
fn async_wait_sleeps_until_a_started_or_ready_drain_completes() {
    use std::future::Future;
    struct Wakes(AtomicUsize);
    impl futures::task::ArcWake for Wakes {
        fn wake_by_ref(this: &Arc<Self>) {
            this.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    for started in [false, true] {
        for timeout in [None, Some(Duration::from_secs(5))] {
            let (closing, entered) = std::sync::mpsc::channel();
            let (release, blocked) = std::sync::mpsc::channel();
            let (dropped, _) = std::sync::mpsc::channel();
            let server = EnvServer::new(
                ShutdownProbe {
                    server: Arc::new(Mutex::new(std::sync::Weak::new())),
                    closing: Some(closing),
                    release: Some(blocked),
                    dropped,
                    fail: false,
                    reenter_drop: false,
                },
                quick(),
            )
            .unwrap();
            if started {
                server.start().unwrap();
            }
            let mut shutdown = Box::pin(server.shutdown_async());
            let noop = futures::task::noop_waker();
            assert!(
                shutdown
                    .as_mut()
                    .poll(&mut std::task::Context::from_waker(&noop))
                    .is_pending()
            );
            entered.recv_timeout(Duration::from_secs(5)).unwrap();
            let wakes = Arc::new(Wakes(AtomicUsize::new(0)));
            let waker = futures::task::waker(Arc::clone(&wakes));
            let mut waiting = Box::pin(server.wait_async(timeout));
            assert!(
                waiting
                    .as_mut()
                    .poll(&mut std::task::Context::from_waker(&waker))
                    .is_pending()
            );
            std::thread::sleep(Duration::from_millis(120));
            let count = wakes.0.load(Ordering::Relaxed);
            release.send(()).unwrap();
            assert!(futures::executor::block_on(waiting).unwrap());
            futures::executor::block_on(shutdown).unwrap();
            assert_eq!(count, 0, "no timer may wake wait_async every 50ms");
        }
    }
}

#[test]
fn close_callback_can_drop_the_last_server_arc() {
    struct DropOnClose {
        slot: Arc<Mutex<Option<Arc<EnvServer>>>>,
        done: std::sync::mpsc::Sender<()>,
    }
    impl Env for DropOnClose {
        fn observation_space(&self) -> Space {
            Discrete::new(2, DiscreteOptions::default()).unwrap().into()
        }
        fn action_space(&self) -> Space {
            self.observation_space()
        }
        fn reset(&mut self, _: Option<i64>, _: Option<&InfoDict>) -> Result<(Value, InfoDict)> {
            unreachable!()
        }
        fn step(&mut self, _: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
            unreachable!()
        }
        fn close(&mut self) -> Result<()> {
            drop(self.slot.lock().unwrap().take());
            self.done.send(()).unwrap();
            Ok(())
        }
    }
    let slot = Arc::new(Mutex::new(None));
    let (done, observed) = std::sync::mpsc::channel();
    let server = Arc::new(
        EnvServer::new(
            DropOnClose {
                slot: Arc::clone(&slot),
                done,
            },
            quick(),
        )
        .unwrap(),
    );
    let signal = server.shutdown_handle();
    server.start().unwrap();
    *slot.lock().unwrap() = Some(Arc::clone(&server));
    drop(server);
    signal.trigger("test last Arc from close");
    observed
        .recv_timeout(Duration::from_secs(5))
        .expect("callback Drop joined its own server");
    assert!(slot.lock().unwrap().is_none());
}

#[test]
fn vector_callback_can_drop_the_last_server_arc() {
    struct DropOnStep {
        slot: Arc<Mutex<Option<Arc<EnvServer>>>>,
        done: std::sync::mpsc::Sender<()>,
        pair: Pair,
    }
    impl VectorEnv for DropOnStep {
        fn num_envs(&self) -> u64 {
            self.pair.num_envs()
        }
        fn single_observation_space(&self) -> Space {
            self.pair.single_observation_space()
        }
        fn single_action_space(&self) -> Space {
            self.pair.single_action_space()
        }
        fn reset(
            &mut self,
            seed: Option<VectorSeed>,
            options: Option<&InfoDict>,
        ) -> Result<(Value, InfoDict)> {
            self.pair.reset(seed, options)
        }
        fn step(
            &mut self,
            actions: Value,
        ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
            drop(self.slot.lock().unwrap().take());
            self.done.send(()).unwrap();
            self.pair.step(actions)
        }
    }
    let slot = Arc::new(Mutex::new(None));
    let (done, observed) = std::sync::mpsc::channel();
    let server = Arc::new(
        EnvServer::vector(
            DropOnStep {
                slot: Arc::clone(&slot),
                done,
                pair: Pair {
                    num_envs: 2,
                    seen: Arc::new(Mutex::new(Vec::new())),
                },
            },
            quick(),
        )
        .unwrap(),
    );
    server.start().unwrap();
    let mut remote =
        RemoteVectorEnv::new(RemoteEnvOptions::default().address(server.address())).unwrap();
    remote.reset(None, None).unwrap();
    *slot.lock().unwrap() = Some(Arc::clone(&server));
    drop(server);
    let _ = remote.step(Value::List(vec![Value::Int(1), Value::Int(2)]));
    observed
        .recv_timeout(Duration::from_secs(5))
        .expect("vector callback Drop joined its own actor");
    assert!(slot.lock().unwrap().is_none());
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

#[test]
fn closed_client_refusal_keeps_its_idle_firing_pending() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    let token = remote.cancel_handle();
    remote.close().unwrap();
    token.cancel();
    assert_eq!(
        remote.reset(None, None).unwrap_err().class(),
        ErrorClass::RuntimeError
    );
    assert_eq!(
        remote.step(Value::Int(0)).unwrap_err().class(),
        ErrorClass::RuntimeError
    );
    assert_eq!(
        remote.render(None).unwrap_err().class(),
        ErrorClass::RuntimeError
    );
    assert!(token.take());
    assert!(!token.take());
    server.shutdown().unwrap();
}

#[test]
fn immediate_remote_completion_observes_handle_cancellation() {
    struct Fire {
        token: Arc<Mutex<Option<crate::ext::CancelToken>>>,
    }
    impl crate::Env for Fire {
        fn observation_space(&self) -> crate::spaces::Space {
            Discrete::new(4, DiscreteOptions::default()).unwrap().into()
        }
        fn action_space(&self) -> crate::spaces::Space {
            self.observation_space()
        }
        fn reset(
            &mut self,
            _: Option<i64>,
            _: Option<&crate::InfoDict>,
        ) -> crate::Result<(crate::Value, crate::InfoDict)> {
            Ok((crate::Value::Int(0), crate::InfoDict::new()))
        }
        fn step(
            &mut self,
            _: Option<crate::Value>,
        ) -> crate::Result<(crate::Value, f64, bool, bool, crate::InfoDict)> {
            if let Some(token) = self.token.lock().unwrap().take() {
                token.cancel();
            }
            Ok((
                crate::Value::Int(1),
                1.,
                false,
                false,
                crate::InfoDict::new(),
            ))
        }
    }
    let token = Arc::new(Mutex::new(None));
    let server = EnvServer::new(
        Fire {
            token: token.clone(),
        },
        quick(),
    )
    .unwrap();
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    remote.reset(None, None).unwrap();
    *token.lock().unwrap() = Some(remote.cancel_handle());
    assert_eq!(
        remote.step(crate::Value::Int(0)).unwrap_err(),
        crate::Error::Cancelled
    );
    remote.step(crate::Value::Int(0)).unwrap();
    remote.close().unwrap();
    server.shutdown().unwrap();
}
