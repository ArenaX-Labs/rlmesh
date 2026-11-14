//! Shutdown, drain and close: concurrent and reentrant shutdowns, async
//! drains, and callbacks that drop the last server handle.

use super::*;

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
