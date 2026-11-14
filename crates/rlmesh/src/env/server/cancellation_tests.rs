//! Cancellation of a server's startup, serve and shutdown.

use super::*;
use crate::ext::CancelToken;
use std::sync::atomic::{AtomicUsize, Ordering};

struct ClosingEnv {
    token: CancelToken,
    closes: Arc<AtomicUsize>,
    cancel_on_close: bool,
    fail_close: bool,
    close_gate: Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>,
}
impl Env for ClosingEnv {
    fn observation_space(&self) -> crate::spaces::Space {
        crate::spaces::Discrete::new(2, Default::default())
            .unwrap()
            .into()
    }
    fn action_space(&self) -> crate::spaces::Space {
        self.observation_space()
    }
    fn reset(
        &mut self,
        _: Option<i64>,
        _: Option<&crate::InfoDict>,
    ) -> Result<(crate::Value, crate::InfoDict)> {
        Ok((0.into(), Default::default()))
    }
    fn step(
        &mut self,
        _: Option<crate::Value>,
    ) -> Result<(crate::Value, f64, bool, bool, crate::InfoDict)> {
        Ok((0.into(), 0.0, true, false, Default::default()))
    }
    fn close(&mut self) -> Result<()> {
        self.closes.fetch_add(1, Ordering::SeqCst);
        if let Some((entered, release)) = self.close_gate.take() {
            entered.send(()).unwrap();
            release.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        if self.cancel_on_close {
            self.token.cancel();
            std::thread::sleep(Duration::from_millis(60));
        }
        if self.fail_close {
            Err(Error::Internal("close failed".into()))
        } else {
            Ok(())
        }
    }
}
fn server(
    token: &CancelToken,
    cancel_on_close: bool,
    fail_close: bool,
) -> (EnvServer, Arc<AtomicUsize>) {
    let closes = Arc::new(AtomicUsize::new(0));
    let server = EnvServer::new(
        ClosingEnv {
            token: token.clone(),
            closes: Arc::clone(&closes),
            cancel_on_close,
            fail_close,
            close_gate: None,
        },
        EnvServerOptions {
            address: Some(":0/env".into()),
            options: Some(crate::ServeOptions {
                workflow_edition: Some(crate::current_workflow_edition().into()),
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .unwrap();
    (server, closes)
}
#[test]
fn subscribed_start_samples_before_local_shutdown_and_reports_once() {
    let token = CancelToken::new();
    let (server, closes) = server(&token, false, false);
    server.start_subscribed(Some(token.subscribe())).unwrap();
    token.cancel();
    assert!(matches!(server.shutdown(), Err(Error::Cancelled)));
    assert!(server.wait(Some(Duration::ZERO)).unwrap());
    server.shutdown().unwrap();
    assert_eq!(closes.load(Ordering::SeqCst), 1);
    token.cancel();
    assert!(token.take(), "terminal path releases its subscription");
}
#[test]
fn subscribed_cancellation_joins_a_shared_drain_and_reports_once() {
    use std::future::Future;
    use std::sync::mpsc;

    let token = CancelToken::new();
    let closes = Arc::new(AtomicUsize::new(0));
    let (entered, closing) = mpsc::channel();
    let (release, held) = mpsc::channel();
    let server = Arc::new(
        EnvServer::new(
            ClosingEnv {
                token: token.clone(),
                closes: Arc::clone(&closes),
                cancel_on_close: false,
                fail_close: false,
                close_gate: Some((entered, held)),
            },
            EnvServerOptions::default(),
        )
        .unwrap(),
    );
    server.start_subscribed(Some(token.subscribe())).unwrap();
    token.cancel();
    let mut shutdown = Box::pin(server.shutdown_async());
    let waker = futures::task::noop_waker();
    assert!(
        shutdown
            .as_mut()
            .poll(&mut std::task::Context::from_waker(&waker))
            .is_pending()
    );
    closing.recv_timeout(Duration::from_secs(5)).unwrap();
    let (completed, observed) = mpsc::channel();
    let second = {
        let server = Arc::clone(&server);
        std::thread::spawn(move || completed.send(server.shutdown()).unwrap())
    };
    let early = observed.recv_timeout(Duration::from_millis(50));
    let polled = server.wait(Some(Duration::ZERO));
    release.send(()).unwrap();
    second.join().unwrap();
    let sync_result = observed.recv_timeout(Duration::from_secs(5)).unwrap();
    let async_result = futures::executor::block_on(shutdown);
    assert!(matches!(early, Err(mpsc::RecvTimeoutError::Timeout)));
    assert!(!polled.unwrap());
    assert!(matches!(sync_result, Err(Error::Cancelled)));
    async_result.unwrap();
    assert!(server.wait(Some(Duration::ZERO)).unwrap());
    server.shutdown().unwrap();
    assert_eq!(closes.load(Ordering::SeqCst), 1);
    token.cancel();
    assert!(token.take(), "joined drain releases its subscription");
}
#[test]
fn local_stop_ignores_cancellation_during_close() {
    let token = CancelToken::new();
    let (server, closes) = server(&token, true, false);
    server.start_subscribed(Some(token.subscribe())).unwrap();
    server.shutdown().unwrap();
    assert_eq!(closes.load(Ordering::SeqCst), 1);
    assert!(!token.take(), "late firing is consumed through cleanup");
    token.cancel();
    assert!(token.take());
}
#[test]
fn late_cancellation_does_not_replace_a_close_error() {
    let token = CancelToken::new();
    let (server, _) = server(&token, true, true);
    server.start_subscribed(Some(token.subscribe())).unwrap();
    assert!(matches!(server.shutdown(), Err(Error::Internal(_))));
}
#[test]
fn dropped_subscribed_server_closes_and_releases_subscription() {
    let token = CancelToken::new();
    let (server, closes) = server(&token, true, false);
    server.start_subscribed(Some(token.subscribe())).unwrap();
    drop(server);
    assert_eq!(closes.load(Ordering::SeqCst), 1);
    assert!(!token.take());
    token.cancel();
    assert!(token.take());
}
#[test]
fn drop_samples_a_firing_before_its_local_stop() {
    let token = CancelToken::new();
    let (server, closes) = server(&token, false, false);
    server.start_subscribed(Some(token.subscribe())).unwrap();
    let trigger = server.shutdown_handle();
    token.cancel();
    drop(server);
    assert_eq!(trigger.reason().as_deref(), Some("cancelled"));
    assert_eq!(closes.load(Ordering::SeqCst), 1);
    token.cancel();
    assert!(token.take());
}
#[test]
fn refused_subscribed_start_releases_only_its_own_subscription() {
    let token = CancelToken::new();
    let other = CancelToken::new();
    let (server, _) = server(&token, false, false);
    server.start_subscribed(Some(token.subscribe())).unwrap();
    assert!(server.start_subscribed(Some(other.subscribe())).is_err());
    other.cancel();
    assert!(other.take());
    token.cancel();
    assert!(matches!(server.shutdown(), Err(Error::Cancelled)));
}
#[test]
fn background_cancel_records_only_the_winning_trigger() {
    for cancel_first in [false, true] {
        let token = CancelToken::new();
        let trigger = ShutdownTrigger::new();
        let watcher = BackgroundCancel::start(token.subscribe(), trigger.clone()).unwrap();
        if !cancel_first {
            assert!(trigger.trigger("remote shutdown"));
        }
        token.cancel();
        watcher.probe.sample();
        assert_eq!(watcher.stop(), cancel_first);
        assert_eq!(
            trigger.reason().as_deref(),
            Some(if cancel_first {
                "cancelled"
            } else {
                "remote shutdown"
            })
        );
        assert!(!token.take());
        token.cancel();
        assert!(token.take());
    }
}
#[test]
fn early_serve_error_is_not_replaced_while_retiring_the_watcher() {
    use std::sync::atomic::AtomicBool;
    let token = CancelToken::new();
    let (server, _) = server(&token, false, false);
    let stop = Arc::new(AtomicBool::new(false));
    let probe = Arc::new(CancelProbe {
        subscription: token.subscribe(),
        trigger: server.shutdown_handle(),
        cancelled: Mutex::new(false),
    });
    let (stopping, watched) = (Arc::clone(&stop), Arc::clone(&probe));
    let thread = std::thread::spawn(move || {
        while !stopping.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        token.cancel();
        watched.sample();
    });
    let watcher = BackgroundCancel {
        stop,
        probe,
        thread: Some(thread),
    };
    let result = background_outcome(
        Err(Error::Internal("injected early serve failure".into())),
        &server.shared,
        Some(watcher),
    );
    assert!(
        matches!(result, Err(Error::Internal(message)) if message == "injected early serve failure")
    );
    assert_eq!(
        server.shared.shutdown.reason().as_deref(),
        Some("serve ended")
    );
}
#[test]
fn watcher_startup_failure_closes_ready_envs_and_notifies_waiters() {
    let token = CancelToken::new();
    let (server, closes) = server(&token, false, true);
    let mut done = server.shared.done.subscribe();
    assert!(!*done.borrow_and_update());
    let result = server.spawn_background_with_watcher(
        "start",
        State::Starting,
        Some(token.subscribe()),
        |_, _| Err(Error::Internal("injected watcher spawn failure".into())),
    );
    assert!(
        matches!(result, Err(Error::Internal(message)) if message == "injected watcher spawn failure")
    );
    assert_eq!(closes.load(Ordering::SeqCst), 1);
    assert!(done.has_changed().unwrap(), "wake wait_async after cleanup");
    assert!(*done.borrow_and_update());
    assert!(server.wait(Some(Duration::ZERO)).unwrap());
    server.shutdown().unwrap();
    assert!(server.start().is_err());
    token.cancel();
    assert!(token.take(), "failed startup releases its subscription");
}
