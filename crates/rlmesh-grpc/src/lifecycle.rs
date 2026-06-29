//! Server lifecycle: serve options, idle/drain/close timeouts, the shutdown
//! trigger, and the in-flight activity tracking that drives idle shutdown.
//!
//! The env and model servers share this machinery. Each request is bracketed by
//! an [`IdleActivity::Started`] and an [`ActivityFinishedGuard`] that emits the
//! matching `Finished` on drop, so the in-flight count returns to zero even when
//! a handler panics; idle shutdown then fires once the count stays at zero for
//! `idle_timeout`.

use std::future::Future;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Transport lifecycle policy shared by the env and model servers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServeOptions {
    /// Honor a client `shutdown` RPC. When `false`, remote shutdown is rejected
    /// and the server stops only via its own idle/drain policy.
    pub allow_remote_shutdown: bool,
    /// Shut down after this much inactivity. `None` never times out.
    pub idle_timeout: Option<Duration>,
    /// Maximum time to drain in-flight requests on shutdown. `None` waits
    /// indefinitely.
    pub drain_timeout: Option<Duration>,
    /// Maximum time the env/handler close hook may take. `None` waits
    /// indefinitely.
    pub close_timeout: Option<Duration>,
    /// Optional bearer token required on the `authorization` gRPC metadata
    /// header for every request. `None` (the default) disables authentication.
    pub token: Option<String>,
    /// Maximum number of model Join-stream requests processed concurrently per
    /// connection (pipelined predict). `None` applies
    /// [`DEFAULT_PREDICT_CONCURRENCY`]. Per-route lifecycle ordering is always
    /// preserved regardless of the cap; this only bounds how many decode/encode
    /// and handler critical sections may overlap. Ignored by the env server.
    pub predict_concurrency: Option<usize>,
}

/// Default per-connection concurrency cap for pipelined model predict requests.
///
/// Bounds outstanding per-request tasks so a flood of requests cannot spawn
/// unboundedly. Decode/encode overlap up to this many requests; handler calls
/// still serialize behind the handler mutex (see the model server docs).
pub const DEFAULT_PREDICT_CONCURRENCY: usize = 32;

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdleActivity {
    Started,
    Finished,
}

/// RAII guard that emits [`IdleActivity::Finished`] when dropped, pairing the
/// [`IdleActivity::Started`] a server's read loop sends before dispatching a
/// request.
///
/// Both the env and model servers spawn each request on its own task. If that
/// task panics (e.g. a user `step`/`reset`/`predict` panics), tokio swallows the
/// unwind and any inline `Finished` send after the handler never runs — so the
/// idle-shutdown tracker's in-flight count stays elevated forever and a server
/// with an `idle_timeout` never shuts down (`wait_for_idle_shutdown` does a
/// timeout-free `recv().await` while `in_flight > 0`). Holding this guard inside
/// the spawned task guarantees the `Finished` fires on every exit path —
/// success, error, or panic.
#[doc(hidden)]
pub struct ActivityFinishedGuard(Option<mpsc::UnboundedSender<IdleActivity>>);

impl ActivityFinishedGuard {
    /// Create a guard that will send [`IdleActivity::Finished`] on drop.
    ///
    /// The caller is responsible for having already sent the paired
    /// [`IdleActivity::Started`]. A `None` sender (idle shutdown disabled) makes
    /// the guard a no-op.
    pub fn new(activity_tx: Option<mpsc::UnboundedSender<IdleActivity>>) -> Self {
        Self(activity_tx)
    }
}

impl Drop for ActivityFinishedGuard {
    fn drop(&mut self) {
        if let Some(activity_tx) = &self.0 {
            let _ = activity_tx.send(IdleActivity::Finished);
        }
    }
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ShutdownTrigger {
    token: CancellationToken,
    reason: Arc<Mutex<Option<String>>>,
    triggered: Arc<AtomicBool>,
}

impl ShutdownTrigger {
    #[doc(hidden)]
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            reason: Arc::new(Mutex::new(None)),
            triggered: Arc::new(AtomicBool::new(false)),
        }
    }

    #[doc(hidden)]
    pub fn trigger(&self, reason: impl Into<String>) -> bool {
        let first = !self.triggered.swap(true, Ordering::SeqCst);
        if first {
            *self.reason.lock().expect("shutdown reason lock poisoned") = Some(reason.into());
            self.token.cancel();
        }
        first
    }

    #[doc(hidden)]
    pub async fn cancelled(&self) {
        self.token.cancelled().await;
    }

    #[doc(hidden)]
    pub fn cancelled_owned(&self) -> impl std::future::Future<Output = ()> + Send + 'static {
        self.token.clone().cancelled_owned()
    }

    #[doc(hidden)]
    pub fn reason(&self) -> Option<String> {
        self.reason
            .lock()
            .expect("shutdown reason lock poisoned")
            .clone()
    }
}

#[doc(hidden)]
pub async fn wait_for_idle_shutdown(
    activity_rx: &mut mpsc::UnboundedReceiver<IdleActivity>,
    idle_timeout: Duration,
) {
    let mut in_flight = 0_usize;
    loop {
        let activity = if in_flight == 0 {
            match tokio::time::timeout(idle_timeout, activity_rx.recv()).await {
                Ok(Some(activity)) => activity,
                Ok(None) | Err(_) => return,
            }
        } else {
            match activity_rx.recv().await {
                Some(activity) => activity,
                None => return,
            }
        };

        match activity {
            IdleActivity::Started => in_flight = in_flight.saturating_add(1),
            IdleActivity::Finished => in_flight = in_flight.saturating_sub(1),
        }
    }
}

#[doc(hidden)]
pub fn start_idle_shutdown(
    idle_timeout: Option<Duration>,
    shutdown: ShutdownTrigger,
) -> Option<mpsc::UnboundedSender<IdleActivity>> {
    let idle_timeout = idle_timeout?;
    let (tx, mut rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        wait_for_idle_shutdown(&mut rx, idle_timeout).await;
        shutdown.trigger("idle timeout");
    });
    Some(tx)
}

#[doc(hidden)]
pub async fn await_server_shutdown<F>(
    server: F,
    shutdown: ShutdownTrigger,
    drain_timeout: Option<Duration>,
) -> Result<(), tonic::transport::Error>
where
    F: Future<Output = Result<(), tonic::transport::Error>>,
{
    tokio::pin!(server);
    tokio::select! {
        result = server.as_mut() => result,
        _ = shutdown.cancelled() => {
            if let Some(drain_timeout) = drain_timeout {
                match tokio::time::timeout(drain_timeout, server.as_mut()).await {
                    Ok(result) => result,
                    Err(_) => Ok(()),
                }
            } else {
                server.as_mut().await
            }
        }
    }
}

#[doc(hidden)]
pub async fn await_close_with_timeout<F, T>(
    close: F,
    close_timeout: Option<Duration>,
) -> Result<T, Duration>
where
    F: Future<Output = T>,
{
    match close_timeout {
        Some(close_timeout) => tokio::time::timeout(close_timeout, close)
            .await
            .map_err(|_| close_timeout),
        None => Ok(close.await),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_options_default_disables_remote_shutdown() {
        assert_eq!(
            ServeOptions::default(),
            ServeOptions {
                allow_remote_shutdown: false,
                idle_timeout: None,
                drain_timeout: None,
                close_timeout: None,
                token: None,
                predict_concurrency: None,
            }
        );
    }

    #[tokio::test]
    async fn idle_shutdown_triggers_after_quiet_window() {
        let shutdown = ShutdownTrigger::new();
        let _tx = start_idle_shutdown(Some(Duration::from_millis(10)), shutdown.clone()).unwrap();

        tokio::time::timeout(Duration::from_millis(250), shutdown.cancelled())
            .await
            .unwrap();
        assert_eq!(shutdown.reason().as_deref(), Some("idle timeout"));
    }

    #[tokio::test]
    async fn idle_shutdown_does_not_fire_while_request_is_in_flight() {
        let shutdown = ShutdownTrigger::new();
        let tx = start_idle_shutdown(Some(Duration::from_millis(10)), shutdown.clone()).unwrap();

        tx.send(IdleActivity::Started)
            .expect("idle activity receiver should be open");
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!shutdown.token.is_cancelled());

        tx.send(IdleActivity::Finished)
            .expect("idle activity receiver should be open");
        tokio::time::timeout(Duration::from_millis(250), shutdown.cancelled())
            .await
            .expect("idle shutdown should fire after in-flight request finishes");
        assert_eq!(shutdown.reason().as_deref(), Some("idle timeout"));
    }

    #[tokio::test]
    async fn activity_finished_guard_sends_finished_on_drop() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        {
            let _guard = ActivityFinishedGuard::new(Some(tx));
            // Nothing sent yet while the guard is alive.
            assert!(rx.try_recv().is_err());
        }
        // Dropping the guard emits exactly one Finished.
        assert_eq!(rx.try_recv().ok(), Some(IdleActivity::Finished));
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn activity_finished_guard_pairs_started_even_on_panic() {
        // Model the spawned-request task panicking: the Started/Finished pair must
        // still net to zero in-flight so idle shutdown can fire afterward.
        let shutdown = ShutdownTrigger::new();
        let tx = start_idle_shutdown(Some(Duration::from_millis(10)), shutdown.clone()).unwrap();

        let task_tx = tx.clone();
        let handle = tokio::spawn(async move {
            let _guard = ActivityFinishedGuard::new(Some(task_tx.clone()));
            task_tx
                .send(IdleActivity::Started)
                .expect("idle activity receiver should be open");
            panic!("request handler panicked");
        });
        assert!(handle.await.is_err(), "task was expected to panic");

        // Despite the panic, the guard's Drop sent Finished, so the in-flight
        // count returns to zero and idle shutdown fires.
        tokio::time::timeout(Duration::from_millis(250), shutdown.cancelled())
            .await
            .expect("idle shutdown must fire after a panicking request's guard drops");
        assert_eq!(shutdown.reason().as_deref(), Some("idle timeout"));
    }

    #[tokio::test]
    async fn close_timeout_reports_timeout_duration() {
        let timeout = Duration::from_millis(5);
        let err = await_close_with_timeout(
            async {
                tokio::time::sleep(Duration::from_millis(100)).await;
            },
            Some(timeout),
        )
        .await
        .unwrap_err();
        assert_eq!(err, timeout);
    }
}
