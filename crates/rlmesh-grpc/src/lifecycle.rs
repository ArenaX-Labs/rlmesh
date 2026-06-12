use std::future::Future;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ServeOptions {
    pub allow_remote_shutdown: bool,
    pub idle_timeout: Option<Duration>,
    pub drain_timeout: Option<Duration>,
    pub close_timeout: Option<Duration>,
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdleActivity {
    Started,
    Finished,
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
