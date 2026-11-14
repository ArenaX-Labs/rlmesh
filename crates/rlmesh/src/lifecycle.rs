//! Shutdown and bounded-close helpers shared by session servers.
use std::future::Future;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

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

impl Default for ShutdownTrigger {
    fn default() -> Self {
        Self::new()
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
