//! Retry/deadline connect helper.
//!
//! "Connect and retry until the server is accepting connections" is otherwise
//! hand-rolled (poll-connect loops) across the env client, model client, and
//! their embedders. [`retry_connect`] centralizes the deadline + backoff +
//! cancellation policy so those call sites collapse to a single call.

use std::future::Future;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::error::{Error as GrpcError, TransportError};

/// Policy for [`retry_connect`] and the client `connect_with_retry` helpers.
#[derive(Clone, Default)]
pub struct ConnectOptions {
    /// Overall time budget. `None` retries until success or cancellation.
    pub deadline: Option<Duration>,
    /// Delay between attempts. Zero falls back to a small default.
    pub backoff: Duration,
    /// Optional cancellation token; when cancelled, the retry loop aborts.
    pub cancellation: Option<CancellationToken>,
}

impl ConnectOptions {
    /// Options with an overall deadline and the default backoff.
    pub fn with_deadline(deadline: Duration) -> Self {
        Self {
            deadline: Some(deadline),
            ..Self::default()
        }
    }

    /// Set the per-attempt backoff.
    pub fn backoff(mut self, backoff: Duration) -> Self {
        self.backoff = backoff;
        self
    }

    /// Set the cancellation token.
    pub fn cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }
}

const DEFAULT_BACKOFF: Duration = Duration::from_millis(50);

/// Repeatedly invoke `attempt` until it succeeds, the deadline elapses, or the
/// cancellation token fires.
///
/// A successful attempt is returned immediately. On failure the error is
/// retried (after `backoff`) until the deadline; the last error is returned if
/// the deadline elapses first. Cancellation returns a [`GrpcError::Cancelled`].
pub async fn retry_connect<F, Fut, T>(
    options: &ConnectOptions,
    mut attempt: F,
) -> Result<T, GrpcError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, GrpcError>>,
{
    let backoff = if options.backoff.is_zero() {
        DEFAULT_BACKOFF
    } else {
        options.backoff
    };
    let deadline = options
        .deadline
        .map(|budget| tokio::time::Instant::now() + budget);

    // Records the most recent attempt error so a deadline expiry can surface a
    // real connection failure rather than a synthetic timeout. Seeded for the
    // case where the very first attempt is clipped before it ever fails.
    let mut last_error: GrpcError =
        TransportError::ConnectFailed("connect deadline exceeded".to_string()).into();

    loop {
        if let Some(token) = &options.cancellation
            && token.is_cancelled()
        {
            return Err(GrpcError::Cancelled("connect cancelled".to_string()));
        }

        // Clip each attempt to the caller's remaining budget.
        let remaining = match deadline {
            Some(deadline) => {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    return Err(last_error);
                }
                Some(remaining)
            }
            None => None,
        };

        // Race the attempt against cancellation so a hanging attempt aborts
        // promptly; without a deadline this is the only thing that can interrupt
        // an attempt that never resolves.
        let attempt_result = match (&options.cancellation, remaining) {
            (Some(token), Some(remaining)) => tokio::select! {
                _ = token.cancelled() => {
                    return Err(GrpcError::Cancelled("connect cancelled".to_string()));
                }
                result = tokio::time::timeout(remaining, attempt()) => match result {
                    // The attempt outran the deadline; surface the last error.
                    Err(_elapsed) => return Err(last_error),
                    Ok(result) => result,
                },
            },
            (Some(token), None) => tokio::select! {
                _ = token.cancelled() => {
                    return Err(GrpcError::Cancelled("connect cancelled".to_string()));
                }
                result = attempt() => result,
            },
            (None, Some(remaining)) => match tokio::time::timeout(remaining, attempt()).await {
                Err(_elapsed) => return Err(last_error),
                Ok(result) => result,
            },
            (None, None) => attempt().await,
        };

        last_error = match attempt_result {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };

        // Out of time? Surface the last error.
        if let Some(deadline) = deadline
            && tokio::time::Instant::now() + backoff >= deadline
        {
            return Err(last_error);
        }

        match &options.cancellation {
            Some(token) => {
                tokio::select! {
                    _ = tokio::time::sleep(backoff) => {}
                    _ = token.cancelled() => {
                        return Err(GrpcError::Cancelled("connect cancelled".to_string()));
                    }
                }
            }
            None => tokio::time::sleep(backoff).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[tokio::test]
    async fn retries_until_success() {
        let attempts = AtomicUsize::new(0);
        let result = retry_connect(
            &ConnectOptions::default().backoff(Duration::from_millis(1)),
            || {
                let n = attempts.fetch_add(1, Ordering::SeqCst);
                async move {
                    if n < 2 {
                        Err(TransportError::ConnectFailed("not yet".to_string()).into())
                    } else {
                        Ok(n)
                    }
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(result, 2);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn surfaces_last_error_after_deadline() {
        let result: Result<(), GrpcError> = retry_connect(
            &ConnectOptions::with_deadline(Duration::from_millis(15))
                .backoff(Duration::from_millis(10)),
            || async { Err(TransportError::ConnectFailed("down".to_string()).into()) },
        )
        .await;
        let error = result.unwrap_err();
        assert!(error.to_string().contains("down"), "got: {error}");
    }

    #[tokio::test]
    async fn deadline_clips_a_hanging_attempt() {
        // A SYN-blackholing address makes each attempt hang for the endpoint's
        // fixed connect timeout (longer than the caller's budget). The deadline
        // must still be honored mid-attempt rather than only between attempts:
        // a never-resolving attempt must not outlast the caller's budget.
        let started = tokio::time::Instant::now();
        let result: Result<(), GrpcError> = retry_connect(
            &ConnectOptions::with_deadline(Duration::from_millis(50)),
            // Never resolves; without mid-attempt clipping this hangs forever.
            std::future::pending,
        )
        .await;

        let error = result.unwrap_err();
        // The deadline surfaces a connect failure (not a hang), and we returned
        // at the deadline rather than waiting on the unbounded attempt.
        assert!(
            matches!(
                error,
                GrpcError::Transport(TransportError::ConnectFailed(_))
            ),
            "got: {error}"
        );
        // Generous real-time bound: the 50ms deadline must be honored well
        // before any fixed per-attempt connect timeout (seconds) would elapse.
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "retry_connect waited {:?}, past the 50ms deadline",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn cancellation_aborts_retry() {
        let token = CancellationToken::new();
        token.cancel();
        let result: Result<(), GrpcError> =
            retry_connect(&ConnectOptions::default().cancellation(token), || async {
                Err(TransportError::ConnectFailed("down".to_string()).into())
            })
            .await;
        assert!(matches!(result, Err(GrpcError::Cancelled(_))));
    }

    #[tokio::test]
    async fn cancellation_aborts_a_hanging_attempt_without_deadline() {
        // A never-resolving attempt with no deadline: only the cancellation
        // token can interrupt it. Cancelling mid-attempt must abort retry_connect
        // rather than waiting for the (never-arriving) attempt to return.
        let token = CancellationToken::new();
        let canceller = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            canceller.cancel();
        });

        let started = tokio::time::Instant::now();
        let result: Result<(), GrpcError> = retry_connect(
            &ConnectOptions::default().cancellation(token),
            // Never resolves; without mid-attempt cancellation this hangs forever.
            std::future::pending,
        )
        .await;

        let error = result.unwrap_err();
        assert!(matches!(error, GrpcError::Cancelled(_)), "got: {error}");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "retry_connect waited {:?}, past cancellation",
            started.elapsed()
        );
    }
}
