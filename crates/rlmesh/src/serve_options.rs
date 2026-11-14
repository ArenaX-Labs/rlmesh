//! Lifecycle and batching options for session servers.
use crate::{Error, Result};
use std::time::Duration;

/// Lifecycle policy for a session server.
#[derive(Debug, Clone, Default)]
pub struct ServeOptions {
    /// Maximum time the env or model close hook may take (default 5 s).
    pub close_timeout: Option<Duration>,
    /// The workflow edition this server declares.
    pub workflow_edition: Option<String>,
}
impl ServeOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.close_timeout.is_some_and(|t| t.is_zero()) {
            return Err(Error::invalid_argument(
                "close_timeout_seconds must be positive",
            ));
        }
        Ok(())
    }
}

/// Experimental (Python's `BatchOptions`). How a session-served model
/// coalesces the predicts it receives, from every session it answers in, into
/// one handler call (one fused forward for a
/// model that fuses). The model still runs one forward at a time: batching
/// raises fill, not concurrency.
///
/// A batch closes at `max_batch_size` groups or `batch_byte_budget`
/// observation bytes, or as soon as every route the model has resolved has a
/// request in it. While the model is idle it also closes `batch_window_ms`
/// after its first request; while a forward runs, it stays open until that
/// forward ends, so under load the window costs nothing.
/// A model that cannot fuse (`allow_fusion = false`, or no batched corner)
/// is not batched.
///
/// The defaults are: `max_batch_size` and `batch_window_ms`
/// as its request compiler sets them, the byte budget and in-flight cap as
/// its runner reads them. Build one with `..Default::default()`.
#[derive(Debug, Clone)]
pub struct BatchOptions {
    /// The most predict groups (one route's rows each) one handler call
    /// carries; 1 answers each request on its own, at once. A request with
    /// more groups runs as a batch of its own.
    pub max_batch_size: usize,
    /// The longest an idle model holds a batch open for more requests.
    pub batch_window_ms: u64,
    /// The most observation bytes one handler call carries. A request over
    /// it runs as a batch of its own.
    pub batch_byte_budget: usize,
    /// Batches past assembly at once: one on the model and the next
    /// assembling behind it at 2; 1 assembles only while the model is idle.
    pub max_in_flight_batches: usize,
    /// Append one JSON line per batch here, in the
    /// `telemetry/batches.jsonl` shape. `None` records nothing.
    pub batch_records: Option<std::path::PathBuf>,
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            batch_window_ms: 5,
            batch_byte_budget: 256 * 1024 * 1024 - 4 * 1024,
            max_in_flight_batches: 2,
            batch_records: None,
        }
    }
}
