//! Run hooks and the context a run hands them (RUN-6, RUN-7, RUN-8).

use std::sync::{Arc, Mutex};

use super::{EpisodeResult, RunResult, StepEvent};
use crate::{EnvContract, Error, Result};

/// Observer callbacks for a run: Python's `RunHooks`. Every default is a
/// no-op; implement any subset.
///
/// A sync run calls hooks on the thread that called it, outside any async
/// execution context, so a hook may make sync rlmesh calls; under an
/// `_async` twin they run inside its future, where a blocking rlmesh call
/// fails with the `RuntimeError` class (RUN-8). A hook error aborts the run;
/// [`on_run_end`](RunHooks::on_run_end) still fires exactly once with the
/// completed episodes, and the run's own error wins over its error.
pub trait RunHooks: Send {
    /// Once per run, before the first episode, with the run's context.
    fn on_run_start(&mut self, _context: Arc<dyn RunContext>) -> Result<()> {
        Ok(())
    }

    /// After each episode's reset returns (`episode` is 0-based).
    fn on_episode_start(&mut self, _episode: i64, _seed: Option<i64>) -> Result<()> {
        Ok(())
    }

    /// After each env step.
    fn on_step(&mut self, _event: &StepEvent) -> Result<()> {
        Ok(())
    }

    /// With each completed episode's result.
    fn on_episode_end(&mut self, _result: &EpisodeResult) -> Result<()> {
        Ok(())
    }

    /// Exactly once when the run ends, also on an error or a cancel, with
    /// the (possibly partial) result.
    fn on_run_end(&mut self, _result: &RunResult) -> Result<()> {
        Ok(())
    }
}

/// What a run exposes to its hooks: Python's `RunContext` protocol.
///
/// Python's `read` and `frame_sources` members arrive with the adapters
/// surface's read items.
pub trait RunContext: Send + Sync {
    /// The connected env's contract; `None` before the env is connected.
    fn contract(&self) -> Result<Option<EnvContract>>;

    /// Lanes the connected env runs (1 before connect).
    fn num_envs(&self) -> Result<u64>;
}

/// The library's context: a native run's (valid for its whole life) or a
/// session's (failing once the session has closed, RUN-7).
pub(crate) struct LibraryContext {
    contract: Mutex<Option<EnvContract>>,
    closed: std::sync::atomic::AtomicBool,
}

impl LibraryContext {
    pub(crate) fn new() -> Arc<LibraryContext> {
        Arc::new(LibraryContext {
            contract: Mutex::new(None),
            closed: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub(crate) fn connect(&self, contract: EnvContract) {
        *self.contract.lock().unwrap_or_else(|err| err.into_inner()) = Some(contract);
    }

    pub(crate) fn close(&self) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn open(&self) -> Result<()> {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(Error::Runtime("session is closed".to_string()));
        }
        Ok(())
    }
}

impl RunContext for LibraryContext {
    fn contract(&self) -> Result<Option<EnvContract>> {
        self.open()?;
        Ok(self
            .contract
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone())
    }

    fn num_envs(&self) -> Result<u64> {
        Ok(self
            .contract()?
            .map_or(1, |contract| contract.num_envs().max(1)))
    }
}
