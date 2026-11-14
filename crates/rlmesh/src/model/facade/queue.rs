//! The model's queue and lifecycle: the [`Shared`] internals every handle
//! holds, the FIFO of jobs its library thread runs, and close.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use crate::lifecycle::ShutdownTrigger;

use super::calls::{Core, Resolved};
use super::erased::DynPolicy;
use crate::model::context::EpisodeStore;
use crate::{Error, Result};

type Job = Box<dyn FnOnce(&mut Core) + Send>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Open,
    Closing,
    Closed,
}

struct LifeState {
    phase: Phase,
    serves: std::collections::HashMap<u64, ShutdownTrigger>,
    next_serve: u64,
}

/// The model's lifecycle, shared with its library thread (which marks it
/// closed once the close hook returned).
pub(super) struct Life {
    state: Mutex<LifeState>,
    changed: Condvar,
}

impl Life {
    fn lock(&self) -> MutexGuard<'_, LifeState> {
        self.state.lock().unwrap_or_else(|err| err.into_inner())
    }

    pub(super) fn is_closed(&self) -> bool {
        self.lock().phase == Phase::Closed
    }

    fn set_closed(&self) {
        self.lock().phase = Phase::Closed;
        self.changed.notify_all();
    }

    /// Wait until `done` holds, or `timeout` passes; whether it held.
    fn wait_until(
        &self,
        timeout: Option<Duration>,
        mut done: impl FnMut(&LifeState) -> bool,
    ) -> bool {
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        let mut state = self.lock();
        while !done(&state) {
            match deadline {
                None => {
                    state = self
                        .changed
                        .wait(state)
                        .unwrap_or_else(|err| err.into_inner());
                }
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        return false;
                    }
                    state = self
                        .changed
                        .wait_timeout(state, deadline - now)
                        .unwrap_or_else(|err| err.into_inner())
                        .0;
                }
            }
        }
        true
    }
}

/// The model's shared internals: declarations, queue and lifecycle.
pub(crate) struct Shared {
    pub(crate) decl: Resolved,
    queue: mpsc::Sender<Job>,
    worker: ThreadId,
    pub(super) life: Arc<Life>,
    refuse: AtomicBool,
    /// Set while a run drives this model.
    running: Arc<AtomicBool>,
}

/// Holds a model's run slot; released on drop.
pub struct RunSlot(Arc<AtomicBool>);

impl RunSlot {
    pub(crate) fn claim(gate: Arc<AtomicBool>) -> Result<Self> {
        if gate.swap(true, Ordering::SeqCst) {
            return Err(Error::Runtime(
                "run() is already in progress on this model; a model runs one run at a time".into(),
            ));
        }
        Ok(Self(gate))
    }
}

impl Drop for RunSlot {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

fn closed_error() -> Error {
    Error::Runtime("the model is closed".to_string())
}

fn panic_error(payload: Box<dyn std::any::Any + Send>) -> Error {
    let message = payload
        .downcast_ref::<&str>()
        .map(|text| text.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a model call panicked".to_string());
    Error::Internal(format!("model call panicked: {message}"))
}

impl Shared {
    #[allow(clippy::type_complexity)]
    pub(super) fn start(
        policy: Box<dyn DynPolicy>,
        decl: Resolved,
        on_episode_end: Option<Box<dyn FnMut(&str) -> Result<()> + Send>>,
        on_close: Option<Box<dyn FnOnce() -> Result<()> + Send>>,
    ) -> Result<Arc<Shared>> {
        let mut core = Core {
            policy,
            corners: decl.corners,
            context: decl.context,
            store: EpisodeStore::from_env(),
            on_episode_end,
            on_close,
            closed: false,
        };
        let (queue, jobs) = mpsc::channel::<Job>();
        let worker = std::thread::Builder::new()
            .name("rlmesh-model".to_string())
            .spawn(move || {
                for job in jobs {
                    job(&mut core);
                }
            })
            .map_err(|err| Error::Internal(format!("failed to spawn the model thread: {err}")))?;
        let worker = worker.thread().id();
        Ok(Arc::new(Shared {
            decl,
            queue,
            worker,
            life: Arc::new(Life {
                state: Mutex::new(LifeState {
                    phase: Phase::Open,
                    serves: std::collections::HashMap::new(),
                    next_serve: 0,
                }),
                changed: Condvar::new(),
            }),
            refuse: AtomicBool::new(false),
            running: Arc::new(AtomicBool::new(false)),
        }))
    }

    pub(crate) fn on_worker(&self) -> bool {
        std::thread::current().id() == self.worker
    }

    /// Refuse a call from inside this model's own corner or hook.
    pub(crate) fn check_not_reentrant(&self, what: &str) -> Result<()> {
        if self.on_worker() {
            return Err(Error::Runtime(format!(
                "{what} re-entered its own Model from inside a corner or hook; a model's \
                 calls are serialized, so this would deadlock"
            )));
        }
        Ok(())
    }

    fn submit<R: Send + 'static>(
        &self,
        f: impl FnOnce(&mut Core) -> Result<R> + Send + 'static,
    ) -> Result<mpsc::Receiver<Result<R>>> {
        let (tx, rx) = mpsc::sync_channel(1);
        let work = crate::warnings::work_context();
        let job: Job = Box::new(move |core: &mut Core| {
            let out = std::panic::catch_unwind(AssertUnwindSafe(|| {
                crate::warnings::with_work(work, || {
                    crate::warnings::check_cancelled()?;
                    f(core)
                })
            }))
            .unwrap_or_else(|payload| Err(panic_error(payload)));
            let _ = tx.send(out);
        });
        self.queue
            .send(job)
            .map_err(|_| Error::Runtime("the model's call queue has stopped".to_string()))?;
        Ok(rx)
    }

    /// Run `f` on the model's thread, behind every call queued before it.
    pub(crate) fn call<R: Send + 'static>(
        &self,
        f: impl FnOnce(&mut Core) -> Result<R> + Send + 'static,
    ) -> Result<R> {
        self.check_not_reentrant("a corner call")?;
        let admission = self.life.lock();
        if self.refuse.load(Ordering::SeqCst) {
            return Err(closed_error());
        }
        let rx = self.submit(move |core| {
            if core.closed {
                return Err(closed_error());
            }
            f(core)
        })?;
        drop(admission);
        rx.recv()
            .map_err(|_| Error::Internal("the model call was dropped".to_string()))?
    }

    /// Claim the model's one run slot: a second run while one is in progress
    /// fails at once with the `RuntimeError` class.
    pub(crate) fn claim_run(self: &Arc<Self>) -> Result<RunSlot> {
        RunSlot::claim(self.running.clone())
    }

    /// Refuse a new run or serve once `close()` has begun.
    pub(crate) fn admit(&self) -> Result<()> {
        if self.life.lock().phase != Phase::Open {
            return Err(closed_error());
        }
        Ok(())
    }

    pub(super) fn register_serve(&self, trigger: ShutdownTrigger) -> Result<u64> {
        let mut state = self.life.lock();
        if state.phase != Phase::Open {
            return Err(closed_error());
        }
        let id = state.next_serve;
        state.next_serve += 1;
        state.serves.insert(id, trigger);
        Ok(id)
    }

    pub(super) fn deregister_serve(&self, id: u64) {
        self.life.lock().serves.remove(&id);
        self.life.changed.notify_all();
    }

    /// `close()`: stop every serve, wait for queued calls, fire the hook
    /// once. A second or concurrent close waits for the first and succeeds.
    pub(crate) fn close(&self, timeout: Option<Duration>) -> Result<()> {
        self.check_not_reentrant("close()")?;
        let triggers: Vec<ShutdownTrigger> = {
            let mut state = self.life.lock();
            match state.phase {
                Phase::Closed => return Ok(()),
                Phase::Closing => {
                    // Another close is running (a user close stopping this
                    // serve, or a detached Drop close): it owns the hook and
                    // its errors, so wait for it and succeed.
                    drop(state);
                    self.life
                        .wait_until(None, |state| state.phase == Phase::Closed);
                    return Ok(());
                }
                Phase::Open => {
                    state.phase = Phase::Closing;
                    state.serves.values().cloned().collect()
                }
            }
        };
        for trigger in triggers {
            trigger.trigger("model closed");
        }
        self.life.wait_until(None, |state| state.serves.is_empty());
        let admission = self.life.lock();
        self.refuse.store(true, Ordering::SeqCst);
        let life = Arc::clone(&self.life);
        let rx = match self.submit(move |core| {
            struct Complete(Arc<Life>);
            impl Drop for Complete {
                fn drop(&mut self) {
                    self.0.set_closed();
                }
            }
            let _complete = Complete(life);
            core.close_hook()
        }) {
            Ok(rx) => rx,
            Err(err) => {
                drop(admission);
                self.life.set_closed();
                return Err(err);
            }
        };
        drop(admission);
        match timeout {
            None => rx
                .recv()
                .map_err(|_| Error::Internal("the model close was dropped".to_string()))?,
            Some(timeout) => match rx.recv_timeout(timeout) {
                Ok(result) => result,
                Err(mpsc::RecvTimeoutError::Timeout) => Err(Error::Timeout(timeout)),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    Err(Error::Internal("the model close was dropped".to_string()))
                }
            },
        }
    }
}
