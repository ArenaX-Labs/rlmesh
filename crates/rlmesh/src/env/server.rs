//! One non-generic environment server for an env, a list of lane envs, or a
//! vector env.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::lifecycle::ShutdownTrigger;

use super::{Env, VectorEnv, signals};
use crate::adapters::EnvTags;
use crate::specs::EnvContract;
use crate::{Error, Result, ServeOptions};

mod build;
mod runner;
mod wrappers;

#[cfg(test)]
mod cancellation_tests;

use runner::{BackgroundCancel, CancelProbe, cleanup_ready, run};
pub(crate) use wrappers::Tagged;

/// The bound an env server puts on a drain or close the options leave unset.
const DEFAULT_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Options for constructing an environment server.
///
/// Defaults: the session address `:0/env`, no serve options or tags, and
/// `close_env_on_shutdown` true. Build one with `..Default::default()`;
/// appending a field is a minor change by project policy.
#[derive(Clone, Debug)]
pub struct EnvServerOptions {
    /// Session participant address; `None` binds `:0/env`.
    pub address: Option<String>,
    /// Transport lifecycle policy.
    pub options: Option<ServeOptions>,
    /// Tags to merge into each served env's metadata, validated against a
    /// scalar env's spaces.
    pub tags: Option<EnvTags>,
    /// Close the env when the server stops (default `true`); `false` for a
    /// server in front of an env its caller closes.
    pub close_env_on_shutdown: bool,
    /// Rust-only fields.
    pub ext: Option<crate::ext::EnvServerOptions>,
}

impl Default for EnvServerOptions {
    fn default() -> Self {
        EnvServerOptions {
            address: None,
            options: None,
            tags: None,
            close_env_on_shutdown: true,
            ext: None,
        }
    }
}

/// A factory serve's describe publication: the factory's full envelope, or
/// why it could not be built (step 5 is best-effort).
pub(crate) struct Publish {
    pub(crate) envelope: Result<String>,
}

/// What the server serves.
enum Served {
    Lanes(Vec<Box<dyn Env + Send>>),
    Vector(Box<dyn VectorEnv + Send>),
}

/// A bound robot participant in a session.
enum Listener {
    Lockstep(Box<crate::lockstep::LockstepBinding>),
}

impl Listener {
    fn bind(
        address: &str,
        runtime: &tokio::runtime::Runtime,
        contract: &crate::EnvContract,
    ) -> Result<(Self, String)> {
        if !crate::address::is_v2(address) {
            return Err(Error::invalid_argument(crate::address::V1_REFUSAL));
        }
        crate::lockstep::check(contract.to_wire()?)
            .map_err(|err| Error::invalid_argument(err.to_string()))?;
        let binding = std::thread::scope(|scope| {
            scope
                .spawn(|| runtime.block_on(crate::lockstep::bind(address)))
                .join()
        })
        .map_err(|_| Error::Internal("binding the lockstep env panicked".into()))?
        .map_err(|err| match err {
            crate::lockstep::GymError::Config(why) => Error::invalid_argument(why),
            other => Error::Connection(other.to_string()),
        })?;
        let bound = binding.address();
        Ok((Self::Lockstep(Box::new(binding)), bound))
    }

    fn release(self, runtime: &tokio::runtime::Runtime) {
        match self {
            // Off this thread: a failed serve_async start releases from
            // inside an executor.
            Self::Lockstep(binding) => std::thread::scope(|scope| {
                scope.spawn(|| runtime.block_on(binding.close()));
            }),
        }
    }
}

struct Resources {
    served: Served,
    runtime: tokio::runtime::Runtime,
    listener: Listener,
    options: ServeOptions,
    close_timeout: Duration,
    close_env_on_shutdown: bool,
    /// Announce the env once it answers (`None` for a run's own server).
    announce: Option<crate::binding::Announce>,
}

enum State {
    Ready(Box<Resources>),
    Starting,
    Foreground,
    Background(Arc<Drain>),
    Stopped,
}

struct Shared {
    id: u64,
    address: String,
    contract: EnvContract,
    binding_description: Mutex<Option<String>>,
    shutdown: ShutdownTrigger,
    background_cancel: Mutex<std::sync::Weak<CancelProbe>>,
    state: Mutex<State>,
    /// A serve thread nobody waits on any more (a dropped `serve_async`,
    /// `shutdown_async`), or a live serve_async's result-owned drain.
    /// The server's Drop joins it.
    detached: Mutex<Option<Arc<Drain>>>,
    abandoned: super::lanes::JoinThreads,
    /// Serialize only state transitions and handle publication, never a drain
    /// wait: a suspended future must not block synchronous observers.
    drain: futures::lock::Mutex<()>,
    /// Notify after start publication or any joined drain, including ready close.
    done: tokio::sync::watch::Sender<bool>,
}

/// One join operation, shared by observers without sharing result ownership.
struct Drain {
    result: Mutex<Option<Result<()>>>,
    done: tokio::sync::watch::Sender<bool>,
    joiner: Mutex<Option<JoinHandle<()>>>,
    serve_owned: AtomicBool,
}
impl Drain {
    fn new(
        handle: JoinHandle<Result<()>>,
        serve_owned: bool,
        notify: tokio::sync::watch::Sender<bool>,
    ) -> Arc<Self> {
        let (done, _) = tokio::sync::watch::channel(false);
        let drain = Arc::new(Self {
            result: Mutex::new(None),
            done,
            joiner: Mutex::new(None),
            serve_owned: AtomicBool::new(serve_owned),
        });
        let worker = Arc::clone(&drain);
        let joiner = std::thread::spawn(move || {
            *lock(&worker.result) = Some(join(handle));
            worker.done.send_replace(true);
            notify.send_replace(true);
        });
        *lock(&drain.joiner) = Some(joiner);
        drain
    }
    async fn wait(&self) {
        let mut done = self.done.subscribe();
        while !*done.borrow_and_update() {
            if done.changed().await.is_err() {
                break;
            }
        }
        if let Some(joiner) = lock(&self.joiner).take()
            && joiner.thread().id() != std::thread::current().id()
        {
            let _ = joiner.join();
        }
    }
    fn finished(&self) -> bool {
        *self.done.borrow()
    }
    fn result(&self, owner: bool) -> Result<()> {
        if owner || !self.serve_owned.load(Ordering::Acquire) {
            lock(&self.result).take().unwrap_or(Ok(()))
        } else {
            Ok(())
        }
    }
}

/// Serves an env, a list of lane envs, or a vector env.
///
/// The server binds at construction, so [`address`](Self::address) and
/// [`env_contract`](Self::env_contract) are readable before serving. Then
/// [`serve`](Self::serve) blocks with lane 0 (or the vector env) on the
/// calling thread, or [`start`](Self::start) serves from library threads and
/// [`wait`](Self::wait) / [`shutdown`](Self::shutdown) observe the stop.
/// Each lane runs on one thread for the server's life.
///
/// Dropping a server runs [`shutdown`](Self::shutdown) and discards its
/// error; call `shutdown()` to observe it. Inside an async runtime, Drop
/// triggers the drain and leaves the rest to the server's own threads.
/// Rust detaches close/actor threads that exceed their timeout or teardown
/// grace; only the C binding's free is an unbounded join point. A callback must
/// call shutdown directly rather than joining a helper thread that calls it
/// (a cross-thread join cycle).
pub struct EnvServer {
    shared: Arc<Shared>,
}

static NEXT_SERVER: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// The server whose env callback this thread is running, if any.
    static IN_CALLBACK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Run `body` marked as a callback of server `id` (so a `shutdown()` from it
/// only triggers the drain).
pub(crate) fn in_callback<R>(id: u64, body: impl FnOnce() -> R) -> R {
    struct Restore(u64);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_CALLBACK.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(IN_CALLBACK.with(|cell| cell.replace(id)));
    body()
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

/// Async-context condition: the thread is polling a tokio task or a library
/// driver segment.
pub(crate) fn in_async_context() -> bool {
    crate::run::drive::in_async_context()
}

/// A blocking call made from async code fails with the RuntimeError class
/// naming its `_async` twin, before it blocks.
pub(crate) fn refuse_in_async_context(call: &str) -> Result<()> {
    if in_async_context() {
        return Err(Error::Runtime(format!(
            "EnvServer::{call}() blocks; call {call}_async() from async code"
        )));
    }
    Ok(())
}

impl EnvServer {
    /// The bound session participant address.
    pub fn address(&self) -> &str {
        &self.shared.address
    }

    /// The contract this endpoint serves.
    pub fn env_contract(&self) -> &EnvContract {
        &self.shared.contract
    }

    /// Serve in the foreground until the server stops: lane 0 (or the vector
    /// env) runs on this thread, every other lane on its own. SIGINT stops the
    /// server and this returns the cancellation class after the drain and
    /// close; SIGTERM stops it and this returns `Ok`.
    pub fn serve(&self) -> Result<()> {
        refuse_in_async_context("serve")?;
        let resources = self.take_resources("serve", State::Foreground)?;
        let signals = signals::enter(self.shared.shutdown.clone());
        let result = run(*resources, &self.shared, true);
        self.set_stopped();
        let interrupted = signals.interrupted();
        drop(signals);
        match result {
            Err(err) => Err(err),
            Ok(()) if interrupted => Err(Error::Cancelled),
            Ok(()) => Ok(()),
        }
    }

    /// Serve from library threads and return at once.
    pub fn start(&self) -> Result<()> {
        self.start_subscribed(None)
    }

    pub(crate) fn start_subscribed(
        &self,
        subscription: Option<crate::ext::CancelSubscription>,
    ) -> Result<()> {
        let _drain = crate::run::drive::run_sync(self.shared.drain.lock());
        let handle = self.spawn_background("start", State::Starting, subscription)?;
        *lock(&self.shared.state) =
            State::Background(Drain::new(handle, false, self.shared.done.clone()));
        self.shared.done.send_replace(true);
        Ok(())
    }

    fn spawn_background(
        &self,
        action: &str,
        next: State,
        subscription: Option<crate::ext::CancelSubscription>,
    ) -> Result<JoinHandle<Result<()>>> {
        self.spawn_background_with_watcher(action, next, subscription, BackgroundCancel::start)
    }

    fn spawn_background_with_watcher(
        &self,
        action: &str,
        next: State,
        subscription: Option<crate::ext::CancelSubscription>,
        start_watcher: impl FnOnce(
            crate::ext::CancelSubscription,
            ShutdownTrigger,
        ) -> Result<BackgroundCancel>,
    ) -> Result<JoinHandle<Result<()>>> {
        let resources = self.take_resources(action, next)?;
        let watcher = match subscription
            .map(|subscription| start_watcher(subscription, self.shared.shutdown.clone()))
            .transpose()
        {
            Ok(watcher) => watcher,
            Err(error) => {
                self.shared.shutdown.trigger("watcher startup failed");
                self.set_stopped();
                let _ = cleanup_ready(*resources, &self.shared);
                self.shared.done.send_replace(true);
                return Err(error);
            }
        };
        if let Some(watcher) = &watcher {
            *lock(&self.shared.background_cancel) = Arc::downgrade(&watcher.probe);
        }
        let shared = Arc::clone(&self.shared);
        std::thread::Builder::new()
            .name("rlmesh-env-server".to_string())
            .spawn(move || {
                let result = run(*resources, &shared, false);
                background_outcome(result, &shared, watcher)
            })
            .map_err(|err| Error::Internal(format!("failed to spawn server thread: {err}")))
    }

    /// Wait for a started server to stop: `Ok(false)` when `timeout` passes
    /// first, `Ok(true)` once it has stopped. The call that observes the stop
    /// returns the serve error, once. `timeout` may be zero.
    pub fn wait(&self, timeout: Option<Duration>) -> Result<bool> {
        self.wait_outcome(timeout)?
            .transpose()
            .map(|stopped| stopped.is_some())
    }

    /// Keep a stopped server's serve result separate from wait misuse, so
    /// bindings can report the stopped flag even when serving failed.
    pub(crate) fn wait_outcome(&self, timeout: Option<Duration>) -> Result<Option<Result<()>>> {
        refuse_in_async_context("wait")?;
        let started = Instant::now();
        loop {
            if let Some(result) = self.poll_wait()? {
                return Ok(Some(result));
            }
            let nap = match timeout {
                Some(timeout) => {
                    let elapsed = started.elapsed();
                    if elapsed >= timeout {
                        return Ok(None);
                    }
                    (timeout - elapsed).min(Duration::from_millis(10))
                }
                None => Duration::from_millis(10),
            };
            std::thread::sleep(nap);
        }
    }

    /// `Some` once the started server has stopped (joining it), `None` while
    /// it runs; misuse is the RuntimeError class.
    fn poll_wait(&self) -> Result<Option<Result<()>>> {
        if IN_CALLBACK.with(std::cell::Cell::get) == self.shared.id {
            return Ok(if matches!(*lock(&self.shared.state), State::Stopped) {
                Some(Ok(()))
            } else {
                None
            });
        }
        let drain = {
            let state = lock(&self.shared.state);
            match &*state {
                State::Background(drain) => Arc::clone(drain),
                State::Starting => return Ok(None),
                State::Stopped => match lock(&self.shared.detached).clone() {
                    Some(drain) => drain,
                    None => return Ok(Some(Ok(()))),
                },
                State::Ready(_) => {
                    return Err(Error::Runtime("wait() called before start()".into()));
                }
                State::Foreground => {
                    return Err(Error::Runtime(
                        "wait() called while serve() is already running in the foreground".into(),
                    ));
                }
            }
        };
        if !drain.finished() {
            return Ok(None);
        }
        crate::run::drive::run_sync(drain.wait());
        // Retain the shared task if another observer already transitioned it.
        let mut state = lock(&self.shared.state);
        if matches!(*state, State::Background(_)) {
            *state = State::Stopped;
            *lock(&self.shared.detached) = Some(Arc::clone(&drain));
        }
        Ok(Some(drain.result(false)))
    }

    /// Stop the server: release a never-served listener (closing the envs if
    /// `close_env_on_shutdown`), or trigger the drain and join a started one,
    /// returning its serve error unless a `wait()` already returned it.
    /// Called from one of this server's own env callbacks, it only triggers
    /// the drain.
    pub fn shutdown(&self) -> Result<()> {
        refuse_in_async_context("shutdown")?;
        self.shared.trigger_shutdown("local shutdown");
        if IN_CALLBACK.with(std::cell::Cell::get) == self.shared.id {
            let mut state = lock(&self.shared.state);
            if matches!(*state, State::Foreground) {
                *state = State::Stopped;
            }
            return Ok(());
        }
        let drain = self.begin_shutdown();
        if let Some(drain) = drain {
            crate::run::drive::run_sync(drain.wait());
            drain.result(false)
        } else {
            Ok(())
        }
    }

    fn begin_shutdown(&self) -> Option<Arc<Drain>> {
        let _transition = crate::run::drive::run_sync(self.shared.drain.lock());
        let mut state = lock(&self.shared.state);
        let previous = std::mem::replace(&mut *state, State::Stopped);
        let drain = match previous {
            State::Ready(resources) => {
                let shared = Arc::clone(&self.shared);
                Some(Drain::new(
                    std::thread::spawn(move || cleanup_ready(*resources, &shared)),
                    false,
                    self.shared.done.clone(),
                ))
            }
            State::Background(drain) => Some(drain),
            State::Stopped => {
                return lock(&self.shared.detached)
                    .clone()
                    .filter(|drain| !drain.serve_owned.load(Ordering::Acquire));
            }
            State::Foreground | State::Starting => return None,
        };
        *lock(&self.shared.detached) = drain.clone();
        drain
    }

    /// The async twin of [`serve`](Self::serve): every lane, lane 0 included,
    /// runs on library threads and no signal handler is installed; resolves
    /// with the serve error when the server stops. Dropping it stops the
    /// server (the drain and close finish detached).
    pub async fn serve_async(&self) -> Result<()> {
        let drain = {
            let _transition = self.shared.drain.lock().await;
            let handle = self.spawn_background("serve", State::Foreground, None)?;
            let drain = Drain::new(handle, true, self.shared.done.clone());
            *lock(&self.shared.detached) = Some(Arc::clone(&drain));
            drain
        };
        let guard = StopOnDrop(&self.shared);
        drain.wait().await;
        std::mem::forget(guard);
        self.set_stopped();
        drain.result(true)
    }

    /// The async twin of [`wait`](Self::wait). Dropping it leaves the server
    /// as it was.
    pub async fn wait_async(&self, timeout: Option<Duration>) -> Result<bool> {
        let mut done = self.shared.done.subscribe();
        // At most one timer for the entire timeout; None creates no timer.
        let timer = timeout.map(sleep_off_runtime);
        futures::pin_mut!(timer);
        loop {
            // Snapshot before inspecting state, so a publication or completion
            // between the poll and changed().await cannot be missed.
            done.borrow_and_update();
            if let Some(result) = self.poll_wait()? {
                return result.map(|()| true);
            }
            match timer.as_mut().as_pin_mut() {
                Some(timer) => {
                    let changed = done.changed();
                    futures::pin_mut!(changed);
                    if matches!(
                        futures::future::select(changed, timer).await,
                        futures::future::Either::Right(_)
                    ) {
                        return Ok(false);
                    }
                }
                None => {
                    let _ = done.changed().await;
                }
            }
        }
    }

    /// The async twin of [`shutdown`](Self::shutdown). Dropping it leaves the
    /// drain it triggered to finish detached.
    pub async fn shutdown_async(&self) -> Result<()> {
        self.shared.trigger_shutdown("local shutdown");
        if IN_CALLBACK.with(std::cell::Cell::get) == self.shared.id {
            return Ok(());
        }
        // No transition lock is retained by a suspended future. A synchronous
        // observer can therefore join even while this future remains unpolled.
        if let Some(drain) = self.begin_shutdown() {
            drain.wait().await;
            drain.result(false)
        } else {
            Ok(())
        }
    }

    fn take_resources(&self, action: &str, next: State) -> Result<Box<Resources>> {
        let mut state = lock(&self.shared.state);
        match std::mem::replace(&mut *state, next) {
            State::Ready(resources) => Ok(resources),
            previous => {
                let message = match &previous {
                    State::Starting => {
                        format!("{action}() called while server is starting in the background")
                    }
                    State::Foreground => {
                        format!("{action}() called while server is already running")
                    }
                    State::Background(_) => format!(
                        "{action}() called while server is already running in the background"
                    ),
                    _ => format!("{action}() called after the server has been stopped"),
                };
                *state = previous;
                Err(Error::Runtime(message))
            }
        }
    }

    /// The trigger that stops this server without waiting (a factory
    /// serve's cancel).
    pub(crate) fn shutdown_handle(&self) -> ShutdownTrigger {
        self.shared.shutdown.clone()
    }

    /// C free's unbounded join point, after the binding released its borrows.
    pub(crate) fn join_abandoned(&self) {
        if IN_CALLBACK.with(std::cell::Cell::get) == self.shared.id {
            return;
        }
        let threads = std::mem::take(&mut *lock(&self.shared.abandoned));
        for thread in threads {
            if thread.thread().id() != std::thread::current().id() {
                let _ = thread.join();
            }
        }
    }

    fn set_stopped(&self) {
        *lock(&self.shared.state) = State::Stopped;
    }
}

fn background_outcome(
    result: Result<()>,
    shared: &Shared,
    watcher: Option<BackgroundCancel>,
) -> Result<()> {
    // Early setup errors can return before drive() triggers shutdown. Select
    // their stop before retiring the watcher so a later firing cannot mask them.
    shared.shutdown.trigger("serve ended");
    if watcher.is_some_and(BackgroundCancel::stop) {
        Err(Error::Cancelled)
    } else {
        result
    }
}

/// A dropped `serve_async`: stop the server, as `shutdown()` does on a
/// foreground serve.
struct StopOnDrop<'a>(&'a Shared);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        if let Some(drain) = lock(&self.0.detached).as_ref() {
            drain.serve_owned.store(false, Ordering::Release);
        }
        self.0.trigger_shutdown("dropped serve_async");
        *lock(&self.0.state) = State::Stopped;
    }
}

impl Drop for EnvServer {
    fn drop(&mut self) {
        self.shared.trigger_shutdown("drop");
        let previous = std::mem::replace(&mut *lock(&self.shared.state), State::Stopped);
        let detached = lock(&self.shared.detached).take();
        if in_async_context() || IN_CALLBACK.with(std::cell::Cell::get) == self.shared.id {
            // Never join a callback's own drain, directly or via its joiner.
            // ENV-10: never block an executor; the server's own threads finish.
            if let State::Ready(resources) = previous {
                let shared = Arc::clone(&self.shared);
                std::thread::spawn(move || {
                    let _ = cleanup_ready(*resources, &shared);
                });
            }
            return;
        }
        let _ = match previous {
            State::Ready(resources) => cleanup_ready(*resources, &self.shared),
            State::Background(drain) => {
                crate::run::drive::run_sync(drain.wait());
                Ok(())
            }
            _ => Ok(()),
        };
        if let Some(drain) = detached {
            crate::run::drive::run_sync(drain.wait());
        }
    }
}

impl fmt::Display for EnvServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EnvServer(address='{}')", self.shared.address)
    }
}

impl fmt::Debug for EnvServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// A sleep that needs no runtime timer (the twin runs on the caller's
/// executor, whatever it is).
fn sleep_off_runtime(duration: Duration) -> impl std::future::Future<Output = ()> {
    let (tx, rx) = futures::channel::oneshot::channel::<()>();
    std::thread::spawn(move || {
        std::thread::sleep(duration);
        let _ = tx.send(());
    });
    async move {
        let _ = rx.await;
    }
}

fn join(handle: JoinHandle<Result<()>>) -> Result<()> {
    if handle.thread().id() == std::thread::current().id() {
        return Ok(());
    }
    handle.join().unwrap_or_else(|_| {
        Err(Error::Internal(
            "serve failed: the server thread panicked".into(),
        ))
    })
}
