//! Driving a borrowed future on the caller.
//!
//! A run or session call is one future over owned (`'static`) state.
//! Whatever it needs from the caller's borrows -- a local env's `reset` /
//! `step`, a run hook -- it sends as a [`Job`] through a [`CallerPort`] and
//! awaits. The sync entry points poll the future on the calling thread in
//! segments (the library's own parking executor, inside the library
//! runtime's `Handle::enter`, never tokio's `block_on`) and run each job
//! between segments, outside any async execution context, so a hook or an
//! env method may make sync rlmesh calls. The `_async` twins run the jobs
//! inside their own future, with the driving flag set.

use std::cell::Cell;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use crate::{Env, Error, Result, RunHooks, VectorEnv};

/// The library runtime runs and sessions poll under: its workers drive I/O
/// and timers while a sync caller parks.
pub(crate) fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("rlmesh-run")
            .build()
            .expect("failed to create the rlmesh run runtime")
    })
}

thread_local! {
    /// Set while this thread polls a library driver segment or handles a job
    /// inside an `_async` twin.
    static DRIVING: Cell<bool> = const { Cell::new(false) };
}

/// Whether this thread is driving a library future right now.
pub(crate) fn driving() -> bool {
    DRIVING.with(Cell::get)
}

struct Driving(bool);

impl Driving {
    fn set() -> Driving {
        Driving(DRIVING.with(|cell| cell.replace(true)))
    }
}

impl Drop for Driving {
    fn drop(&mut self) {
        DRIVING.with(|cell| cell.set(self.0));
    }
}

/// A local env a run or session borrows from its caller.
pub(crate) enum LocalEnv<'c> {
    Env(&'c mut (dyn Env + Send + 'c)),
    Vector(&'c mut (dyn VectorEnv + Send + 'c)),
}

/// What jobs reach on the caller's side.
pub(crate) struct CallerSide<'c> {
    pub(crate) env: Option<LocalEnv<'c>>,
    pub(crate) hooks: Option<&'c mut (dyn RunHooks + 'c)>,
}

impl CallerSide<'_> {
    /// The borrowed scalar env.
    pub(crate) fn env(&mut self) -> Result<&mut (dyn Env + Send + '_)> {
        match &mut self.env {
            Some(LocalEnv::Env(env)) => Ok(&mut **env),
            _ => Err(Error::Internal("no local env is lent to this call".into())),
        }
    }

    /// The borrowed vector env.
    pub(crate) fn vector(&mut self) -> Result<&mut (dyn VectorEnv + Send + '_)> {
        match &mut self.env {
            Some(LocalEnv::Vector(env)) => Ok(&mut **env),
            _ => Err(Error::Internal(
                "no local vector env is lent to this call".into(),
            )),
        }
    }
}

/// Shorten a lent hooks borrow to the call's.
pub(crate) fn lend_hooks<'c>(
    hooks: Option<&'c mut (dyn RunHooks + '_)>,
) -> Option<&'c mut (dyn RunHooks + 'c)> {
    match hooks {
        Some(hooks) => Some(hooks),
        None => None,
    }
}

/// One piece of work for the caller's side.
pub(crate) type Job = Box<dyn for<'x, 'c> FnOnce(&'x mut CallerSide<'c>) + Send>;

/// Sends jobs to the caller and waits for their results.
#[derive(Clone)]
pub(crate) struct CallerPort(UnboundedSender<Job>);

/// The caller's end of a [`CallerPort`].
pub(crate) type CallerQueue = UnboundedReceiver<Job>;

/// A port and its queue.
pub(crate) fn caller_channel() -> (CallerPort, CallerQueue) {
    let (tx, rx) = unbounded_channel();
    (CallerPort(tx), rx)
}

fn panicked(what: &str) -> Error {
    Error::Internal(format!("{what} panicked"))
}

fn gone() -> Error {
    Error::Runtime("the call that lent its env or hooks has returned".to_string())
}

impl CallerPort {
    /// Run `f` on the caller's side and await its result.
    pub(crate) async fn call<R: Send + 'static>(
        &self,
        f: impl for<'x, 'c> FnOnce(&'x mut CallerSide<'c>) -> Result<R> + Send + 'static,
    ) -> Result<R> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let job: Job = Box::new(move |side: &mut CallerSide<'_>| {
            if tx.is_closed() {
                return;
            }
            let out = catch_unwind(AssertUnwindSafe(|| f(side)))
                .unwrap_or_else(|_| Err(panicked("a lent env or hook call")));
            let _ = tx.send(out);
        });
        self.0.send(job).map_err(|_| gone())?;
        rx.await.unwrap_or_else(|_| Err(gone()))
    }

    /// The blocking form of [`call`](Self::call), for a plain library thread
    /// (an env server's lane).
    pub(crate) fn call_blocking<R: Send + 'static>(
        &self,
        f: impl for<'x, 'c> FnOnce(&'x mut CallerSide<'c>) -> Result<R> + Send + 'static,
    ) -> Result<R> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let job: Job = Box::new(move |side: &mut CallerSide<'_>| {
            let out = catch_unwind(AssertUnwindSafe(|| f(side)))
                .unwrap_or_else(|_| Err(panicked("a lent env call")));
            let _ = tx.send(out);
        });
        self.0.send(job).map_err(|_| gone())?;
        rx.recv().unwrap_or_else(|_| Err(gone()))
    }
}

struct Unpark(std::thread::Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

enum Segment<T> {
    Done(T),
    Job(Job),
}

/// Drive `future` on this thread in segments, running each job the future
/// sends between segments with the driving flag clear (the sync entry
/// points). The caller has checked it is not in an async context.
pub(crate) fn block_on_caller<F: Future>(
    future: F,
    queue: &mut CallerQueue,
    side: &mut CallerSide<'_>,
) -> F::Output {
    let _enter = runtime().enter();
    let mut future = pin!(future);
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        let segment = {
            let _driving = Driving::set();
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(out) => Some(Segment::Done(out)),
                Poll::Pending => match queue.poll_recv(&mut cx) {
                    Poll::Ready(Some(job)) => Some(Segment::Job(job)),
                    _ => None,
                },
            }
        };
        match segment {
            Some(Segment::Done(out)) => return out,
            Some(Segment::Job(job)) => job(side),
            None => std::thread::park(),
        }
    }
}

/// The `_async` form of [`block_on_caller`]: each poll enters the library
/// runtime and jobs run inside this future with the driving flag set.
pub(crate) async fn await_on_caller<F: Future>(
    future: F,
    queue: &mut CallerQueue,
    side: &mut CallerSide<'_>,
) -> F::Output {
    let mut future = pin!(future);
    std::future::poll_fn(move |cx| {
        let _enter = runtime().enter();
        let _driving = Driving::set();
        loop {
            if let Poll::Ready(out) = future.as_mut().poll(cx) {
                return Poll::Ready(out);
            }
            match queue.poll_recv(cx) {
                Poll::Ready(Some(job)) => job(side),
                _ => return Poll::Pending,
            }
        }
    })
    .await
}

/// Run `f` with the driving flag set: a callback an `_async` twin runs
/// inside its own future.
pub(crate) fn as_driving<R>(f: impl FnOnce() -> R) -> R {
    let _driving = Driving::set();
    f()
}

/// Run `f` with the driving flag clear: a sync entry point's caller-side
/// callback, which may make sync rlmesh calls of its own.
pub(crate) fn as_caller<R>(f: impl FnOnce() -> R) -> R {
    let _caller = Driving(DRIVING.with(|cell| cell.replace(false)));
    f()
}

/// Poll a sync entry point's outer future, which waits only inside its own
/// [`block_on_caller`] segments, on this thread without the driving flag.
pub(crate) fn run_sync<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        if let Poll::Ready(out) = future.as_mut().poll(&mut cx) {
            return out;
        }
        std::thread::park();
    }
}

/// Drive `future` to completion on this thread inside the library runtime's
/// context (a client's sync call): the runtime's workers drive its I/O and
/// timers while this thread parks. Never tokio's `block_on`, so it cannot
/// panic on nesting; the caller has checked it is not in an async context.
pub(crate) fn block_on_library<F: Future>(future: F) -> F::Output {
    let _enter = runtime().enter();
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        let polled = {
            let _driving = Driving::set();
            future.as_mut().poll(&mut cx)
        };
        if let Poll::Ready(out) = polled {
            return out;
        }
        std::thread::park();
    }
}

/// Poll `future` inside the library runtime's context on whichever executor
/// awaits it (a client's `_async` twin), so its timers and sockets register
/// with the library runtime.
pub(crate) async fn in_library<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    std::future::poll_fn(move |cx| {
        let _enter = runtime().enter();
        future.as_mut().poll(cx)
    })
    .await
}

/// Async-context condition: a tokio task (or `spawn_blocking` closure) or a
/// library driver segment is running on this thread.
pub(crate) fn in_async_context() -> bool {
    tokio::task::try_id().is_some() || driving()
}

/// Refuse a blocking call from async code, naming its twin.
pub(crate) fn refuse_in_async_context(call: &str) -> Result<()> {
    if in_async_context() {
        return Err(Error::Runtime(format!(
            "{call}() blocks; call {call}_async() from async code"
        )));
    }
    Ok(())
}

/// Watches cancellation tokens and cancels the driver when one fires.
pub(crate) struct CancelWatch {
    fired: Arc<std::sync::atomic::AtomicBool>,
    tokens: Arc<Vec<crate::ext::CancelSubscription>>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl CancelWatch {
    /// Fail at once when a token already fired; otherwise watch them.
    pub(crate) fn start(
        tokens: Vec<crate::ext::CancelToken>,
        driver: tokio_util::sync::CancellationToken,
    ) -> Result<CancelWatch> {
        Self::from_subscriptions(
            tokens
                .iter()
                .map(crate::ext::CancelToken::subscribe)
                .collect(),
            driver,
        )
    }

    /// Own the subscriptions captured at admission, without subscribing again.
    pub(crate) fn from_subscriptions(
        tokens: Vec<crate::ext::CancelSubscription>,
        driver: tokio_util::sync::CancellationToken,
    ) -> Result<CancelWatch> {
        let tokens = Arc::new(tokens);
        if tokens.iter().any(crate::ext::CancelSubscription::fired) {
            return Err(Error::Cancelled);
        }
        let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task = (!tokens.is_empty()).then(|| {
            let fired = Arc::clone(&fired);
            let tokens = Arc::clone(&tokens);
            runtime().spawn(async move {
                let mut tick = tokio::time::interval(Duration::from_millis(20));
                loop {
                    tick.tick().await;
                    if tokens.iter().any(crate::ext::CancelSubscription::fired) {
                        fired.store(true, std::sync::atomic::Ordering::SeqCst);
                        driver.cancel();
                        return;
                    }
                }
            })
        });
        Ok(CancelWatch {
            fired,
            tokens,
            task,
        })
    }

    pub(crate) fn subscriptions(&self) -> Arc<Vec<crate::ext::CancelSubscription>> {
        Arc::clone(&self.tokens)
    }

    /// Complete only after output processing. Checking and detaching each
    /// subscription is atomic with cancel(), so a last-moment firing cannot
    /// fall between the final check and unsubscribe.
    pub(crate) fn finish<T>(&self, result: Result<T>) -> Result<T> {
        let fired = self.tokens.iter().fold(
            self.fired.load(std::sync::atomic::Ordering::SeqCst),
            |fired, token| token.finish() || fired,
        );
        if fired { Err(Error::Cancelled) } else { result }
    }

    /// Whether a token fired during the call.
    pub(crate) fn fired(&self) -> bool {
        self.fired.load(std::sync::atomic::Ordering::SeqCst)
            || self
                .tokens
                .iter()
                .any(crate::ext::CancelSubscription::fired)
    }
}

impl Drop for CancelWatch {
    fn drop(&mut self) {
        for token in self.tokens.iter() {
            token.unsubscribe();
        }
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

impl CallerSide<'_> {
    pub(crate) fn hook<R>(
        &mut self,
        id: u64,
        call: impl FnOnce(&mut dyn RunHooks) -> Result<R>,
    ) -> Result<Option<R>> {
        let Self { env, hooks } = self;
        match hooks.as_deref_mut() {
            Some(hooks) => super::read::dispatch(id, env.as_mut(), || call(hooks)).map(Some),
            None => Ok(None),
        }
    }
}
