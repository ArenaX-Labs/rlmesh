//! Many routes against one model in one process: each route is one env
//! target driven by the single-route run, beside the others. Routes start
//! together behind a [`StartBarrier`], retry their env connect, and end with
//! a verdict of their own: one route failing never stops its siblings.
//!
//! Ported from the managed runner (`managed_runtime/runner/runtime.rs`,
//! `start_barrier.rs` and `connections/retry.rs` in rlmesh-middleman). A
//! served model batches across routes on its own side, so each route joins it
//! from its own session; an in-process [`Model`](crate::Model) drives one route at a time.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::watch;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::native::{self, Driven, Mode};
use super::targets::ModelTarget;
use super::{EnvTarget, RunOptions, RunResult, drive};
use crate::{Error, Result};

/// How long the first connected route waits for the rest, as the managed
/// controller's default (`RLMESH_START_BARRIER_SECONDS`).
const START_BARRIER_TIMEOUT: Duration = Duration::from_secs(180);

/// The env connect deadline of a route, retries included: the runtime's
/// `env_connect_timeout` default.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// The first retry backoff, doubled per failed attempt up to
/// [`MAX_BACKOFF`].
const RETRY_BACKOFF: Duration = Duration::from_millis(100);
const MAX_BACKOFF: Duration = Duration::from_secs(2);

/// One route of [`run_routes`]: an env target, the run options it is driven
/// with, and how its reset seeds derive.
#[derive(Debug)]
pub struct Route<'a> {
    /// The env this route drives.
    pub env: EnvTarget<'a>,
    /// This route's options: its episode budget or seeds, step and time
    /// caps, execution horizon, first trial ordinal, hooks and cancellation.
    pub options: RunOptions<'a>,
    /// Derive each episode's reset seed from this base, the seed key and the
    /// episode's trial slot (`trial_index_base` + the route's slot), so
    /// routes that split one trial range draw the episodes one route would.
    /// Explicit `options.seeds` win.
    pub base_seed: Option<i64>,
    /// What the seeds key on besides the base and the slot; `None` keys on
    /// the env's own id, which every container of one env image shares.
    pub seed_key: Option<String>,
}

impl<'a> Route<'a> {
    /// A route over `env` with `options` and no seed base.
    pub fn new(env: impl Into<EnvTarget<'a>>, options: RunOptions<'a>) -> Self {
        Route {
            env: env.into(),
            options,
            base_seed: None,
            seed_key: None,
        }
    }
}

thread_local! {
    /// Set on a route's own thread, where its hooks and local env run.
    static IN_ROUTE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread is running a route of a [`run_routes`] call. Routes
/// started there would never see the outer call's cancellation or signals,
/// so the outer call could not finish: they are refused.
pub fn in_route() -> bool {
    IN_ROUTE.with(std::cell::Cell::get)
}

fn refuse_in_route() -> Result<()> {
    if in_route() {
        return Err(Error::Runtime(
            "run_routes: called from inside a running route (its env, model or hooks); \
             run the routes from the caller of the outer run_routes instead"
                .into(),
        ));
    }
    Ok(())
}

/// Drive `model` against every route at once and return each route's own
/// result, in route order: a route that fails holds its error while the
/// others run to completion. The outer error is one refused before any
/// route starts. An in-process [`Model`](crate::Model) drives one route; reach a shared
/// model at its served address (a [`RemoteModel`](crate::RemoteModel)) to
/// fan it out. Each route runs on a thread of its own, so its hooks and
/// local env are called there.
pub fn run_routes<'a>(
    model: impl Into<ModelTarget<'a>>,
    routes: Vec<Route<'a>>,
) -> Result<Vec<Result<RunResult>>> {
    drive::refuse_in_async_context("run_routes")?;
    refuse_in_route()?;
    let driven = fan_out(super::driven(model.into())?, routes.len())?;
    Ok(run_sync(driven.into_iter().zip(routes).collect()))
}

/// The async twin of [`run_routes`]: every route runs inside this future.
pub async fn run_routes_async<'a>(
    model: impl Into<ModelTarget<'a>>,
    routes: Vec<Route<'a>>,
) -> Result<Vec<Result<RunResult>>> {
    let driven = fan_out(super::driven(model.into())?, routes.len())?;
    let barrier = Arc::new(StartBarrier::new(routes.len(), START_BARRIER_TIMEOUT));
    Ok(
        futures::future::join_all(driven.into_iter().zip(routes).map(|(model, route)| {
            run_one(
                model,
                route,
                Mode::Async,
                Arrival(Some(Arc::clone(&barrier))),
            )
        }))
        .await,
    )
}

/// [`run_routes`] for a binding's handlers, one per route: each answers its
/// own route, as each would a single run.
pub fn run_routes_prepared(
    routes: Vec<(crate::binding::model::PreparedModel, Route<'_>)>,
) -> Vec<Result<RunResult>> {
    run_sync(
        routes
            .into_iter()
            .map(|(model, route)| (Driven::Prepared(model), route))
            .collect(),
    )
}

/// Several results as one: episodes in order, indexed afresh from 0 (each
/// keeps its seed and trial ordinal), telemetry merged per series as
/// [`RunResult::load`] merges workloads, and advisories once each.
pub fn merge_results<'r>(results: impl IntoIterator<Item = &'r RunResult>) -> RunResult {
    let results: Vec<&RunResult> = results.into_iter().collect();
    let episodes = results
        .iter()
        .flat_map(|result| &result.episodes)
        .enumerate()
        .map(|(index, episode)| super::EpisodeResult {
            index: index as i64,
            ..episode.clone()
        })
        .collect();
    let mut advisories: Vec<crate::adapters::Advisory> = Vec::new();
    for advisory in results.iter().flat_map(|result| &result.advisories) {
        if !advisories.contains(advisory) {
            advisories.push(advisory.clone());
        }
    }
    RunResult {
        episodes,
        telemetry: super::report::merge_telemetry(results.iter().flat_map(|r| &r.telemetry)),
        advisories,
    }
}

/// One driven model per route: a served model and the random baseline fan
/// out; an in-process model drives a single route.
fn fan_out(model: Driven<'_>, routes: usize) -> Result<Vec<Driven<'_>>> {
    if routes <= 1 {
        return Ok((routes == 1).then_some(model).into_iter().collect());
    }
    (0..routes)
        .map(|_| match &model {
            Driven::Local(_) => Err(Error::invalid_argument(format!(
                "run_routes: an in-process Model drives one route at a time, got {routes}; \
                 serve it and pass its address to share it across routes"
            ))),
            #[cfg(feature = "lockstep")]
            Driven::Served(model) => Ok(Driven::Served(model)),
            Driven::Random => Ok(Driven::Random),
            Driven::Prepared(_) => Err(Error::Internal(
                "run_routes: a prepared handler answers one route".into(),
            )),
        })
        .collect()
}

/// Every route on a thread of its own, each driven exactly as a sync run.
/// This thread waits, polling the caller's signals as a sync run does.
fn run_sync(routes: Vec<(Driven<'_>, Route<'_>)>) -> Vec<Result<RunResult>> {
    let barrier = Arc::new(StartBarrier::new(routes.len(), START_BARRIER_TIMEOUT));
    let total = routes.len();
    let done = AtomicUsize::new(0);
    let waiter = std::thread::current();
    std::thread::scope(|scope| {
        let handles: Vec<_> = routes
            .into_iter()
            .enumerate()
            .map(|(index, (model, route))| {
                // Made before the spawn: a route whose thread never starts
                // still counts at the barrier, and as done.
                let finished = Finished(&done, waiter.clone());
                let arrival = Arrival(Some(Arc::clone(&barrier)));
                std::thread::Builder::new()
                    .name(format!("rlmesh-route-{index}"))
                    .spawn_scoped(scope, move || {
                        let _finished = finished;
                        IN_ROUTE.with(|cell| cell.set(true));
                        drive::run_sync(run_one(model, route, Mode::Sync, arrival))
                    })
            })
            .collect();
        while done.load(Ordering::SeqCst) < total {
            if drive::caller_poll() {
                std::thread::park_timeout(Duration::from_millis(10));
            } else {
                std::thread::park();
            }
        }
        handles
            .into_iter()
            .map(|handle| match handle {
                Ok(handle) => handle
                    .join()
                    .unwrap_or_else(|_| Err(Error::Internal("a route panicked".into()))),
                Err(err) => Err(Error::Runtime(format!("could not start a route: {err}"))),
            })
            .collect()
    })
}

/// Counts a route done and wakes the waiting caller, however it ends.
struct Finished<'d>(&'d AtomicUsize, std::thread::Thread);

impl Drop for Finished<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
        self.1.unpark();
    }
}

async fn run_one(
    model: Driven<'_>,
    route: Route<'_>,
    mode: Mode,
    arrival: Arrival,
) -> Result<RunResult> {
    let Route {
        env,
        options,
        base_seed,
        seed_key,
    } = route;
    native::run_route(
        model,
        env,
        options,
        mode,
        RouteSpec {
            start: Some(RouteStart {
                arrival,
                connect_timeout: CONNECT_TIMEOUT,
            }),
            base_seed,
            seed_key,
        },
    )
    .await
}

/// What a route adds to a single run.
#[derive(Default)]
pub(crate) struct RouteSpec {
    pub(crate) start: Option<RouteStart>,
    pub(crate) base_seed: Option<i64>,
    pub(crate) seed_key: Option<String>,
}

/// A route's place in its run's start: its barrier arrival and its env
/// connect deadline.
pub(crate) struct RouteStart {
    pub(crate) arrival: Arrival,
    pub(crate) connect_timeout: Duration,
}

/// A route's arrival at the start barrier. Dropped before
/// [`arrive`](Self::arrive), as a route that failed before it was ready, it
/// counts toward the target without waiting.
pub(crate) struct Arrival(Option<Arc<StartBarrier>>);

impl Arrival {
    /// Hold until the routes start together; see [`StartBarrier::arrive`].
    pub(crate) async fn arrive(mut self, cancellation: &CancellationToken) {
        if let Some(barrier) = self.0.take() {
            barrier.arrive(cancellation).await;
        }
    }
}

impl Drop for Arrival {
    fn drop(&mut self) {
        if let Some(barrier) = self.0.take() {
            barrier.arrive_failed();
        }
    }
}

/// Holds a run's routes after they connect until they can step together.
/// Without it, routes whose envs come up first run a whole wave ahead of the
/// slow ones, the waves never overlap, and a batching model's batches run
/// half empty.
pub(crate) struct StartBarrier {
    target: usize,
    timeout: Duration,
    state: Mutex<BarrierState>,
    opened: watch::Sender<bool>,
}

#[derive(Default)]
struct BarrierState {
    arrived: usize,
    // The clock starts when a route is first held, not at run start, so a
    // slow model or env startup does not spend the timeout.
    first_ready: Option<Instant>,
}

impl StartBarrier {
    /// A target of one or none is open from the start.
    pub(crate) fn new(target: usize, timeout: Duration) -> Self {
        Self {
            target,
            timeout,
            state: Mutex::default(),
            opened: watch::Sender::new(target <= 1),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, BarrierState> {
        self.state.lock().unwrap_or_else(|err| err.into_inner())
    }

    /// A route that failed before it was ready counts toward the target
    /// without waiting, so it cannot hold the others.
    fn arrive_failed(&self) {
        let mut state = self.state();
        state.arrived += 1;
        if state.arrived >= self.target {
            self.open(&state, "count");
        }
    }

    /// Waits until every route has arrived, the timeout since the first held
    /// route passes, or the route is cancelled. Once open, never waits.
    async fn arrive(&self, cancellation: &CancellationToken) {
        if *self.opened.borrow() {
            return;
        }
        let deadline = {
            let mut state = self.state();
            state.arrived += 1;
            if state.arrived >= self.target {
                self.open(&state, "count");
                return;
            }
            *state.first_ready.get_or_insert_with(Instant::now) + self.timeout
        };
        let mut opened = self.opened.subscribe();
        tokio::select! {
            _ = opened.wait_for(|open| *open) => {}
            () = tokio::time::sleep_until(deadline) => {
                let state = self.state();
                self.open(&state, "timeout");
            }
            () = cancellation.cancelled() => {}
        }
    }

    fn open(&self, state: &BarrierState, reason: &'static str) {
        if self
            .opened
            .send_if_modified(|open| !std::mem::replace(open, true))
        {
            tracing::info!(
                target_routes = self.target,
                arrived = state.arrived,
                waited_ms = state.first_ready.map_or(0, |t| t.elapsed().as_millis()),
                opened_on = reason,
                "route start barrier open",
            );
        }
    }
}

/// Retry `attempt` until it succeeds, fails for good, `timeout` passes or
/// the route is cancelled, backing off from [`RETRY_BACKOFF`] to
/// [`MAX_BACKOFF`]. Only a transient failure (a refused or timed-out
/// connect) is retried; a refusal such as an edition mismatch or an env
/// another driver holds is final. The last failure is the error.
pub(crate) async fn connect_with_retry<T, F, Fut>(
    timeout: Duration,
    cancellation: &CancellationToken,
    mut attempt: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let deadline = Instant::now() + timeout;
    let mut backoff = RETRY_BACKOFF;
    let mut attempts: u32 = 0;
    loop {
        let error = tokio::select! {
            () = cancellation.cancelled() => return Err(Error::Cancelled),
            result = attempt() => match result {
                Ok(connected) => return Ok(connected),
                Err(error) => error,
            },
        };
        attempts += 1;
        let held = matches!(&error, Error::Connection(message)
            if message.starts_with(crate::env::client::ENV_HELD));
        let sleep_for = backoff.min(deadline.saturating_duration_since(Instant::now()));
        if !error.is_transient() || held || sleep_for.is_zero() {
            return Err(error);
        }
        tracing::warn!(attempt = attempts, %error, "env connect failed; retrying");
        backoff = (backoff * 2).min(MAX_BACKOFF);
        tokio::select! {
            () = cancellation.cancelled() => return Err(Error::Cancelled),
            () = tokio::time::sleep(sleep_for) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TICK: Duration = Duration::from_millis(50);

    fn spawn_arrival(barrier: &Arc<StartBarrier>) -> tokio::task::JoinHandle<Instant> {
        let barrier = Arc::clone(barrier);
        tokio::spawn(async move {
            barrier.arrive(&CancellationToken::new()).await;
            Instant::now()
        })
    }

    #[tokio::test]
    async fn the_start_barrier_opens_on_the_last_arrival_or_a_timeout_from_the_first() {
        // Every route arrives: the held ones start with the last.
        let barrier = Arc::new(StartBarrier::new(3, Duration::from_secs(60)));
        let first = spawn_arrival(&barrier);
        let second = spawn_arrival(&barrier);
        tokio::time::sleep(TICK).await;
        assert!(!first.is_finished() && !second.is_finished());
        barrier.arrive(&CancellationToken::new()).await;
        let opened = Instant::now();
        assert!(first.await.unwrap() >= opened && second.await.unwrap() >= opened);

        // Fewer arrive: the timeout runs from the first held route, not
        // from the run's start.
        let barrier = Arc::new(StartBarrier::new(3, 4 * TICK));
        tokio::time::sleep(4 * TICK).await;
        let first_ready = Instant::now();
        let first = spawn_arrival(&barrier);
        tokio::time::sleep(TICK).await;
        let second = spawn_arrival(&barrier);
        let released = first.await.unwrap();
        assert!(released >= first_ready + 4 * TICK, "held for the timeout");
        assert!(second.await.unwrap() >= released);

        // A failed route counts without waiting; a dropped arrival is one.
        let barrier = Arc::new(StartBarrier::new(3, Duration::from_secs(60)));
        let held = spawn_arrival(&barrier);
        tokio::time::sleep(TICK).await;
        barrier.arrive_failed();
        assert!(!held.is_finished());
        drop(Arrival(Some(Arc::clone(&barrier))));
        tokio::time::timeout(Duration::from_secs(5), held)
            .await
            .expect("the barrier opened")
            .unwrap();

        // Cancellation ends a wait without opening the barrier.
        let barrier = StartBarrier::new(2, Duration::from_secs(60));
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        barrier.arrive(&cancellation).await;
        assert!(!*barrier.opened.borrow());
    }

    #[tokio::test]
    async fn connect_retries_only_transient_failures_until_the_deadline() {
        type Case = (&'static str, fn() -> Error, bool);
        let cases: [Case; 3] = [
            ("refused", || Error::Connection("refused".into()), true),
            (
                "held",
                || Error::Connection(crate::env::client::ENV_HELD.into()),
                false,
            ),
            (
                "edition",
                || Error::invalid_argument("no mutual edition"),
                false,
            ),
        ];
        for (name, error, retried) in cases {
            let attempts = AtomicUsize::new(0);
            let result: Result<()> =
                connect_with_retry(4 * TICK, &CancellationToken::new(), || async {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Err(error())
                })
                .await;
            assert_eq!(
                result.unwrap_err().to_string(),
                error().to_string(),
                "{name}"
            );
            assert_eq!(attempts.load(Ordering::SeqCst) > 1, retried, "{name}");
        }
    }
}
