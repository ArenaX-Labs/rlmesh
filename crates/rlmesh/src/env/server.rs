//! [`EnvServer`]: Python's `EnvServer`, one non-generic server for an env, a
//! list of lane envs, or a vector env.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rlmesh_adapters::v1::{DESCRIBE_METADATA_KEY, ENV_BRANCH_METADATA_KEY};
use rlmesh_grpc::env::Environment;
use rlmesh_grpc::lifecycle::{ShutdownTrigger, await_close_with_timeout, start_idle_shutdown};

use super::envelope::{DescribingHost, env_envelope};
use super::single_lane::SingleLaneEnv;
use super::sync_wire::{
    Actor, ActorRunner, SyncLane, SyncVector, env_contract_of, vector_contract_of,
};
use super::wire::{WireEnvAdapter, WireLaneAdapter};
use super::{Env, VectorEnv, signals};
use crate::adapters::EnvTags;
use crate::bound::BoundListener;
use crate::editions::EditionSources;
use crate::serve_options::{WireExtras, env_endpoint_token};
use crate::specs::{EnvContract, value_to_json};
use crate::{BindAddress, Error, Result, ServeOptions};

/// The bound an env server puts on a drain or close the options leave unset
/// (ENV-7).
const DEFAULT_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Python's `Transport` (`Literal["tcp", "unix"]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Transport {
    /// TCP.
    Tcp,
    /// A Unix-domain socket.
    Unix,
}

impl Transport {
    /// Python's spelling: `"tcp"` or `"unix"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Transport::Tcp => "tcp",
            Transport::Unix => "unix",
        }
    }
}

impl fmt::Display for Transport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The keywords of Python's `EnvServer(env, address=None, *, ...)`.
///
/// Defaults are Python's: the address `tcp://127.0.0.1:0`, no serve options or
/// tags, and `close_env_on_shutdown` true. Build one with
/// `..Default::default()`; appending a field is a minor change by project
/// policy.
#[derive(Clone, Debug)]
pub struct EnvServerOptions {
    /// The bind address (`tcp://host:port` or `unix:///path`); `None` uses the
    /// helpers below, or `tcp://127.0.0.1:0`. Excludes the helpers.
    pub address: Option<String>,
    /// TCP host helper (default `127.0.0.1`).
    pub host: Option<String>,
    /// TCP port helper (default 0, an OS-assigned port).
    pub port: Option<u16>,
    /// Unix socket path helper; excludes `host` and `port`.
    pub path: Option<String>,
    /// Explicit transport; `Unix` needs `path`.
    pub transport: Option<Transport>,
    /// Transport lifecycle policy.
    pub options: Option<ServeOptions>,
    /// Tags to merge into each served env's metadata, validated against a
    /// scalar env's spaces.
    pub tags: Option<EnvTags>,
    /// Close the env when the server stops (default `true`); `false` for a
    /// server in front of an env its caller closes.
    pub close_env_on_shutdown: bool,
    /// Rust-only fields (ENV-12).
    pub ext: Option<crate::ext::EnvServerOptions>,
}

impl Default for EnvServerOptions {
    fn default() -> Self {
        EnvServerOptions {
            address: None,
            host: None,
            port: None,
            path: None,
            transport: None,
            options: None,
            tags: None,
            close_env_on_shutdown: true,
            ext: None,
        }
    }
}

/// A factory serve's describe publication (the envelope's target and
/// catalog).
pub(crate) struct Publish {
    pub(crate) qualname: String,
    pub(crate) catalog: Option<serde_json::Value>,
}

#[allow(clippy::print_stderr)]
fn describe_failed(err: &Error) {
    eprintln!(
        "RLMesh could not describe the served env ({err}); the handshake carries no describe \
         envelope"
    );
}

/// What the server serves.
enum Served {
    Lanes(Vec<Box<dyn Env + Send>>),
    Vector(Box<dyn VectorEnv + Send>),
}

struct Resources {
    served: Served,
    runtime: tokio::runtime::Runtime,
    listener: BoundListener,
    grpc: rlmesh_grpc::ServeOptions,
    idle_timeout: Option<Duration>,
    drain_timeout: Duration,
    close_timeout: Duration,
    close_env_on_shutdown: bool,
}

enum State {
    Ready(Box<Resources>),
    Starting,
    Foreground,
    Background(JoinHandle<Result<()>>),
    Stopped,
}

struct Shared {
    id: u64,
    address: String,
    contract: EnvContract,
    shutdown: ShutdownTrigger,
    state: Mutex<State>,
    /// A serve thread nobody waits on any more (a dropped `serve_async`,
    /// `shutdown_async`): the server's Drop joins it.
    detached: Mutex<Option<JoinHandle<Result<()>>>>,
    /// Set when a serve thread finishes, for the async twins.
    done: tokio::sync::watch::Sender<bool>,
}

/// Serves an env, a list of lane envs, or a vector env: Python's `EnvServer`.
///
/// The server binds at construction, so [`address`](Self::address) and
/// [`env_contract`](Self::env_contract) are readable before serving. Then
/// [`serve`](Self::serve) blocks with lane 0 (or the vector env) on the
/// calling thread, or [`start`](Self::start) serves from library threads and
/// [`wait`](Self::wait) / [`shutdown`](Self::shutdown) observe the stop. Each
/// lane runs on one thread for the server's life.
///
/// Dropping a server runs [`shutdown`](Self::shutdown) and discards its error;
/// call `shutdown()` to observe it. Inside an async runtime, Drop triggers the
/// drain and leaves the rest to the server's own threads.
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
    let previous = IN_CALLBACK.with(|cell| cell.replace(id));
    let result = body();
    IN_CALLBACK.with(|cell| cell.set(previous));
    result
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

/// R-16's condition: the thread is polling a tokio task.
fn in_async_context() -> bool {
    tokio::task::try_id().is_some()
}

impl EnvServer {
    /// Serve one env: Python's `EnvServer(env, ...)`.
    pub fn new<E: Env + Send + 'static>(env: E, options: EnvServerOptions) -> Result<EnvServer> {
        Self::build(Served::Lanes(vec![Box::new(env)]), options, None)
    }

    /// Serve `envs` as the lanes of one `num_envs = N` endpoint, each on its
    /// own thread: Python's `EnvServer([env, ...])`. Use `Box<dyn Env + Send>`
    /// to mix env types.
    pub fn lanes<E: Env + Send + 'static>(
        envs: Vec<E>,
        options: EnvServerOptions,
    ) -> Result<EnvServer> {
        Self::build(
            Served::Lanes(
                envs.into_iter()
                    .map(|env| Box::new(env) as Box<dyn Env + Send>)
                    .collect(),
            ),
            options,
            None,
        )
    }

    /// Serve a vector env: Python's `EnvServer(vector_env)`. A vector env with
    /// `num_envs == 1` is served as a single scalar lane.
    pub fn vector<V: VectorEnv + Send + 'static>(
        env: V,
        options: EnvServerOptions,
    ) -> Result<EnvServer> {
        Self::build(Served::Vector(Box::new(env)), options, None)
    }

    /// A factory serve's server: the edition was resolved by the caller.
    pub(crate) fn lanes_resolved(
        envs: Vec<Box<dyn Env + Send>>,
        options: EnvServerOptions,
        publish: Publish,
    ) -> Result<EnvServer> {
        Self::build(Served::Lanes(envs), options, Some(publish))
    }

    pub(crate) fn vector_resolved(
        env: Box<dyn VectorEnv + Send>,
        options: EnvServerOptions,
        publish: Publish,
    ) -> Result<EnvServer> {
        Self::build(Served::Vector(env), options, Some(publish))
    }

    /// `factory`: a factory serve's publication, whose edition the caller
    /// resolved.
    fn build(
        served: Served,
        options: EnvServerOptions,
        factory: Option<Publish>,
    ) -> Result<EnvServer> {
        let resolved_by_caller = factory.is_some();
        // Step 1: at least one lane (the traits guarantee the spaces).
        let mut served = match served {
            Served::Lanes(lanes) if lanes.is_empty() => {
                return Err(Error::invalid_argument(
                    "EnvServer needs at least one environment",
                ));
            }
            // Step 2: a one-lane vector env is a scalar lane; 0 is refused.
            Served::Vector(env) if env.num_envs() == 1 => {
                Served::Lanes(vec![Box::new(SingleLaneEnv::new(env)?)])
            }
            Served::Vector(env) if env.num_envs() < 2 => {
                return Err(Error::invalid_argument(format!(
                    "a vector env must have num_envs >= 1, got {}",
                    env.num_envs()
                )));
            }
            other => other,
        };

        // Step 3: tags, merged into each scalar env's metadata and validated
        // against its spaces; without tags, published tags are re-validated.
        // Steps 4 and 7: the expected branch and each lane's contract.
        let contract = match &mut served {
            Served::Lanes(lanes) => {
                if let Some(tags) = &options.tags {
                    *lanes = std::mem::take(lanes)
                        .into_iter()
                        .map(|lane| {
                            Tagged::new(lane, tags.clone())
                                .map(|tagged| Box::new(tagged) as Box<dyn Env + Send>)
                        })
                        .collect::<Result<_>>()?;
                }
                let mut contracts = Vec::with_capacity(lanes.len());
                for lane in lanes.iter() {
                    let contract = env_contract_of(lane.as_ref())?;
                    if options.tags.is_none()
                        && let Some(metadata) = contract.metadata()
                        && let Some(published) = EnvTags::from_metadata(metadata)?
                    {
                        published.validate(&lane.observation_space(), &lane.action_space())?;
                    }
                    check_expected_branch(contract.metadata())?;
                    contracts.push(contract);
                }
                let first = contracts[0].clone();
                for (index, contract) in contracts.iter().enumerate().skip(1) {
                    if !contract.same_lane_contract(&first) {
                        return Err(Error::invalid_argument(format!(
                            "lane {index} disagrees with lane 0 on the env contract (spaces, \
                             metadata, tags): every lane of one endpoint must be the same make()"
                        )));
                    }
                }
                first.with_num_envs(lanes.len() as u64)
            }
            Served::Vector(env) => {
                let contract = match &options.tags {
                    // Deferred validation: tags describe one lane, the vector's
                    // spaces are its single spaces.
                    Some(tags) => {
                        let mut metadata = env.metadata().unwrap_or_default();
                        tags.stamp(&mut metadata);
                        let base = vector_contract_of(env.as_ref())?;
                        EnvContract::new(
                            base.id().to_string(),
                            base.observation_space().clone(),
                            base.action_space().clone(),
                            base.render_mode().map(str::to_string),
                            base.num_envs(),
                            Some(metadata),
                            base.autoreset_mode(),
                        )
                    }
                    None => vector_contract_of(env.as_ref())?,
                };
                check_expected_branch(contract.metadata())?;
                contract
            }
        };

        // Step 5: the address.
        let address = normalize_bind_address(&options)?;
        let serve = options.options.clone().unwrap_or_default();
        serve.validate()?;

        // Step 6: the edition. An explicit options value is the call's own;
        // a served env is an authored participant, so a float warns.
        let edition = crate::editions::resolve_native(
            EditionSources {
                call: serve.workflow_edition.as_deref(),
                option: None,
                declared: None,
            },
            !resolved_by_caller,
        )?;

        // Step 8: the endpoint token, before binding.
        let ext = options.ext.clone().unwrap_or_default();
        let token = env_endpoint_token(ext.token.clone())?;
        let mut peer_info_extra: BTreeMap<String, String> = ext.peer_info_extra.clone();
        match &factory {
            // A factory serve's describe is best-effort (step 5).
            Some(publish) => match env_envelope(
                &contract,
                Some(&publish.qualname),
                publish.catalog.as_ref(),
                &DescribingHost::rust(),
                edition.as_deref(),
            ) {
                Ok(envelope) => {
                    peer_info_extra.insert(DESCRIBE_METADATA_KEY.to_string(), envelope);
                }
                Err(err) => describe_failed(&err),
            },
            None if ext.publish_describe => {
                let envelope = env_envelope(
                    &contract,
                    None,
                    None,
                    &DescribingHost::rust(),
                    edition.as_deref(),
                )?;
                peer_info_extra.insert(DESCRIBE_METADATA_KEY.to_string(), envelope);
            }
            None => {}
        }

        // Step 9: bind.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("rlmesh-env-server")
            .build()
            .map_err(|err| Error::Internal(format!("failed to create tokio runtime: {err}")))?;
        let listener = BoundListener::bind_sync(BindAddress::parse(&address)?, runtime.handle())?;
        let address = listener.local_addr()?.to_string();

        let idle_timeout = serve.idle_timeout;
        let drain_timeout = serve.drain_timeout.unwrap_or(DEFAULT_SHUTDOWN_GRACE);
        let close_timeout = serve.close_timeout.unwrap_or(DEFAULT_SHUTDOWN_GRACE);
        let grpc = ServeOptions {
            workflow_edition: edition,
            ..serve
        }
        .to_grpc(WireExtras {
            token,
            predict_concurrency: None,
            peer_info_extra,
        });
        let (done, _) = tokio::sync::watch::channel(false);
        Ok(EnvServer {
            shared: Arc::new(Shared {
                id: NEXT_SERVER.fetch_add(1, Ordering::Relaxed),
                address,
                contract,
                shutdown: ShutdownTrigger::new(),
                state: Mutex::new(State::Ready(Box::new(Resources {
                    served,
                    runtime,
                    listener,
                    grpc,
                    idle_timeout,
                    drain_timeout,
                    close_timeout,
                    close_env_on_shutdown: options.close_env_on_shutdown,
                }))),
                detached: Mutex::new(None),
                done,
            }),
        })
    }

    /// The bound address (`tcp://127.0.0.1:<port>` for port 0).
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
        if in_async_context() {
            return Err(Error::Runtime(
                "EnvServer::serve() blocks; call serve_async() from async code".to_string(),
            ));
        }
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
        let handle = self.spawn_background("start", State::Starting)?;
        let mut state = lock(&self.shared.state);
        match std::mem::replace(&mut *state, State::Stopped) {
            State::Starting => *state = State::Background(handle),
            // A shutdown() raced the start: it already moved to stopped.
            State::Stopped => *lock(&self.shared.detached) = Some(handle),
            other => *state = other,
        }
        Ok(())
    }

    fn spawn_background(&self, action: &str, next: State) -> Result<JoinHandle<Result<()>>> {
        let resources = self.take_resources(action, next)?;
        let shared = Arc::clone(&self.shared);
        std::thread::Builder::new()
            .name("rlmesh-env-server".to_string())
            .spawn(move || run(*resources, &shared, false))
            .map_err(|err| Error::Internal(format!("failed to spawn server thread: {err}")))
    }

    /// Wait for a started server to stop: `Ok(false)` when `timeout` passes
    /// first, `Ok(true)` once it has stopped. The call that observes the stop
    /// returns the serve error, once. `timeout` may be zero.
    pub fn wait(&self, timeout: Option<Duration>) -> Result<bool> {
        let started = Instant::now();
        loop {
            if let Some(result) = self.poll_wait()? {
                return result.map(|()| true);
            }
            let nap = match timeout {
                Some(timeout) => {
                    let elapsed = started.elapsed();
                    if elapsed >= timeout {
                        return Ok(false);
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
        let handle = {
            let mut state = lock(&self.shared.state);
            match &*state {
                State::Background(handle) if handle.is_finished() => {
                    match std::mem::replace(&mut *state, State::Stopped) {
                        State::Background(handle) => handle,
                        _ => unreachable!("state changed while locked"),
                    }
                }
                State::Background(_) | State::Starting => return Ok(None),
                State::Stopped => return Ok(Some(Ok(()))),
                State::Ready(_) => {
                    return Err(Error::Runtime("wait() called before start()".to_string()));
                }
                State::Foreground => {
                    return Err(Error::Runtime(
                        "wait() called while serve() is already running in the foreground"
                            .to_string(),
                    ));
                }
            }
        };
        Ok(Some(join(handle)))
    }

    /// Stop the server: release a never-served listener (closing the envs if
    /// `close_env_on_shutdown`), or trigger the drain and join a started one,
    /// returning its serve error unless a `wait()` already returned it.
    /// Called from one of this server's own env callbacks, it only triggers
    /// the drain.
    pub fn shutdown(&self) -> Result<()> {
        self.shared.shutdown.trigger("local shutdown");
        if IN_CALLBACK.with(std::cell::Cell::get) == self.shared.id {
            return Ok(());
        }
        let previous = std::mem::replace(&mut *lock(&self.shared.state), State::Stopped);
        match previous {
            State::Ready(resources) => cleanup_ready(*resources),
            State::Background(handle) => join(handle),
            State::Foreground => {
                // The drain and close finish inside serve().
                *lock(&self.shared.state) = State::Stopped;
                Ok(())
            }
            State::Starting | State::Stopped => Ok(()),
        }
    }

    /// The async twin of [`serve`](Self::serve): every lane, lane 0 included,
    /// runs on library threads and no signal handler is installed; resolves
    /// with the serve error when the server stops. Dropping it stops the
    /// server (the drain and close finish detached).
    pub async fn serve_async(&self) -> Result<()> {
        let mut done = self.shared.done.subscribe();
        let handle = self.spawn_background("serve", State::Foreground)?;
        *lock(&self.shared.detached) = Some(handle);
        let guard = StopOnDrop(&self.shared);
        while !*done.borrow_and_update() {
            if done.changed().await.is_err() {
                break;
            }
        }
        std::mem::forget(guard);
        self.set_stopped();
        let handle = lock(&self.shared.detached).take();
        handle.map_or(Ok(()), join)
    }

    /// The async twin of [`wait`](Self::wait). Dropping it leaves the server
    /// as it was.
    pub async fn wait_async(&self, timeout: Option<Duration>) -> Result<bool> {
        let mut done = self.shared.done.subscribe();
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        loop {
            if let Some(result) = self.poll_wait()? {
                return result.map(|()| true);
            }
            let remaining = match deadline {
                Some(deadline) => match deadline.checked_duration_since(Instant::now()) {
                    Some(remaining) if !remaining.is_zero() => Some(remaining),
                    _ => return Ok(false),
                },
                None => None,
            };
            let changed = done.changed();
            match remaining {
                Some(remaining) => {
                    let timer = sleep_off_runtime(remaining.min(Duration::from_millis(50)));
                    futures::pin_mut!(changed, timer);
                    let _ = futures::future::select(changed, timer).await;
                }
                None => {
                    let timer = sleep_off_runtime(Duration::from_millis(50));
                    futures::pin_mut!(changed, timer);
                    let _ = futures::future::select(changed, timer).await;
                }
            }
        }
    }

    /// The async twin of [`shutdown`](Self::shutdown). Dropping it leaves the
    /// drain it triggered to finish detached.
    pub async fn shutdown_async(&self) -> Result<()> {
        self.shared.shutdown.trigger("local shutdown");
        let mut done = self.shared.done.subscribe();
        let previous = std::mem::replace(&mut *lock(&self.shared.state), State::Stopped);
        match previous {
            State::Ready(resources) => {
                let (tx, rx) = futures::channel::oneshot::channel();
                std::thread::spawn(move || {
                    let _ = tx.send(cleanup_ready(*resources));
                });
                rx.await
                    .unwrap_or_else(|_| Err(Error::Internal("server cleanup panicked".into())))
            }
            State::Background(handle) => {
                *lock(&self.shared.detached) = Some(handle);
                while !*done.borrow_and_update() {
                    if done.changed().await.is_err() {
                        break;
                    }
                }
                let handle = lock(&self.shared.detached).take();
                handle.map_or(Ok(()), join)
            }
            State::Foreground | State::Starting | State::Stopped => Ok(()),
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

    fn set_stopped(&self) {
        *lock(&self.shared.state) = State::Stopped;
    }
}

/// A dropped `serve_async`: stop the server, as `shutdown()` does on a
/// foreground serve.
struct StopOnDrop<'a>(&'a Shared);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.shutdown.trigger("dropped serve_async");
        *lock(&self.0.state) = State::Stopped;
    }
}

impl Drop for EnvServer {
    fn drop(&mut self) {
        self.shared.shutdown.trigger("drop");
        let previous = std::mem::replace(&mut *lock(&self.shared.state), State::Stopped);
        let detached = lock(&self.shared.detached).take();
        if in_async_context() {
            // ENV-10: never block an executor; the server's own threads finish.
            if let State::Ready(resources) = previous {
                std::thread::spawn(move || {
                    let _ = cleanup_ready(*resources);
                });
            }
            return;
        }
        let _ = match previous {
            State::Ready(resources) => cleanup_ready(*resources),
            State::Background(handle) => join(handle),
            _ => Ok(()),
        };
        if let Some(handle) = detached {
            let _ = join(handle);
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
    handle.join().unwrap_or_else(|_| {
        Err(Error::Internal(
            "serve failed: the server thread panicked".into(),
        ))
    })
}

/// Step 4: `RLMESH_EXPECTED_ENV_BRANCH`, when set and non-empty, must equal the
/// env's published branch.
fn check_expected_branch(metadata: Option<&crate::Metadata>) -> Result<()> {
    let Ok(expected) = std::env::var("RLMESH_EXPECTED_ENV_BRANCH") else {
        return Ok(());
    };
    if expected.trim().is_empty() {
        return Ok(());
    }
    let expected_json: serde_json::Value = serde_json::from_str(&expected).map_err(|err| {
        Error::invalid_argument(format!(
            "RLMESH_EXPECTED_ENV_BRANCH is not valid JSON: {err}"
        ))
    })?;
    let actual = match metadata.and_then(|metadata| metadata.get(ENV_BRANCH_METADATA_KEY)) {
        Some(value) => value_to_json(value)?,
        None => serde_json::Value::Null,
    };
    if actual != expected_json {
        return Err(Error::invalid_argument(format!(
            "this image was published for env branch {expected_json} but the served env is \
             branch {actual}; build and push the image for the branch it serves"
        )));
    }
    Ok(())
}

/// Step 5: Python's `normalize_bind_address`.
fn normalize_bind_address(options: &EnvServerOptions) -> Result<String> {
    let helpers = options.host.is_some()
        || options.port.is_some()
        || options.path.is_some()
        || options.transport.is_some();
    if let Some(address) = &options.address {
        if address.is_empty() {
            return Err(Error::invalid_argument("address cannot be empty"));
        }
        if helpers {
            return Err(Error::invalid_argument(
                "address cannot be combined with host, port, path, or transport",
            ));
        }
        return Ok(address.clone());
    }
    let tcp_helpers = options.host.is_some() || options.port.is_some();
    if options.path.is_some() && tcp_helpers {
        return Err(Error::invalid_argument(
            "path cannot be combined with host or port",
        ));
    }
    if options.transport == Some(Transport::Unix) && tcp_helpers {
        return Err(Error::invalid_argument(
            "unix transport cannot be combined with host or port",
        ));
    }
    if options.transport == Some(Transport::Tcp) && options.path.is_some() {
        return Err(Error::invalid_argument(
            "path cannot be combined with transport='tcp'",
        ));
    }
    if options.path.is_some() || options.transport == Some(Transport::Unix) {
        let Some(path) = &options.path else {
            return Err(Error::invalid_argument(
                "path is required when transport='unix'",
            ));
        };
        if cfg!(windows) {
            return Err(Error::invalid_argument(
                "unix sockets are not supported on Windows; use tcp://host:port instead",
            ));
        }
        return Ok(format!("unix://{path}"));
    }
    Ok(format!(
        "tcp://{}:{}",
        options.host.as_deref().unwrap_or("127.0.0.1"),
        options.port.unwrap_or(0)
    ))
}

/// An env with tags merged into its metadata (Python's `adapters.tag`).
pub(crate) struct Tagged {
    env: Box<dyn Env + Send>,
    metadata: crate::Metadata,
}

impl Tagged {
    /// Merge and validate `tags` against `env`'s spaces.
    pub(crate) fn new(env: Box<dyn Env + Send>, tags: EnvTags) -> Result<Tagged> {
        tags.validate(&env.observation_space(), &env.action_space())?;
        Ok(Self::unchecked(env, tags))
    }

    /// Merge `tags` without validation.
    pub(crate) fn unchecked(env: Box<dyn Env + Send>, tags: EnvTags) -> Tagged {
        let mut metadata = env.metadata().unwrap_or_default();
        tags.stamp(&mut metadata);
        Tagged { env, metadata }
    }

    /// `env` serving `metadata` as its own (a factory's stamp).
    pub(crate) fn with_metadata(env: Box<dyn Env + Send>, metadata: crate::Metadata) -> Tagged {
        Tagged { env, metadata }
    }
}

impl Env for Tagged {
    fn observation_space(&self) -> crate::spaces::Space {
        self.env.observation_space()
    }
    fn action_space(&self) -> crate::spaces::Space {
        self.env.action_space()
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&crate::InfoDict>,
    ) -> Result<(crate::Value, crate::InfoDict)> {
        self.env.reset(seed, options)
    }
    fn step(
        &mut self,
        action: Option<crate::Value>,
    ) -> Result<(crate::Value, f64, bool, bool, crate::InfoDict)> {
        self.env.step(action)
    }
    fn render(&mut self) -> Result<Option<crate::Value>> {
        self.env.render()
    }
    fn close(&mut self) -> Result<()> {
        self.env.close()
    }
    fn render_mode(&self) -> Option<String> {
        self.env.render_mode()
    }
    fn metadata(&self) -> Option<crate::Metadata> {
        Some(self.metadata.clone())
    }
    fn id(&self) -> Option<String> {
        self.env.id()
    }
}

/// An env run marked as server `id`'s callback.
struct Marked<E> {
    id: u64,
    env: E,
}

impl<E: Env + ?Sized> Env for Marked<Box<E>> {
    fn observation_space(&self) -> crate::spaces::Space {
        self.env.observation_space()
    }
    fn action_space(&self) -> crate::spaces::Space {
        self.env.action_space()
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&crate::InfoDict>,
    ) -> Result<(crate::Value, crate::InfoDict)> {
        in_callback(self.id, || self.env.reset(seed, options))
    }
    fn step(
        &mut self,
        action: Option<crate::Value>,
    ) -> Result<(crate::Value, f64, bool, bool, crate::InfoDict)> {
        in_callback(self.id, || self.env.step(action))
    }
    fn render(&mut self) -> Result<Option<crate::Value>> {
        in_callback(self.id, || self.env.render())
    }
    fn close(&mut self) -> Result<()> {
        in_callback(self.id, || self.env.close())
    }
}

impl<V: VectorEnv + ?Sized> VectorEnv for Marked<Box<V>> {
    fn num_envs(&self) -> u64 {
        self.env.num_envs()
    }
    fn single_observation_space(&self) -> crate::spaces::Space {
        self.env.single_observation_space()
    }
    fn single_action_space(&self) -> crate::spaces::Space {
        self.env.single_action_space()
    }
    fn reset(
        &mut self,
        seed: Option<super::VectorSeed>,
        options: Option<&crate::InfoDict>,
    ) -> Result<(crate::Value, crate::InfoDict)> {
        in_callback(self.id, || self.env.reset(seed, options))
    }
    fn step(
        &mut self,
        actions: crate::Value,
    ) -> Result<(
        crate::Value,
        Vec<f64>,
        Vec<bool>,
        Vec<bool>,
        crate::InfoDict,
    )> {
        in_callback(self.id, || self.env.step(actions))
    }
    fn render(&mut self) -> Result<Option<crate::Value>> {
        in_callback(self.id, || self.env.render())
    }
    fn close(&mut self) -> Result<()> {
        in_callback(self.id, || self.env.close())
    }
}

/// The foreground job runner of a lane or a vector env, if this serve keeps
/// one on the calling thread.
enum Runner {
    None,
    Lane(ActorRunner<Box<dyn Env + Send>>),
    Vector(ActorRunner<Box<dyn VectorEnv + Send>>),
}

impl Runner {
    fn run(self) {
        match self {
            Runner::None => {}
            Runner::Lane(runner) => runner.run(),
            Runner::Vector(runner) => runner.run(),
        }
    }
}

/// Serve until the shutdown trigger fires, then drain and close. In the
/// foreground the calling thread runs lane 0's (or the vector env's) jobs
/// while the transport runs on a helper thread.
fn run(resources: Resources, shared: &Arc<Shared>, foreground: bool) -> Result<()> {
    let result = run_inner(resources, shared, foreground);
    let _ = shared.done.send(true);
    result
}

fn run_inner(resources: Resources, shared: &Arc<Shared>, foreground: bool) -> Result<()> {
    let Resources {
        served,
        runtime,
        listener,
        grpc,
        idle_timeout,
        drain_timeout,
        close_timeout,
        close_env_on_shutdown,
    } = resources;
    let id = shared.id;
    let contract = shared.contract.clone();
    let parts = ServeParts {
        runtime,
        listener,
        grpc,
        idle_timeout,
        drain_timeout,
        close_timeout,
        close_env_on_shutdown,
        shutdown: shared.shutdown.clone(),
    };
    match served {
        Served::Lanes(lanes) => {
            let mut wire_lanes = Vec::with_capacity(lanes.len());
            let mut runner = Runner::None;
            for (index, lane) in lanes.into_iter().enumerate() {
                let lane: Box<dyn Env + Send> = Box::new(Marked { id, env: lane });
                if foreground && index == 0 {
                    let (relayed, lane_runner) = SyncLane::relayed(lane, &contract)?;
                    wire_lanes.push(relayed);
                    runner = Runner::Lane(lane_runner);
                } else {
                    wire_lanes.push(SyncLane::new(lane, &contract)?);
                }
            }
            let adapter = WireLaneAdapter::new(wire_lanes)
                .map_err(|err| Error::invalid_argument(err.to_string()))?;
            drive(adapter, runner, parts)
        }
        Served::Vector(env) => {
            let env: Box<dyn VectorEnv + Send> = Box::new(Marked { id, env });
            let (actor, runner) = if foreground {
                let (actor, runner) = Actor::foreground(env);
                (actor, Runner::Vector(runner))
            } else {
                let (actor, _thread) = Actor::spawn("rlmesh-vector-env".to_string(), env)?;
                (actor, Runner::None)
            };
            drive(
                WireEnvAdapter::new(SyncVector::new(actor, &contract)?),
                runner,
                parts,
            )
        }
    }
}

/// What the transport needs to serve one env.
struct ServeParts {
    runtime: tokio::runtime::Runtime,
    listener: BoundListener,
    grpc: rlmesh_grpc::ServeOptions,
    idle_timeout: Option<Duration>,
    drain_timeout: Duration,
    close_timeout: Duration,
    close_env_on_shutdown: bool,
    shutdown: ShutdownTrigger,
}

/// Serve `environment`; with a foreground runner, the transport runs on a
/// helper thread while this thread runs the runner's jobs.
fn drive<E: Environment + Send + Sync + 'static>(
    environment: E,
    runner: Runner,
    parts: ServeParts,
) -> Result<()> {
    let ServeParts {
        runtime,
        listener,
        grpc,
        idle_timeout,
        drain_timeout,
        close_timeout,
        close_env_on_shutdown,
        shutdown,
    } = parts;
    let serve = move || -> Result<()> {
        runtime.block_on(async move {
            let environment = Arc::new(environment);
            let activity_tx = start_idle_shutdown(idle_timeout, shutdown.clone());
            let service = rlmesh_grpc::env::env_service_from_shared(
                Arc::clone(&environment),
                shutdown.clone(),
                grpc,
                activity_tx,
            );
            let (_health, health_service) = rlmesh_grpc::health::serving_health_service().await;
            let router = tonic::transport::Server::builder()
                .add_service(health_service)
                .add_service(service);
            let serve_result = listener.serve(router, shutdown, Some(drain_timeout)).await;
            let close_result = if close_env_on_shutdown {
                let close = async { environment.close().await.map(|_| ()) };
                match await_close_with_timeout(close, Some(close_timeout)).await {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(err)) => Err(Error::Internal(err.message)),
                    Err(timeout) => Err(Error::Internal(format!(
                        "close timed out after {}ms",
                        timeout.as_millis()
                    ))),
                }
            } else {
                Ok(())
            };
            // The env (and with it the lane threads and a foreground actor)
            // drops here, which ends a foreground runner.
            drop(environment);
            serve_failure(serve_result, close_result)
        })
    };

    if matches!(runner, Runner::None) {
        return serve();
    }
    let server = std::thread::Builder::new()
        .name("rlmesh-env-server".to_string())
        .spawn(serve)
        .map_err(|err| Error::Internal(format!("failed to spawn server thread: {err}")))?;
    runner.run();
    server.join().unwrap_or_else(|_| {
        Err(Error::Internal(
            "serve failed: the server thread panicked".into(),
        ))
    })
}

/// The one serve-failure text (ENV-P1): `serve failed: <message>`.
fn serve_failure(serve: Result<()>, close: Result<()>) -> Result<()> {
    let text = |err: Error| match err {
        Error::Internal(message) => message,
        other => other.to_string(),
    };
    match (serve, close) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(err), Ok(())) | (Ok(()), Err(err)) => {
            Err(Error::Internal(format!("serve failed: {}", text(err))))
        }
        (Err(serve), Err(close)) => Err(Error::Internal(format!(
            "serve failed: {}; close failed: {}",
            text(serve),
            text(close)
        ))),
    }
}

/// Release a never-served server: drop the listener and close the envs (when
/// `close_env_on_shutdown`), bounded by `close_timeout`.
fn cleanup_ready(resources: Resources) -> Result<()> {
    let Resources {
        served,
        runtime,
        listener,
        close_timeout,
        close_env_on_shutdown,
        ..
    } = resources;
    listener.release();
    drop(runtime);
    if !close_env_on_shutdown {
        return Ok(());
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("rlmesh-env-close".to_string())
        .spawn(move || {
            let result = match served {
                Served::Lanes(mut lanes) => lanes.iter_mut().try_for_each(|lane| {
                    lane.close().map_err(|err| {
                        super::sync_wire::authored_failure("close", &err).to_string()
                    })
                }),
                Served::Vector(mut env) => env
                    .close()
                    .map_err(|err| super::sync_wire::authored_failure("close", &err).to_string()),
            };
            let _ = tx.send(result);
        })
        .map_err(|err| Error::Internal(format!("failed to spawn a close thread: {err}")))?;
    match rx.recv_timeout(close_timeout) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(message)) => Err(Error::Internal(format!(
            "serve failed: {}",
            message.trim_start_matches("runtime error: ")
        ))),
        Err(_) => Err(Error::Internal(format!(
            "serve failed: close timed out after {}ms",
            close_timeout.as_millis()
        ))),
    }
}
