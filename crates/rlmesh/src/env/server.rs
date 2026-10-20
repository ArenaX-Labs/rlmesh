//! One non-generic environment server for an env, a list of lane envs, or a
//! vector env.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

/// The bound an env server puts on a drain or close the options leave unset.
const DEFAULT_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Endpoint transport: TCP or Unix-domain sockets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Transport {
    /// TCP.
    Tcp,
    /// A Unix-domain socket.
    Unix,
}

impl Transport {
    /// The canonical spelling: `"tcp"` or `"unix"`.
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

/// Options for constructing an environment server.
///
/// Defaults: the address `tcp://127.0.0.1:0`, no serve options or tags, and
/// `close_env_on_shutdown` true. Build one with `..Default::default()`;
/// appending a field is a minor change by project policy.
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
    /// Rust-only fields.
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

/// A factory serve's describe publication: the factory's full envelope, or
/// why it could not be built (step 5 is best-effort).
pub(crate) struct Publish {
    pub(crate) envelope: Result<String>,
    pub(crate) peer_runtime: Option<rlmesh_grpc::lifecycle::PeerRuntime>,
}

pub(crate) fn peer_runtime(
    language: &str,
    extra: &serde_json::Map<String, serde_json::Value>,
) -> rlmesh_grpc::lifecycle::PeerRuntime {
    rlmesh_grpc::lifecycle::PeerRuntime {
        language: extra
            .get("language")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(language)
            .into(),
        language_version: extra
            .get("language_version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .into(),
        framework_versions: extra
            .get("framework_versions")
            .and_then(serde_json::Value::as_object)
            .into_iter()
            .flatten()
            .filter_map(|(key, value)| value.as_str().map(|v| (key.clone(), v.into())))
            .collect(),
    }
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
    Background(Arc<Drain>),
    Stopped,
}

struct Shared {
    id: u64,
    address: String,
    contract: EnvContract,
    binding_description: Mutex<Option<String>>,
    token: Option<String>,
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
    /// Serve one env.
    pub fn new<E: Env + Send + 'static>(env: E, options: EnvServerOptions) -> Result<EnvServer> {
        Self::build(Served::Lanes(vec![Box::new(env)]), options, None, false)
    }

    /// Serve `envs` as the lanes of one `num_envs = N` endpoint, each on its
    /// own thread. Use `Box<dyn Env + Send>` to mix env types.
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
            false,
        )
    }

    /// Serve a vector env. A vector env with `num_envs == 1` is served as a
    /// single scalar lane.
    pub fn vector<V: VectorEnv + Send + 'static>(
        env: V,
        options: EnvServerOptions,
    ) -> Result<EnvServer> {
        Self::build(Served::Vector(Box::new(env)), options, None, false)
    }

    /// A factory serve's server: the edition was resolved by the caller.
    pub(crate) fn lanes_resolved(
        envs: Vec<Box<dyn Env + Send>>,
        options: EnvServerOptions,
        publish: Publish,
    ) -> Result<EnvServer> {
        Self::build(Served::Lanes(envs), options, Some(publish), true)
    }

    pub(crate) fn vector_resolved(
        env: Box<dyn VectorEnv + Send>,
        options: EnvServerOptions,
        publish: Publish,
    ) -> Result<EnvServer> {
        Self::build(Served::Vector(env), options, Some(publish), true)
    }

    /// Override publication identity for an in-repo non-Rust binding, before
    /// starting the server. The facade still builds and validates the envelope.
    pub(crate) fn binding_host(
        &self,
        component: &'static str,
        language: &'static str,
        extra: serde_json::Map<String, serde_json::Value>,
    ) -> Result<()> {
        let mut state = lock(&self.shared.state);
        let State::Ready(resources) = &mut *state else {
            return Err(Error::Runtime("server has already started".into()));
        };
        resources.grpc.peer_runtime = Some(peer_runtime(language, &extra));
        let envelope = env_envelope(
            &self.shared.contract,
            None,
            None,
            &DescribingHost {
                component,
                language,
                extra,
            },
            resources.grpc.workflow_edition.as_deref(),
            matches!(resources.served, Served::Vector(_)),
        )?;
        *lock(&self.shared.binding_description) = Some(envelope.clone());
        if resources
            .grpc
            .peer_info_extra
            .contains_key(DESCRIBE_METADATA_KEY)
        {
            resources
                .grpc
                .peer_info_extra
                .insert(DESCRIBE_METADATA_KEY.into(), envelope);
        }
        Ok(())
    }

    pub(crate) fn binding_description(&self) -> Result<String> {
        lock(&self.shared.binding_description)
            .clone()
            .ok_or_else(|| Error::Runtime("server has no binding description".into()))
    }

    /// The loopback server a run or session stands up in front of a local
    /// env: bound on `127.0.0.1`, publishing nothing, never closing the env
    /// (its caller does) and taking the caller's resolved edition.
    pub(crate) fn loopback(
        env: crate::run::LoopbackEnv,
        edition: Option<String>,
    ) -> Result<EnvServer> {
        let options = EnvServerOptions {
            options: Some(ServeOptions {
                workflow_edition: edition,
                drain_timeout: Some(Duration::from_millis(100)),
                ..Default::default()
            }),
            close_env_on_shutdown: false,
            ..Default::default()
        };
        let served = match env {
            crate::run::LoopbackEnv::Env(env) => Served::Lanes(vec![env]),
            crate::run::LoopbackEnv::Vector(env) => Served::Vector(env),
        };
        Self::build(served, options, None, true)
    }

    /// The endpoint token this server checks, if any.
    pub(crate) fn token(&self) -> Option<&str> {
        self.shared.token.as_deref()
    }

    /// `factory`: a factory serve's publication, whose edition the caller
    /// resolved; `resolved_by_caller` also covers a run's loopback.
    fn build(
        served: Served,
        options: EnvServerOptions,
        factory: Option<Publish>,
        resolved_by_caller: bool,
    ) -> Result<EnvServer> {
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
                        tags.stamp(&mut metadata)?;
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
        let checked_token = token.clone().filter(|token| !token.is_empty());
        let mut peer_info_extra: BTreeMap<String, String> = ext.peer_info_extra.clone();
        match &factory {
            // A factory serve's describe is best-effort (step 5).
            Some(publish) => match &publish.envelope {
                Ok(envelope) => {
                    peer_info_extra.insert(DESCRIBE_METADATA_KEY.to_string(), envelope.clone());
                }
                Err(err) => describe_failed(err),
            },
            None if ext.publish_describe => {
                let envelope = env_envelope(
                    &contract,
                    None,
                    None,
                    &DescribingHost::rust(),
                    edition.as_deref(),
                    matches!(served, Served::Vector(_)),
                )?;
                peer_info_extra.insert(DESCRIBE_METADATA_KEY.to_string(), envelope);
            }
            None => {}
        }

        // Retain the full factory envelope for binding server introspection.
        // Rebuilding it from the contract would lose branches and variants.
        let binding_description = factory
            .as_ref()
            .and_then(|publish| publish.envelope.as_ref().ok())
            .cloned();

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
        let mut grpc = ServeOptions {
            workflow_edition: edition,
            ..serve
        }
        .to_grpc(WireExtras {
            token,
            predict_concurrency: None,
            peer_info_extra,
        });
        grpc.peer_runtime = factory
            .as_ref()
            .and_then(|publish| publish.peer_runtime.clone());
        let (done, _) = tokio::sync::watch::channel(false);
        Ok(EnvServer {
            shared: Arc::new(Shared {
                id: NEXT_SERVER.fetch_add(1, Ordering::Relaxed),
                address,
                contract,
                binding_description: Mutex::new(binding_description),
                token: checked_token,
                shutdown: ShutdownTrigger::new(),
                background_cancel: Mutex::new(std::sync::Weak::new()),
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
                abandoned: Arc::new(Mutex::new(Vec::new())),
                drain: futures::lock::Mutex::new(()),
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

/// Normalize the bind address.
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

/// An env with tags merged into its metadata.
pub(crate) struct Tagged {
    env: Box<dyn Env + Send>,
    metadata: crate::Metadata,
}

impl Tagged {
    /// Merge and validate `tags` against `env`'s spaces.
    pub(crate) fn new(env: Box<dyn Env + Send>, tags: EnvTags) -> Result<Tagged> {
        tags.validate(&env.observation_space(), &env.action_space())?;
        Self::unchecked(env, tags)
    }

    /// Merge `tags` without validation.
    pub(crate) fn unchecked(env: Box<dyn Env + Send>, tags: EnvTags) -> Result<Tagged> {
        let mut metadata = env.metadata().unwrap_or_default();
        tags.stamp(&mut metadata)?;
        Ok(Tagged { env, metadata })
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
    env: Option<E>,
}

impl<E> Drop for Marked<E> {
    fn drop(&mut self) {
        in_callback(self.id, || drop(self.env.take()));
    }
}

impl<E: Env + ?Sized> Env for Marked<Box<E>> {
    fn observation_space(&self) -> crate::spaces::Space {
        self.env.as_ref().expect("marked env").observation_space()
    }
    fn action_space(&self) -> crate::spaces::Space {
        self.env.as_ref().expect("marked env").action_space()
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&crate::InfoDict>,
    ) -> Result<(crate::Value, crate::InfoDict)> {
        in_callback(self.id, || {
            self.env.as_mut().expect("marked env").reset(seed, options)
        })
    }
    fn step(
        &mut self,
        action: Option<crate::Value>,
    ) -> Result<(crate::Value, f64, bool, bool, crate::InfoDict)> {
        in_callback(self.id, || {
            self.env.as_mut().expect("marked env").step(action)
        })
    }
    fn render(&mut self) -> Result<Option<crate::Value>> {
        in_callback(self.id, || self.env.as_mut().expect("marked env").render())
    }
    fn close(&mut self) -> Result<()> {
        in_callback(self.id, || self.env.as_mut().expect("marked env").close())
    }
}

impl<V: VectorEnv + ?Sized> VectorEnv for Marked<Box<V>> {
    fn num_envs(&self) -> u64 {
        self.env.as_ref().expect("marked env").num_envs()
    }
    fn single_observation_space(&self) -> crate::spaces::Space {
        self.env
            .as_ref()
            .expect("marked env")
            .single_observation_space()
    }
    fn single_action_space(&self) -> crate::spaces::Space {
        self.env.as_ref().expect("marked env").single_action_space()
    }
    fn reset(
        &mut self,
        seed: Option<super::VectorSeed>,
        options: Option<&crate::InfoDict>,
    ) -> Result<(crate::Value, crate::InfoDict)> {
        in_callback(self.id, || {
            self.env.as_mut().expect("marked env").reset(seed, options)
        })
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
        in_callback(self.id, || {
            self.env.as_mut().expect("marked env").step(actions)
        })
    }
    fn render(&mut self) -> Result<Option<crate::Value>> {
        in_callback(self.id, || self.env.as_mut().expect("marked env").render())
    }
    fn close(&mut self) -> Result<()> {
        in_callback(self.id, || self.env.as_mut().expect("marked env").close())
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

/// A background factory serve owns its cancellation subscription until serving
/// ends. Joining the serve thread releases the watcher before returning its result.
struct CancelProbe {
    subscription: crate::ext::CancelSubscription,
    trigger: ShutdownTrigger,
    cancelled: Mutex<bool>,
}
impl CancelProbe {
    fn sample(&self) {
        let mut cancelled = lock(&self.cancelled);
        if !*cancelled && self.subscription.fired() && self.trigger.trigger("cancelled") {
            *cancelled = true;
        }
    }
}
impl Shared {
    fn trigger_shutdown(&self, reason: &str) {
        // A firing before this call must compete synchronously, even between
        // watcher polls. Once another trigger wins, later firings cannot alter
        // the serve result, including firings during drain/close.
        if let Some(probe) = lock(&self.background_cancel).upgrade() {
            probe.sample();
        }
        self.shutdown.trigger(reason);
    }
}
struct BackgroundCancel {
    stop: Arc<std::sync::atomic::AtomicBool>,
    probe: Arc<CancelProbe>,
    thread: Option<JoinHandle<()>>,
}
impl BackgroundCancel {
    fn start(
        subscription: crate::ext::CancelSubscription,
        trigger: ShutdownTrigger,
    ) -> Result<Self> {
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop = Arc::new(AtomicBool::new(false));
        let probe = Arc::new(CancelProbe {
            subscription,
            trigger,
            cancelled: Mutex::new(false),
        });
        let (stop_flag, watched) = (Arc::clone(&stop), Arc::clone(&probe));
        let thread = std::thread::Builder::new()
            .name("rlmesh-factory-cancel".into())
            .spawn(move || {
                while !stop_flag.load(Ordering::SeqCst) {
                    watched.sample();
                    std::thread::sleep(Duration::from_millis(20));
                }
                // Retain the subscription through drain/close, so later firings
                // remain consumed without changing the selected serve result.
            })
            .map_err(|err| {
                Error::Internal(format!("failed to spawn cancellation watcher: {err}"))
            })?;
        Ok(Self {
            stop,
            probe,
            thread: Some(thread),
        })
    }
    fn stop(mut self) -> bool {
        self.finish();
        *lock(&self.probe.cancelled)
    }
    fn finish(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Drop for BackgroundCancel {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Serve until the shutdown trigger fires, then drain and close. In the
/// foreground the calling thread runs lane 0's (or the vector env's) jobs
/// while the transport runs on a helper thread.
fn run(resources: Resources, shared: &Arc<Shared>, foreground: bool) -> Result<()> {
    run_inner(resources, shared, foreground)
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
                let lane: Box<dyn Env + Send> = Box::new(Marked {
                    id,
                    env: Some(lane),
                });
                if foreground && index == 0 {
                    let (relayed, lane_runner) = SyncLane::relayed(lane, &contract)?;
                    wire_lanes.push(relayed);
                    runner = Runner::Lane(lane_runner);
                } else {
                    wire_lanes.push(SyncLane::new(lane, &contract)?);
                }
            }
            let mut adapter = WireLaneAdapter::new(wire_lanes)
                .map_err(|err| Error::invalid_argument(err.to_string()))?;
            adapter.track_abandoned(Arc::clone(&shared.abandoned));
            drive(adapter, runner, parts)
        }
        Served::Vector(env) => {
            let env: Box<dyn VectorEnv + Send> = Box::new(Marked { id, env: Some(env) });
            let (actor, runner) = if foreground {
                let (actor, runner) = Actor::foreground(env);
                (actor, Runner::Vector(runner))
            } else {
                let (actor, thread) = Actor::spawn("rlmesh-vector-env".to_string(), env)?;
                lock(&shared.abandoned).push(thread);
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
            let router = rlmesh_grpc::lifecycle::server_builder()
                .add_service(health_service)
                .add_service(service);
            let serve_result = listener
                .serve(router, shutdown.clone(), Some(drain_timeout))
                .await;
            // A transport failure also begins stopping before close callbacks.
            // Later token firings cannot replace that selected serve error.
            shutdown.trigger("serve ended");
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

/// The one serve-failure text: `serve failed: <message>`.
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
fn cleanup_ready(resources: Resources, shared: &Arc<Shared>) -> Result<()> {
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
    let id = shared.id;
    if !close_env_on_shutdown {
        in_callback(id, || drop(served));
        return Ok(());
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let thread = std::thread::Builder::new()
        .name("rlmesh-env-close".to_string())
        .spawn(move || {
            let result = in_callback(id, || match served {
                Served::Lanes(mut lanes) => lanes.iter_mut().try_for_each(|lane| {
                    in_callback(id, || lane.close()).map_err(|err| {
                        super::sync_wire::authored_failure("close", &err).to_string()
                    })
                }),
                Served::Vector(mut env) => in_callback(id, || env.close())
                    .map_err(|err| super::sync_wire::authored_failure("close", &err).to_string()),
            });
            let _ = tx.send(result);
        })
        .map_err(|err| Error::Internal(format!("failed to spawn a close thread: {err}")))?;
    let result = rx.recv_timeout(close_timeout);
    if result.is_ok() {
        let _ = thread.join();
    } else {
        lock(&shared.abandoned).push(thread);
    }
    match result {
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

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    use crate::ext::CancelToken;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ClosingEnv {
        token: CancelToken,
        closes: Arc<AtomicUsize>,
        cancel_on_close: bool,
        fail_close: bool,
        close_gate: Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>,
    }
    impl Env for ClosingEnv {
        fn observation_space(&self) -> crate::spaces::Space {
            crate::spaces::Discrete::new(2, Default::default())
                .unwrap()
                .into()
        }
        fn action_space(&self) -> crate::spaces::Space {
            self.observation_space()
        }
        fn reset(
            &mut self,
            _: Option<i64>,
            _: Option<&crate::InfoDict>,
        ) -> Result<(crate::Value, crate::InfoDict)> {
            Ok((0.into(), Default::default()))
        }
        fn step(
            &mut self,
            _: Option<crate::Value>,
        ) -> Result<(crate::Value, f64, bool, bool, crate::InfoDict)> {
            Ok((0.into(), 0.0, true, false, Default::default()))
        }
        fn close(&mut self) -> Result<()> {
            self.closes.fetch_add(1, Ordering::SeqCst);
            if let Some((entered, release)) = self.close_gate.take() {
                entered.send(()).unwrap();
                release.recv_timeout(Duration::from_secs(5)).unwrap();
            }
            if self.cancel_on_close {
                self.token.cancel();
                std::thread::sleep(Duration::from_millis(60));
            }
            if self.fail_close {
                Err(Error::Internal("close failed".into()))
            } else {
                Ok(())
            }
        }
    }
    fn server(
        token: &CancelToken,
        cancel_on_close: bool,
        fail_close: bool,
        idle: bool,
    ) -> (EnvServer, Arc<AtomicUsize>) {
        let closes = Arc::new(AtomicUsize::new(0));
        let server = EnvServer::new(
            ClosingEnv {
                token: token.clone(),
                closes: Arc::clone(&closes),
                cancel_on_close,
                fail_close,
                close_gate: None,
            },
            EnvServerOptions {
                address: Some("tcp://127.0.0.1:0".into()),
                options: Some(crate::ServeOptions {
                    workflow_edition: Some(crate::current_workflow_edition().into()),
                    idle_timeout: idle.then_some(Duration::from_millis(20)),
                    drain_timeout: Some(Duration::from_millis(50)),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .unwrap();
        (server, closes)
    }
    #[test]
    fn subscribed_start_samples_before_local_shutdown_and_reports_once() {
        let token = CancelToken::new();
        let (server, closes) = server(&token, false, false, false);
        server.start_subscribed(Some(token.subscribe())).unwrap();
        token.cancel();
        assert!(matches!(server.shutdown(), Err(Error::Cancelled)));
        assert!(server.wait(Some(Duration::ZERO)).unwrap());
        server.shutdown().unwrap();
        assert_eq!(closes.load(Ordering::SeqCst), 1);
        token.cancel();
        assert!(token.take(), "terminal path releases its subscription");
    }
    #[test]
    fn subscribed_cancellation_joins_a_shared_drain_and_reports_once() {
        use std::future::Future;
        use std::sync::mpsc;

        let token = CancelToken::new();
        let closes = Arc::new(AtomicUsize::new(0));
        let (entered, closing) = mpsc::channel();
        let (release, held) = mpsc::channel();
        let server = Arc::new(
            EnvServer::new(
                ClosingEnv {
                    token: token.clone(),
                    closes: Arc::clone(&closes),
                    cancel_on_close: false,
                    fail_close: false,
                    close_gate: Some((entered, held)),
                },
                EnvServerOptions::default(),
            )
            .unwrap(),
        );
        server.start_subscribed(Some(token.subscribe())).unwrap();
        token.cancel();
        let mut shutdown = Box::pin(server.shutdown_async());
        let waker = futures::task::noop_waker();
        assert!(
            shutdown
                .as_mut()
                .poll(&mut std::task::Context::from_waker(&waker))
                .is_pending()
        );
        closing.recv_timeout(Duration::from_secs(5)).unwrap();
        let (completed, observed) = mpsc::channel();
        let second = {
            let server = Arc::clone(&server);
            std::thread::spawn(move || completed.send(server.shutdown()).unwrap())
        };
        let early = observed.recv_timeout(Duration::from_millis(50));
        let polled = server.wait(Some(Duration::ZERO));
        release.send(()).unwrap();
        second.join().unwrap();
        let sync_result = observed.recv_timeout(Duration::from_secs(5)).unwrap();
        let async_result = futures::executor::block_on(shutdown);
        assert!(matches!(early, Err(mpsc::RecvTimeoutError::Timeout)));
        assert!(!polled.unwrap());
        assert!(matches!(sync_result, Err(Error::Cancelled)));
        async_result.unwrap();
        assert!(server.wait(Some(Duration::ZERO)).unwrap());
        server.shutdown().unwrap();
        assert_eq!(closes.load(Ordering::SeqCst), 1);
        token.cancel();
        assert!(token.take(), "joined drain releases its subscription");
    }
    #[test]
    fn local_and_idle_stops_ignore_cancellation_during_close() {
        for idle in [false, true] {
            let token = CancelToken::new();
            let (server, closes) = server(&token, true, false, idle);
            server.start_subscribed(Some(token.subscribe())).unwrap();
            if idle {
                assert!(server.wait(Some(Duration::from_secs(2))).unwrap());
            }
            server.shutdown().unwrap();
            assert_eq!(closes.load(Ordering::SeqCst), 1);
            assert!(!token.take(), "late firing is consumed through cleanup");
            token.cancel();
            assert!(token.take());
        }
    }
    #[test]
    fn late_cancellation_does_not_replace_a_close_error() {
        let token = CancelToken::new();
        let (server, _) = server(&token, true, true, false);
        server.start_subscribed(Some(token.subscribe())).unwrap();
        assert!(matches!(server.shutdown(), Err(Error::Internal(_))));
    }
    #[test]
    fn dropped_subscribed_server_closes_and_releases_subscription() {
        let token = CancelToken::new();
        let (server, closes) = server(&token, true, false, false);
        server.start_subscribed(Some(token.subscribe())).unwrap();
        drop(server);
        assert_eq!(closes.load(Ordering::SeqCst), 1);
        assert!(!token.take());
        token.cancel();
        assert!(token.take());
    }
    #[test]
    fn drop_samples_a_firing_before_its_local_stop() {
        let token = CancelToken::new();
        let (server, closes) = server(&token, false, false, false);
        server.start_subscribed(Some(token.subscribe())).unwrap();
        let trigger = server.shutdown_handle();
        token.cancel();
        drop(server);
        assert_eq!(trigger.reason().as_deref(), Some("cancelled"));
        assert_eq!(closes.load(Ordering::SeqCst), 1);
        token.cancel();
        assert!(token.take());
    }
    #[test]
    fn refused_subscribed_start_releases_only_its_own_subscription() {
        let token = CancelToken::new();
        let other = CancelToken::new();
        let (server, _) = server(&token, false, false, false);
        server.start_subscribed(Some(token.subscribe())).unwrap();
        assert!(server.start_subscribed(Some(other.subscribe())).is_err());
        other.cancel();
        assert!(other.take());
        token.cancel();
        assert!(matches!(server.shutdown(), Err(Error::Cancelled)));
    }
    #[test]
    fn background_cancel_records_only_the_winning_trigger() {
        for cancel_first in [false, true] {
            let token = CancelToken::new();
            let trigger = ShutdownTrigger::new();
            let watcher = BackgroundCancel::start(token.subscribe(), trigger.clone()).unwrap();
            if !cancel_first {
                assert!(trigger.trigger("remote shutdown"));
            }
            token.cancel();
            watcher.probe.sample();
            assert_eq!(watcher.stop(), cancel_first);
            assert_eq!(
                trigger.reason().as_deref(),
                Some(if cancel_first {
                    "cancelled"
                } else {
                    "remote shutdown"
                })
            );
            assert!(!token.take());
            token.cancel();
            assert!(token.take());
        }
    }
    #[test]
    fn early_serve_error_is_not_replaced_while_retiring_the_watcher() {
        use std::sync::atomic::AtomicBool;
        let token = CancelToken::new();
        let (server, _) = server(&token, false, false, false);
        let stop = Arc::new(AtomicBool::new(false));
        let probe = Arc::new(CancelProbe {
            subscription: token.subscribe(),
            trigger: server.shutdown_handle(),
            cancelled: Mutex::new(false),
        });
        let (stopping, watched) = (Arc::clone(&stop), Arc::clone(&probe));
        let thread = std::thread::spawn(move || {
            while !stopping.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            token.cancel();
            watched.sample();
        });
        let watcher = BackgroundCancel {
            stop,
            probe,
            thread: Some(thread),
        };
        let result = background_outcome(
            Err(Error::Internal("injected early serve failure".into())),
            &server.shared,
            Some(watcher),
        );
        assert!(
            matches!(result, Err(Error::Internal(message)) if message == "injected early serve failure")
        );
        assert_eq!(
            server.shared.shutdown.reason().as_deref(),
            Some("serve ended")
        );
    }
    #[test]
    fn watcher_startup_failure_closes_ready_envs_and_notifies_waiters() {
        let token = CancelToken::new();
        let (server, closes) = server(&token, false, true, false);
        let mut done = server.shared.done.subscribe();
        assert!(!*done.borrow_and_update());
        let result = server.spawn_background_with_watcher(
            "start",
            State::Starting,
            Some(token.subscribe()),
            |_, _| Err(Error::Internal("injected watcher spawn failure".into())),
        );
        assert!(
            matches!(result, Err(Error::Internal(message)) if message == "injected watcher spawn failure")
        );
        assert_eq!(closes.load(Ordering::SeqCst), 1);
        assert!(done.has_changed().unwrap(), "wake wait_async after cleanup");
        assert!(*done.borrow_and_update());
        assert!(server.wait(Some(Duration::ZERO)).unwrap());
        server.shutdown().unwrap();
        assert!(server.start().is_err());
        token.cancel();
        assert!(token.take(), "failed startup releases its subscription");
    }
    #[test]
    fn rust_factory_leaves_endpoint_runtime_identity_unset() {
        struct Factory;
        impl super::super::EnvFactory for Factory {
            fn make(&mut self, _: &crate::Kwargs) -> Result<super::super::MadeEnv> {
                Ok(super::super::MadeEnv::env(ClosingEnv {
                    token: CancelToken::new(),
                    closes: Arc::new(AtomicUsize::new(0)),
                    cancel_on_close: false,
                    fail_close: false,
                    close_gate: None,
                }))
            }
        }
        for num_envs in [1, 2] {
            let server = super::super::factory::prepare_serve(
                &mut Factory,
                "tcp://127.0.0.1:0",
                &crate::EnvFactoryServeOptions {
                    num_envs,
                    workflow_edition: Some(crate::current_workflow_edition().into()),
                    ..Default::default()
                },
                &crate::describe::Host::rust(),
            )
            .unwrap();
            let state = lock(&server.shared.state);
            let State::Ready(resources) = &*state else {
                panic!("expected ready server")
            };
            assert!(resources.grpc.peer_runtime.is_none());
        }
    }
}
