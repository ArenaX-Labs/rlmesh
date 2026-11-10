//! The concrete [`Model`]: a shared handle over one [`Policy`], its episode
//! store, its resolved declarations and its hooks, with every corner and hook
//! call serialized through one FIFO queue run on a plain library thread.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use crate::lifecycle::ShutdownTrigger;

use super::policy::{Policy, SpecArg};
use crate::{Error, Kwargs, Result, ServeOptions};

mod calls;
mod erased;
mod queue;
mod serve;

pub(crate) use calls::Row;
pub use erased::{DynPolicy, PolicyDecl};
pub use queue::RunSlot;
pub(crate) use queue::Shared;
pub(crate) use serve::run_local;

use calls::Resolved;
use erased::{Erased, check_native_chunk};
use serve::serve_blocking;

/// The cap on a `None` drain or close timeout, which a model serve takes
/// too.
const SERVE_GRACE: Duration = Duration::from_secs(5);

/// Options for [`Model::new`].
#[derive(Default)]
#[allow(clippy::type_complexity)] // Python's hook signatures, spelled out (MDL-21).
pub struct ModelOptions {
    /// Replaces [`Policy::spec`]; `None` keeps the policy's.
    pub spec: Option<SpecArg>,
    /// Replaces [`Policy::on_episode_end`].
    pub on_episode_end: Option<Box<dyn FnMut(&str) -> Result<()> + Send>>,
    /// Replaces [`Policy::close`].
    pub on_close: Option<Box<dyn FnOnce() -> Result<()> + Send>>,
    /// Allow `module:callable` custom-input entrypoints in the spec.
    pub trust_entrypoints: bool,
}

impl fmt::Debug for ModelOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelOptions")
            .field("spec", &self.spec)
            .field("on_episode_end", &self.on_episode_end.is_some())
            .field("on_close", &self.on_close.is_some())
            .field("trust_entrypoints", &self.trust_entrypoints)
            .finish()
    }
}

/// Options for [`Model::serve`], including cancellation and native extras.
#[derive(Clone, Debug, Default)]
pub struct ModelServeOptions {
    /// The participant the model joins the session at `address` as:
    /// `serve("lab:7447", name: "policy")` is reached at `lab:7447/policy`.
    /// `None` reads it off `address` (`"lab:7447/policy"`).
    pub name: Option<String>,
    /// The transport lifecycle policy; `None` drain and close timeouts take
    /// a 5 s cap.
    pub options: Option<ServeOptions>,
    /// Cancelling it stops the server (drain, close), and `serve` fails with
    /// the cancellation class.
    pub cancel: Option<crate::ext::CancelToken>,
    /// The Rust- and C-only extras.
    pub ext: Option<crate::ext::ModelServeOptions>,
    /// How a session-served model coalesces predicts; `None` is
    /// [`BatchOptions::default`](crate::BatchOptions). Only a session
    /// address batches.
    pub batching: Option<crate::BatchOptions>,
}

/// A loaded model built from a [`Policy`] or predict closure. A shared
/// handle: clones drive the same model, and the last one dropped closes it
/// if [`close`](Model::close) has not run.
///
/// Every corner and hook call on one model, across every run and serve on
/// it, runs one at a time, in submission order, on a library thread. A
/// corner or hook that calls back into its own model fails with the
/// `RuntimeError` class instead of deadlocking.
#[derive(Clone)]
pub struct Model {
    handle: Arc<Handle>,
}

struct Handle {
    shared: Arc<Shared>,
}

impl Drop for Handle {
    fn drop(&mut self) {
        if self.shared.life.is_closed() {
            return;
        }
        let shared = Arc::clone(&self.shared);
        let wait = !shared.on_worker() && !crate::env::server::in_async_context();
        let (done, finished) = mpsc::sync_channel(1);
        let _ = std::thread::Builder::new()
            .name("rlmesh-model-close".to_string())
            .spawn(move || {
                let _ = shared.close(None);
                let _ = done.send(());
            });
        if wait {
            let _ = finished.recv_timeout(SERVE_GRACE);
        }
    }
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let decl = &self.handle.shared.decl;
        f.debug_struct("Model")
            .field("policy", &decl.type_name)
            .field("corners", &decl.corners.names())
            .field("spec", &decl.spec)
            .field("native_chunk", &decl.native_chunk)
            .finish()
    }
}

/// Check a config against `load`'s declared keys: a missing required key,
/// then the unlisted keys.
fn check_config(keys: Option<&[(String, bool)]>, config: &Kwargs, from_config: bool) -> Result<()> {
    let Some(keys) = keys else {
        return Ok(());
    };
    if let Some((name, _)) = keys
        .iter()
        .find(|(name, required)| *required && !config.contains_key(name))
    {
        return Err(if from_config {
            Error::invalid_argument(format!("missing required parameter: '{name}'"))
        } else {
            Error::Type(format!("load() missing required argument: '{name}'"))
        });
    }
    let mut unknown: Vec<&str> = config
        .keys()
        .filter(|key| !keys.iter().any(|(name, _)| name == key))
        .collect();
    if !unknown.is_empty() {
        unknown.sort_unstable();
        return Err(Error::invalid_argument(format!(
            "extra='passthrough' but load has no **kwargs to forward: {}",
            unknown.join(", ")
        )));
    }
    Ok(())
}

/// Construct a model, in the native order: declarations, config keys,
/// `load`, the loaded native chunk. On any failure no hook runs.
pub(crate) fn construct(
    mut policy: Box<dyn DynPolicy>,
    decl: PolicyDecl,
    options: ModelOptions,
    config: Option<Kwargs>,
) -> Result<Model> {
    decl.validate()?;
    let from_config = config.is_some();
    let config = config.unwrap_or_default();
    check_config(decl.load_keys.as_deref(), &config, from_config)?;
    policy.load(&config)?;
    let native_chunk = policy.native_chunk();
    check_native_chunk(native_chunk, decl.corners)?;
    let allow_fusion = policy.allow_fusion();
    let reset_options = policy.has_reset_options();
    let execution_horizon = policy.execution_horizon();
    if let Some(horizon) = execution_horizon {
        crate::run::check_horizon(horizon, native_chunk)?;
    }
    let ModelOptions {
        spec,
        on_episode_end,
        on_close,
        trust_entrypoints,
    } = options;
    let resolved = Resolved {
        corners: decl.corners,
        context: decl.context,
        static_native_chunk: decl.native_chunk,
        native_chunk,
        allow_fusion,
        spec: spec.or(decl.spec),
        workflow_edition: decl.workflow_edition,
        authored: decl.authored,
        type_name: decl.type_name,
        load_keys: decl.load_keys,
        trust_entrypoints,
        reset_options,
        execution_horizon,
    };
    let shared = Shared::start(policy, resolved, on_episode_end, on_close)?;
    Ok(Model {
        handle: Arc::new(Handle { shared }),
    })
}

impl Model {
    /// Build a model from `source`, loading it now with an empty config. A
    /// closure is a policy too (`Model::new(|obs| Ok(..),
    /// ModelOptions::default())`).
    pub fn new<P: Policy>(source: P, options: ModelOptions) -> Result<Model> {
        construct(
            Box::new(Erased(source)),
            PolicyDecl::of::<P>(),
            options,
            None,
        )
    }

    /// Build `P::default()` and load it with `config`.
    pub fn from_config<P: Policy + Default>(config: Kwargs) -> Result<Model> {
        construct(
            Box::new(Erased(P::default())),
            PolicyDecl::of::<P>(),
            ModelOptions::default(),
            Some(config),
        )
    }

    pub(crate) fn shared(&self) -> &Arc<Shared> {
        &self.handle.shared
    }

    /// The resolved spec: the options' or the policy's.
    pub fn spec(&self) -> Option<&SpecArg> {
        self.shared().decl.spec.as_ref()
    }

    /// Whether a grouped predict may fuse lanes across envs.
    pub fn allow_fusion(&self) -> bool {
        self.shared().decl.allow_fusion
    }

    /// The native chunk length, read after `load`.
    pub fn native_chunk(&self) -> Option<u32> {
        self.shared().decl.native_chunk
    }

    /// The checkpoint's execution horizon, read once after `load`.
    pub fn execution_horizon(&self) -> Option<u32> {
        self.shared().decl.execution_horizon
    }

    /// The policy's workflow edition declaration.
    pub fn workflow_edition(&self) -> Option<&str> {
        self.shared().decl.workflow_edition.as_deref()
    }

    /// Release the model: stop every serve still running on it (drain),
    /// wait for queued calls, then fire the close hook, at most once. Later
    /// corner calls, runs and serves fail with the `RuntimeError` class.
    pub fn close(&self) -> Result<()> {
        if crate::env::server::in_async_context() {
            return Err(Error::Runtime(
                "Model::close() blocks; call close_async() from async code".to_string(),
            ));
        }
        self.shared().close(None)
    }

    /// The async twin of [`close`](Model::close).
    pub async fn close_async(&self) -> Result<()> {
        let shared = Arc::clone(self.shared());
        shared.check_not_reentrant("close_async()")?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        std::thread::Builder::new()
            .name("rlmesh-model-close".to_string())
            .spawn(move || {
                let _ = tx.send(shared.close(None));
            })
            .map_err(|err| Error::Internal(format!("failed to spawn the close thread: {err}")))?;
        rx.await
            .unwrap_or_else(|_| Err(Error::Internal("the model close was dropped".to_string())))
    }

    /// Serve the model on `address` until the server stops, then close it.
    /// SIGINT stops it (drain, close) and fails with the cancellation class;
    /// SIGTERM stops it and returns `Ok`.
    pub fn serve(&self, address: &str, options: ModelServeOptions) -> Result<()> {
        self.serve_host(address, options, true)
    }

    pub(crate) fn serve_host(
        &self,
        address: &str,
        options: ModelServeOptions,
        foreground: bool,
    ) -> Result<()> {
        if crate::env::server::in_async_context() {
            return Err(Error::Runtime(
                "Model::serve() blocks; call serve_async() from async code".to_string(),
            ));
        }
        self.shared().check_not_reentrant("serve()")?;
        serve_blocking(self.shared(), address, options, foreground, None)
    }

    /// The async twin of [`serve`](Model::serve): installs no signal
    /// handlers. Dropping the future after its first poll stops the server;
    /// the drain and the close finish on a library thread.
    pub fn serve_async(
        &self,
        address: &str,
        options: ModelServeOptions,
    ) -> impl std::future::Future<Output = Result<()>> + Send + 'static {
        let shared = Arc::clone(self.shared());
        let address = address.to_string();
        async move {
            shared.check_not_reentrant("serve_async()")?;
            if options
                .cancel
                .as_ref()
                .is_some_and(crate::ext::CancelToken::take)
            {
                return Err(Error::Cancelled);
            }
            let dropped = crate::ext::CancelToken::new();
            let guard = DropCancels(Some(dropped.clone()));
            let (tx, rx) = tokio::sync::oneshot::channel();
            std::thread::Builder::new()
                .name("rlmesh-model-server".to_string())
                .spawn(move || {
                    let _ = tx.send(serve_blocking(
                        &shared,
                        &address,
                        options,
                        false,
                        Some(dropped),
                    ));
                })
                .map_err(|err| {
                    Error::Internal(format!("failed to spawn the server thread: {err}"))
                })?;
            let result = rx.await.unwrap_or_else(|_| {
                Err(Error::Internal(
                    "serve failed: the server thread stopped".to_string(),
                ))
            });
            guard.disarm();
            result
        }
    }
}

/// Fires its token when dropped armed: a dropped `serve_async` future.
struct DropCancels(Option<crate::ext::CancelToken>);

impl DropCancels {
    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for DropCancels {
    fn drop(&mut self) {
        if let Some(token) = self.0.take() {
            token.cancel();
        }
    }
}

/// Watches cancel tokens and stops a server when one fires.
pub(crate) struct CancelWatcher {
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl CancelWatcher {
    pub(crate) fn start(
        tokens: Vec<crate::ext::CancelToken>,
        trigger: ShutdownTrigger,
    ) -> CancelWatcher {
        let tokens: Vec<_> = tokens
            .iter()
            .map(crate::ext::CancelToken::subscribe)
            .collect();
        let stop = Arc::new(AtomicBool::new(false));
        let fired = Arc::new(AtomicBool::new(false));
        let (stop_flag, fired_flag) = (Arc::clone(&stop), Arc::clone(&fired));
        let thread = if tokens.is_empty() {
            None
        } else {
            std::thread::Builder::new()
                .name("rlmesh-cancel".to_string())
                .spawn(move || {
                    while !stop_flag.load(Ordering::SeqCst) {
                        if tokens.iter().any(crate::ext::CancelSubscription::fired) {
                            fired_flag.store(true, Ordering::SeqCst);
                            trigger.trigger("cancelled");
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(20));
                    }
                })
                .ok()
        };
        CancelWatcher {
            stop,
            fired,
            thread,
        }
    }

    /// Stop watching; whether a token fired.
    pub(crate) fn stop(mut self) -> bool {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        self.fired.load(Ordering::SeqCst)
    }
}

impl Drop for CancelWatcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
