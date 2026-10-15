//! The concrete [`Model`]: a shared handle over one [`Policy`], its episode
//! store, its resolved declarations and its hooks, with every corner and hook
//! call serialized through one FIFO queue run on a plain library thread.

use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use rlmesh_grpc::lifecycle::ShutdownTrigger;

use super::context::{BatchPredictContext, EpisodeStore, OwnedContext, PredictContext};
use super::policy::{Corners, LoadKey, Policy, SpecArg};
use super::types::EpisodeInfo;
use crate::editions::EditionSources;
use crate::{BindAddress, Error, Kwargs, Result, ServeOptions, Value};

/// ENV-7's cap on a `None` drain or close timeout, which a model serve
/// takes too.
const SERVE_GRACE: Duration = Duration::from_secs(5);

/// The options of [`Model::new`]: Python's `Model(..., spec=,
/// on_episode_end=, on_close=, trust_entrypoints=)`.
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

/// The options of [`Model::serve`]: Python's `serve(address, *, options=)`,
/// plus the cancel token (R-18) and the fenced extras (MDL-20).
#[derive(Clone, Debug, Default)]
pub struct ModelServeOptions {
    /// The transport lifecycle policy; `None` drain and close timeouts take
    /// a 5 s cap.
    pub options: Option<ServeOptions>,
    /// Cancelling it stops the server (drain, close), and `serve` fails with
    /// the cancellation class.
    pub cancel: Option<crate::ext::CancelToken>,
    /// The Rust- and C-only extras.
    pub ext: Option<crate::ext::ModelServeOptions>,
}

/// A loaded model: Python's `rlmesh.Model`, built from a [`Policy`] (or a
/// predict closure). A shared handle: clones drive the same model, and the
/// last one dropped closes it if [`close`](Model::close) has not run.
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
        if shared.on_worker() || crate::env::server::in_async_context() {
            // Never wait for our own call or block an executor: the close
            // runs on a detached library thread, after the current call.
            let _ = std::thread::Builder::new()
                .name("rlmesh-model-close".to_string())
                .spawn(move || {
                    let _ = shared.close(None);
                });
        } else {
            let _ = shared.close(None);
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

/// A policy's declarations as the library reads them: the policy type's
/// consts, or a binding's runtime declarations.
#[derive(Clone, Debug)]
pub struct PolicyDecl {
    /// The corners the policy defines.
    pub corners: Corners,
    /// The corners that receive a context.
    pub context: Corners,
    /// The static native chunk length.
    pub native_chunk: Option<u32>,
    /// The keywords `load` accepts (`None`: every key).
    pub load_keys: Option<Vec<(String, bool)>>,
    /// The policy's spec.
    pub spec: Option<SpecArg>,
    /// The policy's edition declaration.
    pub workflow_edition: Option<String>,
    /// Whether the policy is an authored participant (COR-7).
    pub authored: bool,
    /// The describe qualname's type part.
    pub type_name: String,
}

impl PolicyDecl {
    /// The declarations of policy type `P`.
    pub fn of<P: Policy>() -> PolicyDecl {
        PolicyDecl {
            corners: P::CORNERS,
            context: P::CONTEXT,
            native_chunk: P::NATIVE_CHUNK,
            load_keys: P::LOAD_KEYS.map(|keys| {
                keys.iter()
                    .map(|LoadKey { name, required }| (name.to_string(), *required))
                    .collect()
            }),
            spec: P::spec(),
            workflow_edition: P::workflow_edition(),
            authored: P::AUTHORED,
            type_name: std::any::type_name::<P>().to_string(),
        }
    }

    /// Refuse a declaration construction refuses (TypeError class, or the
    /// ValueError class for a native chunk below 1).
    pub fn validate(&self) -> Result<()> {
        let all = Corners::PREDICT
            .union(Corners::PREDICT_CHUNK)
            .union(Corners::PREDICT_BATCH)
            .union(Corners::PREDICT_CHUNK_BATCH);
        if !all.contains(self.corners) || !all.contains(self.context) {
            return Err(Error::Type(
                "corner declarations hold bits outside the four corners".to_string(),
            ));
        }
        if !self.corners.contains(self.context) {
            let extra = Corners::from_bits(self.context.bits() & !self.corners.bits());
            return Err(Error::Type(format!(
                "CONTEXT lists {} but CORNERS does not: a context corner must be one the \
                 policy defines",
                extra.names().join(", ")
            )));
        }
        if !self.corners.contains(Corners::PREDICT)
            && !self.corners.contains(Corners::PREDICT_BATCH)
        {
            return Err(Error::Type(
                "could not derive predict(): define predict(), or a corner it can be derived \
                 from (a chunk-only model on the raw Value bridge needs an explicit predict())."
                    .to_string(),
            ));
        }
        check_native_chunk(self.native_chunk, self.corners)?;
        if let Some(keys) = &self.load_keys {
            let mut seen = std::collections::HashSet::new();
            for (name, _) in keys {
                if name.is_empty() {
                    return Err(Error::Type("a load key name must not be empty".to_string()));
                }
                if !seen.insert(name.as_str()) {
                    return Err(Error::Type(format!("load key '{name}' is declared twice")));
                }
            }
        }
        Ok(())
    }
}

impl Corners {
    pub(crate) const fn from_bits(bits: u32) -> Corners {
        // Only the four corner bits are meaningful; others are refused by
        // `PolicyDecl::validate`.
        let mut corners = Corners::NONE;
        if bits & 1 != 0 {
            corners = corners.union(Corners::PREDICT);
        }
        if bits & 2 != 0 {
            corners = corners.union(Corners::PREDICT_CHUNK);
        }
        if bits & 4 != 0 {
            corners = corners.union(Corners::PREDICT_BATCH);
        }
        if bits & 8 != 0 {
            corners = corners.union(Corners::PREDICT_CHUNK_BATCH);
        }
        corners
    }
}

/// A native chunk declaration: one with no chunk corner is the TypeError
/// class (checked first), one below 1 the ValueError class.
pub(crate) fn check_native_chunk(native_chunk: Option<u32>, corners: Corners) -> Result<()> {
    let Some(native) = native_chunk else {
        return Ok(());
    };
    if !corners.has_chunk() {
        return Err(Error::Type(
            "native_chunk declares how long ONE action chunk is, but this model defines no \
             predict_chunk() / predict_chunk_batch(). Add a chunk corner, or drop native_chunk."
                .to_string(),
        ));
    }
    if native < 1 {
        return Err(Error::invalid_argument(format!(
            "native_chunk must be >= 1 (or None when undeclared), got {native}"
        )));
    }
    Ok(())
}

/// A policy behind the library's erased seam: the [`Policy`] members with
/// the context passed only to the corners that declared it. A binding with
/// runtime declarations implements it directly.
pub trait DynPolicy: Send {
    /// [`Policy::load`].
    fn load(&mut self, config: &Kwargs) -> Result<()>;
    /// [`Policy::allow_fusion`].
    fn allow_fusion(&self) -> bool;
    /// [`Policy::native_chunk`].
    fn native_chunk(&self) -> Option<u32>;
    /// `predict`, with a context when the corner reads one.
    fn predict(
        &mut self,
        observation: Value,
        context: Option<&mut PredictContext<'_>>,
    ) -> Result<Value>;
    /// `predict_batch`, with the row contexts when the corner reads them.
    fn predict_batch(
        &mut self,
        observations: Vec<Value>,
        context: Option<&mut BatchPredictContext<'_>>,
    ) -> Result<Vec<Value>>;
    /// `predict_chunk`, with a context when the corner reads one.
    fn predict_chunk(
        &mut self,
        observation: Value,
        execution_horizon: u32,
        context: Option<&mut PredictContext<'_>>,
    ) -> Result<Value>;
    /// `predict_chunk_batch`, with the row contexts when the corner reads them.
    fn predict_chunk_batch(
        &mut self,
        observations: Vec<Value>,
        execution_horizon: u32,
        context: Option<&mut BatchPredictContext<'_>>,
    ) -> Result<Vec<Value>>;
    /// [`Policy::on_episode_end`].
    fn on_episode_end(&mut self, episode_id: &str) -> Result<()>;
    /// [`Policy::close`].
    fn close(&mut self) -> Result<()>;
}

/// A [`Policy`] on the erased seam: the `_with_context` twin for a corner in
/// `P::CONTEXT`, the plain method otherwise.
struct Erased<P>(P);

impl<P: Policy> DynPolicy for Erased<P> {
    fn load(&mut self, config: &Kwargs) -> Result<()> {
        self.0.load(config)
    }

    fn allow_fusion(&self) -> bool {
        self.0.allow_fusion()
    }

    fn native_chunk(&self) -> Option<u32> {
        self.0.native_chunk()
    }

    fn predict(
        &mut self,
        observation: Value,
        context: Option<&mut PredictContext<'_>>,
    ) -> Result<Value> {
        match context {
            Some(context) if P::CONTEXT.contains(Corners::PREDICT) => {
                self.0.predict_with_context(observation, context)
            }
            _ => self.0.predict(observation),
        }
    }

    fn predict_batch(
        &mut self,
        observations: Vec<Value>,
        context: Option<&mut BatchPredictContext<'_>>,
    ) -> Result<Vec<Value>> {
        match context {
            Some(context) if P::CONTEXT.contains(Corners::PREDICT_BATCH) => {
                self.0.predict_batch_with_context(observations, context)
            }
            _ => self.0.predict_batch(observations),
        }
    }

    fn predict_chunk(
        &mut self,
        observation: Value,
        execution_horizon: u32,
        context: Option<&mut PredictContext<'_>>,
    ) -> Result<Value> {
        match context {
            Some(context) if P::CONTEXT.contains(Corners::PREDICT_CHUNK) => self
                .0
                .predict_chunk_with_context(observation, execution_horizon, context),
            _ => self.0.predict_chunk(observation, execution_horizon),
        }
    }

    fn predict_chunk_batch(
        &mut self,
        observations: Vec<Value>,
        execution_horizon: u32,
        context: Option<&mut BatchPredictContext<'_>>,
    ) -> Result<Vec<Value>> {
        match context {
            Some(context) if P::CONTEXT.contains(Corners::PREDICT_CHUNK_BATCH) => self
                .0
                .predict_chunk_batch_with_context(observations, execution_horizon, context),
            _ => self.0.predict_chunk_batch(observations, execution_horizon),
        }
    }

    fn on_episode_end(&mut self, episode_id: &str) -> Result<()> {
        self.0.on_episode_end(episode_id)
    }

    fn close(&mut self) -> Result<()> {
        self.0.close()
    }
}

/// One row's identity, owned for a queued call.
#[derive(Clone, Debug, Default)]
pub(crate) struct Row {
    pub(crate) episode_id: String,
    pub(crate) env_id: String,
    pub(crate) seed: Option<i64>,
}

impl Row {
    pub(crate) fn of(episode: Option<&EpisodeInfo>) -> Row {
        episode.map_or_else(Row::default, |episode| Row {
            episode_id: episode.episode_id.clone(),
            env_id: episode.env_id.clone(),
            seed: episode.seed,
        })
    }
}

/// The model's resolved declarations, read once after `load`.
#[derive(Debug)]
pub(crate) struct Resolved {
    pub(crate) corners: Corners,
    pub(crate) context: Corners,
    pub(crate) static_native_chunk: Option<u32>,
    pub(crate) native_chunk: Option<u32>,
    pub(crate) allow_fusion: bool,
    pub(crate) spec: Option<SpecArg>,
    pub(crate) workflow_edition: Option<String>,
    pub(crate) authored: bool,
    pub(crate) type_name: String,
    pub(crate) load_keys: Option<Vec<(String, bool)>>,
    pub(crate) trust_entrypoints: bool,
}

impl Resolved {
    /// Whether a chunk corner exists, written or derived (`predict_chunk`
    /// from `predict_chunk_batch`).
    pub(crate) fn has_chunk(&self) -> bool {
        self.corners.has_chunk()
    }
}

/// What runs on the model's library thread: the policy, its store and
/// hooks.
#[allow(clippy::type_complexity)]
pub(crate) struct Core {
    policy: Box<dyn DynPolicy>,
    corners: Corners,
    context: Corners,
    store: EpisodeStore,
    on_episode_end: Option<Box<dyn FnMut(&str) -> Result<()> + Send>>,
    on_close: Option<Box<dyn FnOnce() -> Result<()> + Send>>,
    closed: bool,
}

fn batch_len_error(corner: &str, got: usize, want: usize) -> Error {
    Error::model(format!("{corner} returned {got} actions for {want} lanes"))
}

impl Core {
    fn begin_all(&mut self, rows: &[Row]) -> Result<Vec<OwnedContext>> {
        let mut contexts = Vec::with_capacity(rows.len());
        for row in rows {
            match self.store.begin(&row.episode_id, &row.env_id, row.seed) {
                Ok(context) => contexts.push(context),
                Err(err) => {
                    for context in contexts {
                        self.store.finish(context);
                    }
                    return Err(err);
                }
            }
        }
        Ok(contexts)
    }

    fn finish_all(&mut self, contexts: Vec<OwnedContext>) {
        for context in contexts {
            self.store.finish(context);
        }
    }

    /// `predict` for one row: the policy's own, or a batch of one of
    /// `predict_batch` (row 0).
    pub(crate) fn predict(&mut self, observation: Value, row: &Row) -> Result<Value> {
        if !self.corners.contains(Corners::PREDICT) {
            let mut actions = self.predict_batch(vec![observation], std::slice::from_ref(row))?;
            if actions.len() != 1 {
                return Err(batch_len_error("predict_batch", actions.len(), 1));
            }
            return Ok(actions.remove(0));
        }
        if !self.context.contains(Corners::PREDICT) {
            return self.policy.predict(observation, None);
        }
        let mut context = self.store.begin(&row.episode_id, &row.env_id, row.seed)?;
        let result = self.policy.predict(observation, Some(&mut context.view()));
        self.store.finish(context);
        result
    }

    /// The policy's own `predict_batch`.
    pub(crate) fn predict_batch(
        &mut self,
        observations: Vec<Value>,
        rows: &[Row],
    ) -> Result<Vec<Value>> {
        if !self.context.contains(Corners::PREDICT_BATCH) {
            return self.policy.predict_batch(observations, None);
        }
        let mut contexts = self.begin_all(rows)?;
        let result = {
            let mut views: Vec<PredictContext<'_>> =
                contexts.iter_mut().map(OwnedContext::view).collect();
            self.policy
                .predict_batch(observations, Some(views.as_mut_slice()))
        };
        self.finish_all(contexts);
        result
    }

    /// `predict_chunk` for one row: the policy's own, or a batch of one of
    /// `predict_chunk_batch`.
    pub(crate) fn predict_chunk(
        &mut self,
        observation: Value,
        execution_horizon: u32,
        row: &Row,
    ) -> Result<Value> {
        if !self.corners.contains(Corners::PREDICT_CHUNK) {
            let mut chunks = self.predict_chunk_batch(
                vec![observation],
                execution_horizon,
                std::slice::from_ref(row),
            )?;
            if chunks.len() != 1 {
                return Err(batch_len_error("predict_chunk_batch", chunks.len(), 1));
            }
            return Ok(chunks.remove(0));
        }
        if !self.context.contains(Corners::PREDICT_CHUNK) {
            return self
                .policy
                .predict_chunk(observation, execution_horizon, None);
        }
        let mut context = self.store.begin(&row.episode_id, &row.env_id, row.seed)?;
        let result =
            self.policy
                .predict_chunk(observation, execution_horizon, Some(&mut context.view()));
        self.store.finish(context);
        result
    }

    /// The policy's own `predict_chunk_batch`.
    pub(crate) fn predict_chunk_batch(
        &mut self,
        observations: Vec<Value>,
        execution_horizon: u32,
        rows: &[Row],
    ) -> Result<Vec<Value>> {
        if !self.context.contains(Corners::PREDICT_CHUNK_BATCH) {
            return self
                .policy
                .predict_chunk_batch(observations, execution_horizon, None);
        }
        let mut contexts = self.begin_all(rows)?;
        let result = {
            let mut views: Vec<PredictContext<'_>> =
                contexts.iter_mut().map(OwnedContext::view).collect();
            self.policy.predict_chunk_batch(
                observations,
                execution_horizon,
                Some(views.as_mut_slice()),
            )
        };
        self.finish_all(contexts);
        result
    }

    /// An episode ended: drop its entry, then fire the hook.
    pub(crate) fn episode_end(&mut self, episode_id: &str) -> Result<()> {
        self.store.end(episode_id);
        match self.on_episode_end.as_mut() {
            Some(hook) => hook(episode_id),
            None => self.policy.on_episode_end(episode_id),
        }
    }

    /// An env's teardown: drop its entries, firing no hook.
    pub(crate) fn drop_env(&mut self, env_id: &str) {
        self.store.drop_env(env_id);
    }

    fn close_hook(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        self.store.clear();
        match self.on_close.take() {
            Some(hook) => hook(),
            None => self.policy.close(),
        }
    }
}

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
struct Life {
    state: Mutex<LifeState>,
    changed: Condvar,
}

impl Life {
    fn lock(&self) -> MutexGuard<'_, LifeState> {
        self.state.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn is_closed(&self) -> bool {
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
    life: Arc<Life>,
    refuse: AtomicBool,
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
    fn start(
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
        let job: Job = Box::new(move |core: &mut Core| {
            let out = std::panic::catch_unwind(AssertUnwindSafe(|| f(core)))
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
        if self.refuse.load(Ordering::SeqCst) {
            return Err(closed_error());
        }
        let rx = self.submit(move |core| {
            if core.closed {
                return Err(closed_error());
            }
            f(core)
        })?;
        rx.recv()
            .map_err(|_| Error::Internal("the model call was dropped".to_string()))?
    }

    /// Refuse a new run or serve once `close()` has begun.
    pub(crate) fn admit(&self) -> Result<()> {
        if self.life.lock().phase != Phase::Open {
            return Err(closed_error());
        }
        Ok(())
    }

    fn register_serve(&self, trigger: ShutdownTrigger) -> Result<u64> {
        let mut state = self.life.lock();
        if state.phase != Phase::Open {
            return Err(closed_error());
        }
        let id = state.next_serve;
        state.next_serve += 1;
        state.serves.insert(id, trigger);
        Ok(id)
    }

    fn deregister_serve(&self, id: u64) {
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
        self.refuse.store(true, Ordering::SeqCst);
        let life = Arc::clone(&self.life);
        let rx = match self.submit(move |core| {
            let result = core.close_hook();
            life.set_closed();
            result
        }) {
            Ok(rx) => rx,
            Err(err) => {
                self.life.set_closed();
                return Err(err);
            }
        };
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

/// Check a config against `load`'s declared keys, in Python's order:
/// a missing required key, then the unlisted keys.
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
    };
    let shared = Shared::start(policy, resolved, on_episode_end, on_close)?;
    Ok(Model {
        handle: Arc::new(Handle { shared }),
    })
}

impl Model {
    /// Build a model from `policy`, loading it now with an empty config, as
    /// Python's `MyModel()` does. A closure is a policy too
    /// (`Model::new(|obs| Ok(..), ModelOptions::default())`).
    pub fn new<P: Policy>(policy: P, options: ModelOptions) -> Result<Model> {
        construct(
            Box::new(Erased(policy)),
            PolicyDecl::of::<P>(),
            options,
            None,
        )
    }

    /// Build `P::default()` and load it with `config`: Python's
    /// `MyModel.from_config(**config)`.
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

    /// Serve the model on `address` until the server stops, then close it:
    /// Python's `Model.serve`. SIGINT stops it (drain, close) and fails with
    /// the cancellation class; SIGTERM stops it and returns `Ok`.
    pub fn serve(&self, address: &str, options: ModelServeOptions) -> Result<()> {
        if crate::env::server::in_async_context() {
            return Err(Error::Runtime(
                "Model::serve() blocks; call serve_async() from async code".to_string(),
            ));
        }
        self.shared().check_not_reentrant("serve()")?;
        serve_blocking(self.shared(), address, options, true, None)
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
                        if tokens.iter().any(crate::ext::CancelToken::take) {
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

/// One serve, blocking the calling thread: validate, resolve the edition,
/// bind, serve until stopped, then close the model.
fn serve_blocking(
    shared: &Arc<Shared>,
    address: &str,
    options: ModelServeOptions,
    foreground: bool,
    dropped: Option<crate::ext::CancelToken>,
) -> Result<()> {
    let ModelServeOptions {
        options,
        cancel,
        ext,
    } = options;
    if cancel.as_ref().is_some_and(crate::ext::CancelToken::take) {
        return Err(Error::Cancelled);
    }
    shared.admit()?;
    let mut serve = options.unwrap_or_default();
    serve.validate()?;
    let ext = ext.unwrap_or_default();
    if ext.predict_concurrency == Some(0) {
        return Err(Error::invalid_argument(
            "predict_concurrency must be at least 1",
        ));
    }
    let token = crate::serve_options::model_endpoint_token(ext.token)?.unwrap_or_default();
    let address = BindAddress::parse(address)?;
    // The call's edition wins over the environment variable; a served peer
    // is authored, so a floating one warns (COR-6, COR-7).
    serve.workflow_edition = crate::editions::resolve_native(
        EditionSources {
            call: serve.workflow_edition.as_deref(),
            option: None,
            declared: shared.decl.workflow_edition.as_deref(),
        },
        true,
    )?;
    serve.drain_timeout = Some(serve.drain_timeout.unwrap_or(SERVE_GRACE));
    let close_timeout = serve.close_timeout.unwrap_or(SERVE_GRACE);
    serve.close_timeout = Some(close_timeout);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("rlmesh-model-serve")
        .build()
        .map_err(|err| Error::Internal(format!("failed to create tokio runtime: {err}")))?;
    let handler = super::bridge::engine_handler(Arc::clone(shared));
    let bound = runtime.block_on(super::server::bind_model_with_options(
        handler,
        address,
        &token,
        ext.predict_concurrency,
        serve,
    ))?;
    let trigger = bound.shutdown_trigger();
    let serve_id = shared.register_serve(trigger.clone())?;
    let signals = foreground.then(|| crate::env::signals::enter(trigger.clone()));
    let tokens: Vec<_> = cancel.into_iter().chain(dropped).collect();
    let watcher = CancelWatcher::start(tokens, trigger);
    let result = runtime.block_on(bound.serve());
    drop(runtime);
    shared.deregister_serve(serve_id);
    let cancelled = watcher.stop();
    let interrupted = signals.as_ref().is_some_and(|scope| scope.interrupted());
    drop(signals);
    let close = shared.close(Some(close_timeout));
    match (result, close) {
        (Err(err), _) => Err(err),
        (Ok(()), Err(err)) => Err(err),
        (Ok(()), Ok(())) if cancelled || interrupted => Err(Error::Cancelled),
        (Ok(()), Ok(())) => Ok(()),
    }
}

/// Drive `model` in-process against a remote env (the binding's run path
/// until `run` lands): the run borrows the model and never closes it.
pub(crate) async fn run_local(
    model: &Model,
    options: super::worker::RunLocalOptions,
    cancellation: tokio_util::sync::CancellationToken,
    hooks: Arc<dyn rlmesh_runtime::RuntimeHooks>,
) -> Result<rlmesh_runtime::RuntimeReport> {
    let shared = model.shared();
    shared.check_not_reentrant("run")?;
    shared.admit()?;
    if options.execution_horizon > 1 && !shared.decl.has_chunk() {
        crate::warnings::warn(
            crate::WarningCategory::UserWarning,
            format!(
                "execution_horizon={} was requested but the model defines no predict_chunk(); \
                 running un-chunked.",
                options.execution_horizon
            ),
        )?;
    }
    let mut handler = super::bridge::engine_handler(Arc::clone(shared));
    super::local::run_local(&mut handler, options, cancellation, hooks).await
}
