//! Run hooks and the context a run hands them (RUN-6, RUN-7, RUN-8).

use std::sync::{Arc, Mutex};

use super::{EpisodeResult, RunResult, StepEvent};
use super::{FrameSources, ReadItem, RenderFn};
use crate::{EnvContract, Error, Result, Value};

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
    fn on_run_start(&mut self, context: Arc<dyn RunContext>) -> Result<()> {
        let _ = context;
        Ok(())
    }

    /// After each episode's reset returns (`episode` is 0-based).
    fn on_episode_start(&mut self, episode: i64, seed: Option<i64>) -> Result<()> {
        let _ = (episode, seed);
        Ok(())
    }

    /// After each env step.
    fn on_step(&mut self, event: &StepEvent) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// With each completed episode's result.
    fn on_episode_end(&mut self, result: &EpisodeResult) -> Result<()> {
        let _ = result;
        Ok(())
    }

    /// Exactly once when the run ends, also on an error or a cancel, with
    /// the (possibly partial) result.
    fn on_run_end(&mut self, result: &RunResult) -> Result<()> {
        let _ = result;
        Ok(())
    }
}

/// What a run exposes to its hooks: Python's `RunContext` protocol.
pub trait RunContext: Send + Sync {
    /// The connected env's contract; `None` before the env is connected.
    fn contract(&self) -> Result<Option<EnvContract>>;

    /// Lanes the connected env runs (1 before connect).
    fn num_envs(&self) -> Result<u64>;
    /// Read a semantic role or input leaf through a read-only adapter.
    fn read(&self, observation: &Value, item: &ReadItem) -> Result<Value>;
    /// Discover the currently reachable image and render sources.
    fn frame_sources(&self) -> Result<FrameSources<'_>>;
}

/// The library's context: a native run's (valid for its whole life) or a
/// session's (failing once the session has closed, RUN-7).
type RemoteRender = Arc<dyn Fn() -> Result<Value> + Send + Sync>;
pub(crate) struct LibraryContext {
    contract: Mutex<Option<EnvContract>>,
    closed: std::sync::atomic::AtomicBool,
    pub(crate) id: u64,
    busy: std::sync::atomic::AtomicBool,
    remote_render: Mutex<Option<RemoteRender>>,
    readers: Mutex<Vec<(ReadItem, super::Reader)>>,
    advisories: Mutex<Vec<crate::adapters::Advisory>>,
}

impl LibraryContext {
    pub(crate) fn new() -> Arc<LibraryContext> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Arc::new(LibraryContext {
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            busy: false.into(),
            remote_render: Mutex::new(None),
            readers: Mutex::new(Vec::new()),
            advisories: Mutex::new(Vec::new()),
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

    pub(crate) fn remote_render(&self, render: Arc<dyn Fn() -> Result<Value> + Send + Sync>) {
        *self.remote_render.lock().unwrap_or_else(|e| e.into_inner()) = Some(render);
    }
    pub(crate) fn busy(self: &Arc<Self>) -> Result<ContextBusy> {
        self.busy
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
            )
            .map_err(|_| {
                Error::Runtime(
                    "a session operation or context render is already in progress".into(),
                )
            })?;
        Ok(ContextBusy(self.clone()))
    }
    pub(crate) fn advisories(&self) -> Vec<crate::adapters::Advisory> {
        self.advisories
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    pub(crate) fn set_advisories(&self, notes: Vec<crate::adapters::Advisory>) {
        *self.advisories.lock().unwrap_or_else(|e| e.into_inner()) = notes;
    }
    pub(crate) fn render(&self) -> Result<Value> {
        self.open()?;
        if super::read::has_local_render(self.id) {
            return super::read::local_render(self.id);
        }
        if self.busy.load(std::sync::atomic::Ordering::SeqCst) && !super::read::in_dispatch(self.id)
        {
            return Err(Error::Runtime(
                "render is unavailable while a session operation is in progress".into(),
            ));
        }
        let _busy = if super::read::in_dispatch(self.id) {
            None
        } else {
            self.busy
                .compare_exchange(
                    false,
                    true,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                )
                .map_err(|_| {
                    Error::Runtime(
                        "a session operation or context render is already in progress".into(),
                    )
                })?;
            Some(RenderBusy(&self.busy))
        };
        let render = self
            .remote_render
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or_else(|| {
                Error::Runtime("render is not reachable outside a local hook dispatch".into())
            })?;
        render()
    }
    fn open(&self) -> Result<()> {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(Error::Runtime("session is closed".to_string()));
        }
        Ok(())
    }
}

pub(crate) struct ContextBusy(Arc<LibraryContext>);
impl Drop for ContextBusy {
    fn drop(&mut self) {
        self.0
            .busy
            .store(false, std::sync::atomic::Ordering::SeqCst)
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
    fn read(&self, observation: &Value, item: &ReadItem) -> Result<Value> {
        self.open()?;
        let contract = self
            .contract()?
            .ok_or_else(|| Error::adapter_resolution("reading by role needs an env contract"))?;
        let mut cache = self.readers.lock().unwrap_or_else(|e| e.into_inner());
        let reader = if let Some((_, reader)) = cache.iter().find(|(key, _)| key == item) {
            reader.clone()
        } else {
            let reader = super::read::reader(&contract, std::slice::from_ref(item))?;
            cache.push((item.clone(), reader.clone()));
            reader
        };
        drop(cache);
        Ok(reader
            .call(observation)?
            .get(&reader.roles()[0])
            .cloned()
            .unwrap_or(Value::None))
    }
    fn frame_sources(&self) -> Result<FrameSources<'_>> {
        let contract = self.contract()?;
        let roles = super::read::image_roles(contract.as_ref())?;
        let reachable = super::read::has_local_render(self.id)
            || (self
                .remote_render
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_some()
                && (!self.busy.load(std::sync::atomic::Ordering::SeqCst)
                    || super::read::in_dispatch(self.id)));
        let label = if reachable {
            contract
                .as_ref()
                .and_then(|c| super::read::render_label(c, &roles))
        } else {
            None
        };
        let render = label.as_ref().map(|_| RenderFn::new(|| self.render()));
        Ok(FrameSources {
            roles,
            render_label: label,
            render,
        })
    }
}

struct RenderBusy<'a>(&'a std::sync::atomic::AtomicBool);
impl Drop for RenderBusy<'_> {
    fn drop(&mut self) {
        self.0.store(false, std::sync::atomic::Ordering::SeqCst)
    }
}
