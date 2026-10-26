//! Run hooks and the context a run hands them.

use std::sync::{Arc, Mutex};

use super::{EpisodeResult, RunResult, StepEvent};
use super::{FrameSources, ReadItem, RenderFn};
use crate::{EnvContract, Error, Result, Value};

/// Presentation data emitted by the core after an observation is available.
/// The observer may draw it; episode accounting remains in the driver.
pub struct RunFrame<'a> {
    /// The newly available observation.
    pub observation: &'a Value,
    /// Zero-based episode index.
    pub episode: i64,
    /// The reset seed.
    pub seed: Option<i64>,
    /// Completed steps in this episode.
    pub steps: i64,
    /// Cumulative episode reward.
    pub reward: f64,
    /// An already-known outcome for this frame; otherwise use the last step.
    pub outcome: Option<&'a str>,
    /// Most recent prediction latency in milliseconds.
    pub predict_ms: f64,
    /// Most recent environment latency in milliseconds.
    pub step_ms: f64,
    /// Time since this episode started, in seconds.
    pub elapsed_s: f64,
}

/// Observer callbacks for a run. Every default is a no-op; implement any
/// subset.
///
/// A sync run calls hooks on the thread that called it, outside any async
/// execution context, so a hook may make sync rlmesh calls; under an
/// `_async` twin they run inside its future, where a blocking rlmesh call
/// fails with the `RuntimeError` class. A hook error aborts the run;
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

    /// Binding plumbing for Python's Experimental `RunHooks` pair (R-25):
    /// one lane's observation before the model sees it, with its 0-based
    /// episode and the steps that episode has taken; return a replacement of
    /// the same space, or `None` to keep it. Runs only; a manual binding does
    /// not call it.
    #[doc(hidden)]
    fn transform_observation(
        &mut self,
        episode: i64,
        step: i64,
        observation: &Value,
    ) -> Result<Option<Value>> {
        let _ = (episode, step, observation);
        Ok(None)
    }

    /// Binding plumbing, as `transform_observation`: one lane's action
    /// before the env receives it, once per env step, so a hook keeping
    /// recent actions can delay, drop (hold the previous one) or freeze them.
    #[doc(hidden)]
    fn transform_action(
        &mut self,
        episode: i64,
        step: i64,
        action: &Value,
    ) -> Result<Option<Value>> {
        let _ = (episode, step, action);
        Ok(None)
    }
}

/// What a run exposes to its hooks.
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
/// binding's (failing once the binding has closed).
pub(crate) struct LibraryContext {
    contract: Mutex<Option<EnvContract>>,
    closed: std::sync::atomic::AtomicBool,
    pub(crate) id: u64,
    readers: Mutex<Vec<(ReadItem, super::Reader)>>,
    advisories: Mutex<Vec<crate::adapters::Advisory>>,
    presentation: Mutex<Option<Box<dyn crate::binding::model::Presentation>>>,
}

impl LibraryContext {
    pub(crate) fn new() -> Arc<LibraryContext> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Arc::new(LibraryContext {
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            readers: Mutex::new(Vec::new()),
            presentation: Mutex::new(None),
            advisories: Mutex::new(Vec::new()),
            contract: Mutex::new(None),
            closed: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub(crate) fn set_presentation(
        &self,
        observer: Option<Box<dyn crate::binding::model::Presentation>>,
    ) {
        *self.presentation.lock().unwrap_or_else(|e| e.into_inner()) = observer;
    }
    pub(crate) fn frame(&self, frame: &RunFrame<'_>) -> Result<()> {
        match &mut *self.presentation.lock().unwrap_or_else(|e| e.into_inner()) {
            Some(observer) => observer.on_frame(frame),
            None => Ok(()),
        }
    }
    pub(crate) fn directive(&self) -> Result<super::runtime::RunDirective> {
        match &mut *self.presentation.lock().unwrap_or_else(|e| e.into_inner()) {
            Some(observer) => observer.directive(),
            None => Ok(super::runtime::RunDirective::Continue),
        }
    }

    pub(crate) fn connect(&self, contract: EnvContract) {
        *self.contract.lock().unwrap_or_else(|err| err.into_inner()) = Some(contract);
    }

    #[cfg(feature = "lockstep")]
    pub(crate) fn close(&self) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
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
        super::read::local_render(self.id)
    }
    fn open(&self) -> Result<()> {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(Error::Runtime("the binding is closed".to_string()));
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
        let label = if super::read::has_local_render(self.id) {
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
