//! The authoring side of a model: the [`Policy`] trait, its declarations
//! ([`Corners`], [`LoadKey`], [`SpecArg`]) and the closure forms.

use std::fmt;

use super::context::{BatchPredictContext, PredictContext};
use crate::adapters::ModelSpec;
use crate::{Error, Kwargs, Metadata, Result, Value};

/// A set of predict corners, with C's `RLMESH_CORNER_*` bit values: which
/// corners a [`Policy`] writes ([`Policy::CORNERS`]) and which read the
/// context ([`Policy::CONTEXT`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Corners(u32);

impl Corners {
    /// No corner.
    pub const NONE: Corners = Corners(0);
    /// `predict`.
    pub const PREDICT: Corners = Corners(1);
    /// `predict_chunk`.
    pub const PREDICT_CHUNK: Corners = Corners(2);
    /// `predict_batch`.
    pub const PREDICT_BATCH: Corners = Corners(4);
    /// `predict_chunk_batch`.
    pub const PREDICT_CHUNK_BATCH: Corners = Corners(8);

    /// Every corner in `self` or `other` (bitwise or).
    pub const fn union(self, other: Corners) -> Corners {
        Corners(self.0 | other.0)
    }

    /// Whether every corner of `other` is in `self`.
    pub const fn contains(self, other: Corners) -> bool {
        self.0 & other.0 == other.0
    }

    pub(crate) const fn bits(self) -> u32 {
        self.0
    }

    pub(crate) const fn has_chunk(self) -> bool {
        self.0 & (Corners::PREDICT_CHUNK.0 | Corners::PREDICT_CHUNK_BATCH.0) != 0
    }

    /// The corner names in declaration order.
    pub(crate) fn names(self) -> Vec<&'static str> {
        [
            (Corners::PREDICT, "predict"),
            (Corners::PREDICT_CHUNK, "predict_chunk"),
            (Corners::PREDICT_BATCH, "predict_batch"),
            (Corners::PREDICT_CHUNK_BATCH, "predict_chunk_batch"),
        ]
        .into_iter()
        .filter(|(corner, _)| self.contains(*corner))
        .map(|(_, name)| name)
        .collect()
    }
}

/// One keyword `load` accepts: the native form of `load`'s signature.
/// `required` is a parameter with no default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoadKey {
    /// The keyword.
    pub name: &'static str,
    /// Whether `from_config` must supply it.
    pub required: bool,
}

impl LoadKey {
    /// A keyword with a default (`key=None`).
    pub const fn optional(name: &'static str) -> LoadKey {
        LoadKey {
            name,
            required: false,
        }
    }

    /// A keyword without a default.
    pub const fn required(name: &'static str) -> LoadKey {
        LoadKey {
            name,
            required: true,
        }
    }
}

/// What a model's `spec` may be: a `ModelSpec` or [`NO_ADAPTER`];
/// `Option<SpecArg>`'s `None` leaves the spec undeclared.
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)] // Two arms, as Python's SpecArg.
pub enum SpecArg {
    /// Resolve an adapter from this spec per route.
    ModelSpec(ModelSpec),
    /// Skip adapter resolution: the model adapts its own observations.
    NoAdapter,
}

impl From<ModelSpec> for SpecArg {
    fn from(spec: ModelSpec) -> SpecArg {
        SpecArg::ModelSpec(spec)
    }
}

/// Disable adapter resolution.
pub const NO_ADAPTER: SpecArg = SpecArg::NoAdapter;

/// The text a corner listed in [`Policy::CORNERS`] (or `CONTEXT`) but not
/// written fails with.
fn unwritten(method: &str, list: &str, entry: &str) -> Error {
    Error::Runtime(format!(
        "{method} is not implemented: the policy lists {entry} in {list} but does not \
         write {method}()"
    ))
}

/// A model's declarations, predict operations and lifecycle hooks. Write the
/// corners the model has and list them in [`CORNERS`](Policy::CORNERS); a
/// corner that reads the per-episode context is listed in
/// [`CONTEXT`](Policy::CONTEXT) too and written as its
/// `<corner>_with_context` twin. Build the model with
/// [`Model::new`](crate::Model::new) or
/// [`Model::from_config`](crate::Model::from_config).
///
/// Rust cannot see which methods an impl wrote, so the consts are the
/// declaration: a corner listed but not written fails its first call with
/// the `RuntimeError` class naming the entry, and a corner written but not
/// listed is never called.
pub trait Policy: Send + 'static {
    /// The corners this policy implements.
    const CORNERS: Corners = Corners::PREDICT;
    /// The corners that read the context; each is called through its
    /// `_with_context` twin. Must be a subset of `CORNERS`.
    const CONTEXT: Corners = Corners::NONE;
    /// The static native chunk length K (describe reads this one).
    const NATIVE_CHUNK: Option<u32> = None;
    /// The parameter keys `load` accepts; `None` forwards every key.
    const LOAD_KEYS: Option<&'static [LoadKey]> = None;
    /// Whether the policy writes [`reset_options`](Policy::reset_options);
    /// a run asks it only then.
    const RESET_OPTIONS: bool = false;
    /// Whether the policy type is an authored participant; false only for
    /// the closure forms, as a wrapped bare callable is not one.
    #[doc(hidden)]
    const AUTHORED: bool = true;

    /// The model's adapter spec.
    fn spec() -> Option<SpecArg>
    where
        Self: Sized,
    {
        None
    }

    /// The workflow edition the policy was authored against.
    fn workflow_edition() -> Option<String>
    where
        Self: Sized,
    {
        None
    }

    /// Load weights from `config`, once, in the constructor. The default
    /// accepts only an empty config.
    fn load(&mut self, kwargs: &Kwargs) -> Result<()> {
        if kwargs.is_empty() {
            return Ok(());
        }
        let mut keys: Vec<&str> = kwargs.keys().collect();
        keys.sort_unstable();
        Err(Error::Type(format!(
            "model received construction params ({}) but does not override load(...) to apply \
             them; implement load(**kwargs) to consume the binding",
            keys.join(", ")
        )))
    }

    /// Whether a grouped predict may fuse lanes from different envs into one
    /// batched-corner call; read once, after `load`.
    fn allow_fusion(&self) -> bool {
        true
    }

    /// The native chunk length K after `load` (a checkpoint may decide it);
    /// read once, after `load`.
    fn native_chunk(&self) -> Option<u32> {
        Self::NATIVE_CHUNK
    }

    /// One observation to one action.
    fn predict(&mut self, observation: Value) -> Result<Value> {
        let _ = observation;
        Err(unwritten("predict", "CORNERS", "Corners::PREDICT"))
    }

    /// [`predict`](Policy::predict) reading the context.
    fn predict_with_context(
        &mut self,
        observation: Value,
        context: &mut PredictContext<'_>,
    ) -> Result<Value> {
        let _ = (observation, context);
        Err(unwritten(
            "predict_with_context",
            "CONTEXT",
            "Corners::PREDICT",
        ))
    }

    /// One action per observation, in one call.
    fn predict_batch(&mut self, observations: Vec<Value>) -> Result<Vec<Value>> {
        let _ = observations;
        Err(unwritten(
            "predict_batch",
            "CORNERS",
            "Corners::PREDICT_BATCH",
        ))
    }

    /// [`predict_batch`](Policy::predict_batch) reading one context per row.
    fn predict_batch_with_context(
        &mut self,
        observations: Vec<Value>,
        context: &mut BatchPredictContext<'_>,
    ) -> Result<Vec<Value>> {
        let _ = (observations, context);
        Err(unwritten(
            "predict_batch_with_context",
            "CONTEXT",
            "Corners::PREDICT_BATCH",
        ))
    }

    /// One observation to a chunk of actions (leading axis = frames).
    fn predict_chunk(&mut self, observation: Value, execution_horizon: u32) -> Result<Value> {
        let _ = (observation, execution_horizon);
        Err(unwritten(
            "predict_chunk",
            "CORNERS",
            "Corners::PREDICT_CHUNK",
        ))
    }

    /// [`predict_chunk`](Policy::predict_chunk) reading the context.
    fn predict_chunk_with_context(
        &mut self,
        observation: Value,
        execution_horizon: u32,
        context: &mut PredictContext<'_>,
    ) -> Result<Value> {
        let _ = (observation, execution_horizon, context);
        Err(unwritten(
            "predict_chunk_with_context",
            "CONTEXT",
            "Corners::PREDICT_CHUNK",
        ))
    }

    /// One chunk per observation, in one call.
    fn predict_chunk_batch(
        &mut self,
        observations: Vec<Value>,
        execution_horizon: u32,
    ) -> Result<Vec<Value>> {
        let _ = (observations, execution_horizon);
        Err(unwritten(
            "predict_chunk_batch",
            "CORNERS",
            "Corners::PREDICT_CHUNK_BATCH",
        ))
    }

    /// [`predict_chunk_batch`](Policy::predict_chunk_batch) reading one
    /// context per row.
    fn predict_chunk_batch_with_context(
        &mut self,
        observations: Vec<Value>,
        execution_horizon: u32,
        context: &mut BatchPredictContext<'_>,
    ) -> Result<Vec<Value>> {
        let _ = (observations, execution_horizon, context);
        Err(unwritten(
            "predict_chunk_batch_with_context",
            "CONTEXT",
            "Corners::PREDICT_CHUNK_BATCH",
        ))
    }

    /// Options to add to the env reset that starts `episode`, which holds
    /// its `episode_id`, `seed` and `trial_index`. A run asks before each
    /// reset it sends (when [`RESET_OPTIONS`](Policy::RESET_OPTIONS) is
    /// set), and the env's `reset(options)` receives them under the
    /// runtime's own reserved keys. A reset of several lanes gets each key
    /// as a per-lane list; an env that autoresets restarts its lanes itself,
    /// with no reset to carry them. The default adds none.
    fn reset_options(&mut self, episode: &Metadata) -> Result<Option<Metadata>> {
        let _ = episode;
        Ok(None)
    }

    /// An episode ended (after its context entry is dropped).
    fn on_episode_end(&mut self, episode_id: &str) -> Result<()> {
        let _ = episode_id;
        Ok(())
    }

    /// Release the model's resources: runs at most once.
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A bare predict closure.
impl<F> Policy for F
where
    F: FnMut(Value) -> Result<Value> + Send + 'static,
{
    const AUTHORED: bool = false;

    fn predict(&mut self, observation: Value) -> Result<Value> {
        self(observation)
    }
}

/// A predict closure that reads the context. Built by [`with_context`].
pub struct WithContext<F>(F);

impl<F> fmt::Debug for WithContext<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WithContext(..)")
    }
}

/// Wrap a closure that takes the per-episode context as a policy.
pub fn with_context<F>(f: F) -> WithContext<F>
where
    F: FnMut(Value, &mut PredictContext<'_>) -> Result<Value> + Send + 'static,
{
    WithContext(f)
}

impl<F> Policy for WithContext<F>
where
    F: FnMut(Value, &mut PredictContext<'_>) -> Result<Value> + Send + 'static,
{
    const CONTEXT: Corners = Corners::PREDICT;
    const AUTHORED: bool = false;

    fn predict_with_context(
        &mut self,
        observation: Value,
        context: &mut PredictContext<'_>,
    ) -> Result<Value> {
        (self.0)(observation, context)
    }
}
