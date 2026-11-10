//! A policy behind one object-safe interface: its declarations
//! ([`PolicyDecl`]) and [`DynPolicy`], which every [`Policy`] erases to.

use crate::model::context::{BatchPredictContext, PredictContext};
use crate::model::policy::{Corners, LoadKey, Policy, SpecArg};
use crate::{Error, Kwargs, Metadata, Result, Value};

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
    /// Whether the policy is an authored participant.
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
    /// [`Policy::execution_horizon`].
    fn execution_horizon(&self) -> Option<u32> {
        None
    }
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
    /// Whether [`reset_options`](DynPolicy::reset_options) is written
    /// ([`Policy::RESET_OPTIONS`]); read once, after `load`.
    fn has_reset_options(&self) -> bool {
        false
    }
    /// [`Policy::reset_options`].
    fn reset_options(&mut self, episode: &Metadata) -> Result<Option<Metadata>> {
        let _ = episode;
        Ok(None)
    }
    /// [`Policy::close`].
    fn close(&mut self) -> Result<()>;
}

/// A [`Policy`] on the erased seam: the `_with_context` twin for a corner in
/// `P::CONTEXT`, the plain method otherwise.
pub(super) struct Erased<P>(pub(super) P);

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

    fn execution_horizon(&self) -> Option<u32> {
        self.0.execution_horizon()
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

    fn has_reset_options(&self) -> bool {
        P::RESET_OPTIONS
    }

    fn reset_options(&mut self, episode: &Metadata) -> Result<Option<Metadata>> {
        self.0.reset_options(episode)
    }

    fn close(&mut self) -> Result<()> {
        self.0.close()
    }
}
