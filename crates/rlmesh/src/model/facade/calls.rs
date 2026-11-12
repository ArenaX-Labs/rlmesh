//! What a model call runs on the library thread: the [`Core`] (policy, episode
//! store and hooks), the declarations resolved after `load`, and each row's
//! identity.

use super::erased::DynPolicy;
use crate::model::context::{EpisodeStore, OwnedContext, PredictContext};
use crate::model::policy::{Corners, SpecArg};
use crate::model::types::EpisodeInfo;
use crate::{Error, Metadata, Result, Value};

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
    pub(crate) type_name: String,
    pub(crate) load_keys: Option<Vec<(String, bool)>>,
    pub(crate) trust_entrypoints: bool,
    /// Whether the policy adds reset options (read after `load`).
    pub(crate) reset_options: bool,
    /// The execution horizon `load` chose for its checkpoint.
    pub(crate) execution_horizon: Option<u32>,
}

impl Resolved {
    /// The horizon a run naming none executes at: the one `load` chose, else
    /// the spec's `timing.execution`, else 1.
    pub(crate) fn resolve_horizon(&self, explicit: Option<u32>) -> Result<u32> {
        crate::run::resolve_execution_horizon(
            explicit.map(i128::from),
            self.execution_horizon.map(i128::from),
            match &self.spec {
                Some(SpecArg::ModelSpec(spec)) => spec.timing().execution.map(i128::from),
                _ => None,
            },
            self.native_chunk,
        )
    }

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
    pub(super) policy: Box<dyn DynPolicy>,
    pub(super) corners: Corners,
    pub(super) context: Corners,
    pub(super) store: EpisodeStore,
    pub(super) on_episode_end: Option<Box<dyn FnMut(&str) -> Result<()> + Send>>,
    pub(super) on_close: Option<Box<dyn FnOnce() -> Result<()> + Send>>,
    pub(super) closed: bool,
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

    /// The policy's reset options for one episode.
    pub(crate) fn reset_options(&mut self, episode: &Metadata) -> Result<Option<Metadata>> {
        self.policy.reset_options(episode)
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

    pub(super) fn close_hook(&mut self) -> Result<()> {
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
