//! A model that calls other models from its predict: `self.inner.predict(obs)`
//! inside a run, bind or serve of the outer model runs the inner model's own
//! handler on the episode being predicted, whether the inner model is local
//! or served (`Model("host:port/name")`).
//!
//! The outer model's handler is wrapped ([`NestedHandler`]): it drops the
//! inner models' episodes and routes with its own. The outer predict sets the
//! episode it is answering as its thread's scope ([`scoped`]), which
//! [`nested_predict`] reads.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;
use pyo3::exceptions::{PyLookupError, PyRuntimeError};
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rlmesh::binding::model::{
    EpisodeInfo, HeldState, ModelHandler, ModelObservation, ModelResetOptions, ModelRouteContext,
    ModelRouteSetup, PredictFrames, ResolveOptions, RouteNeeds,
};
use rlmesh_spaces::{EnvContract, SpaceValue};

use crate::spaces::{ValueBackend, py_any_to_space_value_with_backend, space_value_to_py_neutral};
use crate::types::to_py_err;

/// One inner model: its handler, its route setup, and the envs it resolved.
struct Inner {
    handler: tokio::sync::Mutex<Box<dyn ModelHandler>>,
    setup: Option<Arc<dyn ModelRouteSetup>>,
    resolved: Mutex<HashSet<String>>,
}

impl Inner {
    fn resolved(&self) -> std::sync::MutexGuard<'_, HashSet<String>> {
        self.resolved.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// One row's frames for `observation`, resolving its env on first use.
    async fn predict(
        &self,
        observation: ModelObservation,
        contract: &EnvContract,
    ) -> rlmesh::Result<PredictFrames> {
        let env_id = observation.route.env_id.clone();
        if let Some(setup) = &self.setup
            && !self.resolved().contains(&env_id)
        {
            let options = ResolveOptions {
                execution_horizon: 1,
                delivers_history: false,
            };
            setup.resolve_adapter(&env_id, contract, options).await?;
            self.resolved().insert(env_id);
        }
        self.handler.lock().await.predict_chunked(observation).await
    }
}

/// The inner models of one outer model, by the Python `id` of each.
pub(crate) struct Nested {
    inners: HashMap<u64, Inner>,
}

impl Nested {
    pub(crate) fn new(inners: Vec<(u64, Box<dyn ModelHandler>)>) -> Arc<Self> {
        let inners = inners
            .into_iter()
            .map(|(key, handler)| {
                let inner = Inner {
                    setup: handler.route_setup(),
                    handler: tokio::sync::Mutex::new(handler),
                    resolved: Mutex::default(),
                };
                (key, inner)
            })
            .collect();
        Arc::new(Self { inners })
    }

    /// The inners that resolved `env_id`.
    fn on_env<'a>(&'a self, env_id: &'a str) -> impl Iterator<Item = &'a Inner> {
        self.inners
            .values()
            .filter(move |inner| inner.resolved().contains(env_id))
    }
}

type Scope = (Arc<Nested>, EpisodeInfo, Arc<EnvContract>);

thread_local! {
    /// The episode the outer model is answering on this thread.
    static SCOPE: RefCell<Option<Scope>> = const { RefCell::new(None) };
}

/// Runs `f` with `episode` of the env `contract` as this thread's scope for
/// inner predicts (no scope without inner models, an episode or a contract).
pub(crate) fn scoped<R>(
    nested: Option<&Arc<Nested>>,
    episode: Option<&EpisodeInfo>,
    contract: Option<&Arc<EnvContract>>,
    f: impl FnOnce() -> R,
) -> R {
    let (Some(nested), Some(episode), Some(contract)) = (nested, episode, contract) else {
        return f();
    };
    struct Restore(Option<Scope>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SCOPE.with(|scope| *scope.borrow_mut() = self.0.take());
        }
    }
    let scope = (Arc::clone(nested), episode.clone(), Arc::clone(contract));
    let _restore = Restore(SCOPE.with(|slot| slot.replace(Some(scope))));
    f()
}

/// The inner model `key`'s action for `observation` (neutral, in the env's
/// observation space), on the episode the outer model is answering. A
/// `LookupError` when no predict of an outer model holding it is running.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
import typing
def nested_predict(key: int, observation: typing.Any) -> typing.Any: ...
"#
    )
)]
#[pyfunction]
pub fn nested_predict(
    py: Python<'_>,
    key: u64,
    observation: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let Some((nested, episode, contract)) = SCOPE.with(|scope| scope.borrow().clone()) else {
        return Err(PyLookupError::new_err("no predict in progress"));
    };
    if !nested.inners.contains_key(&key) {
        return Err(PyLookupError::new_err("not an inner model of this predict"));
    }
    let space = |space: Option<&rlmesh_spaces::spaces::SpaceSpec>, what: &str| {
        space.cloned().ok_or_else(|| {
            PyRuntimeError::new_err(format!("an inner predict needs the env's {what} space"))
        })
    };
    let observation_space = space(contract.observation_space.as_ref(), "observation")?;
    let action_space = space(contract.action_space.as_ref(), "action")?;
    let value = py_any_to_space_value_with_backend(
        py,
        &observation,
        &observation_space,
        ValueBackend::Native,
    )?;
    let leaves = rlmesh_spaces::leaves::encode_leaf_slab(&[value], &observation_space)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let request = ModelObservation {
        observation: Some(leaves),
        route: ModelRouteContext {
            env_id: episode.env_id.clone(),
            episodes: vec![episode],
            ..Default::default()
        },
        num_envs: 1,
        env_contract: Some(Arc::clone(&contract)),
        ..Default::default()
    };
    // On the shared multi-thread runtime: a served inner model's link needs
    // it, and this thread is the outer predict's.
    let (sent, answer) = tokio::sync::oneshot::channel::<rlmesh::Result<PredictFrames>>();
    crate::model::model_runtime().spawn(async move {
        let inner = &nested.inners[&key];
        let _ = sent.send(inner.predict(request, &contract).await);
    });
    let frames = py
        .detach(|| answer.blocking_recv())
        .map_err(|_| PyRuntimeError::new_err("the inner predict was dropped"))?
        .map_err(to_py_err)?;
    let action: SpaceValue = frames
        .actions
        .into_iter()
        .next()
        .ok_or_else(|| PyRuntimeError::new_err("the inner model returned no action"))?;
    Ok(space_value_to_py_neutral(py, &action, &action_space)?.unbind())
}

/// An outer model's handler that also ends its inner models' episodes and
/// routes, and closes them with it.
pub(crate) struct NestedHandler {
    pub(crate) outer: Box<dyn ModelHandler>,
    pub(crate) nested: Arc<Nested>,
}

#[async_trait]
impl ModelHandler for NestedHandler {
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        self.outer.predict(observation).await
    }

    async fn predict_chunked(
        &mut self,
        observation: ModelObservation,
    ) -> rlmesh::Result<PredictFrames> {
        self.outer.predict_chunked(observation).await
    }

    async fn predict_grouped(
        &mut self,
        observations: Vec<ModelObservation>,
    ) -> Vec<rlmesh::Result<PredictFrames>> {
        self.outer.predict_grouped(observations).await
    }

    fn held_state(&self) -> Option<HeldState> {
        self.outer.held_state()
    }

    fn take_adapter_ns(&mut self) -> u64 {
        self.outer.take_adapter_ns()
    }

    fn route_setup(&self) -> Option<Arc<dyn ModelRouteSetup>> {
        Some(Arc::new(NestedSetup {
            outer: self.outer.route_setup(),
            nested: Arc::clone(&self.nested),
        }))
    }

    async fn reset_adapter(
        &mut self,
        env_id: &str,
        episode_ids: Vec<String>,
    ) -> rlmesh::Result<()> {
        self.outer
            .reset_adapter(env_id, episode_ids.clone())
            .await?;
        for inner in self.nested.on_env(env_id) {
            let mut handler = inner.handler.lock().await;
            handler.reset_adapter(env_id, episode_ids.clone()).await?;
        }
        Ok(())
    }

    async fn join(&mut self, session: &str) -> rlmesh::Result<()> {
        self.outer.join(session).await?;
        for inner in self.nested.inners.values() {
            inner.handler.lock().await.join(session).await?;
        }
        Ok(())
    }

    async fn on_close(&mut self) -> rlmesh::Result<()> {
        let closed = self.outer.on_close().await;
        for inner in self.nested.inners.values() {
            inner.handler.lock().await.on_close().await?;
        }
        closed
    }

    fn reset_options(&self) -> Option<Arc<dyn ModelResetOptions>> {
        self.outer.reset_options()
    }
}

/// The outer model's route setup, ending each env on the inner models too.
struct NestedSetup {
    outer: Option<Arc<dyn ModelRouteSetup>>,
    nested: Arc<Nested>,
}

#[async_trait]
impl ModelRouteSetup for NestedSetup {
    async fn resolve_adapter(
        &self,
        env_id: &str,
        env_contract: &EnvContract,
        options: ResolveOptions,
    ) -> rlmesh::Result<RouteNeeds> {
        match &self.outer {
            Some(outer) => outer.resolve_adapter(env_id, env_contract, options).await,
            None => Ok(RouteNeeds::default()),
        }
    }

    async fn reset_adapter(&self, env_id: &str, episode_ids: &[String]) -> rlmesh::Result<()> {
        if let Some(outer) = &self.outer {
            outer.reset_adapter(env_id, episode_ids).await?;
        }
        for inner in self.nested.on_env(env_id) {
            if let Some(setup) = &inner.setup {
                setup.reset_adapter(env_id, episode_ids).await?;
            }
        }
        Ok(())
    }

    async fn release_adapter(&self, env_id: &str) -> rlmesh::Result<()> {
        if let Some(outer) = &self.outer {
            outer.release_adapter(env_id).await?;
        }
        for inner in self.nested.inners.values() {
            if inner.resolved().remove(env_id)
                && let Some(setup) = &inner.setup
            {
                setup.release_adapter(env_id).await?;
            }
        }
        Ok(())
    }
}
