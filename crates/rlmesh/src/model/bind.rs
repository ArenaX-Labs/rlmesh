//! A model bound to one env, stepped by hand: `Model.bind(target)`.
//!
//! The run loop's single-lane step, driven one call at a time: `reset`
//! starts an episode (with the model's reset options), `step` asks the
//! model for an action (replaying a chunk across steps) unless the caller
//! passes one, and an episode's end evicts its state on the model, as a
//! run does.

use std::collections::VecDeque;
use std::sync::Arc;

use bytes::Bytes;
use rlmesh_spaces::leaves::{decode_leaf_slab, encode_leaf_slab};
use rlmesh_spaces::{EnvContract, MetaMap, SpaceValue};

use super::handler::{ModelHandler, ModelResetOptions, ModelRouteSetup, ResolveOptions};
use super::local::{DrivenEnv, LockstepEnv};
use super::types::{EpisodeInfo, HistoryFrame, ModelObservation, ModelRouteContext};
use crate::env::client::lockstep_error;
use crate::lockstep::LockstepClient;
use crate::{Error, Result};

/// The live episode.
struct Episode {
    info: EpisodeInfo,
    /// The observation the next action answers, as leaf slabs.
    observation: Vec<Bytes>,
    step: i64,
    /// The rest of the last predicted chunk.
    replay: VecDeque<SpaceValue>,
    /// The steps a replayed action took, for the next predict's history.
    history: Vec<HistoryFrame>,
    /// Ended (its state evicted on the model).
    done: bool,
}

/// One env step through a [`Binding`].
pub struct BoundStep {
    /// The next observation.
    pub observation: SpaceValue,
    /// The action applied.
    pub action: SpaceValue,
    /// The step's reward.
    pub reward: f64,
    /// Whether the env reported a terminal state.
    pub terminated: bool,
    /// Whether the episode was cut short.
    pub truncated: bool,
    /// The step's info.
    pub info: Option<MetaMap>,
    /// The model's predict this step took (zero for a replayed or given
    /// action).
    pub predict: std::time::Duration,
    /// The env's step round trip.
    pub env: std::time::Duration,
}

/// A model bound to one env lane.
pub struct Binding {
    handler: Box<dyn ModelHandler>,
    setup: Option<Arc<dyn ModelRouteSetup>>,
    reset_options: Option<Arc<dyn ModelResetOptions>>,
    client: LockstepClient,
    contract: Arc<EnvContract>,
    env_id: String,
    /// The route keeps frame windows: every replayed step goes to the next
    /// predict as a history row.
    history: bool,
    episode: Option<Episode>,
}

impl Binding {
    /// Binds `handler` to the env at the session address `address`: the
    /// model joins the env's session and resolves its route there.
    pub async fn connect(
        mut handler: Box<dyn ModelHandler>,
        address: &str,
        execution_horizon: u32,
        workflow_edition: Option<&str>,
    ) -> Result<Self> {
        if !crate::address::is_v2(address) {
            return Err(Error::invalid_argument(format!(
                "Model.bind({address:?}): a v1 env endpoint, which rlmesh 0.2 no longer dials; \
                 serve the env at a session address (EnvServer(env, \":0/name\")), or pass the \
                 env object itself"
            )));
        }
        let connected = LockstepEnv::connect(address, workflow_edition).await?;
        if connected.num_envs != 1 {
            return Err(Error::invalid_argument(format!(
                "Model.bind drives one env lane and this env serves {}: run it with \
                 Model.run, or serve its lanes as envs of their own",
                connected.num_envs
            )));
        }
        let DrivenEnv::Lockstep(env) = connected.env;
        let client = env.client().clone();
        if let Some(session) = &connected.session {
            handler.join(session).await?;
        }
        let env_id = crate::mint_id();
        let setup = handler.route_setup();
        let mut history = false;
        if let Some(setup) = &setup {
            let needs = setup
                .resolve_adapter(
                    &env_id,
                    &connected.contract,
                    ResolveOptions {
                        execution_horizon,
                        delivers_history: true,
                    },
                )
                .await?;
            history = needs.history.is_some();
        }
        Ok(Self {
            reset_options: handler.reset_options(),
            handler,
            setup,
            client,
            contract: Arc::new(connected.contract),
            env_id,
            history,
            episode: None,
        })
    }

    /// The env's contract.
    pub fn contract(&self) -> &EnvContract {
        &self.contract
    }

    /// Ends the live episode, if any, and starts the next: seeded with
    /// `seed`, with the model's reset options under the caller's `options`.
    /// Returns its first observation and the reset's info.
    pub async fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<MetaMap>,
    ) -> Result<(SpaceValue, Option<MetaMap>)> {
        self.end().await?;
        let info = EpisodeInfo {
            episode_id: crate::mint_id(),
            seed,
            env_id: self.env_id.clone(),
        };
        let mut merged = match &self.reset_options {
            Some(source) => source.reset_options(&info, None).await?.unwrap_or_default(),
            None => MetaMap::new(),
        };
        merged.extend(options.unwrap_or_default());
        let seeds: Vec<i64> = seed.into_iter().collect();
        let options = (!merged.is_empty()).then_some(merged);
        let (observation, reset_info) = self
            .client
            .reset_leaves(&[], &seeds, options.as_ref(), None)
            .await
            .map_err(lockstep_error)?;
        let value = self.observation(&observation)?;
        self.episode = Some(Episode {
            info,
            observation,
            step: 0,
            replay: VecDeque::new(),
            history: Vec::new(),
            done: false,
        });
        Ok((value, reset_info))
    }

    /// One env step: `action`, or the model's (the next of its chunk).
    pub async fn step(&mut self, action: Option<SpaceValue>) -> Result<BoundStep> {
        let episode = match &mut self.episode {
            Some(episode) if !episode.done => episode,
            Some(_) => {
                return Err(Error::Runtime(
                    "the episode ended; reset to start the next".into(),
                ));
            }
            None => return Err(Error::Runtime("reset before the first step".into())),
        };
        let mut predict = std::time::Duration::ZERO;
        let action = match action {
            // The caller's action makes the rest of a chunk stale.
            Some(action) => {
                episode.replay.clear();
                action
            }
            None => match episode.replay.pop_front() {
                Some(action) => action,
                None => {
                    let observation = ModelObservation {
                        observation: Some(episode.observation.clone()),
                        route: ModelRouteContext {
                            env_id: self.env_id.clone(),
                            episodes: vec![episode.info.clone()],
                            ..Default::default()
                        },
                        num_envs: 1,
                        env_contract: Some(Arc::clone(&self.contract)),
                        history: std::mem::take(&mut episode.history),
                        step: Some(episode.step),
                    };
                    let started = std::time::Instant::now();
                    let frames = self.handler.predict_chunked(observation).await?;
                    predict = started.elapsed();
                    episode.replay = frames
                        .replay
                        .into_iter()
                        .filter_map(|frame| frame.into_iter().next())
                        .collect();
                    frames
                        .actions
                        .into_iter()
                        .next()
                        .ok_or_else(|| Error::model("the model returned no action"))?
                }
            },
        };
        let space = self
            .contract
            .action_space
            .as_ref()
            .ok_or_else(|| Error::Internal("the env has no action space".into()))?;
        crate::model::wire::check_actions_conform(space, std::slice::from_ref(&action))?;
        let leaves = encode_leaf_slab(std::slice::from_ref(&action), space)
            .map_err(|e| Error::invalid_argument(e.to_string()))?;
        let started = std::time::Instant::now();
        let step = self
            .client
            .step_leaves(&[], &leaves, None)
            .await
            .map_err(lockstep_error)?;
        let env = started.elapsed();
        let observation = self.observation(&step.observation)?;
        let terminated = step.terminated.first().copied().unwrap_or(false);
        let truncated = step.truncated.first().copied().unwrap_or(false);
        if let Some(episode) = &mut self.episode {
            episode.step += 1;
            // The model sees this step's observation now only when it
            // predicts on it; a step a replayed action takes is history.
            if self.history && !episode.replay.is_empty() {
                episode.history.push(HistoryFrame {
                    observation: Some(step.observation.clone()),
                    episodes: vec![episode.info.clone()],
                    step: episode.step,
                });
            }
            episode.observation = step.observation;
        }
        if terminated || truncated {
            self.end().await?;
        }
        Ok(BoundStep {
            observation,
            action,
            reward: step.rewards.first().copied().unwrap_or(0.0),
            terminated,
            truncated,
            info: step.info,
            predict,
            env,
        })
    }

    /// The env's rendered frame (PNG), when it renders.
    pub async fn render(&self) -> Result<Option<Bytes>> {
        self.client.render(0, None).await.map_err(lockstep_error)
    }

    /// Ends the live episode, releases the route and the env, and closes
    /// what the binding opened on the model.
    pub async fn close(mut self) -> Result<()> {
        let ended = self.end().await;
        if let Some(setup) = &self.setup {
            setup.release_adapter(&self.env_id).await?;
        }
        let closed = self.handler.on_close().await;
        self.client.close().await;
        ended.and(closed)
    }

    /// Evicts the live episode's state on the model, once.
    async fn end(&mut self) -> Result<()> {
        let Some(episode) = &mut self.episode else {
            return Ok(());
        };
        if episode.done {
            return Ok(());
        }
        episode.done = true;
        episode.replay.clear();
        episode.history.clear();
        let ids = vec![episode.info.episode_id.clone()];
        if let Some(setup) = &self.setup {
            setup.reset_adapter(&self.env_id, &ids).await?;
        }
        self.handler.reset_adapter(&self.env_id, ids).await
    }

    fn observation(&self, leaves: &[Bytes]) -> Result<SpaceValue> {
        let space = self
            .contract
            .observation_space
            .as_ref()
            .ok_or_else(|| Error::Internal("the env has no observation space".into()))?;
        decode_leaf_slab(leaves, space, 1)
            .map_err(|e| Error::Protocol(e.to_string()))?
            .pop()
            .ok_or_else(|| Error::Protocol("the env answered no observation".into()))
    }
}
