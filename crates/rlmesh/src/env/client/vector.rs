//! [`RemoteVectorEnv`]: the client for a remote vector env.

use super::client_core::{EnvClientCore, cancellable, connect_core, teardown};
use super::link::wire_lane;
use super::{Lendable, RemoteEnvOptions, Shared};
use crate::run::drive::{block_on_library, in_library, refuse_in_async_context};
use crate::spaces::{Space, SpaceSpec};
use crate::{EnvContract, Error, InfoDict, Metadata, Result, Value, VectorSeed};

/// A client for a remote vector env (two or more lanes stepped in lockstep):
/// A remote vector environment. Observations and actions are one batched
/// [`Value`]; rewards and flags are per-lane vectors.
pub struct RemoteVectorEnv {
    core: Option<Box<EnvClientCore>>,
    shared: Shared,
    lent_edition: String,
    cancel: crate::ext::CancelToken,
}

client_common!(RemoteVectorEnv, "RemoteVectorEnv");

impl RemoteVectorEnv {
    pub(crate) fn reset_cancel(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
        cancel: Option<crate::ext::CancelToken>,
    ) -> Result<(Value, InfoDict)> {
        refuse_in_async_context("RemoteVectorEnv::reset")?;
        let subscriptions = self.admit(cancel.as_ref())?;
        block_on_library(self.reset_in(seed, options, Vec::new(), Some(subscriptions)))
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn step_cancel(
        &mut self,
        action: Value,
        cancel: Option<crate::ext::CancelToken>,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        refuse_in_async_context("RemoteVectorEnv::step")?;
        let lanes = self.unstack_actions(action)?;
        let subscriptions = self.admit(cancel.as_ref())?;
        block_on_library(self.step_lanes_in(lanes, subscriptions))
    }

    /// Dial and handshake. An endpoint with fewer than two envs is refused
    /// (the ValueError class).
    pub fn new(options: RemoteEnvOptions) -> Result<RemoteVectorEnv> {
        refuse_in_async_context("RemoteVectorEnv::new")?;
        block_on_library(Self::new_in(options))
    }

    /// The async twin of [`new`](Self::new).
    pub async fn new_async(options: RemoteEnvOptions) -> Result<RemoteVectorEnv> {
        in_library(Self::new_in(options)).await
    }

    async fn new_in(options: RemoteEnvOptions) -> Result<RemoteVectorEnv> {
        let mut core = connect_core(options).await?;
        core.require_vector()?;
        core.set_name("RemoteVectorEnv");
        Ok(RemoteVectorEnv {
            shared: Shared::of(&core)?,
            lent_edition: String::new(),
            core: Some(Box::new(core)),
            cancel: crate::ext::CancelToken::new(),
        })
    }

    /// Lanes the endpoint steps in lockstep.
    pub fn num_envs(&self) -> u64 {
        self.shared.num_envs
    }

    /// One lane's observation space (built once).
    pub fn single_observation_space(&self) -> &Space {
        &self.shared.observation_space
    }

    /// One lane's action space (built once).
    pub fn single_action_space(&self) -> &Space {
        &self.shared.action_space
    }

    /// The same object as [`single_observation_space`](Self::single_observation_space).
    pub fn observation_space(&self) -> &Space {
        &self.shared.observation_space
    }

    /// The same object as [`single_action_space`](Self::single_action_space).
    pub fn action_space(&self) -> &Space {
        &self.shared.action_space
    }

    /// Reset every lane: the batched observation and the info (with one
    /// `episode_ids` entry per lane). An int seed is sent as a one-element
    /// list.
    pub fn reset(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        refuse_in_async_context("RemoteVectorEnv::reset")?;
        block_on_library(self.reset_in(seed, options, Vec::new(), None))
    }

    /// The async twin of [`reset`](Self::reset).
    pub async fn reset_async(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        in_library(self.reset_in(seed, options, Vec::new(), None)).await
    }

    pub(crate) async fn reset_in(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
        env_indices: Vec<u32>,
        subscriptions: Option<Vec<crate::ext::CancelSubscription>>,
    ) -> Result<(Value, InfoDict)> {
        let subscriptions = match subscriptions {
            Some(s) => s,
            None => self.admit(None)?,
        };
        let seeds = match seed {
            None => Vec::new(),
            Some(VectorSeed::One(seed)) => vec![seed],
            Some(VectorSeed::Each(seeds)) => seeds,
        };
        let options = options.cloned();
        let observation = self.shared.contract.observation_space().wire().clone();
        self.io_subscribed(subscriptions, move |core| {
            Box::pin(async move {
                let (lanes, info) = core
                    .reset_values(seeds, options.as_ref(), env_indices)
                    .await?;
                Ok((super::super::batch::stack_lanes(&observation, lanes)?, info))
            })
        })
        .await
    }

    /// Apply one batched action: the batched observation, per-lane rewards,
    /// terminated and truncated flags, and the info (`completed_episodes` is
    /// the count of lanes whose episode ended).
    #[allow(clippy::type_complexity)]
    pub fn step(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        refuse_in_async_context("RemoteVectorEnv::step")?;
        block_on_library(self.step_in(actions))
    }

    /// The async twin of [`step`](Self::step).
    #[allow(clippy::type_complexity)]
    pub async fn step_async(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        in_library(self.step_in(actions)).await
    }

    fn unstack_actions(&self, actions: Value) -> Result<Vec<Value>> {
        let action = self.shared.contract.action_space().wire().clone();
        super::super::batch::unstack_lanes(&action, actions, self.shared.num_envs as usize)
    }

    #[allow(clippy::type_complexity)]
    async fn step_in(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        let lanes = self.unstack_actions(actions)?;
        let subscriptions = self.admit(None)?;
        self.step_lanes_in(lanes, subscriptions).await
    }

    #[allow(clippy::type_complexity)]
    async fn step_lanes_in(
        &mut self,
        lanes: Vec<Value>,
        subscriptions: Vec<crate::ext::CancelSubscription>,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        let observation = self.shared.contract.observation_space().wire().clone();
        self.io_subscribed(subscriptions, move |core| {
            Box::pin(async move {
                let (lanes, rewards, terminated, truncated, info) = core.step_values(lanes).await?;
                Ok((
                    super::super::batch::stack_lanes(&observation, lanes)?,
                    rewards,
                    terminated,
                    truncated,
                    info,
                ))
            })
        })
        .await
    }
}

/// `ext::reset_lanes`' body.
pub(crate) async fn reset_lanes_in(
    env: &mut RemoteVectorEnv,
    env_indices: &[u64],
    seed: Option<VectorSeed>,
    options: Option<&InfoDict>,
) -> Result<(Value, InfoDict)> {
    let lanes = env_indices
        .iter()
        .map(|index| wire_lane(*index))
        .collect::<Result<Vec<u32>>>()?;
    env.reset_in(seed, options, lanes, None).await
}
