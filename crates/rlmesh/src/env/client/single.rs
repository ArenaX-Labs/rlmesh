//! [`RemoteEnv`]: the client for one remote env.

use super::client_core::{EnvClientCore, cancellable, connect_core, teardown};
use super::{Lendable, RemoteEnvOptions, Shared};
use crate::run::drive::{block_on_library, in_library, refuse_in_async_context};
use crate::spaces::{Space, SpaceSpec};
use crate::{EnvContract, Error, InfoDict, Metadata, Result, Value};

/// A client for one remote env.
///
/// Construction dials and handshakes; the env's exclusive session slot is
/// taken by the first `reset`, `step` or `render`. Calls block (each has an
/// `_async` twin); `&mut self` serializes them. Dropping an unclosed client
/// closes it with `close`'s bound.
pub struct RemoteEnv {
    core: Option<Box<EnvClientCore>>,
    shared: Shared,
    lent_edition: String,
    cancel: crate::ext::CancelToken,
}

client_common!(RemoteEnv, "RemoteEnv");

impl RemoteEnv {
    pub(crate) fn reset_cancel(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
        cancel: Option<crate::ext::CancelToken>,
    ) -> Result<(Value, InfoDict)> {
        refuse_in_async_context("RemoteEnv::reset")?;
        let subscriptions = self.admit(cancel.as_ref())?;
        block_on_library(self.reset_in(seed, options, Some(subscriptions)))
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn step_cancel(
        &mut self,
        action: Value,
        cancel: Option<crate::ext::CancelToken>,
    ) -> Result<(Value, f64, bool, bool, InfoDict)> {
        refuse_in_async_context("RemoteEnv::step")?;
        let subscriptions = self.admit(cancel.as_ref())?;
        block_on_library(self.step_in(action, Some(subscriptions)))
    }

    /// Dial and handshake. Invalid options are the ValueError class, a dial
    /// failure the ConnectionError class, a refused handshake the
    /// ProtocolException class; an endpoint with more than one env is
    /// refused.
    pub fn new(options: RemoteEnvOptions) -> Result<RemoteEnv> {
        refuse_in_async_context("RemoteEnv::new")?;
        block_on_library(Self::new_in(options))
    }

    /// The async twin of [`new`](Self::new).
    pub async fn new_async(options: RemoteEnvOptions) -> Result<RemoteEnv> {
        in_library(Self::new_in(options)).await
    }

    async fn new_in(options: RemoteEnvOptions) -> Result<RemoteEnv> {
        let mut core = connect_core(options).await?;
        core.require_single()?;
        core.set_name("RemoteEnv");
        Ok(RemoteEnv {
            shared: Shared::of(&core)?,
            lent_edition: String::new(),
            core: Some(Box::new(core)),
            cancel: crate::ext::CancelToken::new(),
        })
    }

    /// The observation space (built once).
    pub fn observation_space(&self) -> &Space {
        &self.shared.observation_space
    }

    /// The action space (built once).
    pub fn action_space(&self) -> &Space {
        &self.shared.action_space
    }

    /// Start an episode: `(observation, info)`. `info` carries the
    /// runtime-reserved `episode_ids`.
    pub fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        refuse_in_async_context("RemoteEnv::reset")?;
        block_on_library(self.reset_in(seed, options, None))
    }

    /// The async twin of [`reset`](Self::reset).
    pub async fn reset_async(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        in_library(self.reset_in(seed, options, None)).await
    }

    async fn reset_in(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
        subscriptions: Option<Vec<crate::ext::CancelSubscription>>,
    ) -> Result<(Value, InfoDict)> {
        let subscriptions = match subscriptions {
            Some(s) => s,
            None => self.admit(None)?,
        };
        let options = options.cloned();
        self.io_subscribed(subscriptions, move |core| {
            Box::pin(async move {
                let (lanes, info) = core
                    .reset_values(seed.into_iter().collect(), options.as_ref(), Vec::new())
                    .await?;
                Ok((lanes.into_iter().next().unwrap_or(Value::None), info))
            })
        })
        .await
    }

    /// Apply one action: `(observation, reward, terminated, truncated,
    /// info)`; `info` carries `episode_ids` and `completed_episodes`.
    #[allow(clippy::type_complexity)]
    pub fn step(&mut self, action: Value) -> Result<(Value, f64, bool, bool, InfoDict)> {
        refuse_in_async_context("RemoteEnv::step")?;
        block_on_library(self.step_in(action, None))
    }

    /// The async twin of [`step`](Self::step).
    #[allow(clippy::type_complexity)]
    pub async fn step_async(
        &mut self,
        action: Value,
    ) -> Result<(Value, f64, bool, bool, InfoDict)> {
        in_library(self.step_in(action, None)).await
    }

    #[allow(clippy::type_complexity)]
    async fn step_in(
        &mut self,
        action: Value,
        subscriptions: Option<Vec<crate::ext::CancelSubscription>>,
    ) -> Result<(Value, f64, bool, bool, InfoDict)> {
        let subscriptions = match subscriptions {
            Some(s) => s,
            None => self.admit(None)?,
        };
        self.io_subscribed(subscriptions, move |core| {
            Box::pin(async move {
                let (lanes, rewards, terminated, truncated, info) =
                    core.step_values(vec![action]).await?;
                Ok((
                    lanes.into_iter().next().unwrap_or(Value::None),
                    rewards.first().copied().unwrap_or_default(),
                    terminated.first().copied().unwrap_or_default(),
                    truncated.first().copied().unwrap_or_default(),
                    info,
                ))
            })
        })
        .await
    }
}
