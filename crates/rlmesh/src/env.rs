//! Env authoring and serving: Python's Gymnasium env shape, [`EnvServer`],
//! [`EnvFactory`] and the remote clients.
//!
//! An author writes an env in Gymnasium shape ([`Env`], or [`VectorEnv`] for a
//! natively batched one) and the library derives everything else: the
//! [`EnvServer`] reads the spaces, id, render mode and metadata once at
//! construction and builds the served [`EnvContract`](crate::specs::EnvContract).
//! Authoring is sync; a lane runs on its own thread for the server's life.

mod batch;
mod client;
pub(crate) mod envelope;
mod factory;
pub(crate) mod lanes;
pub(crate) mod legacy;
pub(crate) mod legacy_server;
mod server;
mod signals;
mod single_lane;
mod sync_wire;
mod trial;
pub(crate) mod types;
pub(crate) mod wire;

use std::sync::{Arc, Mutex};

pub use client::{RemoteEnv, RemoteVectorEnv};
pub use factory::{
    EnvFactory, EnvFactoryExt, EnvFactoryServeOptions, MadeEnv, TagParam, Variant, VariantOptions,
};
pub use server::{EnvServer, EnvServerOptions, Transport};
pub use trial::trial_index;

use crate::spaces::Space;
use crate::{InfoDict, Metadata, Result, Value};

/// One environment in Gymnasium shape: Python's `EnvLike` protocol.
///
/// `reset` returns `(observation, info)` and `step` returns `(observation,
/// reward, terminated, truncated, info)`, as Gymnasium's do. The server reads
/// the spaces, `render_mode`, `metadata` and `id` once, when it is built.
///
/// A failure is an [`Error`](crate::Error): return
/// [`Error::environment_recoverable`](crate::Error::environment_recoverable) to
/// fail one request and keep the client's session; any other error ends it,
/// and the client reads `[INTERNAL] {op} failed: {message}`.
pub trait Env {
    /// The space observations belong to.
    fn observation_space(&self) -> Space;

    /// The space actions belong to.
    fn action_space(&self) -> Space;

    /// Start an episode: Python's `reset(*, seed=None, options=None)`.
    fn reset(&mut self, seed: Option<i64>, options: Option<&InfoDict>)
    -> Result<(Value, InfoDict)>;

    /// Apply one action. The action is `None` when the request carries none,
    /// as Python calls `step(None)`.
    fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)>;

    /// The current frame as an image value (cast to uint8; `[H, W, 3]`,
    /// `[H, W, 4]`, `[3, H, W]`, `[4, H, W]` or `[H, W]` are encoded, any other
    /// shape gives no frame). `None` (the default) is no frame.
    fn render(&mut self) -> Result<Option<Value>> {
        Ok(None)
    }

    /// Release the env's resources. The default does nothing, as
    /// `gymnasium.Env.close` does.
    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    /// Python's `render_mode`; `None` by default.
    fn render_mode(&self) -> Option<String> {
        None
    }

    /// Python's `metadata` dict; `None` by default. Its `autoreset_mode` key
    /// sets the served autoreset mode.
    fn metadata(&self) -> Option<Metadata> {
        None
    }

    /// Python's `spec.id`; `None` serves `"UnknownEnv-v1"`.
    fn id(&self) -> Option<String> {
        None
    }
}

/// The seed a [`VectorEnv`] reset receives: Python's `int | list[int]` (None
/// is `Option::None`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VectorSeed {
    /// One seed for the whole vector (Python passes an int).
    One(i64),
    /// One seed per sub-environment.
    Each(Vec<i64>),
}

impl From<i64> for VectorSeed {
    fn from(seed: i64) -> Self {
        VectorSeed::One(seed)
    }
}

impl From<Vec<i64>> for VectorSeed {
    fn from(seeds: Vec<i64>) -> Self {
        VectorSeed::Each(seeds)
    }
}

impl From<&[i64]> for VectorSeed {
    fn from(seeds: &[i64]) -> Self {
        VectorSeed::Each(seeds.to_vec())
    }
}

/// A natively batched environment: Python's `VectorEnvLike` protocol.
///
/// `actions` and observations use the batched layout a default Python vector
/// env sees: a Box, MultiBinary or MultiDiscrete batch is one stacked
/// `Tensor` of shape `[num_envs, *shape]`, a Discrete batch one int64 `Tensor`
/// of shape `[num_envs]`, a Dict or Tuple batch is batched per child, and any
/// other kind is a `List` of per-lane values. Returned observations may also
/// give a stacked kind as a `List` of per-lane values.
pub trait VectorEnv {
    /// The number of sub-environments stepped together.
    fn num_envs(&self) -> u64;

    /// The space one sub-environment's observation belongs to.
    fn single_observation_space(&self) -> Space;

    /// The space one sub-environment's action belongs to.
    fn single_action_space(&self) -> Space;

    /// Reset every sub-environment.
    fn reset(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)>;

    /// Step every sub-environment with its action.
    #[allow(clippy::type_complexity)] // Python's five-tuple, kept as a tuple (R-15).
    fn step(&mut self, actions: Value)
    -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)>;

    /// The current frames, batched; `None` (the default) is no frame.
    fn render(&mut self) -> Result<Option<Value>> {
        Ok(None)
    }

    /// Release the env's resources; the default does nothing.
    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    /// Python's `render_mode`; `None` by default.
    fn render_mode(&self) -> Option<String> {
        None
    }

    /// Python's `metadata` dict; `None` by default.
    fn metadata(&self) -> Option<Metadata> {
        None
    }

    /// Python's `spec.id`; `None` serves `"UnknownEnv-v1"`.
    fn id(&self) -> Option<String> {
        None
    }
}

impl<E: Env + ?Sized> Env for Box<E> {
    fn observation_space(&self) -> Space {
        (**self).observation_space()
    }
    fn action_space(&self) -> Space {
        (**self).action_space()
    }
    fn render_mode(&self) -> Option<String> {
        (**self).render_mode()
    }
    fn metadata(&self) -> Option<Metadata> {
        (**self).metadata()
    }
    fn id(&self) -> Option<String> {
        (**self).id()
    }

    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        (**self).reset(seed, options)
    }
    fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        (**self).step(action)
    }
    fn render(&mut self) -> Result<Option<Value>> {
        (**self).render()
    }
    fn close(&mut self) -> Result<()> {
        (**self).close()
    }
}

fn locked<T: ?Sized>(env: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    env.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl<E: Env> Env for Arc<Mutex<E>> {
    fn observation_space(&self) -> Space {
        locked(self).observation_space()
    }
    fn action_space(&self) -> Space {
        locked(self).action_space()
    }
    fn render_mode(&self) -> Option<String> {
        locked(self).render_mode()
    }
    fn metadata(&self) -> Option<Metadata> {
        locked(self).metadata()
    }
    fn id(&self) -> Option<String> {
        locked(self).id()
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        locked(self).reset(seed, options)
    }
    fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        locked(self).step(action)
    }
    fn render(&mut self) -> Result<Option<Value>> {
        locked(self).render()
    }
    fn close(&mut self) -> Result<()> {
        locked(self).close()
    }
}

impl<V: VectorEnv + ?Sized> VectorEnv for Box<V> {
    fn num_envs(&self) -> u64 {
        (**self).num_envs()
    }
    fn single_observation_space(&self) -> Space {
        (**self).single_observation_space()
    }
    fn single_action_space(&self) -> Space {
        (**self).single_action_space()
    }
    fn render_mode(&self) -> Option<String> {
        (**self).render_mode()
    }
    fn metadata(&self) -> Option<Metadata> {
        (**self).metadata()
    }
    fn id(&self) -> Option<String> {
        (**self).id()
    }

    fn reset(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        (**self).reset(seed, options)
    }
    fn step(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        (**self).step(actions)
    }
    fn render(&mut self) -> Result<Option<Value>> {
        (**self).render()
    }
    fn close(&mut self) -> Result<()> {
        (**self).close()
    }
}

impl<V: VectorEnv> VectorEnv for Arc<Mutex<V>> {
    fn num_envs(&self) -> u64 {
        locked(self).num_envs()
    }
    fn single_observation_space(&self) -> Space {
        locked(self).single_observation_space()
    }
    fn single_action_space(&self) -> Space {
        locked(self).single_action_space()
    }
    fn render_mode(&self) -> Option<String> {
        locked(self).render_mode()
    }
    fn metadata(&self) -> Option<Metadata> {
        locked(self).metadata()
    }
    fn id(&self) -> Option<String> {
        locked(self).id()
    }
    fn reset(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        locked(self).reset(seed, options)
    }
    fn step(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        locked(self).step(actions)
    }
    fn render(&mut self) -> Result<Option<Value>> {
        locked(self).render()
    }
    fn close(&mut self) -> Result<()> {
        locked(self).close()
    }
}

#[cfg(test)]
mod tests;
