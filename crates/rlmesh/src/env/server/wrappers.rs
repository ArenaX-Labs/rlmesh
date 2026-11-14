//! The env wrappers a server serves through: [`Tagged`] carries an env's
//! tags, `Marked` notes which server owns a callback.

use super::in_callback;
use crate::Result;
use crate::adapters::EnvTags;
use crate::env::{Env, VectorEnv};

/// An env with tags merged into its metadata.
pub(crate) struct Tagged {
    env: Box<dyn Env + Send>,
    metadata: crate::Metadata,
}

impl Tagged {
    /// Merge and validate `tags` against `env`'s spaces.
    pub(crate) fn new(env: Box<dyn Env + Send>, tags: EnvTags) -> Result<Tagged> {
        tags.validate(&env.observation_space(), &env.action_space())?;
        Self::unchecked(env, tags)
    }

    /// Merge `tags` without validation.
    pub(crate) fn unchecked(env: Box<dyn Env + Send>, tags: EnvTags) -> Result<Tagged> {
        let mut metadata = env.metadata().unwrap_or_default();
        tags.stamp(&mut metadata)?;
        Ok(Tagged { env, metadata })
    }

    /// `env` serving `metadata` as its own (a factory's stamp).
    pub(crate) fn with_metadata(env: Box<dyn Env + Send>, metadata: crate::Metadata) -> Tagged {
        Tagged { env, metadata }
    }
}

impl Env for Tagged {
    fn observation_space(&self) -> crate::spaces::Space {
        self.env.observation_space()
    }
    fn action_space(&self) -> crate::spaces::Space {
        self.env.action_space()
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&crate::InfoDict>,
    ) -> Result<(crate::Value, crate::InfoDict)> {
        self.env.reset(seed, options)
    }
    fn step(
        &mut self,
        action: Option<crate::Value>,
    ) -> Result<(crate::Value, f64, bool, bool, crate::InfoDict)> {
        self.env.step(action)
    }
    fn render(&mut self) -> Result<Option<crate::Value>> {
        self.env.render()
    }
    fn close(&mut self) -> Result<()> {
        self.env.close()
    }
    fn render_mode(&self) -> Option<String> {
        self.env.render_mode()
    }
    fn metadata(&self) -> Option<crate::Metadata> {
        Some(self.metadata.clone())
    }
    fn id(&self) -> Option<String> {
        self.env.id()
    }
}

/// An env run marked as server `id`'s callback.
pub(super) struct Marked<E> {
    pub(super) id: u64,
    pub(super) env: Option<E>,
}

impl<E> Drop for Marked<E> {
    fn drop(&mut self) {
        in_callback(self.id, || drop(self.env.take()));
    }
}

impl<E: Env + ?Sized> Env for Marked<Box<E>> {
    fn observation_space(&self) -> crate::spaces::Space {
        self.env.as_ref().expect("marked env").observation_space()
    }
    fn action_space(&self) -> crate::spaces::Space {
        self.env.as_ref().expect("marked env").action_space()
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&crate::InfoDict>,
    ) -> Result<(crate::Value, crate::InfoDict)> {
        in_callback(self.id, || {
            self.env.as_mut().expect("marked env").reset(seed, options)
        })
    }
    fn step(
        &mut self,
        action: Option<crate::Value>,
    ) -> Result<(crate::Value, f64, bool, bool, crate::InfoDict)> {
        in_callback(self.id, || {
            self.env.as_mut().expect("marked env").step(action)
        })
    }
    fn render(&mut self) -> Result<Option<crate::Value>> {
        in_callback(self.id, || self.env.as_mut().expect("marked env").render())
    }
    fn close(&mut self) -> Result<()> {
        in_callback(self.id, || self.env.as_mut().expect("marked env").close())
    }
}

impl<V: VectorEnv + ?Sized> VectorEnv for Marked<Box<V>> {
    fn num_envs(&self) -> u64 {
        self.env.as_ref().expect("marked env").num_envs()
    }
    fn single_observation_space(&self) -> crate::spaces::Space {
        self.env
            .as_ref()
            .expect("marked env")
            .single_observation_space()
    }
    fn single_action_space(&self) -> crate::spaces::Space {
        self.env.as_ref().expect("marked env").single_action_space()
    }
    fn reset(
        &mut self,
        seed: Option<super::super::VectorSeed>,
        options: Option<&crate::InfoDict>,
    ) -> Result<(crate::Value, crate::InfoDict)> {
        in_callback(self.id, || {
            self.env.as_mut().expect("marked env").reset(seed, options)
        })
    }
    fn step(
        &mut self,
        actions: crate::Value,
    ) -> Result<(
        crate::Value,
        Vec<f64>,
        Vec<bool>,
        Vec<bool>,
        crate::InfoDict,
    )> {
        in_callback(self.id, || {
            self.env.as_mut().expect("marked env").step(actions)
        })
    }
    fn render(&mut self) -> Result<Option<crate::Value>> {
        in_callback(self.id, || self.env.as_mut().expect("marked env").render())
    }
    fn close(&mut self) -> Result<()> {
        in_callback(self.id, || self.env.as_mut().expect("marked env").close())
    }
}
