use super::*;
use crate::{Env, Error, InfoDict, Metadata, Result, Value, VectorEnv, VectorSeed};
use std::fmt;
/// Tagging keywords. Build with `..Default::default()`; adding an options
/// field is a minor change by project policy.
#[derive(Clone, Debug)]
pub struct TagOptions {
    pub validate: bool,
}
impl Default for TagOptions {
    fn default() -> Self {
        Self { validate: true }
    }
}
impl TagOptions {
    pub fn validate(mut self, value: bool) -> Self {
        self.validate = value;
        self
    }
}
/// An environment with immutable adapter tags merged over its metadata.
pub struct Tagged<E> {
    inner: E,
    tags: EnvTags,
    fragment: Metadata,
}
impl<E> Tagged<E> {
    pub fn get_ref(&self) -> &E {
        &self.inner
    }
    pub fn get_mut(&mut self) -> &mut E {
        &mut self.inner
    }
    pub fn into_inner(self) -> E {
        self.inner
    }
    pub fn tags(&self) -> &EnvTags {
        &self.tags
    }
    fn metadata(&self, base: Option<Metadata>) -> Metadata {
        let mut m = base.unwrap_or_default();
        for (k, v) in &self.fragment {
            m.insert(k, v.clone());
        }
        m
    }
}
/// A failed tag returns ownership of the original environment.
pub struct TagError<E> {
    env: E,
    error: Error,
}
impl<E> TagError<E> {
    pub fn error(&self) -> &Error {
        &self.error
    }
    pub fn into_parts(self) -> (E, Error) {
        (self.env, self.error)
    }
}
impl<E> fmt::Debug for TagError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.error, f)
    }
}
impl<E> fmt::Display for TagError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl<E> std::error::Error for TagError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
impl<E> From<TagError<E>> for Error {
    fn from(e: TagError<E>) -> Self {
        e.error
    }
}
fn prepare(tags: &EnvTags, spaces: Option<(Space, Space)>) -> Result<Metadata> {
    let run = || -> Result<Metadata> {
        let fragment = tags.to_metadata()?;
        if let Some((obs, action)) = spaces {
            let joined = core::join(
                &tags.wire()?,
                &core::SpaceView::from(obs.spec().wire()),
                &core::SpaceView::from(action.spec().wire()),
            )
            .map_err(resolution_error)?;
            for note in joined.advisories {
                crate::warnings::warn(crate::WarningCategory::UserWarning, note.message)?
            }
        }
        Ok(fragment)
    };
    run().map_err(resolution_error)
}
/// Attach tags to an authored single environment, retaining it on failure.
pub fn tag<E: Env>(
    env: E,
    tags: &EnvTags,
    options: TagOptions,
) -> std::result::Result<Tagged<E>, TagError<E>> {
    let spaces = options
        .validate
        .then(|| (env.observation_space(), env.action_space()));
    match prepare(tags, spaces) {
        Ok(fragment) => Ok(Tagged {
            inner: env,
            tags: tags.clone(),
            fragment,
        }),
        Err(error) => Err(TagError { env, error }),
    }
}
/// Attach tags to an authored vector environment, validating its lane spaces.
pub fn tag_vector<E: VectorEnv>(
    env: E,
    tags: &EnvTags,
    options: TagOptions,
) -> std::result::Result<Tagged<E>, TagError<E>> {
    let spaces = options
        .validate
        .then(|| (env.single_observation_space(), env.single_action_space()));
    match prepare(tags, spaces) {
        Ok(fragment) => Ok(Tagged {
            inner: env,
            tags: tags.clone(),
            fragment,
        }),
        Err(error) => Err(TagError { env, error }),
    }
}
impl<E: Env> Env for Tagged<E> {
    fn observation_space(&self) -> Space {
        self.inner.observation_space()
    }
    fn action_space(&self) -> Space {
        self.inner.action_space()
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.inner.reset(seed, options)
    }
    fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.inner.step(action)
    }
    fn render(&mut self) -> Result<Option<Value>> {
        self.inner.render()
    }
    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }
    fn render_mode(&self) -> Option<String> {
        self.inner.render_mode()
    }
    fn metadata(&self) -> Option<Metadata> {
        Some(self.metadata(self.inner.metadata()))
    }
    fn id(&self) -> Option<String> {
        self.inner.id()
    }
}
impl<E: VectorEnv> VectorEnv for Tagged<E> {
    fn num_envs(&self) -> u64 {
        self.inner.num_envs()
    }
    fn single_observation_space(&self) -> Space {
        self.inner.single_observation_space()
    }
    fn single_action_space(&self) -> Space {
        self.inner.single_action_space()
    }
    fn reset(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.inner.reset(seed, options)
    }
    fn step(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        self.inner.step(actions)
    }
    fn render(&mut self) -> Result<Option<Value>> {
        self.inner.render()
    }
    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }
    fn render_mode(&self) -> Option<String> {
        self.inner.render_mode()
    }
    fn metadata(&self) -> Option<Metadata> {
        Some(self.metadata(self.inner.metadata()))
    }
    fn id(&self) -> Option<String> {
        self.inner.id()
    }
}
