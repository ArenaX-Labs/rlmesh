//! The per-episode context a context-reading corner receives, and the
//! model's episode store behind it.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;

use super::types::predict_seed;
use crate::{Error, Result};

/// The live episodes a model tracks before a new one is refused.
pub(crate) const EPISODE_CAPACITY: usize = 65_536;

/// The environment variable overriding [`EPISODE_CAPACITY`].
pub(crate) const EPISODE_CAPACITY_ENV: &str = "RLMESH_EPISODE_CAPACITY";

/// One episode's free state, holding any `Send` value per key. It lives
/// until the episode ends.
#[derive(Default)]
pub struct EpisodeState {
    entries: HashMap<String, Box<dyn Any + Send>>,
}

impl fmt::Debug for EpisodeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut keys: Vec<&String> = self.entries.keys().collect();
        keys.sort();
        f.debug_struct("EpisodeState").field("keys", &keys).finish()
    }
}

impl EpisodeState {
    /// The value under `key`, if present and a `T`.
    pub fn get<T: Any>(&self, key: &str) -> Option<&T> {
        self.entries.get(key)?.downcast_ref()
    }

    /// The value under `key`, mutably, if present and a `T`.
    pub fn get_mut<T: Any>(&mut self, key: &str) -> Option<&mut T> {
        self.entries.get_mut(key)?.downcast_mut()
    }

    /// Store `value` under `key`; the value it replaced, if any.
    pub fn insert<T: Any + Send>(
        &mut self,
        key: impl Into<String>,
        value: T,
    ) -> Option<Box<dyn Any + Send>> {
        self.entries.insert(key.into(), Box::new(value))
    }

    /// The `T` under `key`, inserting `f()` when the key is missing. A key
    /// holding another type is the `TypeError` class naming it.
    pub fn get_or_insert_with<T: Any + Send>(
        &mut self,
        key: &str,
        f: impl FnOnce() -> T,
    ) -> Result<&mut T> {
        let entry = self
            .entries
            .entry(key.to_string())
            .or_insert_with(|| Box::new(f()));
        entry.downcast_mut().ok_or_else(|| {
            Error::Type(format!(
                "episode state key '{key}' holds a value of another type than {}",
                std::any::type_name::<T>()
            ))
        })
    }

    /// Remove the value under `key`.
    pub fn remove(&mut self, key: &str) -> Option<Box<dyn Any + Send>> {
        self.entries.remove(key)
    }

    /// Whether `key` holds a value.
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// How many keys hold a value.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no key holds a value.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// What a context-reading corner learns about the episode it predicts for.
#[derive(Debug)]
#[non_exhaustive]
pub struct PredictContext<'a> {
    /// The episode's id (`""` for an anonymous row).
    pub episode_id: &'a str,
    /// The episode's reset seed, fixed at its first predict.
    pub episode_seed: Option<i64>,
    /// How many context-reading predicts the episode had before this one.
    pub predict_index: u64,
    /// [`predict_seed`](crate::predict_seed)`(episode_seed, predict_index)`,
    /// `None` when the episode is unseeded or anonymous.
    pub predict_seed: Option<i64>,
    /// The episode's free state.
    pub state: &'a mut EpisodeState,
}

/// One context per row of a batched corner.
pub type BatchPredictContext<'a> = [PredictContext<'a>];

/// A row's context, owned while its corner runs: the store lends the
/// episode's state out for the call and takes it back after.
#[derive(Debug)]
pub(crate) struct OwnedContext {
    episode_id: String,
    episode_seed: Option<i64>,
    predict_index: u64,
    predict_seed: Option<i64>,
    state: EpisodeState,
    stored: bool,
}

impl OwnedContext {
    pub(crate) fn view(&mut self) -> PredictContext<'_> {
        PredictContext {
            episode_id: &self.episode_id,
            episode_seed: self.episode_seed,
            predict_index: self.predict_index,
            predict_seed: self.predict_seed,
            state: &mut self.state,
        }
    }

    /// A throwaway context for an anonymous row (or a fused spec-less call).
    pub(crate) fn anonymous(episode_seed: Option<i64>) -> OwnedContext {
        OwnedContext {
            episode_id: String::new(),
            episode_seed,
            predict_index: 0,
            predict_seed: None,
            state: EpisodeState::default(),
            stored: false,
        }
    }
}

struct Entry {
    env_id: String,
    seed: Option<i64>,
    index: u64,
    state: Option<EpisodeState>,
}

/// One model's episode store: entries keyed by `episode_id`, tagged with
/// the env that created them, never evicted.
pub(crate) struct EpisodeStore {
    capacity: usize,
    entries: HashMap<String, Entry>,
}

/// The capacity `RLMESH_EPISODE_CAPACITY` sets: a non-digit value is
/// ignored and `0` is clamped to `1`.
pub(crate) fn capacity_from(raw: Option<&str>) -> usize {
    match raw.map(str::trim) {
        Some(raw) if !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit()) => {
            raw.parse::<usize>().unwrap_or(usize::MAX).max(1)
        }
        _ => EPISODE_CAPACITY,
    }
}

impl EpisodeStore {
    /// A store whose capacity is read once from the environment.
    pub(crate) fn from_env() -> EpisodeStore {
        let raw = std::env::var(EPISODE_CAPACITY_ENV).ok();
        EpisodeStore::with_capacity(capacity_from(raw.as_deref()))
    }

    pub(crate) fn with_capacity(capacity: usize) -> EpisodeStore {
        EpisodeStore {
            capacity: capacity.max(1),
            entries: HashMap::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// The context for one predict of `episode_id`, advancing its
    /// `predict_index`. A new episode past capacity is the builtin
    /// `RuntimeError` class; live episodes are untouched.
    pub(crate) fn begin(
        &mut self,
        episode_id: &str,
        env_id: &str,
        seed: Option<i64>,
    ) -> Result<OwnedContext> {
        if episode_id.is_empty() {
            return Ok(OwnedContext::anonymous(seed));
        }
        if !self.entries.contains_key(episode_id) {
            if self.entries.len() >= self.capacity {
                return Err(Error::Runtime(format!(
                    "model already tracks {} live episodes; refusing episode {episode_id} \
                     rather than evicting another episode's live state. Either episode ends \
                     are not reaching this model, or the deployment runs more concurrent \
                     episodes than {EPISODE_CAPACITY_ENV} allows",
                    self.capacity
                )));
            }
            self.entries.insert(
                episode_id.to_string(),
                Entry {
                    env_id: env_id.to_string(),
                    seed,
                    index: 0,
                    state: Some(EpisodeState::default()),
                },
            );
        }
        let entry = self
            .entries
            .get_mut(episode_id)
            .expect("inserted above when missing");
        let index = entry.index;
        entry.index += 1;
        Ok(OwnedContext {
            episode_id: episode_id.to_string(),
            episode_seed: entry.seed,
            predict_index: index,
            predict_seed: entry.seed.map(|seed| predict_seed(seed, index)),
            state: entry.state.take().unwrap_or_default(),
            stored: true,
        })
    }

    /// Take a context's state back after its corner returned.
    pub(crate) fn finish(&mut self, context: OwnedContext) {
        if !context.stored {
            return;
        }
        if let Some(entry) = self.entries.get_mut(&context.episode_id)
            && entry.state.is_none()
        {
            entry.state = Some(context.state);
        }
    }

    /// Drop `episode_id`'s entry (its destructors run here).
    pub(crate) fn end(&mut self, episode_id: &str) {
        self.entries.remove(episode_id);
    }

    /// Drop every entry `env_id` created, silently.
    pub(crate) fn drop_env(&mut self, env_id: &str) {
        self.entries.retain(|_, entry| entry.env_id != env_id);
    }

    /// Drop every entry.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_reads_digits_only_and_clamps_zero() {
        assert_eq!(capacity_from(None), EPISODE_CAPACITY);
        assert_eq!(capacity_from(Some("12")), 12);
        assert_eq!(capacity_from(Some(" 7 ")), 7);
        assert_eq!(capacity_from(Some("0")), 1);
        assert_eq!(capacity_from(Some("-3")), EPISODE_CAPACITY);
        assert_eq!(capacity_from(Some("ten")), EPISODE_CAPACITY);
        assert_eq!(capacity_from(Some("")), EPISODE_CAPACITY);
    }

    #[test]
    fn store_counts_predicts_fixes_the_seed_and_keeps_state() {
        let mut store = EpisodeStore::with_capacity(4);
        let mut first = store.begin("a", "env", Some(3)).unwrap();
        assert_eq!(first.predict_index, 0);
        assert_eq!(first.predict_seed, Some(predict_seed(3, 0)));
        *first.view().state.get_or_insert_with("n", || 0u32).unwrap() += 5;
        store.finish(first);
        // The seed is fixed at the first predict.
        let second = store.begin("a", "env", Some(99)).unwrap();
        assert_eq!(second.predict_index, 1);
        assert_eq!(second.episode_seed, Some(3));
        assert_eq!(second.state.get::<u32>("n"), Some(&5));
        store.finish(second);
    }

    #[test]
    fn anonymous_rows_are_never_stored() {
        let mut store = EpisodeStore::with_capacity(1);
        let context = store.begin("", "env", Some(1)).unwrap();
        assert_eq!(context.predict_index, 0);
        assert_eq!(context.predict_seed, None);
        assert_eq!(context.episode_seed, Some(1));
        store.finish(context);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn a_new_episode_past_capacity_is_refused_without_evicting() {
        let mut store = EpisodeStore::with_capacity(1);
        let context = store.begin("a", "env", None).unwrap();
        store.finish(context);
        let error = store.begin("b", "env", None).unwrap_err();
        assert_eq!(error.class(), crate::ErrorClass::RuntimeError);
        assert!(
            error.to_string().contains("RLMESH_EPISODE_CAPACITY"),
            "{error}"
        );
        assert!(store.begin("a", "env", None).is_ok());
    }

    #[test]
    fn drop_env_drops_only_that_envs_entries() {
        let mut store = EpisodeStore::with_capacity(8);
        for (episode, env) in [("a", "x"), ("b", "y"), ("c", "x")] {
            let context = store.begin(episode, env, None).unwrap();
            store.finish(context);
        }
        store.drop_env("x");
        assert_eq!(store.len(), 1);
        assert_eq!(store.begin("b", "y", None).unwrap().predict_index, 1);
    }

    #[test]
    fn state_type_mismatch_names_the_key() {
        let mut state = EpisodeState::default();
        state.insert("plans", 1u64);
        let error = state.get_or_insert_with("plans", || 0i32).unwrap_err();
        assert_eq!(error.class(), crate::ErrorClass::TypeError);
        assert!(error.to_string().contains("plans"));
        assert_eq!(state.get::<i32>("plans"), None);
        assert!(state.remove("plans").is_some());
        assert!(state.is_empty());
    }
}
