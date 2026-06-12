//! Episode tracking for evaluation.

use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use rlmesh_proto::env::v1::EpisodeMetadata;

/// Single episode state (internal).
struct Episode {
    id: String,
    seed: i64,
    env_index: i32,
    step_count: i64,
    cumulative_reward: f64,
    start_time: Instant,
    start_timestamp_ns: i64,
}

impl Episode {
    fn new(env_index: i32, seed: i64) -> Self {
        let start_time = Instant::now();
        let start_timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        Self {
            id: Uuid::new_v4().to_string(),
            seed,
            env_index,
            step_count: 0,
            cumulative_reward: 0.0,
            start_time,
            start_timestamp_ns,
        }
    }

    fn record_step(&mut self, reward: f64) {
        self.step_count += 1;
        self.cumulative_reward += reward;
    }

    fn complete(
        self,
        terminated: bool,
        truncated: bool,
        final_info: Option<prost_types::Struct>,
    ) -> EpisodeMetadata {
        let end_timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        let duration_ms = self.start_time.elapsed().as_millis() as i64;

        EpisodeMetadata {
            episode_id: self.id,
            seed: self.seed,
            env_index: self.env_index,
            step_count: self.step_count,
            cumulative_reward: self.cumulative_reward,
            terminated,
            truncated,
            start_timestamp_ns: self.start_timestamp_ns,
            end_timestamp_ns,
            duration_ms,
            final_info,
        }
    }
}

/// Manages episode tracking for all environments.
pub struct EpisodeTracker {
    active: HashMap<i32, Episode>,
}

impl EpisodeTracker {
    /// Create a new episode tracker.
    pub fn new() -> Self {
        Self {
            active: HashMap::new(),
        }
    }

    /// Start a new episode for the given environment index.
    ///
    /// Returns the episode ID.
    pub fn start_episode(&mut self, env_index: i32, seed: i64) -> String {
        let episode = Episode::new(env_index, seed);
        let episode_id = episode.id.clone();

        // Replace any existing episode for this env_index
        if let Some(old_episode) = self.active.insert(env_index, episode) {
            tracing::warn!(
                "Starting new episode for env {} while previous episode {} was still active",
                env_index,
                old_episode.id
            );
        }

        episode_id
    }

    /// Record a step for the given environment.
    pub fn record_step(&mut self, env_index: i32, reward: f64) {
        if let Some(episode) = self.active.get_mut(&env_index) {
            episode.record_step(reward);
        } else {
            tracing::warn!(
                "Attempted to record step for env {} with no active episode",
                env_index
            );
        }
    }

    /// Complete an episode and return its metadata.
    ///
    /// Returns None if no episode was active for the given environment.
    pub fn complete_episode(
        &mut self,
        env_index: i32,
        terminated: bool,
        truncated: bool,
        final_info: Option<prost_types::Struct>,
    ) -> Option<EpisodeMetadata> {
        let episode = self.active.remove(&env_index)?;
        Some(episode.complete(terminated, truncated, final_info))
    }

    /// Complete all active episodes (e.g., on cancellation).
    ///
    /// Returns metadata for all episodes that were active.
    pub fn complete_all(&mut self, reason: &str) -> Vec<EpisodeMetadata> {
        tracing::info!(
            "Completing all {} active episodes: {}",
            self.active.len(),
            reason
        );

        let mut completed = Vec::new();
        for (_env_index, episode) in self.active.drain() {
            let metadata = episode.complete(false, true, None);
            completed.push(metadata);
        }

        completed
    }

    /// Get the active episode ID for a specific environment index.
    pub fn active_episode_id(&self, env_index: i32) -> Option<&str> {
        self.active
            .get(&env_index)
            .map(|episode| episode.id.as_str())
    }
}

impl Default for EpisodeTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn active_count(tracker: &EpisodeTracker) -> usize {
        tracker.active.len()
    }

    #[test]
    fn test_episode_lifecycle() {
        let mut tracker = EpisodeTracker::new();

        // Start episode
        let ep_id = tracker.start_episode(0, 42);
        assert_eq!(active_count(&tracker), 1);

        // Record steps
        tracker.record_step(0, 1.0);
        tracker.record_step(0, 2.5);

        // Complete episode
        let metadata = tracker.complete_episode(0, true, false, None).unwrap();
        assert_eq!(metadata.episode_id, ep_id);
        assert_eq!(metadata.seed, 42);
        assert_eq!(metadata.env_index, 0);
        assert_eq!(metadata.step_count, 2);
        assert_eq!(metadata.cumulative_reward, 3.5);
        assert!(metadata.terminated);
        assert!(!metadata.truncated);

        assert_eq!(active_count(&tracker), 0);
    }

    #[test]
    fn test_vectorized_episodes() {
        let mut tracker = EpisodeTracker::new();

        // Start multiple episodes
        let ep0 = tracker.start_episode(0, 100);
        let ep1 = tracker.start_episode(1, 200);
        let _ep2 = tracker.start_episode(2, 300);
        assert_eq!(active_count(&tracker), 3);

        // Record steps for each
        tracker.record_step(0, 1.0);
        tracker.record_step(1, 2.0);
        tracker.record_step(2, 3.0);

        // Complete env 1
        let meta1 = tracker.complete_episode(1, true, false, None).unwrap();
        assert_eq!(meta1.episode_id, ep1);
        assert_eq!(active_count(&tracker), 2);

        // Complete env 0
        let meta0 = tracker.complete_episode(0, false, true, None).unwrap();
        assert_eq!(meta0.episode_id, ep0);
        assert!(!meta0.terminated);
        assert!(meta0.truncated);
        assert_eq!(active_count(&tracker), 1);

        // Complete env 2
        tracker.complete_episode(2, true, false, None);
        assert_eq!(active_count(&tracker), 0);
    }

    #[test]
    fn test_complete_all() {
        let mut tracker = EpisodeTracker::new();

        tracker.start_episode(0, 1);
        tracker.start_episode(1, 2);
        tracker.start_episode(2, 3);

        tracker.record_step(0, 1.0);
        tracker.record_step(1, 2.0);

        let interrupted = tracker.complete_all("test cancellation");
        assert_eq!(interrupted.len(), 3);
        assert_eq!(active_count(&tracker), 0);

        // All should be truncated (not terminated)
        for meta in interrupted {
            assert!(!meta.terminated);
            assert!(meta.truncated);
        }
    }
}
