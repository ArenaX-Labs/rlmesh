//! Episode tracking for evaluation.

use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use rlmesh_proto::env::v1::EpisodeMetadata;
use rlmesh_proto::spaces::v1::MetaMap;

/// Single episode state (internal).
struct Episode {
    id: String,
    seed: Option<i64>,
    env_index: i32,
    step_count: i64,
    cumulative_reward: f64,
    start_time: Instant,
    start_timestamp_ns: i64,
}

impl Episode {
    fn new(env_index: i32, seed: Option<i64>) -> Self {
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
        final_info: Option<MetaMap>,
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

/// Maximum interrupted episodes retained between drains.
///
/// Reset-only clients can produce interrupted episodes faster than they are
/// reported. The cap bounds memory and response size while retaining recent
/// interruptions.
const MAX_INTERRUPTED_EPISODES: usize = 100;

/// Manages episode tracking for all environments.
pub struct EpisodeTracker {
    active: HashMap<i32, Episode>,
    /// Episodes interrupted by a replacing reset, awaiting the next drain.
    /// Bounded by [`MAX_INTERRUPTED_EPISODES`] with oldest-first eviction.
    interrupted: Vec<EpisodeMetadata>,
    /// Count of interrupted episodes dropped because the buffer was full since
    /// the last drain, surfaced in a single warn log when the drain happens.
    interrupted_dropped: u64,
}

impl EpisodeTracker {
    /// Create a new episode tracker.
    pub fn new() -> Self {
        Self {
            active: HashMap::new(),
            interrupted: Vec::new(),
            interrupted_dropped: 0,
        }
    }

    /// Buffer an interrupted episode, evicting the oldest entry (and counting
    /// the drop) when the buffer is already at [`MAX_INTERRUPTED_EPISODES`]. A
    /// client that loops `Reset` without `Step` cannot grow this buffer without
    /// bound; the dropped count is surfaced when the buffer is next drained.
    fn push_interrupted(&mut self, metadata: EpisodeMetadata) {
        if self.interrupted.len() >= MAX_INTERRUPTED_EPISODES {
            // Evict oldest-first so the most recent interruptions are retained.
            self.interrupted.remove(0);
            self.interrupted_dropped = self.interrupted_dropped.saturating_add(1);
        }
        self.interrupted.push(metadata);
    }

    /// Take the buffered interrupted episodes, logging any that were dropped
    /// because the buffer overflowed since the last drain.
    fn take_interrupted(&mut self) -> Vec<EpisodeMetadata> {
        if self.interrupted_dropped > 0 {
            tracing::warn!(
                dropped = self.interrupted_dropped,
                cap = MAX_INTERRUPTED_EPISODES,
                "interrupted-episode buffer overflowed; oldest interrupted episodes were \
                 dropped before they could be delivered (a client looping Reset without Step \
                 grows this buffer); their accounting is lost"
            );
            self.interrupted_dropped = 0;
        }
        std::mem::take(&mut self.interrupted)
    }

    /// Start a new episode for the given environment index.
    ///
    /// `seed` is `None` when the environment was reset without an explicit seed
    /// (it seeded itself from entropy), so the episode metadata honestly records
    /// the absence of a seed instead of fabricating one.
    ///
    /// Returns the episode ID.
    pub fn start_episode(&mut self, env_index: i32, seed: Option<i64>) -> String {
        let episode = Episode::new(env_index, seed);
        let episode_id = episode.id.clone();

        // A reset can legitimately interrupt an in-flight episode (manual
        // vector reset, runtime reset racing a lane autoreset). Complete the
        // replaced episode as truncated and buffer it so the next drain point
        // (step completed_episodes / close final_episodes) surfaces it instead
        // of silently dropping its accounting.
        if let Some(old_episode) = self.active.insert(env_index, episode) {
            tracing::debug!(
                "Episode {} for env {} interrupted by a new episode; completing as truncated",
                old_episode.id,
                env_index
            );
            self.push_interrupted(old_episode.complete(false, true, None));
        }

        episode_id
    }

    /// Drain episodes that were interrupted by a replacing reset since the
    /// last drain. Callers fold these into the completed-episode stream so
    /// interrupted episodes surface exactly once.
    pub fn drain_interrupted(&mut self) -> Vec<EpisodeMetadata> {
        self.take_interrupted()
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
        final_info: Option<MetaMap>,
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

        let mut completed = self.take_interrupted();
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
        let ep_id = tracker.start_episode(0, Some(42));
        assert_eq!(active_count(&tracker), 1);

        // Record steps
        tracker.record_step(0, 1.0);
        tracker.record_step(0, 2.5);

        // Complete episode
        let metadata = tracker.complete_episode(0, true, false, None).unwrap();
        assert_eq!(metadata.episode_id, ep_id);
        assert_eq!(metadata.seed, Some(42));
        assert_eq!(metadata.env_index, 0);
        assert_eq!(metadata.step_count, 2);
        assert_eq!(metadata.cumulative_reward, 3.5);
        assert!(metadata.terminated);
        assert!(!metadata.truncated);

        assert_eq!(active_count(&tracker), 0);
    }

    #[test]
    fn interrupted_episode_is_completed_as_truncated_and_drained_once() {
        let mut tracker = EpisodeTracker::new();

        let first = tracker.start_episode(0, Some(7));
        tracker.record_step(0, 1.5);
        tracker.record_step(0, 2.5);

        // A replacing reset interrupts the in-flight episode: its accounting
        // must surface as a truncated completion instead of being dropped.
        let second = tracker.start_episode(0, None);
        assert_ne!(first, second);
        assert_eq!(active_count(&tracker), 1);

        let interrupted = tracker.drain_interrupted();
        assert_eq!(interrupted.len(), 1);
        assert_eq!(interrupted[0].episode_id, first);
        assert_eq!(interrupted[0].step_count, 2);
        assert_eq!(interrupted[0].cumulative_reward, 4.0);
        assert!(!interrupted[0].terminated);
        assert!(interrupted[0].truncated);

        // Exactly once: a second drain is empty.
        assert!(tracker.drain_interrupted().is_empty());
    }

    #[test]
    fn complete_all_includes_undrained_interrupted_episodes() {
        let mut tracker = EpisodeTracker::new();

        let first = tracker.start_episode(0, Some(1));
        tracker.record_step(0, 1.0);
        let second = tracker.start_episode(0, None);

        let mut all = tracker.complete_all("client close");
        all.sort_by(|a, b| a.episode_id.cmp(&b.episode_id));
        let mut expected = vec![first, second];
        expected.sort();
        let mut got: Vec<String> = all.iter().map(|m| m.episode_id.clone()).collect();
        got.sort();
        assert_eq!(got, expected);
        assert!(tracker.drain_interrupted().is_empty());
    }

    #[test]
    fn interrupted_buffer_is_bounded_under_repeated_reset_without_step() {
        let mut tracker = EpisodeTracker::new();

        let resets = MAX_INTERRUPTED_EPISODES * 5;
        for _ in 0..resets {
            tracker.start_episode(0, None);
            // The buffer must never grow past the cap, regardless of reset count.
            assert!(
                tracker.interrupted.len() <= MAX_INTERRUPTED_EPISODES,
                "interrupted buffer exceeded its cap"
            );
        }

        // It sits exactly at the cap (the very first reset created the active
        // episode without interrupting anything; every subsequent reset added an
        // interrupted entry, then eviction held it at the cap).
        assert_eq!(tracker.interrupted.len(), MAX_INTERRUPTED_EPISODES);
        // The overflow was counted so the next drain can warn.
        assert!(tracker.interrupted_dropped > 0);

        // Draining empties the buffer and clears the dropped counter (the warn
        // log fires here), and yields at most the cap many episodes — never the
        // whole unbounded backlog at once.
        let drained = tracker.drain_interrupted();
        assert_eq!(drained.len(), MAX_INTERRUPTED_EPISODES);
        assert_eq!(tracker.interrupted_dropped, 0);
        assert!(tracker.drain_interrupted().is_empty());
    }

    #[test]
    fn test_vectorized_episodes() {
        let mut tracker = EpisodeTracker::new();

        // Start multiple episodes
        let ep0 = tracker.start_episode(0, Some(100));
        let ep1 = tracker.start_episode(1, Some(200));
        let _ep2 = tracker.start_episode(2, Some(300));
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

        tracker.start_episode(0, Some(1));
        tracker.start_episode(1, Some(2));
        tracker.start_episode(2, Some(3));

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

    #[test]
    fn unseeded_episode_leaves_seed_unset_not_fabricated_zero() {
        let mut tracker = EpisodeTracker::new();

        // Reset without an explicit seed (the env seeded itself from entropy).
        tracker.start_episode(0, None);
        let meta = tracker.complete_episode(0, true, false, None).unwrap();

        // The seed must be absent, never a fabricated 0 that downstream
        // consumers would mistake for a real seed.
        assert_eq!(meta.seed, None);

        // An explicit seed of 0 is still recorded faithfully as Some(0).
        tracker.start_episode(1, Some(0));
        let meta = tracker.complete_episode(1, true, false, None).unwrap();
        assert_eq!(meta.seed, Some(0));
    }
}
