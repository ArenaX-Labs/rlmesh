//! Episode tracking for evaluation and replay.
#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use rlmesh_proto::env::v1::EpisodeMetadata;

/// Configuration for episode tracking.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Whether to log actions for replay (default: true).
    pub log_actions: bool,
    /// Whether to record frames for debugging (default: false).
    pub record_frames: bool,
    /// Maximum frames per episode (ring buffer size, default: 1000).
    pub max_frames_per_episode: usize,
    /// Maximum completed episodes to retain (default: 100).
    pub max_completed: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            log_actions: true,
            record_frames: false,
            max_frames_per_episode: 1000,
            max_completed: 100,
        }
    }
}

/// A bounded circular buffer for frame storage.
struct FrameRingBuffer {
    frames: Vec<Vec<u8>>,
    capacity: usize,
    next_index: usize,
    full: bool,
}

#[allow(dead_code)] // Methods used for future replay export
impl FrameRingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            frames: Vec::with_capacity(capacity),
            capacity,
            next_index: 0,
            full: false,
        }
    }

    fn push(&mut self, frame: Vec<u8>) {
        if self.frames.len() < self.capacity {
            // Still have room, just push
            self.frames.push(frame);
        } else {
            // Buffer is at capacity, start overwriting
            self.full = true;
            self.frames[self.next_index] = frame;
        }

        self.next_index = (self.next_index + 1) % self.capacity;
    }

    fn len(&self) -> usize {
        self.frames.len()
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &Vec<u8>> + '_> {
        if self.full {
            // Return frames in chronological order (starting from next_index)
            Box::new(
                self.frames[self.next_index..]
                    .iter()
                    .chain(self.frames[..self.next_index].iter()),
            )
        } else {
            // Not full yet, return in insertion order
            Box::new(self.frames.iter())
        }
    }
}

/// Single episode state (internal).
struct Episode {
    id: String,
    seed: i64,
    env_index: i32,
    step_count: i64,
    cumulative_reward: f64,
    start_time: Instant,
    start_timestamp_ns: i64,
    action_log: Option<Vec<Vec<u8>>>,
    frame_log: Option<FrameRingBuffer>,
}

impl Episode {
    fn new(env_index: i32, seed: i64, config: &TrackerConfig) -> Self {
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
            action_log: if config.log_actions {
                Some(Vec::new())
            } else {
                None
            },
            frame_log: if config.record_frames {
                Some(FrameRingBuffer::new(config.max_frames_per_episode))
            } else {
                None
            },
        }
    }

    fn record_step(&mut self, reward: f64, action: Option<&[u8]>) {
        self.step_count += 1;
        self.cumulative_reward += reward;

        if let Some(action_log) = &mut self.action_log
            && let Some(action_bytes) = action
        {
            action_log.push(action_bytes.to_vec());
        }
    }

    fn record_frame(&mut self, frame: Vec<u8>) {
        if let Some(frame_log) = &mut self.frame_log {
            frame_log.push(frame);
        }
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
    config: TrackerConfig,
    active: HashMap<i32, Episode>,
    completed: Vec<EpisodeMetadata>,
}

impl EpisodeTracker {
    /// Create a new episode tracker with default configuration.
    pub fn new() -> Self {
        Self::with_config(TrackerConfig::default())
    }

    /// Create a new episode tracker with custom configuration.
    pub fn with_config(config: TrackerConfig) -> Self {
        Self {
            config,
            active: HashMap::new(),
            completed: Vec::new(),
        }
    }

    /// Start a new episode for the given environment index.
    ///
    /// Returns the episode ID.
    pub fn start_episode(&mut self, env_index: i32, seed: i64) -> String {
        let episode = Episode::new(env_index, seed, &self.config);
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
    ///
    /// # Panics
    /// Panics if no episode is active for the given environment index.
    pub fn record_step(&mut self, env_index: i32, reward: f64, action: Option<&[u8]>) {
        if let Some(episode) = self.active.get_mut(&env_index) {
            episode.record_step(reward, action);
        } else {
            tracing::warn!(
                "Attempted to record step for env {} with no active episode",
                env_index
            );
        }
    }

    /// Record a frame for the given environment (for debugging).
    pub fn record_frame(&mut self, env_index: i32, frame: Vec<u8>) {
        if let Some(episode) = self.active.get_mut(&env_index) {
            episode.record_frame(frame);
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
        let metadata = episode.complete(terminated, truncated, final_info);

        // Add to completed buffer with size limit
        self.completed.push(metadata.clone());
        if self.completed.len() > self.config.max_completed {
            let removed = self.completed.remove(0);
            tracing::debug!(
                "Dropped oldest completed episode {} (buffer full)",
                removed.episode_id
            );
        }

        Some(metadata)
    }

    /// Drain all completed episodes.
    ///
    /// Returns all episodes that have been completed since the last drain.
    pub fn drain_completed(&mut self) -> Vec<EpisodeMetadata> {
        std::mem::take(&mut self.completed)
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

    /// Get the number of active episodes.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Get the number of completed episodes (not yet drained).
    pub fn completed_count(&self) -> usize {
        self.completed.len()
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

    #[test]
    fn test_episode_lifecycle() {
        let mut tracker = EpisodeTracker::new();

        // Start episode
        let ep_id = tracker.start_episode(0, 42);
        assert_eq!(tracker.active_count(), 1);
        assert_eq!(tracker.completed_count(), 0);

        // Record steps
        tracker.record_step(0, 1.0, Some(&[0, 1, 2]));
        tracker.record_step(0, 2.5, Some(&[3, 4, 5]));

        // Complete episode
        let metadata = tracker.complete_episode(0, true, false, None).unwrap();
        assert_eq!(metadata.episode_id, ep_id);
        assert_eq!(metadata.seed, 42);
        assert_eq!(metadata.env_index, 0);
        assert_eq!(metadata.step_count, 2);
        assert_eq!(metadata.cumulative_reward, 3.5);
        assert!(metadata.terminated);
        assert!(!metadata.truncated);

        assert_eq!(tracker.active_count(), 0);
        assert_eq!(tracker.completed_count(), 1);

        // Drain completed
        let completed = tracker.drain_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].episode_id, ep_id);
        assert_eq!(tracker.completed_count(), 0);
    }

    #[test]
    fn test_vectorized_episodes() {
        let mut tracker = EpisodeTracker::new();

        // Start multiple episodes
        let ep0 = tracker.start_episode(0, 100);
        let ep1 = tracker.start_episode(1, 200);
        let _ep2 = tracker.start_episode(2, 300);
        assert_eq!(tracker.active_count(), 3);

        // Record steps for each
        tracker.record_step(0, 1.0, None);
        tracker.record_step(1, 2.0, None);
        tracker.record_step(2, 3.0, None);

        // Complete env 1
        let meta1 = tracker.complete_episode(1, true, false, None).unwrap();
        assert_eq!(meta1.episode_id, ep1);
        assert_eq!(tracker.active_count(), 2);

        // Complete env 0
        let meta0 = tracker.complete_episode(0, false, true, None).unwrap();
        assert_eq!(meta0.episode_id, ep0);
        assert!(!meta0.terminated);
        assert!(meta0.truncated);
        assert_eq!(tracker.active_count(), 1);

        // Complete env 2
        tracker.complete_episode(2, true, false, None);
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(tracker.completed_count(), 3);
    }

    #[test]
    fn test_frame_ring_buffer() {
        let mut buffer = FrameRingBuffer::new(3);
        assert_eq!(buffer.len(), 0);

        // Add frames without overflow
        buffer.push(vec![1, 2, 3]);
        buffer.push(vec![4, 5, 6]);
        assert_eq!(buffer.len(), 2);

        // Fill buffer
        buffer.push(vec![7, 8, 9]);
        assert_eq!(buffer.len(), 3);
        assert!(!buffer.full);

        // Overflow - should replace oldest
        buffer.push(vec![10, 11, 12]);
        assert_eq!(buffer.len(), 3);
        assert!(buffer.full);

        // Verify chronological order (oldest first)
        let frames: Vec<_> = buffer.iter().cloned().collect();
        assert_eq!(frames[0], vec![4, 5, 6]);
        assert_eq!(frames[1], vec![7, 8, 9]);
        assert_eq!(frames[2], vec![10, 11, 12]);
    }

    #[test]
    fn test_complete_all() {
        let mut tracker = EpisodeTracker::new();

        tracker.start_episode(0, 1);
        tracker.start_episode(1, 2);
        tracker.start_episode(2, 3);

        tracker.record_step(0, 1.0, None);
        tracker.record_step(1, 2.0, None);

        let interrupted = tracker.complete_all("test cancellation");
        assert_eq!(interrupted.len(), 3);
        assert_eq!(tracker.active_count(), 0);

        // All should be truncated (not terminated)
        for meta in interrupted {
            assert!(!meta.terminated);
            assert!(meta.truncated);
        }
    }

    #[test]
    fn test_action_logging() {
        let config = TrackerConfig {
            log_actions: true,
            ..Default::default()
        };
        let mut tracker = EpisodeTracker::with_config(config);

        tracker.start_episode(0, 42);
        tracker.record_step(0, 1.0, Some(&[1, 2, 3]));
        tracker.record_step(0, 2.0, Some(&[4, 5, 6]));

        // Actions are logged internally but not exposed in metadata
        // This just verifies no panics occur
        let metadata = tracker.complete_episode(0, true, false, None).unwrap();
        assert_eq!(metadata.step_count, 2);
    }

    #[test]
    fn test_frame_recording() {
        let config = TrackerConfig {
            record_frames: true,
            max_frames_per_episode: 5,
            ..Default::default()
        };
        let mut tracker = EpisodeTracker::with_config(config);

        tracker.start_episode(0, 42);
        tracker.record_frame(0, vec![1, 2, 3]);
        tracker.record_frame(0, vec![4, 5, 6]);
        tracker.record_frame(0, vec![7, 8, 9]);

        // Complete and verify no panics
        let metadata = tracker.complete_episode(0, true, false, None).unwrap();
        assert_eq!(metadata.step_count, 0); // No steps recorded, only frames
    }

    #[test]
    fn test_max_completed_limit() {
        let config = TrackerConfig {
            max_completed: 3,
            ..Default::default()
        };
        let mut tracker = EpisodeTracker::with_config(config);

        // Complete 5 episodes
        for i in 0..5 {
            tracker.start_episode(0, i);
            tracker.complete_episode(0, true, false, None);
        }

        // Should only keep last 3
        assert_eq!(tracker.completed_count(), 3);

        let completed = tracker.drain_completed();
        assert_eq!(completed.len(), 3);
        assert_eq!(completed[0].seed, 2); // Episodes 2, 3, 4 retained
        assert_eq!(completed[1].seed, 3);
        assert_eq!(completed[2].seed, 4);
    }
}
