//! Episode bookkeeping: a stable, session-global record id per episode id.

use std::collections::BTreeMap;

/// One episode's session-global identity (`ep-NNNNNN` + index) and origin lane.
#[derive(Debug, Clone)]
pub(crate) struct EpisodeRecord {
    pub(crate) record_id: String,
    pub(crate) index: i64,
    pub(crate) env_index: i32,
    pub(crate) started_from_auto_reset: bool,
}

/// Assigns each distinct episode id a stable record id, session-global and
/// independent of which lane the episode ran in. A repeated id resolves to its
/// existing record (idempotent), so re-observing a slot never re-counts it.
#[derive(Debug, Default)]
pub(crate) struct EpisodeRecordRegistry {
    next_index: i64,
    by_episode_id: BTreeMap<String, EpisodeRecord>,
}

impl EpisodeRecordRegistry {
    pub(crate) fn ensure_for_slots(
        &mut self,
        episode_ids: &[String],
        started_from_auto_reset: bool,
    ) -> (Vec<String>, Vec<(String, EpisodeRecord)>) {
        let mut record_ids = Vec::with_capacity(episode_ids.len());
        let mut started = Vec::new();
        for (env_index, episode_id) in episode_ids.iter().enumerate() {
            if episode_id.is_empty() {
                record_ids.push(String::new());
                continue;
            }
            if let Some(existing) = self.by_episode_id.get(episode_id) {
                record_ids.push(existing.record_id.clone());
                continue;
            }

            self.next_index += 1;
            let record = EpisodeRecord {
                record_id: format!("ep-{:06}", self.next_index),
                index: self.next_index,
                env_index: env_index as i32,
                started_from_auto_reset,
            };
            record_ids.push(record.record_id.clone());
            self.by_episode_id
                .insert(episode_id.clone(), record.clone());
            started.push((episode_id.clone(), record));
        }
        (record_ids, started)
    }

    pub(crate) fn record_for(&self, episode_id: &str) -> Option<&EpisodeRecord> {
        self.by_episode_id.get(episode_id)
    }
}

#[cfg(test)]
mod tests {
    use super::EpisodeRecordRegistry;

    #[test]
    fn episode_record_ids_are_global_across_vector_slots() {
        let mut registry = EpisodeRecordRegistry::default();
        let (initial_ids, started) =
            registry.ensure_for_slots(&["runtime-a".to_string(), "runtime-b".to_string()], false);
        assert_eq!(initial_ids, ["ep-000001", "ep-000002"]);
        assert_eq!(started.len(), 2);
        assert!(!started[0].1.started_from_auto_reset);

        let (next_ids, started) =
            registry.ensure_for_slots(&["runtime-a".to_string(), "runtime-c".to_string()], true);
        assert_eq!(next_ids, ["ep-000001", "ep-000003"]);
        assert_eq!(started.len(), 1);
        assert_eq!(started[0].0, "runtime-c");
        assert_eq!(started[0].1.env_index, 1);
        assert!(started[0].1.started_from_auto_reset);
    }
}
