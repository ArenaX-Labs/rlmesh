use super::*;

#[tokio::test]
async fn next_step_vector_env_completes_lanes_independently_without_whole_vector_reset() {
    let env = VectorTestEnv::new(vec![2, 3, 2, 4]);
    let model = TestModel::default();
    let report = new_driver(
        vector_spec(4, 8),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    assert_eq!(
        env.reset_seeds
            .lock()
            .expect("reset seed recorder lock poisoned")
            .len(),
        1,
        "driver must reset only once (cold start), never on a lane completion"
    );
    assert!(
        report.total_episodes >= 8,
        "expected >= 8 episodes across lanes, got {}",
        report.total_episodes
    );
    assert_eq!(
        model.predicts.load(Ordering::SeqCst) as i64,
        report.total_steps
    );
}

#[tokio::test]
async fn next_step_roll_attributes_reset_infos_to_the_new_episode() {
    let env = VectorTestEnv::new(vec![1, 1]);
    let hooks = Arc::new(RecordingHooks::default());
    new_driver(vector_spec(2, 4), env, TestModel::default(), hooks.clone())
        .run()
        .await
        .unwrap();

    assert_eq!(
        *hooks.step_infos.lock().unwrap(),
        vec![
            Some(vector_infos("step", 2, true)),
            None,
            Some(vector_infos("step", 2, true)),
        ]
    );
    assert_eq!(
        *hooks.step_flags.lock().unwrap(),
        vec![
            (vec![true, true], vec![false, false], vec![false, false]),
            (vec![false, false], vec![false, false], vec![true, true]),
            (vec![true, true], vec![false, false], vec![false, false]),
        ]
    );
    let emitted = hooks.emitted_observations.lock().unwrap();
    assert_eq!(
        emitted.iter().map(|e| e.infos.clone()).collect::<Vec<_>>(),
        vec![None, None, Some(info_map("phase", "roll"))]
    );
    assert!(
        emitted[2]
            .episode_ids
            .iter()
            .all(|id| !emitted[1].episode_ids.contains(id)),
        "post-roll observation carries the new episode ids"
    );
}

#[tokio::test]
async fn next_step_roll_is_marked_per_lane() {
    let env = VectorTestEnv::new(vec![1, 3]);
    let hooks = Arc::new(RecordingHooks::default());
    new_driver(vector_spec(2, 4), env, TestModel::default(), hooks.clone())
        .run()
        .await
        .unwrap();

    let flags = hooks.step_flags.lock().unwrap();
    assert!(
        flags.iter().any(|(_, _, roll)| roll == &[true, false]),
        "a lane rolls on its own: {flags:?}"
    );
    for (terminated, truncated, roll) in flags.iter() {
        for lane in 0..2 {
            assert!(
                !(roll[lane] && (terminated[lane] || truncated[lane])),
                "a roll ends nothing: {flags:?}"
            );
        }
    }
}

#[tokio::test]
async fn the_budget_bounds_episode_starts_on_a_lockstep_vector() {
    let env = VectorTestEnv::new(vec![2, 2]);
    let hooks = Arc::new(RecordingHooks::default());
    let report = new_driver(vector_spec(2, 3), env, TestModel::default(), hooks.clone())
        .run()
        .await
        .unwrap();

    assert_eq!(report.episodes.len(), 3);
    assert_eq!(report.total_episodes, 3);
    let mut indices: Vec<i64> = report.episodes.iter().map(|e| e.episode_index).collect();
    indices.sort_unstable();
    assert_eq!(indices, [1, 2, 3]);
    assert_eq!(hooks.started_seeds.lock().unwrap().len(), 3);
    assert_eq!(hooks.completed_seeds.lock().unwrap().len(), 3);
    for episode in &report.episodes {
        assert_eq!(episode.step_count, 2);
        assert!(episode.step_ms.is_some_and(|ms| ms > 0.0), "{episode:?}");
        assert!(
            episode.predict_ms.is_some_and(|ms| ms >= 0.0),
            "{episode:?}"
        );
    }
}

#[tokio::test]
async fn the_budget_is_exact_on_driver_reset_lanes() {
    let env = VectorTestEnv::new(vec![1, 3]);
    let hooks = Arc::new(RecordingHooks::default());
    let report = new_driver(
        lane_spec(2, 3, vec![]),
        env,
        TestModel::default(),
        hooks.clone(),
    )
    .run()
    .await
    .unwrap();
    assert_eq!(report.episodes.len(), 3);
    assert_eq!(hooks.started_seeds.lock().unwrap().len(), 3);
    assert_eq!(hooks.completed_seeds.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn chunking_does_not_break_autoreset_eviction() {
    let env = VectorTestEnv::new(vec![2, 3]);
    let model = TestModel {
        replay_frames: 4,
        ..Default::default()
    };
    let report = new_driver(
        vector_spec(2, 4),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    assert!(report.total_episodes >= 4);
    assert!(
        (model.predicts.load(Ordering::SeqCst) as i64) < report.total_steps,
        "chunk replay must skip some predict calls"
    );
    let evicted: Vec<String> = model
        .reset_adapters
        .lock()
        .expect("reset_adapter recorder lock poisoned")
        .iter()
        .flatten()
        .cloned()
        .collect();
    let predicted: std::collections::HashSet<String> = model
        .lifecycle
        .lock()
        .expect("lifecycle lock poisoned")
        .iter()
        .filter(|(kind, _)| *kind == "predict")
        .flat_map(|(_, ids)| ids.iter().cloned())
        .collect();
    let distinct: std::collections::HashSet<String> = evicted.iter().cloned().collect();
    assert_eq!(
        evicted.len(),
        distinct.len(),
        "an episode evicted twice: {evicted:?}"
    );
    assert_eq!(distinct, predicted, "one eviction per predicted episode");
    assert!(evicted.len() as i64 >= report.total_episodes);
    for id in &evicted {
        assert_eq!(id.len(), 36, "evicted a UUID id, got {id:?}");
        assert_eq!(id.as_bytes()[14], b'7', "UUIDv7 version nibble: {id:?}");
    }
}

#[tokio::test]
async fn next_step_autoreset_never_predicts_on_an_evicted_episode() {
    let env = VectorTestEnv::new(vec![1, 3]);
    let model = TestModel::default();
    let report = new_driver(
        vector_spec(2, 4),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();
    assert!(report.total_episodes >= 4);

    let lifecycle = model
        .lifecycle
        .lock()
        .expect("lifecycle lock poisoned")
        .clone();
    let mut evicted: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (kind, ids) in &lifecycle {
        match *kind {
            "evict" => evicted.extend(ids.iter().cloned()),
            _ => {
                for id in ids {
                    assert!(
                        !evicted.contains(id),
                        "predict on {id} after its eviction; lifecycle: {lifecycle:?}"
                    );
                }
            }
        }
    }
    let predicted: std::collections::HashSet<String> = lifecycle
        .iter()
        .filter(|(kind, _)| *kind == "predict")
        .flat_map(|(_, ids)| ids.iter().cloned())
        .collect();
    assert_eq!(evicted, predicted);
    assert!(evicted.len() as i64 >= report.total_episodes);
}

#[tokio::test]
async fn autoreset_roll_keeps_a_sibling_lane_evictable_at_teardown() {
    let env = VectorTestEnv::new(vec![2, 3, 7]);
    let model = TestModel::default();
    let report = new_driver(
        vector_spec(3, 2),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();
    assert_eq!(report.total_episodes, 2);

    let lifecycle = model
        .lifecycle
        .lock()
        .expect("lifecycle lock poisoned")
        .clone();
    let predicted: std::collections::HashSet<String> = lifecycle
        .iter()
        .filter(|(kind, _)| *kind == "predict")
        .flat_map(|(_, ids)| ids.iter().cloned())
        .filter(|id| !id.is_empty())
        .collect();
    let evicted: Vec<String> = model
        .reset_adapters
        .lock()
        .expect("reset_adapter recorder lock poisoned")
        .iter()
        .flatten()
        .cloned()
        .collect();
    let distinct: std::collections::HashSet<String> = evicted.iter().cloned().collect();
    assert_eq!(
        evicted.len(),
        distinct.len(),
        "an episode evicted twice: {evicted:?}"
    );
    assert_eq!(
        distinct, predicted,
        "every predicted episode ends on the model exactly once; lifecycle: {lifecycle:?}"
    );
}

#[tokio::test]
async fn next_step_episode_completed_is_the_last_hook_event_under_its_id() {
    let env = VectorTestEnv::new(vec![1, 3]);
    let hooks = Arc::new(RecordingHooks::default());
    let report = new_driver(vector_spec(2, 4), env, TestModel::default(), hooks.clone())
        .run()
        .await
        .unwrap();
    assert_eq!(report.total_episodes, 4);

    let completed = hooks.completed_ids();
    let unique: std::collections::HashSet<&String> = completed.iter().collect();
    assert_eq!(
        completed.len(),
        4,
        "one completion per episode: {completed:?}"
    );
    assert_eq!(unique.len(), 4, "no id completed twice: {completed:?}");
    assert_eq!(report.episodes.len(), 4, "one summary per episode");
    let events = hooks.events.lock().unwrap().clone();
    for id in &completed {
        let last = events
            .iter()
            .rposition(|(_, ids)| ids.contains(id))
            .expect("completed id was seen");
        assert_eq!(
            events[last].0, "completed",
            "episode_completed must be the last event under {id}; events: {events:?}"
        );
    }
    assert_eq!(
        hooks.events_for(&completed[0]),
        [
            "started",
            "observation",
            "step",
            "observation",
            "step",
            "completed"
        ],
        "events: {events:?}"
    );
    assert_eq!(
        hooks.events_for(&completed[3]),
        ["started", "observation", "step", "completed"],
        "events: {events:?}"
    );
    assert_eq!(events.last().map(|(kind, _)| *kind), Some("completed"));
}
