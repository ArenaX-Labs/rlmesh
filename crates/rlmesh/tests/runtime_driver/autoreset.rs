//! Per-lane autoreset: NEXT_STEP rolls, episode budgets, and evictions
//! around autoreset.

use super::*;

#[tokio::test]
async fn next_step_vector_env_completes_lanes_independently_without_whole_vector_reset() {
    // The headline regression. With num_envs=4, NEXT_STEP, and variable-length
    // episodes, the driver must never issue a reset after the cold start; the env
    // auto-resets each lane itself. Previously any single lane completing fired a
    // whole-vector reset that cut every other lane's episode short.
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

    // Exactly one env.reset over the whole run: the initial cold start.
    assert_eq!(
        env.reset_seeds
            .lock()
            .expect("reset seed recorder lock poisoned")
            .len(),
        1,
        "driver must reset only once (cold start), never on a lane completion"
    );
    // Lanes completed episodes independently until the episode budget.
    assert!(
        report.total_episodes >= 8,
        "expected >= 8 episodes across lanes, got {}",
        report.total_episodes
    );
    // One predict per driver step: no stalled lanes.
    assert_eq!(
        model.predicts.load(Ordering::SeqCst) as i64,
        report.total_steps
    );
}

#[tokio::test]
async fn next_step_roll_attributes_reset_infos_to_the_new_episode() {
    // Both lanes complete at step 1 and roll at step 2. The step-2 response is
    // the new episodes' reset observation + infos: those infos belong on the
    // post-roll observation event (the new episodes' first observation), not on
    // the step event still attributed to the old episodes.
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
    // The terminal flags ride the terminal step (1 and 3), not the deferred
    // completion; the roll response (2) is marked as such and ends nothing.
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
    // Lane 0 ends every step and rolls on the next; lane 1 runs three steps.
    // A response that rolls lane 0 is still a real step of lane 1's episode.
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
    // Two lanes of equal length under a budget of three: the second pair of
    // completions would overshoot by one. The budget bounds STARTS, so the
    // fourth lane-episode the env rolls into is surplus: stepped in lockstep,
    // never announced, scored, or counted -- exactly three episodes report.
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
    // Chunking × NEXT_STEP autoreset. Chunk replay skips most predict calls, but
    // env.step + completion detection + ResetAdapter eviction still run every
    // step — so each episode end is evicted exactly once, with a runtime-minted
    // UUIDv7 id, even while the model is mid-chunk and not being re-called.
    let env = VectorTestEnv::new(vec![2, 3]);
    let model = TestModel {
        replay_frames: 4, // chunk of 5; most steps replay without a predict
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
    // Predict was skipped on most steps (chunk replay), proving the eviction
    // below fired independently of the predict cadence.
    assert!(
        (model.predicts.load(Ordering::SeqCst) as i64) < report.total_steps,
        "chunk replay must skip some predict calls"
    );
    // Exactly one ResetAdapter eviction per completed episode, each a UUIDv7.
    let evicted: Vec<String> = model
        .reset_adapters
        .lock()
        .expect("reset_adapter recorder lock poisoned")
        .iter()
        .flatten()
        .cloned()
        .collect();
    // Every episode the model predicted on is evicted exactly once: each
    // completed episode at its end, and the lane the budget cut mid-episode at
    // teardown, so nothing is left behind in the model.
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
    // NEXT_STEP × lockstep vector: a lane that ends at t is predicted on once
    // more (the terminal observation feeds the autoreset step, whose action
    // the env discards) before its slot rolls at t+1. The model's lifecycle
    // for that id must still be predict* → evict, never evict → predict: an
    // eviction-then-predict would re-seed the ended episode's state and leak
    // the entry. Uneven lane lengths make the rolls land on different steps.
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
    // Every episode the model predicted on was evicted — the completed ones at
    // their end, the lane the budget cut mid-episode at teardown — so nothing
    // is left behind in the model.
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
    // NEXT_STEP × lockstep vector, lanes of 2, 3 and 7 steps, budget 2. The
    // lane-0 roll and the lane-1 completion that spends the budget land on the
    // same step, so lane 2's episode is live at teardown and its last predict
    // was before that roll. It must still be ended on the model: a roll must
    // not clear a sibling lane's model-side episode state.
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
    // NEXT_STEP × lockstep vector, lanes of 1 and 3 steps, budget 4. Lane 0
    // ends at steps 1, 3 and 5; lane 1 at step 3. The step-5 completion hits
    // the budget, so that episode's roll never lands. From a hook's side an
    // ended id is still observed once more (the terminal observation feeding
    // the autoreset step) and stepped once more (the autoreset step itself),
    // both under the old id; `episode_completed` follows those, as the last
    // event the hook sees under that id, mirroring the model's
    // predict* → evict lifecycle. The budget-ended episode completes with no
    // roll behind it. Every episode is completed exactly once and summarized.
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
    // A mid-run boundary (lane 0's first episode): its terminal step, the
    // post-end observation, the autoreset step, then the completion.
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
    // The budget-ended episode (lane 0's third, the last completion): the
    // group idles right after its terminal step, so nothing is observed
    // between that step and its completion, and it is the run's last event.
    assert_eq!(
        hooks.events_for(&completed[3]),
        ["started", "observation", "step", "completed"],
        "events: {events:?}"
    );
    assert_eq!(events.last().map(|(kind, _)| *kind), Some("completed"));
}
