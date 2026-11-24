use super::*;

#[tokio::test]
async fn a_history_route_sees_every_env_step_exactly_once_in_order() {
    let expected: [Vec<LedgerEntry>; 3] = [
        vec![
            (vec![], (0, 1)),
            (vec![(1, 1), (2, 2), (3, 3)], (4, 4)),
            (vec![(5, 5), (6, 6), (7, 7)], (8, 8)),
        ],
        vec![
            (vec![], (0, 1)),
            (vec![(1, 1)], (2, 2)),
            (vec![(3, 3), (4, 4), (5, 5)], (6, 6)),
        ],
        vec![
            (vec![], (0, 1)),
            (vec![], (1, 1)),
            (vec![(2, 2), (3, 3), (4, 4)], (5, 5)),
        ],
    ];
    for (lead, expected) in (0u32..).zip(expected) {
        let env = TestEnv {
            terminal_after: 9,
            ..Default::default()
        };
        let model = TestModel {
            replay_frames: 3,
            wants_history: true,
            ..Default::default()
        };
        let report = new_driver(
            one_episode_spec(),
            env.clone(),
            model.clone(),
            Arc::new(RecordingHooks::default()),
        )
        .with_prefetch(lead)
        .run()
        .await
        .unwrap();
        assert_eq!(report.total_episodes, 1);
        assert_eq!(report.total_steps, 9);

        let ledger = model.ledger.lock().unwrap().clone();
        assert_eq!(ledger, expected, "lead {lead}");
        let union = delivered_steps(&ledger);
        let last = *union.last().unwrap();
        assert_eq!(union, (0..=last).collect::<Vec<_>>(), "lead {lead}");
        assert_eq!(last, [8, 6, 5][lead as usize], "lead {lead}");
        assert_eq!(model.predicts.load(Ordering::SeqCst), 3, "lead {lead}");
    }
}

#[tokio::test]
async fn a_prefetch_across_a_reset_delivers_the_reset_observation_once() {
    let env = TestEnv {
        terminal_after: 3,
        ..Default::default()
    };
    let model = TestModel {
        replay_frames: 2,
        wants_history: true,
        predict_delay: Some(Duration::from_millis(20)),
        ..Default::default()
    };
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(2);
    let report = new_driver(
        spec,
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .with_prefetch(1)
    .run()
    .await
    .unwrap();
    assert_eq!(report.total_episodes, 2);
    assert_eq!(report.total_steps, 6);

    let episodes = ledger_per_episode(&model);
    assert_eq!(episodes.len(), 2);
    assert_ne!(episodes[0].0, episodes[1].0);
    assert_eq!(episodes[0].1, vec![(vec![], (0, 1)), (vec![], (1, 1))]);
    assert_eq!(episodes[1].1, vec![(vec![], (3, 1)), (vec![], (4, 1))]);
    assert_eq!(delivered_steps(&episodes[0].1), vec![0, 1]);
    assert_eq!(delivered_steps(&episodes[1].1), vec![3, 4]);
}

#[tokio::test]
async fn a_prefetch_across_a_next_step_roll_delivers_the_terminal_step_once() {
    let env = VectorTestEnv::new(vec![3]);
    let model = TestModel {
        replay_frames: 2,
        wants_history: true,
        predict_delay: Some(Duration::from_millis(20)),
        ..Default::default()
    };
    let report = new_driver(
        vector_spec(1, 2),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .with_prefetch(1)
    .run()
    .await
    .unwrap();
    assert_eq!(report.total_episodes, 2);

    let episodes = ledger_per_episode(&model);
    assert_eq!(episodes.len(), 2);
    assert_ne!(episodes[0].0, episodes[1].0);
    assert_eq!(
        episodes[0].1,
        vec![(vec![], (0, 0)), (vec![], (1, 0)), (vec![(2, 0)], (3, 0))]
    );
    assert_eq!(delivered_steps(&episodes[0].1), vec![0, 1, 2, 3]);
    assert_eq!(episodes[1].1, vec![(vec![], (4, 0))]);
}

#[tokio::test]
async fn a_route_without_history_carries_no_rows_and_no_step() {
    let model = TestModel {
        replay_frames: 3,
        ..Default::default()
    };
    new_driver(
        one_episode_spec(),
        TestEnv {
            terminal_after: 9,
            ..Default::default()
        },
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();
    let ledger = model.ledger.lock().unwrap().clone();
    assert!(!ledger.is_empty());
    for (rows, own) in ledger {
        assert!(rows.is_empty());
        assert_eq!(own.0, -1, "no step stamped without history");
    }
}

#[tokio::test]
async fn a_vector_route_losing_a_chunk_mid_episode_trips_the_wire_once() {
    let logs = chunked_run_logs(vec![2, 3]).await;
    assert_eq!(
        logs.matches("chunk replay is whole-batch").count(),
        1,
        "the tripwire fires once per session, not once per episode: {logs}"
    );
}

#[tokio::test]
async fn a_single_lane_chunked_route_never_trips_the_wire() {
    let logs = chunked_run_logs(vec![2]).await;
    assert!(
        !logs.contains("chunk replay is whole-batch"),
        "a single-lane chunked route is the supported shape: {logs}"
    );
}

#[tokio::test]
async fn prefetch_overlaps_predict_with_replay_and_keeps_the_ledger() {
    let env = TestEnv {
        terminal_after: 6,
        ..Default::default()
    };
    let model = TestModel {
        replay_frames: 2,
        predict_delay: Some(Duration::from_millis(20)),
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let spec = one_episode_spec();

    let report = new_driver(spec, env.clone(), model.clone(), hooks.clone())
        .with_prefetch(1)
        .run()
        .await
        .unwrap();

    assert_eq!(report.total_steps, 6, "env advanced a full 6-step episode");
    assert_eq!(report.total_episodes, 1);
    assert_eq!(
        env.step_count.load(Ordering::SeqCst),
        6,
        "env stepped every step, replay or not",
    );
    assert_eq!(
        hooks.emitted_observations.lock().unwrap().len(),
        6,
        "observation emitted every step, including replay steps",
    );
    assert_eq!(hooks.actions.load(Ordering::SeqCst), 6);
    let predicts = model.predicts.load(Ordering::SeqCst);
    assert!(
        (2..=3).contains(&predicts),
        "expected 2 chunk predicts (+ at most 1 speculative tail), got {predicts}",
    );
}

#[tokio::test]
async fn prefetch_discards_the_stale_chunk_across_episode_boundaries() {
    let env = TestEnv {
        terminal_after: 3,
        ..Default::default()
    };
    let model = TestModel {
        replay_frames: 2,
        predict_delay: Some(Duration::from_millis(20)),
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(2);

    let report = new_driver(spec, env.clone(), model.clone(), hooks.clone())
        .with_prefetch(1)
        .run()
        .await
        .unwrap();

    assert_eq!(report.total_episodes, 2);
    assert_eq!(report.total_steps, 6);
    let seen = model.seen_observations.lock().unwrap();
    assert!(
        seen.len() >= 2,
        "at least one fresh plan per episode, got {}",
        seen.len()
    );
}

#[tokio::test]
async fn a_leftover_chunk_frame_never_crosses_an_episode_boundary() {
    let env = TestEnv {
        terminal_after: 1,
        ..Default::default()
    };
    let model = TestModel {
        replay_frames: 2,
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(3);

    let report = new_driver(spec, env, model.clone(), hooks)
        .run()
        .await
        .unwrap();

    assert_eq!(report.total_episodes, 3);
    assert_eq!(report.total_steps, 3);
    let ledger = model.ledger.lock().expect("ledger poisoned").clone();
    assert_eq!(ledger, vec![(vec![], (-1, 1)); 3]);
}

#[tokio::test]
async fn a_leftover_chunk_frame_never_crosses_a_next_step_autoreset() {
    let env = VectorTestEnv::new(vec![1]);
    let model = TestModel {
        replay_frames: 2,
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());

    let report = new_driver(vector_spec(1, 3), env, model.clone(), hooks)
        .run()
        .await
        .unwrap();

    assert_eq!(report.total_episodes, 3);
    let predicts = model.predicts.load(Ordering::SeqCst);
    assert_eq!(predicts, 3, "one fresh plan per episode, got {predicts}");
}
