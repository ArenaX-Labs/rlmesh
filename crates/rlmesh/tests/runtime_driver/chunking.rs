//! Action chunks: observation history, prefetch, and replay frames across
//! episode boundaries.

use super::*;

#[tokio::test]
async fn a_history_route_sees_every_env_step_exactly_once_in_order() {
    // The exactly-once invariant behind observation history: across an
    // episode, the union of every predict's history rows and its own
    // observation is the env-step sequence, in order, each step once, with no
    // gap within or between requests. A prefetch lead moves each re-plan
    // earlier (it fires with `lead` frames still buffered, from the latest
    // observation, which then rides as the request's own instead of a row),
    // so the rows per request shrink and the steps after the episode's last
    // predict have no request left to ride — but nothing is delivered twice.
    // Chunks of 4 over a 9-step episode (steps 0..=8 observed; the terminal
    // observation ends the episode without a predict):
    //   lead 0: predicts at 0, 4, 8 — every replayed step rides the next one.
    //   lead 1: the second predict fires while frame 4 is still buffered, from
    //           step 2 (step 1 rides); the third from step 6 (3, 4, 5 ride);
    //           the chunk from 6 covers 7, 8 and the episode ends.
    //   lead 2: the second fires from step 1 with the first chunk barely
    //           started (no row yet); its chunk plays 5..=8, so the third
    //           fires from step 5 (2, 3, 4 ride) and steps 6..=8 replay.
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
            replay_frames: 3, // chunks of 4
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

        // Reset observation carries byte 1 at step 0; step n carries byte n.
        let ledger = model.ledger.lock().unwrap().clone();
        assert_eq!(ledger, expected, "lead {lead}");
        // Each step once, consecutive within and across requests, from step 0:
        // the union is a prefix of the episode, and the whole of it without a
        // lead.
        let union = delivered_steps(&ledger);
        let last = *union.last().unwrap();
        assert_eq!(union, (0..=last).collect::<Vec<_>>(), "lead {lead}");
        assert_eq!(last, [8, 6, 5][lead as usize], "lead {lead}");
        assert_eq!(model.predicts.load(Ordering::SeqCst), 3, "lead {lead}");
    }
}

#[tokio::test]
async fn a_prefetch_across_a_reset_delivers_the_reset_observation_once() {
    // Driver-owned resets, chunks of 3, lead 1, two 3-step episodes. The
    // prefetch fired at each episode's tail is conditioned on step 1 and
    // lands stale, AFTER the next episode's reset observation, which was
    // buffered as a row while it was in flight. The re-arm must promote that
    // row to the fresh request's own — not send it as a row AND as the own.
    let env = TestEnv {
        terminal_after: 3,
        ..Default::default()
    };
    let model = TestModel {
        replay_frames: 2,
        wants_history: true,
        // Slower than the env: the stale result lands after the reset.
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

    // The group's step counter runs across episodes: the second episode's
    // reset observation is step 3. Its first predict is the re-arm from that
    // observation, carrying no row (the first episode's rows were dropped
    // with its reset; the reset observation itself is the own). Step 2 and
    // step 5 landed after their episode's last predict fired, so no request
    // was left to carry them.
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
    // NEXT_STEP autoreset, one lane, chunks of 3, lead 1, two 3-step episodes.
    // The prefetch fired from step 1 lands stale after the terminal
    // observation (step 3) was buffered as a row behind step 2. The re-arm
    // promotes step 3 to the own of the predict the ended episode still gets
    // (the autoreset step's action), and step 2 rides it as the row it is —
    // so the ended episode's every step reached the model exactly once,
    // under its own id. The roll observation (step 4) starts the next
    // episode's ledger.
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
    // The residual detector for chunked vector routes. `run_local` and the managed
    // runner refuse the pairing outright, but a host driving RuntimeDriver directly
    // can still reach it — and the loss is silent: the replay buffer is whole-batch,
    // so ONE lane's episode end throws away every lane's remaining frames. Warn
    // once per session, and only when frames were actually discarded.
    let logs = chunked_run_logs(vec![2, 3]).await;
    assert_eq!(
        logs.matches("chunk replay is whole-batch").count(),
        1,
        "the tripwire fires once per session, not once per episode: {logs}"
    );
}

#[tokio::test]
async fn a_single_lane_chunked_route_never_trips_the_wire() {
    // The supported shape: one lane, chunking on, episodes ending mid-chunk. No
    // lane can lose another lane's frames, so nothing is warned about.
    let logs = chunked_run_logs(vec![2]).await;
    assert!(
        !logs.contains("chunk replay is whole-batch"),
        "a single-lane chunked route is the supported shape: {logs}"
    );
}

#[tokio::test]
async fn prefetch_overlaps_predict_with_replay_and_keeps_the_ledger() {
    // Async-inference mode: with chunk size 3 and lead 1, the next chunk's
    // predict fires while the current chunk still has a replay frame left, so
    // the loop never stalls on inference — but every step/observation/action
    // hook still fires exactly as in the synchronous loop.
    let env = TestEnv {
        terminal_after: 6,
        ..Default::default()
    };
    let model = TestModel {
        replay_frames: 2,
        // Slower than an env step: without prefetch this stalls each refill.
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
    // Chunked cadence holds: the model plans once per chunk boundary. Prefetch
    // may run one extra speculative predict at the episode tail (fired before
    // the terminal step landed), never more.
    let predicts = model.predicts.load(Ordering::SeqCst);
    assert!(
        (2..=3).contains(&predicts),
        "expected 2 chunk predicts (+ at most 1 speculative tail), got {predicts}",
    );
}

#[tokio::test]
async fn prefetch_discards_the_stale_chunk_across_episode_boundaries() {
    // Two 3-step episodes with chunk size 3, lead 1: the prefetch fired near
    // each episode's tail is conditioned on a pre-terminal observation and must
    // be discarded — the new episode re-plans from its own reset observation,
    // and no replayed action from the old chunk leaks across the boundary.
    let env = TestEnv {
        terminal_after: 3,
        ..Default::default()
    };
    // The stale prediction must land AFTER the reset observation: that is the
    // ordering in which the driver has to re-plan from the saved observation
    // itself, or the group stalls with nothing in flight.
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
    // Each episode re-planned from its own first observation: the fresh predict
    // after each boundary saw a reset observation, not a stale step one. The
    // reset observation payload is the env's reset marker (see TestEnv::reset),
    // recorded as the first seen observation of each episode's first predict.
    let seen = model.seen_observations.lock().unwrap();
    assert!(
        seen.len() >= 2,
        "at least one fresh plan per episode, got {}",
        seen.len()
    );
}

#[tokio::test]
async fn a_leftover_chunk_frame_never_crosses_an_episode_boundary() {
    // One-step episodes with a chunk of 3: every predict leaves two buffered
    // frames behind when its episode ends. Each new episode must re-plan from
    // its own reset observation rather than replay them, so the model sees
    // exactly one predict per episode, each carrying the reset observation
    // (byte 1) and no step number (no history was negotiated).
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
    // Same invariant when the ENV owns the reset (NEXT_STEP): the driver never
    // issues a reset after the cold start, so the reset-path discard cannot
    // save it — the episode-completion path must drop the buffered frames.
    // Single lane, one-step episodes, chunk of 3.
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
