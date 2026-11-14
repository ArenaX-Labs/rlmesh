//! Lane sessions: one driver per lane over a shared, lane-capable env.

use super::*;

// ---------------------------------------------------------------------------
// Lane sessions: one driver per lane over a shared, lane-capable env.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_abandoned_predict_at_route_end_sends_no_evict() {
    // One lane, a two-frame chunk with lead 1, so the episode's last step runs
    // from replay while the prefetch is in flight: that prefetch is gated for
    // good, and the ended episode's eviction is held behind it. Cancelling
    // leaves the driver draining a predict that never lands, and the endpoint
    // that just missed the close deadline must get no evict -- the RPC carries
    // no deadline of its own, so sending it would hang shutdown forever.
    let (_open, gate) = oneshot::channel();
    let model = TestModel {
        replay_frames: 1,
        gate: Some(gate.shared()),
        gate_at: 1,
        reset_adapter_hangs: true,
        ..TestModel::default()
    };
    let env = TestEnv {
        terminal_after: 2,
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = one_episode_spec();
    spec.limits.service_close_timeout = Duration::from_millis(50);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let run = new_driver(spec, env, model.clone(), hooks.clone())
        .with_prefetch(1)
        .run_with_cancellation(cancellation.clone());
    let cancel = async {
        tokio::time::timeout(Duration::from_secs(5), async {
            while hooks
                .completed_seeds
                .lock()
                .expect("completed seed recorder lock poisoned")
                .is_empty()
            {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("the episode never completed");
        cancellation.cancel();
    };
    let (result, ()) =
        tokio::time::timeout(Duration::from_secs(5), async { tokio::join!(run, cancel) })
            .await
            .expect("driver hung in shutdown on an evict after an abandoned predict");
    assert!(result.is_err(), "the run was cancelled");
    assert!(
        model
            .reset_adapters
            .lock()
            .expect("reset_adapter recorder lock poisoned")
            .is_empty(),
        "no evict goes to an endpoint whose predict missed the close deadline"
    );
}

#[tokio::test]
async fn lane_sessions_score_exactly_the_budgeted_slots_whatever_the_lane_timing() {
    // Three lanes: a 2-step lane, a slow 5-step lane, and a 1-step sprinter.
    // Budget 7 episodes with 10 explicit seeds. The sprinter takes most slots
    // and lane order is timing-dependent, but the scored set is slots 0..7 with
    // seeds 100..107, every slot exactly once, indexed by slot.
    // The sprinter's whole episode is one in-memory round trip; the other
    // lanes' steps take tens of milliseconds, so the slot assignment is not a
    // coin flip under a loaded runner.
    let env = LaneTestEnv::new(vec![
        (2, Duration::from_millis(20)),
        (5, Duration::from_millis(40)),
        (1, Duration::ZERO),
    ]);
    let model = TestModel::default();
    let report = new_driver(
        lane_spec(3, 7, (100..110).collect()),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    assert_eq!(report.total_episodes, 7);
    // The report lists episodes in completion order; by slot they are exactly
    // 1..=7 with seeds 100..107.
    let mut by_slot = report.episodes.clone();
    by_slot.sort_by_key(|e| e.episode_index);
    let indices: Vec<i64> = by_slot.iter().map(|e| e.episode_index).collect();
    assert_eq!(indices, (1..=7).collect::<Vec<_>>(), "one record per slot");
    let seeds: Vec<i64> = by_slot.iter().map(|e| e.seed.unwrap()).collect();
    assert_eq!(
        seeds,
        (100..107).collect::<Vec<_>>(),
        "seed is fixed by slot"
    );
    let lanes_used: std::collections::BTreeSet<i32> =
        report.episodes.iter().map(|e| e.env_index).collect();
    assert_eq!(
        lanes_used,
        [0, 1, 2].into_iter().collect(),
        "every lane ran"
    );
    // Lanes 0 and 1 hold slots 0 and 1 for at least 40ms; the sprinter takes
    // every other slot: 2..7, five episodes.
    let sprinter = report.episodes.iter().filter(|e| e.env_index == 2).count();
    assert_eq!(sprinter, 5, "the fastest lane takes the remaining slots");
    assert!(
        sprinter >= 3,
        "the fastest lane takes the most slots, got {sprinter}"
    );

    // The env saw exactly the seven seeded lane resets and one close; the
    // release handle released once after every lane finished.
    let state = env.inner.lock().expect("lane env state lock poisoned");
    let mut seen: Vec<i64> = state.resets.iter().filter_map(|(_, s)| *s).collect();
    seen.sort_unstable();
    assert_eq!(seen, (100..107).collect::<Vec<_>>());
    assert_eq!(env.closed.load(Ordering::SeqCst), 1);
    assert!(model.closed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn lane_sessions_idle_surplus_lanes_when_the_budget_is_smaller() {
    // Four lanes, two episodes: two lanes run one episode each and the other
    // two never reset (their first slot claim is already past the budget).
    let env = LaneTestEnv::new(vec![(1, Duration::ZERO); 4]);
    let hooks = Arc::new(RecordingHooks::default());
    let report = new_driver(
        lane_spec(4, 2, Vec::new()),
        env.clone(),
        TestModel::default(),
        hooks.clone(),
    )
    .run()
    .await
    .unwrap();
    assert_eq!(report.total_episodes, 2);
    // Driver-owned lane resets complete each episode at its terminal step, as
    // the last event under its id.
    let completed = hooks.completed_ids();
    assert_eq!(completed.len(), 2);
    for id in &completed {
        assert_eq!(
            hooks.events_for(id),
            ["started", "observation", "step", "completed"]
        );
    }
    assert_eq!(
        env.inner
            .lock()
            .expect("lane env state lock poisoned")
            .resets
            .len(),
        2
    );
}

/// A lane-capable env under NEXT_STEP autoreset: every op names exactly one
/// lane, each lane ends its episode on a fixed cadence and rolls itself on the
/// following step. No driver reset ever runs after the cold start, so the
/// route's episode budget is the only thing that can end the run.
#[derive(Clone)]
struct NextStepLaneEnv {
    inner: Arc<Mutex<NextStepLaneState>>,
    length: usize,
}

struct NextStepLaneState {
    resets: usize,
    step: Vec<usize>,
    pending_autoreset: Vec<bool>,
}

impl NextStepLaneEnv {
    fn new(num_envs: usize, length: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(NextStepLaneState {
                resets: 0,
                step: vec![0; num_envs],
                pending_autoreset: vec![false; num_envs],
            })),
            length,
        }
    }
}

#[async_trait]
impl RuntimeEnv for NextStepLaneEnv {
    async fn reset(&mut self, request: EnvResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        assert_eq!(request.env_indices.len(), 1, "lane resets name one lane");
        let lane = request.env_indices[0] as usize;
        let mut state = self.inner.lock().expect("lane env lock poisoned");
        state.resets += 1;
        state.step[lane] = 0;
        state.pending_autoreset[lane] = false;
        Ok(RuntimeEnvReset {
            observation: vec![payload([0])],
            infos: None,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn step(&mut self, request: EnvStepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        assert_eq!(request.env_indices.len(), 1, "lane steps name one lane");
        let lane = request.env_indices[0] as usize;
        let mut state = self.inner.lock().expect("lane env lock poisoned");
        if state.pending_autoreset[lane] {
            // t+1: the env rolls this lane itself and returns the new episode's
            // reset observation.
            state.pending_autoreset[lane] = false;
            state.step[lane] = 0;
            return Ok(RuntimeEnvStep {
                observation: vec![payload([0])],
                rewards: vec![0.0],
                terminated: vec![false],
                truncated: vec![false],
                infos: None,
                endpoint_total_ns: None,
                phases: EndpointPhases::default(),
            });
        }
        state.step[lane] += 1;
        let done = state.step[lane] >= self.length;
        state.pending_autoreset[lane] = done;
        Ok(RuntimeEnvStep {
            observation: vec![payload([0])],
            rewards: vec![1.0],
            terminated: vec![done],
            truncated: vec![false],
            infos: None,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}

#[tokio::test]
async fn a_next_step_lane_session_stops_at_the_episode_budget() {
    // A lane group's budget is normally the slot counter, claimed at reset -- but
    // under NEXT_STEP the env owns the rolls and no reset ever runs, so nothing
    // checked `max_episodes` and the run never terminated.
    let num_envs = 2;
    let budget = 4;
    let env = NextStepLaneEnv::new(num_envs, 2);
    let hooks = Arc::new(RecordingHooks::default());
    let spec = RuntimeSessionSpec {
        subset_step: true,
        ..vector_spec(num_envs, budget)
    };
    let report = tokio::time::timeout(
        Duration::from_secs(10),
        new_driver(spec, env.clone(), TestModel::default(), hooks.clone()).run(),
    )
    .await
    .expect("a lane session under NEXT_STEP must stop at its episode budget")
    .unwrap();

    assert_eq!(
        env.inner.lock().expect("lane env lock poisoned").resets,
        num_envs,
        "one cold-start reset per lane, and never again under NEXT_STEP"
    );
    assert!(
        (budget as i64..=budget as i64 + num_envs as i64).contains(&report.total_episodes),
        "the budget bounds the run (one lane may overshoot by the episode it \
         was already finishing), got {}",
        report.total_episodes
    );
    assert_eq!(report.episodes.len() as i64, report.total_episodes);
    let completed = hooks.completed_ids();
    assert_eq!(
        completed
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        completed.len(),
        "every completion names a distinct episode id: {completed:?}"
    );
}

#[tokio::test]
async fn an_evict_never_waits_behind_a_parked_lane_predict() {
    // Lane episodes end while another lane's predict is in flight. A model
    // serializes evict behind predict on its one lock, and a predict
    // parked in the driver's pending set is not polled while the driver awaits
    // the evict: flushing evictions then would wait out the predict timeout.
    let env = LaneTestEnv::new(vec![
        (2, Duration::from_millis(5)),
        (3, Duration::from_millis(15)),
        (4, Duration::from_millis(25)),
    ]);
    let model = TestModel {
        predict_delay: Some(Duration::from_millis(10)),
        ..Default::default()
    };
    let report = tokio::time::timeout(
        Duration::from_secs(20),
        new_driver(
            lane_spec(3, 9, (100..109).collect()),
            env,
            model,
            Arc::new(RecordingHooks::default()),
        )
        .run(),
    )
    .await
    .expect("the lane run completes without waiting on the evict timeout")
    .unwrap();
    assert_eq!(report.total_episodes, 9);
}
