//! The episode ledger under runtime-side truncation and episode caps.

use super::*;

// ---------------------------------------------------------------------------
// The episode ledger under a runtime-side truncation: the runtime owns episode
// ids (R1), so a peer-reported completion naming a superseded id is an echo.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_endless_episode_under_next_step_fails_at_the_default_bound() {
    // Lane 0 keeps finishing, lane 1 never does. Under NEXT_STEP the env owns
    // its lanes, so the runtime cannot truncate lane 1; the edition's default
    // bound fails the run there rather than letting it wait forever.
    let env = VectorTestEnv::new(vec![1, usize::MAX]);
    let error = tokio::time::timeout(
        Duration::from_secs(60),
        new_driver(
            vector_spec(2, 3),
            env,
            TestModel::default(),
            Arc::new(RecordingHooks::default()),
        )
        .run(),
    )
    .await
    .expect("the default bound ends the run")
    .expect_err("an episode the env never ends fails the run");
    let message = error.to_string();
    assert!(
        message.contains("lane 1") && message.contains("100000 steps"),
        "{message}"
    );
}

#[tokio::test]
async fn a_runtime_truncated_episode_counts_once() {
    // max_episode_steps=4 over 3 episodes on an env that never terminates: the
    // driver truncates and resets mid-episode, and each truncation completes
    // its episode exactly once, after exactly the cap.
    let env = EndlessEnv::default();
    let hooks = Arc::new(RecordingHooks::default());
    let spec = RuntimeSessionSpec {
        max_episodes: Some(3),
        max_episode_steps: Some(4),
        ..one_episode_spec()
    };
    let report = new_driver(spec, env.clone(), TestModel::default(), hooks.clone())
        .run()
        .await
        .unwrap();

    let state = env.inner.lock().expect("endless env lock poisoned");
    assert_eq!(state.steps, 12, "3 episodes of exactly max_episode_steps=4");
    assert_eq!(
        state.resets, 3,
        "one reset per episode, cold start included"
    );
    drop(state);

    assert_eq!(
        report.total_episodes, 3,
        "an echoed id must not count again"
    );
    assert_eq!(report.total_steps, 12);
    assert_eq!(report.episodes.len(), 3, "one summary per episode");
    assert_eq!(
        report
            .episodes
            .iter()
            .map(|episode| episode.step_count)
            .collect::<Vec<_>>(),
        vec![4, 4, 4],
        "every episode runs the full cap, not one step"
    );
    assert!(
        report.episodes.iter().all(|episode| episode.truncated),
        "the runtime truncated every one of them"
    );
    let flags = hooks.step_flags.lock().unwrap();
    assert_eq!(flags.len(), 12);
    assert!(
        flags
            .iter()
            .all(|(term, _, roll)| term == &[false] && roll == &[false]),
        "a runtime cap never terminates or rolls"
    );
    assert_eq!(
        flags
            .iter()
            .map(|(_, trunc, _)| trunc[0])
            .collect::<Vec<_>>(),
        (1..=12).map(|step| step % 4 == 0).collect::<Vec<_>>(),
        "the capped step carries the truncation itself"
    );
    drop(flags);
    assert_eq!(
        report
            .episodes
            .iter()
            .map(|episode| episode.episode_index)
            .collect::<Vec<_>>(),
        vec![1, 2, 3],
        "distinct, contiguous completion indices"
    );
    let completed = hooks.completed_ids();
    assert_eq!(completed.len(), 3);
    assert_eq!(
        completed
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3,
        "three distinct episode ids completed: {completed:?}"
    );
}

/// A scalar env that never ends an episode and reports `is_success: false` on
/// every step, so only a runtime cap can finish it.
#[derive(Clone, Default)]
struct NeverSucceedsEnv;

#[async_trait]
impl RuntimeEnv for NeverSucceedsEnv {
    async fn reset(&mut self, _request: EnvResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        Ok(RuntimeEnvReset {
            observation: vec![payload([0])],
            infos: None,
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn step(&mut self, _request: EnvStepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        let infos = rlmesh_proto::spaces::v1::MetaMap {
            entries: [(
                "is_success".to_string(),
                MetaValue {
                    kind: Some(meta_value::Kind::Bool(false)),
                },
            )]
            .into(),
        };
        Ok(RuntimeEnvStep {
            observation: vec![payload([0])],
            rewards: vec![1.0],
            terminated: vec![false],
            truncated: vec![false],
            infos: Some(infos),
            endpoint_total_ns: None,
            phases: EndpointPhases::default(),
        })
    }

    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}

#[tokio::test]
async fn a_runtime_capped_episode_keeps_the_last_step_success() {
    // The capping step's info is the episode's last word on its outcome, as the
    // Python Session loop reads it; dropping it reported success as unknown.
    let spec = RuntimeSessionSpec {
        max_episode_steps: Some(1),
        ..one_episode_spec()
    };
    let report = new_driver(
        spec,
        NeverSucceedsEnv,
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    assert_eq!(report.episodes.len(), 1);
    let episode = &report.episodes[0];
    assert!(episode.truncated && !episode.terminated);
    assert_eq!(episode.success, Some(false));
}
