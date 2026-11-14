//! Route shutdown: the close timeout on a hung model, evictions yielding to
//! cancellation, and a failed route ending its live episode.

use super::*;

#[tokio::test]
async fn shutdown_enforces_service_close_timeout_on_hung_model() {
    let env = TestEnv::default();
    let model = TestModel {
        release_adapter_hangs: true,
        ..Default::default()
    };
    let mut spec = one_episode_spec();
    spec.limits.service_close_timeout = Duration::from_millis(50);

    // Without driver-side timeout enforcement, run() would hang forever in
    // shutdown on the hung close_route. The driver must give up after
    // service_close_timeout and complete the session.
    let report = tokio::time::timeout(
        Duration::from_secs(5),
        new_driver(
            spec,
            env,
            model.clone(),
            Arc::new(RecordingHooks::default()),
        )
        .run(),
    )
    .await
    .expect("driver hung in shutdown despite service_close_timeout")
    .expect("session should complete");

    assert_eq!(report.total_episodes, 1);
    // The hung close never set `closed`, confirming the driver abandoned it.
    assert!(!model.closed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn hung_eviction_yields_to_cancellation() {
    use tokio_util::sync::CancellationToken;

    // The first episode's eviction goes out at the top of the next loop
    // iteration (the second episode's reset is in flight) and never returns.
    // Cancelling the route must interrupt that await: the run ends as
    // cancelled and still releases the model, instead of sitting on the
    // eviction until the model answers.
    let env = TestEnv::default();
    let model = TestModel {
        reset_adapter_hangs: true,
        ..Default::default()
    };
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(2);
    let cancellation = CancellationToken::new();
    // Bounded so a driver that never evicts at all fails this test instead of
    // spinning here forever.
    let cancel_once_evicting = tokio::time::timeout(Duration::from_secs(5), async {
        while model
            .reset_adapters
            .lock()
            .expect("reset_adapter recorder lock poisoned")
            .is_empty()
        {
            tokio::task::yield_now().await;
        }
        cancellation.cancel();
    });
    let (result, _) = tokio::join!(
        tokio::time::timeout(
            Duration::from_secs(5),
            new_driver(
                spec,
                env,
                model.clone(),
                Arc::new(RecordingHooks::default()),
            )
            .run_with_cancellation(cancellation.clone()),
        ),
        cancel_once_evicting,
    );
    let error = result
        .expect("cancellation must interrupt the hung eviction")
        .unwrap_err();

    assert!(
        matches!(error, RuntimeError::RouteCancelled { .. }),
        "expected RouteCancelled, got {error:?}"
    );
    // The abandoned eviction is the only one attempted (the model never
    // predicted on the second episode, so teardown has nothing to end), and
    // the release still went out after it.
    assert_eq!(
        model
            .reset_adapters
            .lock()
            .expect("reset_adapter recorder lock poisoned")
            .len(),
        1
    );
    assert!(model.closed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn failed_route_ends_its_live_episode_before_release() {
    // Episode 1 runs two steps and completes; episode 2's first predict fails
    // the route. The model has predicted on episode 2 and must hear its end
    // before the release (a context-aware model keeps per-episode state until
    // then), and episode 1, already evicted at completion, is not ended twice.
    let env = TestEnv {
        terminal_after: 2,
        ..Default::default()
    };
    let model = TestModel {
        fail_predict_at: Some(3),
        ..Default::default()
    };
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(2);

    let error = new_driver(
        spec,
        env,
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap_err();
    assert!(
        matches!(error, RuntimeError::ModelRpc { .. }),
        "expected the failed predict, got {error:?}"
    );

    let lifecycle = model
        .lifecycle
        .lock()
        .expect("lifecycle lock poisoned")
        .clone();
    let first = lifecycle[0].1.clone();
    let second = lifecycle[3].1.clone();
    assert_ne!(first, second);
    assert_eq!(
        lifecycle,
        [
            ("predict", first.clone()),
            ("predict", first.clone()),
            ("evict", first),
            ("predict", second.clone()),
            ("evict", second),
        ]
    );
    assert!(model.closed.load(Ordering::SeqCst));
}
