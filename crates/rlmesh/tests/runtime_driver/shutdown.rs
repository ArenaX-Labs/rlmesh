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
    assert!(!model.closed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn hung_eviction_yields_to_cancellation() {
    use tokio_util::sync::CancellationToken;

    let env = TestEnv::default();
    let model = TestModel {
        reset_adapter_hangs: true,
        ..Default::default()
    };
    let mut spec = one_episode_spec();
    spec.max_episodes = Some(2);
    let cancellation = CancellationToken::new();
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
