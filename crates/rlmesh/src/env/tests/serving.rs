//! Serving the Gymnasium shape end to end: single, lane and vector envs,
//! authored failures, and the server's construction checks.

use super::*;

#[test]
fn a_started_server_serves_the_gymnasium_shape_and_closes_the_env() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    assert!(
        crate::address::is_v2(server.address()),
        "{}",
        server.address()
    );
    assert_eq!(server.env_contract().id(), "Counter-v0");
    assert_eq!(server.env_contract().num_envs(), 1);
    assert_eq!(
        server.to_string(),
        format!(
            "EnvServer(address='{}')",
            rlmesh_session::address::redact(server.address())
        )
    );
    let err = server.wait(Some(Duration::ZERO)).unwrap_err();
    assert_eq!(err.class(), ErrorClass::RuntimeError);
    assert_eq!(err.to_string(), "wait() called before start()");

    server.start().unwrap();
    let err = server.start().unwrap_err();
    assert_eq!(err.class(), ErrorClass::RuntimeError);
    let mut remote = dial(server.address()).unwrap();
    let (observation, info) = reset(&mut remote, Some(1));
    assert_eq!(observation, Value::Int(1));
    assert!(matches!(info.get("episode_ids"), Some(Value::List(ids)) if ids.len() == 1));
    let (observation, reward, _, _, info) = step(&mut remote, 1).unwrap();
    assert_eq!(observation, Value::Int(2));
    assert_eq!(reward, 1.0);
    assert_eq!(info.get("completed_episodes"), Some(&Value::Int(0)));
    remote.close().unwrap();
    // close is idempotent; other I/O on a closed client is misuse (RMT-13).
    remote.close().unwrap();
    let err = remote.reset(None, None).unwrap_err();
    assert_eq!(err.class(), ErrorClass::RuntimeError);
    assert_eq!(err.to_string(), "RemoteEnv is closed");
    assert_eq!(remote.env_contract().id(), "Counter-v0");
    drop(remote);

    assert!(!server.wait(Some(Duration::from_millis(20))).unwrap());
    server.shutdown().unwrap();
    assert!(server.wait(None).unwrap());
    assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
    // Every callback, close included, ran on the lane's one thread.
    let threads = counters.threads.lock().unwrap();
    assert!(
        threads.windows(2).all(|pair| pair[0] == pair[1]),
        "{threads:?}"
    );
    assert_ne!(threads[0], std::thread::current().id());
    let err = server.serve().unwrap_err();
    assert_eq!(
        err.to_string(),
        "serve() called after the server has been stopped"
    );
}

#[test]
fn close_env_on_shutdown_false_leaves_the_env_open() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(
        Counter::new(&counters),
        EnvServerOptions {
            close_env_on_shutdown: false,
            ..Default::default()
        },
    )
    .unwrap();
    server.start().unwrap();
    server.shutdown().unwrap();
    assert_eq!(counters.closes.load(Ordering::SeqCst), 0);

    // A never-started server closes its env on shutdown, and drop does too.
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    drop(server);
    assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
}

#[test]
fn an_authored_failure_reads_op_failed_at_the_client() {
    let counters = Arc::new(Counters::default());
    let mut env = Counter::new(&counters);
    env.fail_step = Some(Error::environment_recoverable("try again"));
    let server = EnvServer::new(env, quick()).unwrap();
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    reset(&mut remote, None);
    let err = step(&mut remote, 0).unwrap_err();
    assert_eq!(err.class(), ErrorClass::RecoverableEnvironmentException);
    assert_eq!(err.to_string(), "[INTERNAL] step failed: try again");
    drop(remote);
    server.shutdown().unwrap();

    let mut env = Counter::new(&counters);
    env.fail_step = Some(Error::invalid_argument("boom"));
    let server = EnvServer::new(env, quick()).unwrap();
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    reset(&mut remote, None);
    let err = step(&mut remote, 0).unwrap_err();
    assert_eq!(err.class(), ErrorClass::EnvironmentException);
    assert_eq!(err.to_string(), "[INTERNAL] step failed: boom");
    drop(remote);
    server.shutdown().unwrap();
}

#[test]
fn attached_authored_failures_keep_their_wire_class_and_raw_text() {
    for (error, class, text) in [
        (
            Error::environment_recoverable("try again"),
            ErrorClass::RecoverableEnvironmentException,
            "[INTERNAL] step failed: try again",
        ),
        (
            Error::environment("failed"),
            ErrorClass::EnvironmentException,
            "[INTERNAL] step failed: failed",
        ),
        (
            Error::invalid_argument("bad value"),
            ErrorClass::EnvironmentException,
            "[INTERNAL] step failed: bad value",
        ),
    ] {
        let counters = Arc::new(Counters::default());
        let mut env = Counter::new(&counters);
        env.fail_step = Some(crate::binding::AttachedError::attach(error, ()));
        let server = EnvServer::new(env, quick()).unwrap();
        server.start().unwrap();
        let mut remote = dial(server.address()).unwrap();
        reset(&mut remote, None);
        let err = step(&mut remote, 0).unwrap_err();
        assert_eq!(err.class(), class);
        assert_eq!(err.to_string(), text);
        drop(remote);
        server.shutdown().unwrap();
    }
}

#[test]
fn lanes_serve_one_endpoint_and_must_agree() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::lanes(
        vec![Counter::new(&counters), Counter::new(&counters)],
        quick(),
    )
    .unwrap();
    assert_eq!(server.env_contract().num_envs(), 2);
    server.shutdown().unwrap();
    assert_eq!(counters.closes.load(Ordering::SeqCst), 2);

    let mut odd = Counter::new(&counters);
    let mut metadata = Metadata::new();
    metadata.insert("render_fps", 30);
    odd.metadata = Some(metadata);
    let err = EnvServer::lanes(vec![Counter::new(&counters), odd], quick()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert!(err.to_string().contains("lane 1 disagrees"), "{err}");

    let err = EnvServer::lanes(Vec::<Counter>::new(), quick()).unwrap_err();
    assert_eq!(err.to_string(), "EnvServer needs at least one environment");
}

#[test]
fn v1_bind_addresses_are_refused() {
    let counters = Arc::new(Counters::default());
    for address in [
        "tcp://127.0.0.1:0",
        "unix:///tmp/env.sock",
        "127.0.0.1:5555",
    ] {
        let err = EnvServer::new(
            Counter::new(&counters),
            EnvServerOptions {
                address: Some(address.into()),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains(crate::address::V1_REFUSAL),
            "{err}"
        );
    }
}

#[test]
fn a_zero_timeout_is_refused_at_construction() {
    let counters = Arc::new(Counters::default());
    let err = EnvServer::new(
        Counter::new(&counters),
        EnvServerOptions {
            options: Some(crate::ServeOptions {
                close_timeout: Some(Duration::ZERO),
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert_eq!(err.to_string(), "close_timeout_seconds must be positive");
}

#[test]
fn a_foreground_serve_runs_lane_0_on_the_caller() {
    let counters = Arc::new(Counters::default());
    let server = Arc::new(EnvServer::new(Counter::new(&counters), quick()).unwrap());
    let address = server.address().to_string();
    let serving = {
        let server = Arc::clone(&server);
        std::thread::spawn(move || (std::thread::current().id(), server.serve()))
    };
    let mut remote = loop {
        if let Ok(remote) = dial(&address) {
            break remote;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    reset(&mut remote, None);
    drop(remote);
    let err = server.wait(None).unwrap_err();
    assert!(err.to_string().contains("foreground"), "{err}");
    server.shutdown().unwrap();
    let (thread, result) = serving.join().unwrap();
    result.unwrap();
    assert!(server.wait(None).unwrap());
    let threads = counters.threads.lock().unwrap();
    assert!(threads.iter().all(|id| *id == thread), "{threads:?}");
    assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
}

#[test]
fn serve_async_drops_stop_the_server() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    client(async {
        let serving = server.serve_async();
        futures::pin_mut!(serving);
        // Poll once so the server starts, then drop the future.
        assert!(futures::poll!(serving.as_mut()).is_pending());
    });
    assert!(server.wait(Some(Duration::from_secs(5))).unwrap());
    drop(server);
    assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
}

#[test]
fn a_vector_env_sees_the_batched_layout() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let server = EnvServer::vector(
        Pair {
            num_envs: 2,
            seen: Arc::clone(&seen),
        },
        quick(),
    )
    .unwrap();
    assert_eq!(server.env_contract().num_envs(), 2);
    server.start().unwrap();
    let mut remote =
        RemoteVectorEnv::new(RemoteEnvOptions::default().address(server.address())).unwrap();
    assert_eq!(remote.num_envs(), 2);
    assert!(std::ptr::eq(
        remote.observation_space(),
        remote.single_observation_space()
    ));
    let (observation, info) = remote.reset(Some(VectorSeed::One(7)), None).unwrap();
    assert_eq!(observation.as_tensor().unwrap().shape(), &[2, 2]);
    assert!(matches!(info.get("episode_ids"), Some(Value::List(ids)) if ids.len() == 2));
    let (_, rewards, terminated, _, info) = remote
        .step(Value::List(vec![Value::Int(1), Value::Int(2)]))
        .unwrap();
    assert_eq!(rewards, vec![0.5, 0.5]);
    assert_eq!(terminated, vec![false, false]);
    assert_eq!(info.get("completed_episodes"), Some(&Value::Int(0)));
    let err =
        crate::ext::reset_lanes(&mut remote, &[u64::from(u32::MAX) + 1], None, None).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    drop(remote);
    // A vector endpoint refuses the single-env client.
    let err = dial(server.address()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert!(
        err.to_string().contains("Use RemoteVectorEnv instead"),
        "{err}"
    );
    server.shutdown().unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0], Value::Text("Some(One(7))".into()));
    let actions = seen[1].as_tensor().unwrap();
    assert_eq!(actions.shape(), &[2]);
    assert_eq!(actions.dtype(), DType::Int64);
}

#[test]
fn a_one_lane_vector_env_serves_as_a_scalar_lane() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let server = EnvServer::vector(
        Pair {
            num_envs: 1,
            seen: Arc::clone(&seen),
        },
        quick(),
    )
    .unwrap();
    assert_eq!(server.env_contract().num_envs(), 1);
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    reset(&mut remote, Some(3));
    let (_, reward, _, _, _) = step(&mut remote, 2).unwrap();
    assert_eq!(reward, 0.5);
    drop(remote);
    // A one-lane endpoint refuses the vector client.
    let err =
        RemoteVectorEnv::new(RemoteEnvOptions::default().address(server.address())).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    server.shutdown().unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0], Value::Text("Some(One(3))".into()));
    // Python's `_batch1`: a Discrete action is a one-item list.
    assert_eq!(seen[1], Value::List(vec![Value::Int(2)]));

    let err = EnvServer::vector(
        Pair {
            num_envs: 0,
            seen: Arc::new(Mutex::new(Vec::new())),
        },
        quick(),
    )
    .unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
}

#[test]
fn autoreset_mode_follows_env_27() {
    let counters = Arc::new(Counters::default());
    for (mode, ok) in [
        (Value::Text("NextStep".into()), true),
        (Value::Text("Disabled".into()), true),
        (Value::Text("SameStep".into()), true),
        (Value::Text("Bogus".into()), false),
        (Value::None, false),
        (Value::Int(1), false),
    ] {
        let mut env = Counter::new(&counters);
        let mut metadata = Metadata::new();
        metadata.insert("autoreset_mode", mode.clone());
        env.metadata = Some(metadata);
        let result = EnvServer::new(env, quick());
        assert_eq!(result.is_ok(), ok, "{mode:?}");
        if let Err(err) = result {
            assert_eq!(err.class(), ErrorClass::ValueError);
        }
    }
}
