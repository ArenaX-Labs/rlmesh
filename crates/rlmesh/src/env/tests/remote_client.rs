use super::*;

#[test]
fn blocking_calls_refuse_an_async_context() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let err = runtime
        .block_on(async { tokio::spawn(async move { server.serve() }).await })
        .unwrap()
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::RuntimeError);
    assert_eq!(
        err.to_string(),
        "EnvServer::serve() blocks; call serve_async() from async code"
    );
}

#[test]
fn async_twins_run_on_any_executor() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    server.start().unwrap();
    let address = server.address().to_string();
    futures::executor::block_on(async {
        let mut remote = RemoteEnv::new_async(RemoteEnvOptions::default().address(&address))
            .await
            .unwrap();
        let (observation, _) = remote.reset_async(Some(1), None).await.unwrap();
        assert_eq!(observation, Value::Int(1));
        let (_, reward, _, _, _) = remote.step_async(Value::Int(1)).await.unwrap();
        assert_eq!(reward, 1.0);
        assert_eq!(remote.render_async(None).await.unwrap(), None);
        remote.close_async().await.unwrap();
    });
    let inside = address.clone();
    let err =
        client(async { tokio::spawn(async move { dial(&inside) }).await.unwrap() }).unwrap_err();
    assert_eq!(err.class(), ErrorClass::RuntimeError);
    assert!(err.to_string().contains("RemoteEnv::new_async()"), "{err}");
    server.shutdown().unwrap();
}

#[test]
fn a_request_past_its_timeout_is_a_timeout_error() {
    let counters = Arc::new(Counters::default());
    let mut env = Counter::new(&counters);
    env.slow = Some(Duration::from_secs(1));
    let server = EnvServer::new(env, quick()).unwrap();
    server.start().unwrap();
    let mut remote = RemoteEnv::new(
        RemoteEnvOptions::default()
            .address(server.address())
            .request_timeout(Duration::from_millis(250)),
    )
    .unwrap();
    reset(&mut remote, None);
    let err = step(&mut remote, 1).unwrap_err();
    assert_eq!(err.class(), ErrorClass::TimeoutError);
    assert_eq!(err.to_string(), "operation timed out after 250ms");
    drop(remote);
    server.shutdown().unwrap();
}

#[test]
fn a_fired_cancel_handle_fails_one_call() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    remote.cancel_handle().cancel();
    let err = remote.reset(None, None).unwrap_err();
    assert_eq!(err.class(), ErrorClass::KeyboardInterrupt);
    reset(&mut remote, None);
    drop(remote);
    server.shutdown().unwrap();
}

#[test]
fn construction_errors_follow_rmt_12() {
    let err = RemoteEnv::new(RemoteEnvOptions::default()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    for address in [
        "tcp://127.0.0.1:1",
        "unix:///tmp/env.sock",
        "https://env.example.com",
    ] {
        let err = dial(address).unwrap_err();
        assert_eq!(err.class(), ErrorClass::ValueError);
        assert!(
            err.to_string().contains(crate::address::V1_REFUSAL),
            "{err}"
        );
    }
}

#[test]
fn closed_client_refusal_keeps_its_idle_firing_pending() {
    let counters = Arc::new(Counters::default());
    let server = EnvServer::new(Counter::new(&counters), quick()).unwrap();
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    let token = remote.cancel_handle();
    remote.close().unwrap();
    token.cancel();
    assert_eq!(
        remote.reset(None, None).unwrap_err().class(),
        ErrorClass::RuntimeError
    );
    assert_eq!(
        remote.step(Value::Int(0)).unwrap_err().class(),
        ErrorClass::RuntimeError
    );
    assert_eq!(
        remote.render(None).unwrap_err().class(),
        ErrorClass::RuntimeError
    );
    assert!(token.take());
    assert!(!token.take());
    server.shutdown().unwrap();
}

#[test]
fn immediate_remote_completion_observes_handle_cancellation() {
    struct Fire {
        token: Arc<Mutex<Option<crate::ext::CancelToken>>>,
    }
    impl crate::Env for Fire {
        fn observation_space(&self) -> crate::spaces::Space {
            Discrete::new(4, DiscreteOptions::default()).unwrap().into()
        }
        fn action_space(&self) -> crate::spaces::Space {
            self.observation_space()
        }
        fn reset(
            &mut self,
            _: Option<i64>,
            _: Option<&crate::InfoDict>,
        ) -> crate::Result<(crate::Value, crate::InfoDict)> {
            Ok((crate::Value::Int(0), crate::InfoDict::new()))
        }
        fn step(
            &mut self,
            _: Option<crate::Value>,
        ) -> crate::Result<(crate::Value, f64, bool, bool, crate::InfoDict)> {
            if let Some(token) = self.token.lock().unwrap().take() {
                token.cancel();
            }
            Ok((
                crate::Value::Int(1),
                1.,
                false,
                false,
                crate::InfoDict::new(),
            ))
        }
    }
    let token = Arc::new(Mutex::new(None));
    let server = EnvServer::new(
        Fire {
            token: token.clone(),
        },
        quick(),
    )
    .unwrap();
    server.start().unwrap();
    let mut remote = dial(server.address()).unwrap();
    remote.reset(None, None).unwrap();
    *token.lock().unwrap() = Some(remote.cancel_handle());
    assert_eq!(
        remote.step(crate::Value::Int(0)).unwrap_err(),
        crate::Error::Cancelled
    );
    remote.step(crate::Value::Int(0)).unwrap();
    remote.close().unwrap();
    server.shutdown().unwrap();
}

#[test]
fn a_connect_timeout_reports_why_the_join_ran_out() {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let base = listener.local_addr().unwrap().port();
    let (_, key) = rlmesh_session::address::launch_identity().unwrap();
    let timeout = Duration::from_secs(2);
    let started = std::time::Instant::now();
    let Err(err) = RemoteEnv::new(
        RemoteEnvOptions::default()
            .address(format!("127.0.0.1:{base}#{key}/env"))
            .connect_timeout(timeout),
    ) else {
        panic!("joined a host that never answers");
    };
    let waited = started.elapsed();
    assert!(!err.to_string().contains("refused"), "{err}");
    assert!(waited <= timeout + Duration::from_millis(500), "{waited:?}");
    drop(listener);
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.local_addr().unwrap().port()
}

#[test]
fn a_bare_host_and_port_is_the_env_there() {
    let counters = Arc::new(Counters::default());
    let port = free_port();
    let bare = format!("127.0.0.1:{port}");
    let server = EnvServer::new(
        Counter::new(&counters),
        EnvServerOptions {
            address: Some(bare.clone()),
            ..quick()
        },
    )
    .unwrap();
    server.start().unwrap();
    assert!(server.address().ends_with("/env"), "{}", server.address());
    for address in [bare.clone(), format!("{bare}/env")] {
        let mut remote = dial(&address).unwrap();
        assert_eq!(remote.address(), format!("{bare}/env"));
        let (observation, _) = remote.reset(Some(1), None).unwrap();
        assert_eq!(observation, Value::Int(1));
        remote.close().unwrap();
    }
    server.shutdown().unwrap();
}

#[test]
fn a_bare_port_is_refused_naming_the_host_form() {
    for err in [
        dial("5555").err().unwrap(),
        EnvServer::new(
            Counter::new(&Arc::new(Counters::default())),
            EnvServerOptions {
                address: Some("5555".into()),
                ..quick()
            },
        )
        .err()
        .unwrap(),
    ] {
        assert_eq!(err.class(), ErrorClass::ValueError);
        assert!(err.to_string().contains("\":5555\""), "{err}");
    }
}

fn foreign_server(reply: &'static [u8], first: bool) -> u16 {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut conn) = conn else { return };
            std::thread::spawn(move || {
                let mut buf = [0u8; 256];
                if !first && !matches!(conn.read(&mut buf), Ok(n) if n > 0) {
                    return;
                }
                let _ = conn.write_all(reply);
                let _ = conn.read(&mut buf);
            });
        }
    });
    port
}

const H2_SETTINGS: &[u8] = &[0, 0, 0, 4, 0, 0, 0, 0, 0];

#[test]
fn a_01_grpc_server_is_named_not_timed_out() {
    for (reply, first, http2) in [
        (H2_SETTINGS, true, true),
        (H2_SETTINGS, false, true),
        (&b"HTTP/1.1 400 Bad Request\r\n\r\n"[..], true, false),
        (&b"HTTP/1.1 400 Bad Request\r\n\r\n"[..], false, false),
    ] {
        let port = foreign_server(reply, first);
        let started = std::time::Instant::now();
        let err = RemoteEnv::new(
            RemoteEnvOptions::default()
                .address(format!("127.0.0.1:{port}"))
                .connect_timeout(Duration::from_secs(20)),
        )
        .err()
        .unwrap();
        let text = err.to_string();
        assert_eq!(err.class(), ErrorClass::ProtocolException, "{text}");
        assert!(
            text.contains(&format!(
                "the server at 127.0.0.1:{port} doesn't speak the RLMesh session protocol"
            )) && text.contains("RLMesh 0.2 can't connect to 0.1"),
            "{text}"
        );
        assert_eq!(text.contains("HTTP/2"), http2, "{text}");
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
    }
}
