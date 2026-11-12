//! Whether a failed lockstep connect can succeed on retry: a driver that
//! dials an env still starting retries while the error is transient and
//! stops at once on a refusal the same call meets again.

#![allow(clippy::unwrap_used)]

mod common;

use std::io::BufRead;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::CatalogEnv;
use rlmesh::lockstep::ConnectOptions;
use rlmesh_spaces::AutoresetMode;

/// Set in the child process that serves the env.
const CHILD: &str = "RLMESH_TEST_CONNECT_CHILD";

/// The child half of [`a_connect_failure_says_whether_retrying_can_succeed`]:
/// serves an env in a session it hosts under the launch identity its parent
/// passed, as a launched container does, and prints only its port.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn connect_child() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let binding = rlmesh::lockstep::bind("127.0.0.1:0/env").await.unwrap();
    let address = rlmesh_session::address::Address::participant(&binding.address()).unwrap();
    let port = address.host.and_then(|host| host.port).unwrap();
    println!("PORT {port}");
    binding
        .serve(
            Arc::new(CatalogEnv::new(1, AutoresetMode::Disabled, false)),
            rlmesh::lockstep::ServeOptions::default(),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
}

/// Kills the child env when the test ends, however it ends.
struct Child(std::process::Child);

impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn options(timeout: Duration) -> ConnectOptions {
    ConnectOptions {
        timeout,
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_connect_failure_says_whether_retrying_can_succeed() {
    let (identity, key) = rlmesh_session::address::launch_identity().unwrap();
    let (_, wrong) = rlmesh_session::address::launch_identity().unwrap();
    // The identity reaches the child through its environment, never argv.
    let mut child = Child(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "connect_child", "--nocapture"])
            .env(CHILD, "1")
            .env(
                rlmesh_session::address::HOST_IDENTITY_VAR,
                identity.as_str(),
            )
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut lines = std::io::BufReader::new(child.0.stdout.take().unwrap()).lines();
    let port = loop {
        let line = lines.next().expect("the child serves").unwrap();
        if let Some(port) = line.strip_prefix("PORT ") {
            break port.to_string();
        }
    };

    let env = rlmesh::lockstep::connect(
        &format!("127.0.0.1:{port}#{key}/env"),
        options(Duration::from_secs(20)),
    )
    .await
    .expect("the right key connects");
    drop(env);

    // A key the host does not hold: final, and said well within the
    // timeout instead of waiting it out as a host not up yet.
    let started = Instant::now();
    let refused = rlmesh::lockstep::connect(
        &format!("127.0.0.1:{port}#{wrong}/env"),
        options(Duration::from_secs(30)),
    )
    .await
    .err()
    .expect("a wrong key is refused");
    assert!(!refused.is_transient(), "{refused}");
    assert!(started.elapsed() < Duration::from_secs(15), "{refused}");
    assert!(refused.to_string().contains("refused"), "{refused}");
    assert!(!refused.to_string().contains(&wrong.to_string()));

    // A robot the session does not have reads as one not joined yet, and
    // the error names the robots that are there.
    let missing = rlmesh::lockstep::connect(
        &format!("127.0.0.1:{port}#{key}/envv"),
        options(Duration::from_secs(2)),
    )
    .await
    .err()
    .expect("no such robot");
    assert!(missing.is_transient(), "{missing}");
    assert!(
        missing.to_string().contains("robots there: env"),
        "{missing}"
    );

    // Nobody listening yet: transient, and the keyed address does not host
    // the session in the driver's place, so the env can still bind it.
    drop(child);
    let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let free_port = free.local_addr().unwrap().port();
    drop(free);
    let absent = rlmesh::lockstep::connect(
        &format!("127.0.0.1:{free_port}#{key}/env"),
        options(Duration::from_secs(2)),
    )
    .await
    .err()
    .expect("nobody listens");
    assert!(absent.is_transient(), "{absent}");
    assert!(std::net::TcpListener::bind(("127.0.0.1", free_port)).is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_edition_the_env_does_not_run_is_a_final_refusal() {
    let binding = rlmesh::lockstep::bind(":0/catalog").await.unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    let served = tokio::spawn(binding.serve(
        Arc::new(CatalogEnv::new(1, AutoresetMode::Disabled, false)),
        rlmesh::lockstep::ServeOptions::default(),
        cancel.clone(),
    ));
    // The runner declares an edition older than any the env runs.
    let refused = rlmesh::binding::model::LockstepEnv::connect(
        &address,
        ConnectOptions::default(),
        Some("2026.06"),
    )
    .await
    .err()
    .expect("no mutual edition");
    assert!(!refused.is_transient(), "{refused}");
    assert!(refused.to_string().contains("edition"), "{refused}");
    cancel.cancel();
    served.await.unwrap().unwrap();
}
