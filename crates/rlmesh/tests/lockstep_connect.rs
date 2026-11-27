#![allow(clippy::unwrap_used)]

mod common;

use std::io::BufRead;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::CatalogEnv;
use rlmesh::ext::ErrorExt;
use rlmesh::lockstep::ConnectOptions;
use rlmesh_spaces::AutoresetMode;

const CHILD: &str = "RLMESH_TEST_CONNECT_CHILD";

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

    drop(child);
    let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let free_port = free.local_addr().unwrap().port();
    drop(free);
    let keyed = format!("127.0.0.1:{free_port}#{key}/env");
    let dialing = tokio::spawn(async move {
        rlmesh::lockstep::connect(&keyed, options(Duration::from_secs(3))).await
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(std::net::TcpListener::bind(("127.0.0.1", free_port)).is_ok());
    let absent = dialing.await.unwrap().err().expect("nobody listens");
    assert!(absent.is_transient(), "{absent}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_env_declaring_an_edition_this_build_does_not_know_is_a_final_refusal() {
    let binding = rlmesh::lockstep::bind(":0/catalog").await.unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    let served = tokio::spawn(binding.serve(
        Arc::new(CatalogEnv::new(1, AutoresetMode::Disabled, false)),
        rlmesh::lockstep::ServeOptions {
            workflow_edition: Some("2099.01".into()),
            ..Default::default()
        },
        cancel.clone(),
    ));
    let refused = rlmesh::binding::model::LockstepEnv::connect(&address, ConnectOptions::default())
        .await
        .err()
        .expect("an unknown edition");
    assert!(!refused.is_transient(), "{refused}");
    let message = refused.to_string();
    assert!(
        message.contains("\"2099.01\"")
            && message.contains(rlmesh::current_workflow_edition())
            && message.contains("upgrade the rlmesh runtime"),
        "{message}"
    );
    cancel.cancel();
    served.await.unwrap().unwrap();
}
