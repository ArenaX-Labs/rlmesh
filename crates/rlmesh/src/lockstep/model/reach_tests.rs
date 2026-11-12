//! A model served where its driver's loopback is not its own (a container,
//! another machine) drives a local env, and one asked to join a session it
//! can't reach says so in seconds.
//!
//! Each side is a process of its own: the model's, whose host this process
//! reaches at an address the model's process cannot dial (a port forwarded
//! from another network namespace, as a container publishes one), and the
//! driver's.

use std::io::BufRead;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::{Duration, Instant};

use crate::spaces::{Discrete, DiscreteOptions, Space};
use crate::{
    Env, EnvServer, EnvServerOptions, InfoDict, Model, ModelOptions, ModelServeOptions,
    RemoteModel, RemoteModelOptions, Result, RunOptions, Value,
};

/// A model child: serves at this participant address until stdin closes.
const MODEL_AT: &str = "RLMESH_TEST_REACH_MODEL_AT";
/// With [`MODEL_AT`]: relay each class port from a Unix socket in this
/// directory (`c0` to `c3`), as a container's published ports.
const RELAY: &str = "RLMESH_TEST_REACH_RELAY";
/// A driver child: runs the model at this address against a local env.
const DRIVE: &str = "RLMESH_TEST_REACH_DRIVE";
/// With [`DRIVE`]: serve the env at its own address and pass that.
const ENV_BY_ADDRESS: &str = "RLMESH_TEST_REACH_ENV_BY_ADDRESS";

/// Ends each episode at its third step.
struct Three(i64);

impl Env for Three {
    fn observation_space(&self) -> Space {
        Discrete::new(4, DiscreteOptions::default())
            .expect("space")
            .into()
    }
    fn action_space(&self) -> Space {
        Discrete::new(2, DiscreteOptions::default())
            .expect("space")
            .into()
    }
    fn reset(&mut self, _: Option<i64>, _: Option<&InfoDict>) -> Result<(Value, InfoDict)> {
        self.0 = 0;
        Ok((Value::Int(0), InfoDict::new()))
    }
    fn step(&mut self, _: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.0 += 1;
        Ok((Value::Int(self.0), 1.0, self.0 >= 3, false, InfoDict::new()))
    }
}

#[test]
fn model_child() {
    let Ok(address) = std::env::var(MODEL_AT) else {
        return;
    };
    if let Ok(dir) = std::env::var(RELAY) {
        let base: SocketAddr = address
            .split('/')
            .next()
            .and_then(|base| base.parse().ok())
            .expect("a base");
        relay_in(PathBuf::from(dir), base);
    }
    let model = Model::new(|_: Value| Ok(Value::Int(1)), ModelOptions::default()).expect("model");
    let cancel = crate::ext::CancelToken::new();
    let stop = cancel.clone();
    std::thread::spawn(move || {
        let _ = std::io::stdin().lock().read_line(&mut String::new());
        stop.cancel();
    });
    let _ = model.serve(
        &address,
        ModelServeOptions {
            cancel: Some(cancel),
            ..Default::default()
        },
    );
}

/// Relays each Unix socket `c{i}` in `dir` to class port `i` of `base`.
fn relay_in(dir: PathBuf, base: SocketAddr) {
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    std::thread::spawn(move || {
        runtime.block_on(async move {
            for class in 0..4_u16 {
                let listener =
                    tokio::net::UnixListener::bind(dir.join(format!("c{class}"))).expect("bind");
                let to = SocketAddr::new(base.ip(), base.port() + class);
                tokio::spawn(async move {
                    while let Ok((mut from, _)) = listener.accept().await {
                        tokio::spawn(async move {
                            if let Ok(mut to) = tokio::net::TcpStream::connect(to).await {
                                let _ = tokio::io::copy_bidirectional(&mut from, &mut to).await;
                            }
                        });
                    }
                });
            }
            std::future::pending::<()>().await;
        });
    });
}

#[test]
#[allow(clippy::print_stdout)]
fn driver_child() {
    let Ok(address) = std::env::var(DRIVE) else {
        return;
    };
    let model = RemoteModel::new(
        RemoteModelOptions::default()
            .address(address)
            .connect_timeout(Duration::from_secs(20)),
    )
    .expect("model");
    let mut env = Three(0);
    let options = || RunOptions {
        episodes: Some(2),
        ..Default::default()
    };
    let started = Instant::now();
    let ran = if std::env::var_os(ENV_BY_ADDRESS).is_some() {
        let server = EnvServer::new(
            Three(0),
            EnvServerOptions {
                address: Some("127.0.0.1:0/elsewhere/env".into()),
                ..Default::default()
            },
        )
        .expect("server");
        server.start().expect("start");
        let ran = crate::run(&model, server.address(), options());
        let _ = server.shutdown();
        ran
    } else {
        crate::run(&model, &mut env, options())
    };
    let ms = started.elapsed().as_millis();
    match ran {
        Ok(result) => println!("STEPS={}", result.total_steps()),
        Err(err) => println!("ERR={err} AFTER={ms}"),
    }
}

fn exe() -> PathBuf {
    std::env::current_exe().expect("test binary")
}

fn child_args(test: &str) -> [String; 3] {
    [
        "--exact".into(),
        format!("lockstep::model::reach_tests::{test}"),
        "--nocapture".into(),
    ]
}

/// The scratch directories both children share.
struct Scratch {
    dir: tempfile::TempDir,
}

impl Scratch {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("scratch");
        std::fs::create_dir(dir.path().join("runtime")).expect("runtime");
        std::fs::create_dir(dir.path().join("state")).expect("state");
        Self { dir }
    }

    fn env(&self, command: &mut Command) {
        command
            .env("RLMESH_RUNTIME_DIR", self.dir.path().join("runtime"))
            .env("RLMESH_STATE_DIR", self.dir.path().join("state"))
            .env_remove("RLMESH_HOST_IDENTITY")
            .env_remove(MODEL_AT)
            .env_remove(RELAY)
            .env_remove(DRIVE)
            .env_remove(ENV_BY_ADDRESS);
    }

    fn relay(&self) -> PathBuf {
        self.dir.path().to_path_buf()
    }
}

/// A model child, stopped when dropped (its stdin closed, then killed).
struct ModelProcess {
    child: Child,
    stdin: Option<ChildStdin>,
}

impl ModelProcess {
    fn start(mut command: Command) -> Self {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .expect("spawn the model");
        let stdin = child.stdin.take();
        Self { child, stdin }
    }
}

impl Drop for ModelProcess {
    fn drop(&mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// What a driver child printed: the steps it ran, or its error.
fn drive(scratch: &Scratch, address: &str, env_by_address: bool) -> String {
    let mut command = Command::new(exe());
    command.args(child_args("driver_child"));
    scratch.env(&mut command);
    command.env(DRIVE, address);
    if env_by_address {
        command.env(ENV_BY_ADDRESS, "1");
    }
    let output = command.output().expect("run the driver");
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.lines()
        .find(|line| line.starts_with("STEPS=") || line.starts_with("ERR="))
        .map_or_else(
            || {
                format!(
                    "no outcome: {text}{}",
                    String::from_utf8_lossy(&output.stderr)
                )
            },
            str::to_string,
        )
}

fn free_base() -> SocketAddr {
    rlmesh_session::transport::free_local_base(Ipv4Addr::LOCALHOST.into()).expect("free ports")
}

/// The plain case: the model in another process on this machine, reached
/// by its address alone, drives a local env.
#[test]
fn a_model_in_another_process_drives_a_local_env() {
    let scratch = Scratch::new();
    let base = free_base();
    let mut command = Command::new(exe());
    command.args(child_args("model_child"));
    scratch.env(&mut command);
    command.env(MODEL_AT, format!("{base}/model"));
    let _model = ModelProcess::start(command);
    assert_eq!(drive(&scratch, &format!("{base}/model"), false), "STEPS=6");
}

/// Whether this machine lets an unprivileged user make a network namespace
/// with its loopback up.
fn namespaces() -> bool {
    Command::new("unshare")
        .args(["-rn", "ip", "link", "set", "lo", "up"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Forwards class ports `published`.. on this machine's loopback to the
/// Unix sockets `c0`.. in `relay`, as a container's published ports.
fn publish(published: SocketAddr, relay: PathBuf) -> tokio::runtime::Runtime {
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    for class in 0..4_u16 {
        let at = SocketAddr::new(published.ip(), published.port() + class);
        let listener = runtime
            .block_on(tokio::net::TcpListener::bind(at))
            .expect("publish");
        let socket = relay.join(format!("c{class}"));
        runtime.spawn(async move {
            while let Ok((mut from, _)) = listener.accept().await {
                let socket = socket.clone();
                tokio::spawn(async move {
                    if let Ok(mut to) = tokio::net::UnixStream::connect(&socket).await {
                        let _ = tokio::io::copy_bidirectional(&mut from, &mut to).await;
                    }
                });
            }
        });
    }
    runtime
}

/// The model in a network namespace of its own, its host's ports published
/// on this machine's loopback under other numbers: the model can dial
/// neither this machine's loopback nor the address its driver dials.
fn contained(
    scratch: &Scratch,
    relay: &Path,
) -> Option<(ModelProcess, String, tokio::runtime::Runtime)> {
    if !namespaces() {
        return None;
    }
    let (identity, key) = rlmesh_session::address::launch_identity().expect("identity");
    // Inside the namespace every port is free.
    let inside = SocketAddr::from((Ipv4Addr::LOCALHOST, 20_000));
    let mut command = Command::new("unshare");
    command
        .args(["-rn", "sh", "-c", r#"ip link set lo up && exec "$0" "$@""#])
        .arg(exe())
        .args(child_args("model_child"));
    scratch.env(&mut command);
    // The launcher's identity, through the environment as a sandbox passes it.
    command
        .env("RLMESH_HOST_IDENTITY", identity.as_str())
        .env(MODEL_AT, format!("{inside}/model"))
        .env(RELAY, relay);
    let model = ModelProcess::start(command);
    let published = free_base();
    let runtime = publish(published, relay.to_path_buf());
    Some((model, format!("{published}#{key}/model"), runtime))
}

/// A served model that can't reach its driver's machine still drives a
/// local env: the env is served on the model's host, which the model
/// recognises by its key in the address the driver dialed.
#[test]
fn a_model_that_cannot_reach_its_driver_drives_a_local_env() {
    let scratch = Scratch::new();
    let relay = scratch.relay();
    let Some((_model, address, _published)) = contained(&scratch, &relay) else {
        eprintln!("skipped: no unprivileged network namespaces here");
        return;
    };
    assert_eq!(drive(&scratch, &address, false), "STEPS=6");

    // An env given by address is the model's to reach: it can't, and says
    // so in seconds rather than at the driver's timeout.
    let refused = drive(&scratch, &address, true);
    assert!(
        refused.contains("could not reach the env's session at 127.0.0.1:"),
        "{refused}"
    );
    let ms: u64 = refused
        .rsplit_once("AFTER=")
        .and_then(|(_, ms)| ms.trim().parse().ok())
        .unwrap_or(u64::MAX);
    assert!(ms < 10_000, "{refused}");
}
