use std::io::{BufRead, BufReader};
use std::net::{Ipv4Addr, SocketAddr};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::Duration;

use super::*;
use crate::address::{HOST_IDENTITY_VAR, STATE_DIR_VAR, SessionAddress, host_at, launch_identity};
use crate::wire::{PriorityClass, Role};

fn mode(path: &Path) -> u32 {
    std::fs::metadata(path).expect("stat").permissions().mode() & 0o777
}

fn loopback(port: u16) -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, port))
}

fn fresh_key() -> AddressKey {
    launch_identity().expect("identity").1
}

#[test]
fn only_this_users_private_key_files_are_trusted() {
    assert!(trusted(1000, 0o100600, 1000).is_ok());
    assert!(trusted(1000, 0o100400, 1000).is_ok());
    let other = trusted(0, 0o100600, 1000).expect_err("another user's");
    assert!(other.contains("owned by uid 0"), "{other}");
    for lax in [0o100640, 0o100604, 0o100620, 0o100666] {
        let refused = trusted(1000, lax, 1000).expect_err("readable by others");
        assert!(refused.contains("lets others read or write"), "{refused}");
    }
}

#[test]
fn a_key_file_is_private_while_its_host_runs_and_gone_after() {
    let scratch = tempfile::tempdir().unwrap();
    let dir = scratch.path().join("hosts");
    let key = fresh_key();
    let file = KeyFile::write_in(&dir, loopback(5587), &key).expect("written");
    assert_eq!(file.path(), dir.join("127.0.0.1-5587.key"));
    assert_eq!(mode(&dir), 0o700);
    assert_eq!(mode(file.path()), 0o600);
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        1,
        "no partial file left"
    );
    let filed = filed_in(&dir, loopback(5587))
        .expect("trusted")
        .expect("found");
    assert_eq!(filed.key, key);
    assert!(filed_in(&dir, loopback(5588)).unwrap().is_none());
    let path = file.path().to_path_buf();
    drop(file);
    assert!(!path.exists(), "the host's close removes its key");
    assert!(filed_in(&dir, loopback(5587)).unwrap().is_none());
}

#[test]
fn a_key_file_its_host_outlived_is_passed_over() {
    let scratch = tempfile::tempdir().unwrap();
    let dir = scratch.path().join("hosts");
    // As a killed host leaves it: whole, private, and held by no one.
    crate::address::write_private(
        &dir.join("127.0.0.1-5587.key"),
        fresh_key().to_string().as_bytes(),
    )
    .unwrap();
    assert!(filed_in(&dir, loopback(5587)).unwrap().is_none());
}

#[test]
fn a_host_closing_leaves_its_successors_key_in_place() {
    let scratch = tempfile::tempdir().unwrap();
    let dir = scratch.path().join("hosts");
    let (first, second) = (fresh_key(), fresh_key());
    let before = KeyFile::write_in(&dir, loopback(5587), &first).unwrap();
    let after = KeyFile::write_in(&dir, loopback(5587), &second).unwrap();
    drop(before);
    let filed = filed_in(&dir, loopback(5587))
        .unwrap()
        .expect("the successor's");
    assert_eq!(filed.key, second);
    drop(after);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
}

#[test]
fn a_linked_or_lax_key_file_is_refused_not_read() {
    let scratch = tempfile::tempdir().unwrap();
    let dir = scratch.path().join("hosts");
    let held = KeyFile::write_in(&dir, loopback(5587), &fresh_key()).unwrap();
    std::os::unix::fs::symlink(held.path(), dir.join("127.0.0.1-5588.key")).unwrap();
    let linked = filed_in(&dir, loopback(5588)).expect_err("a link");
    assert!(linked.to_string().contains("a link"), "{linked}");

    std::fs::set_permissions(held.path(), std::fs::Permissions::from_mode(0o640)).unwrap();
    let lax = filed_in(&dir, loopback(5587)).expect_err("group-readable");
    assert!(lax.to_string().contains("mode 640"), "{lax}");
    assert!(
        !lax.to_string().contains(&fresh_key().to_string()[..8]),
        "never the key"
    );
}

// The processes of `a_loopback_host_leaves_its_key_for_this_users_other_processes`.

/// A host child: binds here, prints its port, hosts until its stdin closes.
const HOST: &str = "RLMESH_TEST_KEYS_HOST";
/// With [`HOST`]: host `insecure`.
const INSECURE: &str = "RLMESH_TEST_KEYS_INSECURE";
/// A join child: joins the loopback port here with no key.
const JOIN: &str = "RLMESH_TEST_KEYS_JOIN";
/// With [`JOIN`]: its timeout, in milliseconds.
const JOIN_MS: &str = "RLMESH_TEST_KEYS_JOIN_MS";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[allow(clippy::print_stdout)]
async fn host_child() {
    let Ok(bind) = std::env::var(HOST) else {
        return;
    };
    let insecure = std::env::var_os(INSECURE).is_some();
    let mut placed = host_at(
        bind.parse().expect("bind"),
        "keys",
        Duration::from_millis(250),
        insecure,
    )
    .await
    .expect("host");
    println!("PORT={}", placed.address.base.port());
    let _ = tokio::task::spawn_blocking(|| {
        let mut line = String::new();
        let _ = std::io::stdin().read_line(&mut line);
    })
    .await;
    placed.close().await.expect("close");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[allow(clippy::print_stdout)]
async fn join_child() {
    let Ok(port) = std::env::var(JOIN) else {
        return;
    };
    let timeout = std::env::var(JOIN_MS)
        .ok()
        .and_then(|ms| ms.parse().ok())
        .map_or(Duration::from_secs(10), Duration::from_millis);
    let address = SessionAddress::new(loopback(port.parse().expect("port")), "keys").unwrap();
    let started = std::time::Instant::now();
    let joined = address
        .join(
            "joiner",
            vec![Role::Observer],
            PriorityClass::Scripted,
            timeout,
        )
        .await;
    let ms = started.elapsed().as_millis();
    match joined {
        Ok(participant) => {
            println!("JOINED");
            participant.close().await;
        }
        // Never a key: these name the address and the file, not what it holds.
        Err(err) => println!("ERR={err} AFTER={ms}"),
    }
}

/// The test binary, as one of the children above, in the scratch
/// directories.
fn child(test: &str, scratch: &Scratch) -> Command {
    let mut command = Command::new(std::env::current_exe().expect("test binary"));
    command
        .args([
            "--exact",
            &format!("address::host_keys::tests::{test}"),
            "--nocapture",
        ])
        .env(crate::address::hosting::RUNTIME_DIR_VAR, &scratch.runtime)
        .env(STATE_DIR_VAR, &scratch.state)
        .env_remove(HOST_IDENTITY_VAR)
        .env_remove(HOST)
        .env_remove(INSECURE)
        .env_remove(JOIN);
    command
}

struct Scratch {
    _dir: tempfile::TempDir,
    runtime: PathBuf,
    state: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("scratch");
        let (runtime, state) = (dir.path().join("runtime"), dir.path().join("state"));
        std::fs::create_dir(&state).expect("state dir");
        Self {
            _dir: dir,
            runtime,
            state,
        }
    }

    fn key_file(&self, port: u16) -> PathBuf {
        self.runtime.join("hosts").join(file_name(loopback(port)))
    }
}

/// A host child, stopped (its stdin closed, then killed) when dropped.
struct Host {
    child: Child,
    stdin: Option<ChildStdin>,
    port: u16,
}

impl Host {
    fn start(scratch: &Scratch, bind: &str, setup: impl FnOnce(&mut Command)) -> Self {
        let mut command = child("host_child", scratch);
        command
            .env(HOST, bind)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        setup(&mut command);
        let mut child = command.spawn().expect("spawn the host");
        let stdin = child.stdin.take();
        let mut lines = BufReader::new(child.stdout.take().expect("stdout")).lines();
        let port = loop {
            let line = lines
                .next()
                .expect("the host printed its port")
                .expect("read");
            if let Some(port) = line.strip_prefix("PORT=") {
                break port.trim().parse().expect("a port");
            }
        };
        // The rest of its output, drained so its writes never meet a closed pipe.
        std::thread::spawn(move || lines.for_each(drop));
        Self { child, stdin, port }
    }

    /// Closes the host and waits for its process to end.
    fn close(mut self) {
        self.stdin.take();
        let status = self.child.wait().expect("wait");
        assert!(status.success(), "the host child failed");
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        if self.stdin.take().is_some() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// What a keyless join to `port` printed: `JOINED`, or the error and how
/// long it took.
fn join(scratch: &Scratch, port: u16, timeout_ms: u64) -> String {
    let output = child("join_child", scratch)
        .env(JOIN, port.to_string())
        .env(JOIN_MS, timeout_ms.to_string())
        .output()
        .expect("run the join");
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.lines()
        .find(|line| *line == "JOINED" || line.starts_with("ERR="))
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

fn took_ms(outcome: &str) -> u64 {
    outcome
        .rsplit_once("AFTER=")
        .and_then(|(_, ms)| ms.trim().parse().ok())
        .unwrap_or(u64::MAX)
}

/// A keyless join by another process of this user: what the quickstart's
/// `RemoteEnv("127.0.0.1:5587/cartpole")` beside its `EnvServer` does.
#[test]
fn a_loopback_host_leaves_its_key_for_this_users_other_processes() {
    let scratch = Scratch::new();
    let first = Host::start(&scratch, "127.0.0.1:0", |_| {});
    let second = Host::start(&scratch, "127.0.0.1:0", |_| {});
    let (first_file, second_file) = (scratch.key_file(first.port), scratch.key_file(second.port));
    assert_eq!(mode(&first_file), 0o600);
    assert_eq!(mode(&second_file), 0o600);
    assert_eq!(mode(&scratch.runtime.join("hosts")), 0o700);

    // Two hosts side by side, each joined by address alone.
    assert_eq!(join(&scratch, first.port, 10_000), "JOINED");
    assert_eq!(join(&scratch, second.port, 10_000), "JOINED");

    // A key file others could read is refused, at once, and says why.
    std::fs::set_permissions(&second_file, std::fs::Permissions::from_mode(0o644)).unwrap();
    let lax = join(&scratch, second.port, 10_000);
    assert!(lax.contains("mode 644"), "{lax}");
    assert!(took_ms(&lax) < 2_000, "{lax}");
    std::fs::set_permissions(&second_file, std::fs::Permissions::from_mode(0o600)).unwrap();

    // A host's close removes its key and no other.
    first.close();
    assert!(!first_file.exists());
    assert!(second_file.exists());

    // A launched host's key is its launcher's: it leaves none.
    let (identity, _) = launch_identity().unwrap();
    let launched = Host::start(&scratch, "127.0.0.1:0", |command| {
        command.env(HOST_IDENTITY_VAR, identity.as_str());
    });
    assert!(!scratch.key_file(launched.port).exists());

    // A stale file whose host is gone is passed over: the join reports the
    // missing key, as with no file at all.
    let stale = scratch.key_file(launched.port);
    crate::address::write_private(&stale, fresh_key().to_string().as_bytes()).unwrap();
    let passed_over = join(&scratch, launched.port, 2_000);
    assert!(
        passed_over.contains("no key for the session"),
        "{passed_over}"
    );

    // One whose writer still runs but names another host's key ends soon,
    // saying the key on file is not this host's.
    std::fs::remove_file(&stale).unwrap();
    let held = KeyFile::write_in(
        &scratch.runtime.join("hosts"),
        loopback(launched.port),
        &fresh_key(),
    )
    .unwrap();
    let wrong = join(&scratch, launched.port, 20_000);
    assert!(
        wrong.contains("the key on file is not this host's"),
        "{wrong}"
    );
    assert!(took_ms(&wrong) < 5_000, "{wrong}");
    drop(held);
    launched.close();

    // Nor does a host beyond loopback.
    let wide = Host::start(&scratch, "0.0.0.0:0", |command| {
        command.env(INSECURE, "1");
    });
    let left: Vec<_> = std::fs::read_dir(scratch.runtime.join("hosts"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(
        left,
        [second_file.file_name().unwrap().to_owned()],
        "{left:?}"
    );
    wide.close();
    second.close();
    assert_eq!(
        std::fs::read_dir(scratch.runtime.join("hosts"))
            .unwrap()
            .count(),
        0
    );
}
