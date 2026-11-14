//! A session host listens on one TCP port, and on UDP at the same number
//! when it offers boards the datagram path; a participant in another
//! process joins with only that port reachable, every traffic class on a
//! connection of its own to it.
//!
//! The host runs in a child process of this test binary: a join in the
//! process that hosts rides the host's sessions instead of a link, and the
//! child's sockets are the host's alone.

#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader};
use std::net::{Ipv4Addr, SocketAddr};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use rlmesh_session::address::{Address, HostingOptions, host_at_with, write_private};
use rlmesh_session::host::BoardData;
use rlmesh_session::wire::{PriorityClass, Role};

/// Set in the child that hosts: where it leaves its address.
const CHILD: &str = "RLMESH_TEST_ONE_PORT_CHILD";
const WAIT: Duration = Duration::from_secs(20);

/// This process's TCP listeners and UDP sockets, by local port.
#[allow(unsafe_code)]
fn bound_ports() -> (Vec<u16>, Vec<u16>) {
    let (mut tcp, mut udp) = (Vec::new(), Vec::new());
    for entry in std::fs::read_dir("/proc/self/fd").expect("fds").flatten() {
        let Some(fd) = entry.file_name().to_str().and_then(|n| n.parse().ok()) else {
            continue;
        };
        // SAFETY: borrowed only for this iteration; nothing here closes it.
        let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) };
        let socket = socket2::SockRef::from(&fd);
        let Some(port) = socket
            .local_addr()
            .ok()
            .and_then(|a| a.as_socket())
            .map(|a| a.port())
        else {
            continue;
        };
        match socket.r#type() {
            Ok(socket2::Type::STREAM) if socket.is_listener().unwrap_or(false) => tcp.push(port),
            Ok(socket2::Type::DGRAM) => udp.push(port),
            _ => {}
        }
    }
    (tcp, udp)
}

/// The child: hosts with the datagram path on, checks its own sockets, and
/// leaves its port and key in a private file until its stdin closes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_child() {
    let Some(out) = std::env::var_os(CHILD) else {
        return;
    };
    let options = HostingOptions {
        board_data: Some(BoardData::Udp),
    };
    let base = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
    let mut placed = host_at_with(
        base,
        "one-port",
        Duration::from_millis(250),
        false,
        &options,
    )
    .await
    .expect("host");
    let port = placed.address.base.port();
    let (tcp, udp) = bound_ports();
    assert_eq!(tcp, [port], "TCP listeners");
    assert_eq!(udp, [port], "UDP sockets");
    let key = placed.address.key.clone().expect("a hosted key");
    // A key is a secret: it reaches the parent in a 0600 file, never output.
    write_private(
        std::path::Path::new(&out),
        format!("{port} {key}").as_bytes(),
    )
    .expect("address file");
    println!("hosting");
    let _ = std::io::stdin().lock().read_line(&mut String::new());
    placed.close().await.expect("close");
}

/// Forwards `listen` to `to`, counting the connections it accepts.
async fn relay(listen: tokio::net::TcpListener, to: SocketAddr, accepted: Arc<AtomicUsize>) {
    while let Ok((mut from, _)) = listen.accept().await {
        accepted.fetch_add(1, Ordering::Relaxed);
        tokio::spawn(async move {
            if let Ok(mut to) = tokio::net::TcpStream::connect(to).await {
                let _ = tokio::io::copy_bidirectional(&mut from, &mut to).await;
            }
        });
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_host_listens_on_one_port_and_a_participant_joins_through_it_alone() {
    if std::env::var_os(CHILD).is_some() {
        return;
    }
    let scratch = tempfile::tempdir().expect("scratch");
    let out = scratch.path().join("address");
    let mut child = Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", "host_child", "--nocapture"])
        .env(CHILD, &out)
        .env("RLMESH_RUNTIME_DIR", scratch.path())
        .env_remove(rlmesh_session::address::HOST_IDENTITY_VAR)
        .env_remove(rlmesh_session::host::BOARD_DATA_VAR)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn the host");
    // Read to the end, so the child's last lines never meet a closed pipe.
    let stdout = child.stdout.take().expect("stdout");
    let (hosting, hosted) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if line.contains("hosting") {
                let _ = hosting.send(());
            }
        }
    });
    if hosted.recv_timeout(WAIT).is_err() {
        let _ = child.kill();
        let status = child.wait().expect("wait");
        panic!("the host child never hosted ({status})");
    }
    let text = std::fs::read_to_string(&out).expect("address file");
    let (port, key) = text.trim().split_once(' ').expect("port and key");
    let host = SocketAddr::from((Ipv4Addr::LOCALHOST, port.parse::<u16>().expect("port")));

    // Only one port reaches the host: a relay of it, at another number.
    let listen = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("relay");
    let relayed = listen.local_addr().expect("relay addr");
    let accepted = Arc::new(AtomicUsize::new(0));
    let relaying = tokio::spawn(relay(listen, host, accepted.clone()));
    let address = Address::session(&format!("{relayed}/one-port#{key}"))
        .expect("address")
        .joining()
        .expect("joining");
    let participant = address
        .join("probe", vec![Role::Observer], PriorityClass::Scripted, WAIT)
        .await
        .expect("joins through the one port");
    let view = participant
        .fetch_authority()
        .await
        .expect("a control query");
    assert!(
        view.participant(participant.id())
            .is_some_and(|p| p.present)
    );
    assert!(
        accepted.load(Ordering::Relaxed) >= 4,
        "one connection per traffic class, all on the one port"
    );
    participant.close().await;
    relaying.abort();
    drop(child.stdin.take());
    let status = child.wait().expect("wait");
    assert!(status.success(), "the host child: {status}");
}
