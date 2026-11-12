//! An embedded host offers boards the datagram path by default (UDP on its
//! base port + 1): a session without boards sees no change, and a UDP port
//! that won't open leaves the host running on TCP.

use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use rlmesh_session::address::{Address, host_at};
use rlmesh_session::host::BoardData;
use rlmesh_session::sdk;
use rlmesh_session::transport::free_local_base;
use rlmesh_session::wire::DataPath;

const WAIT: Duration = Duration::from_secs(10);

const EYE: &str = r#"{
    "eyes": {"kind": "joints", "names": ["pan", "tilt"], "fields": ["pos"], "hz": 50,
             "command": "position", "safe": "hold", "hold_max_ms": 100}
}"#;

fn udp_taken(at: SocketAddr) -> bool {
    UdpSocket::bind(at).is_err_and(|e| e.kind() == std::io::ErrorKind::AddrInUse)
}

/// Waits for the host's health row of `robot`, as of its next interval.
async fn row(host: &sdk::EmbeddedHost, robot: &str) -> rlmesh_session::ParticipantHealth {
    let deadline = Instant::now() + WAIT;
    loop {
        let found = host
            .health()
            .await
            .into_iter()
            .find(|r| r.participant.as_str() == robot && r.heartbeats > 0);
        if let Some(row) = found {
            return row;
        }
        assert!(Instant::now() < deadline, "no health row for {robot}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[test]
fn the_default_is_udp() {
    assert_eq!(BoardData::default(), BoardData::Udp);
    assert!(!BoardData::default().strict());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_embedded_host_offers_udp_and_a_session_without_boards_sees_no_change() {
    let host = sdk::EmbeddedHost::start(
        &Address::session(":0").expect("address"),
        Duration::from_millis(250),
    )
    .await
    .expect("host");
    let base = host.address().base;
    let udp = SocketAddr::new(base.ip(), base.port() + 1);
    assert!(udp_taken(udp), "the datagram path listens on udp/{udp}");

    // A robot from the SDK does not take the path: its parts ride TCP, and
    // its health row says so as it did before.
    let robot = sdk::connect_robot(host.address(), "eye", EYE, WAIT)
        .await
        .expect("robot");
    let row = row(&host, "eye").await;
    assert_eq!(row.data_path, DataPath::Session);
    assert_eq!(row.data_path.name(), "tcp");
    assert!(row.datagram_board.is_none() && row.datagram_host.is_none());
    assert!(row.path_time.is_none() && row.datagram_relay.is_none());
    drop(robot);
    host.close().await.expect("close");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_udp_port_that_wont_open_leaves_the_host_on_tcp() {
    let base = free_local_base(Ipv4Addr::LOCALHOST.into()).expect("ports");
    let at = SocketAddr::new(base.ip(), base.port() + 1);
    let _taken = UdpSocket::bind(at).expect("hold udp/base+1");

    assert!(
        BoardData::Udp
            .open(base)
            .await
            .expect("udp degrades")
            .is_none(),
        "no path, no error"
    );
    assert!(BoardData::Tcp.open(base).await.expect("tcp").is_none());
    let strict = BoardData::UdpOnly
        .open(base)
        .await
        .expect_err("udp-only fails");
    assert_eq!(strict.kind(), std::io::ErrorKind::AddrInUse, "{strict}");

    let mut placed = host_at(base, "taken", Duration::from_millis(250), false)
        .await
        .expect("hosts on TCP");
    let robot = sdk::connect_robot(&placed.address, "eye", EYE, WAIT)
        .await
        .expect("a robot joins");
    drop(robot);
    placed.close().await.expect("close");
}
