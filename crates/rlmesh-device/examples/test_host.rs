//! A session host for `mise run test:esp32:host` and a board on the LAN:
//! the default session of tenant `local` on free ports of `[ip]` (default
//! 127.0.0.1; a LAN address for a board), printing its address with its key
//! (as `rlmesh host` does), until killed:
//!
//! ```sh
//! cargo run -p rlmesh-device --example test_host -- 192.168.1.20
//! ```
//!
//! Its links are not gated: an embedded host's are (Zenoh `usrpwd`), which
//! zenoh-pico can't present (until the host takes ungated LAN links, as its
//! design says). Session traffic is still authenticated and sealed by the
//! handshake. For tests only.

#![allow(clippy::print_stdout)]

use std::io::Write;

use rlmesh_session::address::{AddressKey, DEFAULT_SESSION, TENANT, session_id};
use rlmesh_session::transport::open_local_listeners;
use rlmesh_session::wire::TenantId;
use rlmesh_session::{HostConfig, SessionHost};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    let ip = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1".into())
        .parse()
        .expect("an IP address");
    let (transport, base) = open_local_listeners(ip).await.expect("listeners");
    let config = HostConfig::new(
        TenantId::new(TENANT).expect("tenant"),
        session_id(DEFAULT_SESSION),
    );
    let key = AddressKey::of(&config.identity);
    let _host = SessionHost::start(transport, config).await.expect("host");
    println!("{base}#{key}");
    std::io::stdout().flush().expect("stdout");
    std::future::pending::<()>().await;
}
