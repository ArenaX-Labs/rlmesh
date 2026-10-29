//! A session host for `mise run test:esp32:host`: the default session of
//! tenant `local` on free loopback ports, printing its address with its key
//! (as `rlmesh host` does), until killed.
//!
//! Its links are not gated. A loopback embedded host gates them (Zenoh
//! `usrpwd`, keyed by the session), which zenoh-pico can't present; a board
//! on the LAN joins `rlmesh host --insecure` instead, whose LAN links are
//! plain. Session traffic is authenticated and sealed by the handshake
//! either way. For tests only.

#![allow(clippy::print_stdout)]

use std::io::Write;

use rlmesh_session::address::{AddressKey, DEFAULT_SESSION, TENANT, session_id};
use rlmesh_session::transport::open_local_listeners;
use rlmesh_session::wire::TenantId;
use rlmesh_session::{HostConfig, SessionHost};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
        .await
        .expect("listeners");
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
