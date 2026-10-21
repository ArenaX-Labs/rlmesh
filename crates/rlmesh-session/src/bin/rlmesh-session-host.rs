//! `rlmesh-session-host`: runs a session host until interrupted.
//!
//! ```text
//! rlmesh-session-host --listen-base 0.0.0.0:7447            # per-class TCP on 7447-7450
//! rlmesh-session-host --listen unixsock-stream//tmp/s.sock  # one shared session
//! ```
//!
//! It hosts the default session unless `--session` names another (a name or
//! a UUID), prints the session id on stdout (and writes it to `--session-file`),
//! logs to stderr (`RUST_LOG`), and shuts down cleanly on SIGINT or SIGTERM.

#![allow(clippy::print_stdout)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use rlmesh_session::wire::TenantId;
use rlmesh_session::{HostConfig, SessionError, SessionHost, Transport, TransportConfig};
use tracing::info;

/// Runs an RLMesh session host.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// The tenant namespace.
    #[arg(long, default_value = "local")]
    tenant: String,
    /// The session: a name (`bench-3`) or a UUID.
    #[arg(long, default_value = rlmesh_session::address::DEFAULT_SESSION)]
    session: String,
    /// Per-class TCP: class i listens on this port + i (control, realtime,
    /// data, bulk).
    #[arg(long, conflicts_with = "listen")]
    listen_base: Option<SocketAddr>,
    /// One shared session listening on these Zenoh endpoints.
    #[arg(long)]
    listen: Vec<String>,
    /// How long a lease holder may stay silent before its lease lapses.
    #[arg(long, default_value_t = 250)]
    lease_grace_ms: u64,
    /// Log each participant's link health this often (0: never).
    #[arg(long, default_value_t = 0)]
    health_log_ms: u64,
    /// Write the session id here once the host is up.
    #[arg(long)]
    session_file: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), SessionError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    let tenant = TenantId::new(args.tenant)?;
    let session = rlmesh_session::address::session_id(&args.session);
    let transport = match args.listen_base {
        Some(base) => TransportConfig::per_class_tcp(Some(base), None)?,
        None if args.listen.is_empty() => {
            return Err(SessionError::Config(
                "pass --listen-base or --listen".into(),
            ));
        }
        None => {
            let listen: Vec<&str> = args.listen.iter().map(String::as_str).collect();
            TransportConfig::shared(&listen, &[])
        }
    };
    let transport = Transport::open(&transport).await?;
    let mut config = HostConfig::new(tenant, session);
    config.lease_grace = Duration::from_millis(args.lease_grace_ms.max(1));
    config.health_log = (args.health_log_ms > 0).then(|| Duration::from_millis(args.health_log_ms));
    let host = SessionHost::start(transport.clone(), config).await?;
    println!("{session}");
    if let Some(path) = &args.session_file {
        std::fs::write(path, session.to_string())?;
    }
    shutdown_signal().await?;
    info!("shutting down");
    host.shutdown().await;
    transport.close().await?;
    Ok(())
}

#[cfg(unix)]
async fn shutdown_signal() -> Result<(), SessionError> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut term = signal(SignalKind::terminate())?;
    tokio::select! {
        result = tokio::signal::ctrl_c() => result?,
        _ = term.recv() => {}
    }
    Ok(())
}

#[cfg(not(unix))]
async fn shutdown_signal() -> Result<(), SessionError> {
    Ok(tokio::signal::ctrl_c().await?)
}
