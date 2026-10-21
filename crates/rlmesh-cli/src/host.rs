//! `rlmesh host`: v2 sessions hosted in the foreground, so participants on
//! this machine (or, listening beyond loopback, others) join instead of
//! hosting their own.

use std::io::Write;
use std::net::{Ipv4Addr, SocketAddr};

use anyhow::{Context, Result};
use rlmesh_session::address::{self, DEFAULT_SESSION, LEASE_GRACE};

use crate::cli::HostArgs;

pub async fn host(args: &HostArgs, stdout: &mut impl Write) -> Result<()> {
    let base = match args.listen {
        Some(base) => base,
        None => SocketAddr::from((Ipv4Addr::LOCALHOST, address::well_known_port()?)),
    };
    let mut placed = address::host_at(base, DEFAULT_SESSION, LEASE_GRACE)
        .await
        .with_context(|| format!("could not listen on {base} (is a host already there?)"))?;
    writeln!(stdout, "{}", placed.address.base)?;
    stdout.flush()?;
    shutdown_signal().await?;
    placed.close().await?;
    Ok(())
}

#[cfg(unix)]
async fn shutdown_signal() -> std::io::Result<()> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut term = signal(SignalKind::terminate())?;
    tokio::select! {
        result = tokio::signal::ctrl_c() => result?,
        _ = term.recv() => {}
    }
    Ok(())
}

#[cfg(not(unix))]
async fn shutdown_signal() -> std::io::Result<()> {
    tokio::signal::ctrl_c().await
}
