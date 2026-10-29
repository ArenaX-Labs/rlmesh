//! `rlmesh-gateway`: serves browsers a session until interrupted.
//!
//! ```text
//! RLMESH_GATEWAY_TOKEN=… rlmesh-gateway --address 127.0.0.1:17600/<session>#<key> --static examples/web/dist
//! ```
//!
//! It logs to stderr (`RUST_LOG`) and prints `READY <url>` once it serves
//! (never with a token: the page's comes in its fragment, `#t=…`, from
//! whoever minted it).

// The READY line is for the scripts that wait on it.
#![allow(clippy::print_stderr)]

use std::sync::Arc;

use axum::serve::ListenerExt;
use clap::Parser;
use rlmesh_session::gateway::{AppState, Args, Config, Gateway, router, tls};
use tracing::{error, info};

#[tokio::main]
async fn main() -> std::process::ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    match run(Args::parse()).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            error!("{err}");
            std::process::ExitCode::FAILURE
        }
    }
}

async fn run(args: Args) -> Result<(), String> {
    let config = Config::from_args(args, std::env::var("RLMESH_GATEWAY_TOKEN").ok())?;
    let tls = match &config.tls {
        Some((cert, key)) => Some(tls::load_config(cert, key)?),
        None => None,
    };
    // As every participant joins: a client of the host, through the
    // password gate of a loopback host, and without one to a host that
    // gates no link (rlmeshd), which the handshake alone admits.
    let address = &config.address;
    let wait = std::time::Duration::from_secs(10);
    let transport = address
        .link(tokio::time::Instant::now() + wait, wait)
        .await
        .map_err(|e| e.to_string())?;
    let credential = address.credential().map_err(|e| e.to_string())?;
    let gateway = Gateway::open(
        transport,
        address.tenant.clone(),
        address.session,
        credential,
        config.max_priority,
    )
    .await
    .map_err(|e| format!("attach to session {}: {e}", address.session))?;
    let state = AppState {
        gateway,
        policy: Arc::new(config.policy.clone()),
    };
    let ended = state.gateway.ended();
    let app = router(state, config.static_dir.as_deref());
    let tcp = tokio::net::TcpListener::bind((config.bind, config.port))
        .await
        .map_err(|e| format!("bind {}:{}: {e}", config.bind, config.port))?;
    let scheme = if tls.is_some() { "https" } else { "http" };
    let host = if config.bind.is_ipv6() {
        format!("[{}]", config.bind)
    } else {
        config.bind.to_string()
    };
    let policy = &config.policy;
    info!(
        page_token = policy.page.is_some(),
        browsers = ?policy.browsers,
        origins = ?policy.origins,
        max_message_bytes = policy.max_message_bytes,
        max_subscriptions = policy.max_subscriptions,
        max_priority = ?config.max_priority,
        "serving session {} of {}",
        address.session,
        address.base
    );
    // The demo scripts wait for this line.
    eprintln!("READY {scheme}://{host}:{}/", config.port);
    let served = match tls {
        Some(tls) => {
            let listener = tls::TlsListener::new(tcp, tls).map_err(|e| e.to_string())?;
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown(ended))
                .await
        }
        None => {
            let listener = tcp.tap_io(|tcp| {
                let _ = tcp.set_nodelay(true);
            });
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown(ended))
                .await
        }
    };
    served.map_err(|e| e.to_string())
}

/// Until interrupted, or until the session ends: this gateway serves one
/// session, so it exits with it (after its sockets got their close frames).
async fn shutdown(mut ended: tokio::sync::watch::Receiver<bool>) {
    let session_ended = async {
        if ended.wait_for(|e| *e).await.is_ok() {
            info!("the session ended; exiting");
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        if let Ok(mut term) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
                () = session_ended => {}
            }
            return;
        }
    }
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        () = session_ended => {}
    }
}
