//! `rlmesh-gateway`: serves browsers a session until interrupted.
//!
//! ```text
//! rlmesh-gateway --connect 127.0.0.1:17600 --session-file session.id --static examples/web/dist
//! ```
//!
//! It logs to stderr (`RUST_LOG`) and prints `READY <url>` once it serves.

// The READY line is for the scripts that wait on it.
#![allow(clippy::print_stderr)]

use std::sync::Arc;

use axum::serve::ListenerExt;
use clap::Parser;
use rlmesh_gateway::{AppState, Args, Config, Gateway, router, tls};
use rlmesh_session::{Transport, TransportConfig};
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
    let transport =
        TransportConfig::per_class_tcp(None, Some(config.connect)).map_err(|e| e.to_string())?;
    let transport = Transport::open(&transport)
        .await
        .map_err(|e| e.to_string())?;
    let gateway = Gateway::open(
        transport,
        config.tenant.clone(),
        config.session,
        config.max_priority,
    )
    .await
    .map_err(|e| format!("attach to session {}: {e}", config.session))?;
    let state = AppState {
        gateway,
        policy: Arc::new(config.policy.clone()),
    };
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
        token = if policy.token.is_some() { "required" } else { "off" },
        origins = ?policy.origins,
        max_message_bytes = policy.max_message_bytes,
        max_subscriptions = policy.max_subscriptions,
        max_priority = ?config.max_priority,
        "serving session {} of {}",
        config.session,
        config.connect
    );
    // The demo scripts wait for this line.
    eprintln!("READY {scheme}://{host}:{}/", config.port);
    let served = match tls {
        Some(tls) => {
            let listener = tls::TlsListener::new(tcp, tls).map_err(|e| e.to_string())?;
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown())
                .await
        }
        None => {
            let listener = tcp.tap_io(|tcp| {
                let _ = tcp.set_nodelay(true);
            });
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown())
                .await
        }
    };
    served.map_err(|e| e.to_string())
}

async fn shutdown() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        if let Ok(mut term) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}
