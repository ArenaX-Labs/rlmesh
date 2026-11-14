//! Optional TLS (`wss://`, `https://`) with rustls (ring provider), as an axum `Listener`.
//!
//! Handshakes run in their own tasks with a timeout, so one slow or stalled client cannot hold up the
//! accept loop; finished streams are handed to axum through a channel.

use rustls::ServerConfig;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::io;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_rustls::TlsAcceptor;
use tokio_rustls::server::TlsStream;
use tracing::{debug, warn};

/// How long one client may take to finish its TLS handshake.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Load a PEM certificate chain and private key into a rustls server config (TLS 1.2 and 1.3).
pub fn load_config(cert: &Path, key: &Path) -> Result<Arc<ServerConfig>, String> {
    let certs = CertificateDer::pem_file_iter(cert)
        .map_err(|e| format!("{}: {e}", cert.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{}: {e}", cert.display()))?;
    if certs.is_empty() {
        return Err(format!("{}: no certificates", cert.display()));
    }
    let key = PrivateKeyDer::from_pem_file(key).map_err(|e| format!("{}: {e}", key.display()))?;
    let cfg =
        ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .map_err(|e| e.to_string())?
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|e| e.to_string())?;
    Ok(Arc::new(cfg))
}

/// TLS on a TCP listener, as an axum `Listener`.
pub struct TlsListener {
    rx: mpsc::Receiver<(TlsStream<TcpStream>, SocketAddr)>,
    local: SocketAddr,
}

impl TlsListener {
    /// Accepts on `tcp` and handshakes with `cfg`, each client in its own
    /// task.
    pub fn new(tcp: TcpListener, cfg: Arc<ServerConfig>) -> io::Result<Self> {
        let local = tcp.local_addr()?;
        let (tx, rx) = mpsc::channel(64);
        let acceptor = TlsAcceptor::from(cfg);
        tokio::spawn(async move {
            loop {
                let (s, addr) = match tcp.accept().await {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("tls accept: {e}");
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                };
                let _ = s.set_nodelay(true);
                let (acceptor, tx) = (acceptor.clone(), tx.clone());
                tokio::spawn(async move {
                    match tokio::time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(s)).await {
                        Ok(Ok(t)) => {
                            let _ = tx.send((t, addr)).await;
                        }
                        Ok(Err(e)) => debug!(%addr, "tls handshake: {e}"),
                        Err(_) => debug!(%addr, "tls handshake timed out"),
                    }
                });
            }
        });
        Ok(Self { rx, local })
    }
}

impl axum::serve::Listener for TlsListener {
    type Io = TlsStream<TcpStream>;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self.rx.recv().await {
            Some(v) => v,
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        Ok(self.local)
    }
}
