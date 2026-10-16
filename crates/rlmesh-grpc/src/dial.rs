//! One connector for every client transport: plaintext HTTP/2 over TCP, a Unix
//! socket, or TLS (HTTP/2 negotiated by ALPN `h2`) for an `https://` target.
//!
//! The env and model clients both dial through one function, so every language
//! inherits the same trust rules and the same failure text.

use std::error::Error as StdError;
use std::path::Path;
use std::sync::Arc;

use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};

use crate::error::{ClientError, Error, TransportError};
use crate::helpers::EnvConnectTarget;

/// Which certificate authorities a TLS dial trusts.
#[derive(Clone, Default)]
pub enum TlsTrust {
    /// The system trust store, falling back to the bundled web PKI roots
    /// when the system has none.
    #[default]
    System,
    /// Only the certificates of this PEM bundle (like `curl --cacert`).
    Bundle(Arc<[u8]>),
}

impl std::fmt::Debug for TlsTrust {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => f.write_str("System"),
            Self::Bundle(pem) => write!(f, "Bundle({} bytes)", pem.len()),
        }
    }
}

impl TlsTrust {
    /// Read and check the PEM bundle at `path`: it must hold at least one
    /// certificate and nothing that fails to parse. A failure is
    /// [`ClientError::InvalidTlsCaFile`], before any dial.
    pub fn from_ca_file(path: &Path) -> Result<Self, Error> {
        use rustls::pki_types::{CertificateDer, pem::PemObject};
        let invalid = |reason: String| -> Error {
            ClientError::InvalidTlsCaFile(format!("tls_ca_file '{}' {reason}", path.display()))
                .into()
        };
        let pem =
            std::fs::read(path).map_err(|error| invalid(format!("cannot be read: {error}")))?;
        let mut count = 0usize;
        for certificate in CertificateDer::pem_slice_iter(&pem) {
            certificate.map_err(|error| {
                invalid(format!("is not a valid PEM certificate bundle: {error}"))
            })?;
            count += 1;
        }
        if count == 0 {
            return Err(invalid("holds no PEM certificate".to_string()));
        }
        Ok(Self::Bundle(pem.into()))
    }
}

/// Dial `target` and return the HTTP/2 channel: TLS for an `https://` target
/// (trusting `trust`), a Unix socket for `unix://`, plaintext otherwise.
///
/// A failed TLS handshake (an untrusted or mismatched certificate, a
/// plaintext server, no `h2` in ALPN) reads `TLS handshake with host:port
/// failed: <reason>` ([`tls_failure`]); other failures keep the transport's
/// own text.
pub(crate) async fn dial(target: &EnvConnectTarget, trust: &TlsTrust) -> Result<Channel, Error> {
    let endpoint = crate::configure_endpoint(
        Endpoint::from_shared(target.endpoint().to_string())
            .map_err(|e| TransportError::InvalidAddress(e.to_string()))?,
    );

    #[cfg(unix)]
    if let Some(socket_path) = target.unix_path().cloned() {
        use hyper_util::rt::TokioIo;
        use tokio::net::UnixStream;
        use tower::service_fn;
        return endpoint
            .connect_with_connector(service_fn(move |_: tonic::transport::Uri| {
                let socket_path = socket_path.clone();
                async move { UnixStream::connect(socket_path).await.map(TokioIo::new) }
            }))
            .await
            .map_err(|e| TransportError::ConnectFailed(e.to_string()).into());
    }

    let endpoint = match target.tls_server() {
        None => endpoint,
        Some((host, _)) => {
            let base = ClientTlsConfig::new().domain_name(host.to_string());
            match trust {
                TlsTrust::Bundle(pem) => {
                    endpoint.tls_config(base.ca_certificate(Certificate::from_pem(pem)))
                }
                TlsTrust::System => endpoint
                    .clone()
                    .tls_config(base.clone().with_native_roots().with_webpki_roots())
                    // No system roots on this host: the bundled web PKI roots.
                    .or_else(|_| endpoint.tls_config(base.with_webpki_roots())),
            }
            .map_err(|e| TransportError::ConnectFailed(tls_failure(target, &e)))?
        }
    };
    endpoint.connect().await.map_err(|e| {
        let message = if target.is_tls() {
            tls_failure(target, &e)
        } else {
            e.to_string()
        };
        TransportError::ConnectFailed(message).into()
    })
}

/// The one text every language shows for a failed TLS dial of `target`:
/// `TLS handshake with host:port failed: <reason>` when the handshake itself
/// failed, else the transport's own text (a refused TCP connect, say).
pub fn tls_failure(target: &EnvConnectTarget, error: &(dyn StdError + 'static)) -> String {
    let authority = target
        .display_address()
        .strip_prefix("https://")
        .unwrap_or(target.display_address());
    let mut source: Option<&(dyn StdError + 'static)> = Some(error);
    while let Some(current) = source {
        let rustls_error = current.downcast_ref::<rustls::Error>().or_else(|| {
            current
                .downcast_ref::<std::io::Error>()
                .and_then(|io| io.get_ref())
                .and_then(|inner| inner.downcast_ref::<rustls::Error>())
        });
        if let Some(rustls_error) = rustls_error {
            return format!("TLS handshake with {authority} failed: {rustls_error}");
        }
        if current.to_string().contains("HTTP/2 was not negotiated") {
            return format!(
                "TLS handshake with {authority} failed: the server did not negotiate HTTP/2 \
                 (ALPN h2)"
            );
        }
        source = current.source();
    }
    error.to_string()
}

/// What a client sends and trusts when it dials: the endpoint token (sent as
/// `authorization: Bearer <token>`; empty sends none) and the TLS trust of an
/// `https://` target. `Debug` never prints the token.
#[derive(Clone, Default)]
pub struct Credentials {
    /// The endpoint token (an API key); empty sends none.
    pub token: String,
    /// The certificate authorities an `https://` dial trusts.
    pub trust: TlsTrust,
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials")
            .field(
                "token",
                &if self.token.is_empty() {
                    "<none>"
                } else {
                    "<redacted>"
                },
            )
            .field("trust", &self.trust)
            .finish()
    }
}

impl Credentials {
    /// Credentials sending `token` and trusting the system roots.
    pub fn token(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            trust: TlsTrust::System,
        }
    }
}
