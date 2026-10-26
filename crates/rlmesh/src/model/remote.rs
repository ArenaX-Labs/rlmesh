use std::sync::Arc;
use std::time::Duration;

use crate::{ConnectAddress, Result};

/// Options for connecting to a served model: the same as
/// [`RemoteEnvOptions`](crate::RemoteEnvOptions) without `workflow_edition`.
/// `Debug` never prints the `api_key`.
#[derive(Clone, Default)]
pub struct RemoteModelOptions {
    /// The endpoint (`tcp://host:port`, `host:port`, `https://host[:port]`
    /// or `unix:///path`).
    pub address: Option<String>,
    /// TCP host helper (default `127.0.0.1`); needs `port`.
    pub host: Option<String>,
    /// TCP port helper.
    pub port: Option<u16>,
    /// Unix socket path helper; excludes `host` and `port`.
    pub path: Option<String>,
    /// Explicit transport.
    pub transport: Option<crate::Transport>,
    /// Bound on each session's dial to the model (zero: none).
    pub connect_timeout: Option<Duration>,
    /// Bound on each `predict` and route close of sessions off this handle
    /// (zero: none).
    pub request_timeout: Option<Duration>,
    /// The key sent as `authorization: Bearer <key>` on every model request;
    /// `None` reads `RLMESH_API_KEY` for an `https://` address only, `""`
    /// sends none.
    pub api_key: Option<String>,
    /// A PEM bundle an `https://` dial trusts instead of the system roots.
    pub tls_ca_file: Option<std::path::PathBuf>,
}

impl std::fmt::Debug for RemoteModelOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteModelOptions")
            .field("address", &self.address)
            .field("host", &self.host)
            .field("port", &self.port)
            .field("path", &self.path)
            .field("transport", &self.transport)
            .field("connect_timeout", &self.connect_timeout)
            .field("request_timeout", &self.request_timeout)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("tls_ca_file", &self.tls_ca_file)
            .finish()
    }
}

impl RemoteModelOptions {
    /// Set `address`.
    pub fn address(mut self, address: impl Into<String>) -> Self {
        self.address = Some(address.into());
        self
    }

    /// Set `host`.
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    /// Set `port`.
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Set `path`.
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set `transport`.
    pub fn transport(mut self, transport: crate::Transport) -> Self {
        self.transport = Some(transport);
        self
    }

    /// Set `connect_timeout`.
    pub fn connect_timeout(mut self, timeout: impl Into<Duration>) -> Self {
        self.connect_timeout = Some(timeout.into());
        self
    }

    /// Set `request_timeout`.
    pub fn request_timeout(mut self, timeout: impl Into<Duration>) -> Self {
        self.request_timeout = Some(timeout.into());
        self
    }

    /// Set `api_key`.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set `tls_ca_file`.
    pub fn tls_ca_file(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.tls_ca_file = Some(path.into());
        self
    }
}

struct RemoteModelInner {
    address: String,
    #[cfg_attr(not(feature = "lockstep"), allow(dead_code))]
    connect_timeout: Option<Duration>,
}

/// A served model (policy) to bind envs to.
///
/// Construction does no I/O: it checks the address and timeouts, and a run
/// or a binding dials. Clones share the handle; it is `Send + Sync`.
#[derive(Clone)]
pub struct RemoteModel(Arc<RemoteModelInner>);

impl RemoteModel {
    /// Check the address, timeouts, `api_key` and `tls_ca_file` (the
    /// ValueError class) and keep the address as given, so
    /// `"127.0.0.1:5555"` stays scheme-less. Checks `tls_ca_file` and
    /// `RLMESH_API_KEY` now (each session dial reads them again); no network
    /// I/O. An explicit `api_key` for a plaintext address that is not
    /// loopback warns (`UserWarning`, once per address per process).
    pub fn new(options: RemoteModelOptions) -> Result<RemoteModel> {
        let address = crate::env::client::normalize_connect_address(
            options.address.as_deref(),
            options.host.as_deref(),
            options.port,
            options.path.as_deref(),
            options.transport,
        )?;
        ConnectAddress::parse(&address)?;
        let connect_timeout = crate::env::client::normalize_timeout(options.connect_timeout);
        crate::credentials::resolve(
            &address,
            options.api_key.as_deref(),
            options.tls_ca_file.as_deref(),
            true,
        )?;
        Ok(RemoteModel(Arc::new(RemoteModelInner {
            address,
            connect_timeout,
        })))
    }

    /// The model endpoint this handle dials, as given (normalized, not
    /// canonicalized).
    pub fn address(&self) -> &str {
        &self.0.address
    }

    #[cfg(feature = "lockstep")]
    pub(crate) fn connect_timeout(&self) -> Option<Duration> {
        self.0.connect_timeout
    }
}

impl std::fmt::Display for RemoteModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RemoteModel(address='{}')", self.0.address)
    }
}

impl std::fmt::Debug for RemoteModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}
