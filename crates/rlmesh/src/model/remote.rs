use std::sync::Arc;
use std::time::Duration;

use crate::{ConnectAddress, Result};

/// Options for connecting to a served model: the same as
/// [`RemoteEnvOptions`](crate::RemoteEnvOptions) without `workflow_edition`.
#[derive(Clone, Default)]
pub struct RemoteModelOptions {
    /// A session participant address.
    pub address: Option<String>,
    /// Bound on each session's dial to the model (zero: none).
    pub connect_timeout: Option<Duration>,
    /// Bound on each `predict` and route close of sessions off this handle
    /// (zero: none).
    pub request_timeout: Option<Duration>,
}

impl std::fmt::Debug for RemoteModelOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteModelOptions")
            .field("address", &self.address)
            .field("connect_timeout", &self.connect_timeout)
            .field("request_timeout", &self.request_timeout)
            .finish()
    }
}

impl RemoteModelOptions {
    /// Set `address`.
    pub fn address(mut self, address: impl Into<String>) -> Self {
        self.address = Some(address.into());
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
}

struct RemoteModelInner {
    address: String,
    connect_timeout: Option<Duration>,
    request_timeout: Option<Duration>,
}

/// A served model (policy) to bind envs to.
///
/// Construction does no I/O: it checks the address and timeouts, and a run
/// or a binding dials. Clones share the handle; it is `Send + Sync`.
#[derive(Clone)]
pub struct RemoteModel(Arc<RemoteModelInner>);

impl RemoteModel {
    /// Check the session participant address and local timeouts without I/O.
    pub fn new(options: RemoteModelOptions) -> Result<RemoteModel> {
        let address = crate::env::client::normalize_connect_address(options.address.as_deref())?;
        ConnectAddress::parse(&address)?;
        let connect_timeout = crate::env::client::normalize_timeout(options.connect_timeout);
        let request_timeout = crate::env::client::normalize_timeout(options.request_timeout);
        Ok(RemoteModel(Arc::new(RemoteModelInner {
            address,
            connect_timeout,
            request_timeout,
        })))
    }

    /// The model endpoint this handle dials, as given (normalized, not
    /// canonicalized).
    pub fn address(&self) -> &str {
        &self.0.address
    }

    pub(crate) fn connect_timeout(&self) -> Option<Duration> {
        self.0.connect_timeout
    }

    /// The model this handle dials, its requests bounded by
    /// `request_timeout` when set.
    pub(crate) fn served(&self, connect_timeout: Duration) -> crate::lockstep::ServedModel {
        let served = crate::lockstep::ServedModel::new(self.address(), connect_timeout);
        match self.0.request_timeout {
            Some(deadline) => served.with_request_timeout(deadline),
            None => served,
        }
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
