//! Remote environment clients over one core supporting any number of lanes.

use std::time::Duration;

use crate::spaces::Space;
use crate::{ConnectAddress, EnvContract, Error, Metadata, Result};

macro_rules! client_common {
    ($Client:ident, $name:literal) => {
        impl $Client {
            fn core_mut(&mut self) -> Result<&mut EnvClientCore> {
                self.core.as_deref_mut().ok_or_else(|| {
                    Error::Runtime(concat!($name, " is in use by a run").to_string())
                })
            }

            /// Run one I/O call on the core, cancellable through the handle's
            /// token.
            async fn io<T>(
                &mut self,
                op: impl for<'c> FnOnce(
                    &'c mut EnvClientCore,
                ) -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<T>> + Send + 'c>,
                >,
            ) -> Result<T> {
                let subscriptions = self.admit(None)?;
                self.io_subscribed(subscriptions, op).await
            }

            fn admit(
                &mut self,
                cancel: Option<&crate::ext::CancelToken>,
            ) -> Result<Vec<crate::ext::CancelSubscription>> {
                let core = self.core_mut()?;
                core.check_open()?;
                let subscriptions: Vec<_> = std::iter::once(&self.cancel)
                    .chain(cancel)
                    .map(crate::ext::CancelToken::subscribe)
                    .collect();
                if subscriptions
                    .iter()
                    .any(crate::ext::CancelSubscription::fired)
                {
                    return Err(Error::Cancelled);
                }
                Ok(subscriptions)
            }

            async fn io_subscribed<T>(
                &mut self,
                subscriptions: Vec<crate::ext::CancelSubscription>,
                op: impl for<'c> FnOnce(
                    &'c mut EnvClientCore,
                ) -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<T>> + Send + 'c>,
                >,
            ) -> Result<T> {
                let core = self.core_mut()?;
                cancellable(subscriptions, op(core)).await
            }

            /// The canonical connect string of the dialed endpoint.
            pub fn address(&self) -> &str {
                &self.shared.address
            }

            /// This connection's container id (UUIDv7), distinct from the
            /// human env name (`env_contract().id()`).
            pub fn env_id(&self) -> &str {
                &self.shared.env_id
            }

            /// The workflow edition this connection negotiated, or the last
            /// edition a session pinned it to.
            pub fn selected_workflow_edition(&self) -> &str {
                match &self.core {
                    Some(core) => core.selected_workflow_edition(),
                    None => &self.lent_edition,
                }
            }

            /// The contract the server reported at handshake.
            pub fn env_contract(&self) -> &EnvContract {
                &self.shared.contract
            }

            /// The env's render mode, if any.
            pub fn render_mode(&self) -> Option<&str> {
                self.shared.contract.render_mode()
            }

            /// The endpoint's metadata (empty when it reports none).
            pub fn metadata(&self) -> &Metadata {
                &self.shared.metadata
            }

            /// The observation space spec reported by the endpoint.
            pub fn observation_space_spec(&self) -> &SpaceSpec {
                self.shared.contract.observation_space()
            }

            /// The action space spec reported by the endpoint.
            pub fn action_space_spec(&self) -> &SpaceSpec {
                self.shared.contract.action_space()
            }

            /// This handle's cancel token: firing it fails the call in
            /// flight, or the next call.
            pub fn cancel_handle(&self) -> crate::ext::CancelToken {
                self.cancel.clone()
            }

            /// Render lane `env_index` (`None` = 0) as a uint8 frame, or
            /// `None` when the env has no frame.
            pub fn render(&mut self, env_index: Option<u64>) -> Result<Option<Value>> {
                refuse_in_async_context(concat!($name, "::render"))?;
                block_on_library(self.render_in(env_index))
            }

            /// The async twin of [`render`](Self::render).
            pub async fn render_async(&mut self, env_index: Option<u64>) -> Result<Option<Value>> {
                in_library(self.render_in(env_index)).await
            }

            async fn render_in(&mut self, env_index: Option<u64>) -> Result<Option<Value>> {
                let index = env_index.unwrap_or(0);
                self.io(move |core| Box::pin(core.render_value(index)))
                    .await
            }

            /// Close this client's session (idempotent); the server keeps
            /// running. Errors other than a lost connection or an expired
            /// bound are returned.
            pub fn close(&mut self) -> Result<()> {
                refuse_in_async_context(concat!($name, "::close"))?;
                block_on_library(self.close_in())
            }

            /// The async twin of [`close`](Self::close).
            pub async fn close_async(&mut self) -> Result<()> {
                in_library(self.close_in()).await
            }

            async fn close_in(&mut self) -> Result<()> {
                self.core_mut()?.close().await
            }

            pub(crate) fn core_ref(&self) -> Option<&EnvClientCore> {
                self.core.as_deref()
            }
        }

        impl Lendable for $Client {
            fn remote_address(&self) -> &str {
                &self.shared.address
            }

            fn lend(&mut self) -> Result<Box<EnvClientCore>> {
                let core = self.core.take().ok_or_else(|| {
                    Error::Runtime(concat!($name, " is in use by a run").to_string())
                })?;
                self.lent_edition = core.selected_workflow_edition().to_string();
                Ok(core)
            }

            fn give_back(&mut self, core: Box<EnvClientCore>) {
                self.core = Some(core);
            }
        }

        impl Drop for $Client {
            fn drop(&mut self) {
                if let Some(core) = self.core.take() {
                    teardown(core);
                }
            }
        }

        impl std::fmt::Debug for $Client {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct($name)
                    .field("address", &crate::address::redact(&self.shared.address))
                    .field("env_id", &self.shared.env_id)
                    .finish_non_exhaustive()
            }
        }
    };
}

mod client_core;
mod episodes;
mod link;
mod single;
mod vector;

pub use client_core::{CoreConnect, EnvClientCore};
pub(crate) use client_core::{decode_render_frame, decode_render_png};
pub use link::EditionCell;
pub(crate) use link::{ENV_HELD, lockstep_error};
pub use single::RemoteEnv;
pub use vector::RemoteVectorEnv;
pub(crate) use vector::reset_lanes_in;

/// How long `close` waits for the server when the client has no request
/// timeout: past it the client detaches locally.
const CLOSE_GRACE: Duration = Duration::from_secs(5);

/// The runtime-reserved info keys.
const RUNTIME_INFO_KEYS: [&str; 2] = ["episode_ids", "completed_episodes"];

/// Options for connecting to a single or vector environment.
///
/// `address` names a session participant. A zero timeout means no timeout.
#[derive(Clone, Default)]
pub struct RemoteEnvOptions {
    /// A session participant address, such as `localhost:5555/env`.
    pub address: Option<String>,
    /// Bound on the dial and handshake.
    pub connect_timeout: Option<Duration>,
    /// Local bound on each reset, step, render and close.
    pub request_timeout: Option<Duration>,
    /// The workflow edition this client declares on the handshake; empty or
    /// whitespace declares nothing and resolution continues.
    pub workflow_edition: Option<String>,
}

impl std::fmt::Debug for RemoteEnvOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteEnvOptions")
            .field(
                "address",
                &self.address.as_deref().map(crate::address::redact),
            )
            .field("connect_timeout", &self.connect_timeout)
            .field("request_timeout", &self.request_timeout)
            .field("workflow_edition", &self.workflow_edition)
            .finish()
    }
}

impl RemoteEnvOptions {
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

    /// Set `workflow_edition`.
    pub fn workflow_edition(mut self, edition: impl Into<String>) -> Self {
        self.workflow_edition = Some(edition.into());
        self
    }
}

/// Normalize a connect address: one connect string from an address or the
/// helpers, every refusal the ValueError class.
pub(crate) fn normalize_connect_address(address: Option<&str>) -> Result<String> {
    let address = address.ok_or_else(|| Error::invalid_argument("address is required"))?;
    ConnectAddress::parse(address)?;
    Ok(address.to_string())
}

/// Zero means no timeout, as `None` does.
pub(crate) fn normalize_timeout(timeout: Option<Duration>) -> Option<Duration> {
    timeout.filter(|timeout| !timeout.is_zero())
}

/// A timeout given in seconds (the bindings' `*_timeout_seconds`): `None`
/// and zero are no timeout; negative or non-finite is the ValueError class.
pub(crate) fn timeout_from_secs(value: Option<f64>, field: &str) -> Result<Option<Duration>> {
    match value {
        None => Ok(None),
        Some(seconds) => Duration::try_from_secs_f64(seconds)
            .map(|timeout| normalize_timeout(Some(timeout)))
            .map_err(|_| {
                Error::invalid_argument(format!(
                    "{field} must be a non-negative finite float or None"
                ))
            }),
    }
}

/// Bound `future` by `timeout`: a reply handled at or after the deadline is
/// discarded and the call fails with the TimeoutError class.
async fn bounded<T>(
    timeout: Option<Duration>,
    future: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    match timeout {
        Some(timeout) => {
            let deadline = tokio::time::Instant::now() + timeout;
            tokio::time::timeout_at(deadline, future)
                .await
                .map_err(|_| Error::Timeout(timeout))?
        }
        None => future.await,
    }
}

/// What both clients keep outside the per-call core, so accessors never wait
/// on a call.
struct Shared {
    address: String,
    env_id: String,
    contract: EnvContract,
    metadata: Metadata,
    observation_space: Space,
    action_space: Space,
    num_envs: u64,
}

impl Shared {
    fn of(core: &EnvClientCore) -> Result<Shared> {
        let contract = EnvContract::from_wire(core.env_contract())
            .map_err(|err| Error::Protocol(err.to_string()))?;
        Ok(Shared {
            address: core.address().to_string(),
            env_id: core.env_id().to_string(),
            metadata: contract.metadata().cloned().unwrap_or_default(),
            observation_space: Space::new(contract.observation_space().clone()),
            action_space: Space::new(contract.action_space().clone()),
            num_envs: core.num_envs() as u64,
            contract,
        })
    }
}

/// The client half a session can borrow: its core moves into the session for
/// the borrow and comes back when the session ends.
pub(crate) trait Lendable {
    /// The handle's canonical connect string.
    fn remote_address(&self) -> &str;
    /// Move the core out for a run's borrow.
    fn lend(&mut self) -> Result<Box<EnvClientCore>>;
    /// Return a lent core.
    fn give_back(&mut self, core: Box<EnvClientCore>);
}
