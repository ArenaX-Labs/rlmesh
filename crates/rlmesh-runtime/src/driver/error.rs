//! The driver's error type, [`RuntimeError`].

use std::time::Duration;

use crate::hooks::HookError;

/// Type-erased structured error preserved as the `#[source]` of RPC failures.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RuntimeError {
    #[error("invalid runtime session spec: {0}")]
    InvalidSpec(String),

    #[error(
        "{operation} timed out on route {route_id} component {component_id} at runtime step {step} after {timeout:?}"
    )]
    OperationTimeout {
        route_id: String,
        component_id: String,
        operation: &'static str,
        step: i64,
        timeout: Duration,
    },

    #[error("route {route_id} cancelled at runtime step {step}: {reason}")]
    RouteCancelled {
        route_id: String,
        step: i64,
        reason: String,
    },

    #[error(
        "environment {operation} failed at runtime step {step}: {message}. If the source is 'transport error: connection closed', the environment server exited, crashed, or received SIGTERM before replying; inspect the environment container logs immediately before the runtime error timestamp"
    )]
    EnvRpc {
        operation: &'static str,
        step: i64,
        message: String,
        /// Whether the underlying transport error is retryable. Captured at
        /// construction by the adapter, which owns the structured error.
        recoverable: bool,
        /// The structured underlying error, preserved so callers can downcast
        /// or inspect the chain. `rlmesh-runtime` does not depend on
        /// `rlmesh-grpc`, so the concrete type is erased here.
        #[source]
        source: Option<BoxError>,
    },

    #[error("model endpoint {component_id} request failed: {message}")]
    ModelRpc {
        component_id: String,
        message: String,
        /// Whether the underlying error is retryable. Captured at construction.
        recoverable: bool,
        #[source]
        source: Option<BoxError>,
    },

    #[error(
        "model endpoint {component_id} returned mismatched route identity for request {request_id}"
    )]
    ModelRouteMismatch {
        component_id: String,
        request_id: String,
    },

    #[error("protocol error: {0}")]
    Protocol(String),

    #[error("runtime hook failed: {0}")]
    Hook(HookError),
}

impl RuntimeError {
    pub fn operation_timeout(
        route_id: impl Into<String>,
        component_id: impl Into<String>,
        operation: &'static str,
        step: i64,
        timeout: Duration,
    ) -> Self {
        Self::OperationTimeout {
            route_id: route_id.into(),
            component_id: component_id.into(),
            operation,
            step,
            timeout,
        }
    }

    pub fn route_cancelled(
        route_id: impl Into<String>,
        step: i64,
        reason: impl Into<String>,
    ) -> Self {
        Self::RouteCancelled {
            route_id: route_id.into(),
            step,
            reason: reason.into(),
        }
    }

    /// Constructs an [`EnvRpc`](Self::EnvRpc) error, capturing the underlying
    /// error's recoverability and preserving it as a structured `#[source]`.
    pub fn env_rpc<E>(operation: &'static str, step: i64, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::env_rpc_with_recoverability(operation, step, false, source)
    }

    /// Constructs an [`EnvRpc`](Self::EnvRpc) error with an explicit
    /// recoverability flag (e.g. from `GrpcError::is_recoverable`), preserving
    /// the structured source.
    pub fn env_rpc_with_recoverability<E>(
        operation: &'static str,
        step: i64,
        recoverable: bool,
        source: E,
    ) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::EnvRpc {
            operation,
            step,
            message: source.to_string(),
            recoverable,
            source: Some(Box::new(source)),
        }
    }

    /// Constructs a [`ModelRpc`](Self::ModelRpc) error, preserving the
    /// structured source. Recoverability defaults to `false`.
    pub fn model_rpc<E>(component_id: impl Into<String>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::model_rpc_with_recoverability(component_id, false, source)
    }

    /// Constructs a [`ModelRpc`](Self::ModelRpc) error with an explicit
    /// recoverability flag, preserving the structured source.
    pub fn model_rpc_with_recoverability<E>(
        component_id: impl Into<String>,
        recoverable: bool,
        source: E,
    ) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::ModelRpc {
            component_id: component_id.into(),
            message: source.to_string(),
            recoverable,
            source: Some(Box::new(source)),
        }
    }

    /// Whether this error is recoverable (retryable).
    ///
    /// For RPC failures this reflects the recoverability captured from the
    /// underlying transport error at construction. All other variants are
    /// treated as non-recoverable.
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::EnvRpc { recoverable, .. } | Self::ModelRpc { recoverable, .. } => *recoverable,
            _ => false,
        }
    }
}
