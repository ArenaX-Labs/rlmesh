//! Server lifecycle options shared by the env and model servers.

use std::time::Duration;

/// Transport lifecycle policy for a server (shutdown, timeouts, auth).
///
/// Defaults are conservative: remote shutdown is disabled, every timeout is
/// unset (no idle shutdown, no drain/close bound), and no token is required.
/// Pass an instance to [`EnvServer::bind_with_options`](crate::EnvServer::bind_with_options)
/// or via [`ServeModelOptions`](crate::ServeModelOptions) for the model server.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServeOptions {
    /// Honor a client `shutdown` RPC. When `false`, remote shutdown requests
    /// are rejected and the server stops only via its own idle/drain policy.
    pub allow_remote_shutdown: bool,
    /// Shut the server down after this much inactivity. `None` never times out.
    pub idle_timeout: Option<Duration>,
    /// Maximum time to drain in-flight requests during shutdown. `None` waits
    /// indefinitely.
    pub drain_timeout: Option<Duration>,
    /// Maximum time the environment/handler close hook may take on shutdown.
    /// `None` waits indefinitely.
    pub close_timeout: Option<Duration>,
    /// Bearer token required on the `authorization` metadata header of every
    /// request to this endpoint.
    ///
    /// `None` (or an empty string) **disables authentication**: the endpoint
    /// accepts every request without a token. Set this to require a token.
    pub token: Option<String>,
}

impl From<ServeOptions> for rlmesh_grpc::ServeOptions {
    fn from(value: ServeOptions) -> Self {
        Self {
            allow_remote_shutdown: value.allow_remote_shutdown,
            idle_timeout: value.idle_timeout,
            drain_timeout: value.drain_timeout,
            close_timeout: value.close_timeout,
            token: value.token.filter(|token| !token.is_empty()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_options_default_disables_remote_shutdown() {
        assert_eq!(
            ServeOptions::default(),
            ServeOptions {
                allow_remote_shutdown: false,
                idle_timeout: None,
                drain_timeout: None,
                close_timeout: None,
                token: None,
            }
        );
    }

    #[test]
    fn facade_serve_options_convert_to_transport_options() {
        let options = ServeOptions {
            allow_remote_shutdown: true,
            idle_timeout: Some(Duration::from_secs(1)),
            drain_timeout: Some(Duration::from_secs(2)),
            close_timeout: Some(Duration::from_secs(3)),
            token: Some("s3cret".to_string()),
        };
        let grpc_options = rlmesh_grpc::ServeOptions::from(options.clone());
        assert_eq!(
            grpc_options.allow_remote_shutdown,
            options.allow_remote_shutdown
        );
        assert_eq!(grpc_options.idle_timeout, options.idle_timeout);
        assert_eq!(grpc_options.drain_timeout, options.drain_timeout);
        assert_eq!(grpc_options.close_timeout, options.close_timeout);
        assert_eq!(grpc_options.token.as_deref(), Some("s3cret"));
    }

    #[test]
    fn empty_token_disables_env_auth_after_conversion() {
        let options = ServeOptions {
            token: Some(String::new()),
            ..ServeOptions::default()
        };
        let grpc_options = rlmesh_grpc::ServeOptions::from(options);
        assert_eq!(grpc_options.token, None);
    }
}
