//! Transport serve options shared by the env and model servers.

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
    ///
    /// The model server also reads
    /// [`ServeModelOptions::token`](crate::ServeModelOptions::token); a
    /// non-empty value here wins over that field.
    pub token: Option<String>,
    /// Maximum number of model Join-stream requests a served model processes
    /// concurrently per connection (pipelined predict). `None` applies the
    /// default cap. Per-route lifecycle ordering is preserved regardless; this
    /// only bounds how many decode/encode and handler critical sections overlap.
    /// Has no effect on the environment server.
    pub predict_concurrency: Option<usize>,
    /// The workflow edition this served peer **declares** — the sticky
    /// statement of what it was authored against, kept until its author bumps
    /// it. It rides every handshake response as the peer's WANT, so a runtime
    /// runs the session at this edition even when both builds could go higher.
    ///
    /// `None` (the default) declares nothing: the response carries this build's
    /// current edition, which is its own `max(can)` and therefore caps no
    /// runtime — byte-identical to a build without this field.
    ///
    /// Declare the bare `YYYY.MM` base this peer was authored against
    /// (`rlmesh_proto::WORKFLOW_EDITION_BASE` on the build you author on): it
    /// names the contract and selects whichever spelling of it both sides offer,
    /// a dev build's `YYYY.MM-dev.<git>` included. A cohort spelling pins to that
    /// exact moving build instead. Either must admit an edition this build offers
    /// ([`rlmesh_proto::parse_declared_edition`]); the surfaces that take this
    /// from a user — the Python `ServeOptions`, `--workflow-edition`, the C API's
    /// `RlmeshServeOptions` — refuse any other value where it is typed, while a
    /// value set directly on this bare `pub` struct is only trimmed here and is
    /// refused at negotiation instead, by the refusal naming every tier's WANT
    /// and CAN.
    pub workflow_edition: Option<String>,
}

impl From<ServeOptions> for rlmesh_grpc::ServeOptions {
    fn from(value: ServeOptions) -> Self {
        Self {
            allow_remote_shutdown: value.allow_remote_shutdown,
            idle_timeout: value.idle_timeout,
            drain_timeout: value.drain_timeout,
            close_timeout: value.close_timeout,
            token: value.token.filter(|token| !token.is_empty()),
            predict_concurrency: value.predict_concurrency,
            workflow_edition: value
                .workflow_edition
                .map(|edition| edition.trim().to_string())
                .filter(|edition| !edition.is_empty()),
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
                predict_concurrency: None,
                workflow_edition: None,
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
            predict_concurrency: Some(8),
            workflow_edition: Some(rlmesh_proto::CURRENT_WORKFLOW_EDITION.to_string()),
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
        assert_eq!(
            grpc_options.predict_concurrency,
            options.predict_concurrency
        );
        assert_eq!(
            grpc_options.workflow_edition.as_deref(),
            Some(rlmesh_proto::CURRENT_WORKFLOW_EDITION)
        );
    }

    #[test]
    fn blank_workflow_edition_declares_nothing_after_conversion() {
        let options = ServeOptions {
            workflow_edition: Some("   ".to_string()),
            ..ServeOptions::default()
        };
        let grpc_options = rlmesh_grpc::ServeOptions::from(options);
        assert_eq!(grpc_options.workflow_edition, None);
    }

    /// The user-facing surfaces normalize before they get here; this bare field
    /// does not, so the conversion is where a padded value is trimmed.
    #[test]
    fn padded_workflow_edition_is_trimmed_after_conversion() {
        let options = ServeOptions {
            workflow_edition: Some(format!("  {}  ", rlmesh_proto::CURRENT_WORKFLOW_EDITION)),
            ..ServeOptions::default()
        };
        let grpc_options = rlmesh_grpc::ServeOptions::from(options);
        assert_eq!(
            grpc_options.workflow_edition.as_deref(),
            Some(rlmesh_proto::CURRENT_WORKFLOW_EDITION)
        );
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
