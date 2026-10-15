//! Python's `ServeOptions`: the transport lifecycle policy shared by the env
//! and model servers.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::{Error, Result};

/// Environment variable holding the token a served model endpoint requires.
pub const MODEL_ENDPOINT_TOKEN_ENV: &str = "RLMESH_MODEL_ENDPOINT_TOKEN";

/// Environment variable holding the token a served environment endpoint requires.
pub const ENV_ENDPOINT_TOKEN_ENV: &str = "RLMESH_ENV_ENDPOINT_TOKEN";

/// Transport lifecycle policy for a server: Python's `ServeOptions`.
///
/// Fields keep Python's names, with the `_seconds` suffix dropped from the
/// durations. The defaults are Python's: remote shutdown off, no idle
/// shutdown, no drain or close bound (an env server caps each at 5 s), no
/// declared edition, uncompressed responses. Options are plain data: the
/// server that consumes them validates them (a timeout must be positive) and
/// resolves the edition.
///
/// Build one with `..Default::default()`; appending a field is a minor change
/// by project policy, so a full struct literal is unsupported.
#[derive(Debug, Clone, Default)]
pub struct ServeOptions {
    /// Honor a client `shutdown` RPC. When `false`, remote shutdown requests
    /// are rejected.
    pub allow_remote_shutdown: bool,
    /// Shut the server down after this much inactivity. `None` never times out.
    pub idle_timeout: Option<Duration>,
    /// Maximum time to drain in-flight requests during shutdown. `None` is the
    /// server's default (5 s for an env server).
    pub drain_timeout: Option<Duration>,
    /// Maximum time the env's (or model's) close may take on shutdown. `None`
    /// is the server's default (5 s for an env server).
    pub close_timeout: Option<Duration>,
    /// The workflow edition this server declares: the serve call's own value,
    /// adopted ahead of `RLMESH_WORKFLOW_EDITION`. A blank value declares
    /// nothing.
    pub workflow_edition: Option<String>,
    /// Compress environment responses (zstd, else gzip) for a client that
    /// accepts them. Has no effect on the model server.
    pub compress_responses: bool,
}

impl ServeOptions {
    /// Python's checks, run by the consuming call: every timeout positive.
    pub(crate) fn validate(&self) -> Result<()> {
        for (name, timeout) in [
            ("idle_timeout_seconds", self.idle_timeout),
            ("drain_timeout_seconds", self.drain_timeout),
            ("close_timeout_seconds", self.close_timeout),
        ] {
            if timeout.is_some_and(|timeout| timeout.is_zero()) {
                return Err(Error::invalid_argument(format!("{name} must be positive")));
            }
        }
        Ok(())
    }

    /// The transport's options for these, with the language-only extras the
    /// serve call carries beside them.
    #[allow(clippy::wrong_self_convention)] // The name R-25 gives it; consuming avoids a clone.
    pub(crate) fn to_grpc(self, extras: WireExtras) -> rlmesh_grpc::ServeOptions {
        rlmesh_grpc::ServeOptions {
            allow_remote_shutdown: self.allow_remote_shutdown,
            idle_timeout: self.idle_timeout,
            drain_timeout: self.drain_timeout,
            close_timeout: self.close_timeout,
            token: extras.token.filter(|token| !token.is_empty()),
            predict_concurrency: extras.predict_concurrency,
            workflow_edition: self
                .workflow_edition
                .map(|edition| edition.trim().to_string())
                .filter(|edition| !edition.is_empty()),
            compress_responses: self.compress_responses,
            peer_info_extra: extras.peer_info_extra,
        }
    }
}

/// What a serve call adds to its [`ServeOptions`] on the wire: the fenced
/// token and peer-info extras, and the model's predict concurrency.
#[derive(Debug, Clone, Default)]
pub(crate) struct WireExtras {
    pub(crate) token: Option<String>,
    pub(crate) predict_concurrency: Option<usize>,
    pub(crate) peer_info_extra: BTreeMap<String, String>,
}

/// An env server's token: `RLMESH_ENV_ENDPOINT_TOKEN` when set (its failures
/// are the `RLMeshException` class), else the configured `ext.token` (a token
/// no header can carry is the `ValueError` class, an invalid options field).
pub(crate) fn env_endpoint_token(configured: Option<String>) -> Result<Option<String>> {
    env_endpoint_token_from(configured, std::env::var(ENV_ENDPOINT_TOKEN_ENV))
}

/// A model server's token (MDL-20): `RLMESH_MODEL_ENDPOINT_TOKEN` when set
/// (its failures are the `RLMeshException` class), else the configured
/// `ext.token` (a token no header can carry is the `ValueError` class).
pub(crate) fn model_endpoint_token(configured: Option<String>) -> Result<Option<String>> {
    configured_endpoint_token(
        MODEL_ENDPOINT_TOKEN_ENV,
        configured,
        std::env::var(MODEL_ENDPOINT_TOKEN_ENV),
    )
}

fn env_endpoint_token_from(
    configured: Option<String>,
    environment: std::result::Result<String, std::env::VarError>,
) -> Result<Option<String>> {
    configured_endpoint_token(ENV_ENDPOINT_TOKEN_ENV, configured, environment)
}

pub(crate) fn configured_endpoint_token(
    variable: &str,
    configured: Option<String>,
    environment: std::result::Result<String, std::env::VarError>,
) -> Result<Option<String>> {
    let configured = configured.unwrap_or_default();
    if matches!(environment, Err(std::env::VarError::NotPresent))
        && !configured.is_empty()
        && let Some(problem) = rlmesh_grpc::helpers::token_header_problem(&configured)
    {
        return Err(Error::invalid_argument(format!(
            "the configured endpoint token {problem}; a token must be printable ASCII \
             without surrounding whitespace, since clients send it in an HTTP header"
        )));
    }
    let token = endpoint_token(variable, configured, environment)?;
    Ok(Some(token).filter(|token| !token.is_empty()))
}

/// Resolve a served endpoint's token from the deployment `variable` reading,
/// falling back to `configured` only when the variable is unset. The reading is
/// passed in so precedence and invalid values can be tested without mutating
/// the process environment.
///
/// Either token must be one a client can send in an HTTP header value: one
/// with leading or trailing whitespace (a secret file's trailing newline),
/// control characters, or non-ASCII characters fails here, naming its source
/// and the problem, instead of serving an endpoint no client can authenticate
/// to. It is never trimmed: a token is a secret, so a malformed one fails
/// closed rather than being guessed at.
pub(crate) fn endpoint_token(
    variable: &str,
    configured: String,
    environment: std::result::Result<String, std::env::VarError>,
) -> Result<String> {
    let header_safe = |source: &str, token: String| match rlmesh_grpc::helpers::token_header_problem(
        &token,
    ) {
        Some(problem) => Err(Error::server(format!(
            "{source} {problem}; a token must be printable ASCII without surrounding whitespace, since clients send it in an HTTP header"
        ))),
        None => Ok(token),
    };
    match environment {
        Ok(token) if token.trim().is_empty() => Err(Error::server(format!(
            "{variable} must not be empty or whitespace-only"
        ))),
        Ok(token) => header_safe(variable, token),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(Error::server(format!("{variable} must be valid Unicode")))
        }
        Err(std::env::VarError::NotPresent) if configured.is_empty() => Ok(configured),
        Err(std::env::VarError::NotPresent) => {
            header_safe("the configured endpoint token", configured)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorClass;

    #[test]
    fn serve_options_default_is_pythons() {
        let options = ServeOptions::default();
        assert!(!options.allow_remote_shutdown);
        assert_eq!(options.idle_timeout, None);
        assert_eq!(options.drain_timeout, None);
        assert_eq!(options.close_timeout, None);
        assert_eq!(options.workflow_edition, None);
        assert!(!options.compress_responses);
    }

    #[test]
    fn a_zero_timeout_is_a_value_error() {
        let options = ServeOptions {
            drain_timeout: Some(Duration::ZERO),
            ..ServeOptions::default()
        };
        let err = options.validate().unwrap_err();
        assert_eq!(err.class(), ErrorClass::ValueError);
        assert!(err.to_string().contains("drain_timeout_seconds"), "{err}");
    }

    #[test]
    fn facade_serve_options_convert_to_transport_options() {
        let options = ServeOptions {
            allow_remote_shutdown: true,
            idle_timeout: Some(Duration::from_secs(1)),
            drain_timeout: Some(Duration::from_secs(2)),
            close_timeout: Some(Duration::from_secs(3)),
            workflow_edition: Some(format!("  {}  ", rlmesh_proto::CURRENT_WORKFLOW_EDITION)),
            compress_responses: true,
        };
        let grpc_options = options.clone().to_grpc(WireExtras {
            token: Some("s3cret".to_string()),
            predict_concurrency: Some(8),
            peer_info_extra: BTreeMap::from([("k".to_string(), "v".to_string())]),
        });
        assert!(grpc_options.allow_remote_shutdown);
        assert_eq!(grpc_options.idle_timeout, options.idle_timeout);
        assert_eq!(grpc_options.drain_timeout, options.drain_timeout);
        assert_eq!(grpc_options.close_timeout, options.close_timeout);
        assert_eq!(grpc_options.token.as_deref(), Some("s3cret"));
        assert_eq!(grpc_options.predict_concurrency, Some(8));
        assert_eq!(
            grpc_options.workflow_edition.as_deref(),
            Some(rlmesh_proto::CURRENT_WORKFLOW_EDITION)
        );
        assert!(grpc_options.compress_responses);
        assert_eq!(grpc_options.peer_info_extra.len(), 1);
        let blank = ServeOptions {
            workflow_edition: Some("   ".to_string()),
            ..ServeOptions::default()
        };
        assert_eq!(blank.to_grpc(WireExtras::default()).workflow_edition, None);
    }

    #[test]
    fn env_endpoint_token_precedence() {
        for (configured, environment, expected) in [
            (None, None, None),
            (Some(""), None, None),
            (Some("direct"), None, Some("direct")),
            (None, Some("deployment"), Some("deployment")),
            (Some("direct"), Some("deployment"), Some("deployment")),
            // The variable replaces a configured token it does not check.
            (Some("bad\n"), Some("deployment"), Some("deployment")),
        ] {
            let environment = environment
                .map(str::to_string)
                .ok_or(std::env::VarError::NotPresent);
            let resolved =
                env_endpoint_token_from(configured.map(str::to_string), environment).unwrap();
            assert_eq!(resolved.as_deref(), expected);
        }
    }

    #[test]
    fn env_endpoint_token_invalid_environment_fails_closed() {
        for value in ["", " ", "\t\n"] {
            let error = env_endpoint_token_from(
                Some("fallback-must-not-be-used".to_string()),
                Ok(value.to_string()),
            )
            .unwrap_err();
            assert_eq!(error.class(), ErrorClass::RLMeshException);
            assert!(error.to_string().contains(ENV_ENDPOINT_TOKEN_ENV));
            assert!(!error.to_string().contains("fallback-must-not-be-used"));
        }
        let error = env_endpoint_token_from(
            None,
            Err(std::env::VarError::NotUnicode("private-value".into())),
        )
        .unwrap_err();
        assert!(error.to_string().contains("valid Unicode"));
        assert!(!error.to_string().contains("private-value"));
    }

    /// A secret mounted from a file usually ends in a newline, which no client
    /// can send in a header: startup fails naming the variable and the
    /// problem, never trimming the value or echoing it.
    #[test]
    fn env_endpoint_token_a_header_cannot_carry_fails_startup() {
        for (value, problem) in [
            ("private-value\n", "trailing whitespace"),
            ("private-value\r\n", "trailing whitespace"),
            (" private-value", "leading or trailing whitespace"),
            ("private\u{1b}value", "control character"),
            ("private-v\u{e4}lue", "non-ASCII"),
        ] {
            let error = env_endpoint_token_from(None, Ok(value.to_string())).unwrap_err();
            assert_eq!(error.class(), ErrorClass::RLMeshException);
            let error = error.to_string();
            assert!(error.contains(ENV_ENDPOINT_TOKEN_ENV), "{error}");
            assert!(error.contains(problem), "{error}");
            assert!(!error.contains("private"), "{error}");
        }
        // A configured token is held to the same rule, as an invalid options
        // field (the ValueError class).
        let error = env_endpoint_token_from(
            Some("private-value\n".to_string()),
            Err(std::env::VarError::NotPresent),
        )
        .unwrap_err();
        assert_eq!(error.class(), ErrorClass::ValueError);
        let error = error.to_string();
        assert!(error.contains("configured endpoint token"), "{error}");
        assert!(!error.contains("private"), "{error}");
        // Interior spaces travel in a header and stay accepted.
        let resolved = env_endpoint_token_from(None, Ok("a b".to_string())).unwrap();
        assert_eq!(resolved.as_deref(), Some("a b"));
    }
}
