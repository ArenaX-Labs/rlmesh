//! Client credentials: the API key a remote client sends and the certificate
//! authorities an `https://` dial trusts, resolved once for every language.
//!
//! The key is the explicit `api_key`, else the `RLMESH_API_KEY` environment
//! variable, which is attached only to a TLS (`https://`) endpoint, never to
//! plaintext TCP or a Unix socket. An explicit key sent in plaintext to a host
//! other than loopback warns once. The key never appears in error text or
//! `Debug` output.

use std::path::Path;

use rlmesh_grpc::{Credentials, TlsTrust};

use crate::{Error, Result, WarningCategory};

/// The environment variable a remote client reads its API key from when no
/// `api_key` is given (TLS endpoints only).
pub const API_KEY_ENV: &str = "RLMESH_API_KEY";

/// A client's resolved credentials for one address. `Debug` redacts the key.
#[derive(Clone, Debug, Default)]
pub struct ClientAuth(pub(crate) Credentials);

impl ClientAuth {
    /// The resolved key (empty: none). Never log or print it.
    pub fn token(&self) -> &str {
        &self.0.token
    }

    /// The TLS trust of an `https://` dial.
    pub fn trust(&self) -> &TlsTrust {
        &self.0.trust
    }
}

/// Resolve the credentials a client dialing `address` sends: the ValueError
/// class for a key a header cannot carry, a `tls_ca_file` that cannot be
/// read or parsed, or one given for an address that is not `https://`. With
/// `warn`, an explicit key bound for a plaintext non-loopback host warns
/// (`UserWarning`). No network I/O.
pub fn resolve(
    address: &str,
    api_key: Option<&str>,
    tls_ca_file: Option<&Path>,
    warn: bool,
) -> Result<ClientAuth> {
    let target =
        rlmesh_grpc::helpers::parse_env_connect_target(address).map_err(Error::from_grpc)?;
    let trust = match tls_ca_file {
        None => TlsTrust::System,
        Some(_) if !target.is_tls() => {
            return Err(Error::invalid_argument(format!(
                "tls_ca_file applies only to an https:// address; got '{}'",
                target.display_address()
            )));
        }
        Some(path) => TlsTrust::from_ca_file(path).map_err(Error::from_grpc)?,
    };
    let token = match api_key {
        Some(key) => {
            check_key(key, "the api_key")?;
            if warn && !key.is_empty() && !target.is_tls() && !is_local(&target) {
                crate::warnings::warn(
                    WarningCategory::UserWarning,
                    format!(
                        "api_key is sent unencrypted to {}; use an https:// address to keep it \
                         private",
                        target.display_address()
                    ),
                )?;
            }
            key.to_string()
        }
        None if target.is_tls() => match std::env::var(API_KEY_ENV) {
            Ok(key) => {
                check_key(&key, API_KEY_ENV)?;
                key
            }
            Err(_) => String::new(),
        },
        None => String::new(),
    };
    Ok(ClientAuth(Credentials { token, trust }))
}

/// A key a header cannot carry is the ValueError class; the text names
/// `what` and the problem, never the key.
fn check_key(key: &str, what: &str) -> Result<()> {
    if let Some(problem) = rlmesh_grpc::helpers::auth::token_header_problem(key) {
        return Err(Error::invalid_argument(format!(
            "{what} cannot be sent in the authorization header: it {problem}"
        )));
    }
    Ok(())
}

/// A Unix socket or a loopback TCP host: traffic that stays on this machine.
fn is_local(target: &rlmesh_grpc::helpers::EnvConnectTarget) -> bool {
    match target.tcp_host() {
        None => true,
        Some(host) => {
            host.eq_ignore_ascii_case("localhost")
                || host
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|ip| ip.is_loopback())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_keys_are_checked_and_never_printed() {
        let auth = resolve("tcp://127.0.0.1:1", Some("s3cret"), None, false).unwrap();
        assert_eq!(auth.token(), "s3cret");
        assert!(!format!("{auth:?}").contains("s3cret"));
        let error = resolve("tcp://127.0.0.1:1", Some("s3cret\n"), None, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("the api_key cannot be sent"), "{error}");
        assert!(!error.contains("s3cret"), "{error}");
    }

    #[test]
    fn ca_files_need_https_and_a_pem_bundle() {
        let error = resolve(
            "tcp://127.0.0.1:1",
            None,
            Some(Path::new("/nope.pem")),
            false,
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidArgument(_)), "{error}");
        let error = resolve(
            "https://localhost:1",
            None,
            Some(Path::new("/definitely/missing.pem")),
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("cannot be read"), "{error}");
        let path = std::env::temp_dir().join(format!("rlmesh-bad-ca-{}.pem", std::process::id()));
        std::fs::write(&path, "not a certificate").unwrap();
        let error = resolve("https://localhost:1", None, Some(&path), false).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(
            error.to_string().contains("holds no PEM certificate"),
            "{error}"
        );
    }

    #[test]
    fn loopback_and_unix_are_local() {
        for (address, local) in [
            ("tcp://127.0.0.1:1", true),
            ("localhost:1", true),
            ("tcp://[::1]:1", true),
            ("tcp://10.0.0.1:1", false),
            ("models.example:80", false),
        ] {
            let target = rlmesh_grpc::helpers::parse_env_connect_target(address).unwrap();
            assert_eq!(is_local(&target), local, "{address}");
        }
    }
}
