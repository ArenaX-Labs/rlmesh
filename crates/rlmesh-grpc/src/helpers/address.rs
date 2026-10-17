//! Address parsing and normalization for connect and bind targets.
//!
//! A connect target accepts `host:port`, `tcp://host:port` and
//! `http://host:port` (plaintext HTTP/2), `https://host[:port]` (TLS, port 443
//! by default, no path) and, on Unix, `unix:///path`; any other scheme is
//! refused. It carries both the dial endpoint and a normalized display
//! address. A bind target carries the host/port or socket path.

#[cfg(not(unix))]
use std::path::PathBuf;
#[cfg(unix)]
use std::path::{Path, PathBuf};

use crate::error::{Error, TransportError};

/// A parsed connect target: the gRPC dial endpoint plus a normalized display
/// address, and the socket path for a Unix target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvConnectTarget {
    endpoint: String,
    address: String,
    kind: EnvConnectTargetKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EnvConnectTargetKind {
    Tcp,
    Tls {
        host: String,
        port: u16,
    },
    #[cfg(unix)]
    Unix {
        path: PathBuf,
    },
}

/// A parsed bind target: a TCP host/port (port `0` requests an OS-assigned
/// port) or a Unix socket path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindTarget {
    Tcp { host: String, port: u16 },
    Unix { path: PathBuf },
}

impl EnvConnectTarget {
    /// The normalized display address (`tcp://...` or `unix://...`).
    pub fn display_address(&self) -> &str {
        &self.address
    }

    /// The `http://...` endpoint to dial (a placeholder authority for a Unix
    /// target, which connects through a custom connector).
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// The socket path for a Unix target, or `None` for TCP.
    pub fn unix_path(&self) -> Option<&PathBuf> {
        match &self.kind {
            EnvConnectTargetKind::Tcp | EnvConnectTargetKind::Tls { .. } => None,
            #[cfg(unix)]
            EnvConnectTargetKind::Unix { path } => Some(path),
        }
    }

    /// The server host and port of an `https://` target (the host without
    /// IPv6 brackets: the name its certificate must carry), or `None` for a
    /// plaintext or Unix target.
    pub fn tls_server(&self) -> Option<(&str, u16)> {
        match &self.kind {
            EnvConnectTargetKind::Tls { host, port } => Some((host, *port)),
            _ => None,
        }
    }

    /// Whether this is an `https://` (TLS) target.
    pub fn is_tls(&self) -> bool {
        self.tls_server().is_some()
    }

    /// The host of a TCP target, plaintext or TLS (IPv6 without brackets),
    /// or `None` for a Unix target.
    pub fn tcp_host(&self) -> Option<&str> {
        match &self.kind {
            EnvConnectTargetKind::Tls { host, .. } => Some(host),
            EnvConnectTargetKind::Tcp => {
                let authority = self
                    .endpoint
                    .strip_prefix("http://")
                    .unwrap_or(&self.endpoint);
                let host = authority
                    .rsplit_once(':')
                    .map_or(authority, |(host, _)| host);
                Some(
                    host.strip_prefix('[')
                        .and_then(|host| host.strip_suffix(']'))
                        .unwrap_or(host),
                )
            }
            #[cfg(unix)]
            EnvConnectTargetKind::Unix { .. } => None,
        }
    }
}

/// Parse a connect address into an [`EnvConnectTarget`], rejecting unsupported
/// schemes and an `https://` address with a path.
pub fn parse_env_connect_target(addr: &str) -> Result<EnvConnectTarget, Error> {
    let addr = addr.trim();
    if addr.is_empty() {
        return Err(TransportError::InvalidAddress("empty address".to_string()).into());
    }

    if let Some(path) = addr.strip_prefix("unix://") {
        #[cfg(unix)]
        {
            let socket_path = normalize_unix_path(path)?;
            return Ok(EnvConnectTarget {
                endpoint: "http://[::]:50051".to_string(),
                address: format!("unix://{}", socket_path.display()),
                kind: EnvConnectTargetKind::Unix { path: socket_path },
            });
        }

        #[cfg(not(unix))]
        {
            let _ = path;
            return Err(TransportError::InvalidAddress(
                "unix sockets are not supported on Windows; use tcp://host:port instead"
                    .to_string(),
            )
            .into());
        }
    }

    if let Some(rest) = addr.strip_prefix("https://") {
        let (host, port) = parse_https_authority(rest)?;
        let authority = if host.contains(':') {
            format!("[{host}]:{port}")
        } else {
            format!("{host}:{port}")
        };
        return Ok(EnvConnectTarget {
            endpoint: format!("https://{authority}"),
            address: format!("https://{authority}"),
            kind: EnvConnectTargetKind::Tls { host, port },
        });
    }

    if let Some(target) = addr.strip_prefix("tcp://") {
        validate_tcp_authority("tcp://", target)?;
        return Ok(EnvConnectTarget {
            endpoint: format!("http://{target}"),
            address: format!("tcp://{target}"),
            kind: EnvConnectTargetKind::Tcp,
        });
    }

    if let Some(target) = addr.strip_prefix("http://") {
        validate_tcp_authority("http://", target)?;
        return Ok(EnvConnectTarget {
            endpoint: addr.to_string(),
            address: addr.to_string(),
            kind: EnvConnectTargetKind::Tcp,
        });
    }

    reject_nontcp_scheme(addr)?;

    validate_tcp_authority("", addr)?;
    Ok(EnvConnectTarget {
        endpoint: format!("http://{addr}"),
        address: format!("tcp://{addr}"),
        kind: EnvConnectTargetKind::Tcp,
    })
}

/// Parse a bind address into a [`BindTarget`]. A bare port or `host:port`
/// defaults the host to `127.0.0.1`; a missing host is normalized likewise.
pub fn parse_bind_target(addr: &str) -> Result<BindTarget, Error> {
    let addr = addr.trim();
    if addr.is_empty() {
        return Err(TransportError::InvalidAddress("empty address".to_string()).into());
    }

    if let Some(path) = addr.strip_prefix("unix://") {
        return Ok(BindTarget::Unix {
            path: normalize_unix_path(path)?,
        });
    }

    if addr.contains("://") && !addr.starts_with("tcp://") {
        let scheme = addr
            .split_once("://")
            .map(|(scheme, _)| scheme)
            .unwrap_or(addr);
        return Err(TransportError::InvalidAddress(format!(
            "unsupported address scheme '{scheme}'"
        ))
        .into());
    }

    let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
    if tcp_addr.is_empty() {
        return Err(TransportError::InvalidAddress("empty tcp address".to_string()).into());
    }

    if let Some((host, port)) = tcp_addr.rsplit_once(':') {
        return Ok(BindTarget::Tcp {
            host: normalize_bind_host(host).to_string(),
            port: parse_port(port, "tcp port")?,
        });
    }

    Ok(BindTarget::Tcp {
        host: "127.0.0.1".to_string(),
        port: parse_port(tcp_addr, "port")?,
    })
}

/// Validate a tcp/http authority and re-emit it as a `tcp://...` session
/// address, refusing `unix://`, `https://` and other schemes.
///
/// Superseded by [`parse_env_connect_target`], which every RLMesh client now
/// dials through (it also accepts `unix://` and `https://`); kept, with its
/// previous behaviour, for existing callers.
pub fn normalize_tcp_session_address(addr: &str) -> Result<String, Error> {
    let addr = addr.trim();
    if addr.is_empty() {
        return Err(TransportError::InvalidAddress("empty address".to_string()).into());
    }
    if addr.starts_with("unix://") {
        return Err(TransportError::InvalidAddress(
            "unix sockets are not supported by rlmesh-grpc; use tcp://host:port instead"
                .to_string(),
        )
        .into());
    }
    if addr.starts_with("https://") {
        return Err(TransportError::InvalidAddress(
            "https:// is reserved for control-plane URLs, not session links".to_string(),
        )
        .into());
    }
    if let Some(target) = addr.strip_prefix("tcp://") {
        validate_tcp_authority("tcp://", target)?;
        return Ok(format!("tcp://{target}"));
    }
    if let Some(target) = addr.strip_prefix("http://") {
        validate_tcp_authority("http://", target)?;
        return Ok(format!("tcp://{target}"));
    }
    reject_nontcp_scheme(addr)?;
    validate_tcp_authority("", addr)?;
    Ok(format!("tcp://{addr}"))
}

/// Reject a leftover scheme on an address that should be a bare tcp
/// authority. A schemeless address passes through.
fn reject_nontcp_scheme(addr: &str) -> Result<(), Error> {
    if let Some((scheme, _)) = addr.split_once("://") {
        return Err(TransportError::InvalidAddress(format!(
            "unsupported address scheme '{scheme}'"
        ))
        .into());
    }

    Ok(())
}

/// The host (IPv6 without brackets) and port of an `https://` address's
/// remainder: port 443 when omitted; a path other than `/`, a query, a
/// fragment or userinfo is refused.
fn parse_https_authority(rest: &str) -> Result<(String, u16), Error> {
    let shown = shown_address("https://", rest);
    let invalid = |reason: &str| -> Error {
        TransportError::InvalidAddress(format!("{shown}: {reason}")).into()
    };
    let authority = rest.strip_suffix('/').unwrap_or(rest);
    if authority.contains('/') {
        return Err(invalid(
            "an https endpoint cannot have a path; give only https://host[:port]",
        ));
    }
    if authority.contains(['?', '#']) {
        return Err(invalid(
            "an https endpoint cannot have a query or fragment; give only https://host[:port]",
        ));
    }
    if authority.contains('@') {
        return Err(invalid(
            "an https endpoint cannot carry credentials in the address; use api_key",
        ));
    }
    let (host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
        let (host, after) = bracketed
            .split_once(']')
            .ok_or_else(|| invalid("unterminated IPv6 address"))?;
        match after {
            "" => (host, None),
            _ => match after.strip_prefix(':') {
                Some(port) => (host, Some(port)),
                None => return Err(invalid("unexpected text after the IPv6 address")),
            },
        }
    } else {
        match authority.rsplit_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (authority, None),
        }
    };
    if host.is_empty() {
        return Err(invalid("missing host"));
    }
    let port = match port {
        None => 443,
        Some(port) => port
            .parse::<u16>()
            .map_err(|error| invalid(&format!("invalid port '{port}': {error}")))?,
    };
    Ok((host.to_string(), port))
}

/// A plaintext address's `host:port` remainder after `scheme` (`tcp://`,
/// `http://` or none): userinfo, a query, a fragment and a path are refused.
fn validate_tcp_authority(scheme: &str, authority: &str) -> Result<(), Error> {
    let shown = shown_address(scheme, authority);
    let invalid = |reason: &str| -> Error {
        TransportError::InvalidAddress(format!("{shown}: {reason}")).into()
    };
    if authority.contains('@') {
        return Err(invalid("an address cannot carry credentials; use api_key"));
    }
    if authority.contains(['?', '#']) {
        return Err(invalid(
            "an address cannot have a query or fragment; give only host:port",
        ));
    }
    if authority.contains('/') {
        return Err(invalid(
            "a tcp endpoint cannot have a path; give only host:port",
        ));
    }
    let (host, port) = authority
        .rsplit_once(':')
        .ok_or_else(|| invalid("missing port"))?;
    if host.is_empty() {
        return Err(invalid("missing host"));
    }
    let _ = port
        .parse::<u16>()
        .map_err(|error| invalid(&format!("invalid port '{port}': {error}")))?;
    Ok(())
}

/// A refused address as its errors show it: never its userinfo, query or
/// fragment (they can carry a secret), nor a path; only the authority.
fn shown_address(scheme: &str, rest: &str) -> String {
    if rest.contains(['@', '?', '#']) {
        return format!("{scheme}<redacted>");
    }
    match rest.split_once('/') {
        Some((authority, path)) if !path.is_empty() => format!("{scheme}{authority}/…"),
        _ => format!("{scheme}{}", rest.trim_end_matches('/')),
    }
}

fn parse_port(value: &str, context: &str) -> Result<u16, Error> {
    value.parse::<u16>().map_err(|err| {
        TransportError::InvalidAddress(format!("invalid {context} '{value}': {err}")).into()
    })
}

fn normalize_bind_host(host: &str) -> &str {
    match host {
        "" | "localhost" => "127.0.0.1",
        _ => host,
    }
}

#[cfg(unix)]
fn normalize_unix_path(path: &str) -> Result<PathBuf, Error> {
    if path.is_empty() {
        return Err(
            TransportError::InvalidAddress("unix socket path cannot be empty".to_string()).into(),
        );
    }

    let socket_path = Path::new(path);
    if socket_path.is_absolute() {
        return Ok(socket_path.to_path_buf());
    }

    let cwd = std::env::current_dir()
        .map_err(|e| TransportError::InvalidAddress(format!("failed to read cwd: {e}")))?;
    Ok(cwd.join(socket_path))
}

#[cfg(not(unix))]
fn normalize_unix_path(path: &str) -> Result<PathBuf, Error> {
    let _ = path;
    Err(TransportError::InvalidAddress(
        "unix sockets are not supported on Windows; use tcp://host:port instead".to_string(),
    )
    .into())
}

#[cfg(test)]
mod tests {
    use super::{
        BindTarget, normalize_tcp_session_address, parse_bind_target, parse_env_connect_target,
    };

    #[test]
    fn tcp_session_normalization_keeps_its_previous_behaviour() {
        for (address, expected) in [
            ("localhost:50052", "tcp://localhost:50052"),
            ("tcp://localhost:50052", "tcp://localhost:50052"),
            ("http://localhost:50052", "tcp://localhost:50052"),
        ] {
            assert_eq!(normalize_tcp_session_address(address).unwrap(), expected);
        }
        assert!(normalize_tcp_session_address("unix:///tmp/rlmesh.sock").is_err());
        assert!(normalize_tcp_session_address("https://control-plane.example").is_err());
    }

    #[test]
    fn https_targets_default_to_port_443_and_dial_tls() {
        let bare = parse_env_connect_target("https://models.example").unwrap();
        assert_eq!(bare.endpoint(), "https://models.example:443");
        assert_eq!(bare.display_address(), "https://models.example:443");
        assert_eq!(bare.tls_server(), Some(("models.example", 443)));
        assert!(bare.is_tls());

        let slash = parse_env_connect_target("https://models.example:8443/").unwrap();
        assert_eq!(slash.display_address(), "https://models.example:8443");
        assert_eq!(slash.tls_server(), Some(("models.example", 8443)));

        let v6 = parse_env_connect_target("https://[::1]:9443").unwrap();
        assert_eq!(v6.endpoint(), "https://[::1]:9443");
        assert_eq!(v6.tls_server(), Some(("::1", 9443)));
        assert_eq!(v6.tcp_host(), Some("::1"));

        let plain = parse_env_connect_target("tcp://localhost:50051").unwrap();
        assert!(!plain.is_tls());
        assert_eq!(plain.tcp_host(), Some("localhost"));
    }

    #[test]
    fn https_targets_refuse_paths_queries_and_credentials() {
        for address in [
            "https://models.example/v1",
            "https://models.example:443/v1/",
            "https://models.example?x=1",
            "https://user:pw@models.example",
            "https://",
            "https://models.example:notaport",
            "https://[::1",
        ] {
            let error = parse_env_connect_target(address).unwrap_err().to_string();
            assert!(error.contains("invalid address"), "{address}: {error}");
        }
        let error = parse_env_connect_target("https://models.example/v1")
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot have a path"), "{error}");
    }

    #[test]
    fn refused_addresses_never_echo_a_secret() {
        for address in [
            "https://user:SECRET@models.example",
            "https://models.example?api_key=SECRET",
            "https://models.example/v1?api_key=SECRET",
            "https://models.example#SECRET",
            "https://models.example/SECRET",
            "tcp://user:SECRET@models.example:1",
            "http://user:SECRET@models.example:1",
            "user:SECRET@models.example:1",
            "http://localhost:1?api_key=SECRET",
            "http://localhost?api_key=SECRET",
            "tcp://localhost:1#SECRET",
            "tcp://localhost:1/SECRET",
            "localhost:1?api_key=SECRET",
            "localhost:1/v1?api_key=SECRET",
        ] {
            let error = parse_env_connect_target(address).unwrap_err().to_string();
            assert!(!error.contains("SECRET"), "{address}: {error}");
            assert!(error.contains("invalid address"), "{address}: {error}");
        }
    }

    #[test]
    fn bind_target_accepts_tcp_shortcuts() {
        assert_eq!(
            parse_bind_target("7000").unwrap(),
            BindTarget::Tcp {
                host: "127.0.0.1".to_string(),
                port: 7000,
            }
        );
        assert_eq!(
            parse_bind_target("localhost:7001").unwrap(),
            BindTarget::Tcp {
                host: "127.0.0.1".to_string(),
                port: 7001,
            }
        );
        assert_eq!(
            parse_bind_target("tcp://0.0.0.0:7002").unwrap(),
            BindTarget::Tcp {
                host: "0.0.0.0".to_string(),
                port: 7002,
            }
        );
    }

    #[test]
    fn bind_target_rejects_invalid_addresses() {
        assert!(parse_bind_target("").is_err());
        assert!(parse_bind_target("tcp://").is_err());
        assert!(parse_bind_target("http://localhost:50051").is_err());
        assert!(parse_bind_target("localhost:not-a-port").is_err());
    }

    #[test]
    fn env_connect_target_accepts_tcp_and_http_forms() {
        let bare = parse_env_connect_target("localhost:50053").unwrap();
        assert_eq!(bare.endpoint(), "http://localhost:50053");
        assert_eq!(bare.display_address(), "tcp://localhost:50053");

        let tcp = parse_env_connect_target("tcp://localhost:50054").unwrap();
        assert_eq!(tcp.endpoint(), "http://localhost:50054");
        assert_eq!(tcp.display_address(), "tcp://localhost:50054");

        let http = parse_env_connect_target("http://localhost:50055").unwrap();
        assert_eq!(http.endpoint(), "http://localhost:50055");
        assert_eq!(http.display_address(), "http://localhost:50055");
    }

    #[test]
    fn env_connect_target_rejects_invalid_session_urls() {
        assert!(parse_env_connect_target("ftp://localhost:50051").is_err());
        assert!(parse_env_connect_target("localhost").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn env_connect_target_accepts_unix_socket_paths() {
        let target = parse_env_connect_target("unix:///tmp/rlmesh.sock").unwrap();
        assert_eq!(target.endpoint(), "http://[::]:50051");
        assert_eq!(target.display_address(), "unix:///tmp/rlmesh.sock");
        assert_eq!(
            target.unix_path().unwrap(),
            &std::path::PathBuf::from("/tmp/rlmesh.sock")
        );
    }

    #[cfg(unix)]
    #[test]
    fn bind_target_accepts_unix_socket_paths() {
        assert_eq!(
            parse_bind_target("unix:///tmp/rlmesh-bind.sock").unwrap(),
            BindTarget::Unix {
                path: std::path::PathBuf::from("/tmp/rlmesh-bind.sock"),
            }
        );
    }
}
