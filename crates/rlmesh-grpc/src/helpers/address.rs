//! Address parsing and normalization for connect and bind targets.
//!
//! Accepts the `host:port`, `tcp://host:port`, `http://host:port`, and (on Unix)
//! `unix:///path` forms, rejecting `https://` (reserved for control-plane URLs)
//! and other schemes. A connect target carries both the dial endpoint and a
//! normalized display address; a bind target carries the host/port or socket path.

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
            EnvConnectTargetKind::Tcp => None,
            #[cfg(unix)]
            EnvConnectTargetKind::Unix { path } => Some(path),
        }
    }
}

/// Parse a connect address into an [`EnvConnectTarget`], rejecting `https://`
/// and unsupported schemes.
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

    if let Some(target) = addr.strip_prefix("tcp://") {
        validate_tcp_authority(target)?;
        return Ok(EnvConnectTarget {
            endpoint: format!("http://{target}"),
            address: format!("tcp://{target}"),
            kind: EnvConnectTargetKind::Tcp,
        });
    }

    if let Some(target) = addr.strip_prefix("http://") {
        validate_tcp_authority(target)?;
        return Ok(EnvConnectTarget {
            endpoint: addr.to_string(),
            address: addr.to_string(),
            kind: EnvConnectTargetKind::Tcp,
        });
    }

    reject_nontcp_scheme(addr)?;

    validate_tcp_authority(addr)?;
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

/// Validate a tcp/http authority and re-emit it as an `http://...` dial endpoint.
pub fn normalize_endpoint(addr: &str) -> Result<String, Error> {
    normalize_tcp_to_scheme(addr, "http://")
}

/// Validate a tcp/http authority and re-emit it as a `tcp://...` session address.
pub fn normalize_tcp_session_address(addr: &str) -> Result<String, Error> {
    normalize_tcp_to_scheme(addr, "tcp://")
}

/// Validate a tcp/http authority and re-emit it under `out_scheme`. An
/// `http://` input is preserved verbatim when `out_scheme` is `http://`.
fn normalize_tcp_to_scheme(addr: &str, out_scheme: &str) -> Result<String, Error> {
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

    if let Some(target) = addr.strip_prefix("tcp://") {
        validate_tcp_authority(target)?;
        return Ok(format!("{out_scheme}{target}"));
    }

    if let Some(target) = addr.strip_prefix("http://") {
        validate_tcp_authority(target)?;
        return Ok(if out_scheme == "http://" {
            addr.to_string()
        } else {
            format!("{out_scheme}{target}")
        });
    }

    reject_nontcp_scheme(addr)?;

    validate_tcp_authority(addr)?;
    Ok(format!("{out_scheme}{addr}"))
}

/// Reject a leftover scheme on an address that should be a bare tcp authority:
/// `https://` is reserved for control-plane URLs, any other `scheme://` is
/// unsupported. A schemeless address passes through.
fn reject_nontcp_scheme(addr: &str) -> Result<(), Error> {
    if addr.starts_with("https://") {
        return Err(TransportError::InvalidAddress(
            "https:// is reserved for control-plane URLs, not session links".to_string(),
        )
        .into());
    }

    if let Some((scheme, _)) = addr.split_once("://") {
        return Err(TransportError::InvalidAddress(format!(
            "unsupported address scheme '{scheme}'"
        ))
        .into());
    }

    Ok(())
}

fn validate_tcp_authority(authority: &str) -> Result<(), Error> {
    let (host, port) = authority.rsplit_once(':').ok_or_else(|| {
        TransportError::InvalidAddress(format!(
            "missing port in tcp session endpoint '{authority}'"
        ))
    })?;
    if host.is_empty() {
        return Err(TransportError::InvalidAddress(format!(
            "missing host in tcp session endpoint '{authority}'"
        ))
        .into());
    }
    let _ = port.parse::<u16>().map_err(|error| {
        TransportError::InvalidAddress(format!("invalid tcp port '{port}': {error}"))
    })?;
    Ok(())
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
        BindTarget, normalize_endpoint, normalize_tcp_session_address, parse_bind_target,
        parse_env_connect_target,
    };

    #[test]
    fn endpoint_normalization_accepts_tcp_runtime_forms() {
        assert_eq!(
            normalize_endpoint("localhost:50051").unwrap(),
            "http://localhost:50051"
        );
        assert_eq!(
            normalize_endpoint("tcp://localhost:50051").unwrap(),
            "http://localhost:50051"
        );
        assert_eq!(
            normalize_endpoint("http://localhost:50051").unwrap(),
            "http://localhost:50051"
        );
    }

    #[test]
    fn endpoint_normalization_rejects_unix_and_https() {
        assert!(normalize_endpoint("unix:///tmp/rlmesh.sock").is_err());
        assert!(normalize_endpoint("https://control-plane.example").is_err());
    }

    #[test]
    fn tcp_session_normalization_uses_tcp_scheme() {
        assert_eq!(
            normalize_tcp_session_address("localhost:50052").unwrap(),
            "tcp://localhost:50052"
        );
        assert_eq!(
            normalize_tcp_session_address("http://localhost:50052").unwrap(),
            "tcp://localhost:50052"
        );
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
        assert!(parse_env_connect_target("https://control-plane.example").is_err());
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
